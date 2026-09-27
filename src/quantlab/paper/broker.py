"""Broker abstraction, a paper broker, and the live-trading lock.

``Broker`` is the only interface execution code talks to, so a real brokerage adapter can be added
later without touching research or risk code. **No real-money broker is implemented.** The
:func:`assert_live_trading_permitted` lock refuses unless BOTH an explicit configuration flag and a
signed safety-review file are present — and even then ``LiveBroker`` is a stub that raises.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from quantlab.config import Settings, get_settings


class LiveTradingDisabled(PermissionError):
    pass


def assert_live_trading_permitted(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    if not s.live_trading_enabled:
        raise LiveTradingDisabled("live trading is disabled (QUANTLAB_LIVE_TRADING_ENABLED is not true)")
    f = s.live_safety_review_file
    if f is None or not Path(f).exists():
        raise LiveTradingDisabled("no safety-review file (QUANTLAB_LIVE_SAFETY_REVIEW_FILE)")
    text = Path(f).read_text()
    if "APPROVED-BY:" not in text or "SHA256-OF-CONFIG:" not in text:
        raise LiveTradingDisabled("safety-review file lacks APPROVED-BY and SHA256-OF-CONFIG lines")


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    ts: pd.Timestamp
    bid_size: float = float("nan")
    ask_size: float = float("nan")
    source: str = "unknown"

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class Order:
    symbol: str
    quantity: float  # + buy / − sell
    kind: str = "market"
    limit: float | None = None
    multiplier: float = 1.0
    signal_id: str | None = None
    client_id: str = ""


@dataclass
class Fill:
    order: Order
    status: str  # FILLED | REJECTED
    price: float | None
    quantity: float
    commission: float
    quote: Quote | None
    reason: str = ""
    ts: pd.Timestamp | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quote"] = asdict(self.quote) if self.quote else None
        return d


class Broker(ABC):
    @abstractmethod
    def quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def submit(self, order: Order) -> Fill: ...

    @abstractmethod
    def positions(self) -> dict[str, float]: ...

    @abstractmethod
    def cash(self) -> float: ...


@dataclass
class PaperBroker(Broker):
    """Fills against supplied quotes: buys at ask + slippage, sells at bid − slippage. Never at mid."""

    starting_cash: float
    quotes: dict[str, Quote] = field(default_factory=dict)
    slippage_bps: float = 2.0
    commission_per_share: float = 0.0
    commission_min: float = 0.0
    max_quote_age: pd.Timedelta = pd.Timedelta(minutes=5)
    now: pd.Timestamp | None = None
    _cash: float = field(init=False)
    _pos: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._cash = self.starting_cash

    def set_quote(self, q: Quote) -> None:
        self.quotes[q.symbol] = q

    def quote(self, symbol: str) -> Quote:
        if symbol not in self.quotes:
            raise KeyError(f"no quote for {symbol}")
        return self.quotes[symbol]

    def submit(self, order: Order) -> Fill:
        q = self.quotes.get(order.symbol)
        now = self.now or (q.ts if q else None)
        if q is None:
            return Fill(order, "REJECTED", None, 0, 0, None, "no quote", now)
        if now is not None and now - q.ts > self.max_quote_age:
            return Fill(order, "REJECTED", None, 0, 0, q, "stale quote", now)
        side = 1 if order.quantity > 0 else -1
        ref = q.ask if side > 0 else q.bid
        if ref <= 0:
            return Fill(order, "REJECTED", None, 0, 0, q, "no executable price", now)
        px = ref * (1 + side * self.slippage_bps / 1e4)
        if order.kind == "limit" and order.limit is not None and ((side > 0 and px > order.limit) or (side < 0 and px < order.limit)):
            return Fill(order, "REJECTED", None, 0, 0, q, "limit not marketable", now)
        comm = max(self.commission_min, self.commission_per_share * abs(order.quantity)) if order.quantity else 0.0
        cost = order.quantity * px * order.multiplier + comm
        if side > 0 and cost > self._cash:
            return Fill(order, "REJECTED", None, 0, 0, q, "insufficient cash", now)
        self._cash -= cost
        self._pos[order.symbol] = self._pos.get(order.symbol, 0.0) + order.quantity
        return Fill(order, "FILLED", px, order.quantity, comm, q, "", now)

    def positions(self) -> dict[str, float]:
        return {k: v for k, v in self._pos.items() if v != 0}

    def cash(self) -> float:
        return self._cash


class LiveBroker(Broker):  # pragma: no cover - intentionally unimplemented
    """Placeholder. Constructing it checks the lock; every method raises until a reviewed adapter exists."""

    def __init__(self, settings: Settings | None = None) -> None:
        assert_live_trading_permitted(settings)
        raise NotImplementedError("no live brokerage adapter has been implemented or reviewed")

    def quote(self, symbol): raise NotImplementedError
    def submit(self, order): raise NotImplementedError
    def positions(self): raise NotImplementedError
    def cash(self): raise NotImplementedError


def config_fingerprint(settings: Settings) -> str:
    """Hash of the non-secret configuration, to be quoted in a safety review."""
    d = settings.model_dump(exclude={"fred_api_key", "tiingo_api_key", "orats_api_key", "polygon_api_key"})
    return hashlib.sha256(repr(sorted(d.items())).encode()).hexdigest()
