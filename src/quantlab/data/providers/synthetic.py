"""Synthetic market generator — every output is labeled ``SIMULATED``.

Purpose: exercise and *calibrate* the research machinery (does it reject pure noise at the nominal
rate? does it detect an effect of known size?). Synthetic results are never research findings.

Model (daily, on the real NYSE session calendar)
------------------------------------------------
* Market return ``m_t`` with GARCH(1,1) variance, Student-t shocks and a two-state (calm/stressed)
  Markov regime that scales volatility and shifts drift.
* Sector returns ``s_{k,t}`` (Gaussian, with their own vol) and asset returns
  ``r_{i,t} = beta_i m_t + s_{k(i),t} + e_{i,t}``.
* Optional **planted effects** (default: none):
    - ``idio_ar1``: AR(1) coefficient of idiosyncratic returns (negative ⇒ short-term reversal),
    - ``momentum_loading``: drift proportional to the trailing 120-day idiosyncratic return.
* Each return is split into overnight and intraday parts to form OHLC; volume is lognormal and
  rises with |return|.
* Corporate actions: quarterly cash dividends and occasional splits when price is high. Raw
  (unadjusted) prices reflect them exactly as a real tape would.
* Quarterly earnings events with announcement jumps (timing BMO/AMC).
* ``SYN_VIX``: a VIX-like index = model conditional vol × a variance premium + noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.calendar import get_calendar
from quantlab.data.providers.base import DataProvider, ProviderInfo
from quantlab.provenance import DataLabel, Dataset, Provenance

SOURCE = "synthetic"


@dataclass
class SyntheticConfig:
    start: str = "2005-01-01"
    end: str = "2024-12-31"
    n_symbols: int = 20
    n_sectors: int = 4
    seed: int = 7
    annual_drift: float = 0.07
    garch_omega: float = 2e-6
    garch_alpha: float = 0.08
    garch_beta: float = 0.90
    t_dof: float = 5.0
    stress_prob_enter: float = 0.01
    stress_prob_exit: float = 0.05
    stress_vol_mult: float = 2.0
    stress_drift: float = -0.20  # annualized drift while stressed
    idio_vol_annual: float = 0.25
    sector_vol_annual: float = 0.10
    overnight_share: float = 0.35  # fraction of variance realized overnight
    dividend_yield: float = 0.015
    split_threshold: float = 400.0
    earnings_jump_vol: float = 0.05
    # planted effects (0 = none)
    idio_ar1: float = 0.0
    momentum_loading: float = 0.0
    market_symbol: str = "SYNMKT"
    base_volume: float = 2e6
    extra: dict = field(default_factory=dict)


class SyntheticMarket:
    """Generates a complete, internally consistent SIMULATED market."""

    def __init__(self, config: SyntheticConfig | None = None, delay_minutes: int = 15) -> None:
        self.cfg = config or SyntheticConfig()
        self.cal = get_calendar()
        self.delay_minutes = delay_minutes
        self._generated = False

    # ---------------------------------------------------------------------------------------------
    def generate(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        sessions = self.cal.sessions(cfg.start, cfg.end)
        T, N, K = len(sessions), cfg.n_symbols, cfg.n_sectors
        if T < 300:
            raise ValueError("synthetic history too short; need >= 300 sessions")

        # --- market with GARCH + regimes --------------------------------------------------------
        shocks = rng.standard_t(cfg.t_dof, size=T) / np.sqrt(cfg.t_dof / (cfg.t_dof - 2))
        var = np.empty(T)
        m = np.empty(T)
        regime = np.zeros(T, dtype=int)
        uncond = cfg.garch_omega / max(1e-9, 1 - cfg.garch_alpha - cfg.garch_beta)
        v = uncond
        state = 0
        u = rng.random(T)
        for t in range(T):
            if t > 0:
                if state == 0 and u[t] < cfg.stress_prob_enter:
                    state = 1
                elif state == 1 and u[t] < cfg.stress_prob_exit:
                    state = 0
            regime[t] = state
            scale = cfg.stress_vol_mult if state else 1.0
            var[t] = v * scale**2
            drift = (cfg.stress_drift if state else cfg.annual_drift) / 252
            m[t] = drift + np.sqrt(var[t]) * shocks[t]
            v = cfg.garch_omega + cfg.garch_alpha * (m[t] - drift) ** 2 / scale**2 + cfg.garch_beta * v

        # --- sectors and idiosyncratic ---------------------------------------------------------
        sector_of = np.arange(N) % K
        betas = rng.uniform(0.6, 1.4, size=N)
        s = rng.normal(0, cfg.sector_vol_annual / np.sqrt(252), size=(T, K))
        idio_sd = cfg.idio_vol_annual / np.sqrt(252) * rng.uniform(0.7, 1.3, size=N)
        eps = rng.normal(0, 1, size=(T, N)) * idio_sd
        e = np.zeros((T, N))
        cum_idio = np.zeros((T, N))
        for t in range(T):
            e[t] = eps[t]
            if t > 0 and cfg.idio_ar1:
                e[t] += cfg.idio_ar1 * e[t - 1]
            if t > 120 and cfg.momentum_loading:
                trailing = cum_idio[t - 1] - cum_idio[t - 121]
                e[t] += cfg.momentum_loading * trailing / 120
            cum_idio[t] = (cum_idio[t - 1] if t else 0) + e[t]

        # --- earnings events: quarterly, jump on announcement reaction day ----------------------
        events = []
        jumps = np.zeros((T, N))
        for i in range(N):
            offset = rng.integers(20, 60)
            for t0 in range(offset, T, 63):
                timing = "BMO" if rng.random() < 0.5 else "AMC"
                reaction = t0 if timing == "BMO" else t0 + 1
                if reaction >= T:
                    continue
                jumps[reaction, i] = rng.normal(0, cfg.earnings_jump_vol)
                sess = sessions[t0]
                if timing == "BMO":
                    ann = self.cal.session_open(sess) - pd.Timedelta(minutes=60)
                else:
                    ann = self.cal.session_close(sess) + pd.Timedelta(minutes=20)
                events.append({
                    "symbol": self._sym(i), "fiscal_period": f"Q{len(events)}",
                    "announce_time": ann, "timing": timing, "available_at": ann, "source": SOURCE,
                })
        total = betas * m[:, None] + s[:, sector_of] + e + jumps

        # --- prices, corporate actions ----------------------------------------------------------
        on_share = cfg.overnight_share
        on_noise = rng.normal(0, 1, size=(T, N))
        on_ret = total * on_share + on_noise * idio_sd * 0.3
        on_ret[jumps != 0] = total[jumps != 0]  # announcement reaction happens at the open
        intra_ret = total - on_ret

        closes = np.empty((T, N))
        opens = np.empty((T, N))
        actions = []
        price = rng.uniform(20, 200, size=N)
        div_days = {i: set(range(int(rng.integers(5, 60)), T, 63)) for i in range(N)}
        for t in range(T):
            prev_close = price.copy()
            ratio = np.ones(N)
            div = np.zeros(N)
            for i in range(N):
                if t > 0 and prev_close[i] > cfg.split_threshold and rng.random() < 0.05:
                    ratio[i] = 2.0 if prev_close[i] < 2 * cfg.split_threshold else 4.0
                    actions.append((i, t, "split", ratio[i]))
                if t in div_days[i]:
                    div[i] = round(prev_close[i] * cfg.dividend_yield / 4, 2)
                    if div[i] > 0:
                        actions.append((i, t, "dividend", div[i]))
            # total return identity: (close_t * ratio + div) / prev_close = exp(total)
            opens[t] = (prev_close * np.exp(on_ret[t]) - div) / ratio
            closes[t] = (prev_close * np.exp(total[t]) - div) / ratio
            price = closes[t]
        closes = np.maximum(closes, 0.01)
        opens = np.maximum(opens, 0.01)
        intra_sd = idio_sd * np.sqrt(1 - on_share) + np.sqrt(var)[:, None] * 0.6
        hi_ext = np.abs(rng.normal(0, 1, size=(T, N))) * intra_sd * 0.6
        lo_ext = np.abs(rng.normal(0, 1, size=(T, N))) * intra_sd * 0.6
        highs = np.maximum(opens, closes) * np.exp(hi_ext)
        lows = np.minimum(opens, closes) * np.exp(-lo_ext)
        vol_mult = np.exp(rng.normal(0, 0.3, size=(T, N)) + 8 * np.abs(total))
        volume = np.round(cfg.base_volume * rng.uniform(0.2, 3, size=N) * vol_mult)

        avail = self.cal.daily_bar_available_at(sessions, self.delay_minutes)
        frames = []
        for i in range(N):
            frames.append(pd.DataFrame({
                "symbol": self._sym(i), "session": sessions, "open": opens[:, i], "high": highs[:, i],
                "low": lows[:, i], "close": closes[:, i], "volume": volume[:, i].astype(float),
                "available_at": avail, "source": SOURCE,
            }))
        # a market "ETF": beta-1 exposure to m_t, no corporate actions
        mkt_close = 100 * np.exp(np.cumsum(m))
        mkt_prev = np.concatenate([[100.0], mkt_close[:-1]])
        mkt_open = mkt_prev * np.exp(m * on_share)
        mkt_ext = np.abs(rng.normal(0, 1, size=(T, 2))) * np.sqrt(var)[:, None] * 0.5
        frames.append(pd.DataFrame({
            "symbol": cfg.market_symbol, "session": sessions, "open": mkt_open,
            "high": np.maximum(mkt_open, mkt_close) * np.exp(mkt_ext[:, 0]),
            "low": np.minimum(mkt_open, mkt_close) * np.exp(-mkt_ext[:, 1]),
            "close": mkt_close, "volume": np.round(cfg.base_volume * 20 * np.exp(8 * np.abs(m))),
            "available_at": avail, "source": SOURCE,
        }))
        self._bars = pd.concat(frames, ignore_index=True).sort_values(["symbol", "session"], ignore_index=True)

        ca = pd.DataFrame(actions, columns=["i", "t", "action", "value"])
        if len(ca):
            ex = sessions[ca["t"].to_numpy()]
            # announced (conservatively) at the prior session's close
            prev = sessions[np.maximum(ca["t"].to_numpy() - 5, 0)]
            self._actions = pd.DataFrame({
                "symbol": [self._sym(i) for i in ca["i"]], "ex_date": ex, "action": ca["action"],
                "value": ca["value"].astype(float), "available_at": self.cal.session_closes(prev),
                "source": SOURCE,
            })
        else:
            self._actions = pd.DataFrame(columns=["symbol", "ex_date", "action", "value", "available_at", "source"])
        self._events = pd.DataFrame(events)

        cond_vol = np.sqrt(var * 252)
        vix = 100 * cond_vol * 1.15 + rng.normal(0, 0.8, size=T)
        self._vix = pd.DataFrame({
            "series_id": "SYN_VIX", "observation_date": sessions, "value": np.maximum(vix, 5.0),
            "available_at": avail, "source": SOURCE,
        })
        self._truth = {
            "sessions": sessions, "market_return": m, "regime": regime, "betas": betas,
            "sector_of": sector_of, "idio": e, "total_return": total,
        }
        self._generated = True

    # ---------------------------------------------------------------------------------------------
    def _sym(self, i: int) -> str:
        return f"SYN{i:03d}"

    def _ensure(self) -> None:
        if not self._generated:
            self.generate()

    def _ds(self, name: str, df: pd.DataFrame) -> Dataset:
        return Dataset(name, df.reset_index(drop=True),
                       Provenance(DataLabel.SIMULATED, SOURCE, notes=f"seed={self.cfg.seed}"))

    @property
    def symbols(self) -> list[str]:
        return [self._sym(i) for i in range(self.cfg.n_symbols)]

    def bars(self) -> Dataset:
        self._ensure()
        return self._ds("bars_daily", self._bars)

    def corporate_actions(self) -> Dataset:
        self._ensure()
        return self._ds("corporate_actions", self._actions)

    def earnings_events(self) -> Dataset:
        self._ensure()
        return self._ds("earnings_events", self._events)

    def vix(self) -> Dataset:
        self._ensure()
        return self._ds("macro_series", self._vix)

    @property
    def truth(self) -> dict:
        """Ground truth of the simulation (for calibration tests only)."""
        self._ensure()
        return self._truth


class SyntheticProvider(DataProvider):
    """Provider facade over :class:`SyntheticMarket` (label: SIMULATED)."""

    info = ProviderInfo("synthetic", DataLabel.SIMULATED, notes="Generated data for machinery tests.")

    def __init__(self, market: SyntheticMarket | None = None) -> None:
        self.market = market or SyntheticMarket()

    def _slice(self, ds: Dataset, col: str, symbols: list[str] | None, start: str, end: str,
               sym_col: str = "symbol") -> Dataset:
        df = ds.frame
        mask = (df[col] >= pd.Timestamp(start)) & (df[col] <= pd.Timestamp(end))
        if symbols is not None:
            mask &= df[sym_col].isin(symbols)
        return Dataset(ds.name, df[mask].reset_index(drop=True), ds.provenance)

    def get_daily_bars(self, symbols, start, end):
        return self._slice(self.market.bars(), "session", symbols, start, end)

    def get_corporate_actions(self, symbols, start, end):
        return self._slice(self.market.corporate_actions(), "ex_date", symbols, start, end)

    def get_earnings_events(self, symbols, start, end):
        ds = self.market.earnings_events()
        df = ds.frame
        sess = df["announce_time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        mask = (sess >= pd.Timestamp(start)) & (sess <= pd.Timestamp(end)) & df["symbol"].isin(symbols)
        return Dataset(ds.name, df[mask].reset_index(drop=True), ds.provenance)

    def get_macro_series(self, series_id, start, end, vintages=False):
        if series_id != "SYN_VIX":
            raise KeyError(f"synthetic provider has no series {series_id}")
        return self._slice(self.market.vix(), "observation_date", None, start, end)
