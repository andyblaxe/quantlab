"""Option-chain analytics: IVs from bid/ask/mid, term structure, skew, smile, surface sanity checks,
plus a SIMULATED chain generator for tests and demos (labelled MODEL_PRICED — never research evidence
about option mispricing, since model prices cannot reveal model mispricing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.options.pricing import bs_greeks, bs_price, implied_vol


def enrich_chain(chain: pd.DataFrame, r: float, q: float = 0.0, session_col: str = "session",
                 max_rel_spread: float = 0.25) -> pd.DataFrame:
    """Add T, mid, spread, moneyness and IVs (bid/mid/ask) and Greeks at mid IV.

    ``chain`` follows the OPTION_QUOTES_EOD schema. Time to expiry uses calendar days/365 from the quote
    session to expiry (EOD convention).
    """
    c = chain.copy()
    c["T"] = (pd.to_datetime(c["expiration"]) - pd.to_datetime(c[session_col])).dt.days.clip(lower=0) / 365.0
    c["mid"] = (c["bid"] + c["ask"]) / 2
    c["spread"] = c["ask"] - c["bid"]
    c["rel_spread"] = np.where(c["mid"] > 0, c["spread"] / c["mid"], np.nan)
    S = c["underlying_price"].to_numpy(float)
    K = c["strike"].to_numpy(float)
    T = c["T"].to_numpy(float)
    rt = c["right"].to_numpy()
    c["log_moneyness"] = np.log(K / S)
    for col in ("bid", "mid", "ask"):
        px = c[col].to_numpy(float)
        c[f"iv_{col}"] = implied_vol(np.where(px > 0, px, np.nan), S, K, T, r, q, rt)
    # Quotes near the minimum tick (bid 0 / ask 0.01) or with very wide spreads give meaningless mid IVs
    # (the tick, not the market, sets the price). They stay in the frame but are flagged and excluded
    # from skew/term-structure summaries.
    # Surface convention: use out-of-the-money (or near-ATM) options — deep ITM prices are mostly
    # intrinsic value, so the bid/ask spread swamps the time value that carries the IV information.
    otm = ((c["right"] == "C") & (c["strike"] >= 0.98 * c["underlying_price"])) | \
          ((c["right"] == "P") & (c["strike"] <= 1.02 * c["underlying_price"]))
    c["iv_reliable"] = (c["bid"] > 0) & (c["rel_spread"] < max_rel_spread) & c["iv_mid"].notna() & otm
    iv = c["iv_mid"].to_numpy(float)
    g = bs_greeks(S, K, np.maximum(T, 1e-9), r, q, np.where(np.isfinite(iv), iv, np.nan), rt)
    for k, v in g.items():
        c[f"model_{k}"] = v
    return c


def atm_term_structure(enriched: pd.DataFrame) -> pd.DataFrame:
    """ATM IV per expiry: average of call and put mid-IV at the strike nearest the spot."""
    rows = []
    for exp, g in enriched[enriched["iv_reliable"]].groupby("expiration"):
        k = g.loc[(g["strike"] - g["underlying_price"]).abs().idxmin(), "strike"]
        atm = g[g["strike"] == k]
        rows.append({"expiration": exp, "T": float(atm["T"].iloc[0]), "atm_strike": float(k),
                     "atm_iv": float(atm["iv_mid"].mean()),
                     "straddle_mid": float(atm["mid"].sum()) if len(atm) == 2 else np.nan})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def skew_25d(enriched: pd.DataFrame) -> pd.DataFrame:
    """Per expiry: IV(25-delta put) − IV(25-delta call), and put wing minus ATM (risk-reversal style)."""
    rows = []
    for exp, g in enriched[enriched["iv_reliable"]].groupby("expiration"):
        puts, calls = g[g["right"] == "P"].dropna(subset=["model_delta", "iv_mid"]), g[g["right"] == "C"].dropna(subset=["model_delta", "iv_mid"])
        if puts.empty or calls.empty:
            continue
        p25 = puts.iloc[(puts["model_delta"] + 0.25).abs().argsort().iloc[0]]
        c25 = calls.iloc[(calls["model_delta"] - 0.25).abs().argsort().iloc[0]]
        atm = g.iloc[(g["strike"] - g["underlying_price"]).abs().argsort().iloc[0]]
        rows.append({"expiration": exp, "T": float(g["T"].iloc[0]), "iv_put25": float(p25["iv_mid"]), "iv_call25": float(c25["iv_mid"]),
                     "risk_reversal_25d": float(p25["iv_mid"] - c25["iv_mid"]), "put_wing_minus_atm": float(p25["iv_mid"] - atm["iv_mid"])})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def arbitrage_checks(enriched: pd.DataFrame) -> dict:
    """Static-arbitrage diagnostics on mid prices: call-price convexity in strike (butterfly) and
    total implied variance non-decreasing in maturity (calendar). Violations usually mean stale or
    wide quotes, not free money — especially after bid/ask costs."""
    butterfly = 0
    for (exp, right), g in enriched.groupby(["expiration", "right"]):
        g = g.sort_values("strike")
        if len(g) < 3:
            continue
        m = g["mid"].to_numpy()
        k = g["strike"].to_numpy()
        slopes = np.diff(m) / np.diff(k)
        butterfly += int(np.sum(np.diff(slopes) < -1e-9))
    ts = atm_term_structure(enriched)
    tv = ts["atm_iv"] ** 2 * ts["T"]
    calendar = int(np.sum(np.diff(tv.to_numpy()) < -1e-9))
    return {"butterfly_violations": butterfly, "calendar_violations": calendar,
            "note": "Mid-price diagnostics; executable arbitrage requires crossing bid/ask and is rarely available."}


def simulated_chain(S: float, session: str, r: float = 0.04, q: float = 0.0, base_iv: float = 0.2,
                    skew: float = -0.1, term_slope: float = 0.02, event_move: float = 0.0,
                    event_before: str | None = None, expiries_days=(7, 14, 30, 60, 90),
                    strikes_pct=np.arange(0.8, 1.21, 0.025), rel_spread: float = 0.04, min_tick: float = 0.01,
                    strikes: list[float] | None = None) -> pd.DataFrame:
    """SIMULATED (model-priced) chain with skew, term structure and an optional event (earnings)
    variance bump for expiries after ``event_before``. Labelled by the caller as SIMULATED/MODEL_PRICED."""
    rows = []
    sess = pd.Timestamp(session)
    for d in expiries_days:
        exp = sess + pd.Timedelta(days=int(d))
        T = d / 365
        for K in (strikes if strikes is not None else [round(S * kp, 2) for kp in strikes_pct]):
            K = float(K)
            iv = base_iv + skew * np.log(K / S) + term_slope * np.sqrt(T)
            if event_move and event_before is not None and exp >= pd.Timestamp(event_before):
                iv = np.sqrt(iv**2 + event_move**2 / T)
            for right in ("C", "P"):
                mid = float(bs_price(S, K, T, r, q, iv, right))
                half = max(mid * rel_spread / 2, min_tick)
                bid = max(round(mid - half, 2), 0.0)
                rows.append({"underlying": "SIM", "session": sess, "expiration": exp, "strike": K, "right": right,
                             "bid": bid, "ask": round(mid + half, 2), "volume": 100.0, "open_interest": 1000.0,
                             "underlying_price": S, "true_iv": iv})
    return pd.DataFrame(rows)
