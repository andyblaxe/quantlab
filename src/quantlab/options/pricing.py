"""Option pricing baseline: Black–Scholes–Merton (continuous dividend yield), Greeks, implied
volatility, Cox–Ross–Rubinstein binomial trees for American exercise, and parity/no-arbitrage checks.

Black–Scholes is a *baseline*, not a claim that markets follow it: implied volatility is simply the
number that makes the model match an observed price. All functions are vectorised over numpy arrays.

Conventions: ``T`` in years, ``r`` and ``q`` continuously compounded, ``right`` ∈ {"C", "P"},
``sigma`` annualised. Greeks per one option on one share (multiply by contract multiplier ×
quantity for positions); vega per 1.00 vol (divide by 100 for "per vol point"), theta per year
(divide by 365 for per calendar day).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _arr(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


def _is_call(right) -> np.ndarray:
    r = np.asarray(right)
    return np.char.upper(r.astype(str)) == "C"


def d1_d2(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        vt = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vt
    return d1, d1 - vt


def bs_price(S, K, T, r, q, sigma, right):
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    call = _is_call(right)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    dfq, dfr = np.exp(-q * T), np.exp(-r * T)
    c = S * dfq * norm.cdf(d1) - K * dfr * norm.cdf(d2)
    p = K * dfr * norm.cdf(-d2) - S * dfq * norm.cdf(-d1)
    price = np.where(call, c, p)
    # expired or zero-vol: discounted intrinsic on the forward
    fwd_intr = np.where(call, np.maximum(S * dfq - K * dfr, 0), np.maximum(K * dfr - S * dfq, 0))
    degenerate = (T <= 0) | (sigma <= 0)
    intr = np.where(call, np.maximum(S - K, 0), np.maximum(K - S, 0))
    return np.where(T <= 0, intr, np.where(degenerate, fwd_intr, price))


def bs_greeks(S, K, T, r, q, sigma, right) -> dict[str, np.ndarray]:
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    call = _is_call(right)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    dfq, dfr = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)
    sq = np.sqrt(T)
    delta = np.where(call, dfq * norm.cdf(d1), -dfq * norm.cdf(-d1))
    gamma = dfq * pdf / (S * sigma * sq)
    vega = S * dfq * pdf * sq
    theta_c = -S * dfq * pdf * sigma / (2 * sq) - r * K * dfr * norm.cdf(d2) + q * S * dfq * norm.cdf(d1)
    theta_p = -S * dfq * pdf * sigma / (2 * sq) + r * K * dfr * norm.cdf(-d2) - q * S * dfq * norm.cdf(-d1)
    rho = np.where(call, K * T * dfr * norm.cdf(d2), -K * T * dfr * norm.cdf(-d2))
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": np.where(call, theta_c, theta_p), "rho": rho}


def price_bounds(S, K, T, r, q, right):
    """No-arbitrage bounds for a European option price (lower, upper)."""
    S, K, T, r, q = _arr(S, K, T, r, q)
    call = _is_call(right)
    dfq, dfr = np.exp(-q * T), np.exp(-r * T)
    lo = np.where(call, np.maximum(S * dfq - K * dfr, 0), np.maximum(K * dfr - S * dfq, 0))
    hi = np.where(call, S * dfq, K * dfr)
    return lo, hi


def implied_vol(price, S, K, T, r, q, right, lo: float = 1e-4, hi: float = 5.0) -> np.ndarray:
    """European implied volatility (Brent). NaN where the price violates no-arbitrage bounds or T<=0.

    A NaN is information (bad quote, stale print, or American early-exercise premium), never filled in.
    """
    price, S, K, T, r, q = np.broadcast_arrays(*_arr(price, S, K, T, r, q))
    right = np.broadcast_to(np.asarray(right), price.shape)
    out = np.full(price.shape, np.nan)
    lb, ub = price_bounds(S, K, T, r, q, right)
    for idx in np.ndindex(price.shape):
        p = price[idx]
        if not np.isfinite(p) or T[idx] <= 0 or p <= lb[idx] + 1e-12 or p >= ub[idx]:
            continue
        f = lambda s: float(bs_price(S[idx], K[idx], T[idx], r[idx], q[idx], s, right[idx])) - p
        try:
            if f(lo) > 0 or f(hi) < 0:
                continue
            out[idx] = brentq(f, lo, hi, xtol=1e-10, maxiter=200)
        except ValueError:
            continue
    return out


def crr_price(S, K, T, r, q, sigma, right, american: bool = True, steps: int = 400) -> float:
    """Cox–Ross–Rubinstein binomial price (scalar). American exercise checked at every node."""
    if T <= 0:
        return float(max(S - K, 0) if right == "C" else max(K - S, 0))
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("binomial probabilities outside (0,1); increase steps")
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    ST = S * u ** (steps - j) * d**j
    V = np.maximum(ST - K, 0) if right == "C" else np.maximum(K - ST, 0)
    for n in range(steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            Sn = S * u ** (n - np.arange(n + 1)) * d ** np.arange(n + 1)
            ex = np.maximum(Sn - K, 0) if right == "C" else np.maximum(K - Sn, 0)
            V = np.maximum(V, ex)
    return float(V[0])


def early_exercise_premium(S, K, T, r, q, sigma, right, steps: int = 400) -> float:
    return crr_price(S, K, T, r, q, sigma, right, True, steps) - crr_price(S, K, T, r, q, sigma, right, False, steps)


def parity_gap(call, put, S, K, T, r, q):
    """European put–call parity residual: C − P − (S e^{-qT} − K e^{-rT}). ≈0 for European options."""
    call, put, S, K, T, r, q = _arr(call, put, S, K, T, r, q)
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))


def parity_arbitrage(call_bid, call_ask, put_bid, put_ask, S_bid, S_ask, K, T, r, q=0.0, cost: float = 0.0,
                     american: bool = True) -> dict:
    """Executable parity violation using bid/ask (not mids), net of ``cost`` per share.

    Conversion (sell call, buy put, buy stock) and reversal (buy call, sell put, short stock) profits.
    For American options only the inequality bounds hold, which weakens reversals (early exercise)
    — flagged rather than assumed away. Retail feasibility is usually nil: these gaps close in
    microseconds and require hard-to-borrow stock for reversals.
    """
    dfr, dfq = np.exp(-r * T), np.exp(-q * T)
    conversion = call_bid - put_ask - (S_ask * dfq - K * dfr) - cost
    reversal = put_bid - call_ask + (S_bid * dfq - K * dfr) - cost
    return {"conversion_profit": float(conversion), "reversal_profit": float(reversal),
            "violation": bool(conversion > 0 or reversal > 0),
            "caveat": ("American exercise: reversal parity is an inequality; early exercise and dividends can erase it"
                       if american else "")}


# ---------------------------------------------------------------------------------------------
# Expected moves and distributions
# ---------------------------------------------------------------------------------------------
SQRT_2_OVER_PI = np.sqrt(2 / np.pi)


def expected_move_from_iv(S, sigma, T) -> dict:
    """One-standard-deviation move (S σ √T) and expected absolute move (≈0.798 × that)."""
    sd = float(S * sigma * np.sqrt(T))
    return {"one_sd_move": sd, "expected_abs_move": sd * SQRT_2_OVER_PI, "one_sd_pct": sd / S}


def expected_move_from_straddle(straddle_price, S) -> dict:
    """ATM straddle ≈ expected absolute move; implied 1-sd move ≈ straddle / 0.798."""
    return {"expected_abs_move": float(straddle_price), "one_sd_move": float(straddle_price / SQRT_2_OVER_PI),
            "expected_abs_pct": float(straddle_price / S)}


def implied_event_move(iv_front, T_front, iv_back, T_back) -> dict:
    """Implied one-day event (e.g. earnings) move from two expiries straddling the event.

    Assumes constant diffusive vol σ_d and one event jump with variance E² inside both expiries:
      σ1² T1 = σ_d² T1 + E²,  σ2² T2 = σ_d² T2 + E²
    ⇒ σ_d² = (σ2² T2 − σ1² T1) / (T2 − T1),  E² = σ1² T1 − σ_d² T1.
    Returns NaN when the term structure is inconsistent with the model (E² < 0).
    """
    if T_back <= T_front:
        raise ValueError("back expiry must be later than front expiry")
    var_d = (iv_back**2 * T_back - iv_front**2 * T_front) / (T_back - T_front)
    e2 = iv_front**2 * T_front - var_d * T_front
    if e2 <= 0 or var_d < 0:
        return {"event_sd": float("nan"), "event_expected_abs": float("nan"), "diffusive_vol": float(np.sqrt(max(var_d, 0))),
                "consistent": False}
    e = float(np.sqrt(e2))
    return {"event_sd": e, "event_expected_abs": e * SQRT_2_OVER_PI, "diffusive_vol": float(np.sqrt(var_d)), "consistent": True}


def prob_above(S, K, T, r, q, sigma) -> float:
    """Risk-neutral P(S_T > K) = N(d2). Not a real-world probability (it embeds risk premia)."""
    _, d2 = d1_d2(S, K, T, r, q, sigma)
    return float(norm.cdf(d2))


def lognormal_pdf(x, S, T, mu, sigma):
    """Density of S_T under GBM with drift mu (risk-neutral: mu = r − q)."""
    x = np.asarray(x, dtype=float)
    m = np.log(S) + (mu - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, np.exp(-((np.log(x) - m) ** 2) / (2 * s**2)) / (x * s * np.sqrt(2 * np.pi)), 0.0)


def breeden_litzenberger(strikes, call_prices, r, T) -> tuple[np.ndarray, np.ndarray]:
    """Risk-neutral density from call prices: f(K) = e^{rT} ∂²C/∂K². Negative values signal butterfly
    arbitrage or noisy quotes — returned as-is, not clipped, so they can be investigated."""
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_prices, dtype=float)
    d2 = np.gradient(np.gradient(C, K), K)
    return K, np.exp(r * T) * d2
