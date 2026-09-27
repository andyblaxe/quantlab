import numpy as np
import pandas as pd
import pytest

from quantlab.features import get_feature
from quantlab.research.labels import forward_return
from quantlab.research.models import (
    calibration_table, evaluate_predictions, explain, stack_panel, walk_forward_classifier,
)
from quantlab.research.relative_value import (
    adf_pvalue, correlation_clusters, engle_granger, explained_variance, half_life, pca_residuals, rolling_spread,
)


def _pair(n=1500, seed=0, coint=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    x = np.cumsum(rng.normal(0, 0.01, n))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.95 * e[t - 1] + rng.normal(0, 0.005)
    y = 0.8 * x + (e if coint else np.cumsum(rng.normal(0, 0.01, n)))
    return pd.Series(np.exp(y + 4), idx), pd.Series(np.exp(x + 4), idx)


def test_cointegration_and_half_life():
    y, x = _pair()
    eg = engle_granger(y, x)
    assert eg["p_value"] < 0.01 and eg["hedge_ratio_full_sample"] == pytest.approx(0.8, abs=0.1)
    yn, xn = _pair(coint=False, seed=1)
    assert engle_granger(yn, xn)["p_value"] > 0.05
    rs = rolling_spread(y, x)
    assert rs["hedge_ratio"].dropna().between(0.5, 1.1).mean() > 0.9
    assert 5 < half_life(rs["spread"]) < 40  # AR(0.95) → ~13.5 sessions
    assert adf_pvalue(rs["spread"]) < 0.05


def test_rolling_spread_is_point_in_time():
    y, x = _pair()
    full = rolling_spread(y, x)
    cut = y.index[900]
    part = rolling_spread(y.loc[:cut], x.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], part)


def test_pca_residuals_remove_common_factor():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=700)
    m = rng.normal(0, 0.01, 700)
    R = pd.DataFrame({f"S{i}": m * rng.uniform(0.8, 1.2) + rng.normal(0, 0.005, 700) for i in range(8)}, index=idx)
    assert explained_variance(R)[0] > 0.5
    res = pca_residuals(R, n_factors=1, window=252).dropna()
    assert res.corr().abs().to_numpy()[np.triu_indices(8, 1)].mean() < R.corr().to_numpy()[np.triu_indices(8, 1)].mean()
    cl = correlation_clusters(pd.concat([R, pd.DataFrame(rng.normal(size=(700, 2)), index=idx, columns=["N1", "N2"])], axis=1))
    assert len({cl[f"S{i}"] for i in range(8)}) == 1 and cl["N1"] != cl["S0"]


def _design(small_panel, noise_only=False):
    feats = {f: get_feature(f).compute(small_panel) for f in ["zret(h=1,vol_n=63)", "rsi(n=14)", "sma_ratio(n=50)"]}
    y = forward_return(small_panel, 1)
    if noise_only:
        rng = np.random.default_rng(0)
        y = pd.DataFrame(rng.normal(0, 0.01, y.shape), index=y.index, columns=y.columns).where(y.notna())
    return stack_panel(feats, y)


def test_walk_forward_model_has_no_edge_on_noise(small_panel):
    df = _design(small_panel, noise_only=True)
    res = walk_forward_classifier(df, "logistic", min_train_sessions=500, test_sessions=250)
    assert res.fold_scores["test_start"].min() > df.index.get_level_values("session").unique()[499]  # OOS only
    ev = evaluate_predictions(res.predictions, cost=0.001, threshold=0.5)
    assert abs(ev["hit_rate"] - 0.5) < 0.05  # coin flip on noise
    cal = calibration_table(res.predictions)
    assert len(cal) == 10


def test_model_explanation_and_boosting(small_panel):
    df = _design(small_panel)
    res = walk_forward_classifier(df, "gradient_boosting", min_train_sessions=750, test_sessions=375)
    ex = explain(res, n_repeats=2)
    assert ex["status"] == "OK" and {r["feature"] for r in ex["permutation_importance"]} == set(df.columns) - {"y"}
    assert "not causal" in ex["caveat"] or "causal" in ex["caveat"]
