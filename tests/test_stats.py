import numpy as np
import pandas as pd
import pytest

from quantlab.stats.core import (
    bootstrap_ci, circular_shift_test, effective_n, mean_test_hac, random_entry_test, sign_flip_test,
    stationary_bootstrap_indices,
)
from quantlab.stats.multiple import (
    adjust, benjamini_hochberg, benjamini_yekutieli, bonferroni, deflated_sharpe_ratio, expected_max_sharpe, holm,
    min_track_record_length, probabilistic_sharpe_ratio,
)
from quantlab.stats.signal import cross_sectional_ic, normal_shrinkage_posterior


def test_multiple_testing_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(bonferroni(p), [0.04, 0.16, 0.12, 0.02])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    c = 1 + 1 / 2 + 1 / 3 + 1 / 4
    assert np.allclose(benjamini_yekutieli(p), np.minimum(benjamini_hochberg(p) * c, 1))
    assert np.allclose(adjust([0.01, np.nan], "bonferroni"), [0.02, 1.0])  # NaN counts toward family


def test_hac_t_test_size_under_null_is_close_to_nominal():
    """Under a true null with autocorrelated data (overlapping sums), rejection rate ≈ 5%."""
    rng = np.random.default_rng(0)
    rejections = 0
    trials = 400
    for _ in range(trials):
        e = rng.normal(size=520)
        x = pd.Series(e).rolling(5).sum().dropna().to_numpy()  # overlapping 5-day returns
        rejections += mean_test_hac(x, lags=8, alternative="two-sided").p_value < 0.05
    assert 0.02 <= rejections / trials <= 0.10


def test_naive_t_test_would_over_reject_overlapping_data():
    """Documents why HAC is needed: iid formula on overlapping sums rejects far too often."""
    rng = np.random.default_rng(1)
    naive = 0
    for _ in range(300):
        x = pd.Series(rng.normal(size=520)).rolling(10).sum().dropna().to_numpy()
        t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        naive += abs(t) > 1.96
    assert naive / 300 > 0.2


def test_hac_detects_real_effect():
    rng = np.random.default_rng(2)
    x = rng.normal(0.2, 1, size=1000)
    assert mean_test_hac(x).p_value < 0.001


def test_bootstrap_ci_covers_truth_and_indices_valid():
    rng = np.random.default_rng(3)
    idx = stationary_bootstrap_indices(100, 5, rng)
    assert idx.min() >= 0 and idx.max() < 100
    covered = 0
    for s in range(60):
        x = np.random.default_rng(100 + s).normal(0.1, 1, 400)
        ci = bootstrap_ci(x, n_boot=400, seed=s, mean_block=1)
        covered += ci["lo"] <= 0.1 <= ci["hi"]
    assert covered / 60 >= 0.85


def test_sign_flip_and_circular_shift():
    rng = np.random.default_rng(4)
    assert sign_flip_test(rng.normal(0.3, 1, 200), n_perm=2000).p_value < 0.01
    assert sign_flip_test(rng.normal(0.0, 1, 200), n_perm=2000).p_value > 0.01
    idx = pd.bdate_range("2010-01-01", periods=800)
    s = pd.Series(rng.normal(size=800), index=idx)
    y_related = s * 0.3 + pd.Series(rng.normal(size=800), index=idx)
    y_noise = pd.Series(rng.normal(size=800), index=idx)
    assert circular_shift_test(s, y_related, n_perm=500).p_value < 0.01
    assert circular_shift_test(s, y_noise, n_perm=500).p_value > 0.01


def test_random_entry_controls_for_drift():
    """A long-only rule in an uptrending market beats zero but NOT random entries."""
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    fwd = pd.DataFrame(rng.normal(0.006, 0.02, size=(1500, 4)), index=idx, columns=list("ABCD"))
    pick = rng.random((1500, 4)) < 0.05
    rows = [(idx[i], fwd.columns[j]) for i, j in zip(*np.nonzero(pick))]
    trades = pd.DataFrame({"signal_session": [r[0] for r in rows], "symbol": [r[1] for r in rows], "direction": 1.0})
    trades["gross_ret"] = [fwd.loc[d, s] for d, s in rows]
    assert mean_test_hac(trades["gross_ret"]).p_value < 0.01  # "significant" vs zero ...
    res = random_entry_test(trades, fwd, n_perm=1000)
    assert res.p_value > 0.05  # ... but no timing skill


def test_effective_n_collapses_clusters():
    sessions = pd.bdate_range("2020-01-01", periods=100)
    same_day = pd.Series([sessions[10]] * 20)
    assert effective_n(same_day, sessions, holding=5) == 1
    daily = pd.Series(sessions[:50])
    assert effective_n(daily, sessions, holding=5) == 10


def test_psr_dsr_behave_sensibly():
    assert probabilistic_sharpe_ratio(0.1, 1000) > 0.99
    assert probabilistic_sharpe_ratio(0.0, 1000) == pytest.approx(0.5)
    # the more strategies tried, the higher the bar
    assert expected_max_sharpe(1000, 0.05) > expected_max_sharpe(10, 0.05) > 0
    d1 = deflated_sharpe_ratio(0.08, 1000, n_trials=1)
    d100 = deflated_sharpe_ratio(0.08, 1000, n_trials=100)
    assert d100["dsr"] < d1["dsr"]
    assert min_track_record_length(0.1) < min_track_record_length(0.05)
    assert min_track_record_length(-0.1) == float("inf")


def test_dsr_calibration_best_of_many_noise_strategies_is_not_significant():
    rng = np.random.default_rng(6)
    n, k = 750, 200
    R = rng.normal(0, 0.01, size=(n, k))
    srs = R.mean(0) / R.std(0, ddof=1)
    best = srs.max()
    assert probabilistic_sharpe_ratio(best, n) > 0.95  # naive PSR is fooled by selection
    assert deflated_sharpe_ratio(best, n, n_trials=k, sr_std_across_trials=srs.std())["dsr"] < 0.95


def test_ic_and_shrinkage():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=200)
    sig = pd.DataFrame(rng.normal(size=(200, 20)), index=idx)
    fwd = sig * 0.2 + rng.normal(size=(200, 20))
    ic = cross_sectional_ic(sig, fwd)
    assert 0.1 < ic.mean() < 0.35
    post = normal_shrinkage_posterior(0.01, 0.01)
    assert post["post_mean"] == pytest.approx(0.005)  # halfway to zero with prior sd = se
    assert 0.5 < post["p_positive"] < 0.8
