"""Model benchmarks with walk-forward training, and model explanation tools.

Order of evidence: random prediction → always-long (drift) → simple rules → linear/logistic models →
tree ensembles. A more complex model is only interesting if it beats the simpler ones **out of sample
after costs**; in-sample fit is never reported as evidence.

MODEL PERFORMANCE (out-of-sample, walk-forward) and MODEL EXPLANATION (permutation importance,
partial dependence, calibration) are returned separately. Feature importance describes what the
model uses — it is not evidence that a feature *causes* returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "logistic": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
    "gradient_boosting": lambda seed: HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05,
                                                                     l2_regularization=1.0, random_state=seed),
}


def stack_panel(features: dict[str, pd.DataFrame], target: pd.DataFrame) -> pd.DataFrame:
    """Long (date, symbol) design matrix; rows with any missing value dropped."""
    cols = {k: v.stack() for k, v in features.items()}
    cols["y"] = target.stack()
    df = pd.DataFrame(cols).dropna()
    df.index.names = ["session", "symbol"]
    return df


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame  # session, symbol, prob, y (forward return), fold
    fold_scores: pd.DataFrame
    last_model: object
    last_train: pd.DataFrame


def walk_forward_classifier(df: pd.DataFrame, model: str = "logistic", min_train_sessions: int = 756,
                            test_sessions: int = 252, purge: int = 5, seed: int = 0) -> WalkForwardResult:
    """Train on the past, predict P(forward return > 0) on the next block, roll forward (expanding)."""
    sessions = df.index.get_level_values("session").unique().sort_values()
    feats = [c for c in df.columns if c != "y"]
    preds, scores = [], []
    i, k = min_train_sessions, 0
    mdl, train = None, None
    while i + purge < len(sessions):
        tr_end, te_start = sessions[i - 1], sessions[i + purge]
        te_end = sessions[min(i + purge + test_sessions, len(sessions)) - 1]
        s = df.index.get_level_values("session")
        train, test = df[s <= tr_end], df[(s >= te_start) & (s <= te_end)]
        if len(test) == 0 or train["y"].gt(0).nunique() < 2:
            break
        mdl = MODELS[model](seed)
        mdl.fit(train[feats], (train["y"] > 0).astype(int))
        p = mdl.predict_proba(test[feats])[:, 1]
        preds.append(pd.DataFrame({"prob": p, "y": test["y"].to_numpy(), "fold": k}, index=test.index))
        hit = float(np.mean((p > 0.5) == (test["y"] > 0)))
        scores.append({"fold": k, "test_start": te_start, "test_end": te_end, "n": len(test), "hit_rate": hit,
                       "base_rate": float((test["y"] > 0).mean())})
        i += test_sessions
        k += 1
    return WalkForwardResult(pd.concat(preds) if preds else pd.DataFrame(), pd.DataFrame(scores), mdl, train)


def evaluate_predictions(pred: pd.DataFrame, cost: float, threshold: float = 0.55, seed: int = 0) -> dict:
    """Out-of-sample economic evaluation vs baselines: trade (long) when prob > threshold.

    Baselines: always-long (drift), random entries at the same frequency. All net of ``cost`` per trade.
    """
    if pred.empty:
        return {"status": "NO_PREDICTIONS"}
    rng = np.random.default_rng(seed)
    take = pred["prob"] > threshold
    model_net = pred.loc[take, "y"] - cost
    rand_mask = rng.random(len(pred)) < take.mean()
    out = {
        "n_oos": int(len(pred)), "n_trades": int(take.sum()), "trade_rate": float(take.mean()),
        "model_mean_net": float(model_net.mean()) if take.any() else None,
        "always_long_mean_net": float((pred["y"] - cost).mean()),
        "random_mean_net": float((pred.loc[rand_mask, "y"] - cost).mean()) if rand_mask.any() else None,
        "hit_rate": float(np.mean((pred["prob"] > 0.5) == (pred["y"] > 0))),
        "base_rate": float((pred["y"] > 0).mean()),
        "brier": float(np.mean((pred["prob"] - (pred["y"] > 0)) ** 2)),
        "brier_base_rate": float(np.mean(((pred["y"] > 0).mean() - (pred["y"] > 0)) ** 2)),
    }
    out["beats_baselines"] = bool(out["model_mean_net"] is not None
                                  and out["model_mean_net"] > max(out["always_long_mean_net"], out["random_mean_net"] or -np.inf))
    return out


def calibration_table(pred: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs realised frequency (are 60% predictions right 60% of the time?)."""
    b = pd.qcut(pred["prob"].rank(method="first"), bins, labels=False)
    return pred.assign(bin=b, hit=(pred["y"] > 0).astype(float)).groupby("bin").agg(
        mean_prob=("prob", "mean"), realized=("hit", "mean"), n=("hit", "size"))


def explain(result: WalkForwardResult, seed: int = 0, n_repeats: int = 5) -> dict:
    """MODEL EXPLANATION on the last training window: permutation importance and partial dependence.

    Descriptive only — importance ≠ causality, and correlated features share/steal importance.
    """
    if result.last_model is None or result.last_train is None:
        return {"status": "NO_MODEL"}
    feats = [c for c in result.last_train.columns if c != "y"]
    X, y = result.last_train[feats], (result.last_train["y"] > 0).astype(int)
    sample = X.sample(min(len(X), 5000), random_state=seed)
    pi = permutation_importance(result.last_model, sample, y.loc[sample.index], n_repeats=n_repeats, random_state=seed,
                                scoring="neg_log_loss")
    imp = pd.DataFrame({"feature": feats, "importance": pi.importances_mean, "sd": pi.importances_std}).sort_values(
        "importance", ascending=False)
    pdp = {}
    for f in feats[:5]:
        r = partial_dependence(result.last_model, sample, [f], grid_resolution=10)
        pdp[f] = {"grid": r["grid_values"][0].tolist(), "avg_prob": r["average"][0].tolist()}
    return {"status": "OK", "permutation_importance": imp.to_dict("records"), "partial_dependence": pdp,
            "prediction_distribution": np.quantile(result.predictions["prob"], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
            if not result.predictions.empty else None,
            "caveat": "Importance describes the fitted model, not causal effects; SHAP not implemented (optional dependency)."}
