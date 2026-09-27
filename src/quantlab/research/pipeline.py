"""The research pipeline: registered hypothesis → evidence → decision → catalog.

Stage 1 — ``evaluate(hid)`` (development data only; the test partition is not in memory):
    backtest on the development view → partition trades into TRAIN / VALIDATION (purged) →
    screening decision on TRAIN → walk-forward of the whole parameter-selection process →
    stress tests (costs ×2, outliers, year concentration, parameter neighbours) →
    multiple-testing correction over the family (BY) and deflated Sharpe over all trials →
    regime / stability / capacity / Monte Carlo → validation decision → catalog status.
    A hypothesis that passes is FROZEN (its spec cannot change); one that fails is REJECTED and
    never touches the untouched test data.

Stage 2 — ``run_untouched_test(hid)``: unseal the vault once, evaluate the frozen spec on the TEST
partition, and ACCEPT or REJECT. No re-tuning is possible: the spec hash is fixed.

Everything is recorded: an experiment row (all numbers), artifacts (trades, daily returns), journal
entries, and catalog versions. Failed and inconclusive results are recorded exactly like successes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from quantlab.backtest.costs import get_cost_model
from quantlab.backtest.metrics import summarize_returns, summarize_trades, yearly_returns
from quantlab.montecarlo.simulate import MonteCarloConfig, monte_carlo
from quantlab.research.acceptance import AcceptanceCriteria, CriterionResult, decide
from quantlab.research.artifacts import ArtifactStore
from quantlab.research.catalog import (
    RESEARCH_GRADE, InvalidTransition, SignalStatus, evidence_grade, latest_record, signal_id_for, transition,
    upsert_record,
)
from quantlab.research.integrity import assess
from quantlab.research.data import ResearchData
from quantlab.research.hypotheses import (
    HypothesisSpec, HypothesisStatus, advance_hypothesis, hypothesis_status, load_hypothesis, require_registered,
)
from quantlab.research.labels import forward_return
from quantlab.research.measurements import MEASUREMENTS
from quantlab.research.regimes import performance_by_regime, regime_labels
from quantlab.research.registry import Registry, code_version
from quantlab.research.splits import SplitPlan, partition_of, walk_forward_folds
from quantlab.research.strategies import StrategyRun, run_strategy, template_kind
from quantlab.research.vault import TestVault
from quantlab.stats.core import bootstrap_ci, effective_n, mean_test_hac, random_entry_test
from quantlab.stats.multiple import adjust, deflated_sharpe_ratio
from quantlab.stats.signal import cross_sectional_ic, time_series_ic
from quantlab.features import check_no_lookahead, get_feature

CAPACITY_LEVELS = (100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)


class PipelineError(RuntimeError):
    pass


def _window(s: pd.Series, a, b) -> pd.Series:
    return s.loc[(s.index >= a) & (s.index <= b)]


class ResearchPipeline:
    def __init__(self, reg: Registry, data: ResearchData, plan: SplitPlan, artifacts: ArtifactStore,
                 capital: float = 100_000.0, max_concurrent: int = 5, seed: int = 20260927,
                 n_boot: int = 1000, n_perm: int = 1000, mc_paths: int = 1000,
                 wf_min_train: int = 756, wf_test: int = 252) -> None:
        self.reg, self.data, self.plan, self.art = reg, data, plan, artifacts
        self.vault = TestVault(reg, plan)
        self.capital, self.max_concurrent, self.seed = capital, max_concurrent, seed
        self.n_boot, self.n_perm, self.mc_paths = n_boot, n_perm, mc_paths
        self.wf_min_train, self.wf_test = wf_min_train, wf_test
        self.all_sessions = data.panel["close"].index
        self.bounds = plan.boundaries(self.all_sessions)

    # ------------------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------------------
    def _run(self, panel, spec: HypothesisSpec, params: dict | None = None, cost=None) -> StrategyRun:
        return run_strategy(panel, spec.strategy, params if params is not None else spec.params,
                            self.data.universes[spec.universe], cost or get_cost_model(spec.cost_profile),
                            holding=spec.holding_period, execution=spec.execution, capital=self.capital,
                            max_concurrent=self.max_concurrent)

    def _benchmark_daily(self, panel, spec: HypothesisSpec) -> pd.Series:
        bh = run_strategy(panel, "buy_and_hold", {}, self.data.universes[spec.universe],
                          get_cost_model(spec.cost_profile), execution=spec.execution, capital=self.capital)
        return bh.daily_net

    def _warmup_end(self, spec: HypothesisSpec) -> pd.Timestamp:
        """First session at which every feature the strategy uses has enough history."""
        specs = [c[0] for c in spec.params.get("conditions", [])] + ([spec.params["feature"]] if "feature" in spec.params else [])
        lb = max([get_feature(f).lookback for f in specs], default=0)
        return self.all_sessions[min(lb + 1, len(self.all_sessions) - 1)]

    def _partition(self, run: StrategyRun, name: str, panel, spec, fwd, bh_daily) -> dict:
        a, b = self.bounds[name]
        a = max(a, self._warmup_end(spec))  # never score a strategy (or its benchmark) before it can act
        tr = run.trades
        if len(tr):
            tr = tr[partition_of(tr["entry_session"], tr["exit_session"], self.bounds) == name]
        dg, dn = _window(run.daily_gross, a, b), _window(run.daily_net, a, b)
        sessions = self.all_sessions
        n_eff = effective_n(tr["entry_session"], sessions, spec.holding_period) if len(tr) else 0
        out: dict[str, Any] = {
            "partition": name, "start": str(a.date()), "end": str(b.date()),
            "trades_gross": summarize_trades(tr["gross_ret"]) if len(tr) else {"n_trades": 0},
            "trades_net": summarize_trades(tr["net_ret"]) if len(tr) else {"n_trades": 0},
            "daily_gross": summarize_returns(dg), "daily_net": summarize_returns(dn),
            "effective_n": n_eff,
        }
        alt = spec.alternative if spec.alternative in ("greater", "less") else "two-sided"
        out["hac_daily_net"] = mean_test_hac(dn, lags=max(spec.holding_period, 5), alternative=alt).to_dict()
        p_parts = [out["hac_daily_net"]["p_value"]]
        if spec.primary_metric == "excess_vs_buy_and_hold":
            ex = (dn - _window(bh_daily, a, b)).dropna()
            out["excess_vs_buy_and_hold"] = {
                **mean_test_hac(ex, lags=max(spec.holding_period, 5), alternative=alt).to_dict(),
                "ann_excess": float(ex.mean() * 252),
                "sharpe_strategy": out["daily_net"].get("sharpe"),
                "sharpe_buy_and_hold": summarize_returns(_window(bh_daily, a, b)).get("sharpe"),
            }
            p_parts = [out["excess_vs_buy_and_hold"]["p_value"]]
        elif template_kind(spec.strategy) in ("event", "position") and len(tr) >= 5:
            elig = pd.DataFrame(True, index=fwd.index, columns=self.data.universes[spec.universe])
            out["random_entry"] = random_entry_test(tr, fwd, n_perm=self.n_perm, seed=self.seed, eligible=elig).to_dict()
            p_parts.append(out["random_entry"]["p_value"])
        if len(tr) >= 5:
            out["bootstrap_mean_net_trade"] = bootstrap_ci(tr["net_ret"], n_boot=self.n_boot, seed=self.seed)
        # conservative primary p-value: the strategy must beat zero after costs AND the relevant null
        ps = [p for p in p_parts if p is not None and np.isfinite(p)]
        out["p_primary"] = float(max(ps)) if ps else np.nan
        out["_trades"] = tr
        return out

    def _family_pvalues(self, family: str, this_hid: str, this_p: float) -> dict:
        hyps = self.reg.find("hypotheses", family=family)
        ps, ids = [], []
        for h in hyps:
            if h["id"] == this_hid:
                p = this_p
            else:
                ex = self.reg.find("experiments", order="DESC", limit=1, hypothesis_id=h["id"], kind="development")
                p = ex[0]["payload"].get("train", {}).get("p_primary") if ex else None
            ps.append(np.nan if p is None else p)
            ids.append(h["id"])
        q = adjust(ps, "by")
        return {"family": family, "family_size": len(ids), "method": "benjamini_yekutieli",
                "q_value": float(q[ids.index(this_hid)]), "n_untested_counted_as_p1": int(np.isnan(ps).sum())}

    def _trial_sharpes(self) -> list[float]:
        vals = []
        for ex in self.reg.find("experiments", kind="development"):
            v = ex["payload"].get("train", {}).get("daily_net", {}).get("sharpe")
            if v is not None:
                vals.append(v / np.sqrt(252))
        return vals

    def _walk_forward(self, dev_panel, spec: HypothesisSpec, base_run: StrategyRun) -> dict:
        candidates = [spec.params] + list(spec.param_neighbors)
        runs = [base_run] + [self._run(dev_panel, spec, p) for p in spec.param_neighbors]
        folds = walk_forward_folds(dev_panel["close"].index, self.wf_min_train, self.wf_test,
                                   purge_sessions=max(spec.holding_period, 1), end=self.plan.development_end)
        rows, oos = [], []
        for f in folds:
            scores = []
            for r in runs:
                d = _window(r.daily_net, f.train_start, f.train_end)
                sd = d.std()
                scores.append(d.mean() / sd if sd and sd > 0 else -np.inf)
            k = int(np.argmax(scores))
            d_oos = _window(runs[k].daily_net, f.test_start, f.test_end)
            tr = runs[k].trades
            tr = tr[(tr["entry_session"] >= f.test_start) & (tr["exit_session"] <= f.test_end)] if len(tr) else tr
            oos.append(d_oos)
            rows.append({"fold": f.index, "train_end": str(f.train_end.date()), "test_start": str(f.test_start.date()),
                         "test_end": str(f.test_end.date()), "selected_params": candidates[k],
                         "oos_total_net_return": float((1 + d_oos).prod() - 1), "oos_n_trades": int(len(tr)),
                         "oos_mean_net_trade": float(tr["net_ret"].mean()) if len(tr) else None})
        if not rows:
            return {"n_folds": 0}
        pooled = pd.concat(oos)
        pos = np.mean([r["oos_total_net_return"] > 0 for r in rows])
        return {"n_folds": len(rows), "folds": rows, "positive_fold_frac": float(pos),
                "pooled_oos": summarize_returns(pooled), "selection_varied": len({str(r["selected_params"]) for r in rows}) > 1,
                "_pooled": pooled}

    def _stability(self, trades: pd.DataFrame) -> dict:
        if len(trades) < 20:
            return {"assessment": "INSUFFICIENT_DATA"}
        t = pd.DatetimeIndex(trades["entry_session"])
        years = (t - t.min()).days / 365.25
        y = trades["net_ret"].to_numpy()
        reg = sps.linregress(years, y)
        half = len(trades) // 2
        first, second = float(np.mean(y[:half])), float(np.mean(y[half:]))
        if first > 0 and second <= 0:
            a = "DISAPPEARED"
        elif reg.pvalue < 0.10:
            a = "STRENGTHENED" if reg.slope > 0 else "WEAKENED"
        else:
            a = "STABLE" if second > 0 else "NO_EDGE_IN_EITHER_HALF"
        by_year = trades.groupby(t.year)["net_ret"].agg(["mean", "count"]).reset_index(names="year")
        return {"assessment": a, "slope_per_year": float(reg.slope), "slope_p": float(reg.pvalue),
                "first_half_mean": first, "second_half_mean": second, "by_year": by_year.to_dict("records")}

    def _capacity(self, trades: pd.DataFrame, spec: HypothesisSpec) -> dict:
        if trades.empty or "participation" not in trades:
            return {"assessment": "NOT_ESTIMATED"}
        cm = get_cost_model(spec.cost_profile)
        notional = self.capital / self.max_concurrent
        med_part = float(trades["participation"].median())
        max_trade = notional * cm.max_participation / med_part if med_part > 0 else np.nan
        levels = {}
        for cap in CAPACITY_LEVELS:
            pos = cap / self.max_concurrent
            one_share_unaffordable = float((trades["entry_price_raw"] > pos).mean())
            levels[str(int(cap))] = {
                "position_size": pos, "frac_trades_unaffordable_whole_share": one_share_unaffordable,
                "median_participation": med_part * pos / notional,
                "feasible": bool(one_share_unaffordable < 0.5 and med_part * pos / notional <= cm.max_participation),
            }
        return {"max_trade_notional_at_participation_cap": max_trade,
                "strategy_capacity_estimate": max_trade * self.max_concurrent if np.isfinite(max_trade) else None,
                "trades_per_year": float(len(trades) / max(1e-9, (trades["entry_session"].max() - trades["entry_session"].min()).days / 365.25)),
                "by_capital": levels}

    def _ic(self, dev_panel, spec, fwd) -> float | None:
        feats = [c[0] for c in spec.params.get("conditions", [])] or ([spec.params["feature"]] if "feature" in spec.params else [])
        if not feats:
            return None
        a, b = self.bounds["train"]
        x = get_feature(feats[0]).compute(dev_panel)[self.data.universes[spec.universe]]
        y = fwd[x.columns]
        x, y = x.loc[a:b], y.loc[a:b]
        if x.shape[1] >= 5:
            ic = cross_sectional_ic(x, y)
            return float(ic.mean()) if len(ic) else None
        return time_series_ic(x.stack(), y.stack())

    def _lookahead_check(self, panel, spec: HypothesisSpec) -> dict[str, list[str]]:
        """Automatic truncation test on every feature the hypothesis uses (dates where it failed)."""
        out = {}
        for f in self._condition_features(spec):
            try:
                out[f] = [str(t.date()) for t in check_no_lookahead(get_feature(f), panel, n_checks=4, seed=self.seed)]
            except ValueError as e:  # panel too short to test: record, don't guess
                out[f] = [f"untestable: {e}"]
        return out

    def _condition_features(self, spec: HypothesisSpec) -> list[str]:
        return [c[0] for c in spec.params.get("conditions", [])] + ([spec.params["feature"]] if "feature" in spec.params else [])

    def _annotate_trades(self, panel, spec, trades: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        """Attach the feature values and regime labels known at each trade's signal time."""
        if trades.empty:
            return trades
        out = trades.copy()
        when = pd.DatetimeIndex(out["signal_session"])
        for f in self._condition_features(spec):
            vals = get_feature(f).compute(panel)
            out[f"feat:{f}"] = [vals.at[d, sym] if (d in vals.index and sym in vals.columns) else np.nan
                                for d, sym in zip(when, out["symbol"])]
        for col in labels.columns:
            out[f"regime:{col}"] = labels[col].reindex(when).to_numpy()
        return out

    def _signal_strength(self, spec, trades: pd.DataFrame) -> dict:
        """Mean net trade return by quintile of the primary feature at entry (monotonicity check)."""
        feats = self._condition_features(spec)
        if not feats or trades.empty or f"feat:{feats[0]}" not in trades or len(trades) < 50:
            return {"status": "NOT_AVAILABLE"}
        x = trades[f"feat:{feats[0]}"]
        ok = x.notna()
        if x[ok].nunique() < 5:
            return {"status": "NOT_AVAILABLE"}
        b = pd.qcut(x[ok].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        g = trades.loc[ok, "net_ret"].groupby(b, observed=True)
        rho = sps.spearmanr(x[ok], trades.loc[ok, "net_ret"]).statistic
        return {"status": "OK", "feature": feats[0],
                "buckets": [{"quintile": int(k), "n": int(v.count()), "mean_net": float(v.mean()),
                             "feature_range": [float(x[ok][b == k].min()), float(x[ok][b == k].max())]} for k, v in g],
                "spearman_feature_vs_net_return": float(rho)}

    @staticmethod
    def _strip(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    # ------------------------------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------------------------------
    def evaluate(self, hid: str, reason: str = "") -> dict:
        spec = load_hypothesis(self.reg, hid)
        require_registered(self.reg, hid, spec)
        if hypothesis_status(self.reg, hid) != HypothesisStatus.REGISTERED:
            raise PipelineError(f"{hid} is {hypothesis_status(self.reg, hid)}; development evaluation runs once")
        crit = AcceptanceCriteria(**spec.acceptance)
        if template_kind(spec.strategy) == "measurement":
            return self._evaluate_measurement(hid, spec, crit, reason)
        dev = self.vault.development_view(self.data.panel)
        if spec.universe not in self.data.universes:
            raise PipelineError(f"universe {spec.universe!r} not available in this dataset")
        fwd = forward_return(dev, spec.holding_period, spec.execution)
        run = self._run(dev, spec)
        bh = self._benchmark_daily(dev, spec) if spec.primary_metric == "excess_vs_buy_and_hold" else None
        train = self._partition(run, "train", dev, spec, fwd, bh)
        val = self._partition(run, "validation", dev, spec, fwd, bh)
        dev_trades = pd.concat([train["_trades"], val["_trades"]])

        # ---- screening (TRAIN only) ----
        t_net = train["trades_net"]
        screen_results = [
            CriterionResult("train_effective_n", None if train["effective_n"] < crit.min_effective_n_inconclusive else True,
                            train["effective_n"], crit.min_effective_n_inconclusive,
                            "enough independent observations to test at all"),
            CriterionResult("train_net_mean_positive", (t_net.get("mean", np.nan) > 0) if t_net.get("n_trades") else None,
                            t_net.get("mean"), 0.0, "average trade makes money after estimated costs"),
            CriterionResult("train_p_value", (train["p_primary"] < crit.screen_p_value) if np.isfinite(train["p_primary"]) else None,
                            train["p_primary"], crit.screen_p_value,
                            "conservative primary p-value (max of HAC and the relevant null test)"),
        ]
        screening = decide("screening", screen_results)

        # ---- full evidence for validation ----
        wf = self._walk_forward(dev, spec, run)
        family = self._family_pvalues(spec.family, hid, train["p_primary"])
        tr_daily = _window(run.daily_net, *self.bounds["train"])
        sr = tr_daily.mean() / tr_daily.std() if tr_daily.std() > 0 else np.nan
        # Criterion: DSR against the best of N *skill-less* trials, using the null sampling dispersion of
        # the Sharpe ratio. The empirical dispersion across our (heterogeneous, cost-dominated) trials is
        # reported as a diagnostic only: it conflates differences in costs/structure with luck and would
        # penalise genuine discoveries (see RESEARCH_METHODOLOGY.md §4).
        trials = self._trial_sharpes()
        n_trials = max(self.reg.count("hypotheses"), 1)
        dsr = deflated_sharpe_ratio(sr, len(tr_daily), n_trials=n_trials, sr_std_across_trials=None,
                                    skew=float(tr_daily.skew()), kurt=float(tr_daily.kurt() + 3))
        if len(trials) >= 5:
            emp = deflated_sharpe_ratio(sr, len(tr_daily), n_trials=n_trials, sr_std_across_trials=float(np.std(trials)),
                                        skew=float(tr_daily.skew()), kurt=float(tr_daily.kurt() + 3))
            dsr["diagnostic_empirical_dispersion"] = emp
        stress: dict[str, Any] = {}
        cost_run = self._run(dev, spec, cost=get_cost_model(spec.cost_profile).scaled(crit.cost_stress_multiplier))
        ct = cost_run.trades
        ct = ct[partition_of(ct["entry_session"], ct["exit_session"], self.bounds).isin(["train", "validation"])] if len(ct) else ct
        stress["cost_x2_mean_net_trade"] = float(ct["net_ret"].mean()) if len(ct) else None
        if len(dev_trades) >= 20:
            cut = dev_trades["net_ret"].quantile(1 - crit.outlier_trim_frac)
            stress["trimmed_mean_net_trade"] = float(dev_trades.loc[dev_trades["net_ret"] < cut, "net_ret"].mean())
            yearly = dev_trades.groupby(pd.DatetimeIndex(dev_trades["entry_session"]).year)["net_ret"].sum()
            total = yearly.sum()
            stress["max_single_year_share"] = float(yearly.max() / total) if total > 0 else None
            stress["pnl_by_year"] = {str(k): float(v) for k, v in yearly.items()}
        neigh = []
        for p in spec.param_neighbors:
            nr = self._run(dev, spec, p)
            nt = nr.trades
            nt = nt[partition_of(nt["entry_session"], nt["exit_session"], self.bounds).isin(["train", "validation"])] if len(nt) else nt
            neigh.append({"params": p, "n_trades": int(len(nt)), "mean_net_trade": float(nt["net_ret"].mean()) if len(nt) else None})
        stress["param_neighbors"] = neigh
        pos_frac = (np.mean([(n["mean_net_trade"] or -1) > 0 for n in neigh]) if neigh else None)
        stress["param_neighbors_positive_frac"] = pos_frac
        dev_boot = bootstrap_ci(dev_trades["net_ret"], n_boot=self.n_boot, seed=self.seed,
                                level=crit.bootstrap_ci_level) if len(dev_trades) >= 5 else None
        dev_eff_n = effective_n(dev_trades["entry_session"], self.all_sessions, spec.holding_period) if len(dev_trades) else 0

        v_net, t_sh, v_sh = val["trades_net"], train["daily_net"].get("sharpe"), val["daily_net"].get("sharpe")
        val_results = [
            CriterionResult("dev_effective_n", None if dev_eff_n < crit.min_effective_n_accept else True, dev_eff_n,
                            crit.min_effective_n_accept, "enough independent observations to accept"),
            CriterionResult("validation_effective_n", None if val["effective_n"] < crit.min_effective_n_inconclusive else True,
                            val["effective_n"], crit.min_effective_n_inconclusive, "validation sample adequate"),
            CriterionResult("validation_net_mean_positive", (v_net.get("mean", np.nan) > 0) if v_net.get("n_trades") else None,
                            v_net.get("mean"), 0.0, "edge persists on validation data after costs"),
            CriterionResult("validation_sharpe_vs_train",
                            None if t_sh is None or v_sh is None or not np.isfinite(t_sh) or not np.isfinite(v_sh) else bool(v_sh >= crit.validation_sharpe_ratio_of_train * max(t_sh, 0) and v_sh > 0),
                            v_sh, f"{crit.validation_sharpe_ratio_of_train} x train ({t_sh})", "no collapse from train to validation"),
            CriterionResult("walk_forward", None if not wf.get("n_folds") else bool(
                wf["positive_fold_frac"] >= crit.walk_forward_min_positive_fold_frac and (wf["pooled_oos"].get("sharpe") or -1) > 0),
                wf.get("positive_fold_frac"), crit.walk_forward_min_positive_fold_frac,
                "re-selecting parameters on past data and trading forward stays profitable"),
            CriterionResult("multiple_testing_by_q", bool(family["q_value"] < crit.fdr_q), family["q_value"], crit.fdr_q,
                            f"false-discovery-adjusted over {family['family_size']} hypotheses in family"),
            CriterionResult("deflated_sharpe", None if not np.isfinite(dsr["dsr"]) else bool(dsr["dsr"] >= crit.dsr_min_prob),
                            dsr["dsr"], crit.dsr_min_prob, f"Sharpe beats the luckiest of {dsr['n_trials']} null trials"),
            CriterionResult("bootstrap_ci_excludes_zero", None if dev_boot is None else bool(dev_boot["lo"] > 0),
                            None if dev_boot is None else dev_boot["lo"], 0.0, "lower CI bound of mean net trade > 0"),
            CriterionResult("cost_stress", None if stress["cost_x2_mean_net_trade"] is None else bool(stress["cost_x2_mean_net_trade"] > 0),
                            stress["cost_x2_mean_net_trade"], 0.0, f"still profitable at {crit.cost_stress_multiplier}x costs"),
            CriterionResult("outlier_robustness", None if "trimmed_mean_net_trade" not in stress else bool(stress["trimmed_mean_net_trade"] > 0),
                            stress.get("trimmed_mean_net_trade"), 0.0, f"profitable without the best {crit.outlier_trim_frac:.0%} of trades"),
            CriterionResult("year_concentration", None if "max_single_year_share" not in stress else bool(
                stress["max_single_year_share"] is not None and stress["max_single_year_share"] <= crit.max_single_year_pnl_share),
                stress.get("max_single_year_share"), crit.max_single_year_pnl_share, "no single year dominates P&L"),
            CriterionResult("parameter_sensitivity", None if pos_frac is None else bool(pos_frac >= crit.min_param_neighbor_positive_frac),
                            pos_frac, crit.min_param_neighbor_positive_frac, "neighbouring parameter sets also work"),
        ]
        validation = decide("validation", val_results) if screening.outcome == "PASS" else None

        # ---- descriptive evidence ----
        labels = regime_labels(dev, self.data.market_symbol,
                               vix=self.data.vix.loc[: self.plan.development_end] if self.data.vix is not None else None)
        regimes = performance_by_regime(dev_trades, labels)
        dev_trades = self._annotate_trades(dev, spec, dev_trades, labels)
        strength = self._signal_strength(spec, dev_trades)
        stability = self._stability(dev_trades)
        capacity = self._capacity(dev_trades, spec)
        ic = self._ic(dev, spec, fwd)
        min_pos = float(dev_trades["entry_price_raw"].median()) if "entry_price_raw" in dev_trades and len(dev_trades) else 0.0
        years = max(1e-9, (self.bounds["validation"][1] - self.bounds["train"][0]).days / 365.25)
        cm = get_cost_model(spec.cost_profile)
        mc = monte_carlo(dev_trades["net_ret"], MonteCarloConfig(
            n_paths=self.mc_paths, trades_per_year=max(1.0, len(dev_trades) / years),
            position_fraction=1.0 / self.max_concurrent, fixed_cost_per_trade=2 * cm.commission_min,
            min_position_value=min_pos, seed=self.seed))

        # ---- record ----
        results = {
            "hypothesis_id": hid, "spec_hash": spec.spec_hash(), "stage": "development",
            "data": self.data.describe(), "universe_flags": self.data.universe_flags.get(spec.universe, []),
            "split_plan": self.plan.to_dict(), "bounds": {k: [str(a.date()), str(b.date())] for k, (a, b) in self.bounds.items()},
            "code_version": code_version(), "seed": self.seed, "capital": self.capital, "max_concurrent": self.max_concurrent,
            "cost_model": cm.to_dict(), "strategy_meta": run.meta,
            "train": self._strip(train), "validation": self._strip(val), "walk_forward": self._strip(wf),
            "multiple_testing": family, "deflated_sharpe": dsr, "stress": stress,
            "dev_bootstrap_mean_net_trade": dev_boot, "dev_effective_n": dev_eff_n,
            "screening": screening.to_dict(), "validation_decision": validation.to_dict() if validation else None,
            "regimes": regimes, "stability": stability, "capacity": capacity, "information_coefficient": ic,
            "signal_strength": strength,
            "monte_carlo": mc,
            "integrity_inputs": {"lookahead_failures": self._lookahead_check(dev, spec), "cost_model": cm.to_dict(),
                                 "execution": spec.execution,
                                 "data_quality_issues": list(self.data.quality_issues)},
        }
        eid = self.reg.append("experiments", {"hypothesis_id": hid, "kind": "development",
                                              "status": (validation or screening).outcome}, results)
        refs = {"trades": self.art.save(eid, "dev_trades", dev_trades.reset_index(drop=True)),
                "daily": self.art.save(eid, "dev_daily", pd.DataFrame({"gross": run.daily_gross, "net": run.daily_net})),
                "walk_forward_oos": self.art.save(eid, "wf_oos", wf.get("_pooled", pd.Series(dtype=float)).rename("net"))}
        self.reg.append("experiments", {"hypothesis_id": hid, "kind": "artifacts", "status": "stored"},
                        {"experiment_id": eid, "artifacts": refs})
        outcome = self._apply_development_outcome(hid, spec, eid, screening, validation, results)
        self.reg.journal("evaluate_hypothesis", question=spec.question, hypothesis=spec.statement,
                         reason=reason or "pipeline development evaluation", experiment_id=eid, hypothesis_id=hid,
                         dataset=self.data.describe(), features=[c[0] for c in spec.params.get("conditions", [])],
                         model=spec.strategy, parameters=spec.params, results={
                             "train_p_primary": train["p_primary"], "train_mean_net_trade": t_net.get("mean"),
                             "validation_mean_net_trade": v_net.get("mean"), "q_value": family["q_value"],
                             "dsr": dsr["dsr"]},
                         conclusion=outcome["conclusion"], status=outcome["signal_status"], next_step=outcome["next_step"],
                         data_label=self.data.label.value)
        return {"experiment_id": eid, **outcome, "results": results}

    def _evaluate_measurement(self, hid: str, spec: HypothesisSpec, crit: AcceptanceCriteria, reason: str) -> dict:
        fn = MEASUREMENTS[spec.strategy]
        train_end = self.bounds["train"][1]
        train = fn(self.data, train_end, spec.params)
        dev = fn(self.data, self.plan.development_end, spec.params)
        p = train.get("p_primary", train.get("p_value")) if train.get("status") == "OK" else np.nan
        train["p_primary"] = p
        family = self._family_pvalues(spec.family, hid, p)
        if train.get("status") != "OK":
            outcome, concl = "INCONCLUSIVE", f"Measurement not possible: {train.get('status')} {train.get('missing', '')}".strip()
        elif train["effective_n"] < crit.min_effective_n_inconclusive:
            outcome, concl = "INCONCLUSIVE", f"Only {train['effective_n']} independent observations."
        elif family["q_value"] < crit.fdr_q and dev.get("status") == "OK" and np.sign(dev["estimate"]) == np.sign(train["estimate"]):
            outcome, concl = "SUPPORTED", (f"Relationship measured (estimate {train['estimate']:.4g}, BY q={family['q_value']:.3g}) "
                                           "and same sign over the full development period. " + train.get("conclusion_hint", ""))
        else:
            outcome, concl = "NOT_SUPPORTED", (f"No statistically reliable relationship after multiple-testing correction "
                                               f"(q={family['q_value']:.3g}).")
        grade = evidence_grade({"Data_Flags": set(self.data.universe_flags.get(spec.universe, []))})
        if grade != RESEARCH_GRADE:
            concl = f"{grade}: {concl}"
        results = {"hypothesis_id": hid, "spec_hash": spec.spec_hash(), "stage": "development", "kind": "measurement",
                   "evidence_grade": grade,
                   "data": self.data.describe(), "split_plan": self.plan.to_dict(), "code_version": code_version(),
                   "train": train, "development": dev, "multiple_testing": family, "outcome": outcome}
        eid = self.reg.append("experiments", {"hypothesis_id": hid, "kind": "development", "status": outcome}, results)
        advance_hypothesis(self.reg, hid, HypothesisStatus.EVALUATED, f"measurement {eid}", experiment_id=eid)
        advance_hypothesis(self.reg, hid, HypothesisStatus.CONCLUDED, concl, experiment_id=eid)
        nxt = ("Consider a tradable hypothesis built on this measurement (requires the data it needs)."
               if outcome == "SUPPORTED" else "None, unless new data changes the sample materially.")
        self.reg.journal("evaluate_measurement", question=spec.question, hypothesis=spec.statement,
                         reason=reason or "measurement", experiment_id=eid, hypothesis_id=hid, dataset=self.data.describe(),
                         results={"estimate": train.get("estimate"), "p_value": p, "q_value": family["q_value"]},
                         conclusion=concl, status=outcome, next_step=nxt, data_label=self.data.label.value)
        return {"experiment_id": eid, "signal_id": None, "signal_status": outcome, "conclusion": concl,
                "next_step": nxt, "results": results}

    def _apply_development_outcome(self, hid, spec, eid, screening, validation, r) -> dict:
        sid = signal_id_for(hid)
        rec = self._record_fields(spec, hid, eid, r)
        upsert_record(self.reg, sid, rec, f"development evaluation {eid}")
        advance_hypothesis(self.reg, hid, HypothesisStatus.EVALUATED, f"development evaluation {eid}", experiment_id=eid)
        failed = lambda d: ", ".join(x.name for x in d.failed) or "none"
        unev = lambda d: ", ".join(x.name for x in d.unevaluable) or "none"
        if screening.outcome == "FAIL":
            transition(self.reg, sid, SignalStatus.REJECTED, f"failed screening on training data: {failed(screening)}",
                       {"experiment_id": eid})
            advance_hypothesis(self.reg, hid, HypothesisStatus.CONCLUDED, "rejected at screening")
            return {"signal_id": sid, "signal_status": "REJECTED", "conclusion": f"Rejected at screening ({failed(screening)}).",
                    "next_step": "None for this hypothesis; the untouched test was not used."}
        if screening.outcome == "INCONCLUSIVE":
            return {"signal_id": sid, "signal_status": "EXPERIMENTAL",
                    "conclusion": f"Inconclusive at screening (insufficient evidence: {unev(screening)}).",
                    "next_step": "Needs more data (longer history or broader universe); not rejected, not promoted."}
        transition(self.reg, sid, SignalStatus.VALIDATING, "passed screening on training data", {"experiment_id": eid})
        if validation.outcome == "PASS":
            advance_hypothesis(self.reg, hid, HypothesisStatus.FROZEN, "passed validation; spec locked for untouched test",
                               experiment_id=eid)
            gate = assess(self.reg, sid)
            if not gate.promotable:
                return {"signal_id": sid, "signal_status": "VALIDATING",
                        "conclusion": f"{gate.grade}: passed screening and validation, but "
                                      + ("the evidence has unresolved integrity problems" if gate.integrity_failures
                                         else "there is not yet enough evidence")
                                      + f" ({'; '.join(gate.reasons())}).",
                        "next_step": "Untouched test withheld (vault stays sealed). "
                                     + ("Fix the data/method and re-register; this result cannot be promoted."
                                        if gate.integrity_failures else
                                        "Gather more data (longer history, broader universe) and re-register.")}
            return {"signal_id": sid, "signal_status": "VALIDATING",
                    "conclusion": "Passed screening and every validation criterion on development data.",
                    "next_step": "Run the one-time untouched test (run_untouched_test)."}
        if validation.outcome == "FAIL":
            transition(self.reg, sid, SignalStatus.REJECTED, f"failed validation: {failed(validation)}", {"experiment_id": eid})
            advance_hypothesis(self.reg, hid, HypothesisStatus.CONCLUDED, "rejected at validation")
            return {"signal_id": sid, "signal_status": "REJECTED",
                    "conclusion": f"Passed screening but failed validation ({failed(validation)}).",
                    "next_step": "None for this hypothesis; the untouched test was not used."}
        return {"signal_id": sid, "signal_status": "VALIDATING",
                "conclusion": f"Validation inconclusive (could not evaluate: {unev(validation)}).",
                "next_step": "Gather more data before any decision; untouched test not used."}

    def _record_fields(self, spec: HypothesisSpec, hid: str, eid: str, r: dict) -> dict:
        tn, dn = r["train"]["trades_net"], r["train"]["daily_net"]
        feats = [c[0] for c in spec.params.get("conditions", [])] or ([spec.params.get("feature")] if spec.params.get("feature") else [])
        mc = r["monte_carlo"]
        return {
            "Name": spec.name, "Description": spec.question, "Hypothesis": spec.statement, "Hypothesis_ID": hid,
            "Instrument_Universe": spec.universe, "Holding_Period": spec.holding_period, "Direction": spec.direction,
            "Features": feats, "Expected_Return": tn.get("mean"), "Median_Return": tn.get("median"),
            "Win_Rate": tn.get("win_rate"), "Average_Win": tn.get("avg_win"), "Average_Loss": tn.get("avg_loss"),
            "Profit_Factor": tn.get("profit_factor"), "Information_Coefficient": r["information_coefficient"],
            "Sharpe": dn.get("sharpe"), "Sortino": dn.get("sortino"), "Max_Drawdown": dn.get("max_drawdown"),
            "Volatility": dn.get("ann_vol"), "Sample_Size": tn.get("n_trades"),
            "Effective_Sample_Size": r["train"]["effective_n"], "P_Value": r["train"]["p_primary"],
            "Adjusted_P_Value": r["multiple_testing"]["q_value"],
            "Turnover": r["strategy_meta"].get("avg_turnover"),
            "Estimated_Transaction_Cost": (r["train"]["trades_gross"].get("mean", np.nan) - tn.get("mean", np.nan))
            if tn.get("n_trades") else None,
            "Estimated_Slippage": r["cost_model"]["slippage_bps"] / 1e4,
            "Capacity_Estimate": r["capacity"].get("strategy_capacity_estimate"),
            "Regime_Performance": r["regimes"], "In_Sample_Performance": {"trades_net": tn, "daily_net": dn},
            "Validation_Performance": {"trades_net": r["validation"]["trades_net"], "daily_net": r["validation"]["daily_net"]},
            "Walk_Forward_Performance": {k: v for k, v in r["walk_forward"].items() if k != "folds"},
            "Monte_Carlo_Results": {c: {k: mc["historical"][c][k] for k in ("p_ruin", "p_loss_25", "p_double")}
                                    for c in mc.get("historical", {})} if mc.get("status") == "OK" else mc,
            "Signal_Decay": r["stability"].get("assessment"), "Last_Validated": pd.Timestamp.now(tz="UTC").isoformat(),
            "Data_Label": r["data"]["label"], "Data_Flags": sorted(set(r["data"]["flags"]) | set(r["universe_flags"])),
            "Integrity_Inputs": r["integrity_inputs"],
            "Experiment_IDs": [eid],
        }

    # ------------------------------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------------------------------
    def run_untouched_test(self, hid: str, reason: str = "final one-time out-of-sample test") -> dict:
        spec = load_hypothesis(self.reg, hid)
        require_registered(self.reg, hid, spec)
        if hypothesis_status(self.reg, hid) != HypothesisStatus.FROZEN:
            raise PipelineError(f"{hid} is {hypothesis_status(self.reg, hid)}; only FROZEN hypotheses are tested")
        crit = AcceptanceCriteria(**spec.acceptance)
        gate = assess(self.reg, signal_id_for(hid))
        if not gate.promotable:
            raise PipelineError(f"{hid} has {gate.grade} evidence; the untouched test is withheld so the vault is not "
                                f"spent on non-promotable evidence ({'; '.join(gate.reasons())}).")
        dev_ex = self.reg.find("experiments", order="DESC", limit=1, hypothesis_id=hid, kind="development")[0]
        full = self.vault.unseal(self.data.panel, hid, dev_ex["id"], reason)
        contaminated = self.vault.is_contaminated(hid)
        fwd = forward_return(full, spec.holding_period, spec.execution)
        run = self._run(full, spec)
        bh = self._benchmark_daily(full, spec) if spec.primary_metric == "excess_vs_buy_and_hold" else None
        test = self._partition(run, "test", full, spec, fwd, bh)
        val = dev_ex["payload"]["validation"]
        tn = test["trades_net"]
        v_sh, t_sh = val["daily_net"].get("sharpe"), test["daily_net"].get("sharpe")
        results_list = [
            CriterionResult("test_not_contaminated", not contaminated, contaminated, False, "vault opened only once"),
            CriterionResult("test_effective_n", None if test["effective_n"] < crit.min_effective_n_inconclusive else True,
                            test["effective_n"], crit.min_effective_n_inconclusive, "enough test observations"),
            CriterionResult("test_net_mean_positive", (tn.get("mean", np.nan) > 0) if tn.get("n_trades") else None,
                            tn.get("mean"), 0.0, "profitable after costs on never-seen data"),
            CriterionResult("test_sharpe_positive", None if t_sh is None or not np.isfinite(t_sh) else bool(t_sh > 0),
                            t_sh, 0.0, "positive risk-adjusted return on never-seen data"),
        ]
        decision = decide("untouched_test", results_list)
        results = {"hypothesis_id": hid, "stage": "untouched_test", "spec_hash": spec.spec_hash(),
                   "data": self.data.describe(), "code_version": code_version(), "test": self._strip(test),
                   "degradation": {"validation_sharpe": v_sh, "test_sharpe": t_sh,
                                   "ratio": (t_sh / v_sh) if v_sh and t_sh is not None and np.isfinite(v_sh) and v_sh != 0 else None},
                   "decision": decision.to_dict(), "contaminated": contaminated}
        eid = self.reg.append("experiments", {"hypothesis_id": hid, "kind": "untouched_test", "status": decision.outcome}, results)
        self.art.save(eid, "test_trades", test["_trades"].reset_index(drop=True))
        advance_hypothesis(self.reg, hid, HypothesisStatus.TESTED, f"untouched test {eid}", experiment_id=eid)
        sid = signal_id_for(hid)
        rec = latest_record(self.reg, sid)
        upsert_record(self.reg, sid, {"Out_Of_Sample_Performance": {"trades_net": tn, "daily_net": test["daily_net"],
                                                                    "decision": decision.outcome},
                                      "Integrity_Inputs": {**(rec.get("Integrity_Inputs") or {}),
                                                           "test_contaminated": bool(contaminated)},
                                      "Experiment_IDs": (rec.get("Experiment_IDs") or []) + [eid]}, f"untouched test {eid}")
        gate = assess(self.reg, sid)
        if decision.outcome == "PASS" and not gate.promotable:
            status, concl = "VALIDATING", f"{gate.grade}: passed the untouched test but promotion is blocked."
            nxt = "Integrity gate: " + "; ".join(gate.reasons())
            try:  # records the refusal (journal) and raises
                transition(self.reg, sid, SignalStatus.ACCEPTED, "passed the one-time untouched test", {"experiment_id": eid})
            except InvalidTransition:
                pass
        elif decision.outcome == "PASS":
            transition(self.reg, sid, SignalStatus.ACCEPTED, "passed the one-time untouched test", {"experiment_id": eid})
            status, concl = "ACCEPTED", "Survived development validation and the untouched test."
            nxt = "Eligible for a human decision on PAPER_TRADING (requires approval and risk review)."
        elif decision.outcome == "FAIL":
            transition(self.reg, sid, SignalStatus.REJECTED,
                       "failed untouched test: " + ", ".join(x.name for x in decision.failed), {"experiment_id": eid})
            status, concl = "REJECTED", "Edge did not hold on the untouched test data."
            nxt = "None; re-tuning on the test result is not permitted."
        else:
            status, concl = "VALIDATING", "Untouched test inconclusive (too few observations)."
            nxt = "Collect genuinely new data via paper trading before any promotion."
        advance_hypothesis(self.reg, hid, HypothesisStatus.CONCLUDED, concl, experiment_id=eid)
        self.reg.journal("untouched_test", question=spec.question, hypothesis=spec.statement, reason=reason,
                         experiment_id=eid, hypothesis_id=hid, results={"test_mean_net_trade": tn.get("mean"),
                                                                         "test_sharpe": t_sh}, conclusion=concl,
                         status=status, next_step=nxt, data_label=self.data.label.value)
        return {"experiment_id": eid, "signal_id": sid, "signal_status": status, "conclusion": concl,
                "next_step": nxt, "results": results}
