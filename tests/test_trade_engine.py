from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.costs import COST_PROFILES, get_cost_model
from quantlab.backtest.metrics import summarize_trades, trades_to_daily_returns
from quantlab.backtest.trades import TradeConfig, generate_trades
from quantlab.research.labels import forward_return


def _sig(panel, cells):
    s = pd.DataFrame(0.0, index=panel["close"].index, columns=panel["close"].columns)
    for d, sym, v in cells:
        s.iloc[d, s.columns.get_loc(sym)] = v
    return s


def test_trade_returns_match_labels_and_costs_reduce(small_panel):
    s = _sig(small_panel, [(100, "SYN000", 1), (300, "SYN001", 1)])
    tr, un = generate_trades(small_panel, s, TradeConfig(holding=5), get_cost_model("retail_etf"))
    fr = forward_return(small_panel, 5, "next_open")
    assert len(tr) == 2 and un.empty
    assert np.isclose(tr.loc[0, "gross_ret"], fr["SYN000"].iloc[100])
    assert (tr["net_ret"] < tr["gross_ret"]).all()
    assert (tr["entry_time"] > tr["decision_available_at"]).all()  # timestamp integrity
    assert (tr["entry_session"] > tr["signal_session"]).all()


def test_no_overlap_by_default(small_panel):
    s = _sig(small_panel, [(100, "SYN000", 1), (101, "SYN000", 1), (102, "SYN000", 1), (110, "SYN000", 1)])
    tr, _ = generate_trades(small_panel, s, TradeConfig(holding=5), COST_PROFILES["frictionless"])
    assert len(tr) == 2
    tr2, _ = generate_trades(small_panel, s, TradeConfig(holding=5, allow_overlap=True), COST_PROFILES["frictionless"])
    assert len(tr2) == 4


def test_signals_at_end_of_data_are_dropped_not_extrapolated(small_panel):
    n = len(small_panel["close"])
    s = _sig(small_panel, [(n - 3, "SYN000", 1)])
    tr, _ = generate_trades(small_panel, s, TradeConfig(holding=5), COST_PROFILES["frictionless"])
    assert tr.empty


def test_short_trades_need_permission_and_pay_borrow(small_panel):
    s = _sig(small_panel, [(100, "SYN000", -1)])
    with pytest.raises(ValueError):
        generate_trades(small_panel, s, TradeConfig(holding=5), get_cost_model("retail_etf"))
    cm = replace(get_cost_model("retail_etf"), allow_short=True)
    tr, _ = generate_trades(small_panel, s, TradeConfig(holding=5), cm)
    fr = forward_return(small_panel, 5)["SYN000"].iloc[100]
    assert np.isclose(tr.loc[0, "gross_ret"], -fr)


def test_summarize_trades_known_values():
    r = pd.Series([0.1, -0.05, 0.02, -0.01])
    s = summarize_trades(r)
    assert s["win_rate"] == 0.5
    assert np.isclose(s["profit_factor"], 0.12 / 0.06)
    assert np.isclose(s["payoff_ratio"], 0.06 / 0.03)
    assert summarize_trades(pd.Series([], dtype=float))["n_trades"] == 0


def test_trades_to_daily_respects_capacity(small_panel):
    idx = small_panel["close"].index
    tr = pd.DataFrame({"entry_session": [idx[10]] * 3, "exit_session": [idx[15]] * 3, "net_ret": [0.05] * 3})
    d = trades_to_daily_returns(tr, idx, max_concurrent=2)
    assert np.isclose((1 + d).prod() - 1, (1 + 0.05) ** 1 - 1, atol=0.01)  # 2 slots × 5% / 2 = ~5%
