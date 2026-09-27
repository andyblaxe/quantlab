import pytest

from quantlab.data.adjust import build_panel
from quantlab.data.providers.synthetic import SyntheticConfig, SyntheticMarket
from quantlab.research.registry import Registry


@pytest.fixture(scope="session")
def small_market():
    """SIMULATED market, 6 symbols + market, 2016-2021 (session-scoped: generated once)."""
    return SyntheticMarket(SyntheticConfig(n_symbols=6, start="2016-01-01", end="2021-12-31", seed=11,
                                           split_threshold=150.0))


@pytest.fixture(scope="session")
def small_panel(small_market):
    return build_panel(small_market.bars().frame, small_market.corporate_actions().frame)


@pytest.fixture()
def registry(tmp_path):
    reg = Registry(tmp_path / "registry.sqlite")
    yield reg
    reg.close()
