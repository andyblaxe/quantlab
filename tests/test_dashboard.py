"""Dashboard smoke tests: every page renders from the (demo-style) registry without errors."""

import pytest
from fastapi.testclient import TestClient

from quantlab.dashboard.app import create_app, md_to_html
from quantlab.research.session import Workspace


@pytest.fixture(scope="module")
def client_from_world(world):
    reg = world["reg"]
    ws = Workspace(root=world["tmp"], registry=reg, artifacts=world["fb"].art, simulated=True)
    return TestClient(create_app(workspace=ws))


PAGES = ["/", "/portfolio", "/strategies", "/signals", "/research", "/options", "/risk", "/montecarlo", "/reports",
         "/assistant", "/options?S=50&K=55&days=10&iv=0.4", "/strategy/SIG-000001", "/strategy/SIG-000001?mode=quant",
         "/static/plotly.min.js"]


@pytest.mark.parametrize("path", PAGES)
def test_pages_render(client_from_world, path):
    r = client_from_world.get(path)
    assert r.status_code == 200, r.text[:500]
    if path != "/static/plotly.min.js":
        assert "SIMULATED" in r.text


def test_assistant_and_report_generation(client_from_world):
    r = client_from_world.post("/assistant", data={"question": "What have you rejected?"})
    assert r.status_code == 200 and "SIG-" in r.text
    r = client_from_world.post("/assistant", data={"question": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in r.text  # escaped
    r = client_from_world.post("/reports/generate", data={"mode": "quant"}, follow_redirects=True)
    assert r.status_code == 200 and "Research Report" in r.text


def test_bad_inputs_rejected(client_from_world):
    assert client_from_world.get("/strategy/SIG-000001?mode=evil").status_code == 400
    assert client_from_world.get("/reports/R-999999").status_code == 404


def test_markdown_renderer_escapes():
    out = md_to_html("## Title\n- a <b>\n| x | y |\n|---|---|\n| 1 | <i> |")
    assert "<h3>Title</h3>" in out and "&lt;b&gt;" in out and "&lt;i&gt;" in out
