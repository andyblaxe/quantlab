import sqlite3

import pytest

from quantlab.research.registry import Registry


def test_ids_are_sequential_and_rows_retrievable(registry):
    a = registry.journal("note", question="q1")
    b = registry.journal("note", question="q2")
    assert (a, b) == ("J-000001", "J-000002")
    assert registry.get("journal", b)["payload"]["question"] == "q2"
    assert registry.get("journal", a)["payload"]["code_version"]["commit"]


def test_update_and_delete_are_forbidden(registry):
    rid = registry.journal("note", question="original")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        registry._con.execute("UPDATE journal SET payload='{}' WHERE id=?", (rid,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        registry._con.execute("DELETE FROM journal WHERE id=?", (rid,))
    assert registry.get("journal", rid)["payload"]["question"] == "original"


def test_hash_chain_detects_tampering(tmp_path):
    path = tmp_path / "r.sqlite"
    reg = Registry(path)
    for i in range(3):
        reg.journal("note", question=f"q{i}")
    assert all(not v for v in reg.verify_chain().values())
    reg.close()
    # bypass the API: drop the trigger and edit a row, as someone editing the file might
    con = sqlite3.connect(path)
    con.execute("DROP TRIGGER journal_no_update")
    con.execute("UPDATE journal SET payload = replace(payload, 'q1', 'qX') WHERE id='J-000002'")
    con.commit()
    con.close()
    reg2 = Registry(path)  # triggers are re-created on open
    assert reg2.verify_chain()["journal"] == ["J-000002"]


def test_nan_is_stored_as_null(registry):
    rid = registry.append("experiments", {"hypothesis_id": "H-1", "kind": "k", "status": "done"},
                          {"sharpe": float("nan"), "nested": {"x": float("inf")}})
    p = registry.get("experiments", rid)["payload"]
    assert p["sharpe"] is None and p["nested"]["x"] is None


def test_current_status_is_latest_event(registry):
    registry.status_event("signal", "SIG-1", "EXPERIMENTAL", "created")
    registry.status_event("signal", "SIG-1", "REJECTED", "failed validation")
    assert registry.current_status("signal", "SIG-1") == "REJECTED"
    assert registry.current_status("signal", "SIG-2") is None
