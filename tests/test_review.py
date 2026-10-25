"""Regression tests for the project review fixes."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from forgeml.data import synthetic_transactions, validate_transactions
from forgeml.governance import evaluate_candidate, governance_action, initialize_registry


@pytest.mark.parametrize("column,value", [("TransactionAmt", 0), ("TransactionAmt", np.inf), ("TransactionDT", -1), ("TransactionID", 1.5), ("isFraud", np.nan)])
def test_training_rejects_invalid_required_values(column, value):
    frame = synthetic_transactions(30)
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_transactions(frame)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1])
def test_missing_or_invalid_latency_cannot_promote(tmp_path, value):
    path = tmp_path / "registry.json"
    champion = {"average_precision": .3, "brier": .1, "latency_ms": 10}
    candidate = {"average_precision": .5, "brier": .08, "latency_ms": value}
    initialize_registry(path, "original", champion)
    result = evaluate_candidate(path, "candidate", champion, candidate)
    assert result["champion"] == "original"
    assert result["history"][-1]["action"] == "rejected"
    json.dumps(result, allow_nan=False)


def test_failed_bundle_copy_restores_registry(tmp_path):
    champion = {"average_precision": .3, "brier": .1, "latency_ms": 10}
    candidate = {"average_precision": .5, "brier": .08, "latency_ms": 10}
    path = tmp_path / "registry.json"
    before = initialize_registry(path, "original", champion)
    (tmp_path / "report.json").write_text(json.dumps({"governance_comparison": {
        "champion_version": "original", "challenger_version": "candidate",
        "champion_metrics": champion, "challenger_metrics": candidate}}))
    (tmp_path / "bundle.joblib").write_bytes(b"original")
    (tmp_path / "challenger.joblib").write_bytes(b"candidate")
    with patch("shutil.copyfile", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            governance_action(tmp_path, "promote")
    assert json.loads(path.read_text()) == before
    assert (tmp_path / "bundle.joblib").read_bytes() == b"original"


def test_offline_viewer_embeds_assets(tmp_path):
    import shutil
    from pathlib import Path

    from scripts.build_snapshot import build

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "frontend", tmp_path / "frontend")
    result = build(tmp_path, root / "artifacts")
    html = result.read_text(encoding="utf-8")
    assert 'src="/static/app.js"' not in html
    assert 'href="/static/styles.css"' not in html
    assert 'id="snapshot-artifacts"' in html
    assert "async function load()" in html

def test_metrics_totals_survive_rolling_window(tmp_path):
    from forgeml.api import create_app

    app = create_app(tmp_path)
    app.state.telemetry.update(requests=15000, latency_sum_ms=30000, latency_ms=[2.0] * 10000)
    route = next(route for route in app.routes if getattr(route, "path", None) == "/metrics")
    handler = getattr(route.endpoint, "sync_handler", route.endpoint)
    result = handler()
    assert "forgeml_request_latency_ms_count 15000" in result
    assert "forgeml_request_latency_ms_sum 30000.000000" in result


def test_real_data_cannot_overwrite_committed_demo():
    from pathlib import Path

    from forgeml.training import run_training

    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="runs/ieee"):
        run_training(root / "artifacts", data_dir=root / "data/raw")
