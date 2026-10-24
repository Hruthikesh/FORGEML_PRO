"""API safety contracts and real inference artifact integration."""

import socket
import threading
from pathlib import Path

import joblib
import pandas as pd
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from forgeml.api import Transaction, create_app


@pytest.fixture(scope="session")
def local_asyncio_available():
    result = []

    def probe():
        try:
            pair = socket.socketpair()
            for connection in pair:
                connection.close()
            result.append(True)
        except OSError:
            result.append(False)

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(timeout=2)
    if not result or not result[0]:
        pytest.skip("Local socketpair unavailable; ASGI TestClient cannot initialize its event loop")


@pytest.fixture
def empty_client(tmp_path, local_asyncio_available):
    return TestClient(create_app(tmp_path))


def transaction(**changes):
    return {
        "TransactionID": 9999999,
        "TransactionDT": 9999999,
        "TransactionAmt": 49.95,
        "card1": 101,
        "addr1": 20,
        "DeviceInfo": "desktop",
        "ProductCD": "W",
        "P_emaildomain": "gmail.com",
        **changes,
    }


def test_health_without_bundle_and_readiness_error(empty_client):
    assert empty_client.get("/health").json()["status"] == "ok"
    assert empty_client.get("/ready").status_code == 503
    assert "X-Request-ID" in empty_client.get("/health").headers


@pytest.mark.parametrize(
    "change",
    [{"TransactionAmt": -1}, {"TransactionAmt": 0}, {"TransactionDT": -1}, {"isFraud": 1}, {"DeviceInfo": "x" * 257}],
)
def test_scoring_contract_rejects_invalid_input(empty_client, change):
    assert empty_client.post("/predict", json=transaction(**change)).status_code == 422


def test_batch_limits(empty_client):
    assert empty_client.post("/predict-batch", json={"transactions": []}).status_code == 422
    assert empty_client.post("/predict-batch", json={"transactions": [transaction()] * 513}).status_code == 422


def test_monitoring_records_errors_without_raw_payload(empty_client):
    empty_client.post("/predict", json=transaction(TransactionAmt=-1))
    response = empty_client.get("/metrics")
    assert response.status_code == 200
    assert "forgeml_errors_total 1" in response.text
    assert "TransactionAmt" not in response.text


def test_real_bundle_inference_and_artifact_roundtrip(local_asyncio_available):
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    if not (artifacts / "bundle.joblib").is_file():
        pytest.skip("Run scripts/train.py to execute real artifact integration")
    from forgeml.training import predict_bundle

    bundle = joblib.load(artifacts / "bundle.joblib")
    tx = transaction()
    probabilities = predict_bundle(bundle, pd.DataFrame([tx]))
    assert len(probabilities) == 1
    assert 0 <= probabilities[0] <= 1
    client = TestClient(create_app(artifacts))
    response = client.post("/predict", json=tx)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["probability"] == pytest.approx(float(probabilities[0]))
    assert result["model_version"] == bundle["version"]
    assert result["recommendation"] in {"approve", "review"}
    assert result["latency_ms"] >= 0
    batch = client.post("/predict-batch", json={"transactions": [tx, transaction(TransactionID=9999998)]})
    assert batch.status_code == 200, batch.text
    assert batch.json()["count"] == 2
    duplicate = client.post("/predict-batch", json={"transactions": [tx, tx]})
    assert duplicate.status_code == 422
    for endpoint in (
        "/api/report",
        "/api/examples",
        "/api/graph",
        "/api/monitoring",
        "/api/governance",
        "/api/explanations",
        "/model-info",
        "/ready",
    ):
        assert client.get(endpoint).status_code == 200, endpoint


def test_unknown_governance_action_rejected(empty_client):
    assert empty_client.post("/api/governance/unknown", json={}).status_code == 404


@pytest.mark.parametrize(
    "change",
    [{"TransactionAmt": -1}, {"TransactionAmt": 0}, {"TransactionDT": -1}, {"isFraud": 1}, {"DeviceInfo": "x" * 257}],
)
def test_pydantic_scoring_schema_without_network(change):
    with pytest.raises(ValidationError):
        Transaction.model_validate(transaction(**change))


def test_valid_scoring_schema_without_network():
    assert Transaction.model_validate(transaction()).TransactionAmt == 49.95


def test_service_business_logic_with_actual_bundle_without_http():
    """Verify synchronous handlers separately; this is not transport verification."""
    from starlette.requests import Request

    from forgeml.training import predict_bundle

    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    if not (artifacts / "bundle.joblib").is_file():
        pytest.skip("Train actual artifact before direct service integration")
    app = create_app(artifacts)
    handlers = {
        route.path: getattr(route.endpoint, "sync_handler", route.endpoint)
        for route in app.routes
        if hasattr(route, "endpoint")
    }
    assert handlers["/health"]()["status"] == "ok"
    assert handlers["/ready"]()["status"] == "ready"
    tx = Transaction.model_validate(transaction())
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/predict",
            "headers": [],
            "state": {"request_id": "direct-business-test"},
        }
    )
    result = handlers["/predict"](tx, request)
    bundle = joblib.load(artifacts / "bundle.joblib")
    expected = predict_bundle(bundle, pd.DataFrame([tx.model_dump()]))[0]
    assert result["probability"] == pytest.approx(float(expected))
    assert result["request_id"] == "direct-business-test"
    assert 0 <= result["probability"] <= 1
    assert result["model_version"] == bundle["version"]
    for endpoint in (
        "/api/report",
        "/api/examples",
        "/api/graph",
        "/api/monitoring",
        "/api/governance",
        "/api/explanations",
        "/model-info",
    ):
        assert isinstance(handlers[endpoint](), dict), endpoint
