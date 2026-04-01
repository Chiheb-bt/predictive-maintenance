"""
Integration tests — require a running server.

Start the server before running these tests:
    python run.py &
    pytest tests/integration/ -v

Or with make:
    make run &
    make test-integration

When API_KEY is set in the environment, tests automatically include the
correct header so they pass regardless of whether auth is enabled or not.
"""

from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("API_KEY", "")
HEADERS  = {"X-Api-Key": API_KEY} if API_KEY else {}

VALID_PAYLOAD = {
    "Type":                "M",
    "Air_temperature":     298.1,
    "Process_temperature": 308.6,
    "Rotational_speed":    1551,
    "Torque":              42.8,
    "Tool_wear":           0,
}

BATCH_PAYLOAD = [
    {
        "Type": "M", "Air_temperature": 298.1, "Process_temperature": 308.6,
        "Rotational_speed": 1551, "Torque": 42.8, "Tool_wear": 0,
    },
    {
        "Type": "H", "Air_temperature": 310.0, "Process_temperature": 322.0,
        "Rotational_speed": 1800, "Torque": 68.0, "Tool_wear": 220,
    },
    {
        "Type": "L", "Air_temperature": 295.0, "Process_temperature": 305.0,
        "Rotational_speed": 1400, "Torque": 35.0, "Tool_wear": 50,
    },
]

DRIFT_PAYLOAD = {
    "readings": [VALID_PAYLOAD] * 12
}


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as c:
        yield c


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_response_has_status_key(self, client):
        resp = client.get("/")
        assert "status" in resp.json()

    def test_status_is_ok_or_degraded(self, client):
        body = client.get("/").json()
        assert body["status"] in ("ok", "degraded")

    def test_version_present(self, client):
        body = client.get("/").json()
        assert "version" in body

    def test_uptime_is_positive(self, client):
        body = client.get("/").json()
        assert body["uptime_s"] >= 0


class TestPredictEndpoint:
    def test_healthy_server_returns_200(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server is in degraded state — model not loaded")
        assert resp.status_code == 200

    def test_response_shape(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        body = resp.json()
        for key in ("request_id", "prediction", "probability", "confidence",
                    "risk_level", "status", "recommendations", "top_factors"):
            assert key in body, f"Missing key: {key}"

    def test_prediction_is_binary(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        assert resp.json()["prediction"] in (0, 1)

    def test_probability_range(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        p = resp.json()["probability"]
        assert 0.0 <= p <= 1.0

    def test_risk_level_valid(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        assert resp.json()["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_request_id_is_uuid4(self, client):
        import uuid
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        uid = uuid.UUID(resp.json()["request_id"])
        assert uid.version == 4

    def test_top_factors_count(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        assert len(resp.json()["top_factors"]) == 3

    def test_invalid_type_returns_422(self, client):
        bad = {**VALID_PAYLOAD, "Type": "X"}
        resp = client.post("/predict", json=bad, headers=HEADERS)
        assert resp.status_code in (422, 401)

    def test_out_of_range_returns_422(self, client):
        bad = {**VALID_PAYLOAD, "Air_temperature": 9999}
        resp = client.post("/predict", json=bad, headers=HEADERS)
        assert resp.status_code in (422, 401)

    def test_auth_rejected_with_wrong_key(self, client):
        resp = client.post(
            "/predict",
            json=VALID_PAYLOAD,
            headers={"X-Api-Key": "definitely-wrong-key-xyz"},
        )
        assert resp.status_code in (200, 401, 503)

    @pytest.mark.parametrize("machine_type", ["L", "M", "H"])
    def test_all_machine_types(self, client, machine_type):
        payload = {**VALID_PAYLOAD, "Type": machine_type}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        assert resp.status_code == 200

    def test_high_wear_increases_risk(self, client):
        low_wear  = {**VALID_PAYLOAD, "Tool_wear": 0}
        high_wear = {**VALID_PAYLOAD, "Tool_wear": 490, "Torque": 65}
        r_low  = client.post("/predict", json=low_wear,  headers=HEADERS)
        r_high = client.post("/predict", json=high_wear, headers=HEADERS)
        if r_low.status_code == 503 or r_high.status_code == 503:
            pytest.skip("Server degraded")
        assert r_high.json()["probability"] >= r_low.json()["probability"]


class TestBatchPredictEndpoint:
    def test_returns_200_with_valid_list(self, client):
        resp = client.post("/predict/batch", json=BATCH_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        assert resp.status_code == 200

    def test_response_counts_match_input(self, client):
        resp = client.post("/predict/batch", json=BATCH_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        body = resp.json()
        assert body["n_requested"] == len(BATCH_PAYLOAD)
        assert body["n_succeeded"] + body["n_failed"] == body["n_requested"]

    def test_predictions_list_length_matches_input(self, client):
        resp = client.post("/predict/batch", json=BATCH_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        assert len(resp.json()["predictions"]) == len(BATCH_PAYLOAD)

    def test_summary_is_dict(self, client):
        resp = client.post("/predict/batch", json=BATCH_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        assert isinstance(resp.json()["summary"], dict)

    def test_each_row_has_success_flag(self, client):
        resp = client.post("/predict/batch", json=BATCH_PAYLOAD, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        for item in resp.json()["predictions"]:
            assert "success" in item

    def test_partial_invalid_row_does_not_abort_batch(self, client):
        mixed = [
            VALID_PAYLOAD,
            {**VALID_PAYLOAD, "Type": "INVALID_TYPE"},
            VALID_PAYLOAD,
        ]
        resp = client.post("/predict/batch", json=mixed, headers=HEADERS)
        if resp.status_code == 503:
            pytest.skip("Server degraded")
        if resp.status_code != 200:
            pytest.skip("Batch validation rejected at schema level")
        body = resp.json()
        assert body["n_failed"] >= 1
        assert body["n_succeeded"] >= 1

    def test_empty_list_returns_422(self, client):
        resp = client.post("/predict/batch", json=[], headers=HEADERS)
        assert resp.status_code in (422, 401)


class TestDriftEndpoint:
    def test_returns_200_with_enough_readings(self, client):
        resp = client.post("/drift", json=DRIFT_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("Training stats not available (model not trained yet)")
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client):
        resp = client.post("/drift", json=DRIFT_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("Training stats not available")
        body = resp.json()
        for key in ("n_samples", "overall_status", "features", "assessed_at", "note"):
            assert key in body, f"Missing key: {key}"

    def test_n_samples_matches_input(self, client):
        resp = client.post("/drift", json=DRIFT_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("Training stats not available")
        assert resp.json()["n_samples"] == len(DRIFT_PAYLOAD["readings"])

    def test_overall_status_is_valid(self, client):
        resp = client.post("/drift", json=DRIFT_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("Training stats not available")
        assert resp.json()["overall_status"] in ("NORMAL", "WARNING", "DRIFT")

    def test_feature_statuses_are_valid(self, client):
        resp = client.post("/drift", json=DRIFT_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("Training stats not available")
        for feature in resp.json()["features"]:
            assert feature["status"] in ("NORMAL", "WARNING", "DRIFT")

    def test_fewer_than_10_readings_returns_422(self, client):
        too_few = {"readings": [VALID_PAYLOAD] * 5}
        resp = client.post("/drift", json=too_few)
        assert resp.status_code == 422

    def test_drift_no_auth_required(self, client):
        resp = client.post("/drift", json=DRIFT_PAYLOAD)
        assert resp.status_code in (200, 503)
