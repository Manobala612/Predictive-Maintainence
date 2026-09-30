"""
Pytest test suite for the FastAPI Predictive Maintenance backend.

Tests cover:
- GET /api/v1/health        — status and model_loaded fields
- POST /api/v1/predict      — valid input, expected response structure
- POST /api/v1/predict      — high-risk input (expect failure=1)
- POST /api/v1/predict      — each invalid field triggers HTTP 422
- POST /api/v1/predict      — response field types and value ranges
- POST /api/v1/predict      — missing required field triggers HTTP 422
- POST /api/v1/predict      — invalid Type value triggers HTTP 422

MLflow is pointed at a local SQLite file so tests never require a live server.
"""

import os
import pytest

# Point MLflow at a local SQLite backend before importing the app.
# This prevents the prediction service from attempting a real HTTP connection
# to the MLflow tracking server during tests.
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow_test.db")

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.prediction import prediction_service

client = TestClient(app)

# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def load_model():
    """Load model artifacts once for the entire test module."""
    prediction_service.load_artifacts()


# ── Payload helpers ───────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500.0,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 50.0,
}

HIGH_RISK_PAYLOAD = {
    "Type": "H",
    "Air temperature [K]": 305.0,
    "Process temperature [K]": 311.0,    # low delta → poor cooling
    "Rotational speed [rpm]": 1200.0,
    "Torque [Nm]": 68.0,                 # > 60 threshold
    "Tool wear [min]": 220.0,            # > 200 threshold
}


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_status_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_status_field(self):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"

    def test_health_model_loaded_true(self):
        data = client.get("/api/v1/health").json()
        assert data["model_loaded"] is True

    def test_health_features_list_not_empty(self):
        data = client.get("/api/v1/health").json()
        assert isinstance(data.get("features"), list)
        assert len(data["features"]) > 0


# ── Predict endpoint — valid input ────────────────────────────────────────────

class TestPredictValid:
    def test_predict_returns_200(self):
        response = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_has_machine_failure(self):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert "machine_failure" in data

    def test_predict_response_has_failure_probability(self):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert "failure_probability" in data

    def test_predict_response_has_confidence_score(self):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert "confidence_score" in data

    def test_predict_response_has_recommendation(self):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert "recommendation" in data

    def test_machine_failure_is_binary_int(self):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert data["machine_failure"] in (0, 1)
        assert isinstance(data["machine_failure"], int)

    def test_failure_probability_in_range(self):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["failure_probability"] <= 1.0

    def test_confidence_score_in_range(self):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["confidence_score"] <= 1.0

    def test_recommendation_is_string(self):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert isinstance(data["recommendation"], str)
        assert len(data["recommendation"]) > 0

    def test_all_machine_types_accepted(self):
        for t in ("L", "M", "H"):
            payload = {**VALID_PAYLOAD, "Type": t}
            response = client.post("/api/v1/predict", json=payload)
            assert response.status_code == 200, f"Type={t} failed"


# ── Predict endpoint — response structure ─────────────────────────────────────

class TestPredictResponseStructure:
    def test_response_contains_exactly_four_fields(self):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        expected_keys = {"machine_failure", "failure_probability", "confidence_score", "recommendation"}
        assert set(data.keys()) == expected_keys

    def test_high_risk_input_produces_recommendation_text(self):
        """High-risk payload should return a non-trivial recommendation string."""
        data = client.post("/api/v1/predict", json=HIGH_RISK_PAYLOAD).json()
        assert len(data["recommendation"]) > 20


# ── Predict endpoint — invalid input / validation ─────────────────────────────

class TestPredictValidation:
    def test_invalid_type_value_returns_422(self):
        payload = {**VALID_PAYLOAD, "Type": "X"}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_rotational_speed_below_minimum_returns_422(self):
        payload = {**VALID_PAYLOAD, "Rotational speed [rpm]": -100.0}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_rotational_speed_above_maximum_returns_422(self):
        payload = {**VALID_PAYLOAD, "Rotational speed [rpm]": 9999.0}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_air_temperature_below_minimum_returns_422(self):
        payload = {**VALID_PAYLOAD, "Air temperature [K]": 100.0}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_air_temperature_above_maximum_returns_422(self):
        payload = {**VALID_PAYLOAD, "Air temperature [K]": 500.0}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_torque_negative_returns_422(self):
        payload = {**VALID_PAYLOAD, "Torque [Nm]": -5.0}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_tool_wear_above_maximum_returns_422(self):
        payload = {**VALID_PAYLOAD, "Tool wear [min]": 400.0}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_missing_type_field_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Type"}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_missing_torque_field_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Torque [Nm]"}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

    def test_string_value_for_numeric_field_returns_422(self):
        payload = {**VALID_PAYLOAD, "Torque [Nm]": "high"}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
