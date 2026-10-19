
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_prediction_endpoint():
    patient = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    response = client.post("/predict", json=patient)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in [
        "disease_present",
        "no_disease_detected",
    ]
    assert 0 <= result["confidence"] <= 1
    assert result["model"] == "logistic_regression"


def test_invalid_patient_is_rejected():
    response = client.post("/predict", json={"age": -1})

    assert response.status_code == 422
