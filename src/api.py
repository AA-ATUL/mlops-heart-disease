
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational ML model serving demonstration",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "heart_disease_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "heart_disease_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "heart_disease_predictions_total",
    "Total number of predictions by predicted class",
    ["prediction"],
)

# Expose metrics at /metrics
app.mount("/metrics", make_asgi_app())


# Monitor API requests, status codes and response times
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.perf_counter()
    endpoint = request.url.path

    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start_time

        if not endpoint.startswith("/metrics"):
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                http_status="500",
            ).inc()

            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(duration)

        logger.exception(
            "HTTP request failed: method=%s path=%s duration_seconds=%.4f",
            request.method,
            endpoint,
            duration,
        )
        raise

    duration = time.perf_counter() - start_time

    # Avoid counting Prometheus scrapes as application traffic.
    if not endpoint.startswith("/metrics"):
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            http_status=str(response.status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

    logger.info(
        "HTTP request: method=%s path=%s status=%s duration_seconds=%.4f",
        request.method,
        endpoint,
        response.status_code,
        duration,
    )

    return response


class PatientFeatures(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int
    trestbps: float = Field(gt=0)
    chol: float = Field(gt=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int
    ca: float | None = None
    thal: float | None = None


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(patient: PatientFeatures):
    input_data = pd.DataFrame([patient.model_dump()])

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    # Count predictions by predicted class.
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    disease_probability = float(probabilities[1])
    confidence = float(probabilities[prediction])

    logger.info(
        "Prediction completed: prediction=%s, confidence=%.4f, "
        "disease_probability=%.4f",
        prediction,
        confidence,
        disease_probability,
    )

    return {
        "prediction": prediction,
        "prediction_label": (
            "disease_present" if prediction == 1 else "no_disease_detected"
        ),
        "confidence": confidence,
        "disease_probability": disease_probability,
        "model": "logistic_regression",
    }
