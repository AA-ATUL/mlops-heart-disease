
from pathlib import Path
import logging

import joblib
import pandas as pd
from fastapi import FastAPI
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

    positive_probability = float(probabilities[1])

    logger.info(
        "Prediction completed: prediction=%s, probability=%.4f",
        prediction,
        positive_probability,
    )

    return {
        "prediction": prediction,
        "prediction_label": (
            "disease_present" if prediction == 1 else "no_disease_detected"
        ),
        "confidence": positive_probability,
        "model": "logistic_regression",
    }
