
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow

from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from sklearn.exceptions import NotFittedError

from schemas import MachineInput


# 1. PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


# 2. MLflow REGISTRY
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)
client = MlflowClient()


# 3. LOAD PREPROCESSOR
preprocessor_path = MODELS_DIR / "preprocessor.pkl"
preprocessor = joblib.load(preprocessor_path)


# 4. LOAD SELECTED MODELS
# Random Forest models are loaded from their saved local files.
machine_failure_model = joblib.load(
    MODELS_DIR / "machine_failure_random_forest.pkl"
)
quality_model = joblib.load(
    MODELS_DIR / "quality_prediction_random_forest.pkl"
)

# ANN is loaded from the existing Keras model file.
anomaly_model = tf.keras.models.load_model(
    MODELS_DIR / "anomaly_detection_model.keras",
    compile=False,
)


# 5. CREATE FASTAPI APP
app = FastAPI(
    title="Manufacturing Quality Intelligence API",
    description=(
        "API for machine failure prediction, "
        "anomaly detection, and manufacturing quality prediction."
    ),
    version="1.0.0",
)


# 6. PREPARE INPUT FEATURES
def prepare_features(machine_input: MachineInput):
    data = machine_input.model_dump()

    raw_timestamp = data.pop("timestamp")

    print("Received timestamp:", repr(raw_timestamp))

    timestamp = pd.to_datetime(
        raw_timestamp,
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    print("Parsed timestamp:", repr(timestamp))

    if pd.isna(timestamp):
        raise HTTPException(
            status_code=422,
            detail="Invalid timestamp. Use a valid date/time string.",
        )

    data["hour"] = timestamp.hour
    data["day"] = timestamp.day
    data["month"] = timestamp.month
    data["day_of_week"] = timestamp.dayofweek

    # Match the original training feature order.
    feature_names = list(preprocessor.feature_names_in_)
    feature_frame = pd.DataFrame([data]).reindex(
        columns=feature_names
    )

    return preprocessor.transform(feature_frame)


# 7. HEALTH CHECK
@app.get("/")
def home():
    return {
        "message": "Manufacturing Quality Intelligence API is running"
    }


# 8. MACHINE FAILURE PREDICTION
@app.post("/predict/failure")
def predict_failure(machine_input: MachineInput):
    features = prepare_features(machine_input)

    prediction = int(
        machine_failure_model.predict(features)[0]
    )
    probability = float(
        machine_failure_model.predict_proba(features)[0][1]
    )

    return {
        "task": "machine_failure",
        "prediction": prediction,
        "label": "Failure" if prediction == 1 else "No Failure",
        "probability": probability,
    }


# 9. ANOMALY DETECTION
@app.post("/predict/anomaly")
def predict_anomaly(machine_input: MachineInput):
    features = prepare_features(machine_input)

    probabilities = anomaly_model.predict(
        features, verbose=0
    ).ravel()
    probability = float(probabilities[0])
    prediction = int(probability >= 0.5)

    return {
        "task": "is_anomaly",
        "prediction": prediction,
        "label": "Anomaly" if prediction == 1 else "Normal",
        "probability": probability,
    }


# 10. QUALITY PREDICTION
@app.post("/predict/quality")
def predict_quality(machine_input: MachineInput):
    features = prepare_features(machine_input)

    prediction = int(
        quality_model.predict(features)[0]
    )
    probability = float(
        quality_model.predict_proba(features)[0][1]
    )

    return {
        "task": "quality_pass",
        "prediction": prediction,
        "label": "Pass" if prediction == 1 else "Fail",
        "probability": probability,
    }
