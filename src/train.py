# DEEP LEARNING MODEL TRAINING


from pathlib import Path
import numpy as np
import pandas as pd



# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "manufacturing_quality_intelligence.csv"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessor.pkl"
)

MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)


# 2. LOAD DATA AND PREPROCESSOR

import joblib

df = pd.read_csv(RAW_DATA_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Dataset loaded:", df.shape)
print("Preprocessor loaded successfully.")


# 3. FEATURE PREPARATION

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%d-%m-%Y %H:%M"
)

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek

df = df.drop(columns=["timestamp"])

TARGET_COLUMNS = [
    "machine_failure",
    "is_anomaly",
    "quality_pass"
]

ID_COLUMNS = [
    "reading_id"
]

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in TARGET_COLUMNS + ID_COLUMNS
]

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMNS]


# 4. TRAIN-TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 5. APPLY PREPROCESSOR

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

print("\nProcessed Training Shape:")
print(X_train_processed.shape)

print("Processed Testing Shape:")
print(X_test_processed.shape)


# 6. PREPARE TARGETS

y_train_failure = y_train["machine_failure"].values
y_test_failure = y_test["machine_failure"].values

y_train_anomaly = y_train["is_anomaly"].values
y_test_anomaly = y_test["is_anomaly"].values

y_train_quality = y_train["quality_pass"].values
y_test_quality = y_test["quality_pass"].values

print("\nTarget Shapes:")
print("Failure:", y_train_failure.shape)
print("Anomaly:", y_train_anomaly.shape)
print("Quality:", y_train_quality.shape)


# 7. DEEP LEARNING IMPORTS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("\nTensorFlow Version:", tf.__version__)


# 8. BUILD DEEP LEARNING MODEL

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),   # relu because they're learning nonlinear relationships between manufacturing measurements.
        layers.Dropout(0.30),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.20),

        layers.Dense(32, activation="relu"),

        layers.Dense(1, activation="sigmoid")   # sigmoid because all three of our tasks are binary classification.
    ])

    return model


# 9. COMPILE MODEL

input_dim = X_train_processed.shape[1]

model = build_model(input_dim)

model.compile(
    optimizer="adam",     # adam is the optimizer that updates the neural-network weights during training.
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# 10. MODEL ARCHITECTURE CHECK

sample_output = model(X_train_processed[:5])

print("\nSample Output Shape:")
print(sample_output.shape)

print("\nSample Predictions:")
print(sample_output.numpy())


# 11. CLASS WEIGHTS

from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(y):
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    return dict(zip(classes, weights))

failure_weights = calculate_class_weights(y_train_failure)
anomaly_weights = calculate_class_weights(y_train_anomaly)
quality_weights = calculate_class_weights(y_train_quality)

print("\nClass Weights:")
print("Failure:", failure_weights)
print("Anomaly:", anomaly_weights)
print("Quality:", quality_weights)


# 12. MLFLOW

import mlflow
import mlflow.tensorflow

print("\nMLflow Version:", mlflow.__version__)


# 13. MLFLOW EXPERIMENT

MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    "Manufacturing Quality Intelligence"
)

print("\nMLflow experiment configured.")


# 14. MACHINE FAILURE TRAINING RUN

with mlflow.start_run(run_name="machine_failure_ann"):

    print("\nMLflow run started.")

    mlflow.log_param("target", "machine_failure")
    mlflow.log_param("architecture", "128-64-32")
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss", "binary_crossentropy")
    mlflow.log_param("input_features", input_dim)

    print("Model parameters logged.")


    # 15. TRAINING PARAMETERS

    EPOCHS = 30
    BATCH_SIZE = 32

    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)


    # 16. BUILD FAILURE MODEL

    failure_model = build_model(input_dim)

    failure_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    # compile() specifies:
    # - Adam → how weights are updated
    # - Binary crossentropy → how prediction error is measured
    # - Accuracy/Precision/Recall → what we track

    print("\nMachine Failure Model:")
    failure_model.summary()


    # 17. TRAIN FAILURE MODEL

    history = failure_model.fit(
        X_train_processed,
        y_train_failure,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=failure_weights,
        verbose=1
    )


    # 18. LOG TRAINING METRICS

    final_metrics = history.history

    mlflow.log_metric(
        "final_train_loss",
        final_metrics["loss"][-1]
    )

    mlflow.log_metric(
        "final_train_accuracy",
        final_metrics["accuracy"][-1]
    )

    mlflow.log_metric(
        "final_train_precision",
        final_metrics["precision"][-1]
    )

    mlflow.log_metric(
        "final_train_recall",
        final_metrics["recall"][-1]
    )

    mlflow.log_metric(
        "final_val_loss",
        final_metrics["val_loss"][-1]
    )

    mlflow.log_metric(
        "final_val_accuracy",
        final_metrics["val_accuracy"][-1]
    )

    print("\nTraining metrics logged to MLflow.")


    # 19. SAVE FAILURE MODEL

    failure_model_path = (
        MODELS_DIR / "machine_failure_model.keras"
    )

    failure_model.save(failure_model_path)

    mlflow.log_artifact(str(failure_model_path))

    print("\nFailure model saved:")
    print(failure_model_path)


# 20. MACHINE FAILURE TEST PREDICTIONS

failure_probabilities = failure_model.predict(
    X_test_processed,
    verbose=0
)

print("\nFailure Probability Shape:")
print(failure_probabilities.shape)


# 21. CONVERT PROBABILITIES TO CLASSES

failure_predictions = (
    failure_probabilities >= 0.5
).astype(int)

failure_predictions = failure_predictions.ravel()

print("\nFailure Prediction Shape:")
print(failure_predictions.shape)

print("\nFirst 10 Predictions:")
print(failure_predictions[:10])


# 22. CLASSIFICATION METRICS

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

failure_accuracy = accuracy_score(
    y_test_failure,
    failure_predictions
)

failure_precision = precision_score(
    y_test_failure,
    failure_predictions
)

failure_recall = recall_score(
    y_test_failure,
    failure_predictions
)

failure_f1 = f1_score(
    y_test_failure,
    failure_predictions
)

print("\nMachine Failure Evaluation:")
print("Accuracy :", failure_accuracy)
print("Precision:", failure_precision)
print("Recall   :", failure_recall)
print("F1 Score :", failure_f1)


# 23. CLASSIFICATION REPORT

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(
    classification_report(
        y_test_failure,
        failure_predictions,
        target_names=[
            "No Failure",
            "Failure"
        ]
    )
)


# 24. CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(
    y_test_failure,
    failure_predictions
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Machine Failure Confusion Matrix")

plt.tight_layout()

plt.show()


# 25. BUILD ANOMALY DETECTION MODEL

anomaly_model = build_model(input_dim)

anomaly_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

print("\nAnomaly Detection Model:")
anomaly_model.summary()


# 26. TRAIN ANOMALY DETECTION MODEL

with mlflow.start_run(run_name="is_anomaly_ann"):

    mlflow.log_param("target", "is_anomaly")
    mlflow.log_param("architecture", "128-64-32")
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss", "binary_crossentropy")
    mlflow.log_param("input_features", input_dim)

    EPOCHS = 30
    BATCH_SIZE = 32

    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)

    history_anomaly = anomaly_model.fit(
        X_train_processed,
        y_train_anomaly,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=anomaly_weights,
        verbose=1
    )

    
    # 27. LOG ANOMALY TRAINING METRICS
    
    anomaly_metrics = history_anomaly.history

    mlflow.log_metric(
        "final_train_loss",
        anomaly_metrics["loss"][-1]
    )

    mlflow.log_metric(
        "final_train_accuracy",
        anomaly_metrics["accuracy"][-1]
    )

    mlflow.log_metric(
        "final_train_precision",
        anomaly_metrics["precision"][-1]
    )

    mlflow.log_metric(
        "final_train_recall",
        anomaly_metrics["recall"][-1]
    )

    mlflow.log_metric(
        "final_val_loss",
        anomaly_metrics["val_loss"][-1]
    )

    mlflow.log_metric(
        "final_val_accuracy",
        anomaly_metrics["val_accuracy"][-1]
    )

    print("\nAnomaly training metrics logged.")

    
    # 28. SAVE ANOMALY MODEL
    
    anomaly_model_path = (
        MODELS_DIR / "anomaly_detection_model.keras"
    )

    anomaly_model.save(anomaly_model_path)

    mlflow.log_artifact(str(anomaly_model_path))

    print("\nAnomaly model saved:")
    print(anomaly_model_path)


# 29. ANOMALY TEST PREDICTIONS

anomaly_probabilities = anomaly_model.predict(
    X_test_processed,
    verbose=0
)

anomaly_predictions = (
    anomaly_probabilities >= 0.5
).astype(int).ravel()

print("\nAnomaly Prediction Shape:")
print(anomaly_predictions.shape)


# 30. ANOMALY MODEL EVALUATION

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("\nAnomaly Detection Evaluation:")
print("Accuracy :", accuracy_score(y_test_anomaly, anomaly_predictions))
print("Precision:", precision_score(y_test_anomaly, anomaly_predictions, zero_division=0))
print("Recall   :", recall_score(y_test_anomaly, anomaly_predictions, zero_division=0))
print("F1 Score :", f1_score(y_test_anomaly, anomaly_predictions, zero_division=0))

print("\nClassification Report:")
print(
    classification_report(
        y_test_anomaly,
        anomaly_predictions,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)

cm_anomaly = confusion_matrix(
    y_test_anomaly,
    anomaly_predictions
)

print("\nAnomaly Confusion Matrix:")
print(cm_anomaly)