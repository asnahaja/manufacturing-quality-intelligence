
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "manufacturing_quality_intelligence.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Recreate the same test split used by the project.
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d-%m-%Y %H:%M"
)

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek
df = df.drop(columns=["timestamp"])

targets = ["machine_failure", "is_anomaly", "quality_pass"]
X = df.drop(columns=targets + ["reading_id", "maintenance_due"])
y = df[targets]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
X_test_processed = preprocessor.transform(X_test)

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

model = tf.keras.models.load_model(
    MODELS_DIR / "anomaly_detection_model.keras"
)

probabilities = model.predict(X_test_processed, verbose=0).reshape(-1)
predictions = (probabilities >= 0.5).astype(int)
actual = y_test["is_anomaly"].to_numpy()

cm = confusion_matrix(actual, predictions, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal (0)", "Anomaly (1)"],
)
display.plot(values_format="d")
plt.title("Anomaly Detection — Confusion Matrix")
plt.tight_layout()

output = REPORTS_DIR / "anomaly_detection_confusion_matrix.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print("Confusion matrix:")
print(cm)
print(f"Saved: {output}")
