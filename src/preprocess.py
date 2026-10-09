# PREPROCESSING & FEATURE ENGINEERING

from pathlib import Path
import pandas as pd


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "manufacturing_quality_intelligence.csv"
)


# 2. LOAD DATASET

df = pd.read_csv(RAW_DATA_PATH)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# 3. DATA QUALITY CHECK

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


#  4. TIMESTAMP CONVERSION

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%d-%m-%Y %H:%M"
)

print("\nTimestamp Data Type:")
print(df["timestamp"].dtype)


# 5. IDENTIFY ID COLUMNS

ID_COLUMNS = [
    "reading_id"
]

print("\nID Columns:")
print(ID_COLUMNS)


# 6. DEFINE TARGET VARIABLES

TARGET_COLUMNS = [
    "machine_failure",
    "is_anomaly",
    "quality_pass"
]

print("\nTarget Columns:")
print(TARGET_COLUMNS)


# 7. SEPARATE FEATURES AND TARGETS

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in TARGET_COLUMNS + ID_COLUMNS
]

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMNS]

print("\nFeature Columns:")
print(FEATURE_COLUMNS)

print("\nFeatures Shape:", X.shape)
print("Targets Shape:", y.shape)


# 8. TIMESTAMP FEATURE ENGINEERING

X = X.copy()

X["hour"] = X["timestamp"].dt.hour
X["day"] = X["timestamp"].dt.day
X["month"] = X["timestamp"].dt.month
X["day_of_week"] = X["timestamp"].dt.dayofweek

print("\nTimestamp Features Added:")
print([
    "hour",
    "day",
    "month",
    "day_of_week"
])


# 9. REMOVE RAW TIMESTAMP

X = X.drop(columns=["timestamp"])

print("\nFeatures after removing timestamp:")
print(X.columns.tolist())


# 10. IDENTIFY FEATURE TYPES

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# 11. TRAIN-TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Targets:", y_train.shape)
print("Testing Targets:", y_test.shape)


# 12. FINAL FEATURE TYPE LISTS

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# 13. PREPROCESSING PIPELINE

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# 14. FIT PREPROCESSOR

preprocessor.fit(X_train)

print("\nPreprocessor fitted successfully.")


# 15. TRANSFORM TRAINING DATA

X_train_processed = preprocessor.transform(X_train)

print("\nProcessed Training Shape:")
print(X_train_processed.shape)


# 16. TRANSFORM TESTING DATA

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Testing Shape:")
print(X_test_processed.shape)


# 17. CONVERT TO NUMPY ARRAYS

X_train_processed = X_train_processed.toarray()
X_test_processed = X_test_processed.toarray()

print("\nFinal Training Shape:")
print(X_train_processed.shape)

print("\nFinal Testing Shape:")
print(X_test_processed.shape)


# 18. TARGET DISTRIBUTION CHECK

for target in TARGET_COLUMNS:
    print(f"\nTarget Distribution: {target}")
    print(y_train[target].value_counts())
    print(y_train[target].value_counts(normalize=True))


# 19. PREPARE TARGET ARRAYS

y_train = y_train[TARGET_COLUMNS].values
y_test = y_test[TARGET_COLUMNS].values

print("\nTarget Array Shapes:")
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# 20. SEPARATE TARGETS

y_train_failure = y_train[:, 0]
y_test_failure = y_test[:, 0]

y_train_anomaly = y_train[:, 1]
y_test_anomaly = y_test[:, 1]

y_train_quality = y_train[:, 2]
y_test_quality = y_test[:, 2]

print("\nIndividual Target Shapes:")
print("Failure:", y_train_failure.shape)
print("Anomaly:", y_train_anomaly.shape)
print("Quality:", y_train_quality.shape)


# 21. CLASS WEIGHTS

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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
print("Machine Failure:", failure_weights)
print("Anomaly:", anomaly_weights)
print("Quality:", quality_weights)


# 22. SAVE PREPROCESSING OBJECT

import joblib

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

joblib.dump(
    preprocessor,
    MODELS_DIR / "preprocessor.pkl"
)

print("\nPreprocessor saved successfully.")