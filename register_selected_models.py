
from pathlib import Path

import mlflow
import joblib
import tensorflow as tf

from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

client = MlflowClient()

# Register the two selected Random Forest models.
selected_forests = [
    (
        "ManufacturingMachineFailure",
        "machine_failure_random_forest.pkl",
    ),
    (
        "ManufacturingQualityPrediction",
        "quality_prediction_random_forest.pkl",
    ),
]

for registered_name, filename in selected_forests:
    model_path = MODELS / filename
    model = joblib.load(model_path)

    with mlflow.start_run(run_name=f"register_{registered_name}_random_forest"):
        mlflow.log_param("selected_algorithm", "Random Forest")
        mlflow.log_artifact(str(model_path))

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{filename}"

        mlflow.sklearn.log_model(
            sk_model=model,
            name="selected_model",
            registered_model_name=registered_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    latest = client.get_latest_versions(registered_name)
    version = max(int(item.version) for item in latest)

    client.set_registered_model_alias(
        registered_name, "champion", str(version)
    )

    print(f"{registered_name}: champion -> Version {version}")

# The anomaly ANN was already registered during ANN training.
# Point its champion alias to its latest registered version.
anomaly_name = "ManufacturingAnomalyDetection"
anomaly_versions = client.search_model_versions(
    f"name='{anomaly_name}'"
)

if anomaly_versions:
    latest_anomaly = max(
        anomaly_versions, key=lambda item: int(item.version)
    )
    client.set_registered_model_alias(
        anomaly_name, "champion", str(latest_anomaly.version)
    )
    print(
        f"{anomaly_name}: champion -> "
        f"Version {latest_anomaly.version}"
    )
else:
    print("Anomaly model was not found in the registry.")
