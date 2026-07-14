from pathlib import Path
import shutil

import mlflow


BASE_DIR = Path(__file__).resolve().parent.parent

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Paste the complete Linear_SVM winning Run ID here
BEST_RUN_ID = "70cf316c0e7540d9ba0e03b97a81d97a"

MODEL_URI = f"runs:/{BEST_RUN_ID}/model"

OUTPUT_DIR = BASE_DIR / "models"


def export_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from: {MODEL_URI}")

    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=MODEL_URI,
        dst_path=str(OUTPUT_DIR)
    )

    print("Model exported successfully.")
    print(f"Saved at: {local_path}")


if __name__ == "__main__":
    export_model()