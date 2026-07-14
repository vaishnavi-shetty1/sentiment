import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

client.set_registered_model_alias(
    name="IMDb-Sentiment-Classifier",
    alias="champion",
    version="1"
)

print("Alias 'champion' assigned successfully.")
