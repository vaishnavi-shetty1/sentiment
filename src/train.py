from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_imdb.csv"

EXPERIMENT_NAME = "IMDb-Sentiment-Analysis"

REGISTERED_MODEL_NAME = "IMDb-Sentiment-Classifier"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_data():
    print("Loading processed dataset...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=["cleaned_review", "label"]
    )

    # Remove any empty rows
    df = df.dropna()

    print(f"Dataset shape: {df.shape}")

    return df


# --------------------------------------------------
# Split dataset
# --------------------------------------------------

def split_data(df):
    X = df["cleaned_review"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions
        ),
        "recall": recall_score(
            y_test,
            predictions
        ),
        "f1_score": f1_score(
            y_test,
            predictions
        ),
    }

    return metrics


# --------------------------------------------------
# Train models with MLflow
# --------------------------------------------------

def train_models(X_train, X_test, y_train, y_test):

    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000
        ),

        "Multinomial_Naive_Bayes": MultinomialNB(),

        "Linear_SVM": LinearSVC()
    }

    results = []

    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, classifier in models.items():

        print(f"\nTraining: {model_name}")

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2)
                )
            ),
            (
                "classifier",
                classifier
            )
        ])

        with mlflow.start_run(
            run_name=model_name
        ) as run:

            # Train model
            pipeline.fit(
                X_train,
                y_train
            )

            # Evaluate model
            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test
            )

            # Log parameters
            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "tfidf_max_features",
                50000
            )

            mlflow.log_param(
                "ngram_range",
                "1,2"
            )

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log complete pipeline
            mlflow.sklearn.log_model(
                pipeline,
                name="model"
            )

            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"Recall: "
                f"{metrics['recall']:.4f}"
            )

            print(
                f"F1 Score: "
                f"{metrics['f1_score']:.4f}"
            )

            results.append({
                "model_name": model_name,
                "run_id": run.info.run_id,
                **metrics
            })

    return results


# --------------------------------------------------
# Find best model
# --------------------------------------------------

def find_best_model(results):

    best_model = max(
        results,
        key=lambda result: result["f1_score"]
    )

    print("\n================================")
    print("BEST MODEL")
    print("================================")

    print(
        f"Model: "
        f"{best_model['model_name']}"
    )

    print(
        f"F1 Score: "
        f"{best_model['f1_score']:.4f}"
    )

    print(
        f"Run ID: "
        f"{best_model['run_id']}"
    )

    return best_model



# ------------------------------------------------
def register_best_model(best_model):
    """
    Register the best-performing model
    in the MLflow Model Registry.
    """

    model_uri = (
        f"runs:/{best_model['run_id']}/model"
    )

    print("\nRegistering best model...")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print("\n================================")
    print("MODEL REGISTERED SUCCESSFULLY")
    print("================================")

    print(f"Model Name: {REGISTERED_MODEL_NAME}")
    print(f"Version: {model_version.version}")
    print(f"Source Run ID: {best_model['run_id']}")

    return model_version


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    results = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    best_model = find_best_model(results)
    register_best_model(best_model)


if __name__ == "__main__":
    main()