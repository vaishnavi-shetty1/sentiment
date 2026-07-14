from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.pyfunc

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model"

model = None


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    prediction: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print(f"Loading local model from: {MODEL_PATH}")

    try:
        model = mlflow.pyfunc.load_model(
            str(MODEL_PATH)
        )
        print("Model loaded successfully.")

    except Exception as error:
        print(f"Failed to load model: {error}")
        model = None

    yield

    model = None


app = FastAPI(
    title="IMDb Sentiment Analysis API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "IMDb Sentiment Analysis API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    prediction = model.predict([request.text])
    prediction_value = int(prediction[0])

    return PredictionResponse(
        text=request.text,
        sentiment=(
            "positive"
            if prediction_value == 1
            else "negative"
        ),
        prediction=prediction_value
    )