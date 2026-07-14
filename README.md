#  Sentiment Analysis MLOps

A production-ready Sentiment Analysis application built using **Hugging Face DistilBERT**, **FastAPI**, **MLflow**, **Docker**, and **AWS EC2**. The application predicts whether a given movie review is **Positive** or **Negative**.

---

##  Project Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow for NLP sentiment analysis.

The workflow includes:

- Data preprocessing
- Transformer model fine-tuning
- Experiment tracking using MLflow
- Model Registry
- REST API with FastAPI
- Docker containerization
- Cloud deployment on Render

---

#  Features

- ✅ IMDb Movie Review Sentiment Classification
- ✅ Hugging Face DistilBERT Model
- ✅ FastAPI REST API
- ✅ Beautiful HTML/CSS Frontend
- ✅ MLflow Experiment Tracking
- ✅ MLflow Model Registry
- ✅ Model Versioning
- ✅ Dockerized Application
- ✅ Model Deployment
- ✅ Easy Model Inference
- ✅ Production Ready Structure

---

#  Tech Stack

## Machine Learning

- Python
- Hugging Face Transformers
- DistilBERT
- PyTorch
- Scikit-learn

## Backend

- FastAPI
- Uvicorn

## Experiment Tracking

- MLflow

## DevOps

- Docker
- Git
- GitHub

## Cloud

- Render

---

#  Project Structure

```
sentiment-analysis-mlops/
│
├── app/
│   ├── main.py
│   ├── predict.py
│   ├── utils.py
│   └── templates/
│
├── model/
│
├── notebooks/
│
├── static/
│   ├── css/
│   └── images/
│
├── mlruns/
│
├── Dockerfile
├── requirements.txt
├── train.py
├── register_model.py
├── set_alias.py
├── README.md
└── .gitignore
```

---

#  Installation

## Clone Repository

```bash
git clone https://github.com/vaishnavi-shetty1/sentiment.git

cd sentiment
```

---

## Create Virtual Environment

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  Running the Application

## Start FastAPI

```bash
uvicorn app.main:app --reload
```

API

```
http://localhost:8000
```

Swagger Documentation

```
http://localhost:8000/docs
```

---

#  MLflow

Start MLflow UI

```bash
mlflow ui --port 5000
```

Open

```
http://localhost:5000
```

Track

- Experiments
- Parameters
- Metrics
- Artifacts
- Registered Models
- Model Versions

---

#  Model Training

Train the model

```bash
python train.py
```

---

#  Register Model

```bash
python register_model.py
```

---

#  Set Production Alias

```bash
python set_alias.py
```

Example

```python
client.set_registered_model_alias(
    name="IMDb-Sentiment-Classifier",
    alias="champion",
    version="1"
)
```

---

#  Docker

## Build Docker Image

```bash
docker build -t sentiment-analysis .
```

Run Container

```bash
docker run -p 8000:8000 sentiment-analysis
```

---

#  API Endpoint

POST

```
/predict
```

Example Request

```json
{
  "text": "This movie was absolutely amazing!"
}
```

Response

```json
{
  "prediction": "Positive"
}
```

---

#  Model

Model Used

```
DistilBERT Base Uncased
```

Dataset

```
IMDb Movie Reviews
```

Classes

- Positive
- Negative
  
---

# Future Improvements

- User Authentication
- Batch Predictions
- CI/CD using GitHub Actions
- Kubernetes Deployment
- Monitoring with Prometheus & Grafana
- Model Drift Detection
- Automated Retraining Pipeline

---
