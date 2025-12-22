import sys
from pathlib import Path
import time
import json

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from src.lstm.model import LSTMTextClassifier
from src.xgb.model import XGBTextClassifier
from src.kmeans.model import KMeansTextInferencer

# Initialize logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instances
models = {"lstm": None, "xgboost": None, "kmeans": None}

# Model metadata
model_metadata = {}

VALID_MODEL_IDS = ["xgboost", "lstm", "kmeans"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup"""
    global models, model_metadata

    print("Loading models...")

    try:
        # Load LSTM model
        lstm_dir = project_root / "models/lstm/deep_bilstm_chosen"
        models["lstm"] = LSTMTextClassifier(str(lstm_dir))

        # Load LSTM metadata
        with open(lstm_dir / "lstm_metadata.json", "r") as f:
            lstm_meta = json.load(f)
        model_metadata["lstm"] = {
            "accuracy": lstm_meta.get("test_accuracy", 0.90),
            "precision": lstm_meta.get("test_precision", 0.89),
            "recall": lstm_meta.get("test_recall", 0.88),
            "f1": lstm_meta.get("test_f1", 0.88),
        }
        print("✓ LSTM model loaded")

    except Exception as e:
        print(f"✗ Failed to load LSTM model: {e}")
        models["lstm"] = None

    try:
        # Load XGBoost model
        xgb_dir = project_root / "models/xgboost"
        models["xgboost"] = XGBTextClassifier(str(xgb_dir))

        # Load XGBoost metadata
        with open(xgb_dir / "xgb_results_summary_randomized.json", "r") as f:
            xgb_meta = json.load(f)
        model_metadata["xgboost"] = {
            "accuracy": xgb_meta["test_metrics"]["accuracy"],
            "precision": xgb_meta["test_metrics"]["precision"],
            "recall": xgb_meta["test_metrics"]["recall"],
            "f1": xgb_meta["test_metrics"]["f1"],
        }
        print("✓ XGBoost model loaded")

    except Exception as e:
        print(f"✗ Failed to load XGBoost model: {e}")
        models["xgboost"] = None

    try:
        # Load K-Means model
        kmeans_dir = project_root / "models/kmeans/k3/tfidf_1000"
        models["kmeans"] = KMeansTextInferencer(str(kmeans_dir))

        # Load K-Means metadata
        with open(kmeans_dir / "clustering_results.json", "r") as f:
            kmeans_meta = json.load(f)
        model_metadata["kmeans"] = {
            "silhouette_score": kmeans_meta.get("silhouette_score", 0.15),
            "n_clusters": kmeans_meta.get("n_clusters", 3),
            "inertia": kmeans_meta.get("inertia", 0),
        }
        print("✓ K-Means model loaded")

    except Exception as e:
        print(f"✗ Failed to load K-Means model: {e}")
        models["kmeans"] = None

    print(
        f"Loaded {sum(1 for m in models.values() if m is not None)}/3 models successfully"
    )

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Spam Detection API",
    description="Multi-model spam detection API supporting XGBoost, LSTM, and K-Means",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    text: str
    models: List[str] = ["xgboost"]  # Default to xgboost


class BatchPredictionRequest(BaseModel):
    texts: List[str]
    models: List[str] = ["xgboost"]


class ModelPrediction(BaseModel):
    model_id: str
    model_name: str
    prediction: int  # 0 = ham, 1 = spam
    spam_probability: float
    confidence: float
    processing_time_ms: float
    # Additional model-specific data
    cluster_id: Optional[int] = None
    cluster_distances: Optional[Dict[int, float]] = None
    user_point_2d: Optional[List[float]] = None


class PredictionResponse(BaseModel):
    predictions: List[ModelPrediction]
    text_stats: Dict[str, Any]
    total_processing_time_ms: float


class BatchPredictionItem(BaseModel):
    text_index: int
    predictions: List[ModelPrediction]
    text_stats: Dict[str, Any]
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    items: List[BatchPredictionItem]
    summary: Dict[str, Any]
    total_processing_time_ms: float


def _validate_model_ids(model_ids: List[str]) -> List[str]:
    if not model_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one model must be selected",
        )

    invalid = [m for m in model_ids if m not in VALID_MODEL_IDS]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid model(s): {', '.join(invalid)}. Valid models: {', '.join(VALID_MODEL_IDS)}"
            ),
        )

    # Preserve order but remove duplicates
    seen = set()
    unique_models = []
    for model_id in model_ids:
        if model_id not in seen:
            seen.add(model_id)
            unique_models.append(model_id)
    return unique_models


def _compute_text_stats(text: str) -> Dict[str, Any]:
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": max(1, text.count(".") + text.count("!") + text.count("?")),
        "avg_word_length": round(np.mean([len(w) for w in words]) if words else 0, 2),
    }


async def _run_models_for_text(
    text: str, model_ids: List[str]
) -> Tuple[List[ModelPrediction], Dict[str, Any], float]:
    start_time = time.time()
    predictions: List[ModelPrediction] = []

    for model_id in model_ids:
        model_start = time.time()

        try:
            if model_id == "xgboost":
                pred = await predict_xgboost(text)
            elif model_id == "lstm":
                pred = await predict_lstm(text)
            elif model_id == "kmeans":
                pred = await predict_kmeans(text)
            else:
                continue

            pred["processing_time_ms"] = round((time.time() - model_start) * 1000, 2)
            predictions.append(ModelPrediction(**pred))

        except Exception as exc:
            print(f"Error predicting with {model_id}: {exc}")
            continue

    text_stats = _compute_text_stats(text)
    total_time_ms = round((time.time() - start_time) * 1000, 2)

    return predictions, text_stats, total_time_ms


@app.get("/")
async def root():
    """Health check endpoint"""
    available_models = [name for name, model in models.items() if model is not None]
    return {
        "status": "ok",
        "message": "Spam Detection API is running",
        "available_models": available_models,
        "endpoints": {"predict": "/predict", "model_info": "/models/info"},
    }


@app.get("/models/info")
async def get_model_info():
    """Get information about all loaded models"""
    info = {}

    for model_id, model in models.items():
        if model is not None:
            info[model_id] = {
                "status": "loaded",
                "metadata": model_metadata.get(model_id, {}),
                "description": {
                    "xgboost": "Gradient Boosting classifier - Fast and accurate",
                    "lstm": "Deep Bidirectional LSTM - Captures context and word order",
                    "kmeans": "Unsupervised clustering - Distance-based classification",
                }.get(model_id, ""),
            }
        else:
            info[model_id] = {"status": "not_loaded", "error": "Model failed to load"}

    return info


# Add global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": str(exc.errors()),
            "body": str(exc.body),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
        },
    )


# Enhance predict endpoint with validation
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict spam probability using selected models

    - **text**: The message text to analyze
    - **models**: List of model IDs to use (xgboost, lstm, kmeans)
    """
    start_time = time.time()

    # Input validation
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Text input cannot be empty"
        )

    if len(request.text) > 10000:  # Set reasonable limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text exceeds maximum length of 10000 characters",
        )

    model_ids = _validate_model_ids(request.models)

    predictions, text_stats, total_time = await _run_models_for_text(
        request.text, model_ids
    )

    return PredictionResponse(
        predictions=predictions,
        text_stats=text_stats,
        total_processing_time_ms=round(total_time, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict spam probabilities for a batch of texts."""

    batch_start = time.time()

    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one text must be provided",
        )

    model_ids = _validate_model_ids(request.models)

    items: List[BatchPredictionItem] = []
    summary_data: Dict[str, Dict[str, Any]] = {
        model_id: {
            "spam": 0,
            "ham": 0,
            "confidences": [],
            "spam_probabilities": [],
        }
        for model_id in model_ids
    }

    for index, text in enumerate(request.texts):
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text at index {index} cannot be empty",
            )

        if len(text) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text at index {index} exceeds maximum length of 10000 characters",
            )

        predictions, text_stats, processing_time = await _run_models_for_text(
            text, model_ids
        )

        items.append(
            BatchPredictionItem(
                text_index=index,
                predictions=predictions,
                text_stats=text_stats,
                processing_time_ms=processing_time,
            )
        )

        for prediction in predictions:
            model_summary = summary_data.get(prediction.model_id)
            if model_summary is None:
                continue

            if prediction.prediction == 1:
                model_summary["spam"] += 1
            else:
                model_summary["ham"] += 1

            model_summary["confidences"].append(prediction.confidence)
            model_summary["spam_probabilities"].append(prediction.spam_probability)

    summary = {
        "texts_processed": len(request.texts),
        "per_model": {},
    }

    for model_id, data in summary_data.items():
        total_predictions = data["spam"] + data["ham"]
        avg_confidence = (
            sum(data["confidences"]) / len(data["confidences"])
            if data["confidences"]
            else 0
        )
        avg_spam_probability = (
            sum(data["spam_probabilities"]) / len(data["spam_probabilities"])
            if data["spam_probabilities"]
            else 0
        )

        summary["per_model"][model_id] = {
            "predictions": total_predictions,
            "spam": data["spam"],
            "ham": data["ham"],
            "avg_confidence": round(avg_confidence, 4),
            "avg_spam_probability": round(avg_spam_probability, 4),
        }

    total_processing_time_ms = round((time.time() - batch_start) * 1000, 2)

    return BatchPredictionResponse(
        items=items,
        summary=summary,
        total_processing_time_ms=total_processing_time_ms,
    )


async def predict_xgboost(text: str) -> Dict[str, Any]:
    """Predict using XGBoost model"""
    model = models["xgboost"]

    proba = model.predict_proba([text])[0]
    pred = model.predict([text])[0]

    spam_prob = float(proba)
    confidence = abs(spam_prob - 0.5) * 2  # 0-1 scale

    return {
        "model_id": "xgboost",
        "model_name": "XGBoost",
        "prediction": int(pred),
        "spam_probability": round(spam_prob, 4),
        "confidence": round(confidence, 4),
    }


async def predict_lstm(text: str) -> Dict[str, Any]:
    """Predict using LSTM model"""
    model = models["lstm"]

    proba = model.predict_proba([text])[0]
    pred = model.predict([text])[0]

    spam_prob = float(proba)
    confidence = abs(spam_prob - 0.5) * 2

    return {
        "model_id": "lstm",
        "model_name": "LSTM",
        "prediction": int(pred),
        "spam_probability": round(spam_prob, 4),
        "confidence": round(confidence, 4),
    }


async def predict_kmeans(text: str) -> Dict[str, Any]:
    """Predict using K-Means clustering model"""
    model = models["kmeans"]

    # Get cluster assignment and distances
    cluster_id, distances, point_2d = model.predict_with_details([text])

    # Debug logging
    print(
        f"K-Means prediction - cluster_id: {cluster_id}, point_2d: {point_2d}, PCA loaded: {model.pca_2d is not None}"
    )

    # Convert distances to spam probability
    # Assuming cluster 0 is ham, cluster 1+ are spam-like
    spam_clusters = [1, 2]  # Adjust based on your cluster analysis

    total_dist = sum(distances.values())
    if total_dist > 0:
        spam_prob = sum(distances.get(c, 0) for c in spam_clusters) / total_dist
        # Invert because closer distance = higher probability
        spam_prob = 1 - spam_prob
    else:
        spam_prob = 0.5

    # Determine prediction based on closest cluster
    prediction = 1 if cluster_id in spam_clusters else 0

    # Calculate confidence based on distance separation
    dist_values = list(distances.values())
    if len(dist_values) > 1:
        confidence = abs(dist_values[0] - dist_values[1]) / max(dist_values)
    else:
        confidence = 0.5

    return {
        "model_id": "kmeans",
        "model_name": "K-Means",
        "prediction": prediction,
        "spam_probability": round(float(spam_prob), 4),
        "confidence": round(float(confidence), 4),
        "cluster_id": int(cluster_id),
        "cluster_distances": {int(k): round(float(v), 4) for k, v in distances.items()},
        "user_point_2d": [round(float(x), 4) for x in point_2d] if point_2d else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
