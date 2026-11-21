"""API routes for RoadSafe AI."""
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.schemas.prediction import (
    PredictionRequest, PredictionResponse, 
    BatchPredictionRequest, BatchPredictionResponse,
    ModelInfoResponse, ExplanationResponse,
)
from app.services.model_service import model_service
from app.database.connection import get_db
from app.database import repository

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to RoadSafe AI API", "docs": "/docs"}

@router.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "model_loaded": model_service.pipeline is not None}

@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    if not model_service.pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="RoadSafe-Severity-Classifier",
        model_version=model_service.model_version,
        algorithm=type(model_service.pipeline.named_steps['classifier']).__name__,
        features=list(model_service.pipeline.feature_names_in_) if hasattr(model_service.pipeline, 'feature_names_in_') else [],
        is_loaded=True
    )

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest, raw_request: Request, db: Session = Depends(get_db)):
    req_id = raw_request.headers.get("X-Request-ID", str(uuid.uuid4()))
    logger.info(f"Received /predict request. ID: {req_id}")
    
    try:
        # 1. Dapatkan hasil prediksi dari model
        result = model_service.predict(request, request_id=req_id)
        
        # 2. Simpan ke Database
        repository.create_prediction(
            db=db,
            request_id=req_id,
            input_data=request.model_dump(),
            prediction=result["prediction"],
            probabilities=result["probabilities"],
            model_version=result["model_version"]
        )
        
        return result
    except Exception as e:
        logger.error(f"Prediction or DB save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest, raw_request: Request, db: Session = Depends(get_db)):
    req_id = raw_request.headers.get("X-Request-ID", str(uuid.uuid4()))
    logger.info(f"Received /predict/batch request. ID: {req_id}, Size: {len(request.items)}")
    
    try:
        results = model_service.predict_batch(request.items)
        return BatchPredictionResponse(predictions=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions", tags=["Prediction History"])
def get_predictions(limit: int = 10, offset: int = 0, db: Session = Depends(get_db)):
    """Retrieve prediction history from database."""
    predictions = repository.get_predictions(db, skip=offset, limit=limit)
    total = repository.get_total_predictions(db)
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": predictions
    }

@router.get("/metrics", tags=["Metrics"])
def get_metrics():
    # Placeholder: Akan mengembalikan metrics dari MLflow/DB
    return {"total_predictions": 0, "avg_latency_ms": 0.0}

@router.post("/predict/explain", response_model=ExplanationResponse, tags=["Prediction"])
def predict_explain(request: PredictionRequest, raw_request: Request, db: Session = Depends(get_db)):
    req_id = raw_request.headers.get("X-Request-ID", str(uuid.uuid4()))
    logger.info(f"Received /predict/explain request. ID: {req_id}")
    
    try:
        result = model_service.explain_prediction(request, request_id=req_id)
        
        # Opsional: Tetap simpan ke database
        repository.create_prediction(
            db=db,
            request_id=req_id,
            input_data=request.model_dump(),
            prediction=result["prediction"],
            probabilities=result["probabilities"],
            model_version=result["model_version"]
        )
        
        return result
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))