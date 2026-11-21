"""Repository layer for database operations."""
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.database import models

def create_prediction(
    db: Session,
    request_id: str,
    input_data: Dict[str, Any],
    prediction: str,
    probabilities: Dict[str, float],
    model_version: str
) -> models.PredictionHistory:
    """Save a new prediction to the database."""
    db_record = models.PredictionHistory(
        request_id=request_id,
        input_data=input_data,
        prediction=prediction,
        probabilities=probabilities,
        model_version=model_version
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

def get_predictions(db: Session, skip: int = 0, limit: int = 10) -> List[models.PredictionHistory]:
    """Fetch prediction history with pagination."""
    return db.query(models.PredictionHistory)\
             .order_by(models.PredictionHistory.created_at.desc())\
             .offset(skip).limit(limit).all()

def get_total_predictions(db: Session) -> int:
    """Get the total number of predictions made."""
    return db.query(models.PredictionHistory).count()