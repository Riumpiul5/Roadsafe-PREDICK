"""SQLAlchemy ORM models."""
from sqlalchemy import JSON, Column, DateTime, Integer, String, func  # Ubah import ini

from app.database.connection import Base


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True, nullable=False)
    input_data = Column(JSON, nullable=False) # Ganti JSONB ke JSON
    prediction = Column(String, nullable=False)
    probabilities = Column(JSON, nullable=False) # Ganti JSONB ke JSON
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True, index=True, nullable=False)
    version = Column(String, nullable=False)
    algorithm = Column(String, nullable=False)
    trained_at = Column(DateTime(timezone=True), server_default=func.now())
    metrics = Column(JSON, nullable=True) # Ganti JSONB ke JSON