"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    # ... (Biarkan kode ini sama seperti sebelumnya) ...
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    speed_limit: int = Field(..., ge=0, le=150, description="Speed limit in mph")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month of year")
    road_type: str = Field(...)
    weather_conditions: str = Field(...)
    road_surface_conditions: str = Field(...)
    light_conditions: str = Field(...)
    urban_or_rural_area: str = Field(...)
    junction_detail: str = Field(...)

    model_config = {
        "json_schema_extra": {
            "example": {
                "longitude": -0.123, "latitude": 51.507, "speed_limit": 30,
                "hour": 18, "day_of_week": 5, "month": 11,
                "road_type": "Single carriageway", "weather_conditions": "Raining",
                "road_surface_conditions": "Wet", "light_conditions": "Darkness - lights lit",
                "urban_or_rural_area": "Urban", "junction_detail": "T or staggered junction"
            }
        }
    }

class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(..., max_length=100)

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    model_version: str
    request_id: str

    # TAMBAHKAN BARIS INI untuk menghilangkan warning
    model_config = {"protected_namespaces": ()}

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    algorithm: str
    features: list[str]
    is_loaded: bool

    # TAMBAHKAN BARIS INI
    model_config = {"protected_namespaces": ()}

class FeatureContribution(BaseModel):
    feature: str
    value: float
    impact: str

class ExplanationResponse(PredictionResponse):
    explanation: list[FeatureContribution]

    # TAMBAHKAN BARIS INI
    model_config = {"protected_namespaces": ()}