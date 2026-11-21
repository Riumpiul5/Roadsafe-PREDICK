"""Test ModelService loading and inference."""
from app.schemas.prediction import PredictionRequest
from app.services.model_service import ModelService


def test_model_load():
    service = ModelService()
    service.load_model()
    assert service.pipeline is not None

def test_prediction_output():
    service = ModelService()
    service.load_model()
    
    req = PredictionRequest(
        longitude=-0.123, latitude=51.507, speed_limit=30,
        hour=18, day_of_week=5, month=11,
        road_type="Single carriageway", weather_conditions="Raining",
        road_surface_conditions="Wet", light_conditions="Darkness - lights lit",
        urban_or_rural_area="Urban", junction_detail="T or staggered junction"
    )
    result = service.predict(req)
    
    assert "prediction" in result
    assert result["prediction"] in ["Fatal", "Serious", "Slight"]
    assert "probabilities" in result
    # Total probabilitas mendekati 1.0
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.01