"""Test FastAPI endpoints."""
import pytest

VALID_PAYLOAD = {
    "longitude": -0.123, "latitude": 51.507, "speed_limit": 30,
    "hour": 18, "day_of_week": 5, "month": 11,
    "road_type": "Single carriageway", "weather_conditions": "Raining",
    "road_surface_conditions": "Wet", "light_conditions": "Darkness - lights lit",
    "urban_or_rural_area": "Urban", "junction_detail": "T or staggered junction"
}

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    assert "model_name" in response.json()

def test_predict_success(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "request_id" in data
    assert "probabilities" in data

def test_predict_missing_field(client):
    invalid_payload = VALID_PAYLOAD.copy()
    del invalid_payload["speed_limit"] # Hapus field wajib
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422 # Unprocessable Entity (Pydantic error)

def test_predict_invalid_type(client):
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["speed_limit"] = "sangat cepat" # Harusnya integer
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_batch_predict(client):
    payload = {"items": [VALID_PAYLOAD, VALID_PAYLOAD]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert len(response.json()["predictions"]) == 2

def test_get_predictions_pagination(client):
    # Lakukan 1 prediksi agar ada data di SQLite
    client.post("/predict", json=VALID_PAYLOAD)
    
    response = client.get("/predictions?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["limit"] == 5
    assert data["offset"] == 0
    assert "data" in data