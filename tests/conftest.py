"""Pytest fixtures for RoadSafe AI."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.connection import Base, get_db
from app.main import app
from app.services.model_service import model_service

# Setup SQLite in-memory database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Buat tabel di SQLite
Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Timpa dependency database asli dengan SQLite
app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="session", autouse=True)
def load_model_once():
    """Pastikan model dimuat sebelum semua test dijalankan."""
    try:
        model_service.load_model()
    except Exception as e:
        pytest.exit(f"Gagal memuat model untuk testing: {e}")

@pytest.fixture(scope="module")
def client():
    """TestClient untuk API."""
    with TestClient(app) as c:
        yield c