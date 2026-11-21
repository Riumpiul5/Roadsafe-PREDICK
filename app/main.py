"""Main FastAPI application."""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.services.model_service import model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RoadSafe AI API...")
    try:
        model_service.load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    yield
    # Shutdown
    logger.info("Shutting down RoadSafe AI API...")

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    description="Intelligent Traffic Accident Severity Prediction System",
    version=settings.model_version,
    lifespan=lifespan
)

# CORS (izinkan semua untuk development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware untuk Request ID
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    # Simpan di state agar bisa diakses di route
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Include routes
app.include_router(router)