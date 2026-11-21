"""Centralised configuration via pydantic-settings.

Loads environment variables from .env (local) or container env (Docker).
No secrets are hardcoded.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = Field(default="RoadSafe AI", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- Database ---
    database_url: str = Field(
        default="postgresql+psycopg2://roadsafe:roadsafe@localhost:5432/roadsafe",
        alias="DATABASE_URL",
    )

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="roadsafe-severity", alias="MLFLOW_EXPERIMENT_NAME"
    )
    mlflow_model_name: str = Field(
        default="roadsafe-severity-model", alias="MLFLOW_MODEL_NAME"
    )

    # --- Model artifacts ---
    model_dir: Path = Field(default=Path("artifacts/model"), alias="MODEL_DIR")
    model_filename: str = Field(
        default="roadsafe_pipeline.joblib", alias="MODEL_FILENAME"
    )
    model_version: str = Field(default="0.0.0", alias="MODEL_VERSION")

    # --- Dataset ---
    dataset_raw_dir: Path = Field(
        default=Path("dataset/raw"), alias="DATASET_RAW_DIR"
    )
    dataset_processed_dir: Path = Field(
        default=Path("dataset/processed"), alias="DATASET_PROCESSED_DIR"
    )

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()