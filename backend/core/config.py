import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True,
    )

    APP_NAME: str = "AI-Based Predictive Maintenance API"
    APP_VERSION: str = "1.0.0"
    ENV: str = os.getenv("ENV", "development")

    # Security: explicit list of allowed frontend origins
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Model artifact file paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR: str = os.path.join(os.path.dirname(BASE_DIR), "model")
    MODEL_PATH: str = os.path.join(MODEL_DIR, "machine_failure_model.pkl")
    ENCODERS_PATH: str = os.path.join(MODEL_DIR, "label_encoders.pkl")
    FEATURES_PATH: str = os.path.join(MODEL_DIR, "feature_names.pkl")

    # MLflow Tracking Settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = "Predictive_Maintenance_Inference"
    MLFLOW_TRAINING_EXPERIMENT_NAME: str = "Predictive_Maintenance_Training"

    # MLflow HTTP client tuning: int types, sensible defaults for a local server
    # MLFLOW_TIMEOUT=3: wait up to 3 seconds before giving up on tracking server
    # MLFLOW_MAX_RETRIES=1: attempt only once to avoid blocking the request
    MLFLOW_TIMEOUT: int = 3
    MLFLOW_MAX_RETRIES: int = 1


settings = Settings()
