from fastapi import APIRouter, HTTPException
from backend.schemas.telemetry import PredictRequest, PredictResponse
from backend.services.prediction import prediction_service
from backend.core.logging_config import logger

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    """Diagnoses machine operational telemetry for failure signals."""
    logger.info(f"Received prediction request for machine Type={payload.type_}")
    try:
        response = prediction_service.predict(payload)
        logger.info(f"Diagnosis completed. Failure={response.machine_failure}, Confidence={response.confidence_score:.2f}")
        return response
    except Exception as e:
        logger.error(f"Inference error during calculation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal inference error: {str(e)}"
        )

@router.get("/health")
def health_endpoint():
    """Returns application status and model loading indicator."""
    is_ready = prediction_service.model is not None
    return {
        "status": "healthy" if is_ready else "initializing",
        "model_loaded": is_ready,
        "features": prediction_service.feature_names
    }
