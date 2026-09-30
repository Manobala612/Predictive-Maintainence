from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import settings
from backend.core.logging_config import setup_logging, logger
from backend.api.endpoints import router as api_router
from backend.services.prediction import prediction_service

# Initialize central logging configurations
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing FastAPI application services...")
    try:
        prediction_service.load_artifacts()
        logger.info("FastAPI service lifespan context initialized.")
    except Exception as e:
        logger.critical(f"Failed to load model artifacts on startup: {str(e)}")
    
    yield
    
    # Shutdown actions (if any required in future, e.g. closing session pools)
    logger.info("Shutting down FastAPI application services...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="REST API service utilizing a Random Forest model to predict machine failure from operational parameters.",
    lifespan=lifespan
)

# Enable CORS for explicit domains defined in settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router, prefix="/api/v1")
