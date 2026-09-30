import logging
import sys

def setup_logging():
    # Setup base configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Minimize noisy loggers from external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.ERROR)

logger = logging.getLogger("PredictiveMaintenance")
