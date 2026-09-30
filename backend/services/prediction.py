import os
import time
import joblib
import pandas as pd
import mlflow
from datetime import datetime
from backend.core.config import settings
from backend.core.logging_config import logger
from backend.schemas.telemetry import PredictRequest, PredictResponse


class PredictionService:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_names = None

    def load_artifacts(self):
        """Loads machine learning assets from the localized models folder.
        
        This only loads model files. MLflow configuration is applied per-prediction
        so that temporary MLflow outages never permanently disable logging.
        """
        logger.info("Initializing prediction model artifacts...")

        if not os.path.exists(settings.MODEL_PATH):
            raise FileNotFoundError(f"Model pkl not found at: {settings.MODEL_PATH}")
        if not os.path.exists(settings.ENCODERS_PATH):
            raise FileNotFoundError(f"Label encoders pkl not found at: {settings.ENCODERS_PATH}")
        if not os.path.exists(settings.FEATURES_PATH):
            raise FileNotFoundError(f"Feature names pkl not found at: {settings.FEATURES_PATH}")

        self.model = joblib.load(settings.MODEL_PATH)
        self.encoders = joblib.load(settings.ENCODERS_PATH)
        self.feature_names = joblib.load(settings.FEATURES_PATH)

        logger.info(f"Loaded feature names: {self.feature_names}")
        logger.info("All model artifacts loaded successfully.")

        # Set MLflow HTTP client environment variables once at startup.
        # These control the underlying HTTP request timeout and retry behavior
        # used by the MLflow Python client when connecting to the tracking server.
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(settings.MLFLOW_TIMEOUT)
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(settings.MLFLOW_MAX_RETRIES)

    def _log_to_mlflow(
        self,
        raw_type: str,
        input_dict: dict,
        prediction: int,
        failure_prob: float,
        confidence: float,
        inference_time_ms: float,
    ) -> None:
        """Attempts to log a single prediction run to MLflow.
        
        Why per-prediction instead of once at startup:
        - If MLflow is temporarily offline at startup, this approach retries on
          every subsequent prediction instead of silently skipping forever.
        - If MLflow recovers mid-session, logging resumes automatically.
        - Prediction requests are NEVER blocked or failed due to MLflow issues.
        
        Parameters logged:
        - All six raw telemetry sensor inputs (human-readable, not encoded)
        
        Metrics logged:
        - prediction, failure_probability, confidence_score, inference_time_ms
        
        Tags logged:
        - model_name, model_version, api_version for experiment traceability
        """
        # Generate a unique, readable run name using a timestamp
        run_name = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Parameters use the raw human-readable input (not the encoded integer)
        params = {
            "Type": raw_type,
            "Air_temperature_K": input_dict["Air temperature [K]"],
            "Process_temperature_K": input_dict["Process temperature [K]"],
            "Rotational_speed_rpm": input_dict["Rotational speed [rpm]"],
            "Torque_Nm": input_dict["Torque [Nm]"],
            "Tool_wear_min": input_dict["Tool wear [min]"],
        }

        metrics = {
            "prediction": float(prediction),
            "failure_probability": failure_prob,
            "confidence_score": confidence,
            "inference_time_ms": inference_time_ms,
        }

        tags = {
            "model_name": "RandomForest",
            "model_version": "1.0",
            "api_version": settings.APP_VERSION,
        }

        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.set_tags(tags)

            logger.info(f"MLflow run logged: {run_name}")
        except Exception as e:
            # Log warning only — never raise, never block the prediction response
            logger.warning(f"MLflow logging skipped for run '{run_name}': {str(e)}")

    def _generate_recommendation(self, payload: PredictRequest, failure_predicted: int) -> str:
        """Heuristic maintenance checklist based on telemetry thresholds."""
        recs = []
        if failure_predicted == 1:
            recs.append("CRITICAL WARNING: High probability of machine failure detected!")
            if payload.tool_wear > 200:
                recs.append("- Check Tool Wear: Elapsed time exceeds safety threshold (>200 min). Replace tool.")
            if payload.torque > 60:
                recs.append("- Check Motor Load: Excessive torque load detected. Reduce spindle torque.")
            if (payload.proc_temp - payload.air_temp) < 8.6 and payload.rot_speed < 1400:
                recs.append("- Inspect Coolant: Insufficient heat dissipation profile. Check cooling fans/lines.")
            if payload.rot_speed > 2800:
                recs.append("- Adjust Speed: Rotational speed is overstraining tool components.")
            if len(recs) == 1:
                recs.append("- General System Check: Conduct physical maintenance checks on moving components.")
        else:
            recs.append("Machine is operating within normal performance bounds.")
            if payload.tool_wear > 150:
                recs.append("- Tool wear is elevated (>150 min). Schedule maintenance review soon.")
            if payload.torque > 55:
                recs.append("- Monitor Torque: Operating torque is approaching maximum parameters.")

        return "\n".join(recs)

    def predict(self, payload: PredictRequest) -> PredictResponse:
        """Runs inference on telemetry input and logs results to MLflow.
        
        Flow:
        1. Build raw input dictionary preserving original human-readable values.
        2. Encode the 'Type' field (L/M/H → integer) for model inference only.
        3. Construct DataFrame in exact training feature order.
        4. Run prediction and measure latency in milliseconds.
        5. Attempt MLflow logging — failures here NEVER affect the API response.
        6. Return structured response.
        """
        if self.model is None or self.encoders is None or self.feature_names is None:
            raise RuntimeError("Model artifacts not loaded. Call load_artifacts() first.")

        # Step 1: Preserve raw input values for readable MLflow logging
        raw_input = {
            "Type": payload.type_,
            "Air temperature [K]": payload.air_temp,
            "Process temperature [K]": payload.proc_temp,
            "Rotational speed [rpm]": payload.rot_speed,
            "Torque [Nm]": payload.torque,
            "Tool wear [min]": payload.tool_wear,
        }

        # Step 2: Build encoded input for model inference
        encoded_input = dict(raw_input)
        type_encoder = self.encoders.get("Type")
        if type_encoder:
            try:
                encoded_input["Type"] = int(type_encoder.transform([payload.type_])[0])
            except ValueError:
                # Fallback to 'L' if unknown type provided (won't happen with Pydantic Literal)
                encoded_input["Type"] = int(type_encoder.transform(["L"])[0])

        # Step 3: Build DataFrame in exact training feature order
        df_input = pd.DataFrame([encoded_input])[self.feature_names]

        # Step 4: Run model inference and measure latency
        start_time = time.perf_counter()
        prediction = int(self.model.predict(df_input)[0])
        probabilities = self.model.predict_proba(df_input)[0]
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0  # Convert to milliseconds

        confidence = float(probabilities[prediction])
        failure_prob = float(probabilities[1])

        # Step 5: Attempt MLflow logging — isolated from prediction logic
        self._log_to_mlflow(
            raw_type=payload.type_,
            input_dict=raw_input,
            prediction=prediction,
            failure_prob=failure_prob,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
        )

        # Step 6: Generate recommendations and return response
        recommendation = self._generate_recommendation(payload, prediction)

        return PredictResponse(
            machine_failure=prediction,
            failure_probability=failure_prob,
            confidence_score=confidence,
            recommendation=recommendation,
        )


# Singleton instance — loaded once and reused across all requests
prediction_service = PredictionService()
