from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class PredictRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 50.0
            }
        }
    )

    type_: Literal["L", "M", "H"] = Field(..., alias="Type", description="Machine quality variant: L (low), M (medium), H (high)")
    
    # Valid ranges based on dataset limits
    air_temp: float = Field(
        ..., 
        alias="Air temperature [K]", 
        ge=280.0, 
        le=320.0, 
        description="Air temperature in Kelvin (realistic operational limit: 280K - 320K)"
    )
    proc_temp: float = Field(
        ..., 
        alias="Process temperature [K]", 
        ge=280.0, 
        le=320.0, 
        description="Process temperature in Kelvin (realistic operational limit: 280K - 320K)"
    )
    rot_speed: float = Field(
        ..., 
        alias="Rotational speed [rpm]", 
        ge=500.0, 
        le=4000.0, 
        description="Rotational speed in RPM (realistic operational limit: 500 - 4000 rpm)"
    )
    torque: float = Field(
        ..., 
        alias="Torque [Nm]", 
        ge=0.0, 
        le=100.0, 
        description="Torque in Nm (realistic operational limit: 0 - 100 Nm)"
    )
    tool_wear: float = Field(
        ..., 
        alias="Tool wear [min]", 
        ge=0.0, 
        le=300.0, 
        description="Tool wear time in minutes (realistic operational limit: 0 - 300 min)"
    )

class PredictResponse(BaseModel):
    machine_failure: int = Field(..., description="Binary prediction: 1 = Failure, 0 = No Failure")
    failure_probability: float = Field(..., description="Model probability output for machine failure class")
    confidence_score: float = Field(..., description="Model confidence score for the predicted class")
    recommendation: str = Field(..., description="Heuristic operational recommendation based on input conditions")
