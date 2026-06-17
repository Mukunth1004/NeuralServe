from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Label(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text for sentiment analysis")
    request_id: Optional[str] = Field(None, description="Optional client-side request ID")

    model_config = {
        "json_schema_extra": {"example": {"text": "The product quality is absolutely excellent!"}}
    }


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32, description="List of texts (max 32)")
    request_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {"texts": ["Great product!", "Terrible experience."]}
        }
    }


class PredictionResult(BaseModel):
    label: Label
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: dict[str, float]
    text_length: int
    inference_ms: float


class PredictResponse(BaseModel):
    request_id: Optional[str]
    result: PredictionResult
    model_name: str
    runtime: str
    timestamp: datetime


class BatchPredictResponse(BaseModel):
    request_id: Optional[str]
    results: List[PredictionResult]
    count: int
    model_name: str
    runtime: str
    total_inference_ms: float
    timestamp: datetime


class ModelInfo(BaseModel):
    name: str
    architecture: str
    task: str
    runtime: str
    onnx_optimized: bool
    max_sequence_length: int
    labels: List[str]
    status: str
