from typing import Any, Literal

from pydantic import BaseModel, Field


InputType = Literal["url", "sms"]


class URLAnalysisRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="URL to analyze.",
    )


class SMSAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="SMS or message text to analyze.",
    )


class UnifiedAnalysisRequest(BaseModel):
    input_type: InputType
    value: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="URL or SMS content.",
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class AnalysisResponse(BaseModel):
    input_type: str
    prediction: str
    risk_score: float
    risk_band: str
    confidence: float
    threshold: float
    model: dict[str, Any]
    reasons: list[str]


class ModelInfoResponse(BaseModel):
    service: str
    models: dict[str, dict[str, Any]]
