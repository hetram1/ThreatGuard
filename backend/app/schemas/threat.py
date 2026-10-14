from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from ml.reliable.preprocessing import normalize_sms
from .url_validation import validated_url as normalize_url

InputType = Literal["url", "sms"]


class URLAnalysisRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=8192)

    @field_validator("url")
    @classmethod
    def valid_url(cls, value):
        return normalize_url(value)


class SMSAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)

    @field_validator("text")
    @classmethod
    def valid_sms(cls, value):
        return normalize_sms(value)


class UnifiedAnalysisRequest(BaseModel):
    input_type: InputType
    value: str = Field(..., min_length=1, max_length=10000)

    @model_validator(mode="after")
    def validate_value(self):
        self.value = (normalize_url if self.input_type == "url" else normalize_sms)(
            self.value
        )
        return self


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class AnalysisResponse(BaseModel):
    input_type: str
    prediction: str
    risk_score: float
    risk_band: str
    # Retained for database/client compatibility, never displayed as certainty.
    confidence: float = Field(
        description="Legacy max(p,1-p), NOT probability of correctness."
    )
    probability: float
    threshold: float
    model: dict[str, Any]
    reasons: list[str]
    model_evidence: list[dict[str, Any]]
    warnings: list[str]
    calibration: str
    score_description: str
    normalized_input: str


class ModelInfoResponse(BaseModel):
    service: str
    models: dict[str, dict[str, Any]]


class AnalysisHistoryItem(BaseModel):
    id: int
    input_type: str
    input_value: str
    prediction: str
    risk_score: float
    risk_band: str
    confidence: float
    model_name: str
    model_version: str
    reasons: list[str]
    created_at: str


class AnalysisHistoryResponse(BaseModel):
    total: int
    items: list[AnalysisHistoryItem]
