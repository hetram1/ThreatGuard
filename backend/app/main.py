"""ThreatGuard HTTP API. Validation, inference and persistence have separate roles."""

import logging
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from .api.history import router as history_router
from .database.database import get_db
from .schemas.threat import (
    AnalysisResponse,
    HealthResponse,
    ModelInfoResponse,
    SMSAnalysisRequest,
    UnifiedAnalysisRequest,
    URLAnalysisRequest,
)
from .services.analysis_repository import AnalysisRepository
from .services.threat_analysis_service import ThreatAnalysisService

APP_VERSION = "1.0.0"
logger = logging.getLogger(__name__)
app = FastAPI(
    title="ThreatGuard API",
    description="Real-Time AI Digital Threat Intelligence API for URL phishing and SMS scam detection.",
    version=APP_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
threat_service = ThreatAnalysisService()


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {
        "service": "ThreatGuard API",
        "version": APP_VERSION,
        "status": "online",
        "docs": "/docs",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy", service="ThreatGuard API", version=APP_VERSION
    )


def analyze_and_store(
    input_type: str, value: str, db: Session, error_detail: str
) -> AnalysisResponse:
    """All analysis routes use the same inference/persistence/error contract."""
    try:
        result = threat_service.analyze(input_type, value)
        AnalysisRepository.create(db, result, value)
        return AnalysisResponse(**result)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        # Avoid recording user input or database parameter values in logs.
        logger.error("%s (%s)", error_detail, type(exc).__name__)
        raise HTTPException(status_code=500, detail=error_detail) from exc


@app.post(
    "/api/v1/analyze/url", response_model=AnalysisResponse, tags=["Threat Analysis"]
)
def analyze_url(
    request: URLAnalysisRequest, db: Session = Depends(get_db)
) -> AnalysisResponse:
    return analyze_and_store("url", request.url, db, "URL analysis failed.")


@app.post(
    "/api/v1/analyze/sms", response_model=AnalysisResponse, tags=["Threat Analysis"]
)
def analyze_sms(
    request: SMSAnalysisRequest, db: Session = Depends(get_db)
) -> AnalysisResponse:
    return analyze_and_store("sms", request.text, db, "SMS analysis failed.")


@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Threat Analysis"])
def analyze(
    request: UnifiedAnalysisRequest, db: Session = Depends(get_db)
) -> AnalysisResponse:
    return analyze_and_store(
        request.input_type, request.value, db, "Threat analysis failed."
    )


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        service="ThreatGuard",
        models={
            kind: {
                **engine.metadata,
                "threshold": engine.info["threshold"],
                "calibration": engine.info["calibration_method"],
            }
            for kind, engine in [
                ("url", threat_service.url_engine),
                ("sms", threat_service.sms_engine),
            ]
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    errors = [
        {
            "field": ".".join(str(part) for part in error["loc"] if part != "body"),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": "; ".join(error["message"] for error in errors),
            "errors": errors,
        },
    )


app.include_router(history_router)
