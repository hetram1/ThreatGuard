from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas.threat import (
    AnalysisResponse,
    HealthResponse,
    ModelInfoResponse,
    SMSAnalysisRequest,
    UnifiedAnalysisRequest,
    URLAnalysisRequest,
)
from .services.threat_analysis_service import (
    ThreatAnalysisService,
)


APP_VERSION = "1.0.0"


app = FastAPI(
    title="ThreatGuard API",
    description=(
        "Real-Time AI Digital Threat Intelligence API "
        "for URL phishing and SMS scam detection."
    ),
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


@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, str]:
    return {
        "service": "ThreatGuard API",
        "version": APP_VERSION,
        "status": "online",
        "docs": "/docs",
    }


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="ThreatGuard API",
        version=APP_VERSION,
    )


@app.post(
    "/api/v1/analyze/url",
    response_model=AnalysisResponse,
    tags=["Threat Analysis"],
)
def analyze_url(
    request: URLAnalysisRequest,
) -> AnalysisResponse:
    try:
        result = threat_service.analyze_url(
            request.url
        )

        return AnalysisResponse(
            input_type=result["input_type"],
            prediction=result["prediction"],
            risk_score=result["risk_score"],
            risk_band=result["risk_band"],
            confidence=result["confidence"],
            threshold=result["threshold"],
            model=result["model"],
            reasons=result["reasons"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="URL analysis failed.",
        ) from exc


@app.post(
    "/api/v1/analyze/sms",
    response_model=AnalysisResponse,
    tags=["Threat Analysis"],
)
def analyze_sms(
    request: SMSAnalysisRequest,
) -> AnalysisResponse:
    try:
        result = threat_service.analyze_sms(
            request.text
        )

        return AnalysisResponse(
            input_type=result["input_type"],
            prediction=result["prediction"],
            risk_score=result["risk_score"],
            risk_band=result["risk_band"],
            confidence=result["confidence"],
            threshold=result["threshold"],
            model=result["model"],
            reasons=result["reasons"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="SMS analysis failed.",
        ) from exc


@app.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    tags=["Threat Analysis"],
)
def analyze(
    request: UnifiedAnalysisRequest,
) -> AnalysisResponse:
    try:
        result = threat_service.analyze(
            request.input_type,
            request.value,
        )

        return AnalysisResponse(
            input_type=result["input_type"],
            prediction=result["prediction"],
            risk_score=result["risk_score"],
            risk_band=result["risk_band"],
            confidence=result["confidence"],
            threshold=result["threshold"],
            model=result["model"],
            reasons=result["reasons"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Threat analysis failed.",
        ) from exc


@app.get(
    "/api/v1/model-info",
    response_model=ModelInfoResponse,
    tags=["System"],
)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        service="ThreatGuard",
        models={
            "url": {
                "name": "ThreatGuard URL Detector",
                "version": "v5",
                "type": (
                    "Character TF-IDF + engineered "
                    "URL features + Logistic Regression"
                ),
                "threshold": 0.50,
            },
            "sms": {
                "name": "ThreatGuard SMS Scam Detector",
                "version": "v1",
                "type": (
                    "Word + Character TF-IDF "
                    "+ Logistic Regression"
                ),
                "threshold": 0.50,
            },
        },
    )
