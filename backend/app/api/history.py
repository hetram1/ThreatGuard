import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.services.analysis_repository import (
    AnalysisRepository,
)
from backend.app.schemas.threat import (
    AnalysisHistoryItem,
    AnalysisHistoryResponse,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["History"],
)


@router.get(
    "/history",
    response_model=AnalysisHistoryResponse,
)
def get_history(
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
) -> AnalysisHistoryResponse:

    analyses = AnalysisRepository.get_recent(
        db,
        limit=limit,
    )

    items = []

    for analysis in analyses:
        try:
            reasons = json.loads(
                analysis.reasons
            )
        except (
            json.JSONDecodeError,
            TypeError,
        ):
            reasons = []

        items.append(
            AnalysisHistoryItem(
                id=analysis.id,
                input_type=analysis.input_type,
                input_value=analysis.input_value,
                prediction=analysis.prediction,
                risk_score=analysis.risk_score,
                risk_band=analysis.risk_band,
                confidence=analysis.confidence,
                model_name=analysis.model_name,
                model_version=analysis.model_version,
                reasons=reasons,
                created_at=analysis.created_at.isoformat(),
            )
        )

    total = AnalysisRepository.count(db)

    return AnalysisHistoryResponse(
        total=total,
        items=items,
    )
