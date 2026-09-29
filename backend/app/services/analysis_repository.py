import json
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from backend.app.models.analysis import Analysis


class AnalysisRepository:
    """
    Persistence layer for threat analyses.

    Keeps database operations separate from:
        - ML inference
        - FastAPI routing
        - frontend logic
    """

    @staticmethod
    def create(
        db: Session,
        result: dict[str, Any],
        input_value: str,
    ) -> Analysis:

        model_info = result.get(
            "model",
            {},
        )

        analysis = Analysis(
            input_type=result["input_type"],
            input_value=input_value,
            prediction=result["prediction"],
            risk_score=result["risk_score"],
            risk_band=result["risk_band"],
            confidence=result["confidence"],
            model_name=model_info.get(
                "name",
                "Unknown",
            ),
            model_version=model_info.get(
                "version",
                "Unknown",
            ),
            reasons=json.dumps(
                result.get(
                    "reasons",
                    [],
                )
            ),
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return analysis

    @staticmethod
    def get_recent(
        db: Session,
        limit: int = 50,
    ) -> list[Analysis]:

        limit = max(
            1,
            min(limit, 100),
        )

        statement = (
            select(Analysis)
            .order_by(
                desc(
                    Analysis.created_at
                )
            )
            .limit(limit)
        )

        return list(
            db.scalars(statement).all()
        )

    @staticmethod
    def count(
        db: Session,
    ) -> int:
        statement = select(
            func.count(Analysis.id)
        )

        return int(
            db.scalar(statement) or 0
        )
