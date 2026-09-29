from backend.app.database.database import (
    Base,
    engine,
)

# Import models so SQLAlchemy registers their tables.
from backend.app.models.analysis import Analysis  # noqa: F401


def init_db() -> None:
    Base.metadata.create_all(
        bind=engine
    )


if __name__ == "__main__":
    init_db()
    print(
        "ThreatGuard database tables initialized."
    )
