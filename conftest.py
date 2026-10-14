"""Never run API tests against the user's configured database."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(autouse=True)
def isolated_database(request):
    if "api" not in str(request.node.fspath) and "integration" not in str(
        request.node.fspath
    ):
        yield
        return
    from backend.app.main import app
    from backend.app.database.database import Base, get_db

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    factory = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    def override():
        with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override
    yield
    app.dependency_overrides.pop(get_db, None)
    engine.dispose()


@pytest.fixture
def client(isolated_database):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    with TestClient(app) as c:
        yield c
