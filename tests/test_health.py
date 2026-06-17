import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app


@pytest.fixture(autouse=True)
def mock_db():
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=MagicMock())
    with patch("app.db.database.AsyncSessionLocal", return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=False))):
        yield mock_session


@pytest.mark.asyncio
async def test_health_returns_ok():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "NeuralServe"
    assert "version" in data


@pytest.mark.asyncio
async def test_liveness():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


@pytest.mark.asyncio
async def test_readiness_model_not_loaded():
    with patch("app.routers.health.get_model_status", return_value="not_loaded"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "not_ready"
