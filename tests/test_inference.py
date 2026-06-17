import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app
from app.models.schemas import PredictionResult, Label


MOCK_POSITIVE = PredictionResult(
    label=Label.POSITIVE,
    confidence=0.9987,
    scores={"POSITIVE": 0.9987, "NEGATIVE": 0.0013},
    text_length=30,
    inference_ms=12.5,
)

MOCK_NEGATIVE = PredictionResult(
    label=Label.NEGATIVE,
    confidence=0.9871,
    scores={"POSITIVE": 0.0129, "NEGATIVE": 0.9871},
    text_length=25,
    inference_ms=11.2,
)


@pytest.fixture(autouse=True)
def mock_db():
    mock_session = AsyncMock()
    with patch(
        "app.db.database.AsyncSessionLocal",
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(),
        ),
    ):
        yield mock_session


@pytest.mark.asyncio
async def test_predict_positive():
    with (
        patch("app.routers.inference.model_service.get_model_status", return_value="ready"),
        patch("app.routers.inference.model_service.get_model_runtime", return_value="pytorch"),
        patch(
            "app.routers.inference.model_service.predict_batch",
            return_value=([MOCK_POSITIVE], 12.5),
        ),
        patch("app.routers.inference.db_service.log_inference", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/predict",
                json={"text": "This product is absolutely amazing!"},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["label"] == "POSITIVE"
    assert data["result"]["confidence"] > 0.9
    assert "runtime" in data


@pytest.mark.asyncio
async def test_predict_negative():
    with (
        patch("app.routers.inference.model_service.get_model_status", return_value="ready"),
        patch("app.routers.inference.model_service.get_model_runtime", return_value="pytorch"),
        patch(
            "app.routers.inference.model_service.predict_batch",
            return_value=([MOCK_NEGATIVE], 11.2),
        ),
        patch("app.routers.inference.db_service.log_inference", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/predict",
                json={"text": "Terrible experience, very disappointed."},
            )
    assert response.status_code == 200
    assert response.json()["result"]["label"] == "NEGATIVE"


@pytest.mark.asyncio
async def test_predict_model_not_ready():
    with patch("app.routers.inference.model_service.get_model_status", return_value="not_loaded"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/predict", json={"text": "Test"})
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_batch_predict():
    mock_results = [MOCK_POSITIVE, MOCK_NEGATIVE]
    with (
        patch("app.routers.inference.model_service.get_model_status", return_value="ready"),
        patch("app.routers.inference.model_service.get_model_runtime", return_value="pytorch"),
        patch(
            "app.routers.inference.model_service.predict_batch",
            return_value=(mock_results, 24.0),
        ),
        patch("app.routers.inference.db_service.log_inference", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/predict/batch",
                json={"texts": ["Great product!", "Terrible experience."]},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert "total_inference_ms" in data


@pytest.mark.asyncio
async def test_batch_exceeds_pydantic_max():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict/batch",
            json={"texts": ["text"] * 33},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_model_info():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "DistilBERT" in data["architecture"]
    assert "Transformer" in data["architecture"]
    assert "POSITIVE" in data["labels"]
    assert "NEGATIVE" in data["labels"]


@pytest.mark.asyncio
async def test_predict_empty_text_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/predict", json={"text": ""})
    assert response.status_code == 422
