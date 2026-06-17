from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from app.db.database import get_db
from app.models.schemas import (
    PredictRequest,
    PredictResponse,
    PredictBatchRequest,
    BatchPredictResponse,
    ModelInfo,
)
from app.services import model_service, db_service
from app.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/api/v1", tags=["Inference"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    if model_service.get_model_status() != "ready":
        raise HTTPException(status_code=503, detail="Model not ready — check /health/ready")

    results, total_ms = await model_service.predict_batch(
        texts=[req.text],
        request_id=req.request_id,
    )
    background_tasks.add_task(
        db_service.log_inference,
        db,
        results,
        model_service.get_model_runtime(),
        req.request_id,
        1,
    )
    return PredictResponse(
        request_id=req.request_id,
        result=results[0],
        model_name=settings.MODEL_NAME,
        runtime=model_service.get_model_runtime(),
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    req: PredictBatchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    if model_service.get_model_status() != "ready":
        raise HTTPException(status_code=503, detail="Model not ready — check /health/ready")

    if len(req.texts) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.texts)} exceeds max {settings.MAX_BATCH_SIZE}",
        )

    results, total_ms = await model_service.predict_batch(
        texts=req.texts,
        request_id=req.request_id,
    )
    background_tasks.add_task(
        db_service.log_inference,
        db,
        results,
        model_service.get_model_runtime(),
        req.request_id,
        len(req.texts),
    )
    return BatchPredictResponse(
        request_id=req.request_id,
        results=results,
        count=len(results),
        model_name=settings.MODEL_NAME,
        runtime=model_service.get_model_runtime(),
        total_inference_ms=total_ms,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/model/info", response_model=ModelInfo)
async def model_info():
    return ModelInfo(
        name=settings.MODEL_NAME,
        architecture="DistilBERT (Transformer — 6-layer, 768-dim, 12-heads, 66M params)",
        task="text-classification / sentiment-analysis",
        runtime=model_service.get_model_runtime(),
        onnx_optimized=model_service.get_model_runtime() == "onnx",
        max_sequence_length=settings.MAX_SEQUENCE_LENGTH,
        labels=["POSITIVE", "NEGATIVE"],
        status=model_service.get_model_status(),
    )
