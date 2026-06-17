from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.services import db_service

router = APIRouter(prefix="/api/v1", tags=["Metrics & Logs"])


@router.get("/metrics")
async def metrics(db: AsyncSession = Depends(get_db)):
    return await db_service.get_metrics_summary(db)


@router.get("/logs")
async def logs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    records = await db_service.get_inference_logs(db, limit=limit, offset=offset)
    return {
        "count": len(records),
        "logs": [
            {
                "id": r.id,
                "request_id": r.request_id,
                "input_text_length": r.input_text_length,
                "label": r.label,
                "confidence": r.confidence,
                "inference_ms": r.inference_ms,
                "batch_size": r.batch_size,
                "runtime": r.runtime,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ],
    }
