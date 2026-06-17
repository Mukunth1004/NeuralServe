from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from app.db.database import InferenceRecord
from app.models.schemas import PredictionResult
from app.logger import get_logger

logger = get_logger(__name__)


async def log_inference(
    db: AsyncSession,
    results: List[PredictionResult],
    runtime: str,
    request_id: Optional[str] = None,
    batch_size: int = 1,
):
    try:
        records = [
            InferenceRecord(
                request_id=request_id,
                input_text_length=r.text_length,
                label=r.label.value,
                confidence=r.confidence,
                inference_ms=r.inference_ms,
                batch_size=batch_size,
                runtime=runtime,
            )
            for r in results
        ]
        db.add_all(records)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to log inference: {e}")
        await db.rollback()


async def get_inference_logs(
    db: AsyncSession,
    limit: int = 50,
    offset: int = 0,
) -> List[InferenceRecord]:
    result = await db.execute(
        select(InferenceRecord)
        .order_by(InferenceRecord.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


async def get_metrics_summary(db: AsyncSession) -> dict:
    total = await db.scalar(select(func.count(InferenceRecord.id)))
    avg_ms = await db.scalar(select(func.avg(InferenceRecord.inference_ms)))
    avg_confidence = await db.scalar(select(func.avg(InferenceRecord.confidence)))
    label_rows = await db.execute(
        select(InferenceRecord.label, func.count(InferenceRecord.id)).group_by(
            InferenceRecord.label
        )
    )
    last_24h = await db.scalar(
        select(func.count(InferenceRecord.id)).where(
            InferenceRecord.created_at
            >= datetime.now(timezone.utc) - timedelta(hours=24)
        )
    )
    return {
        "total_inferences": total or 0,
        "avg_inference_ms": round(float(avg_ms or 0), 2),
        "avg_confidence": round(float(avg_confidence or 0), 4),
        "label_distribution": {row[0]: row[1] for row in label_rows},
        "last_24h_count": last_24h or 0,
    }
