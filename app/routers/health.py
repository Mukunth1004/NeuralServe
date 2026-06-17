from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db.database import get_db
from app.services.model_service import get_model_status
from app.config import get_settings

settings = get_settings()
router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.APP_ENV,
    }


@router.get("/health/live")
async def liveness():
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    checks = {"model": get_model_status(), "database": "unreachable"}
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "connected"
    except Exception:
        pass
    ready = all(v in ("ready", "connected") for v in checks.values())
    return {"status": "ready" if ready else "not_ready", "checks": checks}
