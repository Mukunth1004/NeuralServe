import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import get_settings
from app.logger import get_logger
from app.db.database import init_db, close_db
from app.services.model_service import load_model
from app.routers import health, inference, metrics

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.APP_ENV}]")
    await init_db()
    await load_model()
    logger.info("Application startup complete — ready to serve")
    yield
    logger.info("Shutting down...")
    await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade transformer inference platform. "
        "DistilBERT sentiment analysis with ONNX graph optimization, "
        "dynamic memory-aware batching, and PostgreSQL inference logging."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{duration_ms}ms]")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception | {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health.router)
app.include_router(inference.router)
app.include_router(metrics.router)
