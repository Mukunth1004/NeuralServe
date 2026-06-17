from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "NeuralServe"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    MODEL_PATH: str = "./models/onnx"
    USE_ONNX: bool = False
    MAX_BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 512
    INFERENCE_TIMEOUT: float = 30.0

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/neuralserve"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    WORKER_COUNT: int = 4

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
