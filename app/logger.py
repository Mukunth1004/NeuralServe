import logging
import json
import sys
from datetime import datetime, timezone
from app.config import get_settings

settings = get_settings()


class StructuredJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if settings.APP_ENV == "production":
            handler.setFormatter(StructuredJSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger
