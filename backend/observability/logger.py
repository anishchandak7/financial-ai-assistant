import os
import sys
import json
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from typing import Optional, Dict, Any

# Thread-safe / Async-safe ContextVars for Distributed Tracing & Request Correlation
current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)
current_request_id: ContextVar[Optional[str]] = ContextVar("current_request_id", default=None)
current_agent_name: ContextVar[Optional[str]] = ContextVar("current_agent_name", default=None)

BACKEND_DIR = Path(__file__).resolve().parent.parent

class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON for production log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "location": f"{record.filename}:{record.lineno}",
        }

        # Inject context correlation IDs if present
        trace_id = current_trace_id.get()
        if trace_id:
            log_obj["trace_id"] = trace_id

        req_id = current_request_id.get()
        if req_id:
            log_obj["request_id"] = req_id

        agent = current_agent_name.get()
        if agent:
            log_obj["agent"] = agent

        # Inject any extra attributes passed via logger.info(..., extra={...})
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_obj.update(record.extra_fields)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class ContextAwareConsoleFormatter(logging.Formatter):
    """Human-readable console formatter that includes trace/request ID when available."""

    def format(self, record: logging.LogRecord) -> str:
        trace_id = current_trace_id.get()
        req_id = current_request_id.get()
        agent = current_agent_name.get()

        prefix_parts = []
        if trace_id:
            prefix_parts.append(f"trace:{trace_id[:8]}")
        if req_id:
            prefix_parts.append(f"req:{req_id[:8]}")
        if agent:
            prefix_parts.append(f"agent:{agent}")

        context_tag = f"[{' '.join(prefix_parts)}] " if prefix_parts else ""
        record.msg = f"{context_tag}{record.msg}"
        return super().format(record)


def get_custom_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    """Creates a custom logger with console and rotating file handlers."""
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # 1. Console Handler (Streams to Stdout)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ContextAwareConsoleFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # 2. Rotating File Handler (Saves structured logs to disk)
    effective_log_path = Path(log_file)
    if not effective_log_path.is_absolute():
        effective_log_path = BACKEND_DIR / effective_log_path

    file_handler = RotatingFileHandler(
        filename=str(effective_log_path),
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
