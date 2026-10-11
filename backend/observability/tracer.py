import time
import uuid
from contextlib import contextmanager
from typing import Optional, Dict, Any, Generator
from observability.logger import current_trace_id, get_custom_logger

logger = get_custom_logger(name=__name__)

class Span:
    """Represents a discrete unit of work within a trace."""

    def __init__(self, name: str, trace_id: str, span_id: str, parent_id: Optional[str] = None):
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def set_error(self, err: Exception | str):
        self.error = str(err)

    def finish(self):
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        status = "ERROR" if self.error else "OK"
        logger.debug(
            f"Span [{self.name}] finished in {self.duration_ms}ms ({status}) | attributes={self.attributes}"
        )


@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Generator[Span, None, None]:
    """Context manager to create and finish an execution span."""
    trace_id = current_trace_id.get() or str(uuid.uuid4())
    span_id = str(uuid.uuid4())[:16]
    span = Span(name=name, trace_id=trace_id, span_id=span_id)

    if attributes:
        for k, v in attributes.items():
            span.set_attribute(k, v)

    try:
        yield span
    except Exception as exc:
        span.set_error(exc)
        raise exc
    finally:
        span.finish()
