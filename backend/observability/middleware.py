import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from observability.logger import current_trace_id, current_request_id, get_custom_logger
from observability.metrics import metrics

logger = get_custom_logger(name=__name__)

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Intercepts incoming HTTP requests to bind Trace ID & Request ID and collect performance metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract or generate Request and Trace IDs
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))

        # Set ContextVars for downstream logging and tracing
        token_req = current_request_id.set(req_id)
        token_trace = current_trace_id.set(trace_id)

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Record telemetry
            metrics.record_http_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms
            )

            # Attach headers to response
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Trace-ID"] = trace_id

            logger.info(
                f"HTTP {request.method} {request.url.path} - {response.status_code} ({duration_ms:.2f}ms)"
            )
            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_http_request(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=duration_ms
            )
            logger.error(
                f"HTTP {request.method} {request.url.path} - FAILED ({duration_ms:.2f}ms) with error: {exc}",
                exc_info=True
            )
            raise exc

        finally:
            current_request_id.reset(token_req)
            current_trace_id.reset(token_trace)
