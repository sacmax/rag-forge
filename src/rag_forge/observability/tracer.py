from contextlib import contextmanager
from typing import Any

class PipelineTracer:
    def __init__(self, enable: bool = False, endpoint: str = "http://localhost:6006"):
        self._enable = enable
        self._tracer = None
        if self._enable:
            from opentelemetry import trace
            from phoenix.otel import register
            self._tracer = trace.get_tracer("rag-forge")

    @contextmanager
    def span(self, name: str, **attributes: Any):
        """Create a traced span, No-op if tracing is disabled"""
        if not self._enable or self._tracer is None:
            yield
            return
        with self._tracer.start_as_current_span(name) as span:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
            yield span