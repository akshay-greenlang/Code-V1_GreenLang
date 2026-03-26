# Deprecated: Use greenlang.monitoring.telemetry instead
from greenlang.monitoring.telemetry import *
from greenlang.monitoring.telemetry.metrics import track_execution as _track_execution
from greenlang.monitoring.telemetry.metrics import MetricsCollector as _MetricsCollector


def track_execution(
    pipeline: str | None = None,
    tenant_id: str = "default",
    metric_name: str | None = None,
):
    """Backward-compatible wrapper supporting legacy metric_name kwarg."""
    resolved_pipeline = pipeline or metric_name or "default"
    return _track_execution(pipeline=resolved_pipeline, tenant_id=tenant_id)


class MetricsCollector(_MetricsCollector):
    """Backward-compatible metrics collector with legacy namespace arg."""

    def __init__(self, namespace: str | None = None, *args, **kwargs):
        if namespace and "job_name" not in kwargs:
            kwargs["job_name"] = namespace
        super().__init__(*args, **kwargs)
