# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-015: Mobile Data Collector

18 Prometheus metrics for mobile data collector agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_mdc_`` prefix (GreenLang EUDR Mobile
Data Collector) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules across the GreenLang platform.

Metrics (18 per PRD Section 7.3):
    Counters (8):
        1.  gl_eudr_mdc_forms_submitted_total            - Form submissions received
        2.  gl_eudr_mdc_gps_captures_total               - GPS captures received
        3.  gl_eudr_mdc_photos_captured_total             - Photos received from devices
        4.  gl_eudr_mdc_syncs_completed_total             - Successful sync sessions
        5.  gl_eudr_mdc_sync_conflicts_total              - Sync conflicts detected
        6.  gl_eudr_mdc_signatures_captured_total         - Digital signatures captured
        7.  gl_eudr_mdc_packages_built_total              - Data packages assembled
        8.  gl_eudr_mdc_api_errors_total                  - API errors by operation

    Histograms (5):
        9.  gl_eudr_mdc_form_submission_duration_seconds  - Form submission latency
        10. gl_eudr_mdc_gps_capture_duration_seconds      - GPS capture latency
        11. gl_eudr_mdc_sync_duration_seconds             - Full sync session duration
        12. gl_eudr_mdc_photo_upload_duration_seconds     - Photo upload latency
        13. gl_eudr_mdc_package_build_duration_seconds    - Package build duration

    Gauges (5):
        14. gl_eudr_mdc_pending_sync_items                - Items pending sync
        15. gl_eudr_mdc_active_devices                    - Active devices (synced recently)
        16. gl_eudr_mdc_offline_devices                   - Offline devices
        17. gl_eudr_mdc_storage_used_bytes                - Storage consumed by photos/packages
        18. gl_eudr_mdc_pending_uploads                   - Pending upload items

Label Values Reference:
    form_type:
        producer_registration, plot_survey, harvest_log,
        custody_transfer, quality_inspection, smallholder_declaration.
    commodity:
        cattle, cocoa, coffee, oil_palm, rubber, soya, wood.
    accuracy_tier:
        excellent, good, acceptable, poor, rejected.
    photo_type:
        plot_photo, commodity_photo, document_photo,
        facility_photo, transport_photo, identity_photo.
    sync_status:
        queued, in_progress, completed, failed, permanently_failed.
    device_platform:
        android, ios, harmonyos.
    operation:
        submit, capture, upload, sync, resolve, build, register,
        deregister, sign, verify, search, validate.

Example:
    >>> from greenlang.agents.eudr.mobile_data_collector.metrics import (
    ...     record_form_submitted,
    ...     record_gps_capture,
    ...     record_photo_captured,
    ...     observe_form_submission_duration,
    ...     set_active_devices,
    ... )
    >>> record_form_submitted("harvest_log", "coffee")
    >>> record_gps_capture("good")
    >>> record_photo_captured("plot_photo")
    >>> observe_form_submission_duration(0.15)
    >>> set_active_devices(250)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; "
        "mobile data collector metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
        """Create a Counter or retrieve existing one to avoid registry collisions."""
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Counter(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

    def _safe_histogram(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
        buckets: tuple = (),
    ):  # type: ignore[return]
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [], **kw,
            )
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(), **kw,
            )

    def _safe_gauge(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
        """Create a Gauge or retrieve existing one."""
        try:
            return Gauge(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Gauge(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

# ---------------------------------------------------------------------------
# Metric definitions (18 metrics per PRD Section 7.3)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # -- Counters (8) --------------------------------------------------------

    # 1. Form submissions received by form type and commodity
    mdc_forms_submitted_total = _safe_counter(
        "gl_eudr_mdc_forms_submitted_total",
        "Total form submissions received from devices",
        labelnames=["form_type", "commodity"],
    )

    # 2. GPS captures received by accuracy tier
    mdc_gps_captures_total = _safe_counter(
        "gl_eudr_mdc_gps_captures_total",
        "Total GPS captures received (point + polygon vertices)",
        labelnames=["accuracy_tier"],
    )

    # 3. Photos received by photo type
    mdc_photos_captured_total = _safe_counter(
        "gl_eudr_mdc_photos_captured_total",
        "Total photos received from devices",
        labelnames=["photo_type"],
    )

    # 4. Successful sync sessions completed
    mdc_syncs_completed_total = _safe_counter(
        "gl_eudr_mdc_syncs_completed_total",
        "Total successful sync sessions completed",
    )

    # 5. Sync conflicts detected
    mdc_sync_conflicts_total = _safe_counter(
        "gl_eudr_mdc_sync_conflicts_total",
        "Total sync conflicts detected",
    )

    # 6. Digital signatures captured
    mdc_signatures_captured_total = _safe_counter(
        "gl_eudr_mdc_signatures_captured_total",
        "Total digital signatures captured",
    )

    # 7. Data packages assembled and sealed
    mdc_packages_built_total = _safe_counter(
        "gl_eudr_mdc_packages_built_total",
        "Total data packages assembled and sealed",
    )

    # 8. API errors by operation
    mdc_api_errors_total = _safe_counter(
        "gl_eudr_mdc_api_errors_total",
        "Total API errors across all endpoints",
        labelnames=["operation"],
    )

    # -- Histograms (5) ------------------------------------------------------

    # 9. Form submission processing latency
    mdc_form_submission_duration_seconds = _safe_histogram(
        "gl_eudr_mdc_form_submission_duration_seconds",
        "Form submission processing latency (server-side)",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.2,
            0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 10. GPS capture processing latency
    mdc_gps_capture_duration_seconds = _safe_histogram(
        "gl_eudr_mdc_gps_capture_duration_seconds",
        "GPS capture processing latency (server-side)",
        buckets=(
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5, 5.0,
        ),
    )

    # 11. Full sync session duration
    mdc_sync_duration_seconds = _safe_histogram(
        "gl_eudr_mdc_sync_duration_seconds",
        "Full sync session duration",
        buckets=(
            0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
            60.0, 120.0, 300.0, 600.0,
        ),
    )

    # 12. Photo upload processing latency
    mdc_photo_upload_duration_seconds = _safe_histogram(
        "gl_eudr_mdc_photo_upload_duration_seconds",
        "Photo upload processing latency",
        buckets=(
            0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.5, 5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 13. Data package build duration
    mdc_package_build_duration_seconds = _safe_histogram(
        "gl_eudr_mdc_package_build_duration_seconds",
        "Data package build duration",
        buckets=(
            0.1, 0.5, 1.0, 2.5, 5.0, 10.0,
            30.0, 60.0, 120.0, 300.0,
        ),
    )

    # -- Gauges (5) ----------------------------------------------------------

    # 14. Current count of items pending sync
    mdc_pending_sync_items = _safe_gauge(
        "gl_eudr_mdc_pending_sync_items",
        "Current count of items pending synchronization across all devices",
    )

    # 15. Active devices (synced within threshold)
    mdc_active_devices = _safe_gauge(
        "gl_eudr_mdc_active_devices",
        "Devices that have synced within the last 48 hours",
    )

    # 16. Offline devices (not synced within threshold)
    mdc_offline_devices = _safe_gauge(
        "gl_eudr_mdc_offline_devices",
        "Devices that have not synced within 48 hours",
    )

    # 17. Total storage consumed by photos and data packages
    mdc_storage_used_bytes = _safe_gauge(
        "gl_eudr_mdc_storage_used_bytes",
        "Total storage consumed by photos and data packages",
    )

    # 18. Total pending upload items across all device sync queues
    mdc_pending_uploads = _safe_gauge(
        "gl_eudr_mdc_pending_uploads",
        "Total pending upload items across all device sync queues",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    mdc_forms_submitted_total = None              # type: ignore[assignment]
    mdc_gps_captures_total = None                 # type: ignore[assignment]
    mdc_photos_captured_total = None              # type: ignore[assignment]
    mdc_syncs_completed_total = None              # type: ignore[assignment]
    mdc_sync_conflicts_total = None               # type: ignore[assignment]
    mdc_signatures_captured_total = None          # type: ignore[assignment]
    mdc_packages_built_total = None               # type: ignore[assignment]
    mdc_api_errors_total = None                   # type: ignore[assignment]
    mdc_form_submission_duration_seconds = None    # type: ignore[assignment]
    mdc_gps_capture_duration_seconds = None       # type: ignore[assignment]
    mdc_sync_duration_seconds = None              # type: ignore[assignment]
    mdc_photo_upload_duration_seconds = None       # type: ignore[assignment]
    mdc_package_build_duration_seconds = None      # type: ignore[assignment]
    mdc_pending_sync_items = None                 # type: ignore[assignment]
    mdc_active_devices = None                     # type: ignore[assignment]
    mdc_offline_devices = None                    # type: ignore[assignment]
    mdc_storage_used_bytes = None                 # type: ignore[assignment]
    mdc_pending_uploads = None                    # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_form_submitted(form_type: str, commodity: str) -> None:
    """Record a form submission event by form type and commodity.

    Args:
        form_type: EUDR form type (producer_registration, plot_survey,
            harvest_log, custody_transfer, quality_inspection,
            smallholder_declaration).
        commodity: EUDR commodity type (cattle, cocoa, coffee,
            oil_palm, rubber, soya, wood).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_forms_submitted_total.labels(
        form_type=form_type, commodity=commodity,
    ).inc()


def record_gps_capture(accuracy_tier: str) -> None:
    """Record a GPS capture event by accuracy tier.

    Args:
        accuracy_tier: Accuracy classification (excellent, good,
            acceptable, poor, rejected).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_gps_captures_total.labels(
        accuracy_tier=accuracy_tier,
    ).inc()


def record_photo_captured(photo_type: str) -> None:
    """Record a photo capture event by photo type.

    Args:
        photo_type: Photo category (plot_photo, commodity_photo,
            document_photo, facility_photo, transport_photo,
            identity_photo).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_photos_captured_total.labels(
        photo_type=photo_type,
    ).inc()


def record_sync_completed() -> None:
    """Record a successful sync session completion event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_syncs_completed_total.inc()


def record_sync_conflict() -> None:
    """Record a sync conflict detection event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_sync_conflicts_total.inc()


def record_signature_captured() -> None:
    """Record a digital signature capture event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_signatures_captured_total.inc()


def record_package_built() -> None:
    """Record a data package build/seal event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_packages_built_total.inc()


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (submit, capture,
            upload, sync, resolve, build, register, deregister,
            sign, verify, search, validate).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_api_errors_total.labels(operation=operation).inc()


def observe_form_submission_duration(seconds: float) -> None:
    """Record the duration of a form submission operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_form_submission_duration_seconds.observe(seconds)


def observe_gps_capture_duration(seconds: float) -> None:
    """Record the duration of a GPS capture processing operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_gps_capture_duration_seconds.observe(seconds)


def observe_sync_duration(seconds: float) -> None:
    """Record the duration of a full sync session.

    Args:
        seconds: Session wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_sync_duration_seconds.observe(seconds)


def observe_photo_upload_duration(seconds: float) -> None:
    """Record the duration of a photo upload operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_photo_upload_duration_seconds.observe(seconds)


def observe_package_build_duration(seconds: float) -> None:
    """Record the duration of a data package build operation.

    Args:
        seconds: Build wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_package_build_duration_seconds.observe(seconds)


def set_pending_sync_items(count: int) -> None:
    """Set the gauge for items pending synchronization.

    Args:
        count: Number of pending sync items. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_pending_sync_items.set(count)


def set_active_devices(count: int) -> None:
    """Set the gauge for active devices in the fleet.

    Args:
        count: Number of active devices. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_active_devices.set(count)


def set_offline_devices(count: int) -> None:
    """Set the gauge for offline devices in the fleet.

    Args:
        count: Number of offline devices. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_offline_devices.set(count)


def set_storage_used_bytes(byte_count: int) -> None:
    """Set the gauge for total storage consumed.

    Args:
        byte_count: Storage used in bytes. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_storage_used_bytes.set(byte_count)


def set_pending_uploads(count: int) -> None:
    """Set the gauge for total pending upload items.

    Args:
        count: Number of pending uploads. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mdc_pending_uploads.set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "mdc_forms_submitted_total",
    "mdc_gps_captures_total",
    "mdc_photos_captured_total",
    "mdc_syncs_completed_total",
    "mdc_sync_conflicts_total",
    "mdc_signatures_captured_total",
    "mdc_packages_built_total",
    "mdc_api_errors_total",
    "mdc_form_submission_duration_seconds",
    "mdc_gps_capture_duration_seconds",
    "mdc_sync_duration_seconds",
    "mdc_photo_upload_duration_seconds",
    "mdc_package_build_duration_seconds",
    "mdc_pending_sync_items",
    "mdc_active_devices",
    "mdc_offline_devices",
    "mdc_storage_used_bytes",
    "mdc_pending_uploads",
    # Helper functions
    "record_form_submitted",
    "record_gps_capture",
    "record_photo_captured",
    "record_sync_completed",
    "record_sync_conflict",
    "record_signature_captured",
    "record_package_built",
    "record_api_error",
    "observe_form_submission_duration",
    "observe_gps_capture_duration",
    "observe_sync_duration",
    "observe_photo_upload_duration",
    "observe_package_build_duration",
    "set_pending_sync_items",
    "set_active_devices",
    "set_offline_devices",
    "set_storage_used_bytes",
    "set_pending_uploads",
]
