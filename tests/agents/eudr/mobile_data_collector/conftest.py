# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-015 Mobile Data Collector test suite.

Provides reusable fixtures for all 8 engine test modules including config
overrides, engine instances, factory functions, and assertion helpers.

Config Fixtures:
    mdc_config          - Default MobileDataCollectorConfig for tests
    strict_config       - Strict-mode config with tighter validation

Engine Fixtures (8):
    offline_form_engine       - OfflineFormEngine instance
    gps_capture_engine        - GPSCaptureEngine instance
    photo_evidence_collector  - PhotoEvidenceCollector instance
    sync_engine               - SyncEngine instance
    form_template_manager     - FormTemplateManager instance
    digital_signature_engine  - DigitalSignatureEngine instance
    data_package_builder      - DataPackageBuilder instance
    device_fleet_manager      - DeviceFleetManager instance

Factory Functions (10):
    make_form_submission      - Create FormSubmission test data
    make_gps_capture          - Create GPSCapture test data
    make_polygon_trace        - Create PolygonTrace test data
    make_photo_evidence       - Create PhotoEvidence test data
    make_sync_queue_item      - Create SyncQueueItem test data
    make_form_template        - Create FormTemplate test data
    make_digital_signature    - Create DigitalSignature test data
    make_data_package         - Create DataPackage test data
    make_device_registration  - Create DeviceRegistration test data
    make_batch_form_data      - Create batch form submission data

Assertion Helpers (5):
    assert_valid_uuid         - Validate UUID format
    assert_valid_sha256       - Validate SHA-256 hex string
    assert_valid_coordinates  - Validate lat/lon in range
    assert_provenance_recorded- Validate provenance tracking occurred
    assert_within_tolerance   - Validate numeric within relative tolerance

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH: int = 64

EUDR_FORM_TYPES: List[str] = [
    "producer_registration",
    "plot_survey",
    "harvest_log",
    "custody_transfer",
    "quality_inspection",
    "smallholder_declaration",
]

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

ACCURACY_TIERS: List[str] = [
    "excellent", "good", "acceptable", "poor", "rejected",
]

PHOTO_TYPES: List[str] = [
    "plot_photo", "commodity_photo", "document_photo",
    "facility_photo", "transport_photo", "identity_photo",
]

SYNC_STATUSES: List[str] = [
    "queued", "in_progress", "completed", "failed", "permanently_failed",
]

CONFLICT_STRATEGIES: List[str] = [
    "server_wins", "client_wins", "manual",
]

TEMPLATE_STATUSES: List[str] = [
    "draft", "review", "published", "deprecated",
]

SIGNER_ROLES: List[str] = [
    "producer", "collector", "cooperative_manager", "inspector",
    "transport_operator", "warehouse_manager", "buyer",
    "exporter", "importer", "auditor", "system",
]

PACKAGE_STATUSES: List[str] = [
    "building", "sealing", "sealed", "submitted", "accepted", "rejected",
]

DEVICE_PLATFORMS: List[str] = ["android", "ios", "web", "desktop"]

DEVICE_STATUSES: List[str] = [
    "registered", "active", "suspended", "decommissioned",
]

FIELD_TYPES: List[str] = [
    "text", "number", "decimal", "date", "datetime",
    "select", "multi_select", "checkbox", "photo",
    "gps_point", "gps_polygon", "signature", "barcode", "calculated",
]

# Sample GPS coordinates (Accra, Ghana - typical EUDR origin)
SAMPLE_LAT: float = 5.6037
SAMPLE_LON: float = -0.1870
SAMPLE_ACCURACY: float = 2.5
SAMPLE_HDOP: float = 1.5
SAMPLE_SATELLITES: int = 10

# Sample polygon (small plot near Accra)
SAMPLE_POLYGON_VERTICES: List[List[float]] = [
    [5.6030, -0.1860],
    [5.6040, -0.1860],
    [5.6040, -0.1870],
    [5.6030, -0.1870],
    [5.6030, -0.1860],  # closed polygon
]

# Sample device info
SAMPLE_DEVICE_MODEL: str = "Samsung Galaxy A15"
SAMPLE_PLATFORM: str = "android"
SAMPLE_OS_VERSION: str = "14.0"

# Retention per EUDR Article 14
EUDR_RETENTION_YEARS: int = 5


# ---------------------------------------------------------------------------
# Mock provenance tracker
# ---------------------------------------------------------------------------

class MockProvenanceTracker:
    """Mock provenance tracker for testing without actual chain hashing."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self._lock = __import__("threading").Lock()

    def record(
        self,
        entity_type: str = "",
        action: str = "",
        entity_id: str = "",
        data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "entity_type": entity_type,
            "action": action,
            "entity_id": entity_id,
            "data": data,
            "metadata": metadata or {},
            "hash_value": hashlib.sha256(
                json.dumps(
                    {"entity_type": entity_type, "action": action,
                     "entity_id": entity_id},
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        }
        with self._lock:
            self.entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        return True

    def get_entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.entries)

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()


# ---------------------------------------------------------------------------
# Mock metrics
# ---------------------------------------------------------------------------

class MockMetrics:
    """Mock metrics recorder for testing without Prometheus."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append({"args": args, "kwargs": kwargs})


# ---------------------------------------------------------------------------
# Config Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset config singleton before each test to ensure isolation."""
    try:
        from greenlang.agents.eudr.mobile_data_collector.config import reset_config
        reset_config()
    except (ImportError, Exception):
        pass
    yield
    try:
        from greenlang.agents.eudr.mobile_data_collector.config import reset_config
        reset_config()
    except (ImportError, Exception):
        pass


@pytest.fixture(autouse=True)
def _mock_provenance_and_metrics(monkeypatch):
    """Mock provenance and metrics globally so engines initialize cleanly."""
    mock_prov = MockProvenanceTracker()
    mock_metrics = MockMetrics()

    try:
        monkeypatch.setattr(
            "greenlang.agents.eudr.mobile_data_collector.provenance.get_provenance_tracker",
            lambda: mock_prov,
        )
    except Exception:
        pass

    # Mock all metric recording functions
    metric_funcs = [
        "record_form_submitted", "record_gps_capture",
        "record_photo_captured", "record_sync_completed",
        "record_sync_conflict", "record_signature_captured",
        "record_package_built", "record_api_error",
        "observe_form_submission_duration", "observe_gps_capture_duration",
        "observe_sync_duration", "observe_photo_upload_duration",
        "observe_package_build_duration",
        "set_pending_sync_items", "set_active_devices",
        "set_offline_devices", "set_storage_used_bytes",
        "set_pending_uploads",
    ]
    for func_name in metric_funcs:
        try:
            monkeypatch.setattr(
                f"greenlang.agents.eudr.mobile_data_collector.metrics.{func_name}",
                lambda *a, **kw: None,
            )
        except Exception:
            pass

    yield mock_prov


@pytest.fixture
def mdc_config(monkeypatch):
    """Create default MobileDataCollectorConfig for tests."""
    from greenlang.agents.eudr.mobile_data_collector.config import (
        MobileDataCollectorConfig,
        set_config,
    )
    config = MobileDataCollectorConfig(
        database_url="postgresql://test:test@localhost:5432/test_mdc",
        redis_url="redis://localhost:6379/0",
        log_level="DEBUG",
        pool_size=2,
        max_form_size_kb=512,
        local_storage_path="/tmp/test_mdc_storage",
        queue_batch_size=10,
        validation_strictness="strict",
        max_fields_per_form=100,
        form_submission_timeout_s=30,
        draft_expiry_days=7,
        min_accuracy_meters=3.0,
        default_crs="EPSG:4326",
        default_srid=4326,
        polygon_min_vertices=3,
        polygon_max_vertices=1000,
        enable_altitude_capture=True,
        hdop_threshold=2.0,
        satellite_count_threshold=6,
        coordinate_decimal_places=7,
        enable_bounds_checking=True,
        gps_capture_timeout_s=60,
        polygon_accuracy_meters=5.0,
        min_plot_area_ha=0.01,
        max_photo_size_mb=20,
        compression_quality_high=90,
        compression_quality_medium=75,
        compression_quality_low=50,
        enable_exif_extraction=True,
        photo_hash_algorithm="sha256",
        max_photos_per_form=20,
        min_photo_width=640,
        min_photo_height=480,
        min_photo_file_size_bytes=10240,
        timestamp_deviation_threshold_s=300,
        sync_interval_s=60,
        max_retry_count=20,
        retry_backoff_multiplier=2.0,
        enable_delta_compression=True,
        bandwidth_limit_kb_s=0,
        conflict_resolution_strategy="server_wins",
        max_upload_size_per_sync_mb=50,
        enable_idempotency=True,
        sync_timeout_s=300,
        max_templates=500,
        conditional_logic_depth=5,
        enable_template_versioning=True,
        enable_template_inheritance=True,
        enable_meta_validation=True,
        signature_algorithm="ecdsa_p256",
        enable_timestamp_binding=True,
        signature_expiry_days=365,
        revocation_window_hours=24,
        enable_multi_signature=True,
        enable_visual_signature=True,
        max_package_size_mb=500,
        package_compression_format="gzip",
        enable_merkle_tree=True,
        package_ttl_years=5,
        package_compression_level=6,
        enable_incremental_build=True,
        supported_export_formats=["zip", "tar_gz", "json_ld"],
        max_devices=5000,
        heartbeat_interval_s=300,
        offline_threshold_minutes=60,
        storage_warning_threshold_pct=85,
        low_battery_threshold_pct=20,
        enable_version_enforcement=True,
        enable_decommission=True,
        batch_size=100,
        batch_concurrency=4,
        batch_timeout_s=600,
        retention_years=5,
        provenance_genesis_hash="0" * 64,
        enable_metrics_export=True,
        supported_languages=["en", "fr", "pt", "es"],
    )
    set_config(config)
    return config


@pytest.fixture
def strict_config(monkeypatch):
    """Create strict-mode config with tighter validation thresholds."""
    from greenlang.agents.eudr.mobile_data_collector.config import (
        MobileDataCollectorConfig,
        set_config,
    )
    config = MobileDataCollectorConfig(
        database_url="postgresql://test:test@localhost:5432/test_mdc_strict",
        redis_url="redis://localhost:6379/1",
        log_level="WARNING",
        pool_size=1,
        max_form_size_kb=256,
        local_storage_path="/tmp/test_mdc_strict",
        queue_batch_size=5,
        validation_strictness="strict",
        max_fields_per_form=50,
        form_submission_timeout_s=15,
        draft_expiry_days=3,
        min_accuracy_meters=1.0,
        default_crs="EPSG:4326",
        default_srid=4326,
        polygon_min_vertices=4,
        polygon_max_vertices=500,
        enable_altitude_capture=True,
        hdop_threshold=1.0,
        satellite_count_threshold=8,
        coordinate_decimal_places=7,
        enable_bounds_checking=True,
        gps_capture_timeout_s=30,
        polygon_accuracy_meters=2.0,
        min_plot_area_ha=0.1,
        max_photo_size_mb=10,
        compression_quality_high=95,
        compression_quality_medium=80,
        compression_quality_low=60,
        enable_exif_extraction=True,
        photo_hash_algorithm="sha256",
        max_photos_per_form=10,
        min_photo_width=1024,
        min_photo_height=768,
        min_photo_file_size_bytes=50000,
        timestamp_deviation_threshold_s=60,
        sync_interval_s=30,
        max_retry_count=5,
        retry_backoff_multiplier=3.0,
        enable_delta_compression=True,
        bandwidth_limit_kb_s=100,
        conflict_resolution_strategy="server_wins",
        max_upload_size_per_sync_mb=10,
        enable_idempotency=True,
        sync_timeout_s=120,
        max_templates=100,
        conditional_logic_depth=3,
        enable_template_versioning=True,
        enable_template_inheritance=False,
        enable_meta_validation=True,
        signature_algorithm="ecdsa_p256",
        enable_timestamp_binding=True,
        signature_expiry_days=90,
        revocation_window_hours=12,
        enable_multi_signature=True,
        enable_visual_signature=False,
        max_package_size_mb=50,
        package_compression_format="gzip",
        enable_merkle_tree=True,
        package_ttl_years=5,
        package_compression_level=9,
        enable_incremental_build=True,
        supported_export_formats=["zip", "tar_gz", "json_ld"],
        max_devices=100,
        heartbeat_interval_s=60,
        offline_threshold_minutes=15,
        storage_warning_threshold_pct=70,
        low_battery_threshold_pct=30,
        enable_version_enforcement=True,
        enable_decommission=True,
        batch_size=50,
        batch_concurrency=2,
        batch_timeout_s=300,
        retention_years=5,
        provenance_genesis_hash="0" * 64,
        enable_metrics_export=False,
        supported_languages=["en", "fr"],
    )
    set_config(config)
    return config


# ---------------------------------------------------------------------------
# Engine Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def offline_form_engine(mdc_config):
    """Create OfflineFormEngine instance for testing."""
    from greenlang.agents.eudr.mobile_data_collector.offline_form_engine import (
        OfflineFormEngine,
    )
    return OfflineFormEngine()


@pytest.fixture
def gps_capture_engine(mdc_config):
    """Create GPSCaptureEngine instance for testing."""
    from greenlang.agents.eudr.mobile_data_collector.gps_capture_engine import (
        GPSCaptureEngine,
    )
    return GPSCaptureEngine()


@pytest.fixture
def photo_evidence_collector(mdc_config):
    """Create PhotoEvidenceCollector instance for testing."""
    from greenlang.agents.eudr.mobile_data_collector.photo_evidence_collector import (
        PhotoEvidenceCollector,
    )
    return PhotoEvidenceCollector()


@pytest.fixture
def sync_engine(mdc_config):
    """Create SyncEngine instance for testing."""
    from greenlang.agents.eudr.mobile_data_collector.sync_engine import (
        SyncEngine,
    )
    return SyncEngine()


@pytest.fixture
def form_template_manager(mdc_config):
    """Create FormTemplateManager instance for testing."""
    from greenlang.agents.eudr.mobile_data_collector.form_template_manager import (
        FormTemplateManager,
    )
    return FormTemplateManager()


@pytest.fixture
def digital_signature_engine(mdc_config):
    """Create DigitalSignatureEngine instance for testing."""
    from greenlang.agents.eudr.mobile_data_collector.digital_signature_engine import (
        DigitalSignatureEngine,
    )
    return DigitalSignatureEngine()


@pytest.fixture
def data_package_builder(mdc_config):
    """Create DataPackageBuilder instance for testing."""
    from greenlang.agents.eudr.mobile_data_collector.data_package_builder import (
        DataPackageBuilder,
    )
    return DataPackageBuilder()


@pytest.fixture
def device_fleet_manager(mdc_config):
    """Create DeviceFleetManager instance for testing."""
    from greenlang.agents.eudr.mobile_data_collector.device_fleet_manager import (
        DeviceFleetManager,
    )
    return DeviceFleetManager()


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

@pytest.fixture
def make_form_submission():
    """Factory for creating form submission test data."""

    def _factory(
        form_type: str = "harvest_log",
        commodity: str = "coffee",
        device_id: str = "dev-001",
        operator_id: str = "op-001",
        status: str = "draft",
        data: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "device_id": device_id,
            "operator_id": operator_id,
            "form_type": form_type,
            "commodity_type": commodity,
            "data": data or {
                "producer_id": "PROD-001",
                "plot_id": "PLOT-001",
                "commodity": commodity,
                "harvest_date": "2026-01-15",
                "quantity_kg": 250.5,
                "harvest_gps": {
                    "latitude": SAMPLE_LAT,
                    "longitude": SAMPLE_LON,
                },
            },
        }
        base.update(overrides)
        return base

    return _factory


@pytest.fixture
def make_gps_capture():
    """Factory for creating GPS capture test data."""

    def _factory(
        latitude: float = SAMPLE_LAT,
        longitude: float = SAMPLE_LON,
        accuracy: float = SAMPLE_ACCURACY,
        hdop: float = SAMPLE_HDOP,
        satellites: int = SAMPLE_SATELLITES,
        **overrides: Any,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "device_id": "dev-001",
            "operator_id": "op-001",
            "latitude": latitude,
            "longitude": longitude,
            "horizontal_accuracy_m": accuracy,
            "hdop": hdop,
            "satellite_count": satellites,
            "altitude_m": 78.5,
        }
        base.update(overrides)
        return base

    return _factory


@pytest.fixture
def make_polygon_trace():
    """Factory for creating polygon trace test data."""

    def _factory(
        vertices: Optional[List[List[float]]] = None,
        device_id: str = "dev-001",
        **overrides: Any,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "device_id": device_id,
            "operator_id": "op-001",
            "vertices": vertices or copy.deepcopy(SAMPLE_POLYGON_VERTICES),
        }
        base.update(overrides)
        return base

    return _factory


@pytest.fixture
def make_photo_evidence():
    """Factory for creating photo evidence test data."""

    def _factory(
        photo_type: str = "plot_photo",
        file_size_bytes: int = 1_500_000,
        device_id: str = "dev-001",
        **overrides: Any,
    ) -> Dict[str, Any]:
        photo_bytes = b"SIMULATED_JPEG_DATA_" + uuid.uuid4().bytes
        integrity_hash = hashlib.sha256(photo_bytes).hexdigest()
        base: Dict[str, Any] = {
            "device_id": device_id,
            "operator_id": "op-001",
            "photo_type": photo_type,
            "file_size_bytes": file_size_bytes,
            "width": 1920,
            "height": 1080,
            "format": "jpeg",
            "integrity_hash": integrity_hash,
            "latitude": SAMPLE_LAT,
            "longitude": SAMPLE_LON,
        }
        base.update(overrides)
        return base

    return _factory


@pytest.fixture
def make_sync_queue_item():
    """Factory for creating sync queue item test data."""

    def _factory(
        item_type: str = "form",
        priority: int = 5,
        device_id: str = "dev-001",
        **overrides: Any,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "device_id": device_id,
            "item_type": item_type,
            "item_id": str(uuid.uuid4()),
            "priority": priority,
            "data": {"key": "value"},
            "size_bytes": 2048,
        }
        base.update(overrides)
        return base

    return _factory


@pytest.fixture
def make_form_template():
    """Factory for creating form template test data."""

    def _factory(
        name: str = "Test Harvest Log",
        form_type: str = "harvest_log",
        fields: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        default_fields = fields or [
            {"field_id": "producer_id", "type": "text",
             "label": "Producer ID", "required": True, "max_length": 50},
            {"field_id": "quantity_kg", "type": "decimal",
             "label": "Quantity (kg)", "required": True, "min_value": 0.0},
            {"field_id": "harvest_date", "type": "date",
             "label": "Harvest Date", "required": True},
            {"field_id": "commodity", "type": "select",
             "label": "Commodity", "required": True,
             "options": EUDR_COMMODITIES},
        ]
        base: Dict[str, Any] = {
            "name": name,
            "form_type": form_type,
            "fields": default_fields,
        }
        base.update(overrides)
        return base

    return _factory


@pytest.fixture
def make_digital_signature():
    """Factory for creating digital signature test data."""

    def _factory(
        signer_id: str = "signer-001",
        form_id: str = "form-001",
        data_hash: Optional[str] = None,
        device_id: str = "dev-001",
        **overrides: Any,
    ) -> Dict[str, Any]:
        if data_hash is None:
            data_hash = hashlib.sha256(b"test_data").hexdigest()
        base: Dict[str, Any] = {
            "form_id": form_id,
            "signer_id": signer_id,
            "data_hash": data_hash,
            "device_id": device_id,
        }
        base.update(overrides)
        return base

    return _factory


@pytest.fixture
def make_data_package():
    """Factory for creating data package test data."""

    def _factory(
        device_id: str = "dev-001",
        operator_id: str = "op-001",
        export_format: str = "zip",
        **overrides: Any,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "device_id": device_id,
            "operator_id": operator_id,
            "export_format": export_format,
        }
        base.update(overrides)
        return base

    return _factory


@pytest.fixture
def make_device_registration():
    """Factory for creating device registration test data."""

    def _factory(
        device_model: str = SAMPLE_DEVICE_MODEL,
        platform: str = SAMPLE_PLATFORM,
        os_version: str = SAMPLE_OS_VERSION,
        **overrides: Any,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "device_model": device_model,
            "platform": platform,
            "os_version": os_version,
            "agent_version": "1.0.0",
        }
        base.update(overrides)
        return base

    return _factory


@pytest.fixture
def make_batch_form_data():
    """Factory for creating batch form submission data."""

    def _factory(
        count: int = 10,
        form_type: str = "harvest_log",
        commodity: str = "coffee",
    ) -> List[Dict[str, Any]]:
        batch = []
        for i in range(count):
            batch.append({
                "device_id": f"dev-{i:03d}",
                "operator_id": f"op-{i:03d}",
                "form_type": form_type,
                "commodity_type": commodity,
                "data": {
                    "producer_id": f"PROD-{i:03d}",
                    "plot_id": f"PLOT-{i:03d}",
                    "commodity": commodity,
                    "harvest_date": f"2026-01-{(i % 28) + 1:02d}",
                    "quantity_kg": 100.0 + i * 10,
                },
            })
        return batch

    return _factory


# ---------------------------------------------------------------------------
# Assertion Helpers
# ---------------------------------------------------------------------------

def assert_valid_uuid(value: str, msg: str = "") -> None:
    """Assert that value is a valid UUID-formatted string.

    Args:
        value: String to validate.
        msg: Optional assertion message.
    """
    try:
        uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        # Also handle sig-{hash} style IDs
        if not re.match(r"^sig-[a-f0-9]{32}$", str(value)):
            pytest.fail(f"Not a valid UUID or signature ID: {value!r} {msg}")


def assert_valid_sha256(value: str, msg: str = "") -> None:
    """Assert that value is a valid 64-character SHA-256 hex string.

    Args:
        value: String to validate.
        msg: Optional assertion message.
    """
    assert isinstance(value, str), f"Expected str, got {type(value).__name__} {msg}"
    assert len(value) == SHA256_HEX_LENGTH, (
        f"SHA-256 hex must be {SHA256_HEX_LENGTH} chars, got {len(value)} {msg}"
    )
    assert re.match(r"^[a-f0-9]{64}$", value), (
        f"Invalid SHA-256 hex: {value!r} {msg}"
    )


def assert_valid_coordinates(
    latitude: float,
    longitude: float,
    msg: str = "",
) -> None:
    """Assert that latitude and longitude are within valid ranges.

    Args:
        latitude: Latitude value (-90 to 90).
        longitude: Longitude value (-180 to 180).
        msg: Optional assertion message.
    """
    assert -90 <= latitude <= 90, (
        f"Latitude {latitude} out of range [-90, 90] {msg}"
    )
    assert -180 <= longitude <= 180, (
        f"Longitude {longitude} out of range [-180, 180] {msg}"
    )


def assert_provenance_recorded(
    provenance_tracker: MockProvenanceTracker,
    entity_type: str = "",
    action: str = "",
    msg: str = "",
) -> None:
    """Assert that a provenance entry was recorded.

    Args:
        provenance_tracker: Mock provenance tracker instance.
        entity_type: Expected entity type (optional filter).
        action: Expected action (optional filter).
        msg: Optional assertion message.
    """
    entries = provenance_tracker.get_entries()
    assert len(entries) > 0, f"No provenance entries recorded {msg}"

    if entity_type or action:
        matches = [
            e for e in entries
            if (not entity_type or e["entity_type"] == entity_type)
            and (not action or e["action"] == action)
        ]
        assert len(matches) > 0, (
            f"No provenance entry found with entity_type={entity_type!r}, "
            f"action={action!r}. Available: "
            f"{[(e['entity_type'], e['action']) for e in entries]} {msg}"
        )


def assert_within_tolerance(
    actual: float,
    expected: float,
    rel_tolerance: float = 1e-6,
    msg: str = "",
) -> None:
    """Assert that actual is within relative tolerance of expected.

    Args:
        actual: Actual value.
        expected: Expected value.
        rel_tolerance: Relative tolerance (default 1e-6).
        msg: Optional assertion message.
    """
    if expected == 0:
        assert abs(actual) < rel_tolerance, (
            f"Expected ~0, got {actual} (tol={rel_tolerance}) {msg}"
        )
    else:
        ratio = abs((actual - expected) / expected)
        assert ratio < rel_tolerance, (
            f"Expected ~{expected}, got {actual} "
            f"(ratio={ratio:.2e}, tol={rel_tolerance}) {msg}"
        )
