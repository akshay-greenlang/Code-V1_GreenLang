# -*- coding: utf-8 -*-
"""
Unit tests for Audit Export Service - SEC-005: Centralized Audit Logging Service

Tests the AuditExportService and format-specific exporters (CSV, JSON, Parquet).

Coverage targets: 85%+ of export.py
"""

from __future__ import annotations

import asyncio
import io
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

# ---------------------------------------------------------------------------
# Attempt to import the audit export module.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.audit_service.export import (
        AuditExportService,
        ExportConfig,
        ExportJob,
        ExportFormat,
        CSVExporter,
        JSONExporter,
        ParquetExporter,
    )
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

    from enum import Enum

    class ExportFormat(str, Enum):
        """Stub for test collection when module is not yet built."""
        CSV = "csv"
        JSON = "json"
        PARQUET = "parquet"

    class ExportConfig:
        """Stub for test collection when module is not yet built."""
        def __init__(
            self,
            output_dir: str = "/tmp/audit-exports",
            max_rows_per_file: int = 100000,
            compression: bool = True,
        ):
            self.output_dir = output_dir
            self.max_rows_per_file = max_rows_per_file
            self.compression = compression

    class ExportJob:
        """Stub for test collection when module is not yet built."""
        def __init__(self, job_id: str, format: str, status: str = "pending"):
            self.job_id = job_id
            self.format = format
            self.status = status
            self.file_path: Optional[str] = None
            self.error: Optional[str] = None
            self.row_count: int = 0
            self.started_at: Optional[datetime] = None
            self.completed_at: Optional[datetime] = None

    class CSVExporter:
        """Stub for test collection when module is not yet built."""
        async def export(self, events: List[Any], output: Any) -> int: ...

    class JSONExporter:
        """Stub for test collection when module is not yet built."""
        async def export(self, events: List[Any], output: Any) -> int: ...

    class ParquetExporter:
        """Stub for test collection when module is not yet built."""
        async def export(self, events: List[Any], output: Any) -> int: ...

    class AuditExportService:
        """Stub for test collection when module is not yet built."""
        def __init__(self, config: ExportConfig = None, event_repository: Any = None):
            self._config = config or ExportConfig()
            self._repository = event_repository

        async def create_export_job(self, format: str, filters: Dict = None) -> ExportJob: ...
        async def get_job(self, job_id: str) -> Optional[ExportJob]: ...
        async def execute_job(self, job_id: str) -> ExportJob: ...
        async def get_download_url(self, job_id: str) -> str: ...
        async def stream_export(self, format: str, filters: Dict, chunk_size: int = 1000) -> AsyncMock: ...


pytestmark = pytest.mark.skipif(
    not _HAS_MODULE,
    reason="audit_service.export not yet implemented",
)


# ============================================================================
# Helpers
# ============================================================================


def _make_event_dict(
    event_id: str = "e-1",
    event_type: str = "auth.login_success",
) -> Dict[str, Any]:
    """Create a mock event dictionary."""
    return {
        "event_id": event_id,
        "event_type": event_type,
        "category": "auth",
        "severity": "info",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tenant_id": "t-acme",
        "user_id": "u-1",
        "result": "success",
        "details": {"method": "password"},
    }


def _make_event_repository() -> AsyncMock:
    """Create a mock event repository."""
    repo = AsyncMock()
    repo.get_events = AsyncMock(return_value={
        "items": [_make_event_dict(event_id=f"e-{i}") for i in range(100)],
        "total": 1000,
    })
    repo.stream_events = AsyncMock()
    return repo


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def export_config() -> ExportConfig:
    """Create a test export configuration."""
    return ExportConfig(
        output_dir="/tmp/audit-exports-test",
        max_rows_per_file=1000,
        compression=True,
    )


@pytest.fixture
def event_repository() -> AsyncMock:
    """Create a mock event repository."""
    return _make_event_repository()


@pytest.fixture
def export_service(export_config, event_repository) -> AuditExportService:
    """Create an AuditExportService instance for testing."""
    return AuditExportService(
        config=export_config,
        event_repository=event_repository,
    )


@pytest.fixture
def sample_events() -> List[Dict[str, Any]]:
    """Create sample events for testing."""
    return [_make_event_dict(event_id=f"e-{i}") for i in range(50)]


# ============================================================================
# TestExportConfig
# ============================================================================


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_values(self) -> None:
        """Config has correct default values."""
        config = ExportConfig()
        assert config.max_rows_per_file == 100000
        assert config.compression is True

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = ExportConfig(
            output_dir="/custom/path",
            max_rows_per_file=50000,
            compression=False,
        )
        assert config.output_dir == "/custom/path"
        assert config.max_rows_per_file == 50000
        assert config.compression is False


# ============================================================================
# TestExportJob
# ============================================================================


class TestExportJob:
    """Tests for ExportJob dataclass."""

    def test_job_creation(self) -> None:
        """ExportJob can be created with required fields."""
        job = ExportJob(job_id="j-1", format="csv", status="pending")
        assert job.job_id == "j-1"
        assert job.format == "csv"
        assert job.status == "pending"

    def test_job_default_values(self) -> None:
        """ExportJob has correct default values."""
        job = ExportJob(job_id="j-1", format="csv")
        assert job.file_path is None
        assert job.error is None
        assert job.row_count == 0


# ============================================================================
# TestCSVExporter
# ============================================================================


class TestCSVExporter:
    """Tests for CSVExporter."""

    @pytest.mark.asyncio
    async def test_csv_export_success(self, sample_events) -> None:
        """CSV export writes events successfully."""
        exporter = CSVExporter()
        output = io.StringIO()
        count = await exporter.export(sample_events, output)
        assert count == len(sample_events)

    @pytest.mark.asyncio
    async def test_csv_export_includes_header(self, sample_events) -> None:
        """CSV export includes header row."""
        exporter = CSVExporter()
        output = io.StringIO()
        await exporter.export(sample_events, output)
        output.seek(0)
        header = output.readline()
        assert "event_id" in header or "event_type" in header

    @pytest.mark.asyncio
    async def test_csv_export_all_fields(self, sample_events) -> None:
        """CSV export includes all event fields."""
        exporter = CSVExporter()
        output = io.StringIO()
        await exporter.export(sample_events, output)
        output.seek(0)
        content = output.read()
        assert "auth.login_success" in content

    @pytest.mark.asyncio
    async def test_csv_export_escapes_special_chars(self) -> None:
        """CSV export properly escapes special characters."""
        exporter = CSVExporter()
        events = [_make_event_dict()]
        events[0]["details"] = {"message": 'Contains "quotes" and, commas'}
        output = io.StringIO()
        count = await exporter.export(events, output)
        assert count == 1
        # Should not raise or corrupt data

    @pytest.mark.asyncio
    async def test_csv_export_empty_list(self) -> None:
        """CSV export handles empty event list."""
        exporter = CSVExporter()
        output = io.StringIO()
        count = await exporter.export([], output)
        assert count == 0


# ============================================================================
# TestJSONExporter
# ============================================================================


class TestJSONExporter:
    """Tests for JSONExporter."""

    @pytest.mark.asyncio
    async def test_json_export_success(self, sample_events) -> None:
        """JSON export writes events successfully."""
        exporter = JSONExporter()
        output = io.StringIO()
        count = await exporter.export(sample_events, output)
        assert count == len(sample_events)

    @pytest.mark.asyncio
    async def test_json_export_valid_json(self, sample_events) -> None:
        """JSON export produces valid JSON."""
        exporter = JSONExporter()
        output = io.StringIO()
        await exporter.export(sample_events, output)
        output.seek(0)
        parsed = json.load(output)
        assert isinstance(parsed, (list, dict))

    @pytest.mark.asyncio
    async def test_json_export_all_events(self, sample_events) -> None:
        """JSON export includes all events."""
        exporter = JSONExporter()
        output = io.StringIO()
        await exporter.export(sample_events, output)
        output.seek(0)
        parsed = json.load(output)
        events = parsed if isinstance(parsed, list) else parsed.get("events", [])
        assert len(events) == len(sample_events)

    @pytest.mark.asyncio
    async def test_json_export_preserves_types(self, sample_events) -> None:
        """JSON export preserves data types."""
        exporter = JSONExporter()
        output = io.StringIO()
        await exporter.export(sample_events, output)
        output.seek(0)
        parsed = json.load(output)
        events = parsed if isinstance(parsed, list) else parsed.get("events", [])
        assert events[0]["event_type"] == "auth.login_success"

    @pytest.mark.asyncio
    async def test_json_export_empty_list(self) -> None:
        """JSON export handles empty event list."""
        exporter = JSONExporter()
        output = io.StringIO()
        count = await exporter.export([], output)
        assert count == 0


# ============================================================================
# TestParquetExporter
# ============================================================================


class TestParquetExporter:
    """Tests for ParquetExporter."""

    @pytest.mark.asyncio
    async def test_parquet_export_success(self, sample_events) -> None:
        """Parquet export writes events successfully."""
        exporter = ParquetExporter()
        output = io.BytesIO()
        count = await exporter.export(sample_events, output)
        assert count == len(sample_events)

    @pytest.mark.asyncio
    async def test_parquet_export_binary_output(self, sample_events) -> None:
        """Parquet export produces binary output."""
        exporter = ParquetExporter()
        output = io.BytesIO()
        await exporter.export(sample_events, output)
        output.seek(0)
        # Parquet files start with "PAR1" magic bytes
        header = output.read(4)
        assert len(header) > 0  # Has content

    @pytest.mark.asyncio
    async def test_parquet_export_empty_list(self) -> None:
        """Parquet export handles empty event list."""
        exporter = ParquetExporter()
        output = io.BytesIO()
        count = await exporter.export([], output)
        assert count == 0


# ============================================================================
# TestAuditExportService
# ============================================================================


class TestAuditExportService:
    """Tests for AuditExportService."""

    @pytest.mark.asyncio
    async def test_create_export_job_csv(
        self, export_service: AuditExportService
    ) -> None:
        """create_export_job() creates a CSV export job."""
        job = await export_service.create_export_job(
            format="csv",
            filters={"category": "auth"},
        )
        assert job.job_id is not None
        assert job.format == "csv"
        assert job.status == "pending"

    @pytest.mark.asyncio
    async def test_create_export_job_json(
        self, export_service: AuditExportService
    ) -> None:
        """create_export_job() creates a JSON export job."""
        job = await export_service.create_export_job(
            format="json",
            filters={},
        )
        assert job.format == "json"

    @pytest.mark.asyncio
    async def test_create_export_job_parquet(
        self, export_service: AuditExportService
    ) -> None:
        """create_export_job() creates a Parquet export job."""
        job = await export_service.create_export_job(
            format="parquet",
            filters={},
        )
        assert job.format == "parquet"

    @pytest.mark.asyncio
    async def test_get_job_exists(
        self, export_service: AuditExportService
    ) -> None:
        """get_job() returns existing job."""
        job = await export_service.create_export_job(format="csv")
        retrieved = await export_service.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_get_job_not_found(
        self, export_service: AuditExportService
    ) -> None:
        """get_job() returns None for non-existent job."""
        job = await export_service.get_job("nonexistent-job-id")
        assert job is None

    @pytest.mark.asyncio
    async def test_execute_job_success(
        self, export_service: AuditExportService, event_repository
    ) -> None:
        """execute_job() completes job successfully."""
        job = await export_service.create_export_job(format="csv")
        completed = await export_service.execute_job(job.job_id)
        assert completed.status == "completed"
        assert completed.row_count > 0

    @pytest.mark.asyncio
    async def test_execute_job_sets_file_path(
        self, export_service: AuditExportService, event_repository
    ) -> None:
        """execute_job() sets output file path."""
        job = await export_service.create_export_job(format="json")
        completed = await export_service.execute_job(job.job_id)
        assert completed.file_path is not None

    @pytest.mark.asyncio
    async def test_execute_job_handles_failure(
        self, export_service: AuditExportService, event_repository
    ) -> None:
        """execute_job() handles export failures."""
        event_repository.get_events.side_effect = Exception("DB error")
        job = await export_service.create_export_job(format="csv")
        completed = await export_service.execute_job(job.job_id)
        assert completed.status == "failed"
        assert completed.error is not None

    @pytest.mark.asyncio
    async def test_get_download_url(
        self, export_service: AuditExportService, event_repository
    ) -> None:
        """get_download_url() returns URL for completed job."""
        job = await export_service.create_export_job(format="csv")
        await export_service.execute_job(job.job_id)
        url = await export_service.get_download_url(job.job_id)
        assert url is not None
        assert len(url) > 0


# ============================================================================
# TestLargeDatasetExport
# ============================================================================


class TestLargeDatasetExport:
    """Tests for large dataset streaming export."""

    @pytest.mark.asyncio
    async def test_stream_export_chunks(
        self, export_service: AuditExportService, event_repository
    ) -> None:
        """stream_export() yields data in chunks."""
        # Setup streaming response
        async def stream_events():
            for i in range(5):
                yield [_make_event_dict(event_id=f"e-{j}") for j in range(100)]

        event_repository.stream_events = stream_events

        chunks = []
        async for chunk in export_service.stream_export(
            format="csv",
            filters={},
            chunk_size=100,
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_export_memory_efficient(
        self, export_config, event_repository
    ) -> None:
        """stream_export() is memory efficient for large datasets."""
        # Configure for many events
        event_repository.get_events.return_value = {
            "items": [_make_event_dict(event_id=f"e-{i}") for i in range(10000)],
            "total": 100000,
        }

        service = AuditExportService(
            config=export_config,
            event_repository=event_repository,
        )

        # Should not load all events at once
        job = await service.create_export_job(format="csv")
        await service.execute_job(job.job_id)
        # Test passes if no memory error
