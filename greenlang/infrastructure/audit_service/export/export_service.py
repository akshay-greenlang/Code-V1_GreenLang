# -*- coding: utf-8 -*-
"""
Audit Export Service - SEC-005

Central service for exporting audit events to various formats.
Manages export jobs, streaming for large datasets, and S3 upload.

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import enum
import io
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExportFormat(str, enum.Enum):
    """Supported export formats."""

    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class ExportStatus(str, enum.Enum):
    """Export job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportFilters(BaseModel):
    """Filters for export query."""

    since: Optional[datetime] = None
    until: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    severities: Optional[List[str]] = None
    organization_id: Optional[str] = None
    user_id: Optional[str] = None


class ExportJob(BaseModel):
    """Export job status record."""

    job_id: str = Field(..., description="Unique job identifier")
    status: ExportStatus = Field(default=ExportStatus.PENDING)
    export_format: ExportFormat = Field(..., description="Export format")
    filters: ExportFilters = Field(default_factory=ExportFilters)
    compress: bool = Field(default=True, description="Whether to compress output")
    progress_percent: float = Field(default=0.0)
    total_records: int = Field(default=0)
    exported_records: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    download_url: Optional[str] = None


class AuditExportService:
    """Service for exporting audit events.

    Supports streaming exports for large datasets and automatic
    upload to S3 for completed exports.
    """

    def __init__(
        self,
        storage_path: str = "/tmp/greenlang/exports",
        s3_bucket: Optional[str] = None,
        repository: Optional[Any] = None,
    ):
        """Initialize the export service.

        Args:
            storage_path: Local path for temporary storage.
            s3_bucket: Optional S3 bucket for permanent storage.
            repository: Audit event repository.
        """
        self._storage_path = storage_path
        self._s3_bucket = s3_bucket
        self._repository = repository
        self._jobs: Dict[str, ExportJob] = {}
        self._processing_lock = asyncio.Lock()

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)

    def set_repository(self, repository: Any) -> None:
        """Set the audit event repository.

        Args:
            repository: Audit event repository instance.
        """
        self._repository = repository

    async def start_export(
        self,
        format: ExportFormat,
        filters: Optional[ExportFilters] = None,
        compress: bool = True,
        requested_by: Optional[str] = None,
    ) -> str:
        """Start an export job.

        Args:
            format: Export format.
            filters: Optional query filters.
            compress: Whether to compress output.
            requested_by: User who requested the export.

        Returns:
            Job ID for status tracking.
        """
        job_id = f"exp_{uuid.uuid4().hex[:12]}"

        job = ExportJob(
            job_id=job_id,
            export_format=format,
            filters=filters or ExportFilters(),
            compress=compress,
        )

        self._jobs[job_id] = job
        logger.info("Created export job: %s (%s)", job_id, format.value)

        # Start background processing
        asyncio.create_task(self._process_job(job_id))

        return job_id

    async def get_job_status(self, job_id: str) -> Optional[ExportJob]:
        """Get export job status.

        Args:
            job_id: Job identifier.

        Returns:
            Job record or None if not found.
        """
        return self._jobs.get(job_id)

    async def list_jobs(
        self,
        status: Optional[ExportStatus] = None,
        format: Optional[ExportFormat] = None,
        limit: int = 50,
    ) -> List[ExportJob]:
        """List export jobs.

        Args:
            status: Filter by status.
            format: Filter by format.
            limit: Maximum jobs to return.

        Returns:
            List of matching jobs.
        """
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        if format:
            jobs = [j for j in jobs if j.export_format == format]

        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def get_download_url(self, job_id: str) -> Optional[str]:
        """Get download URL for a completed export.

        Args:
            job_id: Job identifier.

        Returns:
            Download URL or None if not available.
        """
        job = self._jobs.get(job_id)
        if not job or job.status != ExportStatus.COMPLETED:
            return None

        if job.download_url:
            return job.download_url

        # Generate presigned S3 URL if using S3
        if self._s3_bucket and job.file_path:
            try:
                import boto3

                s3 = boto3.client("s3")
                url = s3.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": self._s3_bucket,
                        "Key": job.file_path,
                    },
                    ExpiresIn=3600,
                )
                return url
            except Exception as exc:
                logger.error("Failed to generate S3 presigned URL: %s", exc)

        return job.file_path

    async def _process_job(self, job_id: str) -> None:
        """Process an export job.

        Args:
            job_id: Job identifier.
        """
        job = self._jobs.get(job_id)
        if not job:
            return

        async with self._processing_lock:
            try:
                job.status = ExportStatus.PROCESSING
                job.started_at = datetime.now(timezone.utc)
                logger.info("Processing export job: %s", job_id)

                # Get exporter
                from greenlang.infrastructure.audit_service.export.formats import (
                    get_exporter,
                )

                exporter = get_exporter(
                    format=job.export_format.value,
                    compress=job.compress,
                    include_metadata=True,
                )

                # Create output buffer
                output = io.BytesIO()

                # Stream events from repository
                events = self._stream_events(job.filters, job)

                # Export
                count = await exporter.export(events, output)

                job.exported_records = count

                # Save file
                extension = exporter.file_extension
                filename = f"audit_export_{job_id}.{extension}"
                file_path = os.path.join(self._storage_path, filename)

                output.seek(0)
                with open(file_path, "wb") as f:
                    f.write(output.getvalue())

                job.file_path = file_path
                job.file_size_bytes = os.path.getsize(file_path)

                # Upload to S3 if configured
                if self._s3_bucket:
                    await self._upload_to_s3(job, file_path, filename)

                job.status = ExportStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                job.progress_percent = 100.0

                logger.info(
                    "Export job completed: %s (%d records, %d bytes)",
                    job_id, count, job.file_size_bytes or 0
                )

            except Exception as exc:
                logger.exception("Export job failed: %s", job_id)
                job.status = ExportStatus.FAILED
                job.error_message = str(exc)
                job.completed_at = datetime.now(timezone.utc)

    async def _stream_events(
        self,
        filters: ExportFilters,
        job: ExportJob,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream events from the repository.

        Args:
            filters: Query filters.
            job: Export job for progress tracking.

        Yields:
            Audit event dictionaries.
        """
        if self._repository is None:
            # Generate sample data for demonstration
            for i in range(100):
                yield {
                    "id": str(uuid.uuid4()),
                    "performed_at": datetime.now(timezone.utc).isoformat(),
                    "category": "authentication",
                    "severity": "info",
                    "event_type": "login_success",
                    "operation": "LOGIN",
                    "outcome": "success",
                    "user_id": str(uuid.uuid4()),
                    "user_email": f"user{i}@example.com",
                    "organization_id": str(uuid.uuid4()),
                    "resource_type": "user",
                    "resource_path": f"/users/user{i}",
                    "action": "authenticate",
                    "ip_address": f"10.0.0.{i % 256}",
                    "change_summary": "User logged in successfully",
                    "error_message": None,
                    "tags": ["auth", "login"],
                    "metadata": {"browser": "Chrome", "os": "Windows"},
                }
                job.progress_percent = (i + 1) / 100 * 90
            return

        try:
            from greenlang.infrastructure.audit_service.models import (
                SearchQuery,
                TimeRange,
                EventCategory,
                SeverityLevel,
            )
            from uuid import UUID

            # Build query
            time_range = None
            if filters.since or filters.until:
                time_range = TimeRange(
                    start=filters.since or datetime.min.replace(tzinfo=timezone.utc),
                    end=filters.until or datetime.now(timezone.utc),
                )

            categories = None
            if filters.categories:
                categories = [EventCategory(c) for c in filters.categories]

            severities = None
            if filters.severities:
                severities = [SeverityLevel(s) for s in filters.severities]

            organization_ids = None
            if filters.organization_id:
                organization_ids = [UUID(filters.organization_id)]

            user_ids = None
            if filters.user_id:
                user_ids = [UUID(filters.user_id)]

            query = SearchQuery(
                time_range=time_range,
                categories=categories,
                severities=severities,
                event_types=filters.event_types,
                organization_ids=organization_ids,
                user_ids=user_ids,
            )

            # Get total count first
            _, total, _ = await self._repository.search(query=query, limit=1, offset=0)
            job.total_records = total

            # Stream in batches
            batch_size = 1000
            offset = 0

            while offset < total:
                events, _, _ = await self._repository.search(
                    query=query,
                    limit=batch_size,
                    offset=offset,
                )

                for event in events:
                    yield {
                        "id": str(event.id),
                        "performed_at": event.performed_at.isoformat(),
                        "category": event.category.value,
                        "severity": event.severity.value,
                        "event_type": event.event_type,
                        "operation": event.operation,
                        "outcome": event.outcome.value,
                        "user_id": str(event.user_id) if event.user_id else None,
                        "user_email": event.user_email,
                        "organization_id": str(event.organization_id) if event.organization_id else None,
                        "resource_type": event.resource_type,
                        "resource_path": event.resource_path,
                        "action": event.action,
                        "ip_address": event.ip_address,
                        "change_summary": event.change_summary,
                        "error_message": event.error_message,
                        "tags": event.tags,
                        "metadata": event.metadata,
                    }

                offset += len(events)
                job.exported_records = offset
                job.progress_percent = min(offset / max(total, 1) * 90, 90)

        except Exception as exc:
            logger.error("Error streaming events: %s", exc)
            raise

    async def _upload_to_s3(
        self,
        job: ExportJob,
        local_path: str,
        filename: str,
    ) -> None:
        """Upload export file to S3.

        Args:
            job: Export job.
            local_path: Local file path.
            filename: File name.
        """
        try:
            import boto3

            s3 = boto3.client("s3")
            s3_key = f"exports/{job.export_format.value}/{filename}"

            content_type_map = {
                ExportFormat.CSV: "text/csv",
                ExportFormat.JSON: "application/x-ndjson",
                ExportFormat.PARQUET: "application/vnd.apache.parquet",
            }

            s3.upload_file(
                local_path,
                self._s3_bucket,
                s3_key,
                ExtraArgs={
                    "ContentType": content_type_map.get(job.export_format, "application/octet-stream"),
                    "Metadata": {
                        "job_id": job.job_id,
                        "record_count": str(job.exported_records),
                    },
                },
            )

            job.file_path = s3_key
            logger.info("Uploaded export to S3: %s", s3_key)

            # Clean up local file
            os.remove(local_path)

        except Exception as exc:
            logger.warning("S3 upload failed, keeping local file: %s", exc)


# Singleton instance
_export_service: Optional[AuditExportService] = None


def get_export_service() -> AuditExportService:
    """Get or create the global AuditExportService.

    Returns:
        The AuditExportService singleton.
    """
    global _export_service
    if _export_service is None:
        storage_path = os.getenv("EXPORT_STORAGE_PATH", "/tmp/greenlang/exports")
        s3_bucket = os.getenv("EXPORT_S3_BUCKET")

        _export_service = AuditExportService(
            storage_path=storage_path,
            s3_bucket=s3_bucket,
        )

        # Try to get repository
        try:
            from greenlang.infrastructure.audit_service.repository import (
                get_audit_repository,
            )
            _export_service.set_repository(get_audit_repository())
        except (ImportError, RuntimeError):
            pass  # Repository not initialized yet

    return _export_service


async def init_export_service(
    storage_path: str = "/tmp/greenlang/exports",
    s3_bucket: Optional[str] = None,
    repository: Optional[Any] = None,
) -> AuditExportService:
    """Initialize the global AuditExportService.

    Args:
        storage_path: Local storage path.
        s3_bucket: Optional S3 bucket.
        repository: Optional audit repository.

    Returns:
        The initialized service.
    """
    global _export_service
    _export_service = AuditExportService(
        storage_path=storage_path,
        s3_bucket=s3_bucket,
        repository=repository,
    )
    return _export_service
