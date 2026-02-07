# -*- coding: utf-8 -*-
"""
Compliance Report Service - SEC-005

Central service for generating compliance reports. Manages report jobs,
background processing, and report storage.

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReportStatus(str, enum.Enum):
    """Report generation status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportFormat(str, enum.Enum):
    """Report output format."""

    PDF = "pdf"
    JSON = "json"
    HTML = "html"
    CSV = "csv"


class ReportPeriod(str, enum.Enum):
    """Pre-defined report periods."""

    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


class ReportJobRecord(BaseModel):
    """Report job status record."""

    job_id: str = Field(..., description="Unique job identifier")
    report_type: str = Field(..., description="Report type: soc2, iso27001, gdpr")
    status: ReportStatus = Field(default=ReportStatus.PENDING)
    progress_percent: float = Field(default=0.0)
    format: ReportFormat = Field(default=ReportFormat.PDF)
    period: ReportPeriod = Field(default=ReportPeriod.LAST_30_DAYS)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    organization_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceReportService:
    """Central service for compliance report generation.

    Manages report jobs, coordinates with specific report generators,
    handles background processing and storage.
    """

    def __init__(
        self,
        storage_path: str = "/tmp/greenlang/reports",
        s3_bucket: Optional[str] = None,
    ):
        """Initialize the report service.

        Args:
            storage_path: Local path for temporary report storage.
            s3_bucket: Optional S3 bucket for permanent storage.
        """
        self._storage_path = storage_path
        self._s3_bucket = s3_bucket
        self._jobs: Dict[str, ReportJobRecord] = {}
        self._generators: Dict[str, Any] = {}
        self._processing_lock = asyncio.Lock()

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)

    def register_generator(self, report_type: str, generator: Any) -> None:
        """Register a report generator.

        Args:
            report_type: Report type identifier (soc2, iso27001, gdpr).
            generator: Report generator instance.
        """
        self._generators[report_type] = generator
        logger.info("Registered report generator: %s", report_type)

    async def generate_report(
        self,
        report_type: str,
        period: ReportPeriod = ReportPeriod.LAST_30_DAYS,
        format: ReportFormat = ReportFormat.PDF,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        organization_id: Optional[str] = None,
        requested_by: Optional[str] = None,
    ) -> str:
        """Start report generation.

        Args:
            report_type: Report type (soc2, iso27001, gdpr).
            period: Pre-defined period or CUSTOM.
            format: Output format.
            period_start: Custom period start (required if period=CUSTOM).
            period_end: Custom period end (required if period=CUSTOM).
            organization_id: Optional organization filter.
            requested_by: User who requested the report.

        Returns:
            Job ID for status tracking.

        Raises:
            ValueError: If report type is not registered or invalid period.
        """
        if report_type not in self._generators:
            raise ValueError(f"Unknown report type: {report_type}")

        # Resolve period dates
        now = datetime.now(timezone.utc)
        if period == ReportPeriod.CUSTOM:
            if not period_start or not period_end:
                raise ValueError("Custom period requires period_start and period_end")
        else:
            period_end = now
            period_map = {
                ReportPeriod.LAST_7_DAYS: timedelta(days=7),
                ReportPeriod.LAST_30_DAYS: timedelta(days=30),
                ReportPeriod.LAST_90_DAYS: timedelta(days=90),
                ReportPeriod.LAST_YEAR: timedelta(days=365),
            }
            period_start = now - period_map.get(period, timedelta(days=30))

        # Create job record
        job_id = f"rpt_{uuid.uuid4().hex[:12]}"
        job = ReportJobRecord(
            job_id=job_id,
            report_type=report_type,
            status=ReportStatus.PENDING,
            format=format,
            period=period,
            period_start=period_start,
            period_end=period_end,
            organization_id=organization_id,
            metadata={"requested_by": requested_by},
        )

        self._jobs[job_id] = job
        logger.info("Created report job: %s (%s)", job_id, report_type)

        # Start background processing
        asyncio.create_task(self._process_job(job_id))

        return job_id

    async def get_job_status(self, job_id: str) -> Optional[ReportJobRecord]:
        """Get report job status.

        Args:
            job_id: Job identifier.

        Returns:
            Job record or None if not found.
        """
        return self._jobs.get(job_id)

    async def list_jobs(
        self,
        report_type: Optional[str] = None,
        status: Optional[ReportStatus] = None,
        organization_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ReportJobRecord]:
        """List report jobs with optional filters.

        Args:
            report_type: Filter by report type.
            status: Filter by status.
            organization_id: Filter by organization.
            limit: Maximum jobs to return.

        Returns:
            List of matching job records.
        """
        jobs = list(self._jobs.values())

        if report_type:
            jobs = [j for j in jobs if j.report_type == report_type]

        if status:
            jobs = [j for j in jobs if j.status == status]

        if organization_id:
            jobs = [j for j in jobs if j.organization_id == organization_id]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def get_download_url(self, job_id: str) -> Optional[str]:
        """Get download URL for a completed report.

        Args:
            job_id: Job identifier.

        Returns:
            Download URL or None if not available.
        """
        job = self._jobs.get(job_id)
        if not job or job.status != ReportStatus.COMPLETED:
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
                    ExpiresIn=3600,  # 1 hour
                )
                return url
            except Exception as exc:
                logger.error("Failed to generate S3 presigned URL: %s", exc)

        # Return local file path
        return job.file_path

    async def _process_job(self, job_id: str) -> None:
        """Process a report generation job.

        Args:
            job_id: Job identifier.
        """
        job = self._jobs.get(job_id)
        if not job:
            return

        async with self._processing_lock:
            try:
                # Update status
                job.status = ReportStatus.PROCESSING
                job.started_at = datetime.now(timezone.utc)
                logger.info("Processing report job: %s", job_id)

                # Get generator
                generator = self._generators.get(job.report_type)
                if not generator:
                    raise ValueError(f"Generator not found: {job.report_type}")

                # Generate report
                result = await generator.generate(
                    period_start=job.period_start,
                    period_end=job.period_end,
                    organization_id=job.organization_id,
                    format=job.format,
                    progress_callback=lambda p: self._update_progress(job_id, p),
                )

                # Store report
                file_path = await self._store_report(job, result)

                # Update job status
                job.status = ReportStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                job.file_path = file_path
                job.progress_percent = 100.0

                if os.path.exists(file_path):
                    job.file_size_bytes = os.path.getsize(file_path)

                logger.info("Report job completed: %s", job_id)

            except Exception as exc:
                logger.exception("Report job failed: %s", job_id)
                job.status = ReportStatus.FAILED
                job.error_message = str(exc)
                job.completed_at = datetime.now(timezone.utc)

    def _update_progress(self, job_id: str, progress: float) -> None:
        """Update job progress.

        Args:
            job_id: Job identifier.
            progress: Progress percentage (0-100).
        """
        job = self._jobs.get(job_id)
        if job:
            job.progress_percent = min(progress, 99.0)  # Reserve 100% for completion

    async def _store_report(
        self, job: ReportJobRecord, content: bytes
    ) -> str:
        """Store generated report.

        Args:
            job: Job record.
            content: Report content bytes.

        Returns:
            Storage path or URL.
        """
        # Generate filename
        timestamp = job.created_at.strftime("%Y%m%d_%H%M%S")
        extension = job.format.value
        filename = f"{job.report_type}_{timestamp}_{job.job_id}.{extension}"

        # Store locally first
        local_path = os.path.join(self._storage_path, filename)
        with open(local_path, "wb") as f:
            f.write(content)

        # Upload to S3 if configured
        if self._s3_bucket:
            try:
                import boto3

                s3 = boto3.client("s3")
                s3_key = f"reports/{job.report_type}/{filename}"

                s3.upload_file(
                    local_path,
                    self._s3_bucket,
                    s3_key,
                    ExtraArgs={
                        "ContentType": self._get_content_type(job.format),
                        "Metadata": {
                            "report_type": job.report_type,
                            "job_id": job.job_id,
                            "organization_id": job.organization_id or "",
                        },
                    },
                )

                # Clean up local file
                os.remove(local_path)

                return s3_key

            except Exception as exc:
                logger.warning("S3 upload failed, keeping local file: %s", exc)

        return local_path

    def _get_content_type(self, format: ReportFormat) -> str:
        """Get MIME content type for format.

        Args:
            format: Report format.

        Returns:
            MIME type string.
        """
        content_types = {
            ReportFormat.PDF: "application/pdf",
            ReportFormat.JSON: "application/json",
            ReportFormat.HTML: "text/html",
            ReportFormat.CSV: "text/csv",
        }
        return content_types.get(format, "application/octet-stream")


# Singleton instance
_report_service: Optional[ComplianceReportService] = None


def get_report_service() -> ComplianceReportService:
    """Get or create the global ComplianceReportService.

    Returns:
        The ComplianceReportService singleton.
    """
    global _report_service
    if _report_service is None:
        storage_path = os.getenv("REPORT_STORAGE_PATH", "/tmp/greenlang/reports")
        s3_bucket = os.getenv("REPORT_S3_BUCKET")
        _report_service = ComplianceReportService(
            storage_path=storage_path,
            s3_bucket=s3_bucket,
        )

        # Register default generators
        try:
            from greenlang.infrastructure.audit_service.reporting.soc2_report import (
                SOC2ReportGenerator,
            )
            from greenlang.infrastructure.audit_service.reporting.iso27001_report import (
                ISO27001ReportGenerator,
            )
            from greenlang.infrastructure.audit_service.reporting.gdpr_report import (
                GDPRReportGenerator,
            )

            _report_service.register_generator("soc2", SOC2ReportGenerator())
            _report_service.register_generator("iso27001", ISO27001ReportGenerator())
            _report_service.register_generator("gdpr", GDPRReportGenerator())
        except ImportError as exc:
            logger.warning("Failed to register report generators: %s", exc)

    return _report_service


async def init_report_service(
    storage_path: str = "/tmp/greenlang/reports",
    s3_bucket: Optional[str] = None,
) -> ComplianceReportService:
    """Initialize the global ComplianceReportService.

    Args:
        storage_path: Local storage path.
        s3_bucket: Optional S3 bucket.

    Returns:
        The initialized service.
    """
    global _report_service
    _report_service = ComplianceReportService(
        storage_path=storage_path,
        s3_bucket=s3_bucket,
    )
    return _report_service
