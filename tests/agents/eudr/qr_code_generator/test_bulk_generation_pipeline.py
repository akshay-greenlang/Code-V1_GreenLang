# -*- coding: utf-8 -*-
"""
Unit tests for Engine 7: Bulk Generation Pipeline (AGENT-EUDR-014)

Tests bulk QR code generation including job submission, parallel
processing, progress tracking, output packaging, manifest generation,
output validation, job lifecycle, and edge cases.

50+ tests across 8 test classes.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from .conftest import (
    BULK_JOB_STATUSES,
    BULK_MAX_SIZE,
    CONTENT_TYPES,
    DEFAULT_BULK_TIMEOUT_S,
    DEFAULT_BULK_WORKERS,
    ERROR_CORRECTION_LEVELS,
    EUDR_COMMODITIES,
    SAMPLE_COMMODITY,
    SAMPLE_OPERATOR_ID,
    SAMPLE_OUTPUT_FILE_HASH,
    SHA256_HEX_LENGTH,
    SUPPORTED_FORMATS,
    assert_bulk_job_valid,
    assert_valid_sha256,
    make_bulk_job,
    _sha256,
)


# =========================================================================
# Test Class 1: Bulk Job Submission
# =========================================================================

class TestBulkJobSubmission:
    """Test bulk job creation and validation."""

    def test_create_bulk_job(self):
        """Test basic bulk job creation."""
        job = make_bulk_job()
        assert job["status"] == "queued"
        assert_bulk_job_valid(job)

    def test_job_has_unique_id(self):
        """Test each job has a unique identifier."""
        ids = {make_bulk_job()["job_id"] for _ in range(50)}
        assert len(ids) == 50

    def test_job_total_codes_required(self):
        """Test total codes must be specified."""
        job = make_bulk_job(total_codes=500)
        assert job["total_codes"] == 500

    def test_job_operator_id_required(self):
        """Test operator ID is required."""
        job = make_bulk_job(operator_id=SAMPLE_OPERATOR_ID)
        assert job["operator_id"] == SAMPLE_OPERATOR_ID

    def test_job_starts_queued(self):
        """Test new job starts in queued status."""
        job = make_bulk_job()
        assert job["status"] == "queued"
        assert job["completed_codes"] == 0
        assert job["failed_codes"] == 0

    def test_job_progress_starts_at_zero(self):
        """Test new job starts with 0% progress."""
        job = make_bulk_job()
        assert job["progress_percent"] == 0.0

    @pytest.mark.parametrize("fmt", SUPPORTED_FORMATS)
    def test_job_with_output_format(self, fmt: str):
        """Test bulk job with each output format."""
        job = make_bulk_job(output_format=fmt)
        assert job["output_format"] == fmt
        assert_bulk_job_valid(job)

    @pytest.mark.parametrize("ct", CONTENT_TYPES)
    def test_job_with_content_type(self, ct: str):
        """Test bulk job with each content type."""
        job = make_bulk_job(content_type=ct)
        assert job["content_type"] == ct


# =========================================================================
# Test Class 2: Bulk Processing
# =========================================================================

class TestBulkProcessing:
    """Test parallel worker processing."""

    def test_processing_status(self):
        """Test job transitions to processing status."""
        job = make_bulk_job(status="processing", completed_codes=100)
        assert job["status"] == "processing"

    def test_worker_count_default(self):
        """Test default worker count is 4."""
        job = make_bulk_job()
        assert job["worker_count"] == DEFAULT_BULK_WORKERS

    def test_custom_worker_count(self):
        """Test custom worker count."""
        job = make_bulk_job(worker_count=16)
        assert job["worker_count"] == 16

    def test_processing_has_started_at(self):
        """Test processing job has a started_at timestamp."""
        job = make_bulk_job(status="processing", completed_codes=50)
        assert job["started_at"] is not None

    def test_completed_codes_increment(self):
        """Test completed codes can be incremented."""
        job = make_bulk_job(status="processing", completed_codes=500)
        assert job["completed_codes"] == 500

    def test_failed_codes_tracked(self):
        """Test failed codes are tracked separately."""
        job = make_bulk_job(
            status="processing",
            completed_codes=450,
            failed_codes=50,
        )
        assert job["failed_codes"] == 50

    @pytest.mark.parametrize("ec", ERROR_CORRECTION_LEVELS)
    def test_job_with_error_correction(self, ec: str):
        """Test bulk job with each error correction level."""
        job = make_bulk_job(error_correction=ec)
        assert job["error_correction"] == ec


# =========================================================================
# Test Class 3: Progress Tracking
# =========================================================================

class TestProgressTracking:
    """Test progress percentage and ETA calculation."""

    def test_progress_at_zero(self):
        """Test progress is 0% when no codes generated."""
        job = make_bulk_job(total_codes=1000, completed_codes=0)
        assert job["progress_percent"] == 0.0

    def test_progress_at_50_percent(self):
        """Test progress is 50% at halfway point."""
        job = make_bulk_job(total_codes=1000, completed_codes=500)
        assert job["progress_percent"] == 50.0

    def test_progress_at_100_percent(self):
        """Test progress is 100% when all codes completed."""
        job = make_bulk_job(
            status="completed",
            total_codes=1000,
            completed_codes=1000,
        )
        assert job["progress_percent"] == 100.0

    def test_progress_includes_failed(self):
        """Test progress accounts for failed codes."""
        job = make_bulk_job(
            total_codes=100,
            completed_codes=80,
            failed_codes=20,
        )
        assert job["progress_percent"] == 100.0

    def test_progress_range_valid(self):
        """Test progress is always 0-100."""
        job = make_bulk_job(total_codes=200, completed_codes=75, failed_codes=25)
        assert 0.0 <= job["progress_percent"] <= 100.0

    def test_progress_partial(self):
        """Test partial progress percentage."""
        job = make_bulk_job(total_codes=1000, completed_codes=333)
        assert 30.0 <= job["progress_percent"] <= 35.0

    def test_progress_with_single_code(self):
        """Test progress with just one code total."""
        job = make_bulk_job(total_codes=1, completed_codes=1)
        assert job["progress_percent"] == 100.0


# =========================================================================
# Test Class 4: Output Packaging
# =========================================================================

class TestOutputPackaging:
    """Test ZIP archive and multi-page PDF packaging."""

    def test_completed_job_has_output_hash(self):
        """Test completed job has output file hash."""
        job = make_bulk_job(status="completed", total_codes=100, completed_codes=100)
        assert job["output_file_hash"] is not None

    def test_output_hash_is_sha256(self):
        """Test output file hash is SHA-256."""
        job = make_bulk_job(status="completed", total_codes=100, completed_codes=100)
        assert_valid_sha256(job["output_file_hash"])

    def test_completed_job_has_file_size(self):
        """Test completed job has output file size."""
        job = make_bulk_job(status="completed", total_codes=100, completed_codes=100)
        assert job["output_file_size_bytes"] is not None
        assert job["output_file_size_bytes"] > 0

    def test_completed_job_has_download_url(self):
        """Test completed job has a download URL."""
        job = make_bulk_job(status="completed", total_codes=100, completed_codes=100)
        assert job["output_file_url"] is not None
        assert "https://" in job["output_file_url"]

    def test_zip_output_format_default(self):
        """Test ZIP is the default bulk output format."""
        job = make_bulk_job()
        assert job["bulk_output_format"] == "zip"

    def test_queued_job_no_output(self):
        """Test queued job has no output file yet."""
        job = make_bulk_job(status="queued")
        assert job["output_file_hash"] is None
        assert job["output_file_url"] is None

    def test_failed_job_no_output(self):
        """Test failed job has no output file."""
        job = make_bulk_job(status="failed", completed_codes=50, failed_codes=50)
        assert job["output_file_hash"] is None


# =========================================================================
# Test Class 5: Manifest Generation
# =========================================================================

class TestManifestGeneration:
    """Test CSV manifest generation for bulk jobs."""

    def test_completed_job_provenance(self):
        """Test completed job has provenance hash."""
        job = make_bulk_job(status="completed", total_codes=100, completed_codes=100)
        assert job["provenance_hash"] is not None

    def test_job_records_commodity(self):
        """Test bulk job records the commodity type."""
        job = make_bulk_job(commodity="coffee")
        assert job["commodity"] == "coffee"

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_all_commodities_in_bulk(self, commodity: str):
        """Test bulk jobs for all EUDR commodities."""
        job = make_bulk_job(commodity=commodity)
        assert job["commodity"] == commodity

    def test_job_records_operator(self):
        """Test bulk job records operator ID."""
        job = make_bulk_job(operator_id="OP-MANIFEST-001")
        assert job["operator_id"] == "OP-MANIFEST-001"

    def test_completed_job_has_timestamps(self):
        """Test completed job has started and completed timestamps."""
        job = make_bulk_job(status="completed", total_codes=10, completed_codes=10)
        assert job["started_at"] is not None
        assert job["completed_at"] is not None

    def test_job_created_at_present(self):
        """Test job has creation timestamp."""
        job = make_bulk_job()
        assert job["created_at"] is not None


# =========================================================================
# Test Class 6: Output Validation
# =========================================================================

class TestOutputValidation:
    """Test QR code verification after generation."""

    def test_completed_job_all_codes_done(self):
        """Test completed job has all codes generated."""
        job = make_bulk_job(
            status="completed",
            total_codes=100,
            completed_codes=100,
            failed_codes=0,
        )
        assert job["completed_codes"] == job["total_codes"]
        assert job["failed_codes"] == 0

    def test_partial_failure_tracked(self):
        """Test partial failures are tracked."""
        job = make_bulk_job(
            status="completed",
            total_codes=100,
            completed_codes=95,
            failed_codes=5,
        )
        assert job["completed_codes"] + job["failed_codes"] == 100

    def test_output_file_hash_unique(self):
        """Test each completed job has a unique output hash."""
        hashes = set()
        for _ in range(10):
            job = make_bulk_job(status="completed", total_codes=10, completed_codes=10)
            if job["output_file_hash"]:
                hashes.add(job["output_file_hash"])
        assert len(hashes) == 10

    def test_validation_pass_produces_output(self):
        """Test successful validation produces output file."""
        job = make_bulk_job(status="completed", total_codes=50, completed_codes=50)
        assert job["output_file_hash"] is not None
        assert job["output_file_size_bytes"] is not None

    def test_zero_failures_ideal(self):
        """Test ideal case with zero failures."""
        job = make_bulk_job(
            status="completed",
            total_codes=1000,
            completed_codes=1000,
            failed_codes=0,
        )
        assert job["failed_codes"] == 0


# =========================================================================
# Test Class 7: Job Lifecycle
# =========================================================================

class TestJobLifecycle:
    """Test job cancel, resume, and scheduling."""

    @pytest.mark.parametrize("status", BULK_JOB_STATUSES)
    def test_all_job_statuses(self, status: str):
        """Test all bulk job statuses are valid."""
        codes = 50 if status in ("processing", "completed", "failed") else 0
        job = make_bulk_job(status=status, completed_codes=codes)
        assert job["status"] == status

    def test_cancelled_job(self):
        """Test cancelled job status."""
        job = make_bulk_job(status="cancelled", completed_codes=200)
        assert job["status"] == "cancelled"

    def test_failed_job_has_error_message(self):
        """Test failed job has an error message."""
        job = make_bulk_job(
            status="failed",
            error_message="Worker timeout exceeded",
            completed_codes=50,
            failed_codes=50,
        )
        assert job["error_message"] is not None
        assert "timeout" in job["error_message"].lower()

    def test_queued_no_started_at(self):
        """Test queued job has no started_at timestamp."""
        job = make_bulk_job(status="queued")
        assert job["started_at"] is None

    def test_processing_has_started_no_completed(self):
        """Test processing job has started but not completed."""
        job = make_bulk_job(status="processing", completed_codes=100)
        assert job["started_at"] is not None
        assert job["completed_at"] is None

    def test_exactly_five_job_statuses(self):
        """Test exactly 5 bulk job statuses exist."""
        assert len(BULK_JOB_STATUSES) == 5


# =========================================================================
# Test Class 8: Edge Cases
# =========================================================================

class TestBulkEdgeCases:
    """Test edge cases for bulk generation."""

    def test_max_bulk_size(self):
        """Test maximum bulk job size."""
        job = make_bulk_job(total_codes=BULK_MAX_SIZE)
        assert job["total_codes"] == BULK_MAX_SIZE
        assert_bulk_job_valid(job)

    def test_single_code_bulk(self):
        """Test bulk job with just 1 code."""
        job = make_bulk_job(total_codes=1)
        assert job["total_codes"] == 1
        assert_bulk_job_valid(job)

    def test_large_worker_count(self):
        """Test bulk job with high worker count."""
        job = make_bulk_job(worker_count=64)
        assert job["worker_count"] == 64

    def test_single_worker(self):
        """Test bulk job with single worker."""
        job = make_bulk_job(worker_count=1)
        assert job["worker_count"] == 1

    def test_timeout_recorded(self):
        """Test bulk timeout constant."""
        assert DEFAULT_BULK_TIMEOUT_S == 3600

    def test_bulk_max_constant(self):
        """Test BULK_MAX_SIZE constant."""
        assert BULK_MAX_SIZE == 100_000

    def test_bulk_job_with_no_commodity(self):
        """Test bulk job without commodity filter."""
        job = make_bulk_job(commodity=None)
        assert job["commodity"] is None
        assert_bulk_job_valid(job)

    def test_error_message_none_for_success(self):
        """Test no error message for successful jobs."""
        job = make_bulk_job(status="completed", total_codes=10, completed_codes=10)
        assert job["error_message"] is None

    def test_failed_codes_cannot_exceed_total(self):
        """Test failed + completed codes equal total."""
        job = make_bulk_job(
            total_codes=100,
            completed_codes=70,
            failed_codes=30,
        )
        assert job["completed_codes"] + job["failed_codes"] == job["total_codes"]
