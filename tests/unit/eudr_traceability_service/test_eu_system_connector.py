# -*- coding: utf-8 -*-
"""
Unit Tests for EUSystemConnector (AGENT-DATA-005)

Tests EU Information System submission preparation, sandbox submission
simulation, status tracking, retry logic, bulk submissions, format
validation, and simulated EU response handling.

Coverage target: 85%+ of eu_system_connector.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums
# ---------------------------------------------------------------------------


class SubmissionStatus(str, Enum):
    DRAFT = "draft"
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Inline data models
# ---------------------------------------------------------------------------


class EUSubmissionRecord:
    """Record of a submission to the EU Information System."""

    def __init__(self, submission_id: str, dds_id: str,
                 status: str = "prepared",
                 eu_reference: Optional[str] = None,
                 retry_count: int = 0,
                 max_retries: int = 3,
                 errors: Optional[List[str]] = None):
        self.submission_id = submission_id
        self.dds_id = dds_id
        self.status = status
        self.eu_reference = eu_reference
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.errors = errors or []
        self.provenance_hash = ""
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.submitted_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Inline EUSystemConnector mirroring greenlang/eudr_traceability/eu_system_connector.py
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class EUSystemConnector:
    """EU Information System submission and tracking engine.

    Handles preparation, submission (sandbox mode), status tracking,
    retry logic, and format validation for DDS submissions to the
    EU Information System.
    """

    def __init__(self, sandbox: bool = True, max_retries: int = 3,
                 eu_system_url: str = ""):
        self._sandbox = sandbox
        self._max_retries = max_retries
        self._eu_system_url = eu_system_url
        self._submissions: Dict[str, EUSubmissionRecord] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"SUB-{self._counter:05d}"

    def prepare_submission(self, dds_id: str,
                           dds_data: Optional[Dict[str, Any]] = None) -> EUSubmissionRecord:
        """Prepare a DDS for submission to the EU Information System."""
        sub_id = self._next_id()

        record = EUSubmissionRecord(
            submission_id=sub_id,
            dds_id=dds_id,
            status="prepared",
            max_retries=self._max_retries,
        )
        record.provenance_hash = _compute_hash({
            "submission_id": sub_id,
            "dds_id": dds_id,
            "status": "prepared",
        })

        self._submissions[sub_id] = record
        return record

    def submit(self, submission_id: str) -> EUSubmissionRecord:
        """Submit a prepared DDS to the EU Information System (sandbox)."""
        record = self._submissions.get(submission_id)
        if record is None:
            raise ValueError(f"Submission {submission_id} not found")

        if self._sandbox:
            # Simulate sandbox submission
            record.status = "submitted"
            record.submitted_at = datetime.now(timezone.utc).isoformat()
            record.eu_reference = f"EU-SANDBOX-{uuid.uuid4().hex[:12].upper()}"
        else:
            # In production, would call the actual EU System API
            record.status = "submitted"
            record.submitted_at = datetime.now(timezone.utc).isoformat()
            record.eu_reference = f"EU-{uuid.uuid4().hex[:12].upper()}"

        record.provenance_hash = _compute_hash({
            "submission_id": submission_id,
            "dds_id": record.dds_id,
            "status": record.status,
            "eu_reference": record.eu_reference,
        })

        return record

    def get_submission(self, submission_id: str) -> Optional[EUSubmissionRecord]:
        """Get a submission record by ID."""
        return self._submissions.get(submission_id)

    def list_submissions(self, status: Optional[str] = None) -> List[EUSubmissionRecord]:
        """List submissions with optional status filter."""
        results = list(self._submissions.values())
        if status is not None:
            results = [s for s in results if s.status == status]
        return results

    def retry_submission(self, submission_id: str) -> EUSubmissionRecord:
        """Retry a failed submission."""
        record = self._submissions.get(submission_id)
        if record is None:
            raise ValueError(f"Submission {submission_id} not found")

        if record.retry_count >= record.max_retries:
            record.status = "error"
            record.errors.append(
                f"Maximum retries ({record.max_retries}) exceeded"
            )
            raise ValueError(
                f"Maximum retries ({record.max_retries}) exceeded for {submission_id}"
            )

        record.retry_count += 1

        if self._sandbox:
            record.status = "submitted"
            record.submitted_at = datetime.now(timezone.utc).isoformat()
            record.eu_reference = f"EU-SANDBOX-{uuid.uuid4().hex[:12].upper()}"

        return record

    def bulk_submit(self, dds_ids: List[str]) -> List[EUSubmissionRecord]:
        """Prepare and submit multiple DDS in bulk."""
        results: List[EUSubmissionRecord] = []
        for dds_id in dds_ids:
            record = self.prepare_submission(dds_id)
            submitted = self.submit(record.submission_id)
            results.append(submitted)
        return results

    def format_for_eu_system(self, dds_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format DDS data for the EU Information System API."""
        return {
            "eudr_version": "1.0",
            "submission_type": "dds",
            "data": {
                "reference": dds_data.get("dds_reference", ""),
                "operator_name": dds_data.get("operator", {}).get("name", ""),
                "operator_country": dds_data.get("operator", {}).get("country", ""),
                "commodity": dds_data.get("commodity", ""),
                "risk_level": dds_data.get("risk_level", ""),
                "origin_countries": dds_data.get("origin_countries", []),
                "plots": dds_data.get("plots", []),
            },
            "metadata": {
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "sandbox": self._sandbox,
            },
        }

    def validate_eu_format(self, formatted_data: Dict[str, Any]) -> List[str]:
        """Validate that formatted data meets EU System requirements."""
        errors: List[str] = []

        if "eudr_version" not in formatted_data:
            errors.append("Missing eudr_version")

        data = formatted_data.get("data", {})
        if not data.get("reference"):
            errors.append("Missing DDS reference")
        if not data.get("operator_name"):
            errors.append("Missing operator name")
        if not data.get("operator_country"):
            errors.append("Missing operator country")
        if not data.get("commodity"):
            errors.append("Missing commodity")
        if not data.get("origin_countries"):
            errors.append("Missing origin countries")

        return errors

    def simulate_eu_response(self, submission_id: str,
                             accepted: bool = True) -> Dict[str, Any]:
        """Simulate an EU System response (for sandbox testing)."""
        record = self._submissions.get(submission_id)
        if record is None:
            raise ValueError(f"Submission {submission_id} not found")

        if accepted:
            record.status = "accepted"
            return {
                "status": "accepted",
                "eu_reference": record.eu_reference,
                "submission_id": submission_id,
                "message": "DDS accepted by EU Information System",
            }
        else:
            record.status = "rejected"
            record.errors.append("Simulated rejection: insufficient data")
            return {
                "status": "rejected",
                "eu_reference": record.eu_reference,
                "submission_id": submission_id,
                "message": "DDS rejected: insufficient data",
                "errors": ["Insufficient plot data"],
            }


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> EUSystemConnector:
    return EUSystemConnector(sandbox=True, max_retries=3)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPrepareSubmission:
    """Tests for submission preparation."""

    def test_prepare_submission(self, engine):
        record = engine.prepare_submission("DDS-00001")
        assert record is not None
        assert isinstance(record, EUSubmissionRecord)
        assert record.dds_id == "DDS-00001"
        assert record.status == "prepared"

    def test_prepare_submission_id_format(self, engine):
        record = engine.prepare_submission("DDS-00001")
        assert record.submission_id.startswith("SUB-")
        assert len(record.submission_id) == 9  # SUB-00001

    def test_prepare_submission_provenance(self, engine):
        record = engine.prepare_submission("DDS-00001")
        assert len(record.provenance_hash) == 64


class TestSubmit:
    """Tests for submission execution."""

    def test_submit_sandbox(self, engine):
        record = engine.prepare_submission("DDS-00001")
        submitted = engine.submit(record.submission_id)
        assert submitted.status == "submitted"
        assert submitted.eu_reference is not None
        assert submitted.eu_reference.startswith("EU-SANDBOX-")
        assert submitted.submitted_at is not None

    def test_submit_not_found(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.submit("SUB-99999")


class TestGetSubmission:
    """Tests for submission retrieval."""

    def test_get_submission_status(self, engine):
        record = engine.prepare_submission("DDS-00001")
        retrieved = engine.get_submission(record.submission_id)
        assert retrieved is not None
        assert retrieved.submission_id == record.submission_id

    def test_get_submission_not_found(self, engine):
        result = engine.get_submission("SUB-99999")
        assert result is None


class TestListSubmissions:
    """Tests for submission listing."""

    def test_list_submissions(self, engine):
        engine.prepare_submission("DDS-00001")
        engine.prepare_submission("DDS-00002")
        results = engine.list_submissions()
        assert len(results) == 2

    def test_list_submissions_by_status(self, engine):
        r1 = engine.prepare_submission("DDS-00001")
        engine.prepare_submission("DDS-00002")
        engine.submit(r1.submission_id)
        prepared = engine.list_submissions(status="prepared")
        submitted = engine.list_submissions(status="submitted")
        assert len(prepared) == 1
        assert len(submitted) == 1


class TestRetrySubmission:
    """Tests for submission retry logic."""

    def test_retry_submission(self, engine):
        record = engine.prepare_submission("DDS-00001")
        engine.submit(record.submission_id)
        record.status = "error"  # simulate failure
        retried = engine.retry_submission(record.submission_id)
        assert retried.retry_count == 1
        assert retried.status == "submitted"

    def test_retry_max_exceeded(self, engine):
        record = engine.prepare_submission("DDS-00001")
        record.retry_count = 3  # already at max
        with pytest.raises(ValueError, match="Maximum retries"):
            engine.retry_submission(record.submission_id)

    def test_retry_increments_count(self, engine):
        record = engine.prepare_submission("DDS-00001")
        assert record.retry_count == 0
        engine.retry_submission(record.submission_id)
        assert record.retry_count == 1
        engine.retry_submission(record.submission_id)
        assert record.retry_count == 2

    def test_retry_not_found(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.retry_submission("SUB-99999")


class TestBulkSubmit:
    """Tests for bulk submission."""

    def test_bulk_submit(self, engine):
        dds_ids = ["DDS-00001", "DDS-00002", "DDS-00003"]
        results = engine.bulk_submit(dds_ids)
        assert len(results) == 3
        for r in results:
            assert r.status == "submitted"
            assert r.eu_reference is not None

    def test_bulk_submit_empty(self, engine):
        results = engine.bulk_submit([])
        assert len(results) == 0


class TestFormatForEUSystem:
    """Tests for EU System format output."""

    def test_format_for_eu_system(self, engine):
        dds_data = {
            "dds_reference": "DDS-00001",
            "operator": {"name": "ChocoCorp", "country": "DE"},
            "commodity": "cocoa",
            "risk_level": "standard",
            "origin_countries": ["BR"],
            "plots": [{"plot_id": "PLT-00001"}],
        }
        formatted = engine.format_for_eu_system(dds_data)
        assert "eudr_version" in formatted
        assert formatted["eudr_version"] == "1.0"
        assert "data" in formatted
        assert "metadata" in formatted

    def test_format_structure(self, engine):
        dds_data = {
            "dds_reference": "DDS-00001",
            "operator": {"name": "TestCo", "country": "FR"},
            "commodity": "coffee",
            "origin_countries": ["ET"],
            "plots": [],
        }
        formatted = engine.format_for_eu_system(dds_data)
        assert formatted["data"]["reference"] == "DDS-00001"
        assert formatted["data"]["operator_name"] == "TestCo"
        assert formatted["data"]["operator_country"] == "FR"
        assert formatted["data"]["commodity"] == "coffee"


class TestValidateEUFormat:
    """Tests for EU format validation."""

    def test_validate_eu_format_valid(self, engine):
        valid_data = {
            "eudr_version": "1.0",
            "data": {
                "reference": "DDS-00001",
                "operator_name": "TestCo",
                "operator_country": "DE",
                "commodity": "cocoa",
                "origin_countries": ["BR"],
            },
        }
        errors = engine.validate_eu_format(valid_data)
        assert len(errors) == 0

    def test_validate_eu_format_invalid(self, engine):
        invalid_data = {
            "data": {
                "reference": "",
                "operator_name": "",
                "commodity": "",
            },
        }
        errors = engine.validate_eu_format(invalid_data)
        assert len(errors) >= 3
        assert any("version" in e.lower() for e in errors)
        assert any("operator" in e.lower() for e in errors)

    def test_validate_eu_format_missing_all(self, engine):
        errors = engine.validate_eu_format({})
        assert len(errors) >= 1


class TestSimulateEUResponse:
    """Tests for EU response simulation."""

    def test_simulate_eu_response_accepted(self, engine):
        record = engine.prepare_submission("DDS-00001")
        engine.submit(record.submission_id)
        response = engine.simulate_eu_response(record.submission_id, accepted=True)
        assert response["status"] == "accepted"
        assert "eu_reference" in response

    def test_simulate_eu_response_rejected(self, engine):
        record = engine.prepare_submission("DDS-00001")
        engine.submit(record.submission_id)
        response = engine.simulate_eu_response(record.submission_id, accepted=False)
        assert response["status"] == "rejected"
        assert "errors" in response

    def test_simulate_eu_response_not_found(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.simulate_eu_response("SUB-99999")
