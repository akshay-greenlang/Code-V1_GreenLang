# -*- coding: utf-8 -*-
"""
Unit Tests for VerificationEngine (AGENT-FOUND-005)

Tests citation verification, batch verification, expiration checks,
supersession detection, hash integrity, DOI validation, required fields,
and verification history tracking.

Coverage target: 85%+ of verification.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline VerificationEngine mirroring greenlang/citations/verification.py
# ---------------------------------------------------------------------------


class CitationRecord:
    """Minimal citation record for verification testing."""

    def __init__(
        self,
        citation_id: str = "",
        citation_type: str = "emission_factor",
        source_authority: str = "defra",
        title: str = "",
        version: Optional[str] = None,
        doi: Optional[str] = None,
        effective_date: str = "2024-01-01",
        expiration_date: Optional[str] = None,
        verification_status: str = "unverified",
        superseded_by: Optional[str] = None,
        content_hash: Optional[str] = None,
        key_values: Optional[Dict[str, Any]] = None,
    ):
        self.citation_id = citation_id or str(uuid.uuid4())
        self.citation_type = citation_type
        self.source_authority = source_authority
        self.title = title
        self.version = version
        self.doi = doi
        self.effective_date = effective_date
        self.expiration_date = expiration_date
        self.verification_status = verification_status
        self.superseded_by = superseded_by
        self.content_hash = content_hash
        self.key_values = key_values or {}

    def calculate_content_hash(self) -> str:
        content = {
            "citation_type": self.citation_type,
            "source_authority": self.source_authority,
            "title": self.title,
            "effective_date": self.effective_date,
            "key_values": self.key_values,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class VerificationRecord:
    def __init__(self, record_id=None, citation_id="", old_status="",
                 new_status="", reason="", verified_by="system", timestamp=None):
        self.record_id = record_id or str(uuid.uuid4())
        self.citation_id = citation_id
        self.old_status = old_status
        self.new_status = new_status
        self.reason = reason
        self.verified_by = verified_by
        self.timestamp = timestamp or datetime.utcnow().isoformat()


class VerificationEngine:
    """Verifies citations for validity, expiration, and integrity."""

    def __init__(self, reference_date: Optional[str] = None):
        self._reference_date = reference_date or date.today().isoformat()
        self._history: List[VerificationRecord] = []

    def verify_citation(self, citation: CitationRecord,
                        verified_by: str = "system") -> str:
        """Verify a single citation. Returns new status."""
        old_status = citation.verification_status
        new_status = self._do_verify(citation)
        citation.verification_status = new_status

        self._history.append(VerificationRecord(
            citation_id=citation.citation_id,
            old_status=old_status,
            new_status=new_status,
            reason=self._get_reason(citation, new_status),
            verified_by=verified_by,
        ))
        return new_status

    def verify_batch(self, citations: List[CitationRecord],
                     verified_by: str = "system") -> Dict[str, str]:
        """Verify multiple citations. Returns dict of id -> status."""
        results = {}
        for c in citations:
            results[c.citation_id] = self.verify_citation(c, verified_by)
        return results

    def check_expiration(self, citation: CitationRecord) -> bool:
        """Check if citation is expired. Returns True if expired."""
        if not citation.expiration_date:
            return False
        return citation.expiration_date < self._reference_date

    def check_supersession(self, citation: CitationRecord) -> bool:
        """Check if citation is superseded. Returns True if superseded."""
        return citation.superseded_by is not None

    def check_hash_integrity(self, citation: CitationRecord) -> bool:
        """Check if content hash matches. Returns True if intact."""
        if not citation.content_hash:
            return True  # No hash to verify
        current_hash = citation.calculate_content_hash()
        return current_hash == citation.content_hash

    def get_verification_history(
        self,
        citation_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[VerificationRecord]:
        results = list(self._history)
        if citation_id:
            results = [r for r in results if r.citation_id == citation_id]
        if limit and limit > 0:
            results = results[-limit:]
        return results

    def _do_verify(self, citation: CitationRecord) -> str:
        """Core verification logic."""
        # Check expiration first
        if self.check_expiration(citation):
            return "expired"

        # Check supersession
        if self.check_supersession(citation):
            return "superseded"

        # Check hash integrity
        if not self.check_hash_integrity(citation):
            return "invalid"

        # Check source-specific requirements
        if citation.source_authority in ("defra", "epa", "ecoinvent"):
            if not citation.version:
                return "unverified"

        # Scientific citations need DOI
        if citation.citation_type == "scientific":
            if not citation.doi:
                return "unverified"

        return "verified"

    def _get_reason(self, citation: CitationRecord, status: str) -> str:
        if status == "expired":
            return f"Citation expired on {citation.expiration_date}"
        if status == "superseded":
            return f"Superseded by {citation.superseded_by}"
        if status == "invalid":
            return "Content hash mismatch detected"
        if status == "unverified":
            if citation.citation_type == "scientific" and not citation.doi:
                return "Scientific citation missing DOI"
            if citation.source_authority in ("defra", "epa", "ecoinvent") and not citation.version:
                return f"{citation.source_authority.upper()} citation missing version"
        if status == "verified":
            return "All verification checks passed"
        return ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return VerificationEngine(reference_date="2025-06-15")


@pytest.fixture
def valid_citation():
    c = CitationRecord(
        citation_id="defra-2024",
        citation_type="emission_factor",
        source_authority="defra",
        title="DEFRA 2024 GHG Factors",
        version="2024",
        effective_date="2024-01-01",
        expiration_date="2025-12-31",
        key_values={"diesel_ef": 2.68},
    )
    c.content_hash = c.calculate_content_hash()
    return c


@pytest.fixture
def expired_citation():
    c = CitationRecord(
        citation_id="defra-2020",
        source_authority="defra",
        version="2020",
        effective_date="2020-01-01",
        expiration_date="2021-12-31",
    )
    c.content_hash = c.calculate_content_hash()
    return c


@pytest.fixture
def superseded_citation():
    return CitationRecord(
        citation_id="defra-2023",
        source_authority="defra",
        version="2023",
        effective_date="2023-01-01",
        superseded_by="defra-2024",
    )


@pytest.fixture
def scientific_citation_no_doi():
    return CitationRecord(
        citation_id="paper-001",
        citation_type="scientific",
        source_authority="ipcc",
        title="Climate Research Paper",
        effective_date="2022-01-01",
    )


@pytest.fixture
def scientific_citation_with_doi():
    c = CitationRecord(
        citation_id="paper-002",
        citation_type="scientific",
        source_authority="ipcc",
        title="Climate Research Paper",
        doi="10.1017/9781009157926",
        effective_date="2022-01-01",
    )
    c.content_hash = c.calculate_content_hash()
    return c


# ===========================================================================
# Test Classes
# ===========================================================================


class TestVerificationEngineVerify:
    """Test verify_citation() for valid citations."""

    def test_verify_valid_citation(self, engine, valid_citation):
        status = engine.verify_citation(valid_citation)
        assert status == "verified"

    def test_verify_updates_status(self, engine, valid_citation):
        engine.verify_citation(valid_citation)
        assert valid_citation.verification_status == "verified"

    def test_verify_records_history(self, engine, valid_citation):
        engine.verify_citation(valid_citation)
        history = engine.get_verification_history()
        assert len(history) == 1
        assert history[0].new_status == "verified"

    def test_verify_with_verified_by(self, engine, valid_citation):
        engine.verify_citation(valid_citation, verified_by="analyst1")
        history = engine.get_verification_history()
        assert history[0].verified_by == "analyst1"

    def test_verify_regulatory_citation(self, engine):
        c = CitationRecord(
            citation_type="regulatory",
            source_authority="eu_commission",
            title="CSRD Directive",
            effective_date="2024-01-01",
        )
        status = engine.verify_citation(c)
        assert status == "verified"

    def test_verify_methodology_citation(self, engine):
        c = CitationRecord(
            citation_type="methodology",
            source_authority="ghg_protocol",
            title="GHG Protocol",
            effective_date="2015-01-01",
        )
        status = engine.verify_citation(c)
        assert status == "verified"


class TestVerificationEngineBatch:
    """Test verify_batch() operation."""

    def test_batch_verify(self, engine, valid_citation, expired_citation):
        results = engine.verify_batch([valid_citation, expired_citation])
        assert results[valid_citation.citation_id] == "verified"
        assert results[expired_citation.citation_id] == "expired"

    def test_batch_verify_all_valid(self, engine):
        citations = [
            CitationRecord(citation_id=f"cid-{i}", citation_type="regulatory",
                           source_authority="eu_commission",
                           effective_date="2024-01-01")
            for i in range(5)
        ]
        results = engine.verify_batch(citations)
        assert all(s == "verified" for s in results.values())
        assert len(results) == 5

    def test_batch_verify_records_history(self, engine, valid_citation, expired_citation):
        engine.verify_batch([valid_citation, expired_citation])
        history = engine.get_verification_history()
        assert len(history) == 2

    def test_batch_verify_empty_list(self, engine):
        results = engine.verify_batch([])
        assert results == {}


class TestVerificationExpiration:
    """Test expiration detection."""

    def test_expired_detected(self, engine, expired_citation):
        assert engine.check_expiration(expired_citation) is True

    def test_valid_not_expired(self, engine, valid_citation):
        assert engine.check_expiration(valid_citation) is False

    def test_no_expiration_date(self, engine):
        c = CitationRecord(effective_date="2020-01-01")
        assert engine.check_expiration(c) is False

    def test_expired_citation_gets_expired_status(self, engine, expired_citation):
        status = engine.verify_citation(expired_citation)
        assert status == "expired"

    def test_expiration_boundary(self):
        engine = VerificationEngine(reference_date="2025-12-31")
        c = CitationRecord(effective_date="2024-01-01", expiration_date="2025-12-31")
        # expiration_date < reference_date is False (equal)
        assert engine.check_expiration(c) is False


class TestVerificationSupersession:
    """Test supersession detection."""

    def test_superseded_detected(self, engine, superseded_citation):
        assert engine.check_supersession(superseded_citation) is True

    def test_not_superseded(self, engine, valid_citation):
        assert engine.check_supersession(valid_citation) is False

    def test_superseded_gets_superseded_status(self, engine, superseded_citation):
        status = engine.verify_citation(superseded_citation)
        assert status == "superseded"

    def test_superseded_reason_includes_new_id(self, engine, superseded_citation):
        engine.verify_citation(superseded_citation)
        history = engine.get_verification_history()
        assert "defra-2024" in history[0].reason


class TestVerificationHashIntegrity:
    """Test hash integrity checking."""

    def test_valid_hash(self, engine, valid_citation):
        assert engine.check_hash_integrity(valid_citation) is True

    def test_no_hash_passes(self, engine):
        c = CitationRecord(content_hash=None)
        assert engine.check_hash_integrity(c) is True

    def test_tampered_hash_detected(self, engine):
        c = CitationRecord(
            citation_type="emission_factor",
            source_authority="defra",
            title="DEFRA Factors",
            version="2024",
            effective_date="2024-01-01",
            key_values={"ef": 2.68},
        )
        c.content_hash = c.calculate_content_hash()
        # Tamper with data
        c.key_values["ef"] = 9999.99
        assert engine.check_hash_integrity(c) is False

    def test_tampered_hash_gets_invalid_status(self, engine):
        c = CitationRecord(
            source_authority="ghg_protocol",
            version="2024",
            effective_date="2024-01-01",
            key_values={"ef": 2.68},
        )
        c.content_hash = c.calculate_content_hash()
        c.key_values["ef"] = 0.0  # tamper
        status = engine.verify_citation(c)
        assert status == "invalid"


class TestVerificationDOI:
    """Test DOI format requirements for scientific citations."""

    def test_scientific_without_doi_unverified(self, engine, scientific_citation_no_doi):
        status = engine.verify_citation(scientific_citation_no_doi)
        assert status == "unverified"

    def test_scientific_with_doi_verified(self, engine, scientific_citation_with_doi):
        status = engine.verify_citation(scientific_citation_with_doi)
        assert status == "verified"

    def test_non_scientific_no_doi_ok(self, engine):
        c = CitationRecord(
            citation_type="regulatory",
            source_authority="eu_commission",
            effective_date="2024-01-01",
        )
        status = engine.verify_citation(c)
        assert status == "verified"


class TestVerificationRequiredFields:
    """Test source-authority-specific field requirements."""

    def test_defra_needs_version(self, engine):
        c = CitationRecord(source_authority="defra", effective_date="2024-01-01")
        status = engine.verify_citation(c)
        assert status == "unverified"

    def test_defra_with_version_verified(self, engine):
        c = CitationRecord(source_authority="defra", version="2024",
                           effective_date="2024-01-01")
        c.content_hash = c.calculate_content_hash()
        status = engine.verify_citation(c)
        assert status == "verified"

    def test_epa_needs_version(self, engine):
        c = CitationRecord(source_authority="epa", effective_date="2024-01-01")
        status = engine.verify_citation(c)
        assert status == "unverified"

    def test_ecoinvent_needs_version(self, engine):
        c = CitationRecord(source_authority="ecoinvent", effective_date="2024-01-01")
        status = engine.verify_citation(c)
        assert status == "unverified"

    def test_ipcc_no_version_ok(self, engine):
        c = CitationRecord(
            citation_type="regulatory",
            source_authority="ipcc",
            effective_date="2020-01-01",
        )
        status = engine.verify_citation(c)
        assert status == "verified"

    def test_internal_no_version_ok(self, engine):
        c = CitationRecord(
            source_authority="internal",
            effective_date="2024-01-01",
        )
        status = engine.verify_citation(c)
        assert status == "verified"


class TestVerificationHistory:
    """Test verification audit trail."""

    def test_history_grows(self, engine, valid_citation):
        engine.verify_citation(valid_citation)
        engine.verify_citation(valid_citation)
        history = engine.get_verification_history()
        assert len(history) == 2

    def test_history_filter_by_citation(self, engine, valid_citation, expired_citation):
        engine.verify_citation(valid_citation)
        engine.verify_citation(expired_citation)
        history = engine.get_verification_history(citation_id=valid_citation.citation_id)
        assert len(history) == 1

    def test_history_limit(self, engine):
        for i in range(10):
            c = CitationRecord(citation_id=f"cid-{i}", citation_type="regulatory",
                               source_authority="eu_commission", effective_date="2024-01-01")
            engine.verify_citation(c)
        history = engine.get_verification_history(limit=3)
        assert len(history) == 3

    def test_history_empty(self, engine):
        assert engine.get_verification_history() == []

    def test_history_records_old_and_new_status(self, engine, valid_citation):
        engine.verify_citation(valid_citation)
        record = engine.get_verification_history()[0]
        assert record.old_status == "unverified"
        assert record.new_status == "verified"

    def test_history_has_reason(self, engine, expired_citation):
        engine.verify_citation(expired_citation)
        record = engine.get_verification_history()[0]
        assert "expired" in record.reason.lower()

    def test_history_has_timestamp(self, engine, valid_citation):
        engine.verify_citation(valid_citation)
        record = engine.get_verification_history()[0]
        assert record.timestamp is not None


class TestVerificationPriorityOrder:
    """Test verification checks priority: expired > superseded > invalid > unverified."""

    def test_expired_takes_priority_over_superseded(self, engine):
        c = CitationRecord(
            source_authority="defra",
            effective_date="2020-01-01",
            expiration_date="2021-01-01",
            superseded_by="newer-id",
        )
        status = engine.verify_citation(c)
        assert status == "expired"

    def test_superseded_takes_priority_over_missing_fields(self, engine):
        c = CitationRecord(
            source_authority="defra",
            effective_date="2024-01-01",
            superseded_by="newer-id",
            # Missing version, but supersession takes priority
        )
        status = engine.verify_citation(c)
        assert status == "superseded"
