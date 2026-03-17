# -*- coding: utf-8 -*-
"""
Unit tests for CrossRegulationEvidenceEngine - PACK-009 Engine 8

Tests unified evidence repository management across CSRD, CBAM,
EU Taxonomy, and EUDR. Validates evidence registration, requirement
mapping, evidence reuse detection, completeness checking, gap
identification, expiring evidence tracking, cost savings calculation,
and provenance hashing.

Coverage target: 85%+
Test count: 15

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, file_path: Path):
    """Import a module from a file path (supports hyphenated directories).

    Registers the module in sys.modules so that pydantic can resolve
    forward-referenced annotations created by ``from __future__ import
    annotations``.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Load the engine module
# ---------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent
_ENGINE_PATH = _PACK_DIR / "engines" / "cross_regulation_evidence_engine.py"

try:
    _mod = _import_from_path("cross_regulation_evidence_engine", _ENGINE_PATH)
    CrossRegulationEvidenceEngine = _mod.CrossRegulationEvidenceEngine
    EvidenceConfig = _mod.EvidenceConfig
    EvidenceResult = _mod.EvidenceResult
    EvidenceItem = _mod.EvidenceItem
    EvidenceMapping = _mod.EvidenceMapping
    EvidenceGap = _mod.EvidenceGap
    ExpiringEvidence = _mod.ExpiringEvidence
    ReuseSavings = _mod.ReuseSavings
    CoverageMatrix = _mod.CoverageMatrix
    EvidenceType = _mod.EvidenceType
    EvidenceStatus = _mod.EvidenceStatus
    RegulationType = _mod.RegulationType
    EVIDENCE_REQUIREMENTS = _mod.EVIDENCE_REQUIREMENTS
    EVIDENCE_REUSE_MAP = _mod.EVIDENCE_REUSE_MAP
    _ENGINE_AVAILABLE = True
except Exception as exc:
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Skip decorator
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not _ENGINE_AVAILABLE,
    reason=f"CrossRegulationEvidenceEngine could not be imported: "
           f"{_ENGINE_IMPORT_ERROR if not _ENGINE_AVAILABLE else ''}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence_item(
    title: str = "Test Evidence Document",
    evidence_type: str = "AUDIT_REPORT",
    regulations: List[str] = None,
    expiry_date: str = "",
    status: str = "ACTIVE",
) -> "EvidenceItem":
    """Create an EvidenceItem instance with sensible defaults."""
    return EvidenceItem(
        title=title,
        evidence_type=evidence_type,
        regulations=regulations or ["CSRD", "CBAM"],
        expiry_date=expiry_date,
        status=status,
    )


def _assert_provenance_hash(hash_str: str) -> None:
    """Assert that a string is a valid SHA-256 hex digest."""
    assert isinstance(hash_str, str)
    assert len(hash_str) == 64
    assert re.match(r"^[0-9a-f]{64}$", hash_str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossRegulationEvidenceEngine:
    """Test suite for CrossRegulationEvidenceEngine."""

    def test_engine_instantiation(self):
        """Engine can be instantiated with default and custom config."""
        engine = CrossRegulationEvidenceEngine()
        assert engine.config is not None
        assert engine.config.require_hash is True
        assert engine.config.max_evidence_age_days == 365
        assert engine.config.enable_reuse_tracking is True
        assert engine.config.expiry_warning_days == 60

        custom = EvidenceConfig(
            require_hash=False,
            max_evidence_age_days=180,
            expiry_warning_days=30,
        )
        engine2 = CrossRegulationEvidenceEngine(custom)
        assert engine2.config.require_hash is False
        assert engine2.config.max_evidence_age_days == 180
        assert engine2.config.expiry_warning_days == 30

    def test_register_evidence(self):
        """register_evidence stores items and generates hashes."""
        engine = CrossRegulationEvidenceEngine()
        item = _make_evidence_item(
            title="GHG Emissions Report 2025",
            evidence_type="AUDIT_REPORT",
            regulations=["CSRD", "CBAM"],
        )
        registered = engine.register_evidence(item)

        assert registered.provenance_hash != ""
        _assert_provenance_hash(registered.provenance_hash)
        assert registered.file_hash != ""
        assert registered.upload_date != ""
        assert len(registered.requirements_satisfied) > 0

    def test_map_to_requirements(self):
        """map_to_requirements maps evidence type to matching requirements."""
        engine = CrossRegulationEvidenceEngine()
        item = _make_evidence_item(
            title="Emission Calculation",
            evidence_type="EMISSION_CALCULATION",
            regulations=["CSRD", "CBAM"],
        )
        mappings = engine.map_to_requirements(item)

        assert len(mappings) > 0
        for m in mappings:
            assert isinstance(m, EvidenceMapping)
            assert m.regulation in ("CSRD", "CBAM")
            assert m.requirement_id != ""
            assert 0.0 < m.coverage_pct <= 100.0

    def test_find_reusable_evidence(self):
        """find_reusable_evidence detects items serving multiple regulations."""
        engine = CrossRegulationEvidenceEngine()
        # Register audit report for CBAM and EUDR -- AUDIT_REPORT type
        # maps to requirements in both CBAM (VER-01, NCA-01) and EUDR (NCA-01)
        item = _make_evidence_item(
            title="Cross-Reg Audit Report",
            evidence_type="AUDIT_REPORT",
            regulations=["CBAM", "EUDR"],
        )
        engine.register_evidence(item)

        reusable = engine.find_reusable_evidence()
        # The audit report maps to requirements in both CBAM and EUDR
        assert len(reusable) >= 1
        for eid, regs in reusable.items():
            assert len(regs) >= 2

    def test_check_completeness(self):
        """check_completeness returns an EvidenceResult with coverage matrix."""
        engine = CrossRegulationEvidenceEngine()
        engine.register_evidence(_make_evidence_item(
            title="Policy Doc",
            evidence_type="POLICY_DOCUMENT",
            regulations=["CSRD", "EU_TAXONOMY"],
        ))
        result = engine.check_completeness()

        assert isinstance(result, EvidenceResult)
        assert len(result.coverage_matrix) > 0
        assert result.processing_time_ms >= 0.0
        for cm in result.coverage_matrix:
            assert isinstance(cm, CoverageMatrix)
            assert cm.total_requirements > 0
            assert 0.0 <= cm.coverage_pct <= 100.0

    def test_identify_gaps(self):
        """identify_gaps finds requirements without sufficient evidence."""
        engine = CrossRegulationEvidenceEngine()
        # Register one item - many requirements will remain uncovered
        engine.register_evidence(_make_evidence_item(
            title="Single Audit",
            evidence_type="AUDIT_REPORT",
            regulations=["CSRD"],
        ))
        gaps = engine.identify_gaps()

        assert len(gaps) > 0
        for gap in gaps:
            assert isinstance(gap, EvidenceGap)
            assert gap.regulation in ("CSRD", "CBAM", "EU_TAXONOMY", "EUDR")
            assert gap.requirement_id != ""
            assert len(gap.missing_types) > 0

    def test_get_expiring_evidence(self):
        """get_expiring_evidence detects items within the warning window."""
        engine = CrossRegulationEvidenceEngine(
            EvidenceConfig(expiry_warning_days=90)
        )
        # Item expiring in 30 days
        soon_date = (date.today() + timedelta(days=30)).isoformat()
        engine.register_evidence(_make_evidence_item(
            title="Expiring Certificate",
            evidence_type="CERTIFICATE",
            regulations=["CBAM"],
            expiry_date=soon_date,
        ))
        # Item expiring in 365 days (should NOT be flagged)
        far_date = (date.today() + timedelta(days=365)).isoformat()
        engine.register_evidence(_make_evidence_item(
            title="Far Future Certificate",
            evidence_type="CERTIFICATE",
            regulations=["CBAM"],
            expiry_date=far_date,
        ))

        expiring = engine.get_expiring_evidence()
        assert len(expiring) >= 1
        # Only the soon-expiring item should appear
        assert any(e.title == "Expiring Certificate" for e in expiring)
        assert not any(e.title == "Far Future Certificate" for e in expiring)

    def test_calculate_reuse_savings(self):
        """calculate_reuse_savings returns cost savings from evidence reuse."""
        engine = CrossRegulationEvidenceEngine()
        # Register several items for reuse
        engine.register_evidence(_make_evidence_item(
            title="Shared Audit Report",
            evidence_type="AUDIT_REPORT",
            regulations=["CSRD", "CBAM", "EUDR"],
        ))
        engine.register_evidence(_make_evidence_item(
            title="Shared Policy",
            evidence_type="POLICY_DOCUMENT",
            regulations=["CSRD", "EU_TAXONOMY", "EUDR"],
        ))

        savings = engine.calculate_reuse_savings()

        assert isinstance(savings, ReuseSavings)
        assert savings.total_evidence_items == 2
        assert savings.without_reuse_cost_eur > 0.0
        assert savings.with_reuse_cost_eur >= 0.0
        assert savings.savings_eur >= 0.0
        assert 0.0 <= savings.savings_pct <= 100.0

    def test_compute_hash(self):
        """compute_hash returns a deterministic SHA-256 hex string."""
        engine = CrossRegulationEvidenceEngine()
        hash1 = engine.compute_hash("test content")
        hash2 = engine.compute_hash("test content")
        hash3 = engine.compute_hash("different content")

        assert hash1 == hash2  # deterministic
        assert hash1 != hash3  # different content => different hash
        _assert_provenance_hash(hash1)

    def test_evidence_requirements_populated(self):
        """EVIDENCE_REQUIREMENTS contains entries for all 4 regulations."""
        assert "CSRD" in EVIDENCE_REQUIREMENTS
        assert "CBAM" in EVIDENCE_REQUIREMENTS
        assert "EU_TAXONOMY" in EVIDENCE_REQUIREMENTS
        assert "EUDR" in EVIDENCE_REQUIREMENTS
        for reg, reqs in EVIDENCE_REQUIREMENTS.items():
            assert len(reqs) >= 10, f"{reg} has fewer than 10 requirements"
            for req_id, req_types in reqs.items():
                assert isinstance(req_types, list)
                assert len(req_types) >= 1

    def test_evidence_reuse_map_populated(self):
        """EVIDENCE_REUSE_MAP contains cross-regulation reuse entries."""
        assert len(EVIDENCE_REUSE_MAP) >= 5
        for key, entry in EVIDENCE_REUSE_MAP.items():
            assert "description" in entry
            assert "satisfies" in entry
            assert len(entry["satisfies"]) >= 2

    def test_result_has_provenance_hash(self):
        """The completeness result carries a valid SHA-256 provenance hash."""
        engine = CrossRegulationEvidenceEngine()
        engine.register_evidence(_make_evidence_item())
        result = engine.check_completeness()
        _assert_provenance_hash(result.provenance_hash)

    def test_coverage_matrix(self):
        """Coverage matrix shows per-regulation coverage after registration."""
        engine = CrossRegulationEvidenceEngine()
        # Register items covering some CSRD requirements
        engine.register_evidence(_make_evidence_item(
            title="CSRD Audit",
            evidence_type="AUDIT_REPORT",
            regulations=["CSRD"],
        ))
        engine.register_evidence(_make_evidence_item(
            title="CSRD Emissions",
            evidence_type="EMISSION_CALCULATION",
            regulations=["CSRD"],
        ))
        result = engine.check_completeness()

        csrd_matrix = [cm for cm in result.coverage_matrix if cm.regulation == "CSRD"]
        assert len(csrd_matrix) == 1
        cm = csrd_matrix[0]
        assert cm.total_requirements == len(EVIDENCE_REQUIREMENTS["CSRD"])
        assert cm.covered_requirements >= 0
        assert len(cm.requirement_details) > 0

    def test_evidence_types_enum(self):
        """EvidenceType enum contains expected evidence categories."""
        expected_types = [
            "AUDIT_REPORT", "POLICY_DOCUMENT", "DATA_EXTRACT",
            "CERTIFICATE", "VERIFICATION_STATEMENT", "RISK_ASSESSMENT",
            "DUE_DILIGENCE_REPORT", "EMISSION_CALCULATION",
        ]
        available = {t.value for t in EvidenceType}
        for et in expected_types:
            assert et in available, f"Missing evidence type: {et}"

    def test_empty_evidence_handling(self):
        """Engine handles empty evidence repository without error."""
        engine = CrossRegulationEvidenceEngine()
        result = engine.check_completeness()

        assert isinstance(result, EvidenceResult)
        assert len(result.items) == 0
        assert len(result.mappings) == 0
        assert result.reuse_count == 0
        # All requirements should show as gaps
        assert len(result.gaps) > 0
        # Savings should still be calculable
        if result.savings is not None:
            assert result.savings.total_evidence_items == 0
        _assert_provenance_hash(result.provenance_hash)
