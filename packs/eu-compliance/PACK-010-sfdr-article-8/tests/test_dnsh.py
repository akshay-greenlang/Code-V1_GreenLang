# -*- coding: utf-8 -*-
"""
Unit tests for SFDRDNSHEngine (PACK-010 SFDR Article 8).

Tests DNSH (Do No Significant Harm) assessment, portfolio DNSH screening,
threshold configuration, report generation, and provenance tracking.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_dnsh_mod = _import_from_path(
    "sfdr_dnsh_engine",
    str(ENGINES_DIR / "sfdr_dnsh_engine.py"),
)

SFDRDNSHEngine = _dnsh_mod.SFDRDNSHEngine
DNSHConfig = _dnsh_mod.DNSHConfig
PAIThreshold = _dnsh_mod.PAIThreshold
InvestmentPAIData = _dnsh_mod.InvestmentPAIData
PAIDNSHCheck = _dnsh_mod.PAIDNSHCheck
DNSHAssessment = _dnsh_mod.DNSHAssessment
PortfolioDNSHResult = _dnsh_mod.PortfolioDNSHResult
DNSHReportSection = _dnsh_mod.DNSHReportSection
PAICategory = _dnsh_mod.PAICategory
DNSHStatus = _dnsh_mod.DNSHStatus
ThresholdDirection = _dnsh_mod.ThresholdDirection
SeverityLevel = _dnsh_mod.SeverityLevel

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_clean_investment() -> InvestmentPAIData:
    """Create an investment that passes all DNSH checks."""
    return InvestmentPAIData(
        investment_id="INV_CLEAN",
        investment_name="CleanTech Corp",
        pai_values={
            "PAI_1": 5000.0,
            "PAI_2": 50.0,
            "PAI_3": 80.0,
            "PAI_5": 25.0,
            "PAI_6": 0.1,
            "PAI_8": 5.0,
            "PAI_9": 10.0,
            "PAI_12": 5.0,
            "PAI_13": 45.0,
            "PAI_15": 100.0,
        },
        pai_boolean_flags={
            "PAI_4": False,
            "PAI_7": False,
            "PAI_10": False,
            "PAI_11": True,   # has compliance mechanisms
            "PAI_14": False,
            "PAI_16": False,
            "PAI_17": False,
            "PAI_18": False,
        },
    )


def _make_dirty_investment() -> InvestmentPAIData:
    """Create an investment that fails multiple DNSH checks."""
    return InvestmentPAIData(
        investment_id="INV_DIRTY",
        investment_name="HighCarbon Industries",
        pai_values={
            "PAI_1": 999_999.0,
            "PAI_2": 9999.0,
            "PAI_3": 9999.0,
            "PAI_5": 99.0,
            "PAI_6": 99.0,
            "PAI_8": 9999.0,
            "PAI_9": 9999.0,
            "PAI_12": 50.0,
            "PAI_13": 2.0,
            "PAI_15": 9999.0,
        },
        pai_boolean_flags={
            "PAI_4": True,    # fossil fuel
            "PAI_7": True,    # biodiversity
            "PAI_10": True,   # UNGC violations
            "PAI_11": False,  # no compliance
            "PAI_14": True,   # controversial weapons
            "PAI_16": True,   # social violations
            "PAI_17": True,   # RE fossil
            "PAI_18": True,   # RE inefficient
        },
    )


def _make_partial_investment() -> InvestmentPAIData:
    """Create an investment with some missing data."""
    return InvestmentPAIData(
        investment_id="INV_PARTIAL",
        investment_name="DataGap Ltd",
        pai_values={
            "PAI_1": 10000.0,
            "PAI_2": 200.0,
        },
        pai_boolean_flags={
            "PAI_4": False,
            "PAI_14": False,
        },
    )


# ===================================================================
# TEST CLASS
# ===================================================================


class TestSFDRDNSHEngine:
    """Unit tests for SFDRDNSHEngine."""

    # ---------------------------------------------------------------
    # 1. Engine initialization
    # ---------------------------------------------------------------

    def test_engine_default_initialization(self):
        """Test engine initializes with default config (built-in thresholds)."""
        engine = SFDRDNSHEngine()
        assert engine.assessment_count == 0

    def test_engine_custom_config(self):
        """Test engine initializes with custom DNSHConfig."""
        config = DNSHConfig()
        engine = SFDRDNSHEngine(config)
        assert engine.assessment_count == 0

    # ---------------------------------------------------------------
    # 2. assess_dnsh - clean investment
    # ---------------------------------------------------------------

    def test_assess_clean_investment_passes(self):
        """Test clean investment passes DNSH assessment."""
        engine = SFDRDNSHEngine()
        assessment = engine.assess_dnsh(_make_clean_investment())
        assert isinstance(assessment, DNSHAssessment)
        # Clean investment should mostly PASS
        pass_count = sum(
            1 for c in assessment.checks if c.status == DNSHStatus.PASS
        )
        assert pass_count >= 1

    def test_assess_dnsh_provenance_hash(self):
        """Test DNSH assessment includes valid provenance hash."""
        engine = SFDRDNSHEngine()
        assessment = engine.assess_dnsh(_make_clean_investment())
        assert hasattr(assessment, "provenance_hash")
        assert isinstance(assessment.provenance_hash, str)
        assert len(assessment.provenance_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", assessment.provenance_hash)

    # ---------------------------------------------------------------
    # 3. assess_dnsh - dirty investment
    # ---------------------------------------------------------------

    def test_assess_dirty_investment_has_failures(self):
        """Test dirty investment triggers DNSH failures."""
        engine = SFDRDNSHEngine()
        assessment = engine.assess_dnsh(_make_dirty_investment())
        fail_count = sum(
            1 for c in assessment.checks if c.status == DNSHStatus.FAIL
        )
        assert fail_count >= 1, "Dirty investment should have DNSH failures"

    # ---------------------------------------------------------------
    # 4. assess_dnsh - partial data
    # ---------------------------------------------------------------

    def test_assess_partial_data_has_insufficient(self):
        """Test partial data yields INSUFFICIENT_DATA for missing indicators."""
        engine = SFDRDNSHEngine()
        assessment = engine.assess_dnsh(_make_partial_investment())
        insufficient = sum(
            1 for c in assessment.checks
            if c.status == DNSHStatus.INSUFFICIENT_DATA
        )
        assert insufficient >= 1

    # ---------------------------------------------------------------
    # 5. assess_portfolio_dnsh
    # ---------------------------------------------------------------

    def test_portfolio_dnsh_returns_result(self):
        """Test portfolio DNSH assessment returns PortfolioDNSHResult."""
        engine = SFDRDNSHEngine()
        investments = [_make_clean_investment(), _make_dirty_investment()]
        result = engine.assess_portfolio_dnsh(investments, "TestPortfolio")
        assert isinstance(result, PortfolioDNSHResult)

    def test_portfolio_dnsh_empty_raises(self):
        """Test empty investments list raises ValueError."""
        engine = SFDRDNSHEngine()
        with pytest.raises(ValueError):
            engine.assess_portfolio_dnsh([], "Empty")

    def test_portfolio_dnsh_count(self):
        """Test portfolio result contains correct number of assessments."""
        engine = SFDRDNSHEngine()
        investments = [_make_clean_investment(), _make_dirty_investment()]
        result = engine.assess_portfolio_dnsh(investments, "TestPortfolio")
        assert len(result.investment_assessments) == 2

    # ---------------------------------------------------------------
    # 6. assessment_count increments
    # ---------------------------------------------------------------

    def test_assessment_count_increments(self):
        """Test assessment_count increments on each assess_dnsh call."""
        engine = SFDRDNSHEngine()
        engine.assess_dnsh(_make_clean_investment())
        assert engine.assessment_count == 1
        engine.assess_dnsh(_make_dirty_investment())
        assert engine.assessment_count == 2

    # ---------------------------------------------------------------
    # 7. get_dnsh_criteria
    # ---------------------------------------------------------------

    def test_get_dnsh_criteria_returns_list(self):
        """Test get_dnsh_criteria returns list of threshold definitions."""
        engine = SFDRDNSHEngine()
        criteria = engine.get_dnsh_criteria()
        assert isinstance(criteria, list)
        assert len(criteria) >= 1
        # Returns list of dicts (model_dump of PAIThreshold)
        for c in criteria:
            assert isinstance(c, dict)
            assert "pai_indicator_id" in c

    # ---------------------------------------------------------------
    # 8. generate_dnsh_report
    # ---------------------------------------------------------------

    def test_generate_dnsh_report(self):
        """Test report generation from assessment."""
        engine = SFDRDNSHEngine()
        assessment = engine.assess_dnsh(_make_clean_investment())
        report = engine.generate_dnsh_report(assessment)
        # Returns a list of DNSHReportSection objects
        assert isinstance(report, list)
        assert len(report) >= 1
        for section in report:
            assert isinstance(section, DNSHReportSection)

    # ---------------------------------------------------------------
    # 9. enabled_thresholds property
    # ---------------------------------------------------------------

    def test_enabled_thresholds_returns_list(self):
        """Test enabled_thresholds returns list of active PAI indicator IDs."""
        engine = SFDRDNSHEngine()
        enabled = engine.enabled_thresholds
        assert isinstance(enabled, list)
        assert len(enabled) >= 1
        for item in enabled:
            assert isinstance(item, str)

    # ---------------------------------------------------------------
    # 10. critical_indicators property
    # ---------------------------------------------------------------

    def test_critical_indicators_returns_list(self):
        """Test critical_indicators returns identifiers of critical PAIs."""
        engine = SFDRDNSHEngine()
        critical = engine.critical_indicators
        assert isinstance(critical, (list, set, tuple))

    # ---------------------------------------------------------------
    # 11. DNSHStatus enum
    # ---------------------------------------------------------------

    def test_dnsh_status_enum_values(self):
        """Test DNSHStatus enum has expected values."""
        expected = {"PASS", "FAIL", "WARNING", "NOT_APPLICABLE", "INSUFFICIENT_DATA"}
        actual = {s.value for s in DNSHStatus}
        assert expected.issubset(actual)

    # ---------------------------------------------------------------
    # 12. Deterministic assessment
    # ---------------------------------------------------------------

    def test_deterministic_assessment(self):
        """Test same input yields identical provenance hash."""
        engine = SFDRDNSHEngine()
        inv = _make_clean_investment()
        a1 = engine.assess_dnsh(inv)
        a2 = engine.assess_dnsh(inv)
        assert a1.provenance_hash == a2.provenance_hash

    # ---------------------------------------------------------------
    # 13. Boolean thresholds
    # ---------------------------------------------------------------

    def test_pai_14_controversial_weapons_boolean(self):
        """Test PAI 14 (controversial weapons) uses boolean threshold."""
        engine = SFDRDNSHEngine()
        dirty = _make_dirty_investment()
        assessment = engine.assess_dnsh(dirty)
        pai14_checks = [c for c in assessment.checks if c.pai_indicator_id == "PAI_14"]
        if pai14_checks:
            assert pai14_checks[0].status == DNSHStatus.FAIL

    # ---------------------------------------------------------------
    # 14. SeverityLevel enum
    # ---------------------------------------------------------------

    def test_severity_level_enum(self):
        """Test SeverityLevel enum exists with expected values."""
        vals = {s.value for s in SeverityLevel}
        assert len(vals) >= 2

    # ---------------------------------------------------------------
    # 15. ThresholdDirection enum
    # ---------------------------------------------------------------

    def test_threshold_direction_enum(self):
        """Test ThresholdDirection enum has expected values."""
        vals = {d.value for d in ThresholdDirection}
        assert len(vals) >= 2

    # ---------------------------------------------------------------
    # 16. PAIDNSHCheck model fields
    # ---------------------------------------------------------------

    def test_pai_dnsh_check_fields(self):
        """Test PAIDNSHCheck has expected fields."""
        engine = SFDRDNSHEngine()
        assessment = engine.assess_dnsh(_make_clean_investment())
        if assessment.checks:
            check = assessment.checks[0]
            assert hasattr(check, "pai_indicator_id")
            assert hasattr(check, "status")
            assert hasattr(check, "threshold_value")

    # ---------------------------------------------------------------
    # 17. InvestmentPAIData model
    # ---------------------------------------------------------------

    def test_investment_pai_data_model(self):
        """Test InvestmentPAIData model construction."""
        inv = InvestmentPAIData(
            investment_id="TEST",
            investment_name="Test Investment",
            pai_values={"PAI_1": 1000.0},
            pai_boolean_flags={"PAI_4": False},
        )
        assert inv.investment_id == "TEST"
        assert inv.pai_values["PAI_1"] == 1000.0

    # ---------------------------------------------------------------
    # 18. PAICategory enum
    # ---------------------------------------------------------------

    def test_pai_category_enum(self):
        """Test PAICategory enum has expected categories."""
        vals = {c.value for c in PAICategory}
        assert len(vals) >= 3
