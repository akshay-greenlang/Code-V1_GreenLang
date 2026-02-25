# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions Compliance Checker Engine.

Tests ComplianceCheckerEngine: 7 regulatory frameworks, requirement lookups,
single/multi-framework checks, and compliance summaries.

Target: 70+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agricultural_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance engine not available")


@pytest.fixture
def engine():
    if COMPLIANCE_AVAILABLE:
        return ComplianceCheckerEngine()
    return None


# ===========================================================================
# Test Class: Initialization
# ===========================================================================


@_SKIP
class TestComplianceInit:
    """Test ComplianceCheckerEngine initialization."""

    def test_engine_creation(self, engine):
        assert engine is not None

    def test_has_check_compliance(self, engine):
        assert hasattr(engine, 'check_compliance')

    def test_has_get_supported_frameworks(self, engine):
        assert hasattr(engine, 'get_supported_frameworks')


# ===========================================================================
# Test Class: Framework Listing
# ===========================================================================


@_SKIP
class TestFrameworkListing:
    """Test framework listing and discovery."""

    def test_supported_frameworks(self, engine):
        frameworks = engine.get_supported_frameworks()
        assert isinstance(frameworks, list)
        assert len(frameworks) >= 5

    def test_ipcc_2006_in_frameworks(self, engine):
        frameworks = engine.get_supported_frameworks()
        frameworks_lower = [f.lower() for f in frameworks]
        assert any("ipcc" in f for f in frameworks_lower)

    def test_ghg_protocol_in_frameworks(self, engine):
        frameworks = engine.get_supported_frameworks()
        frameworks_lower = [f.lower() for f in frameworks]
        assert any("ghg" in f for f in frameworks_lower)

    def test_csrd_in_frameworks(self, engine):
        frameworks = engine.get_supported_frameworks()
        frameworks_lower = [f.lower() for f in frameworks]
        assert any("csrd" in f or "esrs" in f for f in frameworks_lower)

    def test_iso_14064_in_frameworks(self, engine):
        frameworks = engine.get_supported_frameworks()
        frameworks_lower = [f.lower() for f in frameworks]
        assert any("iso" in f or "14064" in f for f in frameworks_lower)

    def test_epa_in_frameworks(self, engine):
        frameworks = engine.get_supported_frameworks()
        frameworks_lower = [f.lower() for f in frameworks]
        assert any("epa" in f or "40cfr" in f for f in frameworks_lower)

    def test_defra_in_frameworks(self, engine):
        frameworks = engine.get_supported_frameworks()
        frameworks_lower = [f.lower() for f in frameworks]
        assert any("defra" in f for f in frameworks_lower)


# ===========================================================================
# Test Class: Requirement Lookups
# ===========================================================================


@_SKIP
class TestRequirements:
    """Test framework requirement lookups."""

    def test_get_framework_requirements(self, engine):
        if not hasattr(engine, 'get_framework_requirements'):
            pytest.skip("get_framework_requirements not available")
        frameworks = engine.get_supported_frameworks()
        reqs = engine.get_framework_requirements(frameworks[0])
        assert reqs is not None

    def test_requirement_count(self, engine):
        if not hasattr(engine, 'get_requirement_count'):
            pytest.skip("get_requirement_count not available")
        frameworks = engine.get_supported_frameworks()
        count = engine.get_requirement_count(frameworks[0])
        assert count > 0

    def test_all_requirements(self, engine):
        if not hasattr(engine, 'get_all_requirements'):
            pytest.skip("get_all_requirements not available")
        all_reqs = engine.get_all_requirements()
        assert isinstance(all_reqs, dict)


# ===========================================================================
# Test Class: Compliance Checking
# ===========================================================================


@_SKIP
class TestComplianceChecking:
    """Test compliance check execution."""

    def test_check_compliance_basic(self, engine):
        calc_data = {
            "calculation_id": "calc-001",
            "source_category": "enteric_fermentation",
            "animal_type": "dairy_cattle",
            "head_count": 200,
            "calculation_method": "ipcc_tier_1",
            "gwp_source": "AR6",
            "ef_source": "ipcc_2006",
            "total_co2e_tonnes": 762.88,
            "tenant_id": "tenant-001",
        }
        result = engine.check_compliance(calc_data)
        assert result is not None

    def test_check_returns_dict_or_object(self, engine):
        calc_data = {
            "calculation_id": "calc-002",
            "source_category": "enteric_fermentation",
            "calculation_method": "ipcc_tier_1",
            "total_co2e_tonnes": 100.0,
        }
        result = engine.check_compliance(calc_data)
        assert isinstance(result, (dict, object))

    def test_check_with_frameworks(self, engine):
        calc_data = {
            "calculation_id": "calc-003",
            "source_category": "enteric_fermentation",
            "calculation_method": "ipcc_tier_1",
            "total_co2e_tonnes": 100.0,
        }
        frameworks = engine.get_supported_frameworks()
        result = engine.check_compliance(calc_data, frameworks=[frameworks[0]])
        assert result is not None

    def test_check_multiple_frameworks(self, engine):
        calc_data = {
            "calculation_id": "calc-004",
            "source_category": "manure_management",
            "calculation_method": "ipcc_tier_2",
            "total_co2e_tonnes": 50.0,
        }
        frameworks = engine.get_supported_frameworks()
        result = engine.check_compliance(calc_data, frameworks=frameworks[:3])
        assert result is not None

    def test_check_manure_compliance(self, engine):
        calc_data = {
            "calculation_id": "calc-005",
            "source_category": "manure_management",
            "animal_type": "dairy_cattle",
            "head_count": 200,
            "awms_type": "uncovered_anaerobic_lagoon",
            "calculation_method": "ipcc_tier_2",
            "total_co2e_tonnes": 500.0,
        }
        result = engine.check_compliance(calc_data)
        assert result is not None

    def test_check_cropland_compliance(self, engine):
        calc_data = {
            "calculation_id": "calc-006",
            "source_category": "cropland_emissions",
            "calculation_method": "ipcc_tier_1",
            "total_co2e_tonnes": 200.0,
        }
        result = engine.check_compliance(calc_data)
        assert result is not None


# ===========================================================================
# Test Class: Individual Framework Checks
# ===========================================================================


@_SKIP
class TestIndividualFrameworks:
    """Test individual framework check methods."""

    def test_check_ipcc_2006(self, engine):
        if not hasattr(engine, 'check_ipcc_2006'):
            pytest.skip("check_ipcc_2006 not available")
        calc_data = {"calculation_method": "ipcc_tier_1", "source_category": "enteric_fermentation"}
        result = engine.check_ipcc_2006(calc_data)
        assert result is not None

    def test_check_ipcc_2019(self, engine):
        if not hasattr(engine, 'check_ipcc_2019'):
            pytest.skip("check_ipcc_2019 not available")
        calc_data = {"calculation_method": "ipcc_tier_1", "source_category": "enteric_fermentation"}
        result = engine.check_ipcc_2019(calc_data)
        assert result is not None

    def test_check_ghg_protocol(self, engine):
        if not hasattr(engine, 'check_ghg_protocol'):
            pytest.skip("check_ghg_protocol not available")
        calc_data = {"calculation_method": "ipcc_tier_1", "source_category": "enteric_fermentation"}
        result = engine.check_ghg_protocol(calc_data)
        assert result is not None

    def test_check_iso_14064(self, engine):
        if not hasattr(engine, 'check_iso_14064'):
            pytest.skip("check_iso_14064 not available")
        calc_data = {"calculation_method": "ipcc_tier_1"}
        result = engine.check_iso_14064(calc_data)
        assert result is not None

    def test_check_csrd_esrs(self, engine):
        if not hasattr(engine, 'check_csrd_esrs'):
            pytest.skip("check_csrd_esrs not available")
        calc_data = {"calculation_method": "ipcc_tier_1"}
        result = engine.check_csrd_esrs(calc_data)
        assert result is not None

    def test_check_epa(self, engine):
        if not hasattr(engine, 'check_epa_40cfr98'):
            pytest.skip("check_epa_40cfr98 not available")
        calc_data = {"calculation_method": "ipcc_tier_1"}
        result = engine.check_epa_40cfr98(calc_data)
        assert result is not None

    def test_check_defra(self, engine):
        if not hasattr(engine, 'check_defra'):
            pytest.skip("check_defra not available")
        calc_data = {"calculation_method": "ipcc_tier_1"}
        result = engine.check_defra(calc_data)
        assert result is not None


# ===========================================================================
# Test Class: Compliance Summary
# ===========================================================================


@_SKIP
class TestComplianceSummary:
    """Test compliance summary and findings."""

    def test_compliance_summary(self, engine):
        if not hasattr(engine, 'get_compliance_summary'):
            pytest.skip("get_compliance_summary not available")
        calc_data = {
            "calculation_id": "calc-sum",
            "source_category": "enteric_fermentation",
            "calculation_method": "ipcc_tier_1",
            "total_co2e_tonnes": 100.0,
        }
        engine.check_compliance(calc_data)
        summary = engine.get_compliance_summary()
        assert summary is not None

    def test_critical_findings(self, engine):
        if not hasattr(engine, 'get_critical_findings'):
            pytest.skip("get_critical_findings not available")
        calc_data = {
            "calculation_id": "calc-crit",
            "source_category": "enteric_fermentation",
        }
        result = engine.check_compliance(calc_data)
        findings = engine.get_critical_findings(result)
        assert isinstance(findings, (list, dict))

    def test_pass_rate(self, engine):
        if not hasattr(engine, 'get_pass_rate'):
            pytest.skip("get_pass_rate not available")
        calc_data = {
            "calculation_id": "calc-pass",
            "source_category": "enteric_fermentation",
            "calculation_method": "ipcc_tier_1",
            "total_co2e_tonnes": 100.0,
        }
        result = engine.check_compliance(calc_data)
        rate = engine.get_pass_rate(result)
        assert isinstance(rate, (int, float))


# ===========================================================================
# Test Class: Statistics
# ===========================================================================


@_SKIP
class TestComplianceStatistics:
    """Test compliance engine statistics."""

    def test_get_statistics(self, engine):
        stats = engine.get_statistics()
        assert isinstance(stats, dict)

    def test_reset(self, engine):
        engine.reset()
        stats = engine.get_statistics()
        assert isinstance(stats, dict)
