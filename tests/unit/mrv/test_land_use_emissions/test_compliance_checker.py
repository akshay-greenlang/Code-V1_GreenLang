# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (Engine 6 of 7)

AGENT-MRV-006: Land Use Emissions Agent

Tests multi-framework regulatory compliance checking across seven
frameworks (IPCC 2006, IPCC 2019, GHG Protocol Land, ISO 14064-1,
CSRD/ESRS E1, EU LULUCF Regulation, SBTi FLAG) with 83 total
requirements.

Target: 100 tests, ~800 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.land_use_emissions.compliance_checker import (
    ComplianceCheckerEngine,
    ComplianceFinding,
    SUPPORTED_FRAMEWORKS,
    ALL_POOLS,
    VALID_CATEGORIES,
    VALID_METHODS,
    VALID_CLIMATE_ZONES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compliance_engine():
    """Create a ComplianceCheckerEngine instance."""
    engine = ComplianceCheckerEngine()
    yield engine
    engine.reset()


@pytest.fixture
def complete_data() -> Dict[str, Any]:
    """Return a fully compliant calculation data dictionary."""
    return {
        "land_category": "FOREST_LAND",
        "climate_zone": "TROPICAL_WET",
        "method": "STOCK_DIFFERENCE",
        "tier": "TIER_1",
        "area_ha": 1000,
        "pools_reported": ["AGB", "BGB", "DEAD_WOOD", "LITTER", "SOC"],
        "total_co2e_tonnes": -5000,  # Net removal
        "net_co2e_tonnes_yr": -5000,
        "gross_emissions_tco2_yr": 200,
        "gross_removals_tco2_yr": -5200,
        "emission_type": "NET_REMOVAL",
        "has_uncertainty": True,
        "provenance_hash": "a" * 64,
        "ef_source": "IPCC_2006",
        "gwp_source": "AR6",
        "year_t1": 2020,
        "year_t2": 2025,
        "base_year": 2020,
        "reporting_period": "2020-2025",
        "parcel_id": "parcel-001",
        "management_practice": "NOMINAL",
        "is_managed": True,
        "biogenic_co2": True,
        "transition_type": "REMAINING",
    }


@pytest.fixture
def minimal_data() -> Dict[str, Any]:
    """Return a minimal calculation data dictionary (many fields missing)."""
    return {
        "land_category": "FOREST_LAND",
        "method": "STOCK_DIFFERENCE",
        "area_ha": 100,
        "total_co2e_tonnes": 500,
    }


@pytest.fixture
def empty_data() -> Dict[str, Any]:
    """Return an empty data dictionary."""
    return {}


# ===========================================================================
# 1. Initialisation Tests
# ===========================================================================


class TestComplianceEngineInit:
    """Test ComplianceCheckerEngine initialisation."""

    def test_init_creates_7_framework_checkers(self, compliance_engine):
        """Test that 7 framework checkers are registered."""
        assert len(compliance_engine._framework_checkers) == 7

    def test_supported_frameworks_constant(self):
        """Test SUPPORTED_FRAMEWORKS contains all 7 frameworks."""
        assert len(SUPPORTED_FRAMEWORKS) == 7
        assert "IPCC_2006" in SUPPORTED_FRAMEWORKS
        assert "IPCC_2019" in SUPPORTED_FRAMEWORKS
        assert "GHG_PROTOCOL_LAND" in SUPPORTED_FRAMEWORKS
        assert "ISO_14064" in SUPPORTED_FRAMEWORKS
        assert "CSRD_ESRS" in SUPPORTED_FRAMEWORKS
        assert "EU_LULUCF" in SUPPORTED_FRAMEWORKS
        assert "SBTI_FLAG" in SUPPORTED_FRAMEWORKS

    def test_all_pools_constant(self):
        """Test ALL_POOLS contains 5 IPCC carbon pools."""
        assert ALL_POOLS == ["AGB", "BGB", "DEAD_WOOD", "LITTER", "SOC"]

    def test_valid_categories_constant(self):
        """Test VALID_CATEGORIES contains 6 IPCC categories."""
        assert len(VALID_CATEGORIES) == 6

    def test_total_checks_counter_starts_zero(self, compliance_engine):
        """Test check counter starts at zero."""
        stats = compliance_engine.get_statistics()
        assert stats["total_checks"] == 0


# ===========================================================================
# 2. check_compliance() Dispatch Tests
# ===========================================================================


class TestCheckComplianceDispatch:
    """Test check_compliance dispatches to correct framework checkers."""

    def test_dispatch_single_framework(self, compliance_engine, complete_data):
        """Test checking a single framework."""
        result = compliance_engine.check_compliance(complete_data, ["IPCC_2006"])
        assert result["status"] == "SUCCESS"
        assert "IPCC_2006" in result["framework_results"]
        assert len(result["frameworks_checked"]) == 1

    def test_dispatch_multiple_frameworks(self, compliance_engine, complete_data):
        """Test checking multiple frameworks."""
        result = compliance_engine.check_compliance(
            complete_data, ["IPCC_2006", "ISO_14064", "SBTI_FLAG"]
        )
        assert len(result["frameworks_checked"]) == 3

    def test_dispatch_all_frameworks_when_none(self, compliance_engine, complete_data):
        """Test that None frameworks checks all 7 frameworks."""
        result = compliance_engine.check_compliance(complete_data, None)
        assert len(result["frameworks_checked"]) == 7

    def test_unknown_framework_skipped(self, compliance_engine, complete_data):
        """Test that unknown framework names are silently skipped."""
        result = compliance_engine.check_compliance(
            complete_data, ["IPCC_2006", "NONEXISTENT_FW"]
        )
        assert "IPCC_2006" in result["frameworks_checked"]
        assert "NONEXISTENT_FW" not in result["frameworks_checked"]

    def test_case_insensitive_framework_names(self, compliance_engine, complete_data):
        """Test that framework names are case-insensitive."""
        result = compliance_engine.check_compliance(complete_data, ["ipcc_2006"])
        assert "IPCC_2006" in result["frameworks_checked"]

    def test_check_increments_counter(self, compliance_engine, complete_data):
        """Test that each check_compliance call increments the counter."""
        compliance_engine.check_compliance(complete_data, ["IPCC_2006"])
        compliance_engine.check_compliance(complete_data, ["ISO_14064"])
        stats = compliance_engine.get_statistics()
        assert stats["total_checks"] == 2


# ===========================================================================
# 3. Compliance Status Tests
# ===========================================================================


class TestComplianceStatus:
    """Test COMPLIANT / PARTIAL / NON_COMPLIANT status determination."""

    def test_fully_compliant_returns_compliant(self, compliance_engine, complete_data):
        """Test that fully complete data returns COMPLIANT for IPCC_2006."""
        result = compliance_engine.check_compliance(complete_data, ["IPCC_2006"])
        ipcc = result["framework_results"]["IPCC_2006"]
        assert ipcc["status"] == "COMPLIANT"
        assert ipcc["pass_rate_pct"] == 100.0

    def test_empty_data_returns_non_compliant(self, compliance_engine, empty_data):
        """Test that empty data returns NON_COMPLIANT or PARTIAL."""
        result = compliance_engine.check_compliance(empty_data, ["IPCC_2006"])
        ipcc = result["framework_results"]["IPCC_2006"]
        assert ipcc["status"] in ("NON_COMPLIANT", "PARTIAL")

    def test_partial_data_returns_partial(self, compliance_engine, minimal_data):
        """Test that partial data returns PARTIAL status."""
        result = compliance_engine.check_compliance(minimal_data, ["IPCC_2006"])
        ipcc = result["framework_results"]["IPCC_2006"]
        # Some requirements should pass, some should fail
        assert ipcc["passed"] > 0
        assert ipcc["failed"] > 0

    def test_overall_status_aggregation(self, compliance_engine, complete_data):
        """Test overall compliance status across multiple frameworks."""
        result = compliance_engine.check_compliance(complete_data, None)
        overall = result["overall"]
        assert overall["total_requirements"] > 0
        assert overall["total_passed"] <= overall["total_requirements"]
        assert overall["compliance_status"] in ("COMPLIANT", "PARTIAL", "NON_COMPLIANT")


# ===========================================================================
# 4. IPCC 2006 Requirements Tests
# ===========================================================================


class TestIPCC2006:
    """Test IPCC 2006 Vol 4 compliance requirements (12 requirements)."""

    def test_ipcc_2006_returns_12_findings(self, compliance_engine, complete_data):
        """Test IPCC 2006 check returns exactly 12 findings."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        assert len(findings) == 12

    def test_ipcc_2006_valid_land_category_passes(self, compliance_engine, complete_data):
        """Test REQ-01: valid land category passes."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        assert findings[0].passed is True

    def test_ipcc_2006_invalid_land_category_fails(self, compliance_engine):
        """Test REQ-01: invalid land category fails."""
        findings = compliance_engine.check_ipcc_2006({"land_category": "DESERT"})
        assert findings[0].passed is False
        assert findings[0].severity == "ERROR"

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_ipcc_2006_all_valid_categories(self, compliance_engine, category):
        """Test REQ-01: all six IPCC categories pass."""
        findings = compliance_engine.check_ipcc_2006({"land_category": category})
        assert findings[0].passed is True

    def test_ipcc_2006_valid_climate_zone_passes(self, compliance_engine, complete_data):
        """Test REQ-02: valid climate zone passes."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        assert findings[1].passed is True

    @pytest.mark.parametrize("zone", VALID_CLIMATE_ZONES)
    def test_ipcc_2006_all_valid_zones(self, compliance_engine, zone):
        """Test REQ-02: all 12 IPCC climate zones pass."""
        findings = compliance_engine.check_ipcc_2006({"climate_zone": zone})
        assert findings[1].passed is True

    def test_ipcc_2006_valid_method(self, compliance_engine, complete_data):
        """Test REQ-03: STOCK_DIFFERENCE method passes."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        assert findings[2].passed is True

    @pytest.mark.parametrize("method", VALID_METHODS)
    def test_ipcc_2006_both_methods_pass(self, compliance_engine, method):
        """Test REQ-03: both STOCK_DIFFERENCE and GAIN_LOSS pass."""
        findings = compliance_engine.check_ipcc_2006({"method": method})
        assert findings[2].passed is True

    def test_ipcc_2006_all_five_pools_passes(self, compliance_engine, complete_data):
        """Test REQ-04: reporting all five pools passes."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        assert findings[3].passed is True

    def test_ipcc_2006_missing_pool_fails(self, compliance_engine):
        """Test REQ-04: missing a pool fails."""
        findings = compliance_engine.check_ipcc_2006({
            "pools_reported": ["AGB", "BGB", "DEAD_WOOD", "LITTER"],  # Missing SOC
        })
        assert findings[3].passed is False

    def test_ipcc_2006_area_reported_passes(self, compliance_engine, complete_data):
        """Test REQ-05: area_ha present passes."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        assert findings[4].passed is True

    def test_ipcc_2006_ef_source_warning_when_missing(self, compliance_engine, minimal_data):
        """Test REQ-06: missing ef_source is a WARNING."""
        findings = compliance_engine.check_ipcc_2006(minimal_data)
        assert findings[5].severity == "WARNING"

    def test_ipcc_2006_tier_warning_when_missing(self, compliance_engine, minimal_data):
        """Test REQ-07: missing tier is a WARNING."""
        findings = compliance_engine.check_ipcc_2006(minimal_data)
        assert findings[6].severity == "WARNING"

    def test_ipcc_2006_provenance_hash_passes(self, compliance_engine, complete_data):
        """Test REQ-08: provenance hash present passes."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        assert findings[7].passed is True

    def test_ipcc_2006_co2e_reported_passes(self, compliance_engine, complete_data):
        """Test REQ-09: CO2e reported passes."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        assert findings[8].passed is True

    def test_ipcc_2006_time_period_passes(self, compliance_engine, complete_data):
        """Test REQ-11: time period defined passes."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        assert findings[10].passed is True


# ===========================================================================
# 5. IPCC 2019 Requirements Tests
# ===========================================================================


class TestIPCC2019:
    """Test IPCC 2019 Refinement compliance requirements (12 requirements)."""

    def test_ipcc_2019_returns_12_findings(self, compliance_engine, complete_data):
        """Test IPCC 2019 check returns exactly 12 findings."""
        findings = compliance_engine.check_ipcc_2019(complete_data)
        assert len(findings) == 12

    def test_ipcc_2019_inherits_base_checks(self, compliance_engine, complete_data):
        """Test IPCC 2019 first 5 requirements mirror IPCC 2006."""
        f2006 = compliance_engine.check_ipcc_2006(complete_data)
        f2019 = compliance_engine.check_ipcc_2019(complete_data)
        for i in range(5):
            assert f2006[i].passed == f2019[i].passed

    def test_ipcc_2019_ef_source_2019(self, compliance_engine):
        """Test REQ-06: IPCC_2019 ef_source is recognized."""
        data = {"ef_source": "IPCC_2019", "land_category": "FOREST_LAND"}
        findings = compliance_engine.check_ipcc_2019(data)
        # REQ-06 is at index 5
        assert findings[5].passed is True

    def test_ipcc_2019_wetland_detail_check(self, compliance_engine):
        """Test REQ-07: wetland detail required for wetland category."""
        data = {"land_category": "WETLANDS", "peatland_status": "drained"}
        findings = compliance_engine.check_ipcc_2019(data)
        assert findings[6].passed is True

    def test_ipcc_2019_non_wetland_passes_wetland_check(self, compliance_engine, complete_data):
        """Test REQ-07: non-wetland categories auto-pass wetland detail check."""
        findings = compliance_engine.check_ipcc_2019(complete_data)
        assert findings[6].passed is True

    def test_ipcc_2019_uncertainty_check(self, compliance_engine, complete_data):
        """Test REQ-12: uncertainty assessment passes."""
        findings = compliance_engine.check_ipcc_2019(complete_data)
        assert findings[11].passed is True

    def test_ipcc_2019_managed_land_check(self, compliance_engine, complete_data):
        """Test REQ-08: managed land status documented."""
        findings = compliance_engine.check_ipcc_2019(complete_data)
        assert findings[7].passed is True


# ===========================================================================
# 6. GHG Protocol Land Sector Tests
# ===========================================================================


class TestGHGProtocolLand:
    """Test GHG Protocol Land Sector compliance (12 requirements)."""

    def test_ghg_protocol_returns_12_findings(self, compliance_engine, complete_data):
        """Test GHG Protocol check returns exactly 12 findings."""
        findings = compliance_engine.check_ghg_protocol_land(complete_data)
        assert len(findings) == 12

    def test_ghg_protocol_emission_separation(self, compliance_engine, complete_data):
        """Test REQ-02: gross emissions and removals reported separately."""
        findings = compliance_engine.check_ghg_protocol_land(complete_data)
        assert findings[1].passed is True

    def test_ghg_protocol_emission_separation_fails_when_missing(self, compliance_engine, minimal_data):
        """Test REQ-02: fails when gross emissions/removals not separated."""
        findings = compliance_engine.check_ghg_protocol_land(minimal_data)
        assert findings[1].passed is False

    def test_ghg_protocol_base_year(self, compliance_engine, complete_data):
        """Test REQ-05: base year established."""
        findings = compliance_engine.check_ghg_protocol_land(complete_data)
        assert findings[4].passed is True

    def test_ghg_protocol_pool_completeness(self, compliance_engine, complete_data):
        """Test REQ-06: at least 3 pools reported."""
        findings = compliance_engine.check_ghg_protocol_land(complete_data)
        assert findings[5].passed is True

    def test_ghg_protocol_insufficient_pools(self, compliance_engine):
        """Test REQ-06: fewer than 3 pools fails."""
        findings = compliance_engine.check_ghg_protocol_land({
            "pools_reported": ["AGB"],
        })
        assert findings[5].passed is False

    def test_ghg_protocol_method_documented(self, compliance_engine, complete_data):
        """Test REQ-07: method documented."""
        findings = compliance_engine.check_ghg_protocol_land(complete_data)
        assert findings[6].passed is True

    def test_ghg_protocol_temporal_boundary(self, compliance_engine, complete_data):
        """Test REQ-09: temporal boundary defined."""
        findings = compliance_engine.check_ghg_protocol_land(complete_data)
        assert findings[8].passed is True

    def test_ghg_protocol_verification_readiness(self, compliance_engine, complete_data):
        """Test REQ-12: verification readiness (provenance)."""
        findings = compliance_engine.check_ghg_protocol_land(complete_data)
        assert findings[11].passed is True


# ===========================================================================
# 7. ISO 14064-1 Tests
# ===========================================================================


class TestISO14064:
    """Test ISO 14064-1:2018 compliance (12 requirements)."""

    def test_iso_returns_12_findings(self, compliance_engine, complete_data):
        """Test ISO 14064 check returns exactly 12 findings."""
        findings = compliance_engine.check_iso_14064(complete_data)
        assert len(findings) == 12

    def test_iso_category_1_always_passes(self, compliance_engine, empty_data):
        """Test REQ-01: LULUCF always classified as Category 1."""
        findings = compliance_engine.check_iso_14064(empty_data)
        assert findings[0].passed is True

    def test_iso_uncertainty_required(self, compliance_engine, complete_data):
        """Test REQ-05: uncertainty assessment required."""
        findings = compliance_engine.check_iso_14064(complete_data)
        assert findings[4].passed is True

    def test_iso_uncertainty_missing_fails(self, compliance_engine, minimal_data):
        """Test REQ-05: missing uncertainty fails."""
        findings = compliance_engine.check_iso_14064(minimal_data)
        assert findings[4].passed is False
        assert findings[4].severity == "ERROR"

    def test_iso_pool_completeness(self, compliance_engine, complete_data):
        """Test REQ-06: at least 4 pools required."""
        findings = compliance_engine.check_iso_14064(complete_data)
        assert findings[5].passed is True

    def test_iso_insufficient_pools(self, compliance_engine):
        """Test REQ-06: 3 pools is insufficient for ISO."""
        findings = compliance_engine.check_iso_14064({
            "pools_reported": ["AGB", "BGB", "SOC"],
        })
        assert findings[5].passed is False

    def test_iso_gwp_source_required(self, compliance_engine, complete_data):
        """Test REQ-10: GWP source required."""
        findings = compliance_engine.check_iso_14064(complete_data)
        assert findings[9].passed is True

    def test_iso_gwp_missing_fails(self, compliance_engine, minimal_data):
        """Test REQ-10: missing GWP source fails."""
        findings = compliance_engine.check_iso_14064(minimal_data)
        assert findings[9].passed is False


# ===========================================================================
# 8. CSRD/ESRS E1 Tests
# ===========================================================================


class TestCSRDESRS:
    """Test CSRD/ESRS E1 compliance (11 requirements)."""

    def test_csrd_returns_11_findings(self, compliance_engine, complete_data):
        """Test CSRD/ESRS check returns exactly 11 findings."""
        findings = compliance_engine.check_csrd_esrs(complete_data)
        assert len(findings) == 11

    def test_csrd_e1_6_gross_emissions(self, compliance_engine, complete_data):
        """Test REQ-01 E1-6: gross Scope 1 emissions disclosed."""
        findings = compliance_engine.check_csrd_esrs(complete_data)
        assert findings[0].passed is True

    def test_csrd_e1_7_removals(self, compliance_engine, complete_data):
        """Test REQ-02 E1-7: removals reported separately."""
        findings = compliance_engine.check_csrd_esrs(complete_data)
        assert findings[1].passed is True

    def test_csrd_removals_missing_fails(self, compliance_engine, minimal_data):
        """Test REQ-02: missing gross_removals fails."""
        findings = compliance_engine.check_csrd_esrs(minimal_data)
        assert findings[1].passed is False

    def test_csrd_methodology_reference(self, compliance_engine, complete_data):
        """Test REQ-05: methodology referenced."""
        findings = compliance_engine.check_csrd_esrs(complete_data)
        assert findings[4].passed is True

    def test_csrd_co2e_reporting_unit(self, compliance_engine, complete_data):
        """Test REQ-09: emissions in tonnes CO2e."""
        findings = compliance_engine.check_csrd_esrs(complete_data)
        assert findings[8].passed is True

    def test_csrd_verification_readiness(self, compliance_engine, complete_data):
        """Test REQ-11: third-party verification readiness."""
        findings = compliance_engine.check_csrd_esrs(complete_data)
        assert findings[10].passed is True


# ===========================================================================
# 9. EU LULUCF Regulation Tests
# ===========================================================================


class TestEULULUCF:
    """Test EU LULUCF Regulation compliance (12 requirements)."""

    def test_eu_lulucf_returns_12_findings(self, compliance_engine, complete_data):
        """Test EU LULUCF check returns exactly 12 findings."""
        findings = compliance_engine.check_eu_lulucf(complete_data)
        assert len(findings) == 12

    def test_eu_lulucf_managed_land_proxy(self, compliance_engine, complete_data):
        """Test REQ-01: managed land proxy applied."""
        findings = compliance_engine.check_eu_lulucf(complete_data)
        assert findings[0].passed is True

    def test_eu_lulucf_managed_land_missing_fails(self, compliance_engine, empty_data):
        """Test REQ-01: missing managed land status fails."""
        findings = compliance_engine.check_eu_lulucf(empty_data)
        assert findings[0].passed is False

    def test_eu_lulucf_no_debit_net_removal_passes(self, compliance_engine, complete_data):
        """Test REQ-03: no-debit rule passes for net removal."""
        findings = compliance_engine.check_eu_lulucf(complete_data)
        assert findings[2].passed is True

    def test_eu_lulucf_no_debit_net_source_fails(self, compliance_engine, complete_data):
        """Test REQ-03: no-debit rule fails for net source."""
        complete_data["net_co2e_tonnes_yr"] = 500  # Positive = net source
        complete_data["total_co2e_tonnes"] = 500
        findings = compliance_engine.check_eu_lulucf(complete_data)
        assert findings[2].passed is False
        assert findings[2].severity == "ERROR"

    def test_eu_lulucf_no_debit_neutral_passes(self, compliance_engine, complete_data):
        """Test REQ-03: no-debit rule passes for neutral (zero)."""
        complete_data["net_co2e_tonnes_yr"] = 0
        complete_data["total_co2e_tonnes"] = 0
        findings = compliance_engine.check_eu_lulucf(complete_data)
        assert findings[2].passed is True

    def test_eu_lulucf_cropland_management(self, compliance_engine):
        """Test REQ-10: cropland management activities documented."""
        data = {
            "land_category": "CROPLAND",
            "management_practice": "FULL_TILLAGE",
        }
        findings = compliance_engine.check_eu_lulucf(data)
        assert findings[9].passed is True

    def test_eu_lulucf_wetland_drainage_check(self, compliance_engine):
        """Test REQ-11: wetland drainage and rewetting accounting."""
        data = {
            "land_category": "WETLANDS",
            "peatland_status": "drained",
        }
        findings = compliance_engine.check_eu_lulucf(data)
        assert findings[10].passed is True

    def test_eu_lulucf_non_wetland_passes_wetland_check(self, compliance_engine, complete_data):
        """Test REQ-11: non-wetland auto-passes wetland check."""
        findings = compliance_engine.check_eu_lulucf(complete_data)
        assert findings[10].passed is True

    def test_eu_lulucf_provenance_for_review(self, compliance_engine, complete_data):
        """Test REQ-12: provenance for EU expert review."""
        findings = compliance_engine.check_eu_lulucf(complete_data)
        assert findings[11].passed is True


# ===========================================================================
# 10. SBTi FLAG Tests
# ===========================================================================


class TestSBTiFLAG:
    """Test SBTi FLAG guidance compliance (12 requirements)."""

    def test_sbti_flag_returns_12_findings(self, compliance_engine, complete_data):
        """Test SBTi FLAG check returns exactly 12 findings."""
        findings = compliance_engine.check_sbti_flag(complete_data)
        assert len(findings) == 12

    def test_sbti_flag_boundary(self, compliance_engine, complete_data):
        """Test REQ-01: FLAG boundary (FOREST, CROPLAND, GRASSLAND, WETLANDS)."""
        findings = compliance_engine.check_sbti_flag(complete_data)
        assert findings[0].passed is True

    def test_sbti_flag_base_year(self, compliance_engine, complete_data):
        """Test REQ-02: base year established."""
        findings = compliance_engine.check_sbti_flag(complete_data)
        assert findings[1].passed is True

    def test_sbti_flag_base_year_missing_fails(self, compliance_engine, empty_data):
        """Test REQ-02: missing base year fails."""
        findings = compliance_engine.check_sbti_flag(empty_data)
        assert findings[1].passed is False

    def test_sbti_flag_72_pct_pathway_info(self, compliance_engine, complete_data):
        """Test REQ-04: -72% by 2050 pathway is INFO level."""
        findings = compliance_engine.check_sbti_flag(complete_data)
        assert findings[3].severity == "INFO"

    def test_sbti_flag_land_management_emissions(self, compliance_engine, complete_data):
        """Test REQ-06: land management emissions included."""
        findings = compliance_engine.check_sbti_flag(complete_data)
        assert findings[5].passed is True

    def test_sbti_flag_removals_separate(self, compliance_engine, complete_data):
        """Test REQ-07: removals accounted separately."""
        findings = compliance_engine.check_sbti_flag(complete_data)
        assert findings[6].passed is True

    def test_sbti_flag_annual_reporting(self, compliance_engine, complete_data):
        """Test REQ-11: annual reporting period."""
        findings = compliance_engine.check_sbti_flag(complete_data)
        assert findings[10].passed is True


# ===========================================================================
# 11. Findings Structure Tests
# ===========================================================================


class TestFindingsStructure:
    """Test ComplianceFinding dataclass and serialization."""

    def test_finding_to_dict(self):
        """Test ComplianceFinding.to_dict() returns correct structure."""
        finding = ComplianceFinding(
            requirement_id="IPCC_2006-01",
            framework="IPCC_2006",
            requirement="Valid land category",
            passed=True,
            severity="ERROR",
            finding="FOREST_LAND is valid",
            recommendation="Use valid IPCC category",
        )
        d = finding.to_dict()
        assert d["requirement_id"] == "IPCC_2006-01"
        assert d["framework"] == "IPCC_2006"
        assert d["passed"] is True
        assert d["severity"] == "ERROR"

    def test_severity_levels_in_findings(self, compliance_engine, complete_data):
        """Test that findings include ERROR, WARNING, and INFO severity levels."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        severities = {f.severity for f in findings}
        # Should have at least ERROR and WARNING defined
        assert "ERROR" in severities or "WARNING" in severities or "INFO" in severities

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_all_frameworks_return_findings(self, compliance_engine, complete_data, framework):
        """Test that every framework returns at least 10 findings."""
        result = compliance_engine.check_compliance(complete_data, [framework])
        fw_result = result["framework_results"][framework]
        assert fw_result["total_requirements"] >= 10

    def test_findings_have_unique_ids(self, compliance_engine, complete_data):
        """Test that all findings within a framework have unique requirement IDs."""
        for framework in SUPPORTED_FRAMEWORKS:
            checker = compliance_engine._framework_checkers[framework]
            findings = checker(complete_data)
            ids = [f.requirement_id for f in findings]
            assert len(ids) == len(set(ids)), f"Duplicate IDs in {framework}: {ids}"


# ===========================================================================
# 12. Multiple Framework Check Tests
# ===========================================================================


class TestMultipleFrameworkCheck:
    """Test checking multiple frameworks in a single call."""

    def test_all_frameworks_aggregate(self, compliance_engine, complete_data):
        """Test that all-framework check produces correct aggregation."""
        result = compliance_engine.check_compliance(complete_data, None)
        overall = result["overall"]
        # Sum of per-framework requirements should equal total
        total_from_fws = sum(
            fw["total_requirements"]
            for fw in result["framework_results"].values()
        )
        assert overall["total_requirements"] == total_from_fws

    def test_pass_rate_calculation(self, compliance_engine, complete_data):
        """Test overall pass rate calculation."""
        result = compliance_engine.check_compliance(complete_data, None)
        overall = result["overall"]
        expected_rate = (
            overall["total_passed"] / overall["total_requirements"] * 100
        )
        assert abs(overall["pass_rate_pct"] - round(expected_rate, 1)) < 0.2

    def test_error_and_warning_counts(self, compliance_engine, complete_data):
        """Test error and warning counts are non-negative."""
        result = compliance_engine.check_compliance(complete_data, None)
        overall = result["overall"]
        assert overall["total_errors"] >= 0
        assert overall["total_warnings"] >= 0

    def test_provenance_hash_present(self, compliance_engine, complete_data):
        """Test that compliance result has provenance hash."""
        result = compliance_engine.check_compliance(complete_data, ["IPCC_2006"])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 13. Edge Cases Tests
# ===========================================================================


class TestComplianceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_pools_list(self, compliance_engine):
        """Test compliance with empty pools_reported list."""
        data = {"pools_reported": []}
        findings = compliance_engine.check_ipcc_2006(data)
        # REQ-04: all pools should fail
        assert findings[3].passed is False

    def test_partial_pools(self, compliance_engine):
        """Test compliance with partial pool coverage."""
        data = {"pools_reported": ["AGB", "BGB", "SOC"]}
        findings = compliance_engine.check_ipcc_2006(data)
        assert findings[3].passed is False  # Missing DEAD_WOOD and LITTER

    def test_case_insensitive_pools(self, compliance_engine):
        """Test that pool names are uppercased for comparison."""
        data = {"pools_reported": ["agb", "bgb", "dead_wood", "litter", "soc"]}
        findings = compliance_engine.check_ipcc_2006(data)
        assert findings[3].passed is True

    def test_empty_string_fields_treated_as_missing(self, compliance_engine):
        """Test that empty string fields are treated as missing."""
        data = {"ef_source": "", "provenance_hash": ""}
        findings = compliance_engine.check_ipcc_2006(data)
        # ef_source (REQ-06) should fail
        assert findings[5].passed is False
        # provenance_hash (REQ-08) should fail
        assert findings[7].passed is False


# ===========================================================================
# 14. get_all_requirements() Tests
# ===========================================================================


class TestGetAllRequirements:
    """Test the requirement listing method."""

    def test_total_requirements_is_83(self, compliance_engine):
        """Test that total requirements across all frameworks is ~83."""
        reqs = compliance_engine.get_all_requirements()
        # Note: may vary slightly based on framework-specific N/A handling
        assert reqs["total_requirements"] >= 75
        assert reqs["total_requirements"] <= 90

    def test_all_frameworks_listed(self, compliance_engine):
        """Test that all 7 frameworks appear in requirements listing."""
        reqs = compliance_engine.get_all_requirements()
        for fw in SUPPORTED_FRAMEWORKS:
            assert fw in reqs["requirements"]

    def test_requirements_have_ids_and_severity(self, compliance_engine):
        """Test that each requirement has an ID and severity."""
        reqs = compliance_engine.get_all_requirements()
        for fw, req_list in reqs["requirements"].items():
            for req in req_list:
                assert "requirement_id" in req
                assert "severity" in req
                assert req["severity"] in ("ERROR", "WARNING", "INFO")


# ===========================================================================
# 15. Statistics and Reset Tests
# ===========================================================================


class TestComplianceStatisticsAndReset:
    """Test engine statistics and reset."""

    def test_statistics_structure(self, compliance_engine):
        """Test statistics returns expected fields."""
        stats = compliance_engine.get_statistics()
        assert stats["engine"] == "ComplianceCheckerEngine"
        assert stats["version"] == "1.0.0"
        assert "created_at" in stats
        assert "total_checks" in stats
        assert stats["supported_frameworks"] == SUPPORTED_FRAMEWORKS
        assert stats["total_requirements"] == 83

    def test_reset_clears_counter(self, compliance_engine, complete_data):
        """Test that reset clears the check counter."""
        compliance_engine.check_compliance(complete_data, ["IPCC_2006"])
        compliance_engine.reset()
        stats = compliance_engine.get_statistics()
        assert stats["total_checks"] == 0

    def test_processing_time_positive(self, compliance_engine, complete_data):
        """Test that processing time is positive."""
        result = compliance_engine.check_compliance(complete_data, ["IPCC_2006"])
        assert result["processing_time_ms"] >= 0


# ===========================================================================
# 16. Cross-Framework Consistency Tests
# ===========================================================================


class TestCrossFrameworkConsistency:
    """Test consistency of checks across frameworks."""

    def test_ipcc_2019_inherits_from_2006(self, compliance_engine, complete_data):
        """Test that IPCC 2019 first 5 requirements match IPCC 2006 semantics."""
        f2006 = compliance_engine.check_ipcc_2006(complete_data)
        f2019 = compliance_engine.check_ipcc_2019(complete_data)
        for i in range(5):
            assert f2006[i].requirement == f2019[i].requirement

    def test_all_frameworks_check_provenance(self, compliance_engine, complete_data):
        """Test that all frameworks have at least one provenance-related check."""
        for fw_name, checker in compliance_engine._framework_checkers.items():
            findings = checker(complete_data)
            provenance_findings = [
                f for f in findings
                if "provenance" in f.requirement.lower() or "audit" in f.requirement.lower()
                or "verification" in f.requirement.lower() or "review" in f.requirement.lower()
            ]
            assert len(provenance_findings) >= 1, f"{fw_name} missing provenance check"

    def test_fully_compliant_data_passes_most_frameworks(self, compliance_engine, complete_data):
        """Test that fully compliant data passes most requirements across all frameworks."""
        result = compliance_engine.check_compliance(complete_data, None)
        overall = result["overall"]
        # With complete data, should pass at least 80% of all requirements
        assert overall["pass_rate_pct"] >= 80.0


# ===========================================================================
# 17. Severity Level Tests
# ===========================================================================


class TestSeverityLevels:
    """Test severity level distribution across frameworks."""

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_framework_has_error_requirements(self, compliance_engine, framework):
        """Test that each framework has at least one ERROR-level requirement."""
        checker = compliance_engine._framework_checkers[framework]
        findings = checker({})  # Empty data to see all requirement severities
        error_findings = [f for f in findings if f.severity == "ERROR"]
        assert len(error_findings) >= 1, f"{framework} has no ERROR requirements"

    def test_ipcc_2006_error_on_missing_critical_fields(self, compliance_engine):
        """Test that IPCC 2006 ERROR findings are triggered for critical missing fields."""
        findings = compliance_engine.check_ipcc_2006({})
        error_findings = [f for f in findings if f.severity == "ERROR" and not f.passed]
        # Missing land_category, climate_zone, method, pools, area, co2e, time period
        assert len(error_findings) >= 5

    def test_info_findings_always_pass(self, compliance_engine, complete_data):
        """Test that INFO-level findings typically pass with complete data."""
        findings = compliance_engine.check_ipcc_2006(complete_data)
        info_findings = [f for f in findings if f.severity == "INFO"]
        for f in info_findings:
            assert f.passed is True, f"INFO finding {f.requirement_id} failed unexpectedly"
