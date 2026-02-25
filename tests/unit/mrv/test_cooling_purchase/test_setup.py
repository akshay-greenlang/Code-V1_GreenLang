# -*- coding: utf-8 -*-
"""
Unit tests for CoolingPurchaseService (setup.py)

AGENT-MRV-012: Cooling Purchase Agent

Tests the service facade layer that provides the main API for the cooling
purchase agent, including calculation delegation, technology specs, emission
factors, unit conversions, and health checks.

Target: 50 tests, ~450 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

import pytest

from greenlang.cooling_purchase.setup import (
    CoolingPurchaseService,
    get_cooling_purchase_service,
)
from greenlang.cooling_purchase.models import (
    CoolingTechnology,
    FreeCoolingSource,
    TESType,
    DataQualityTier,
    GWPSource,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a CoolingPurchaseService instance."""
    svc = CoolingPurchaseService()
    yield svc
    svc.reset()


# ===========================================================================
# 1. Initialization Tests
# ===========================================================================


class TestServiceInit:
    """Test CoolingPurchaseService initialization."""

    def test_singleton_pattern(self):
        """Test singleton returns same instance."""
        s1 = CoolingPurchaseService()
        s2 = CoolingPurchaseService()
        assert s1 is s2

    def test_get_function_returns_singleton(self):
        """Test get_cooling_purchase_service returns singleton."""
        s1 = get_cooling_purchase_service()
        s2 = get_cooling_purchase_service()
        assert s1 is s2

    def test_reset_clears_state(self, service):
        """Test reset clears internal state."""
        _ = service.calculate_electric_chiller(Decimal("1000"), Decimal("4.5"), Decimal("0.45"))
        service.reset()
        stats = service.get_statistics()
        assert stats["total_calculations"] == 0

    def test_get_version(self, service):
        """Test get_version returns 1.0.0."""
        version = service.get_version()
        assert version == "1.0.0"


# ===========================================================================
# 2. Electric Chiller Calculation Tests
# ===========================================================================


class TestElectricChillerCalculation:
    """Test calculate_electric_chiller delegates to engine."""

    def test_calculate_electric_chiller_basic(self, service):
        """Test basic electric chiller calculation."""
        result = service.calculate_electric_chiller(
            cooling_kwh_th=Decimal("100000"),
            cop=Decimal("5.5"),
            grid_ef_kgco2e_kwh=Decimal("0.45"),
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.technology == "WATER_COOLED_CENTRIFUGAL"

    def test_calculate_electric_chiller_with_technology(self, service):
        """Test electric chiller with specific technology."""
        result = service.calculate_electric_chiller(
            cooling_kwh_th=Decimal("50000"),
            cop=Decimal("4.0"),
            grid_ef_kgco2e_kwh=Decimal("0.45"),
            technology=CoolingTechnology.WATER_COOLED_SCREW,
        )
        assert result.technology == "WATER_COOLED_SCREW"

    def test_calculate_electric_chiller_with_iplv(self, service):
        """Test electric chiller with IPLV calculation."""
        result = service.calculate_electric_chiller(
            cooling_kwh_th=Decimal("100000"),
            cop=Decimal("5.5"),
            grid_ef_kgco2e_kwh=Decimal("0.45"),
            use_iplv=True,
        )
        # IPLV should reduce emissions compared to full-load COP
        assert result.emissions_kgco2e > Decimal("0")


# ===========================================================================
# 3. Absorption Cooling Calculation Tests
# ===========================================================================


class TestAbsorptionCoolingCalculation:
    """Test calculate_absorption_cooling delegates to engine."""

    def test_calculate_absorption_cooling_basic(self, service):
        """Test basic absorption cooling calculation."""
        result = service.calculate_absorption_cooling(
            cooling_kwh_th=Decimal("80000"),
            cop_thermal=Decimal("1.2"),
            heat_source="natural_gas",
            heat_ef_kgco2e_kwh=Decimal("0.25"),
        )
        assert result.emissions_kgco2e > Decimal("0")

    def test_calculate_absorption_with_parasitic(self, service):
        """Test absorption cooling with parasitic electricity."""
        result = service.calculate_absorption_cooling(
            cooling_kwh_th=Decimal("80000"),
            cop_thermal=Decimal("1.2"),
            heat_source="natural_gas",
            heat_ef_kgco2e_kwh=Decimal("0.25"),
            parasitic_kwh=Decimal("1000"),
            grid_ef_kgco2e_kwh=Decimal("0.45"),
        )
        # Should include parasitic load emissions
        assert result.emissions_kgco2e > Decimal("0")


# ===========================================================================
# 4. District Cooling Calculation Tests
# ===========================================================================


class TestDistrictCoolingCalculation:
    """Test calculate_district_cooling delegates to engine."""

    def test_calculate_district_cooling_basic(self, service):
        """Test basic district cooling calculation."""
        result = service.calculate_district_cooling(
            cooling_kwh_th=Decimal("100000"),
            region="singapore",
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.technology == "DISTRICT_COOLING"

    def test_calculate_district_cooling_with_losses(self, service):
        """Test district cooling with distribution losses."""
        result = service.calculate_district_cooling(
            cooling_kwh_th=Decimal("100000"),
            region="singapore",
            distribution_loss_pct=Decimal("10.0"),
        )
        # Higher losses should increase emissions
        assert result.emissions_kgco2e > Decimal("0")


# ===========================================================================
# 5. Free Cooling Calculation Tests
# ===========================================================================


class TestFreeCoolingCalculation:
    """Test calculate_free_cooling delegates to engine."""

    def test_calculate_free_cooling_seawater(self, service):
        """Test free cooling with seawater source."""
        result = service.calculate_free_cooling(
            cooling_kwh_th=Decimal("50000"),
            source=FreeCoolingSource.SEAWATER_FREE,
            grid_ef_kgco2e_kwh=Decimal("0.40"),
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.technology == "SEAWATER_FREE"

    def test_calculate_free_cooling_lake(self, service):
        """Test free cooling with lake source."""
        result = service.calculate_free_cooling(
            cooling_kwh_th=Decimal("50000"),
            source=FreeCoolingSource.LAKE_FREE,
            grid_ef_kgco2e_kwh=Decimal("0.40"),
        )
        assert result.technology == "LAKE_FREE"


# ===========================================================================
# 6. TES Calculation Tests
# ===========================================================================


class TestTESCalculation:
    """Test calculate_tes delegates to engine."""

    def test_calculate_tes_ice(self, service):
        """Test TES calculation with ice storage."""
        result = service.calculate_tes(
            cooling_kwh_th=Decimal("80000"),
            tes_type=TESType.ICE_TES,
            capacity_kwh_th=Decimal("20000"),
            cop_charge=Decimal("3.0"),
            grid_ef_charge_kgco2e_kwh=Decimal("0.30"),
            grid_ef_peak_kgco2e_kwh=Decimal("0.60"),
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.tes_savings_kgco2e is not None

    def test_calculate_tes_chilled_water(self, service):
        """Test TES calculation with chilled water storage."""
        result = service.calculate_tes(
            cooling_kwh_th=Decimal("80000"),
            tes_type=TESType.CHILLED_WATER_TES,
            capacity_kwh_th=Decimal("20000"),
            cop_charge=Decimal("5.0"),
            grid_ef_charge_kgco2e_kwh=Decimal("0.30"),
            grid_ef_peak_kgco2e_kwh=Decimal("0.60"),
        )
        assert result.emissions_kgco2e > Decimal("0")


# ===========================================================================
# 7. Technology Spec Tests
# ===========================================================================


class TestTechnologySpec:
    """Test get_technology_spec method."""

    def test_get_technology_spec_centrifugal(self, service):
        """Test get spec for centrifugal chiller."""
        spec = service.get_technology_spec("WATER_COOLED_CENTRIFUGAL")
        assert spec["cop_min"] > Decimal("0")
        assert spec["cop_max"] > spec["cop_min"]
        assert spec["iplv"] > Decimal("0")

    def test_get_technology_spec_screw(self, service):
        """Test get spec for screw chiller."""
        spec = service.get_technology_spec("WATER_COOLED_SCREW")
        assert spec["cop_min"] > Decimal("0")

    def test_get_technology_spec_absorption(self, service):
        """Test get spec for absorption chiller."""
        spec = service.get_technology_spec("DOUBLE_EFFECT_LIBR")
        assert spec["cop_min"] > Decimal("0")


# ===========================================================================
# 8. Default COP Tests
# ===========================================================================


class TestDefaultCOP:
    """Test get_default_cop method."""

    def test_get_default_cop_centrifugal(self, service):
        """Test default COP for centrifugal chiller."""
        cop = service.get_default_cop("WATER_COOLED_CENTRIFUGAL")
        assert cop > Decimal("0")
        assert cop < Decimal("10")

    def test_get_default_cop_screw(self, service):
        """Test default COP for screw chiller."""
        cop = service.get_default_cop("WATER_COOLED_SCREW")
        assert cop > Decimal("0")

    def test_get_default_cop_absorption(self, service):
        """Test default COP for absorption chiller."""
        cop = service.get_default_cop("DOUBLE_EFFECT_LIBR")
        assert cop > Decimal("0")
        assert cop < Decimal("2")


# ===========================================================================
# 9. District EF Tests
# ===========================================================================


class TestDistrictEF:
    """Test get_district_ef method."""

    def test_get_district_ef_singapore(self, service):
        """Test district EF for Singapore."""
        ef = service.get_district_ef("singapore")
        assert ef > Decimal("0")

    def test_get_district_ef_dubai(self, service):
        """Test district EF for Dubai."""
        ef = service.get_district_ef("dubai")
        assert ef > Decimal("0")

    def test_get_district_ef_copenhagen(self, service):
        """Test district EF for Copenhagen."""
        ef = service.get_district_ef("copenhagen")
        assert ef > Decimal("0")


# ===========================================================================
# 10. Heat Source EF Tests
# ===========================================================================


class TestHeatSourceEF:
    """Test get_heat_source_ef method."""

    def test_get_heat_source_ef_natural_gas(self, service):
        """Test heat source EF for natural gas."""
        ef = service.get_heat_source_ef("natural_gas")
        assert ef > Decimal("0")

    def test_get_heat_source_ef_biomass(self, service):
        """Test heat source EF for biomass."""
        ef = service.get_heat_source_ef("biomass")
        assert ef >= Decimal("0")

    def test_get_heat_source_ef_solar_thermal(self, service):
        """Test heat source EF for solar thermal."""
        ef = service.get_heat_source_ef("solar_thermal")
        assert ef >= Decimal("0")


# ===========================================================================
# 11. Refrigerant GWP Tests
# ===========================================================================


class TestRefrigerantGWP:
    """Test get_refrigerant_gwp method."""

    def test_get_refrigerant_gwp_r134a(self, service):
        """Test GWP for R-134a."""
        gwp = service.get_refrigerant_gwp("R134a")
        assert gwp > Decimal("1000")  # R-134a has high GWP

    def test_get_refrigerant_gwp_r410a(self, service):
        """Test GWP for R-410A."""
        gwp = service.get_refrigerant_gwp("R410A")
        assert gwp > Decimal("1000")

    def test_get_refrigerant_gwp_r32(self, service):
        """Test GWP for R-32."""
        gwp = service.get_refrigerant_gwp("R32")
        assert gwp < Decimal("1000")  # R-32 is lower GWP


# ===========================================================================
# 12. Efficiency Conversion Tests
# ===========================================================================


class TestEfficiencyConversion:
    """Test convert_efficiency method."""

    def test_convert_cop_to_eer(self, service):
        """Test converting COP to EER."""
        eer = service.convert_efficiency(Decimal("5.0"), "COP", "EER")
        assert eer > Decimal("0")

    def test_convert_eer_to_cop(self, service):
        """Test converting EER to COP."""
        cop = service.convert_efficiency(Decimal("17.0"), "EER", "COP")
        assert cop > Decimal("0")

    def test_convert_cop_to_kw_per_ton(self, service):
        """Test converting COP to kW/ton."""
        kw_per_ton = service.convert_efficiency(Decimal("5.0"), "COP", "KW_PER_TON")
        assert kw_per_ton > Decimal("0")


# ===========================================================================
# 13. Cooling Unit Conversion Tests
# ===========================================================================


class TestCoolingUnitConversion:
    """Test convert_cooling_units method."""

    def test_convert_ton_hour_to_kwh_th(self, service):
        """Test converting ton-hour to kWh_th."""
        kwh = service.convert_cooling_units(Decimal("100"), "TON_HOUR", "KWH_TH")
        assert kwh > Decimal("0")

    def test_convert_kwh_th_to_gj(self, service):
        """Test converting kWh_th to GJ."""
        gj = service.convert_cooling_units(Decimal("1000"), "KWH_TH", "GJ")
        assert gj > Decimal("0")

    def test_convert_btu_to_kwh_th(self, service):
        """Test converting BTU to kWh_th."""
        kwh = service.convert_cooling_units(Decimal("100000"), "BTU", "KWH_TH")
        assert kwh > Decimal("0")


# ===========================================================================
# 14. Uncertainty Quantification Tests
# ===========================================================================


class TestUncertaintyQuantification:
    """Test quantify_uncertainty method."""

    def test_quantify_uncertainty_basic(self, service):
        """Test basic uncertainty quantification."""
        result = service.quantify_uncertainty(
            total_emissions_kgco2e=Decimal("10000"),
            cooling_kwh_th=Decimal("100000"),
            cop=Decimal("5.0"),
            tier="TIER_1",
        )
        assert result["uncertainty_pct"] > Decimal("0")

    def test_quantify_uncertainty_tier1_vs_tier2(self, service):
        """Test Tier 1 has higher uncertainty than Tier 2."""
        u1 = service.quantify_uncertainty(
            total_emissions_kgco2e=Decimal("10000"),
            tier="TIER_1",
        )
        u2 = service.quantify_uncertainty(
            total_emissions_kgco2e=Decimal("10000"),
            tier="TIER_2",
        )
        assert u1["uncertainty_pct"] > u2["uncertainty_pct"]


# ===========================================================================
# 15. Compliance Check Tests
# ===========================================================================


class TestComplianceCheck:
    """Test check_compliance method."""

    def test_check_compliance_basic(self, service):
        """Test basic compliance check."""
        data = {
            "technology": "WATER_COOLED_CENTRIFUGAL",
            "cooling_kwh_th": Decimal("100000"),
            "emissions_kgco2e": Decimal("10000"),
            "cop": Decimal("5.5"),
        }
        result = service.check_compliance(data)
        assert "status" in result
        assert "framework_results" in result

    def test_check_compliance_single_framework(self, service):
        """Test compliance check for single framework."""
        data = {
            "technology": "WATER_COOLED_CENTRIFUGAL",
            "emissions_kgco2e": Decimal("10000"),
        }
        result = service.check_compliance(data, frameworks=["GHG_PROTOCOL"])
        assert len(result["framework_results"]) == 1


# ===========================================================================
# 16. Health Check Tests
# ===========================================================================


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_returns_healthy(self, service):
        """Test health check returns healthy status."""
        result = service.health_check()
        assert result["status"] == "healthy"

    def test_health_check_includes_version(self, service):
        """Test health check includes version."""
        result = service.health_check()
        assert "version" in result
        assert result["version"] == "1.0.0"


# ===========================================================================
# 17. Statistics Tests
# ===========================================================================


class TestStatistics:
    """Test statistics tracking."""

    def test_statistics_counter_increments(self, service):
        """Test statistics counter increments."""
        stats_before = service.get_statistics()
        _ = service.calculate_electric_chiller(
            Decimal("1000"), Decimal("4.5"), Decimal("0.45")
        )
        stats_after = service.get_statistics()
        assert stats_after["total_calculations"] == stats_before["total_calculations"] + 1

    def test_statistics_tracks_technology_types(self, service):
        """Test statistics tracks technology types."""
        _ = service.calculate_electric_chiller(
            Decimal("1000"), Decimal("4.5"), Decimal("0.45")
        )
        stats = service.get_statistics()
        assert "by_technology" in stats
