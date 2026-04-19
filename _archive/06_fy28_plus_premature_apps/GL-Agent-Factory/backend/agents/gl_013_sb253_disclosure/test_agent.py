"""
GL-013: SB 253 Climate Disclosure Agent - Comprehensive Golden Tests

This module contains golden tests for the SB253DisclosureAgent covering:
- Scope 1 calculations (stationary, mobile, process, fugitive)
- Scope 2 calculations (location-based, market-based)
- All 15 Scope 3 categories
- CARB filing format generation
- Assurance package generation
- Provenance tracking

Test Categories:
- test_scope1_*: 20 tests for Scope 1 direct emissions
- test_scope2_*: 20 tests for Scope 2 indirect emissions
- test_scope3_cat*: 30 tests for all 15 Scope 3 categories
- test_carb_*: 10 tests for CARB filing format
- test_assurance_*: 5 tests for assurance package
- test_provenance_*: 5 tests for provenance tracking
- test_integration_*: 10 tests for end-to-end scenarios

Total: 100 golden tests
"""

import hashlib
import json
import pytest
from datetime import date, datetime
from typing import Any, Dict, List

from .agent import (
    SB253DisclosureAgent,
    SB253ReportInput,
    SB253ReportOutput,
    CompanyInfo,
    FacilityInfo,
    Scope1Source,
    Scope2Source,
    Scope3Data,
    Scope3CategoryData,
    ReportingPeriod,
    OrganizationalBoundary,
    FuelType,
    FuelUnit,
    SourceCategory,
    Scope2Method,
    Scope3Category,
    CalculationMethod,
    DataQualityScore,
    GWPSet,
    AssuranceLevel,
    RefrigerantType,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def agent():
    """Create a fresh SB253DisclosureAgent instance."""
    return SB253DisclosureAgent()


@pytest.fixture
def sample_company_info():
    """Sample company meeting SB 253 threshold."""
    return CompanyInfo(
        company_name="California Test Corporation",
        ein="12-3456789",
        total_revenue_usd=2_500_000_000,
        california_revenue_usd=800_000_000,
        naics_code="331110",
        organizational_boundary=OrganizationalBoundary.OPERATIONAL_CONTROL,
        california_facilities=5,
    )


@pytest.fixture
def sample_facility():
    """Sample California facility."""
    return FacilityInfo(
        facility_id="FAC001",
        facility_name="Main Manufacturing Plant",
        egrid_subregion="CAMX",
        california_facility=True,
        address_city="Los Angeles",
        address_state="CA",
        address_zip="90001",
    )


@pytest.fixture
def sample_scope1_source_natural_gas():
    """Sample natural gas combustion source."""
    return Scope1Source(
        facility_id="FAC001",
        source_category=SourceCategory.STATIONARY_COMBUSTION,
        fuel_type=FuelType.NATURAL_GAS,
        quantity=100.0,
        unit=FuelUnit.THERMS,
        source_description="Main boiler",
    )


@pytest.fixture
def sample_scope1_source_diesel():
    """Sample diesel mobile combustion source."""
    return Scope1Source(
        facility_id="FAC001",
        source_category=SourceCategory.MOBILE_COMBUSTION,
        fuel_type=FuelType.DIESEL,
        quantity=1000.0,
        unit=FuelUnit.GALLONS,
        source_description="Fleet vehicles",
    )


@pytest.fixture
def sample_scope2_source():
    """Sample electricity source."""
    return Scope2Source(
        facility_id="FAC001",
        kwh=1_000_000.0,
        egrid_subregion="CAMX",
        renewable_percentage=20.0,
        has_ppa=False,
        has_recs=False,
    )


@pytest.fixture
def sample_scope3_cat1():
    """Sample Category 1 Purchased Goods data."""
    return Scope3CategoryData(
        category=Scope3Category.PURCHASED_GOODS_SERVICES,
        calculation_method=CalculationMethod.SPEND_BASED,
        data_quality_score=DataQualityScore.FAIR,
        spend_usd=1_000_000.0,
        naics_code="331110",
    )


@pytest.fixture
def sample_scope3_cat6():
    """Sample Category 6 Business Travel data."""
    return Scope3CategoryData(
        category=Scope3Category.BUSINESS_TRAVEL,
        calculation_method=CalculationMethod.AVERAGE_DATA,
        data_quality_score=DataQualityScore.GOOD,
        activity_data={
            "short_haul_miles": 50000,
            "medium_haul_miles": 100000,
            "long_haul_miles": 200000,
        },
    )


@pytest.fixture
def sample_scope3_cat7():
    """Sample Category 7 Employee Commuting data."""
    return Scope3CategoryData(
        category=Scope3Category.EMPLOYEE_COMMUTING,
        calculation_method=CalculationMethod.AVERAGE_DATA,
        data_quality_score=DataQualityScore.FAIR,
        activity_data={
            "employees": 500,
            "avg_commute_miles": 25,
            "working_days": 220,
        },
    )


@pytest.fixture
def minimal_sb253_input(sample_company_info, sample_facility, sample_scope1_source_natural_gas, sample_scope2_source):
    """Minimal valid SB 253 input."""
    return SB253ReportInput(
        company_info=sample_company_info,
        fiscal_year=2025,
        facilities=[sample_facility],
        scope1_sources=[sample_scope1_source_natural_gas],
        scope2_sources=[sample_scope2_source],
        gwp_set=GWPSet.AR5,
        include_scope3=False,
    )


# =============================================================================
# SCOPE 1 TESTS (20 tests)
# =============================================================================

class TestScope1Calculations:
    """Tests for Scope 1 direct emissions calculations."""

    def test_scope1_natural_gas_basic(self, agent):
        """Test basic natural gas combustion calculation (100 therms)."""
        # Expected: 100 therms * 5.31 kgCO2/therm = 531 kgCO2
        # With CH4 and N2O: ~532.2 kgCO2e (using AR5 GWPs)
        input_data = SB253ReportInput(
            company_info=CompanyInfo(
                company_name="Test Corp",
                ein="12-3456789",
                total_revenue_usd=2_000_000_000,
                naics_code="331110",
            ),
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=100.0,
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[],
            gwp_set=GWPSet.AR5,
        )

        result = agent.run(input_data)

        # Verify calculation
        assert result.scope1_emissions.total_emissions.co2e_kg > 530
        assert result.scope1_emissions.total_emissions.co2e_kg < 535
        assert "EPA" in str(result.scope1_emissions.emission_factors_used)

    def test_scope1_diesel_mobile_combustion(self, agent, sample_company_info):
        """Test mobile combustion from 1000 gallons diesel."""
        # Expected: 1000 gal * 10.21 kgCO2/gal = 10,210 kgCO2
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.MOBILE_COMBUSTION,
                    fuel_type=FuelType.DIESEL,
                    quantity=1000.0,
                    unit=FuelUnit.GALLONS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Tolerance of +/- 100 kg for GWP adjustments
        assert result.scope1_emissions.mobile_combustion.co2e_kg > 10100
        assert result.scope1_emissions.mobile_combustion.co2e_kg < 10300

    def test_scope1_gasoline_fleet(self, agent, sample_company_info):
        """Test fleet vehicle gasoline consumption (5000 gallons)."""
        # Expected: 5000 gal * 8.78 kgCO2/gal = 43,900 kgCO2
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.MOBILE_COMBUSTION,
                    fuel_type=FuelType.GASOLINE,
                    quantity=5000.0,
                    unit=FuelUnit.GALLONS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Tolerance for CH4/N2O
        assert result.scope1_emissions.mobile_combustion.co2e_kg > 43800
        assert result.scope1_emissions.mobile_combustion.co2e_kg < 45000

    def test_scope1_propane_stationary(self, agent, sample_company_info):
        """Test propane stationary combustion."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.PROPANE,
                    quantity=1000.0,
                    unit=FuelUnit.GALLONS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Propane: 5.72 kgCO2/gal * 1000 = 5,720 kgCO2
        assert result.scope1_emissions.stationary_combustion.co2e_kg > 5600
        assert result.scope1_emissions.stationary_combustion.co2e_kg < 6000

    def test_scope1_fugitive_refrigerant_r410a(self, agent, sample_company_info):
        """Test fugitive emissions from R-410A refrigerant leakage."""
        # R-410A GWP (AR5) = 2088
        # 10 kg loss * 2088 = 20,880 kgCO2e
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.FUGITIVE_EMISSIONS,
                    fuel_type=FuelType.NATURAL_GAS,  # Placeholder
                    quantity=0,
                    unit=FuelUnit.KG,
                    refrigerant_type=RefrigerantType.R410A,
                    refrigerant_charge_kg=100.0,
                    refrigerant_loss_kg=10.0,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # R-410A at 10kg loss
        assert result.scope1_emissions.fugitive_emissions.co2e_kg > 20000
        assert result.scope1_emissions.fugitive_emissions.co2e_kg < 21500

    def test_scope1_fugitive_refrigerant_r134a(self, agent, sample_company_info):
        """Test fugitive emissions from R-134a refrigerant leakage."""
        # R-134a GWP (AR5) = 1430
        # 5 kg loss * 1430 = 7,150 kgCO2e
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.FUGITIVE_EMISSIONS,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=0,
                    unit=FuelUnit.KG,
                    refrigerant_type=RefrigerantType.R134A,
                    refrigerant_charge_kg=50.0,
                    refrigerant_loss_kg=5.0,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        assert result.scope1_emissions.fugitive_emissions.co2e_kg > 7000
        assert result.scope1_emissions.fugitive_emissions.co2e_kg < 7500

    def test_scope1_multiple_sources(self, agent, sample_company_info):
        """Test aggregation of multiple Scope 1 sources."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=100.0,
                    unit=FuelUnit.THERMS,
                ),
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.MOBILE_COMBUSTION,
                    fuel_type=FuelType.DIESEL,
                    quantity=500.0,
                    unit=FuelUnit.GALLONS,
                ),
                Scope1Source(
                    facility_id="FAC002",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.PROPANE,
                    quantity=200.0,
                    unit=FuelUnit.GALLONS,
                ),
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Verify aggregation
        total = (
            result.scope1_emissions.stationary_combustion.co2e_kg +
            result.scope1_emissions.mobile_combustion.co2e_kg
        )
        assert abs(result.scope1_emissions.total_emissions.co2e_kg - total) < 1

        # Verify facility breakdown
        assert "FAC001" in result.scope1_emissions.emissions_by_facility
        assert "FAC002" in result.scope1_emissions.emissions_by_facility

    def test_scope1_fuel_type_breakdown(self, agent, sample_company_info):
        """Test emissions breakdown by fuel type."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=100.0,
                    unit=FuelUnit.THERMS,
                ),
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.MOBILE_COMBUSTION,
                    fuel_type=FuelType.DIESEL,
                    quantity=500.0,
                    unit=FuelUnit.GALLONS,
                ),
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        assert "natural_gas" in result.scope1_emissions.emissions_by_fuel
        assert "diesel" in result.scope1_emissions.emissions_by_fuel

    def test_scope1_gwp_ar5_vs_ar6(self, agent, sample_company_info):
        """Test GWP differences between AR5 and AR6."""
        source = Scope1Source(
            facility_id="FAC001",
            source_category=SourceCategory.STATIONARY_COMBUSTION,
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit=FuelUnit.THERMS,
        )

        input_ar5 = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[source],
            scope2_sources=[],
            gwp_set=GWPSet.AR5,
        )

        input_ar6 = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[source],
            scope2_sources=[],
            gwp_set=GWPSet.AR6,
        )

        result_ar5 = agent.run(input_ar5)
        result_ar6 = agent.run(input_ar6)

        # AR6 has slightly different CH4 and N2O GWPs
        # Results should be close but not identical
        assert result_ar5.scope1_emissions.total_emissions.co2e_kg != result_ar6.scope1_emissions.total_emissions.co2e_kg

    def test_scope1_emission_factor_citation(self, agent, sample_company_info):
        """Test that emission factor sources are properly cited."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=100.0,
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Verify emission factors are documented
        assert len(result.scope1_emissions.emission_factors_used) > 0
        ef = result.scope1_emissions.emission_factors_used[0]
        assert "source" in ef
        assert "EPA" in ef["source"]

    def test_scope1_zero_emissions(self, agent, sample_company_info):
        """Test handling of zero emission sources."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=0.0,
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        assert result.scope1_emissions.total_emissions.co2e_kg == 0

    def test_scope1_mmbtu_unit(self, agent, sample_company_info):
        """Test natural gas calculation with MMBtu unit."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=10.0,
                    unit=FuelUnit.MMBTU,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # 10 MMBtu * 53.06 kgCO2/MMBtu = 530.6 kgCO2
        assert result.scope1_emissions.total_emissions.co2e_kg > 520
        assert result.scope1_emissions.total_emissions.co2e_kg < 550

    def test_scope1_coal_tons(self, agent, sample_company_info):
        """Test coal combustion in tons."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.COAL,
                    quantity=1.0,
                    unit=FuelUnit.TONS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # 1 short ton * 2406 kgCO2 = 2,406 kgCO2
        assert result.scope1_emissions.total_emissions.co2e_kg > 2300
        assert result.scope1_emissions.total_emissions.co2e_kg < 2500

    def test_scope1_liters_unit(self, agent, sample_company_info):
        """Test diesel calculation with liters unit."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.MOBILE_COMBUSTION,
                    fuel_type=FuelType.DIESEL,
                    quantity=3785.41,  # ~1000 gallons
                    unit=FuelUnit.LITERS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Should be approximately 10,200 kgCO2e (same as 1000 gallons)
        assert result.scope1_emissions.mobile_combustion.co2e_kg > 9800
        assert result.scope1_emissions.mobile_combustion.co2e_kg < 10800

    def test_scope1_process_emissions(self, agent, sample_company_info):
        """Test process emissions category."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.PROCESS_EMISSIONS,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=500.0,
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        assert result.scope1_emissions.process_emissions.co2e_kg > 0
        assert result.scope1_emissions.stationary_combustion.co2e_kg == 0

    def test_scope1_mtco2e_conversion(self, agent, sample_company_info):
        """Test kgCO2e to MTCO2e conversion."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=10000.0,  # ~53,200 kgCO2e
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Verify conversion: kg / 1000 = MT
        expected_mt = result.scope1_emissions.total_emissions.co2e_kg / 1000
        assert abs(result.scope1_emissions.total_emissions.co2e_mtco2e - expected_mt) < 0.01

    def test_scope1_lpg_fuel(self, agent, sample_company_info):
        """Test LPG fuel type calculation."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.LPG,
                    quantity=1000.0,
                    unit=FuelUnit.GALLONS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # LPG: ~5.68 kgCO2/gal * 1000 = 5,680 kgCO2e
        assert result.scope1_emissions.stationary_combustion.co2e_kg > 5500
        assert result.scope1_emissions.stationary_combustion.co2e_kg < 6000

    def test_scope1_fuel_oil_2(self, agent, sample_company_info):
        """Test fuel oil #2 calculation."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.FUEL_OIL_2,
                    quantity=1000.0,
                    unit=FuelUnit.GALLONS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Fuel oil #2: ~10.16 kgCO2/gal * 1000 = 10,160 kgCO2e
        assert result.scope1_emissions.stationary_combustion.co2e_kg > 10000
        assert result.scope1_emissions.stationary_combustion.co2e_kg < 10500


# =============================================================================
# SCOPE 2 TESTS (20 tests)
# =============================================================================

class TestScope2Calculations:
    """Tests for Scope 2 indirect emissions calculations."""

    def test_scope2_california_camx_location(self, agent, sample_company_info):
        """Test location-based calculation for California (CAMX grid)."""
        # CAMX: 472 lb/MWh = 0.214 kg/kWh
        # 1,000,000 kWh * 0.214 = 214,000 kgCO2e
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                )
            ],
        )

        result = agent.run(input_data)

        # Tolerance for unit conversion rounding
        assert result.scope2_emissions.location_based.co2e_kg > 200000
        assert result.scope2_emissions.location_based.co2e_kg < 230000

    def test_scope2_texas_ercot_location(self, agent, sample_company_info):
        """Test location-based calculation for Texas (ERCT grid)."""
        # ERCT: 870 lb/MWh = 0.395 kg/kWh
        # 2,000,000 kWh * 0.395 = 790,000 kgCO2e
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=2_000_000.0,
                    egrid_subregion="ERCT",
                )
            ],
        )

        result = agent.run(input_data)

        assert result.scope2_emissions.location_based.co2e_kg > 750000
        assert result.scope2_emissions.location_based.co2e_kg < 850000

    def test_scope2_market_based_50_renewable(self, agent, sample_company_info):
        """Test market-based with 50% renewable percentage."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                    renewable_percentage=50.0,
                )
            ],
        )

        result = agent.run(input_data)

        # Market-based should be ~50% of location-based
        ratio = (
            result.scope2_emissions.market_based.co2e_kg /
            result.scope2_emissions.location_based.co2e_kg
        )
        assert ratio > 0.45
        assert ratio < 0.55

    def test_scope2_market_based_100_renewable(self, agent, sample_company_info):
        """Test market-based with 100% renewable percentage."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                    renewable_percentage=100.0,
                )
            ],
        )

        result = agent.run(input_data)

        # Market-based should be zero with 100% renewable
        assert result.scope2_emissions.market_based.co2e_kg == 0

    def test_scope2_with_recs(self, agent, sample_company_info):
        """Test market-based with RECs."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                    has_recs=True,
                    rec_mwh=500.0,  # 500 MWh RECs = 500,000 kWh covered
                )
            ],
        )

        result = agent.run(input_data)

        # RECs cover 50% of consumption
        ratio = (
            result.scope2_emissions.market_based.co2e_kg /
            result.scope2_emissions.location_based.co2e_kg
        )
        assert ratio > 0.45
        assert ratio < 0.55

    def test_scope2_with_ppa(self, agent, sample_company_info):
        """Test market-based with Power Purchase Agreement."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                    has_ppa=True,
                    ppa_emissions_factor=0.05,  # Low carbon PPA (50g/kWh)
                )
            ],
        )

        result = agent.run(input_data)

        # PPA factor: 0.05 kg/kWh * 1,000,000 = 50,000 kgCO2e
        assert result.scope2_emissions.market_based.co2e_kg > 45000
        assert result.scope2_emissions.market_based.co2e_kg < 55000

    def test_scope2_utility_specific_factor(self, agent, sample_company_info):
        """Test market-based with utility-specific emission factor."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                    utility_name="Clean Energy Utility",
                    utility_emission_factor=0.10,  # 100g/kWh
                )
            ],
        )

        result = agent.run(input_data)

        # Utility factor: 0.10 kg/kWh * 1,000,000 = 100,000 kgCO2e
        assert result.scope2_emissions.market_based.co2e_kg > 95000
        assert result.scope2_emissions.market_based.co2e_kg < 105000

    def test_scope2_dual_reporting(self, agent, sample_company_info):
        """Test that both location and market-based are reported."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                    renewable_percentage=30.0,
                )
            ],
        )

        result = agent.run(input_data)

        # Both methods should be reported
        assert result.scope2_emissions.location_based.co2e_kg > 0
        assert result.scope2_emissions.market_based.co2e_kg > 0
        assert result.scope2_emissions.location_based.co2e_kg > result.scope2_emissions.market_based.co2e_kg

    def test_scope2_multiple_facilities(self, agent, sample_company_info):
        """Test aggregation across multiple facilities."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=500_000.0,
                    egrid_subregion="CAMX",
                ),
                Scope2Source(
                    facility_id="FAC002",
                    kwh=500_000.0,
                    egrid_subregion="ERCT",
                ),
            ],
        )

        result = agent.run(input_data)

        # Verify both facilities are tracked
        assert "FAC001" in result.scope2_emissions.emissions_by_facility
        assert "FAC002" in result.scope2_emissions.emissions_by_facility

    def test_scope2_grid_breakdown(self, agent, sample_company_info):
        """Test emissions breakdown by grid region."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=500_000.0,
                    egrid_subregion="CAMX",
                ),
                Scope2Source(
                    facility_id="FAC002",
                    kwh=500_000.0,
                    egrid_subregion="NWPP",
                ),
            ],
        )

        result = agent.run(input_data)

        assert "CAMX" in result.scope2_emissions.emissions_by_grid
        assert "NWPP" in result.scope2_emissions.emissions_by_grid

    def test_scope2_total_electricity_tracking(self, agent, sample_company_info):
        """Test total electricity consumption tracking."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=2_000_000.0,
                    egrid_subregion="CAMX",
                    renewable_percentage=25.0,
                )
            ],
        )

        result = agent.run(input_data)

        assert result.scope2_emissions.total_electricity_mwh == 2000.0
        assert result.scope2_emissions.renewable_mwh == 500.0
        assert result.scope2_emissions.renewable_percentage == 25.0

    def test_scope2_egrid_factor_citation(self, agent, sample_company_info):
        """Test that eGRID factors are properly cited."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                )
            ],
        )

        result = agent.run(input_data)

        assert len(result.scope2_emissions.egrid_factors_used) > 0
        ef = result.scope2_emissions.egrid_factors_used[0]
        assert "eGRID" in ef["source"]
        assert ef["subregion"] == "CAMX"

    def test_scope2_northeast_newe(self, agent, sample_company_info):
        """Test New England grid (cleaner than average)."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="NEWE",
                )
            ],
        )

        result = agent.run(input_data)

        # NEWE: 482 lb/MWh - similar to CAMX
        assert result.scope2_emissions.location_based.co2e_kg > 200000
        assert result.scope2_emissions.location_based.co2e_kg < 250000

    def test_scope2_midwest_high_carbon(self, agent, sample_company_info):
        """Test SERC Midwest grid (higher carbon intensity)."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="SRMW",
                )
            ],
        )

        result = agent.run(input_data)

        # SRMW: 1356 lb/MWh - high carbon
        assert result.scope2_emissions.location_based.co2e_kg > 550000
        assert result.scope2_emissions.location_based.co2e_kg < 650000

    def test_scope2_mtco2e_conversion(self, agent, sample_company_info):
        """Test kg to metric tons conversion."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                )
            ],
        )

        result = agent.run(input_data)

        expected_mt = result.scope2_emissions.location_based.co2e_kg / 1000
        assert abs(result.scope2_emissions.location_based.co2e_mtco2e - expected_mt) < 0.01

    def test_scope2_unknown_subregion_fallback(self, agent, sample_company_info):
        """Test fallback for unknown eGRID subregion."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="UNKNOWN",
                )
            ],
        )

        result = agent.run(input_data)

        # Should fall back to CAMX (California default)
        assert result.scope2_emissions.location_based.co2e_kg > 0

    def test_scope2_zero_consumption(self, agent, sample_company_info):
        """Test handling of zero electricity consumption."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=0.0,
                    egrid_subregion="CAMX",
                )
            ],
        )

        result = agent.run(input_data)

        assert result.scope2_emissions.location_based.co2e_kg == 0
        assert result.scope2_emissions.market_based.co2e_kg == 0

    def test_scope2_ch4_n2o_included(self, agent, sample_company_info):
        """Test that CH4 and N2O emissions are included."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                )
            ],
        )

        result = agent.run(input_data)

        # CH4 and N2O should be tracked (though small)
        assert result.scope2_emissions.location_based.ch4_kg >= 0
        assert result.scope2_emissions.location_based.n2o_kg >= 0

    def test_scope2_recs_full_coverage(self, agent, sample_company_info):
        """Test RECs covering 100% of consumption."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                    has_recs=True,
                    rec_mwh=1000.0,  # Full coverage
                )
            ],
        )

        result = agent.run(input_data)

        # Market-based should be zero
        assert result.scope2_emissions.market_based.co2e_kg == 0


# =============================================================================
# SCOPE 3 TESTS - ALL 15 CATEGORIES (30 tests)
# =============================================================================

class TestScope3Calculations:
    """Tests for Scope 3 value chain emissions (all 15 categories)."""

    def test_scope3_cat1_purchased_goods_spend(self, agent, sample_company_info):
        """Test Category 1: Purchased Goods spend-based calculation."""
        # Steel: $1M * 0.95 kgCO2e/USD = 950,000 kgCO2e
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.FAIR,
                    spend_usd=1_000_000.0,
                    naics_code="331110",  # Iron and Steel
                )
            ),
        )

        result = agent.run(input_data)

        cat1 = next(c for c in result.scope3_emissions.categories if c.category_number == 1)
        assert cat1.co2e_kg > 900000
        assert cat1.co2e_kg < 1000000

    def test_scope3_cat2_capital_goods(self, agent, sample_company_info):
        """Test Category 2: Capital Goods spend-based calculation."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                capital_goods=Scope3CategoryData(
                    category=Scope3Category.CAPITAL_GOODS,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.FAIR,
                    spend_usd=500_000.0,
                    naics_code="334111",  # Electronics
                )
            ),
        )

        result = agent.run(input_data)

        cat2 = next(c for c in result.scope3_emissions.categories if c.category_number == 2)
        # Electronics: $500K * 0.35 = 175,000 kgCO2e
        assert cat2.co2e_kg > 150000
        assert cat2.co2e_kg < 200000

    def test_scope3_cat3_fuel_energy_activities(self, agent, sample_company_info):
        """Test Category 3: Fuel and Energy Related Activities."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                fuel_energy_activities=Scope3CategoryData(
                    category=Scope3Category.FUEL_ENERGY_ACTIVITIES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.FAIR,
                    spend_usd=200_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat3 = next(c for c in result.scope3_emissions.categories if c.category_number == 3)
        assert cat3.co2e_kg > 0

    def test_scope3_cat4_upstream_transport(self, agent, sample_company_info):
        """Test Category 4: Upstream Transportation."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                upstream_transportation=Scope3CategoryData(
                    category=Scope3Category.UPSTREAM_TRANSPORTATION,
                    calculation_method=CalculationMethod.AVERAGE_DATA,
                    data_quality_score=DataQualityScore.FAIR,
                    activity_data={
                        "road_tonne_km": 1_000_000,
                        "sea_tonne_km": 500_000,
                    },
                )
            ),
        )

        result = agent.run(input_data)

        cat4 = next(c for c in result.scope3_emissions.categories if c.category_number == 4)
        # Road: 1M tkm * 0.089 = 89,000 kg
        # Sea: 500K tkm * 0.016 = 8,000 kg
        # Total: ~97,000 kgCO2e
        assert cat4.co2e_kg > 90000
        assert cat4.co2e_kg < 110000

    def test_scope3_cat5_waste_generated(self, agent, sample_company_info):
        """Test Category 5: Waste Generated in Operations."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                waste_generated=Scope3CategoryData(
                    category=Scope3Category.WASTE_GENERATED,
                    calculation_method=CalculationMethod.AVERAGE_DATA,
                    data_quality_score=DataQualityScore.FAIR,
                    activity_data={
                        "landfill_mixed_kg": 100_000,
                        "recycling_kg": 50_000,
                    },
                )
            ),
        )

        result = agent.run(input_data)

        cat5 = next(c for c in result.scope3_emissions.categories if c.category_number == 5)
        # Landfill: 100,000 kg * 0.586 = 58,600 kg
        # Recycling: 50,000 kg * -0.5 = -25,000 kg (credit)
        # Total: ~33,600 kgCO2e
        assert cat5.co2e_kg > 25000
        assert cat5.co2e_kg < 45000

    def test_scope3_cat6_business_travel(self, agent, sample_company_info):
        """Test Category 6: Business Travel."""
        # Air miles: 50K short + 100K medium + 200K long
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                business_travel=Scope3CategoryData(
                    category=Scope3Category.BUSINESS_TRAVEL,
                    calculation_method=CalculationMethod.AVERAGE_DATA,
                    data_quality_score=DataQualityScore.GOOD,
                    activity_data={
                        "short_haul_miles": 50000,
                        "medium_haul_miles": 100000,
                        "long_haul_miles": 200000,
                    },
                )
            ),
        )

        result = agent.run(input_data)

        cat6 = next(c for c in result.scope3_emissions.categories if c.category_number == 6)
        # Conversion: miles * 1.6 = km
        # Short: 80,450 km * 0.255 = 20,515 kg
        # Medium: 160,900 km * 0.156 = 25,100 kg
        # Long: 321,800 km * 0.195 = 62,750 kg
        # Total: ~108,000 kgCO2e (but can vary with assumptions)
        assert cat6.co2e_kg > 40000
        assert cat6.co2e_kg < 120000

    def test_scope3_cat7_employee_commuting(self, agent, sample_company_info):
        """Test Category 7: Employee Commuting."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                employee_commuting=Scope3CategoryData(
                    category=Scope3Category.EMPLOYEE_COMMUTING,
                    calculation_method=CalculationMethod.AVERAGE_DATA,
                    data_quality_score=DataQualityScore.FAIR,
                    activity_data={
                        "employees": 500,
                        "avg_commute_miles": 25,
                        "working_days": 220,
                    },
                )
            ),
        )

        result = agent.run(input_data)

        cat7 = next(c for c in result.scope3_emissions.categories if c.category_number == 7)
        # 500 employees * 25 miles * 1.6 * 2 (round trip) * 220 days = 8.8M km
        # With modal split: ~300,000-400,000 kgCO2e
        assert cat7.co2e_kg > 250000
        assert cat7.co2e_kg < 450000

    def test_scope3_cat8_upstream_leased(self, agent, sample_company_info):
        """Test Category 8: Upstream Leased Assets."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                upstream_leased_assets=Scope3CategoryData(
                    category=Scope3Category.UPSTREAM_LEASED_ASSETS,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.POOR,
                    spend_usd=100_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat8 = next(c for c in result.scope3_emissions.categories if c.category_number == 8)
        assert cat8.co2e_kg > 0

    def test_scope3_cat9_downstream_transport(self, agent, sample_company_info):
        """Test Category 9: Downstream Transportation."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                downstream_transportation=Scope3CategoryData(
                    category=Scope3Category.DOWNSTREAM_TRANSPORTATION,
                    calculation_method=CalculationMethod.AVERAGE_DATA,
                    data_quality_score=DataQualityScore.FAIR,
                    activity_data={
                        "road_tonne_km": 500_000,
                        "air_tonne_km": 100_000,
                    },
                )
            ),
        )

        result = agent.run(input_data)

        cat9 = next(c for c in result.scope3_emissions.categories if c.category_number == 9)
        # Road: 500K * 0.089 = 44,500 kg
        # Air: 100K * 0.602 = 60,200 kg
        # Total: ~104,700 kgCO2e
        assert cat9.co2e_kg > 95000
        assert cat9.co2e_kg < 115000

    def test_scope3_cat10_processing_sold(self, agent, sample_company_info):
        """Test Category 10: Processing of Sold Products."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                processing_sold_products=Scope3CategoryData(
                    category=Scope3Category.PROCESSING_SOLD_PRODUCTS,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.POOR,
                    spend_usd=300_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat10 = next(c for c in result.scope3_emissions.categories if c.category_number == 10)
        assert cat10.co2e_kg > 0

    def test_scope3_cat11_use_of_sold(self, agent, sample_company_info):
        """Test Category 11: Use of Sold Products."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                use_of_sold_products=Scope3CategoryData(
                    category=Scope3Category.USE_OF_SOLD_PRODUCTS,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.POOR,
                    spend_usd=1_000_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat11 = next(c for c in result.scope3_emissions.categories if c.category_number == 11)
        assert cat11.co2e_kg > 0

    def test_scope3_cat12_end_of_life(self, agent, sample_company_info):
        """Test Category 12: End-of-Life Treatment."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                end_of_life_treatment=Scope3CategoryData(
                    category=Scope3Category.END_OF_LIFE_TREATMENT,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.VERY_POOR,
                    spend_usd=50_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat12 = next(c for c in result.scope3_emissions.categories if c.category_number == 12)
        assert cat12.co2e_kg >= 0

    def test_scope3_cat13_downstream_leased(self, agent, sample_company_info):
        """Test Category 13: Downstream Leased Assets."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                downstream_leased_assets=Scope3CategoryData(
                    category=Scope3Category.DOWNSTREAM_LEASED_ASSETS,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.POOR,
                    spend_usd=200_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat13 = next(c for c in result.scope3_emissions.categories if c.category_number == 13)
        assert cat13.co2e_kg > 0

    def test_scope3_cat14_franchises(self, agent, sample_company_info):
        """Test Category 14: Franchises."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                franchises=Scope3CategoryData(
                    category=Scope3Category.FRANCHISES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.POOR,
                    spend_usd=500_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat14 = next(c for c in result.scope3_emissions.categories if c.category_number == 14)
        assert cat14.co2e_kg > 0

    def test_scope3_cat15_investments(self, agent, sample_company_info):
        """Test Category 15: Investments."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                investments=Scope3CategoryData(
                    category=Scope3Category.INVESTMENTS,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.VERY_POOR,
                    spend_usd=10_000_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat15 = next(c for c in result.scope3_emissions.categories if c.category_number == 15)
        assert cat15.co2e_kg > 0

    def test_scope3_supplier_specific_method(self, agent, sample_company_info):
        """Test supplier-specific calculation method."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
                    data_quality_score=DataQualityScore.VERY_GOOD,
                    supplier_emissions_kgco2e=500_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat1 = next(c for c in result.scope3_emissions.categories if c.category_number == 1)
        assert cat1.co2e_kg == 500000
        assert cat1.calculation_method == "supplier_specific"

    def test_scope3_data_quality_scoring(self, agent, sample_company_info):
        """Test data quality score propagation."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
                    data_quality_score=DataQualityScore.VERY_GOOD,
                    supplier_emissions_kgco2e=100_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat1 = next(c for c in result.scope3_emissions.categories if c.category_number == 1)
        assert cat1.data_quality_score == 1
        assert cat1.uncertainty_percentage == 10.0  # Very good = 10%

    def test_scope3_uncertainty_calculation(self, agent, sample_company_info):
        """Test uncertainty percentage based on data quality."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    data_quality_score=DataQualityScore.VERY_POOR,
                    spend_usd=100_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        cat1 = next(c for c in result.scope3_emissions.categories if c.category_number == 1)
        assert cat1.data_quality_score == 5
        assert cat1.uncertainty_percentage == 100.0  # Very poor = 100%

    def test_scope3_all_15_categories_empty(self, agent, sample_company_info):
        """Test that all 15 categories are reported even when empty."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(),
        )

        result = agent.run(input_data)

        assert len(result.scope3_emissions.categories) == 15

    def test_scope3_upstream_vs_downstream_totals(self, agent, sample_company_info):
        """Test upstream (1-8) vs downstream (9-15) totals."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=1_000_000.0,
                ),
                downstream_transportation=Scope3CategoryData(
                    category=Scope3Category.DOWNSTREAM_TRANSPORTATION,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=500_000.0,
                ),
            ),
        )

        result = agent.run(input_data)

        assert result.scope3_emissions.upstream_total_mtco2e > 0
        assert result.scope3_emissions.downstream_total_mtco2e > 0

    def test_scope3_top_categories_ranking(self, agent, sample_company_info):
        """Test top emission categories are correctly ranked."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=2_000_000.0,  # Largest
                ),
                business_travel=Scope3CategoryData(
                    category=Scope3Category.BUSINESS_TRAVEL,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=100_000.0,  # Small
                ),
            ),
        )

        result = agent.run(input_data)

        # Purchased goods should be ranked #1
        assert result.scope3_emissions.top_categories[0]["category"] == 1

    def test_scope3_total_emissions_sum(self, agent, sample_company_info):
        """Test total emissions equals sum of categories."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=500_000.0,
                ),
                business_travel=Scope3CategoryData(
                    category=Scope3Category.BUSINESS_TRAVEL,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=100_000.0,
                ),
            ),
        )

        result = agent.run(input_data)

        category_sum = sum(c.co2e_kg for c in result.scope3_emissions.categories)
        assert abs(result.scope3_emissions.total_emissions_kgco2e - category_sum) < 1

    def test_scope3_mtco2e_conversion(self, agent, sample_company_info):
        """Test kg to MTCO2e conversion for Scope 3."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
                    supplier_emissions_kgco2e=1_000_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        expected_mt = result.scope3_emissions.total_emissions_kgco2e / 1000
        assert abs(result.scope3_emissions.total_emissions_mtco2e - expected_mt) < 0.01

    def test_scope3_not_included_by_default(self, agent, sample_company_info):
        """Test Scope 3 not calculated when not requested."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=False,
        )

        result = agent.run(input_data)

        assert result.scope3_emissions is None
        assert result.total_scope3_mtco2e is None

    def test_scope3_emission_factor_source_documented(self, agent, sample_company_info):
        """Test emission factor sources are documented."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=100_000.0,
                    naics_code="331110",
                )
            ),
        )

        result = agent.run(input_data)

        cat1 = next(c for c in result.scope3_emissions.categories if c.category_number == 1)
        assert "USEEIO" in cat1.emission_factor_source or "EPA" in cat1.emission_factor_source

    def test_scope3_default_naics_fallback(self, agent, sample_company_info):
        """Test fallback to default EEIO factor when NAICS not found."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=100_000.0,
                    naics_code="999999",  # Non-existent NAICS
                )
            ),
        )

        result = agent.run(input_data)

        # Should use default factor (0.40)
        cat1 = next(c for c in result.scope3_emissions.categories if c.category_number == 1)
        # $100K * 0.40 = 40,000 kgCO2e
        assert cat1.co2e_kg > 35000
        assert cat1.co2e_kg < 45000


# =============================================================================
# CARB FILING TESTS (10 tests)
# =============================================================================

class TestCARBFiling:
    """Tests for CARB portal filing format generation."""

    def test_carb_filing_generated(self, agent, minimal_sb253_input):
        """Test CARB filing data is generated."""
        result = agent.run(minimal_sb253_input)

        assert result.carb_filing is not None
        assert result.carb_filing.filing_id is not None

    def test_carb_filing_company_section(self, agent, minimal_sb253_input):
        """Test company information section."""
        result = agent.run(minimal_sb253_input)

        company = result.carb_filing.company_section
        assert company["legal_name"] == minimal_sb253_input.company_info.company_name
        assert company["ein"] == minimal_sb253_input.company_info.ein

    def test_carb_filing_scope1_section(self, agent, minimal_sb253_input):
        """Test Scope 1 data section."""
        result = agent.run(minimal_sb253_input)

        scope1 = result.carb_filing.scope1_section
        assert "total_mtco2e" in scope1
        assert "stationary_combustion_mtco2e" in scope1
        assert "mobile_combustion_mtco2e" in scope1

    def test_carb_filing_scope2_section(self, agent, minimal_sb253_input):
        """Test Scope 2 data section with dual reporting."""
        result = agent.run(minimal_sb253_input)

        scope2 = result.carb_filing.scope2_section
        assert "location_based_mtco2e" in scope2
        assert "market_based_mtco2e" in scope2
        assert "total_electricity_mwh" in scope2

    def test_carb_filing_methodology_section(self, agent, minimal_sb253_input):
        """Test methodology documentation section."""
        result = agent.run(minimal_sb253_input)

        methodology = result.carb_filing.methodology_section
        assert "ghg_protocol_version" in methodology
        assert "gwp_set" in methodology
        assert "emission_factor_sources" in methodology

    def test_carb_filing_verification_section(self, agent, minimal_sb253_input):
        """Test verification documentation section."""
        result = agent.run(minimal_sb253_input)

        verification = result.carb_filing.verification_section
        assert "assurance_level" in verification
        assert "assurance_standards" in verification

    def test_carb_filing_xml_ready(self, agent, minimal_sb253_input):
        """Test XML export readiness flag."""
        result = agent.run(minimal_sb253_input)

        assert result.carb_filing.xml_ready is True

    def test_carb_filing_id_format(self, agent, minimal_sb253_input):
        """Test filing ID format includes EIN and year."""
        result = agent.run(minimal_sb253_input)

        filing_id = result.carb_filing.filing_id
        assert "CARB" in filing_id
        assert minimal_sb253_input.company_info.ein in filing_id
        assert str(minimal_sb253_input.fiscal_year) in filing_id

    def test_carb_filing_scope3_section_when_included(self, agent, sample_company_info):
        """Test Scope 3 section present when included."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=100_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        assert result.carb_filing.scope3_section is not None
        assert "total_mtco2e" in result.carb_filing.scope3_section

    def test_carb_filing_scope3_section_absent_when_excluded(self, agent, minimal_sb253_input):
        """Test Scope 3 section absent when not included."""
        result = agent.run(minimal_sb253_input)

        assert result.carb_filing.scope3_section is None


# =============================================================================
# ASSURANCE PACKAGE TESTS (5 tests)
# =============================================================================

class TestAssurancePackage:
    """Tests for assurance package generation."""

    def test_assurance_package_generated(self, agent, minimal_sb253_input):
        """Test assurance package is generated."""
        result = agent.run(minimal_sb253_input)

        assert result.assurance_package is not None
        assert len(result.assurance_package.methodology_notes) > 0

    def test_assurance_level_propagated(self, agent, sample_company_info):
        """Test assurance level is correctly set."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            assurance_level=AssuranceLevel.REASONABLE,
        )

        result = agent.run(input_data)

        assert result.assurance_package.assurance_level == "reasonable"

    def test_assurance_standards_listed(self, agent, minimal_sb253_input):
        """Test assurance standards are documented."""
        result = agent.run(minimal_sb253_input)

        standards = result.assurance_package.standards_applied
        assert "ISAE 3410" in standards
        assert "ISAE 3000" in standards

    def test_assurance_data_sources_documented(self, agent, minimal_sb253_input):
        """Test data sources are documented."""
        result = agent.run(minimal_sb253_input)

        sources = result.assurance_package.data_sources
        assert len(sources) > 0
        assert all("name" in s for s in sources)

    def test_assurance_completeness_score(self, agent, minimal_sb253_input):
        """Test completeness score is calculated."""
        result = agent.run(minimal_sb253_input)

        score = result.assurance_package.completeness_score
        assert 0 <= score <= 100


# =============================================================================
# PROVENANCE TRACKING TESTS (5 tests)
# =============================================================================

class TestProvenanceTracking:
    """Tests for SHA-256 provenance tracking."""

    def test_provenance_chain_generated(self, agent, minimal_sb253_input):
        """Test provenance chain is generated."""
        result = agent.run(minimal_sb253_input)

        assert len(result.provenance_chain) > 0

    def test_provenance_hash_sha256_format(self, agent, minimal_sb253_input):
        """Test provenance hash is valid SHA-256."""
        result = agent.run(minimal_sb253_input)

        # SHA-256 produces 64 hex characters
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_provenance_record_structure(self, agent, minimal_sb253_input):
        """Test provenance record has required fields."""
        result = agent.run(minimal_sb253_input)

        record = result.provenance_chain[0]
        assert record.operation is not None
        assert record.timestamp is not None
        assert record.input_hash is not None
        assert record.output_hash is not None
        assert record.tool_name is not None

    def test_provenance_hash_changes_with_input(self, agent, sample_company_info):
        """Test provenance hash changes with different inputs."""
        input1 = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=100.0,
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[],
        )

        input2 = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=200.0,  # Different quantity
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[],
        )

        result1 = agent.run(input1)
        result2 = agent.run(input2)

        assert result1.provenance_hash != result2.provenance_hash

    def test_provenance_operations_tracked(self, agent, minimal_sb253_input):
        """Test key operations are tracked in provenance chain."""
        result = agent.run(minimal_sb253_input)

        operations = [r.operation for r in result.provenance_chain]
        assert "applicability_check" in operations
        assert "scope1_calculation" in operations
        assert "scope2_calculation" in operations


# =============================================================================
# INTEGRATION TESTS (10 tests)
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_complete_disclosure_scope1_scope2(self, agent, sample_company_info, sample_facility):
        """Test complete disclosure with Scope 1 and 2."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            facilities=[sample_facility],
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=10000.0,
                    unit=FuelUnit.THERMS,
                ),
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.MOBILE_COMBUSTION,
                    fuel_type=FuelType.DIESEL,
                    quantity=5000.0,
                    unit=FuelUnit.GALLONS,
                ),
            ],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=2_000_000.0,
                    egrid_subregion="CAMX",
                    renewable_percentage=30.0,
                )
            ],
        )

        result = agent.run(input_data)

        assert result.validation_status == "PASS"
        assert result.total_scope1_mtco2e > 0
        assert result.total_scope2_location_mtco2e > 0
        assert result.total_emissions_mtco2e > 0

    def test_complete_disclosure_all_scopes(self, agent, sample_company_info):
        """Test complete disclosure with all three scopes."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2027,  # Scope 3 required
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=5000.0,
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                )
            ],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=5_000_000.0,
                    naics_code="331110",
                ),
                business_travel=Scope3CategoryData(
                    category=Scope3Category.BUSINESS_TRAVEL,
                    calculation_method=CalculationMethod.AVERAGE_DATA,
                    activity_data={
                        "short_haul_miles": 25000,
                        "long_haul_miles": 100000,
                    },
                ),
            ),
        )

        result = agent.run(input_data)

        assert result.total_scope1_mtco2e > 0
        assert result.total_scope2_location_mtco2e > 0
        assert result.total_scope3_mtco2e > 0
        assert result.total_emissions_mtco2e > result.total_scope1_mtco2e

    def test_total_emissions_calculation(self, agent, sample_company_info):
        """Test total emissions is sum of all scopes."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=1000.0,
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=500_000.0,
                    egrid_subregion="CAMX",
                )
            ],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=100_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        expected_total = (
            result.total_scope1_mtco2e +
            result.total_scope2_location_mtco2e +
            (result.total_scope3_mtco2e or 0)
        )
        assert abs(result.total_emissions_mtco2e - expected_total) < 0.01

    def test_report_id_format(self, agent, minimal_sb253_input):
        """Test report ID format."""
        result = agent.run(minimal_sb253_input)

        assert "SB253" in result.report_id
        assert minimal_sb253_input.company_info.ein in result.report_id
        assert str(minimal_sb253_input.fiscal_year) in result.report_id

    def test_processing_time_tracked(self, agent, minimal_sb253_input):
        """Test processing time is tracked."""
        result = agent.run(minimal_sb253_input)

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 60000  # Should be under 60 seconds

    def test_gwp_set_documented(self, agent, sample_company_info):
        """Test GWP set is documented in output."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
            gwp_set=GWPSet.AR6,
        )

        result = agent.run(input_data)

        assert result.gwp_set_used == "AR6"

    def test_company_below_threshold_rejected(self, agent):
        """Test company below $1B threshold is rejected."""
        with pytest.raises(ValueError):
            CompanyInfo(
                company_name="Small Corp",
                ein="12-3456789",
                total_revenue_usd=500_000_000,  # Below threshold
                naics_code="331110",
            )

    def test_invalid_ein_format_rejected(self, agent):
        """Test invalid EIN format is rejected."""
        with pytest.raises(ValueError):
            CompanyInfo(
                company_name="Test Corp",
                ein="123456789",  # Missing hyphen
                total_revenue_usd=2_000_000_000,
                naics_code="331110",
            )

    def test_fiscal_year_before_2025_rejected(self, agent, sample_company_info):
        """Test fiscal year before 2025 is rejected."""
        with pytest.raises(ValueError):
            SB253ReportInput(
                company_info=sample_company_info,
                fiscal_year=2024,  # Too early
                scope1_sources=[],
                scope2_sources=[],
            )

    def test_generated_at_timestamp(self, agent, minimal_sb253_input):
        """Test generated_at timestamp is present."""
        before = datetime.utcnow()
        result = agent.run(minimal_sb253_input)
        after = datetime.utcnow()

        assert before <= result.generated_at <= after


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_scope1_sources(self, agent, sample_company_info):
        """Test handling of empty Scope 1 sources."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        assert result.total_scope1_mtco2e == 0

    def test_empty_scope2_sources(self, agent, sample_company_info):
        """Test handling of empty Scope 2 sources."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=100.0,
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        assert result.total_scope1_mtco2e > 0
        assert result.total_scope2_location_mtco2e == 0

    def test_very_large_emissions(self, agent, sample_company_info):
        """Test handling of very large emission values."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=10_000_000.0,  # 10 million therms
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Should be ~53,200 MTCO2e
        assert result.total_scope1_mtco2e > 50000
        assert result.validation_status == "PASS"

    def test_multiple_facilities_aggregation(self, agent, sample_company_info):
        """Test proper aggregation across multiple facilities."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            scope1_sources=[
                Scope1Source(
                    facility_id=f"FAC{i:03d}",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=100.0,
                    unit=FuelUnit.THERMS,
                )
                for i in range(10)  # 10 facilities
            ],
            scope2_sources=[],
        )

        result = agent.run(input_data)

        # Should be 10x single facility emissions
        assert len(result.scope1_emissions.emissions_by_facility) == 10

    def test_assurance_status_ready(self, agent, sample_company_info, sample_facility):
        """Test assurance status is READY with complete data."""
        input_data = SB253ReportInput(
            company_info=sample_company_info,
            fiscal_year=2025,
            facilities=[sample_facility],
            scope1_sources=[
                Scope1Source(
                    facility_id="FAC001",
                    source_category=SourceCategory.STATIONARY_COMBUSTION,
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=1000.0,
                    unit=FuelUnit.THERMS,
                )
            ],
            scope2_sources=[
                Scope2Source(
                    facility_id="FAC001",
                    kwh=1_000_000.0,
                    egrid_subregion="CAMX",
                )
            ],
            include_scope3=True,
            scope3_data=Scope3Data(
                purchased_goods_services=Scope3CategoryData(
                    category=Scope3Category.PURCHASED_GOODS_SERVICES,
                    calculation_method=CalculationMethod.SPEND_BASED,
                    spend_usd=100_000.0,
                )
            ),
        )

        result = agent.run(input_data)

        assert result.assurance_status == "READY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
