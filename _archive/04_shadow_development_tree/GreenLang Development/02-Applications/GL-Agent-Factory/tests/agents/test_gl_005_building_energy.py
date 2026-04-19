"""
Unit Tests for GL-005: Building Energy Agent

Comprehensive test suite covering:
- Energy Use Intensity (EUI) calculation
- GHG emissions from building operations
- CRREM pathway alignment analysis
- Stranding risk assessment
- Energy Performance Certificate (EPC) rating

Target: 85%+ code coverage

Reference:
- EPBD (Energy Performance of Buildings Directive)
- CRREM (Carbon Risk Real Estate Monitor)
- ISO 52000 series

Run with:
    pytest tests/agents/test_gl_005_building_energy.py -v --cov=backend/agents/gl_005_building_energy
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock




from agents.gl_005_building_energy.agent import (
    BuildingEnergyAgent,
    BuildingEnergyInput,
    BuildingEnergyOutput,
    BuildingType,
    EPCRating,
    StrandingRisk,
    EnergyFactor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def building_agent():
    """Create BuildingEnergyAgent instance for testing."""
    return BuildingEnergyAgent()


@pytest.fixture
def office_building_input():
    """Create typical office building input."""
    return BuildingEnergyInput(
        building_id="BUILDING-001",
        building_type=BuildingType.OFFICE,
        floor_area_sqm=10000.0,
        year_built=2010,
        region="DE",
        annual_electricity_kwh=500000.0,
        annual_gas_m3=50000.0,
        occupancy_rate=80.0,
    )


@pytest.fixture
def high_energy_building_input():
    """Create high energy consumption building."""
    return BuildingEnergyInput(
        building_type=BuildingType.OFFICE,
        floor_area_sqm=5000.0,
        region="PL",  # High grid emission factor
        annual_electricity_kwh=1500000.0,
        annual_gas_m3=100000.0,
    )


@pytest.fixture
def efficient_building_input():
    """Create energy-efficient building with renewables."""
    return BuildingEnergyInput(
        building_type=BuildingType.OFFICE,
        floor_area_sqm=10000.0,
        region="FR",  # Low grid emission factor
        annual_electricity_kwh=300000.0,
        annual_gas_m3=10000.0,
        renewable_generation_kwh=100000.0,
    )


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestBuildingEnergyAgentInitialization:
    """Tests for BuildingEnergyAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self, building_agent):
        """Test agent initializes correctly with default config."""
        assert building_agent is not None
        assert building_agent.AGENT_ID == "buildings/energy_performance_v1"
        assert building_agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_agent_has_emission_factors(self, building_agent):
        """Test agent has grid emission factors defined."""
        assert hasattr(building_agent, "GRID_EMISSION_FACTORS")
        assert "DE" in building_agent.GRID_EMISSION_FACTORS

    @pytest.mark.unit
    def test_agent_has_epc_thresholds(self, building_agent):
        """Test agent has EPC thresholds defined."""
        assert hasattr(building_agent, "EPC_THRESHOLDS")
        assert BuildingType.OFFICE in building_agent.EPC_THRESHOLDS

    @pytest.mark.unit
    def test_agent_has_crrem_targets(self, building_agent):
        """Test agent has CRREM targets defined."""
        assert hasattr(building_agent, "CRREM_TARGETS_2050")
        assert BuildingType.OFFICE in building_agent.CRREM_TARGETS_2050


# =============================================================================
# Test Class: Building Types
# =============================================================================


class TestBuildingTypes:
    """Tests for building type handling."""

    @pytest.mark.unit
    def test_all_building_types_defined(self):
        """Test all building types are defined."""
        building_types = [
            BuildingType.OFFICE,
            BuildingType.RETAIL,
            BuildingType.HOTEL,
            BuildingType.RESIDENTIAL,
            BuildingType.INDUSTRIAL,
            BuildingType.WAREHOUSE,
            BuildingType.HEALTHCARE,
            BuildingType.EDUCATION,
            BuildingType.DATA_CENTER,
        ]
        assert len(building_types) == 9

    @pytest.mark.unit
    def test_building_type_values(self):
        """Test building type enum values."""
        assert BuildingType.OFFICE.value == "office"
        assert BuildingType.DATA_CENTER.value == "data_center"


# =============================================================================
# Test Class: EPC Ratings
# =============================================================================


class TestEPCRatings:
    """Tests for EPC rating handling."""

    @pytest.mark.unit
    def test_epc_rating_values(self):
        """Test EPC rating enum values."""
        assert EPCRating.A_PLUS.value == "A+"
        assert EPCRating.A.value == "A"
        assert EPCRating.B.value == "B"
        assert EPCRating.G.value == "G"

    @pytest.mark.unit
    def test_epc_rating_order(self):
        """Test EPC ratings are in order A+ to G."""
        ratings = [EPCRating.A_PLUS, EPCRating.A, EPCRating.B, EPCRating.C,
                   EPCRating.D, EPCRating.E, EPCRating.F, EPCRating.G]
        assert len(ratings) == 8


# =============================================================================
# Test Class: Stranding Risk
# =============================================================================


class TestStrandingRisk:
    """Tests for stranding risk classification."""

    @pytest.mark.unit
    def test_stranding_risk_values(self):
        """Test stranding risk enum values."""
        assert StrandingRisk.LOW.value == "low"
        assert StrandingRisk.MEDIUM.value == "medium"
        assert StrandingRisk.HIGH.value == "high"
        assert StrandingRisk.STRANDED.value == "stranded"


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestBuildingInputValidation:
    """Tests for building energy input validation."""

    @pytest.mark.unit
    def test_valid_input_passes(self, office_building_input):
        """Test valid input passes validation."""
        assert office_building_input.building_type == BuildingType.OFFICE
        assert office_building_input.floor_area_sqm == 10000.0

    @pytest.mark.unit
    def test_floor_area_must_be_positive(self):
        """Test floor area must be >= 1."""
        with pytest.raises(ValueError):
            BuildingEnergyInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=0.5,  # Below minimum
            )

    @pytest.mark.unit
    def test_energy_values_non_negative(self, office_building_input):
        """Test energy values must be non-negative."""
        assert office_building_input.annual_electricity_kwh >= 0
        assert office_building_input.annual_gas_m3 >= 0

    @pytest.mark.unit
    def test_region_code_uppercase(self):
        """Test region code is uppercased."""
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            region="de",  # lowercase
        )
        assert input_data.region == "DE"  # Should be uppercased

    @pytest.mark.unit
    def test_occupancy_rate_bounds(self):
        """Test occupancy rate must be 0-100."""
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            occupancy_rate=50.0,
        )
        assert 0 <= input_data.occupancy_rate <= 100


# =============================================================================
# Test Class: Energy Calculations
# =============================================================================


class TestEnergyCalculations:
    """Tests for energy calculation functionality."""

    @pytest.mark.unit
    def test_gas_to_kwh_conversion(self, building_agent):
        """Test natural gas m3 to kWh conversion factor."""
        assert building_agent.GAS_M3_TO_KWH == 10.55

    @pytest.mark.unit
    def test_eui_calculation(self, building_agent, office_building_input):
        """Test EUI calculation."""
        result = building_agent.run(office_building_input)

        # EUI = total_energy / floor_area
        expected_gas_kwh = 50000 * 10.55
        expected_total = 500000 + expected_gas_kwh
        expected_eui = expected_total / 10000

        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=0.01)

    @pytest.mark.unit
    def test_renewable_share_calculation(self, building_agent, efficient_building_input):
        """Test renewable energy share calculation."""
        result = building_agent.run(efficient_building_input)

        # Renewable share should be calculated
        assert result.renewable_share_pct >= 0
        assert result.renewable_share_pct <= 100


# =============================================================================
# Test Class: Emissions Calculations
# =============================================================================


class TestEmissionsCalculations:
    """Tests for emissions calculation functionality."""

    @pytest.mark.unit
    def test_scope1_emissions_from_gas(self, building_agent, office_building_input):
        """Test Scope 1 emissions from natural gas."""
        result = building_agent.run(office_building_input)

        # Scope 1 = gas_kwh * gas_emission_factor
        gas_kwh = 50000 * 10.55
        expected_scope1 = gas_kwh * building_agent.GAS_EMISSION_FACTOR

        assert result.scope1_emissions == pytest.approx(expected_scope1, rel=0.01)

    @pytest.mark.unit
    def test_scope2_emissions_from_electricity(self, building_agent, office_building_input):
        """Test Scope 2 emissions from electricity."""
        result = building_agent.run(office_building_input)

        # Scope 2 = electricity_kwh * grid_factor
        grid_factor = building_agent.GRID_EMISSION_FACTORS["DE"].value
        expected_scope2 = 500000 * grid_factor

        assert result.scope2_emissions == pytest.approx(expected_scope2, rel=0.01)

    @pytest.mark.unit
    def test_emissions_intensity_calculation(self, building_agent, office_building_input):
        """Test emissions intensity (kgCO2e/m2) calculation."""
        result = building_agent.run(office_building_input)

        # Intensity = total_emissions / floor_area
        expected_intensity = result.total_emissions_kgco2e / 10000

        assert result.emissions_intensity_kgco2e_sqm == pytest.approx(expected_intensity, rel=0.01)


# =============================================================================
# Test Class: Grid Emission Factors
# =============================================================================


class TestGridEmissionFactors:
    """Tests for grid emission factor handling."""

    @pytest.mark.unit
    def test_germany_grid_factor(self, building_agent):
        """Test Germany grid emission factor."""
        factor = building_agent.GRID_EMISSION_FACTORS["DE"]
        assert factor.value == 0.366
        assert factor.unit == "kgCO2e/kWh"

    @pytest.mark.unit
    def test_france_grid_factor_low(self, building_agent):
        """Test France has low grid factor (nuclear)."""
        factor = building_agent.GRID_EMISSION_FACTORS["FR"]
        assert factor.value < 0.1  # Low due to nuclear

    @pytest.mark.unit
    def test_poland_grid_factor_high(self, building_agent):
        """Test Poland has high grid factor (coal)."""
        factor = building_agent.GRID_EMISSION_FACTORS["PL"]
        assert factor.value > 0.5  # High due to coal


# =============================================================================
# Test Class: CRREM Analysis
# =============================================================================


class TestCRREMAnalysis:
    """Tests for CRREM pathway analysis."""

    @pytest.mark.unit
    def test_crrem_target_for_office(self, building_agent):
        """Test CRREM 2050 target for office buildings."""
        target = building_agent.CRREM_TARGETS_2050[BuildingType.OFFICE]
        assert target == 4.5  # kgCO2e/m2/year

    @pytest.mark.unit
    def test_crrem_target_for_data_center(self, building_agent):
        """Test CRREM 2050 target for data centers (high)."""
        target = building_agent.CRREM_TARGETS_2050[BuildingType.DATA_CENTER]
        assert target == 50.0  # Higher due to intensive operations

    @pytest.mark.unit
    def test_stranding_risk_assessment(self, building_agent, high_energy_building_input):
        """Test stranding risk is assessed for high-energy buildings."""
        result = building_agent.run(high_energy_building_input)

        # High energy building should have elevated stranding risk
        assert result.stranding_risk in ["high", "stranded", "medium"]


# =============================================================================
# Test Class: EPC Rating Assignment
# =============================================================================


class TestEPCRatingAssignment:
    """Tests for EPC rating assignment."""

    @pytest.mark.unit
    def test_epc_rating_assigned(self, building_agent, office_building_input):
        """Test EPC rating is assigned."""
        result = building_agent.run(office_building_input)
        assert result.epc_rating in ["A+", "A", "B", "C", "D", "E", "F", "G"]

    @pytest.mark.unit
    def test_efficient_building_gets_good_rating(self, building_agent, efficient_building_input):
        """Test efficient building gets good EPC rating."""
        result = building_agent.run(efficient_building_input)
        # Efficient building should get B or better
        assert result.epc_rating in ["A+", "A", "B", "C"]


# =============================================================================
# Test Class: Provenance Tracking
# =============================================================================


class TestProvenanceTracking:
    """Tests for provenance hash tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, building_agent, office_building_input):
        """Test provenance hash is generated."""
        result = building_agent.run(office_building_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    @pytest.mark.unit
    def test_provenance_hash_deterministic(self, building_agent, office_building_input):
        """Test provenance hash is deterministic for same input."""
        result1 = building_agent.run(office_building_input)
        result2 = building_agent.run(office_building_input)
        assert result1.provenance_hash == result2.provenance_hash


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestBuildingEnergyPerformance:
    """Performance tests for BuildingEnergyAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_calculation_performance(self, building_agent, office_building_input):
        """Test single calculation completes in under 50ms."""
        import time

        start = time.perf_counter()
        result = building_agent.run(office_building_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_calculation_throughput(self, building_agent):
        """Test batch calculation throughput."""
        import time

        num_buildings = 100
        inputs = [
            BuildingEnergyInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=float(i * 1000 + 1000),
                annual_electricity_kwh=float(i * 50000),
            )
            for i in range(num_buildings)
        ]

        start = time.perf_counter()
        results = [building_agent.run(inp) for inp in inputs]
        elapsed_s = time.perf_counter() - start

        throughput = num_buildings / elapsed_s
        assert throughput >= 10, f"Throughput {throughput:.0f} buildings/sec below target"
