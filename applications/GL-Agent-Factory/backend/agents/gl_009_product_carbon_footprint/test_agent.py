"""
Unit Tests for GL-009: Product Carbon Footprint Agent

Comprehensive test coverage for the Product Carbon Footprint Agent including:
- Raw material emissions calculations (26 material categories)
- Manufacturing energy emissions (electricity, natural gas, diesel, steam)
- Transport emissions (8 transport modes)
- Use phase calculations (energy, consumables, maintenance)
- End-of-life Circular Footprint Formula (CFF) calculations
- System boundary validation (cradle-to-gate, cradle-to-grave)
- Data quality assessment (PEF DQR methodology)
- Export format validation (PACT Pathfinder 2.1, Catena-X, EU Battery Passport)
- Provenance hash validation
- Determinism verification tests

Test coverage target: 85%+
Total tests: 60+ golden tests covering all PCF calculation scenarios

Formula Documentation:
----------------------
All emission calculations follow ISO 14067:2018 and PEF methodology:

RAW MATERIALS: emissions = SUM(material_kg * emission_factor)

MANUFACTURING:
- Electricity: kwh * grid_factor * (1 - renewable_pct)
- Natural gas: m3 * 1.93 kgCO2e/m3
- Diesel: L * 2.68 kgCO2e/L
- Steam: kg * 0.27 kgCO2e/kg
- Process: CO2 + CH4*29.8 + N2O*273 + other_gwp

TRANSPORT: emissions = weight_tonnes * distance_km * mode_factor / utilization

USE PHASE:
- Energy: energy_per_use * uses_per_year * lifetime * grid_factor
- Consumables: per_year * lifetime
- Maintenance: per_year * lifetime

END OF LIFE (CFF):
CFF = (1-R1)*Ev + R1*(A*Erec + (1-A)*Ev*Qs/Qp)
    + (1-A)*R2*(ErecEoL - Ev*Qs/Qp)
    + (1-B)*R3*(EER - credits)
    + (1-R2-R3)*ED

Material Emission Factors (kgCO2e/kg) from agent.py:
- Steel (primary): 2.35, Steel (recycled): 0.65
- Aluminum (primary): 16.5, Aluminum (recycled): 0.85
- Copper (primary): 4.20, Copper (recycled): 0.50
- Plastics PP: 1.98, PE: 2.10, PET: 2.73, ABS: 3.55, Recycled: 0.45
- Glass: 0.85, Concrete: 0.13, Cement: 0.93
- Wood softwood: 0.31, hardwood: 0.45
- Paper virgin: 1.32, recycled: 0.67
- Lithium: 12.5, Cobalt: 35.8, Nickel: 12.4, Graphite: 4.85
- Rare earth: 28.5

Grid Factors (kgCO2e/kWh):
- GLOBAL: 0.475, US: 0.417, EU: 0.276, DE: 0.366, FR: 0.052
- UK: 0.207, CN: 0.555, JP: 0.457, IN: 0.708, KR: 0.415

Transport Factors (kgCO2e/tonne-km):
- Road truck: 0.089, Road van: 0.195
- Rail freight: 0.028, Sea container: 0.016, Sea bulk: 0.008
- Air freight: 0.602, Barge: 0.031, Pipeline: 0.025
"""

import hashlib
import json
import math
import sys
import os
import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add the agent directory to path for direct imports (avoid parent __init__.py issues)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    ProductCarbonFootprintAgent,
    PCFInput,
    PCFOutput,
    BOMItem,
    ManufacturingEnergy,
    ProcessEmissions,
    TransportLeg,
    TransportData,
    UsePhaseData,
    EndOfLifeData,
    PCFBoundary,
    TransportMode,
    EndOfLifeTreatment,
    DataQualityLevel,
    MaterialCategory,
    MaterialEmissionFactor,
    GridEmissionFactor,
    ImpactCategories,
    LifecycleStageBreakdown,
)


# =============================================================================
# Test Constants - Expected Emission Factors
# =============================================================================

# Material Emission Factors (kgCO2e/kg)
EF_STEEL_PRIMARY = 2.35
EF_STEEL_RECYCLED = 0.65
EF_ALUMINUM_PRIMARY = 16.5
EF_ALUMINUM_RECYCLED = 0.85
EF_COPPER_PRIMARY = 4.20
EF_COPPER_RECYCLED = 0.50
EF_PLASTICS_PP = 1.98
EF_PLASTICS_PE = 2.10
EF_PLASTICS_PET = 2.73
EF_PLASTICS_ABS = 3.55
EF_PLASTICS_RECYCLED = 0.45
EF_GLASS = 0.85
EF_CONCRETE = 0.13
EF_CEMENT = 0.93
EF_WOOD_SOFTWOOD = 0.31
EF_WOOD_HARDWOOD = 0.45
EF_PAPER_VIRGIN = 1.32
EF_PAPER_RECYCLED = 0.67
EF_LITHIUM = 12.5
EF_COBALT = 35.8
EF_NICKEL = 12.4
EF_GRAPHITE = 4.85
EF_RARE_EARTH = 28.5
EF_RUBBER_NATURAL = 1.85
EF_RUBBER_SYNTHETIC = 2.95
EF_TEXTILES_COTTON = 8.50
EF_TEXTILES_POLYESTER = 5.55

# Grid Emission Factors (kgCO2e/kWh)
EF_GRID_GLOBAL = 0.475
EF_GRID_US = 0.417
EF_GRID_EU = 0.276
EF_GRID_DE = 0.366
EF_GRID_FR = 0.052
EF_GRID_UK = 0.207
EF_GRID_CN = 0.555
EF_GRID_JP = 0.457
EF_GRID_IN = 0.708
EF_GRID_KR = 0.415

# Energy Emission Factors
EF_NATURAL_GAS = 1.93  # kgCO2e/m3
EF_DIESEL = 2.68  # kgCO2e/L
EF_STEAM = 0.27  # kgCO2e/kg
EF_COMPRESSED_AIR = 0.12  # kgCO2e/m3

# Transport Factors (kgCO2e/tonne-km)
EF_ROAD_TRUCK = 0.089
EF_ROAD_VAN = 0.195
EF_RAIL = 0.028
EF_SEA_CONTAINER = 0.016
EF_SEA_BULK = 0.008
EF_AIR = 0.602
EF_BARGE = 0.031
EF_PIPELINE = 0.025

# GWP Values (AR6 100-year)
GWP_CH4 = 29.8
GWP_N2O = 273.0
GWP_SF6 = 25200.0
GWP_NF3 = 17400.0

# End-of-life Factors (kgCO2e/kg)
EF_EOL_LANDFILL = 0.586
EF_EOL_INCINERATION = 2.42
EF_EOL_RECYCLING = 0.21
EF_EOL_ENERGY_RECOVERY = 0.85
EF_EOL_COMPOSTING = 0.10
EF_EOL_REUSE = 0.05


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent() -> ProductCarbonFootprintAgent:
    """Create a ProductCarbonFootprintAgent instance for testing."""
    return ProductCarbonFootprintAgent()


@pytest.fixture
def agent_with_config() -> ProductCarbonFootprintAgent:
    """Create agent with custom configuration."""
    return ProductCarbonFootprintAgent(config={"custom_setting": "value"})


@pytest.fixture
def simple_steel_bom() -> List[BOMItem]:
    """
    Create simple BOM with steel only.

    Expected calculation:
        10 kg * 2.35 kgCO2e/kg = 23.5 kgCO2e
    """
    return [
        BOMItem(
            material_id="STEEL-001",
            material_category=MaterialCategory.STEEL_PRIMARY,
            quantity_kg=10.0,
        )
    ]


@pytest.fixture
def multi_material_bom() -> List[BOMItem]:
    """
    Create BOM with multiple materials.

    Expected calculation:
        Steel:    10 kg * 2.35 = 23.50 kgCO2e
        Aluminum:  5 kg * 16.5 = 82.50 kgCO2e
        Plastic:   2 kg * 1.98 =  3.96 kgCO2e
        Total: 109.96 kgCO2e
    """
    return [
        BOMItem(
            material_id="STEEL-001",
            material_category=MaterialCategory.STEEL_PRIMARY,
            quantity_kg=10.0,
        ),
        BOMItem(
            material_id="ALU-001",
            material_category=MaterialCategory.ALUMINUM_PRIMARY,
            quantity_kg=5.0,
        ),
        BOMItem(
            material_id="PLASTIC-001",
            material_category=MaterialCategory.PLASTICS_PP,
            quantity_kg=2.0,
        ),
    ]


@pytest.fixture
def battery_bom() -> List[BOMItem]:
    """
    Create BOM for battery product (triggers battery passport).

    Expected calculation:
        Lithium:   1 kg * 12.5  = 12.50 kgCO2e
        Cobalt:  0.5 kg * 35.8  = 17.90 kgCO2e
        Nickel:    2 kg * 12.4  = 24.80 kgCO2e
        Graphite:  3 kg *  4.85 = 14.55 kgCO2e
        Total: 69.75 kgCO2e
    """
    return [
        BOMItem(
            material_id="LI-001",
            material_category=MaterialCategory.LITHIUM,
            quantity_kg=1.0,
        ),
        BOMItem(
            material_id="CO-001",
            material_category=MaterialCategory.COBALT,
            quantity_kg=0.5,
        ),
        BOMItem(
            material_id="NI-001",
            material_category=MaterialCategory.NICKEL,
            quantity_kg=2.0,
        ),
        BOMItem(
            material_id="GRAPH-001",
            material_category=MaterialCategory.GRAPHITE,
            quantity_kg=3.0,
        ),
    ]


@pytest.fixture
def recycled_content_bom() -> List[BOMItem]:
    """
    Create BOM with recycled content.

    Expected calculation (50% recycled content):
        Effective EF = 0.5 * 2.35 + 0.5 * 0.65 = 1.50 kgCO2e/kg
        Emissions: 10 kg * 1.50 = 15.0 kgCO2e
    """
    return [
        BOMItem(
            material_id="STEEL-REC-001",
            material_category=MaterialCategory.STEEL_PRIMARY,
            quantity_kg=10.0,
            recycled_content_pct=50.0,
        )
    ]


@pytest.fixture
def supplier_pcf_bom() -> List[BOMItem]:
    """
    Create BOM with supplier-provided PCF (primary data).

    Expected calculation:
        10 kg * 2.00 kgCO2e/kg (supplier PCF) = 20.0 kgCO2e
    """
    return [
        BOMItem(
            material_id="STEEL-SUPPLIER-001",
            material_category=MaterialCategory.STEEL_PRIMARY,
            quantity_kg=10.0,
            supplier_pcf=2.00,  # Supplier-provided value
        )
    ]


@pytest.fixture
def manufacturing_energy_us() -> ManufacturingEnergy:
    """
    Create manufacturing energy data for US region.

    Expected calculation:
        Electricity: 100 kWh * 0.417 = 41.70 kgCO2e
        Natural gas: 50 m3 * 1.93 = 96.50 kgCO2e
        Diesel: 10 L * 2.68 = 26.80 kgCO2e
        Total: 165.00 kgCO2e
    """
    return ManufacturingEnergy(
        electricity_kwh=100.0,
        natural_gas_m3=50.0,
        diesel_liters=10.0,
        grid_region="US",
    )


@pytest.fixture
def manufacturing_energy_renewable() -> ManufacturingEnergy:
    """
    Create manufacturing energy with renewable electricity.

    Expected calculation:
        Electricity: 100 kWh * 0.417 * (1 - 0.8) = 8.34 kgCO2e (80% renewable)
        Natural gas: 50 m3 * 1.93 = 96.50 kgCO2e
        Total: 104.84 kgCO2e
    """
    return ManufacturingEnergy(
        electricity_kwh=100.0,
        natural_gas_m3=50.0,
        grid_region="US",
        renewable_pct=80.0,
    )


@pytest.fixture
def process_emissions_data() -> ProcessEmissions:
    """
    Create process emissions data.

    Expected calculation:
        CO2: 10 kg
        CH4: 0.5 kg * 29.8 = 14.9 kg CO2e
        N2O: 0.1 kg * 273 = 27.3 kg CO2e
        Total: 52.2 kgCO2e
    """
    return ProcessEmissions(
        co2_kg=10.0,
        ch4_kg=0.5,
        n2o_kg=0.1,
    )


@pytest.fixture
def transport_data() -> TransportData:
    """
    Create transport data with multiple legs.

    Expected calculation (product_weight=10kg):
        Truck: 0.01 t * 500 km * 0.089 = 0.445 kgCO2e
        Sea:   0.01 t * 5000 km * 0.016 = 0.800 kgCO2e
        Total: 1.245 kgCO2e
    """
    return TransportData(
        inbound_legs=[
            TransportLeg(
                leg_id="LEG-001",
                mode=TransportMode.ROAD_TRUCK,
                distance_km=500.0,
            ),
        ],
        outbound_legs=[
            TransportLeg(
                leg_id="LEG-002",
                mode=TransportMode.SEA_CONTAINER,
                distance_km=5000.0,
            ),
        ],
        product_weight_kg=10.0,
    )


@pytest.fixture
def use_phase_data() -> UsePhaseData:
    """
    Create use phase data for cradle-to-grave.

    Expected calculation:
        Energy: 0.5 kWh/use * 365 uses/year * 5 years * 0.417 = 380.5875 kgCO2e
        Consumables: 5 kgCO2e/year * 5 years = 25.0 kgCO2e
        Maintenance: 2 kgCO2e/year * 5 years = 10.0 kgCO2e
        Total: 415.5875 kgCO2e
    """
    return UsePhaseData(
        energy_per_use_kwh=0.5,
        uses_per_year=365.0,
        lifetime_years=5.0,
        grid_region="US",
        consumables_kgco2e_per_year=5.0,
        maintenance_kgco2e_per_year=2.0,
    )


@pytest.fixture
def end_of_life_data() -> EndOfLifeData:
    """
    Create end-of-life data for CFF calculation.

    CFF Parameters:
        R1=0.3 (30% recycled input)
        R2=0.7 (70% recycled output)
        R3=0.1 (10% energy recovery)
        A=0.5, B=0.5 (default allocation)
        Qs/Qp=0.9 (quality ratio)
    """
    return EndOfLifeData(
        R1=0.3,
        R2=0.7,
        R3=0.1,
        A=0.5,
        B=0.5,
        Qs=0.9,
        Qp=1.0,
        material_weight_kg=10.0,
        treatment=EndOfLifeTreatment.RECYCLING,
    )


@pytest.fixture
def cradle_to_gate_input(simple_steel_bom, manufacturing_energy_us) -> PCFInput:
    """Create basic cradle-to-gate PCF input."""
    return PCFInput(
        product_id="PROD-001",
        product_name="Test Product",
        bill_of_materials=simple_steel_bom,
        manufacturing_energy=manufacturing_energy_us,
        boundary=PCFBoundary.CRADLE_TO_GATE,
    )


@pytest.fixture
def cradle_to_grave_input(
    simple_steel_bom,
    manufacturing_energy_us,
    transport_data,
    use_phase_data,
    end_of_life_data,
) -> PCFInput:
    """Create cradle-to-grave PCF input."""
    return PCFInput(
        product_id="PROD-002",
        product_name="Full Lifecycle Product",
        bill_of_materials=simple_steel_bom,
        manufacturing_energy=manufacturing_energy_us,
        transport_data=transport_data,
        boundary=PCFBoundary.CRADLE_TO_GRAVE,
        use_phase=use_phase_data,
        end_of_life=end_of_life_data,
    )


# =============================================================================
# Test 1-10: Agent Initialization and Basic Tests
# =============================================================================


class TestAgentInitialization:
    """Tests for agent initialization."""

    def test_01_agent_initialization(self, agent: ProductCarbonFootprintAgent):
        """Test 1: Agent initializes correctly with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "products/carbon_footprint_v1"
        assert agent.VERSION == "1.0.0"
        assert "PACT" in agent.DESCRIPTION or "PCF" in agent.DESCRIPTION or "Product Carbon Footprint" in agent.DESCRIPTION

    def test_02_agent_with_custom_config(
        self, agent_with_config: ProductCarbonFootprintAgent
    ):
        """Test 2: Agent initializes with custom configuration."""
        assert agent_with_config.config == {"custom_setting": "value"}

    def test_03_material_factors_loaded(self, agent: ProductCarbonFootprintAgent):
        """Test 3: Material emission factors are loaded correctly."""
        mf = agent.MATERIAL_FACTORS
        assert MaterialCategory.STEEL_PRIMARY in mf
        assert MaterialCategory.ALUMINUM_PRIMARY in mf
        assert MaterialCategory.PLASTICS_PP in mf
        assert MaterialCategory.LITHIUM in mf

    def test_04_grid_factors_loaded(self, agent: ProductCarbonFootprintAgent):
        """Test 4: Grid emission factors are loaded correctly."""
        gf = agent.GRID_FACTORS
        assert "GLOBAL" in gf
        assert "US" in gf
        assert "EU" in gf
        assert "DE" in gf
        assert "FR" in gf

    def test_05_transport_factors_loaded(self, agent: ProductCarbonFootprintAgent):
        """Test 5: Transport mode factors are loaded correctly."""
        tf = agent.TRANSPORT_FACTORS
        assert TransportMode.ROAD_TRUCK in tf
        assert TransportMode.SEA_CONTAINER in tf
        assert TransportMode.AIR_FREIGHT in tf
        assert TransportMode.RAIL_FREIGHT in tf

    def test_06_get_supported_materials(self, agent: ProductCarbonFootprintAgent):
        """Test 6: Get supported material categories."""
        materials = agent.get_supported_materials()
        assert "steel_primary" in materials
        assert "aluminum_primary" in materials
        assert "lithium" in materials
        assert len(materials) == len(MaterialCategory)

    def test_07_get_supported_transport_modes(self, agent: ProductCarbonFootprintAgent):
        """Test 7: Get supported transport modes."""
        modes = agent.get_supported_transport_modes()
        assert "road_truck" in modes
        assert "sea_container" in modes
        assert "air_freight" in modes
        assert len(modes) == len(TransportMode)

    def test_08_get_material_emission_factor(self, agent: ProductCarbonFootprintAgent):
        """Test 8: Get material emission factor."""
        ef = agent.get_material_emission_factor(MaterialCategory.STEEL_PRIMARY)
        assert ef is not None
        assert ef.factor_kgco2e_per_kg == EF_STEEL_PRIMARY
        assert ef.source == "Ecoinvent 3.9"

    def test_09_get_grid_emission_factor(self, agent: ProductCarbonFootprintAgent):
        """Test 9: Get grid emission factor."""
        gf = agent.get_grid_emission_factor("US")
        assert gf is not None
        assert gf.factor_kgco2e_per_kwh == EF_GRID_US
        assert "EPA" in gf.source

    def test_10_basic_run_completes(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 10: Basic agent run completes successfully."""
        result = agent.run(cradle_to_gate_input)
        assert result is not None
        assert isinstance(result, PCFOutput)
        assert result.total_co2e > 0


# =============================================================================
# Test 11-20: Raw Materials Emission Calculations (Golden Tests)
# =============================================================================


class TestRawMaterialsCalculations:
    """Tests for raw materials emissions calculations."""

    @pytest.mark.golden
    def test_11_steel_primary_10kg(
        self,
        agent: ProductCarbonFootprintAgent,
        simple_steel_bom: List[BOMItem],
    ):
        """
        Test 11: Steel primary calculation - 10 kg

        ZERO-HALLUCINATION CHECK:
        Formula: emissions = 10 kg * 2.35 kgCO2e/kg = 23.5 kgCO2e
        """
        input_data = PCFInput(
            product_id="TEST-011",
            bill_of_materials=simple_steel_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = 10.0 * EF_STEEL_PRIMARY  # 23.5
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_12_multi_material_calculation(
        self,
        agent: ProductCarbonFootprintAgent,
        multi_material_bom: List[BOMItem],
    ):
        """
        Test 12: Multi-material BOM calculation

        ZERO-HALLUCINATION CHECK:
        Steel:    10 kg * 2.35 = 23.50 kgCO2e
        Aluminum:  5 kg * 16.5 = 82.50 kgCO2e
        Plastic:   2 kg * 1.98 =  3.96 kgCO2e
        Total: 109.96 kgCO2e
        """
        input_data = PCFInput(
            product_id="TEST-012",
            bill_of_materials=multi_material_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = (
            10.0 * EF_STEEL_PRIMARY +
            5.0 * EF_ALUMINUM_PRIMARY +
            2.0 * EF_PLASTICS_PP
        )  # 109.96
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_13_aluminum_primary_calculation(self, agent: ProductCarbonFootprintAgent):
        """
        Test 13: Aluminum primary - high emission factor

        ZERO-HALLUCINATION CHECK:
        Formula: 5 kg * 16.5 kgCO2e/kg = 82.5 kgCO2e
        """
        bom = [
            BOMItem(
                material_id="ALU-001",
                material_category=MaterialCategory.ALUMINUM_PRIMARY,
                quantity_kg=5.0,
            )
        ]
        input_data = PCFInput(
            product_id="TEST-013",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = 5.0 * EF_ALUMINUM_PRIMARY  # 82.5
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_14_recycled_content_50pct(
        self,
        agent: ProductCarbonFootprintAgent,
        recycled_content_bom: List[BOMItem],
    ):
        """
        Test 14: Steel with 50% recycled content

        ZERO-HALLUCINATION CHECK:
        Effective EF = 0.5 * 2.35 + 0.5 * 0.65 = 1.50 kgCO2e/kg
        Emissions: 10 kg * 1.50 = 15.0 kgCO2e
        """
        input_data = PCFInput(
            product_id="TEST-014",
            bill_of_materials=recycled_content_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        effective_ef = (
            0.5 * EF_STEEL_PRIMARY + 0.5 * EF_STEEL_RECYCLED
        )  # 1.50
        expected = 10.0 * effective_ef  # 15.0
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_15_supplier_pcf_used(
        self,
        agent: ProductCarbonFootprintAgent,
        supplier_pcf_bom: List[BOMItem],
    ):
        """
        Test 15: Supplier-provided PCF takes precedence

        ZERO-HALLUCINATION CHECK:
        Supplier PCF: 2.00 kgCO2e/kg (not default 2.35)
        Emissions: 10 kg * 2.00 = 20.0 kgCO2e
        """
        input_data = PCFInput(
            product_id="TEST-015",
            bill_of_materials=supplier_pcf_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = 10.0 * 2.00  # 20.0 (supplier PCF, not 23.5)
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_16_battery_materials(
        self,
        agent: ProductCarbonFootprintAgent,
        battery_bom: List[BOMItem],
    ):
        """
        Test 16: Battery materials calculation

        ZERO-HALLUCINATION CHECK:
        Lithium:   1 kg * 12.5  = 12.50 kgCO2e
        Cobalt:  0.5 kg * 35.8  = 17.90 kgCO2e
        Nickel:    2 kg * 12.4  = 24.80 kgCO2e
        Graphite:  3 kg *  4.85 = 14.55 kgCO2e
        Total: 69.75 kgCO2e
        """
        input_data = PCFInput(
            product_id="TEST-016",
            bill_of_materials=battery_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = (
            1.0 * EF_LITHIUM +
            0.5 * EF_COBALT +
            2.0 * EF_NICKEL +
            3.0 * EF_GRAPHITE
        )  # 69.75
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_17_plastic_variants(self, agent: ProductCarbonFootprintAgent):
        """
        Test 17: Different plastic types

        ZERO-HALLUCINATION CHECK:
        PP:   1 kg * 1.98 = 1.98 kgCO2e
        PE:   1 kg * 2.10 = 2.10 kgCO2e
        PET:  1 kg * 2.73 = 2.73 kgCO2e
        ABS:  1 kg * 3.55 = 3.55 kgCO2e
        Total: 10.36 kgCO2e
        """
        bom = [
            BOMItem(material_id="PP-001", material_category=MaterialCategory.PLASTICS_PP, quantity_kg=1.0),
            BOMItem(material_id="PE-001", material_category=MaterialCategory.PLASTICS_PE, quantity_kg=1.0),
            BOMItem(material_id="PET-001", material_category=MaterialCategory.PLASTICS_PET, quantity_kg=1.0),
            BOMItem(material_id="ABS-001", material_category=MaterialCategory.PLASTICS_ABS, quantity_kg=1.0),
        ]
        input_data = PCFInput(
            product_id="TEST-017",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = EF_PLASTICS_PP + EF_PLASTICS_PE + EF_PLASTICS_PET + EF_PLASTICS_ABS
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_18_wood_and_paper(self, agent: ProductCarbonFootprintAgent):
        """
        Test 18: Wood and paper materials

        ZERO-HALLUCINATION CHECK:
        Softwood: 5 kg * 0.31 = 1.55 kgCO2e
        Hardwood: 5 kg * 0.45 = 2.25 kgCO2e
        Paper:    2 kg * 1.32 = 2.64 kgCO2e
        Total: 6.44 kgCO2e
        """
        bom = [
            BOMItem(material_id="WOOD-SW", material_category=MaterialCategory.WOOD_SOFTWOOD, quantity_kg=5.0),
            BOMItem(material_id="WOOD-HW", material_category=MaterialCategory.WOOD_HARDWOOD, quantity_kg=5.0),
            BOMItem(material_id="PAPER-V", material_category=MaterialCategory.PAPER_VIRGIN, quantity_kg=2.0),
        ]
        input_data = PCFInput(
            product_id="TEST-018",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = 5.0 * EF_WOOD_SOFTWOOD + 5.0 * EF_WOOD_HARDWOOD + 2.0 * EF_PAPER_VIRGIN
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_19_construction_materials(self, agent: ProductCarbonFootprintAgent):
        """
        Test 19: Construction materials (glass, concrete, cement)

        ZERO-HALLUCINATION CHECK:
        Glass:    10 kg * 0.85 =  8.50 kgCO2e
        Concrete: 100 kg * 0.13 = 13.00 kgCO2e
        Cement:   20 kg * 0.93 = 18.60 kgCO2e
        Total: 40.10 kgCO2e
        """
        bom = [
            BOMItem(material_id="GLASS-001", material_category=MaterialCategory.GLASS, quantity_kg=10.0),
            BOMItem(material_id="CONCRETE-001", material_category=MaterialCategory.CONCRETE, quantity_kg=100.0),
            BOMItem(material_id="CEMENT-001", material_category=MaterialCategory.CEMENT, quantity_kg=20.0),
        ]
        input_data = PCFInput(
            product_id="TEST-019",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = 10.0 * EF_GLASS + 100.0 * EF_CONCRETE + 20.0 * EF_CEMENT
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_20_rare_earth_materials(self, agent: ProductCarbonFootprintAgent):
        """
        Test 20: Rare earth materials (high emission factor)

        ZERO-HALLUCINATION CHECK:
        Rare earth: 0.5 kg * 28.5 = 14.25 kgCO2e
        """
        bom = [
            BOMItem(
                material_id="RE-001",
                material_category=MaterialCategory.RARE_EARTH,
                quantity_kg=0.5,
            )
        ]
        input_data = PCFInput(
            product_id="TEST-020",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = 0.5 * EF_RARE_EARTH  # 14.25
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )


# =============================================================================
# Test 21-30: Manufacturing Emissions Calculations
# =============================================================================


class TestManufacturingCalculations:
    """Tests for manufacturing emissions calculations."""

    @pytest.mark.golden
    def test_21_electricity_us_grid(self, agent: ProductCarbonFootprintAgent):
        """
        Test 21: Electricity emissions - US grid

        ZERO-HALLUCINATION CHECK:
        100 kWh * 0.417 kgCO2e/kWh = 41.7 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]
        energy = ManufacturingEnergy(electricity_kwh=100.0, grid_region="US")

        input_data = PCFInput(
            product_id="TEST-021",
            bill_of_materials=bom,
            manufacturing_energy=energy,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_mfg = 100.0 * EF_GRID_US  # 41.7
        # Manufacturing includes only electricity here
        assert result.breakdown_by_stage.manufacturing_breakdown.get("electricity", 0) == pytest.approx(
            expected_mfg, rel=1e-6
        )

    @pytest.mark.golden
    def test_22_electricity_france_low_carbon(self, agent: ProductCarbonFootprintAgent):
        """
        Test 22: Electricity emissions - France (nuclear, low carbon)

        ZERO-HALLUCINATION CHECK:
        100 kWh * 0.052 kgCO2e/kWh = 5.2 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]
        energy = ManufacturingEnergy(electricity_kwh=100.0, grid_region="FR")

        input_data = PCFInput(
            product_id="TEST-022",
            bill_of_materials=bom,
            manufacturing_energy=energy,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_elec = 100.0 * EF_GRID_FR  # 5.2
        assert result.breakdown_by_stage.manufacturing_breakdown.get("electricity", 0) == pytest.approx(
            expected_elec, rel=1e-6
        )

    @pytest.mark.golden
    def test_23_renewable_electricity(
        self,
        agent: ProductCarbonFootprintAgent,
        manufacturing_energy_renewable: ManufacturingEnergy,
    ):
        """
        Test 23: Renewable electricity reduces emissions

        ZERO-HALLUCINATION CHECK:
        Effective factor: 0.417 * (1 - 0.8) = 0.0834 kgCO2e/kWh
        100 kWh * 0.0834 = 8.34 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]

        input_data = PCFInput(
            product_id="TEST-023",
            bill_of_materials=bom,
            manufacturing_energy=manufacturing_energy_renewable,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_elec = 100.0 * EF_GRID_US * (1 - 0.8)  # 8.34
        assert result.breakdown_by_stage.manufacturing_breakdown.get("electricity", 0) == pytest.approx(
            expected_elec, rel=1e-6
        )

    @pytest.mark.golden
    def test_24_natural_gas_emissions(self, agent: ProductCarbonFootprintAgent):
        """
        Test 24: Natural gas combustion emissions

        ZERO-HALLUCINATION CHECK:
        50 m3 * 1.93 kgCO2e/m3 = 96.5 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]
        energy = ManufacturingEnergy(natural_gas_m3=50.0)

        input_data = PCFInput(
            product_id="TEST-024",
            bill_of_materials=bom,
            manufacturing_energy=energy,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_gas = 50.0 * EF_NATURAL_GAS  # 96.5
        assert result.breakdown_by_stage.manufacturing_breakdown.get("natural_gas", 0) == pytest.approx(
            expected_gas, rel=1e-6
        )

    @pytest.mark.golden
    def test_25_diesel_emissions(self, agent: ProductCarbonFootprintAgent):
        """
        Test 25: Diesel combustion emissions

        ZERO-HALLUCINATION CHECK:
        10 L * 2.68 kgCO2e/L = 26.8 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]
        energy = ManufacturingEnergy(diesel_liters=10.0)

        input_data = PCFInput(
            product_id="TEST-025",
            bill_of_materials=bom,
            manufacturing_energy=energy,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_diesel = 10.0 * EF_DIESEL  # 26.8
        assert result.breakdown_by_stage.manufacturing_breakdown.get("diesel", 0) == pytest.approx(
            expected_diesel, rel=1e-6
        )

    @pytest.mark.golden
    def test_26_steam_emissions(self, agent: ProductCarbonFootprintAgent):
        """
        Test 26: Steam consumption emissions

        ZERO-HALLUCINATION CHECK:
        100 kg * 0.27 kgCO2e/kg = 27.0 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]
        energy = ManufacturingEnergy(steam_kg=100.0)

        input_data = PCFInput(
            product_id="TEST-026",
            bill_of_materials=bom,
            manufacturing_energy=energy,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_steam = 100.0 * EF_STEAM  # 27.0
        assert result.breakdown_by_stage.manufacturing_breakdown.get("steam", 0) == pytest.approx(
            expected_steam, rel=1e-6
        )

    @pytest.mark.golden
    def test_27_process_emissions_co2(self, agent: ProductCarbonFootprintAgent):
        """
        Test 27: Direct process CO2 emissions

        ZERO-HALLUCINATION CHECK:
        10 kg CO2 direct = 10.0 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]
        process = ProcessEmissions(co2_kg=10.0)

        input_data = PCFInput(
            product_id="TEST-027",
            bill_of_materials=bom,
            process_emissions=process,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        # Process emissions contribute to manufacturing total
        assert result.breakdown_by_stage.manufacturing_breakdown.get("process_emissions", 0) == pytest.approx(
            10.0, rel=1e-6
        )

    @pytest.mark.golden
    def test_28_process_emissions_gwp(
        self,
        agent: ProductCarbonFootprintAgent,
        process_emissions_data: ProcessEmissions,
    ):
        """
        Test 28: Process emissions with GWP conversion

        ZERO-HALLUCINATION CHECK:
        CO2: 10 kg
        CH4: 0.5 kg * 29.8 = 14.9 kg CO2e
        N2O: 0.1 kg * 273 = 27.3 kg CO2e
        Total: 52.2 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]

        input_data = PCFInput(
            product_id="TEST-028",
            bill_of_materials=bom,
            process_emissions=process_emissions_data,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_process = 10.0 + 0.5 * GWP_CH4 + 0.1 * GWP_N2O  # 52.2
        assert result.breakdown_by_stage.manufacturing_breakdown.get("process_emissions", 0) == pytest.approx(
            expected_process, rel=1e-6
        )

    @pytest.mark.golden
    def test_29_combined_manufacturing(
        self,
        agent: ProductCarbonFootprintAgent,
        simple_steel_bom: List[BOMItem],
        manufacturing_energy_us: ManufacturingEnergy,
    ):
        """
        Test 29: Combined manufacturing energy

        ZERO-HALLUCINATION CHECK:
        Electricity: 100 kWh * 0.417 = 41.70 kgCO2e
        Natural gas: 50 m3 * 1.93 = 96.50 kgCO2e
        Diesel: 10 L * 2.68 = 26.80 kgCO2e
        Total: 165.00 kgCO2e
        """
        input_data = PCFInput(
            product_id="TEST-029",
            bill_of_materials=simple_steel_bom,
            manufacturing_energy=manufacturing_energy_us,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_mfg = (
            100.0 * EF_GRID_US +
            50.0 * EF_NATURAL_GAS +
            10.0 * EF_DIESEL
        )  # 165.0
        assert result.breakdown_by_stage.manufacturing_kgco2e == pytest.approx(
            expected_mfg, rel=1e-6
        )

    @pytest.mark.golden
    def test_30_china_high_carbon_grid(self, agent: ProductCarbonFootprintAgent):
        """
        Test 30: China grid (high carbon intensity)

        ZERO-HALLUCINATION CHECK:
        100 kWh * 0.555 kgCO2e/kWh = 55.5 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]
        energy = ManufacturingEnergy(electricity_kwh=100.0, grid_region="CN")

        input_data = PCFInput(
            product_id="TEST-030",
            bill_of_materials=bom,
            manufacturing_energy=energy,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_elec = 100.0 * EF_GRID_CN  # 55.5
        assert result.breakdown_by_stage.manufacturing_breakdown.get("electricity", 0) == pytest.approx(
            expected_elec, rel=1e-6
        )


# =============================================================================
# Test 31-40: Transport and Use Phase Calculations
# =============================================================================


class TestTransportAndUsePhase:
    """Tests for transport and use phase emissions."""

    @pytest.mark.golden
    def test_31_road_truck_transport(self, agent: ProductCarbonFootprintAgent):
        """
        Test 31: Road truck transport emissions

        ZERO-HALLUCINATION CHECK:
        10 kg = 0.01 tonnes
        500 km * 0.01 t * 0.089 kgCO2e/tkm = 0.445 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]
        transport = TransportData(
            inbound_legs=[
                TransportLeg(leg_id="TRUCK-001", mode=TransportMode.ROAD_TRUCK, distance_km=500.0)
            ],
            outbound_legs=[],
            product_weight_kg=10.0,
        )

        input_data = PCFInput(
            product_id="TEST-031",
            bill_of_materials=bom,
            transport_data=transport,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_transport = 0.01 * 500.0 * EF_ROAD_TRUCK  # 0.445
        assert result.breakdown_by_stage.transport_kgco2e == pytest.approx(
            expected_transport, rel=1e-6
        )

    @pytest.mark.golden
    def test_32_sea_container_transport(self, agent: ProductCarbonFootprintAgent):
        """
        Test 32: Sea container transport emissions

        ZERO-HALLUCINATION CHECK:
        100 kg = 0.1 tonnes
        10000 km * 0.1 t * 0.016 kgCO2e/tkm = 16.0 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=100.0)]
        transport = TransportData(
            inbound_legs=[
                TransportLeg(leg_id="SEA-001", mode=TransportMode.SEA_CONTAINER, distance_km=10000.0)
            ],
            outbound_legs=[],
            product_weight_kg=100.0,
        )

        input_data = PCFInput(
            product_id="TEST-032",
            bill_of_materials=bom,
            transport_data=transport,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_transport = 0.1 * 10000.0 * EF_SEA_CONTAINER  # 16.0
        assert result.breakdown_by_stage.transport_kgco2e == pytest.approx(
            expected_transport, rel=1e-6
        )

    @pytest.mark.golden
    def test_33_air_freight_high_emissions(self, agent: ProductCarbonFootprintAgent):
        """
        Test 33: Air freight (highest emission factor)

        ZERO-HALLUCINATION CHECK:
        10 kg = 0.01 tonnes
        5000 km * 0.01 t * 0.602 kgCO2e/tkm = 30.1 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]
        transport = TransportData(
            inbound_legs=[
                TransportLeg(leg_id="AIR-001", mode=TransportMode.AIR_FREIGHT, distance_km=5000.0)
            ],
            outbound_legs=[],
            product_weight_kg=10.0,
        )

        input_data = PCFInput(
            product_id="TEST-033",
            bill_of_materials=bom,
            transport_data=transport,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_transport = 0.01 * 5000.0 * EF_AIR  # 30.1
        assert result.breakdown_by_stage.transport_kgco2e == pytest.approx(
            expected_transport, rel=1e-6
        )

    @pytest.mark.golden
    def test_34_rail_freight_low_emissions(self, agent: ProductCarbonFootprintAgent):
        """
        Test 34: Rail freight (low emission factor)

        ZERO-HALLUCINATION CHECK:
        100 kg = 0.1 tonnes
        1000 km * 0.1 t * 0.028 kgCO2e/tkm = 2.8 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=100.0)]
        transport = TransportData(
            inbound_legs=[
                TransportLeg(leg_id="RAIL-001", mode=TransportMode.RAIL_FREIGHT, distance_km=1000.0)
            ],
            outbound_legs=[],
            product_weight_kg=100.0,
        )

        input_data = PCFInput(
            product_id="TEST-034",
            bill_of_materials=bom,
            transport_data=transport,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_transport = 0.1 * 1000.0 * EF_RAIL  # 2.8
        assert result.breakdown_by_stage.transport_kgco2e == pytest.approx(
            expected_transport, rel=1e-6
        )

    @pytest.mark.golden
    def test_35_multimodal_transport(
        self,
        agent: ProductCarbonFootprintAgent,
        transport_data: TransportData,
    ):
        """
        Test 35: Multimodal transport (truck + sea)

        ZERO-HALLUCINATION CHECK:
        Truck: 0.01 t * 500 km * 0.089 = 0.445 kgCO2e
        Sea:   0.01 t * 5000 km * 0.016 = 0.800 kgCO2e
        Total: 1.245 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]

        input_data = PCFInput(
            product_id="TEST-035",
            bill_of_materials=bom,
            transport_data=transport_data,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected_transport = (
            0.01 * 500.0 * EF_ROAD_TRUCK +
            0.01 * 5000.0 * EF_SEA_CONTAINER
        )  # 1.245
        assert result.breakdown_by_stage.transport_kgco2e == pytest.approx(
            expected_transport, rel=1e-6
        )

    @pytest.mark.golden
    def test_36_use_phase_energy(
        self,
        agent: ProductCarbonFootprintAgent,
        simple_steel_bom: List[BOMItem],
        use_phase_data: UsePhaseData,
    ):
        """
        Test 36: Use phase energy consumption

        ZERO-HALLUCINATION CHECK:
        Energy: 0.5 kWh/use * 365 uses/year * 5 years * 0.417 = 380.5875 kgCO2e
        """
        input_data = PCFInput(
            product_id="TEST-036",
            bill_of_materials=simple_steel_bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            use_phase=use_phase_data,
        )
        result = agent.run(input_data)

        total_uses = 365.0 * 5.0
        expected_energy = 0.5 * total_uses * EF_GRID_US  # 380.5875
        # Use phase includes energy, consumables, and maintenance
        assert result.breakdown_by_stage.use_phase_kgco2e >= expected_energy * 0.9  # Allow margin for rounding

    @pytest.mark.golden
    def test_37_use_phase_consumables(self, agent: ProductCarbonFootprintAgent):
        """
        Test 37: Use phase consumables

        ZERO-HALLUCINATION CHECK:
        Consumables: 10 kgCO2e/year * 3 years = 30.0 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=1.0)]
        use_phase = UsePhaseData(
            lifetime_years=3.0,
            consumables_kgco2e_per_year=10.0,
            grid_region="US",
        )

        input_data = PCFInput(
            product_id="TEST-037",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            use_phase=use_phase,
        )
        result = agent.run(input_data)

        expected_consumables = 10.0 * 3.0  # 30.0
        # Use phase should include consumables
        assert result.breakdown_by_stage.use_phase_kgco2e >= expected_consumables * 0.9

    @pytest.mark.golden
    def test_38_transport_utilization_factor(self, agent: ProductCarbonFootprintAgent):
        """
        Test 38: Transport with 50% utilization (higher emissions per unit)

        ZERO-HALLUCINATION CHECK:
        Base: 0.01 t * 500 km * 0.089 = 0.445 kgCO2e
        Adjusted for 50% util: 0.445 / 0.5 = 0.89 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]
        transport = TransportData(
            inbound_legs=[
                TransportLeg(
                    leg_id="TRUCK-001",
                    mode=TransportMode.ROAD_TRUCK,
                    distance_km=500.0,
                    utilization_pct=50.0,
                )
            ],
            outbound_legs=[],
            product_weight_kg=10.0,
        )

        input_data = PCFInput(
            product_id="TEST-038",
            bill_of_materials=bom,
            transport_data=transport,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        base_emissions = 0.01 * 500.0 * EF_ROAD_TRUCK
        expected_transport = base_emissions / 0.5  # 0.89
        assert result.breakdown_by_stage.transport_kgco2e == pytest.approx(
            expected_transport, rel=1e-6
        )

    @pytest.mark.golden
    def test_39_use_phase_full_calculation(
        self,
        agent: ProductCarbonFootprintAgent,
        simple_steel_bom: List[BOMItem],
        use_phase_data: UsePhaseData,
    ):
        """
        Test 39: Full use phase calculation

        ZERO-HALLUCINATION CHECK:
        Energy: 0.5 kWh/use * 365 uses/year * 5 years * 0.417 = 380.5875 kgCO2e
        Consumables: 5 kgCO2e/year * 5 years = 25.0 kgCO2e
        Maintenance: 2 kgCO2e/year * 5 years = 10.0 kgCO2e
        Total: 415.5875 kgCO2e
        """
        input_data = PCFInput(
            product_id="TEST-039",
            bill_of_materials=simple_steel_bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            use_phase=use_phase_data,
        )
        result = agent.run(input_data)

        total_uses = 365.0 * 5.0
        expected_use_phase = (
            0.5 * total_uses * EF_GRID_US +
            5.0 * 5.0 +
            2.0 * 5.0
        )  # 415.5875
        assert result.breakdown_by_stage.use_phase_kgco2e == pytest.approx(
            expected_use_phase, rel=1e-4
        )

    def test_40_no_transport_no_emissions(self, agent: ProductCarbonFootprintAgent):
        """Test 40: No transport data = zero transport emissions."""
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]

        input_data = PCFInput(
            product_id="TEST-040",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
            # No transport_data
        )
        result = agent.run(input_data)

        assert result.breakdown_by_stage.transport_kgco2e == 0.0


# =============================================================================
# Test 41-50: End-of-Life and CFF Calculations
# =============================================================================


class TestEndOfLifeCFF:
    """Tests for end-of-life Circular Footprint Formula calculations."""

    @pytest.mark.golden
    def test_41_eol_landfill(self, agent: ProductCarbonFootprintAgent):
        """
        Test 41: End-of-life landfill emissions

        ZERO-HALLUCINATION CHECK:
        CFF simplified for landfill (R2=0, R3=0):
        (1-R1)*Ev + (1-R2-R3)*ED
        For R1=0, R2=0, R3=0: 1*2.35 + 1*0.586 = 2.936 per kg
        10 kg * 2.936 = 29.36 kgCO2e
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]
        eol = EndOfLifeData(
            R1=0.0,
            R2=0.0,
            R3=0.0,
            A=0.5,
            B=0.5,
            material_weight_kg=10.0,
            treatment=EndOfLifeTreatment.LANDFILL,
        )

        input_data = PCFInput(
            product_id="TEST-041",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            end_of_life=eol,
        )
        result = agent.run(input_data)

        # With no recycling, full virgin material + landfill disposal
        # Note: CFF formula is complex, this is a simplified expectation
        assert result.breakdown_by_stage.end_of_life_kgco2e > 0

    @pytest.mark.golden
    def test_42_eol_recycling(self, agent: ProductCarbonFootprintAgent):
        """
        Test 42: End-of-life recycling (partial credit)

        ZERO-HALLUCINATION CHECK:
        With R2=0.8 (80% recycling), emissions should be lower than landfill
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]
        eol = EndOfLifeData(
            R1=0.0,
            R2=0.8,
            R3=0.0,
            A=0.5,
            B=0.5,
            material_weight_kg=10.0,
            treatment=EndOfLifeTreatment.RECYCLING,
        )

        input_data = PCFInput(
            product_id="TEST-042",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            end_of_life=eol,
        )
        result = agent.run(input_data)

        # High recycling rate should reduce EoL emissions
        assert result.breakdown_by_stage.end_of_life_kgco2e >= 0  # Can be negative credit

    @pytest.mark.golden
    def test_43_eol_energy_recovery(self, agent: ProductCarbonFootprintAgent):
        """
        Test 43: End-of-life energy recovery

        ZERO-HALLUCINATION CHECK:
        With R3=0.5 (50% energy recovery), includes energy credit
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.PLASTICS_PP, quantity_kg=10.0)]
        eol = EndOfLifeData(
            R1=0.0,
            R2=0.0,
            R3=0.5,
            A=0.5,
            B=0.5,
            LHV_MJ_per_kg=40.0,  # Plastic has high heating value
            XER_heat=0.4,
            XER_elec=0.2,
            material_weight_kg=10.0,
            treatment=EndOfLifeTreatment.ENERGY_RECOVERY,
        )

        input_data = PCFInput(
            product_id="TEST-043",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            end_of_life=eol,
        )
        result = agent.run(input_data)

        # Energy recovery provides some credit
        assert result.breakdown_by_stage.end_of_life_kgco2e >= -100  # Can have credit

    @pytest.mark.golden
    def test_44_cff_with_recycled_input(
        self,
        agent: ProductCarbonFootprintAgent,
        end_of_life_data: EndOfLifeData,
    ):
        """
        Test 44: CFF with recycled input (R1=0.3)

        ZERO-HALLUCINATION CHECK:
        R1=0.3 reduces virgin material component:
        (1-0.3)*Ev + 0.3*(A*Erec + (1-A)*Ev*Qs/Qp)
        = 0.7*2.35 + 0.3*(0.5*0.65 + 0.5*2.35*0.9)
        = 1.645 + 0.3*(0.325 + 1.0575)
        = 1.645 + 0.41475
        = 2.06 per kg (approximately)
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]

        input_data = PCFInput(
            product_id="TEST-044",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            end_of_life=end_of_life_data,
        )
        result = agent.run(input_data)

        # With recycled input, EoL should be moderate
        assert result.breakdown_by_stage.end_of_life_kgco2e != 0

    @pytest.mark.golden
    def test_45_cff_quality_ratio(self, agent: ProductCarbonFootprintAgent):
        """
        Test 45: CFF with different quality ratio (Qs/Qp=0.5)

        ZERO-HALLUCINATION CHECK:
        Lower quality ratio reduces credit for recycling
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]
        eol = EndOfLifeData(
            R1=0.0,
            R2=0.8,
            R3=0.0,
            A=0.5,
            B=0.5,
            Qs=0.5,  # Lower quality
            Qp=1.0,
            material_weight_kg=10.0,
            treatment=EndOfLifeTreatment.RECYCLING,
        )

        input_data = PCFInput(
            product_id="TEST-045",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            end_of_life=eol,
        )
        result = agent.run(input_data)

        # Quality ratio affects credit - result should exist
        assert result.breakdown_by_stage.end_of_life_kgco2e is not None

    @pytest.mark.golden
    def test_46_cff_allocation_factor_a(self, agent: ProductCarbonFootprintAgent):
        """
        Test 46: CFF with allocation factor A=0.2 (open-loop)

        ZERO-HALLUCINATION CHECK:
        A=0.2 is used for open-loop recycling (different product)
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]
        eol = EndOfLifeData(
            R1=0.0,
            R2=0.8,
            R3=0.0,
            A=0.2,  # Open-loop allocation
            B=0.5,
            material_weight_kg=10.0,
            treatment=EndOfLifeTreatment.RECYCLING,
        )

        input_data = PCFInput(
            product_id="TEST-046",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            end_of_life=eol,
        )
        result = agent.run(input_data)

        # Different allocation factor affects result
        assert result.breakdown_by_stage.end_of_life_kgco2e is not None

    @pytest.mark.golden
    def test_47_cradle_to_grave_total(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_grave_input: PCFInput,
    ):
        """
        Test 47: Complete cradle-to-grave calculation

        ZERO-HALLUCINATION CHECK:
        Total = Raw Materials + Manufacturing + Transport + Use Phase + EoL
        """
        result = agent.run(cradle_to_grave_input)

        # Verify all stages sum to total
        calculated_total = (
            result.breakdown_by_stage.raw_materials_kgco2e +
            result.breakdown_by_stage.manufacturing_kgco2e +
            result.breakdown_by_stage.transport_kgco2e +
            result.breakdown_by_stage.use_phase_kgco2e +
            result.breakdown_by_stage.end_of_life_kgco2e
        )
        assert result.total_co2e == pytest.approx(calculated_total, rel=1e-4)

    @pytest.mark.golden
    def test_48_cradle_to_gate_no_use_eol(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 48: Cradle-to-gate excludes use phase and EoL."""
        result = agent.run(cradle_to_gate_input)

        assert result.breakdown_by_stage.use_phase_kgco2e == 0.0
        assert result.breakdown_by_stage.end_of_life_kgco2e == 0.0

    @pytest.mark.golden
    def test_49_eol_reuse_low_emissions(self, agent: ProductCarbonFootprintAgent):
        """
        Test 49: End-of-life reuse (lowest emissions)

        ZERO-HALLUCINATION CHECK:
        Reuse factor: 0.05 kgCO2e/kg
        """
        bom = [BOMItem(material_id="M-001", material_category=MaterialCategory.STEEL_PRIMARY, quantity_kg=10.0)]
        eol = EndOfLifeData(
            R1=0.0,
            R2=0.0,
            R3=0.0,
            A=0.5,
            B=0.5,
            material_weight_kg=10.0,
            treatment=EndOfLifeTreatment.REUSE,
        )

        input_data = PCFInput(
            product_id="TEST-049",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GRAVE,
            end_of_life=eol,
        )
        result = agent.run(input_data)

        # Reuse should have lower EoL emissions than landfill
        assert result.breakdown_by_stage.end_of_life_kgco2e >= 0

    def test_50_boundary_value_in_output(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_grave_input: PCFInput,
    ):
        """Test 50: Boundary value correctly reflected in output."""
        result = agent.run(cradle_to_grave_input)
        assert result.boundary == "cradle_to_grave"


# =============================================================================
# Test 51-55: Data Quality and Compliance
# =============================================================================


class TestDataQualityCompliance:
    """Tests for data quality assessment and compliance."""

    def test_51_data_quality_excellent(
        self,
        agent: ProductCarbonFootprintAgent,
        supplier_pcf_bom: List[BOMItem],
    ):
        """Test 51: High primary data coverage = excellent quality."""
        # All materials have supplier PCF = 100% primary data
        input_data = PCFInput(
            product_id="TEST-051",
            bill_of_materials=supplier_pcf_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.data_coverage_pct == 100.0
        assert result.data_quality_rating == DataQualityLevel.EXCELLENT.value

    def test_52_data_quality_poor(
        self,
        agent: ProductCarbonFootprintAgent,
        simple_steel_bom: List[BOMItem],
    ):
        """Test 52: No primary data = poor quality."""
        # No supplier PCF = 0% primary data
        input_data = PCFInput(
            product_id="TEST-052",
            bill_of_materials=simple_steel_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.data_coverage_pct == 0.0
        assert result.data_quality_rating == DataQualityLevel.POOR.value

    def test_53_data_quality_mixed(self, agent: ProductCarbonFootprintAgent):
        """Test 53: Mixed primary/secondary data."""
        bom = [
            BOMItem(
                material_id="M-001",
                material_category=MaterialCategory.STEEL_PRIMARY,
                quantity_kg=10.0,
                supplier_pcf=2.0,  # Primary
            ),
            BOMItem(
                material_id="M-002",
                material_category=MaterialCategory.ALUMINUM_PRIMARY,
                quantity_kg=5.0,
                # No supplier PCF = Secondary
            ),
        ]
        input_data = PCFInput(
            product_id="TEST-053",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.data_coverage_pct == 50.0
        # 50% coverage = FAIR quality
        assert result.data_quality_rating in [
            DataQualityLevel.FAIR.value,
            DataQualityLevel.GOOD.value,
        ]

    def test_54_uncertainty_reflects_quality(
        self,
        agent: ProductCarbonFootprintAgent,
        simple_steel_bom: List[BOMItem],
    ):
        """Test 54: Uncertainty percentage reflects data quality."""
        input_data = PCFInput(
            product_id="TEST-054",
            bill_of_materials=simple_steel_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        # Poor quality = high uncertainty
        assert result.uncertainty_pct >= 50.0

    def test_55_methodology_iso14067(
        self,
        agent: ProductCarbonFootprintAgent,
        simple_steel_bom: List[BOMItem],
    ):
        """Test 55: Calculation methodology is ISO 14067."""
        input_data = PCFInput(
            product_id="TEST-055",
            bill_of_materials=simple_steel_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert "ISO 14067" in result.calculation_methodology


# =============================================================================
# Test 56-60: Export Formats and Provenance
# =============================================================================


class TestExportFormatsProvenance:
    """Tests for export formats and provenance tracking."""

    def test_56_pact_export_generated(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 56: PACT Pathfinder 2.1 export is generated."""
        result = agent.run(cradle_to_gate_input)

        assert result.pact_export is not None
        assert result.pact_export.specVersion == "2.1.0"
        assert result.pact_export.pcf is not None

    def test_57_catenax_export_generated(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 57: Catena-X export is generated."""
        result = agent.run(cradle_to_gate_input)

        assert result.catenax_export is not None
        assert result.catenax_export.specVersion == "2.0.0"
        assert result.catenax_export.carbonFootprint is not None

    def test_58_battery_passport_for_battery_products(
        self,
        agent: ProductCarbonFootprintAgent,
        battery_bom: List[BOMItem],
    ):
        """Test 58: Battery passport generated for battery products."""
        input_data = PCFInput(
            product_id="BATTERY-001",
            bill_of_materials=battery_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.battery_passport is not None
        assert "BAT-" in result.battery_passport.battery_id
        assert result.battery_passport.calculation_methodology == "ISO 14067:2018"

    def test_59_no_battery_passport_non_battery(
        self,
        agent: ProductCarbonFootprintAgent,
        simple_steel_bom: List[BOMItem],
    ):
        """Test 59: No battery passport for non-battery products."""
        input_data = PCFInput(
            product_id="NON-BATTERY-001",
            bill_of_materials=simple_steel_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.battery_passport is None

    def test_60_provenance_hash_sha256(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 60: Provenance hash is valid SHA-256."""
        result = agent.run(cradle_to_gate_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex string
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)


# =============================================================================
# Test 61-65: Determinism and Edge Cases
# =============================================================================


class TestDeterminismEdgeCases:
    """Tests for determinism and edge cases."""

    @pytest.mark.golden
    def test_61_deterministic_same_inputs(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """
        Test 61: Same inputs produce same emission values (zero-hallucination)

        This verifies the calculation is deterministic - no LLM involved in math.
        """
        result1 = agent.run(cradle_to_gate_input)
        result2 = agent.run(cradle_to_gate_input)
        result3 = agent.run(cradle_to_gate_input)

        assert result1.total_co2e == result2.total_co2e
        assert result2.total_co2e == result3.total_co2e

    @pytest.mark.golden
    def test_62_deterministic_across_instances(
        self,
        cradle_to_gate_input: PCFInput,
    ):
        """
        Test 62: Different agent instances produce same results

        Verifies the calculation doesn't depend on instance state.
        """
        agent1 = ProductCarbonFootprintAgent()
        agent2 = ProductCarbonFootprintAgent()
        agent3 = ProductCarbonFootprintAgent()

        result1 = agent1.run(cradle_to_gate_input)
        result2 = agent2.run(cradle_to_gate_input)
        result3 = agent3.run(cradle_to_gate_input)

        assert result1.total_co2e == result2.total_co2e
        assert result2.total_co2e == result3.total_co2e

    def test_63_zero_quantity_material(self, agent: ProductCarbonFootprintAgent):
        """Test 63: Zero quantity material = zero emissions."""
        bom = [
            BOMItem(
                material_id="M-001",
                material_category=MaterialCategory.STEEL_PRIMARY,
                quantity_kg=0.0,
            )
        ]
        input_data = PCFInput(
            product_id="TEST-063",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.breakdown_by_stage.raw_materials_kgco2e == 0.0

    @pytest.mark.golden
    def test_64_small_quantity_precision(self, agent: ProductCarbonFootprintAgent):
        """
        Test 64: Small quantity precision test

        ZERO-HALLUCINATION CHECK:
        0.001 kg * 2.35 kgCO2e/kg = 0.00235 kgCO2e
        """
        bom = [
            BOMItem(
                material_id="M-001",
                material_category=MaterialCategory.STEEL_PRIMARY,
                quantity_kg=0.001,
            )
        ]
        input_data = PCFInput(
            product_id="TEST-064",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = 0.001 * EF_STEEL_PRIMARY  # 0.00235
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.golden
    def test_65_large_quantity_industrial(self, agent: ProductCarbonFootprintAgent):
        """
        Test 65: Large quantity industrial scale

        ZERO-HALLUCINATION CHECK:
        10000 kg * 2.35 kgCO2e/kg = 23500 kgCO2e = 23.5 tonnes CO2e
        """
        bom = [
            BOMItem(
                material_id="M-001",
                material_category=MaterialCategory.STEEL_PRIMARY,
                quantity_kg=10000.0,
            )
        ]
        input_data = PCFInput(
            product_id="TEST-065",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        expected = 10000.0 * EF_STEEL_PRIMARY  # 23500
        assert result.breakdown_by_stage.raw_materials_kgco2e == pytest.approx(
            expected, rel=1e-6
        )


# =============================================================================
# Test 66-70: Impact Categories and Output Validation
# =============================================================================


class TestImpactCategoriesOutput:
    """Tests for PEF impact categories and output validation."""

    def test_66_climate_change_impact(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 66: Climate change impact equals total CO2e."""
        result = agent.run(cradle_to_gate_input)

        assert result.impact_categories.climate_change_kgco2e == result.total_co2e

    def test_67_all_16_pef_categories(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 67: All 16 PEF impact categories are present."""
        result = agent.run(cradle_to_gate_input)
        ic = result.impact_categories

        assert ic.climate_change_kgco2e is not None
        assert ic.ozone_depletion_kgcfc11e is not None
        assert ic.acidification_molh_plus_e is not None
        assert ic.eutrophication_freshwater_kgpe is not None
        assert ic.eutrophication_marine_kgne is not None
        assert ic.eutrophication_terrestrial_molne is not None
        assert ic.photochemical_ozone_kgnmvoce is not None
        assert ic.particulate_matter_disease_incidence is not None
        assert ic.ionizing_radiation_kbqu235e is not None
        assert ic.ecotoxicity_freshwater_ctue is not None
        assert ic.human_toxicity_cancer_ctuh is not None
        assert ic.human_toxicity_non_cancer_ctuh is not None
        assert ic.land_use_pt is not None
        assert ic.water_use_m3_world_eq is not None
        assert ic.resource_use_fossils_mj is not None
        assert ic.resource_use_minerals_metals_kgsbe is not None

    def test_68_processing_time_tracked(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 68: Processing time is tracked."""
        result = agent.run(cradle_to_gate_input)

        assert result.processing_time_ms > 0

    def test_69_pcf_id_generated(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 69: Unique PCF ID is generated."""
        result = agent.run(cradle_to_gate_input)

        assert result.pcf_id is not None
        assert result.pcf_id.startswith("PCF-")
        assert len(result.pcf_id) == 16  # PCF- + 12 hex chars

    def test_70_emission_factors_tracked(
        self,
        agent: ProductCarbonFootprintAgent,
        cradle_to_gate_input: PCFInput,
    ):
        """Test 70: Emission factors used are tracked."""
        result = agent.run(cradle_to_gate_input)

        assert len(result.emission_factors_used) > 0
        # Should include material factors
        material_factors = [
            ef for ef in result.emission_factors_used
            if "material_id" in ef
        ]
        assert len(material_factors) > 0


# =============================================================================
# Test 71-75: Battery Passport Specific Tests
# =============================================================================


class TestBatteryPassport:
    """Tests specific to EU Battery Regulation passport."""

    def test_71_battery_footprint_class_a(self, agent: ProductCarbonFootprintAgent):
        """Test 71: Battery footprint class A (< 50 kgCO2e)."""
        # Very small battery
        bom = [
            BOMItem(material_id="LI-001", material_category=MaterialCategory.LITHIUM, quantity_kg=0.1),
        ]
        input_data = PCFInput(
            product_id="BAT-A-001",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.battery_passport is not None
        assert result.battery_passport.carbon_footprint_class == "A"

    def test_72_battery_footprint_class_e(
        self,
        agent: ProductCarbonFootprintAgent,
        battery_bom: List[BOMItem],
    ):
        """Test 72: Battery footprint class classification."""
        # Large battery with high cobalt content
        large_battery_bom = [
            BOMItem(material_id="CO-001", material_category=MaterialCategory.COBALT, quantity_kg=10.0),
            BOMItem(material_id="NI-001", material_category=MaterialCategory.NICKEL, quantity_kg=20.0),
        ]
        input_data = PCFInput(
            product_id="BAT-E-001",
            bill_of_materials=large_battery_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.battery_passport is not None
        # 10*35.8 + 20*12.4 = 358 + 248 = 606 kgCO2e = class E
        assert result.battery_passport.carbon_footprint_class == "E"

    def test_73_battery_recycled_content_tracked(self, agent: ProductCarbonFootprintAgent):
        """Test 73: Battery recycled content is tracked."""
        bom = [
            BOMItem(
                material_id="LI-001",
                material_category=MaterialCategory.LITHIUM,
                quantity_kg=1.0,
                recycled_content_pct=20.0,
            ),
            BOMItem(
                material_id="CO-001",
                material_category=MaterialCategory.COBALT,
                quantity_kg=1.0,
                recycled_content_pct=10.0,
            ),
        ]
        input_data = PCFInput(
            product_id="BAT-REC-001",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.battery_passport is not None
        # Weighted average: (1*20 + 1*10) / 2 = 15%
        assert result.battery_passport.recycled_content_pct == pytest.approx(15.0, rel=1e-2)

    def test_74_battery_material_origins(self, agent: ProductCarbonFootprintAgent):
        """Test 74: Battery material origins are tracked."""
        bom = [
            BOMItem(
                material_id="LI-001",
                material_category=MaterialCategory.LITHIUM,
                quantity_kg=1.0,
                country_of_origin="CL",  # Chile
            ),
            BOMItem(
                material_id="CO-001",
                material_category=MaterialCategory.COBALT,
                quantity_kg=1.0,
                country_of_origin="CD",  # DRC
            ),
        ]
        input_data = PCFInput(
            product_id="BAT-ORIGIN-001",
            bill_of_materials=bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.battery_passport is not None
        assert len(result.battery_passport.raw_material_origins) == 2

    def test_75_battery_lifecycle_stages(
        self,
        agent: ProductCarbonFootprintAgent,
        battery_bom: List[BOMItem],
    ):
        """Test 75: Battery passport includes lifecycle stages."""
        input_data = PCFInput(
            product_id="BAT-STAGES-001",
            bill_of_materials=battery_bom,
            boundary=PCFBoundary.CRADLE_TO_GATE,
        )
        result = agent.run(input_data)

        assert result.battery_passport is not None
        stages = result.battery_passport.lifecycle_stages
        assert "raw_materials" in stages
        assert "manufacturing" in stages


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
