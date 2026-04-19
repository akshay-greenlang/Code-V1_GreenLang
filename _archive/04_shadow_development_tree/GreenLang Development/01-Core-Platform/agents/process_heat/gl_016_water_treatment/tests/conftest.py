"""
GL-016 WATERGUARD Agent - Test Fixtures and Configuration

Shared pytest fixtures for water treatment monitoring tests.
Provides standardized test data following ASME/ABMA guidelines.

Author: GL-TestEngineer
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, MagicMock, AsyncMock

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    WaterQualityStatus,
    TreatmentProgram,
    BoilerPressureClass,
    BlowdownType,
    ChemicalType,
    CorrosionMechanism,
    WaterSampleInput,
    WaterQualityResult,
    BoilerWaterInput,
    BoilerWaterOutput,
    BoilerWaterLimits,
    FeedwaterInput,
    FeedwaterOutput,
    FeedwaterLimits,
    CondensateInput,
    CondensateOutput,
    CondensateLimits,
    BlowdownInput,
    BlowdownOutput,
    ChemicalDosingInput,
    ChemicalDosingOutput,
    DeaerationInput,
    DeaerationOutput,
    WaterTreatmentInput,
    WaterTreatmentOutput,
)

from greenlang.agents.process_heat.gl_016_water_treatment.config import (
    WaterTreatmentConfig,
    ASMEBoilerWaterLimits,
    ASMEFeedwaterLimits,
    PhosphateTreatmentConfig,
    OxygenScavengerConfig,
    AmineConfig,
    BlowdownConfig,
    DeaeratorConfig,
    ASME_BOILER_WATER_LIMITS,
    ASME_FEEDWATER_LIMITS,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def water_treatment_config() -> WaterTreatmentConfig:
    """Create test water treatment configuration."""
    return WaterTreatmentConfig(
        system_id="TEST-WT-001",
        name="Test Water Treatment System",
        boiler_pressure_class=BoilerPressureClass.MEDIUM_PRESSURE,
        operating_pressure_psig=450.0,
        design_pressure_psig=600.0,
        steam_capacity_lb_hr=50000.0,
        treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
        oxygen_scavenger_type=ChemicalType.SULFITE,
        amine_treatment_enabled=True,
        amine_type=ChemicalType.MORPHOLINE,
        condensate_return_pct=80.0,
    )


@pytest.fixture
def low_pressure_config() -> WaterTreatmentConfig:
    """Create low pressure boiler configuration."""
    return WaterTreatmentConfig(
        system_id="TEST-WT-LP",
        boiler_pressure_class=BoilerPressureClass.LOW_PRESSURE,
        operating_pressure_psig=150.0,
        design_pressure_psig=200.0,
        steam_capacity_lb_hr=25000.0,
        treatment_program=TreatmentProgram.PHOSPHATE_POLYMER,
    )


@pytest.fixture
def high_pressure_config() -> WaterTreatmentConfig:
    """Create high pressure boiler configuration."""
    return WaterTreatmentConfig(
        system_id="TEST-WT-HP",
        boiler_pressure_class=BoilerPressureClass.HIGH_PRESSURE,
        operating_pressure_psig=1200.0,
        design_pressure_psig=1500.0,
        steam_capacity_lb_hr=100000.0,
        treatment_program=TreatmentProgram.CONGRUENT_PHOSPHATE,
    )


@pytest.fixture
def asme_boiler_limits_medium() -> ASMEBoilerWaterLimits:
    """ASME limits for medium pressure boiler."""
    return ASME_BOILER_WATER_LIMITS["medium_pressure"]


@pytest.fixture
def asme_feedwater_limits_medium() -> ASMEFeedwaterLimits:
    """ASME limits for medium pressure feedwater."""
    return ASME_FEEDWATER_LIMITS["medium_pressure"]


# =============================================================================
# BOILER WATER FIXTURES
# =============================================================================

@pytest.fixture
def boiler_water_input_excellent() -> BoilerWaterInput:
    """Create excellent quality boiler water sample."""
    return BoilerWaterInput(
        sample_id="BW-EXC-001",
        sample_point="boiler_drum",
        ph=10.5,
        p_alkalinity_ppm=100.0,
        m_alkalinity_ppm=150.0,
        phosphate_ppm=8.0,
        specific_conductivity_umho=2500.0,
        cation_conductivity_umho=0.3,
        silica_ppm=15.0,
        tds_ppm=1500.0,
        iron_ppb=30.0,
        copper_ppb=5.0,
        dissolved_oxygen_ppb=2.0,
        operating_pressure_psig=450.0,
        steam_purity_ppb=0.2,
    )


@pytest.fixture
def boiler_water_input_warning() -> BoilerWaterInput:
    """Create warning quality boiler water sample."""
    return BoilerWaterInput(
        sample_id="BW-WARN-001",
        sample_point="boiler_drum",
        ph=11.8,
        p_alkalinity_ppm=400.0,
        m_alkalinity_ppm=500.0,
        phosphate_ppm=15.0,
        specific_conductivity_umho=4500.0,
        silica_ppm=25.0,
        tds_ppm=2300.0,
        iron_ppb=90.0,
        copper_ppb=18.0,
        operating_pressure_psig=450.0,
    )


@pytest.fixture
def boiler_water_input_critical() -> BoilerWaterInput:
    """Create critical quality boiler water sample."""
    return BoilerWaterInput(
        sample_id="BW-CRIT-001",
        sample_point="boiler_drum",
        ph=8.5,  # Too low - hydrogen damage risk
        p_alkalinity_ppm=50.0,
        m_alkalinity_ppm=80.0,
        phosphate_ppm=2.0,
        specific_conductivity_umho=7000.0,  # Exceeds limit
        silica_ppm=50.0,  # High
        tds_ppm=4000.0,  # Exceeds limit
        iron_ppb=250.0,  # Critical
        copper_ppb=60.0,  # Critical
        dissolved_oxygen_ppb=25.0,  # Critical
        operating_pressure_psig=450.0,
    )


@pytest.fixture
def boiler_water_input_acid_phosphate() -> BoilerWaterInput:
    """Create boiler water sample with acid phosphate conditions."""
    return BoilerWaterInput(
        sample_id="BW-AP-001",
        sample_point="boiler_drum",
        ph=9.2,  # Low for coordinated phosphate
        p_alkalinity_ppm=20.0,  # Low alkalinity
        m_alkalinity_ppm=40.0,
        phosphate_ppm=10.0,
        specific_conductivity_umho=3000.0,
        operating_pressure_psig=450.0,
    )


# =============================================================================
# FEEDWATER FIXTURES
# =============================================================================

@pytest.fixture
def feedwater_input_excellent() -> FeedwaterInput:
    """Create excellent quality feedwater sample."""
    return FeedwaterInput(
        sample_id="FW-EXC-001",
        sample_point="deaerator_outlet",
        ph=9.0,
        specific_conductivity_umho=0.3,
        cation_conductivity_umho=0.2,
        dissolved_oxygen_ppb=3.0,
        total_hardness_ppm=0.05,
        iron_ppb=8.0,
        copper_ppb=3.0,
        silica_ppm=0.1,
        oxygen_scavenger_residual_ppm=30.0,
        oxygen_scavenger_type=ChemicalType.SULFITE,
        temperature_f=227.0,
        deaerator_outlet=True,
    )


@pytest.fixture
def feedwater_input_warning() -> FeedwaterInput:
    """Create warning quality feedwater sample."""
    return FeedwaterInput(
        sample_id="FW-WARN-001",
        sample_point="deaerator_outlet",
        ph=8.8,
        specific_conductivity_umho=0.8,
        dissolved_oxygen_ppb=10.0,  # Elevated
        total_hardness_ppm=0.15,
        iron_ppb=25.0,  # Elevated
        copper_ppb=12.0,
        silica_ppm=0.25,
        oxygen_scavenger_residual_ppm=15.0,  # Low
        oxygen_scavenger_type=ChemicalType.SULFITE,
        temperature_f=220.0,
    )


@pytest.fixture
def feedwater_input_critical() -> FeedwaterInput:
    """Create critical quality feedwater sample."""
    return FeedwaterInput(
        sample_id="FW-CRIT-001",
        sample_point="deaerator_outlet",
        ph=8.2,  # Low
        specific_conductivity_umho=1.5,  # High
        dissolved_oxygen_ppb=25.0,  # Critical
        total_hardness_ppm=0.5,  # Hardness breakthrough
        iron_ppb=50.0,  # High
        copper_ppb=25.0,  # High
        silica_ppm=0.5,  # High
        oxygen_scavenger_residual_ppm=5.0,  # Very low
        oxygen_scavenger_type=ChemicalType.SULFITE,
        temperature_f=210.0,
    )


# =============================================================================
# CONDENSATE FIXTURES
# =============================================================================

@pytest.fixture
def condensate_input_excellent() -> CondensateInput:
    """Create excellent quality condensate sample."""
    return CondensateInput(
        sample_id="COND-EXC-001",
        sample_point="main_condensate_return",
        ph=8.7,
        specific_conductivity_umho=0.5,
        cation_conductivity_umho=0.3,
        iron_ppb=15.0,
        copper_ppb=4.0,
        dissolved_oxygen_ppb=8.0,
        amine_residual_ppm=5.0,
        amine_type=ChemicalType.MORPHOLINE,
        condensate_return_pct=85.0,
        condensate_source="main_return",
    )


@pytest.fixture
def condensate_input_contaminated() -> CondensateInput:
    """Create contaminated condensate sample."""
    return CondensateInput(
        sample_id="COND-CONT-001",
        sample_point="process_condensate",
        ph=7.5,
        specific_conductivity_umho=5.0,
        iron_ppb=150.0,
        copper_ppb=30.0,
        dissolved_oxygen_ppb=50.0,
        hardness_ppm=2.0,  # Contamination indicator
        oil_ppm=3.0,  # Oil contamination
        tds_ppm=100.0,
        condensate_return_pct=60.0,
        condensate_source="process_area",
    )


@pytest.fixture
def condensate_input_corrosion() -> CondensateInput:
    """Create condensate sample with corrosion indicators."""
    return CondensateInput(
        sample_id="COND-CORR-001",
        sample_point="far_condensate_return",
        ph=7.0,  # Low - carbonic acid corrosion
        specific_conductivity_umho=1.5,
        iron_ppb=250.0,  # High - active corrosion
        copper_ppb=15.0,
        dissolved_oxygen_ppb=30.0,  # Oxygen ingress
        amine_residual_ppm=1.0,  # Inadequate amine
        amine_type=ChemicalType.MORPHOLINE,
        condensate_return_pct=75.0,
    )


# =============================================================================
# BLOWDOWN FIXTURES
# =============================================================================

@pytest.fixture
def blowdown_input_optimize() -> BlowdownInput:
    """Create blowdown data for optimization testing."""
    return BlowdownInput(
        continuous_blowdown_rate_pct=5.0,
        intermittent_blowdown_frequency_per_shift=2,
        blowdown_type=BlowdownType.CONTINUOUS,
        boiler_tds_ppm=2000.0,
        feedwater_tds_ppm=50.0,
        boiler_conductivity_umho=4000.0,
        feedwater_conductivity_umho=100.0,
        tds_max_ppm=2500.0,
        steam_flow_rate_lb_hr=50000.0,
        operating_pressure_psig=450.0,
        blowdown_heat_recovery_enabled=False,
        flash_tank_pressure_psig=5.0,
        fuel_cost_per_mmbtu=5.0,
        water_cost_per_kgal=3.0,
        chemical_cost_per_kgal=2.0,
    )


@pytest.fixture
def blowdown_input_high_tds() -> BlowdownInput:
    """Create blowdown data with high TDS."""
    return BlowdownInput(
        continuous_blowdown_rate_pct=8.0,
        blowdown_type=BlowdownType.CONTINUOUS,
        boiler_tds_ppm=3500.0,  # High
        feedwater_tds_ppm=100.0,
        tds_max_ppm=2500.0,
        steam_flow_rate_lb_hr=50000.0,
        operating_pressure_psig=450.0,
        blowdown_heat_recovery_enabled=True,
        flash_tank_pressure_psig=5.0,
    )


# =============================================================================
# CHEMICAL DOSING FIXTURES
# =============================================================================

@pytest.fixture
def chemical_dosing_input_standard() -> ChemicalDosingInput:
    """Create standard chemical dosing input."""
    return ChemicalDosingInput(
        feedwater_flow_lb_hr=55000.0,
        makeup_water_flow_lb_hr=10000.0,
        feedwater_do_ppb=5.0,
        current_scavenger_type=ChemicalType.SULFITE,
        current_scavenger_dose_ppm=25.0,
        target_scavenger_residual_ppm=30.0,
        boiler_phosphate_ppm=8.0,
        target_phosphate_ppm=10.0,
        current_phosphate_dose_ppm=0.3,
        blowdown_rate_pct=3.0,
        condensate_return_pct=80.0,
        condensate_ph=8.5,
        target_condensate_ph=8.8,
        current_amine_type=ChemicalType.MORPHOLINE,
        current_amine_dose_ppm=4.0,
        operating_pressure_psig=450.0,
        treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
        scavenger_cost_per_lb=1.50,
        phosphate_cost_per_lb=0.80,
        amine_cost_per_lb=5.00,
    )


@pytest.fixture
def chemical_dosing_input_high_do() -> ChemicalDosingInput:
    """Create chemical dosing input with high dissolved oxygen."""
    return ChemicalDosingInput(
        feedwater_flow_lb_hr=55000.0,
        feedwater_do_ppb=15.0,  # High DO
        current_scavenger_type=ChemicalType.SULFITE,
        current_scavenger_dose_ppm=20.0,  # Insufficient
        target_scavenger_residual_ppm=30.0,
        boiler_phosphate_ppm=5.0,  # Low
        target_phosphate_ppm=10.0,
        current_phosphate_dose_ppm=0.2,
        blowdown_rate_pct=4.0,
        operating_pressure_psig=450.0,
        treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
    )


# =============================================================================
# DEAERATION FIXTURES
# =============================================================================

@pytest.fixture
def deaeration_input_excellent() -> DeaerationInput:
    """Create excellent deaerator performance input."""
    return DeaerationInput(
        deaerator_pressure_psig=5.0,
        deaerator_temperature_f=227.0,
        inlet_water_temperature_f=150.0,
        inlet_dissolved_oxygen_ppb=8000.0,  # 8 ppm typical for cold water
        inlet_co2_ppm=10.0,
        outlet_dissolved_oxygen_ppb=3.0,  # Excellent
        outlet_co2_ppm=0.5,
        total_flow_lb_hr=55000.0,
        steam_flow_lb_hr=2800.0,
        vent_rate_lb_hr=30.0,
        deaerator_type="spray_tray",
        outlet_o2_limit_ppb=7.0,
    )


@pytest.fixture
def deaeration_input_poor() -> DeaerationInput:
    """Create poor deaerator performance input."""
    return DeaerationInput(
        deaerator_pressure_psig=3.0,  # Low pressure
        deaerator_temperature_f=218.0,  # Subcooled
        inlet_water_temperature_f=140.0,
        inlet_dissolved_oxygen_ppb=8000.0,
        inlet_co2_ppm=15.0,
        outlet_dissolved_oxygen_ppb=20.0,  # Poor - exceeds limit
        outlet_co2_ppm=5.0,
        total_flow_lb_hr=55000.0,
        steam_flow_lb_hr=2000.0,  # Low steam
        vent_rate_lb_hr=10.0,  # Low vent
        deaerator_type="spray_tray",
        outlet_o2_limit_ppb=7.0,
    )


# =============================================================================
# COMPREHENSIVE INPUT FIXTURES
# =============================================================================

@pytest.fixture
def water_treatment_input_complete(
    boiler_water_input_excellent,
    feedwater_input_excellent,
    condensate_input_excellent,
    blowdown_input_optimize,
    chemical_dosing_input_standard,
    deaeration_input_excellent,
) -> WaterTreatmentInput:
    """Create complete water treatment input with all components."""
    return WaterTreatmentInput(
        system_id="TEST-WT-001",
        timestamp=datetime.now(timezone.utc),
        boiler_operating_pressure_psig=450.0,
        steam_flow_rate_lb_hr=50000.0,
        treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
        boiler_water=boiler_water_input_excellent,
        feedwater=feedwater_input_excellent,
        condensate=condensate_input_excellent,
        blowdown_data=blowdown_input_optimize,
        chemical_dosing_data=chemical_dosing_input_standard,
        deaerator_data=deaeration_input_excellent,
    )


@pytest.fixture
def water_treatment_input_minimal() -> WaterTreatmentInput:
    """Create minimal water treatment input."""
    return WaterTreatmentInput(
        system_id="TEST-WT-MIN",
        boiler_operating_pressure_psig=150.0,
        steam_flow_rate_lb_hr=25000.0,
        treatment_program=TreatmentProgram.PHOSPHATE_POLYMER,
    )


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

@pytest.fixture
def sample_emission_factor_db() -> Dict[str, float]:
    """Sample emission factors for testing."""
    return {
        ("natural_gas", "US", "boiler"): 53.06,  # kg CO2e per MMBTU
        ("fuel_oil", "US", "boiler"): 73.96,
        ("coal", "US", "boiler"): 95.52,
    }


@pytest.fixture
def chemistry_test_cases() -> list:
    """Parameterized test cases for chemistry calculations."""
    return [
        # (boiler_tds, feedwater_tds, expected_cycles)
        (2000.0, 50.0, 40.0),
        (1500.0, 100.0, 15.0),
        (2500.0, 250.0, 10.0),
        (3000.0, 300.0, 10.0),
        (1000.0, 200.0, 5.0),
    ]


@pytest.fixture
def blowdown_test_cases() -> list:
    """Parameterized test cases for blowdown calculations."""
    return [
        # (cycles, expected_blowdown_pct)
        (5.0, 20.0),
        (10.0, 10.0),
        (4.0, 25.0),
        (8.0, 12.5),
        (6.0, 16.67),
    ]


@pytest.fixture
def scavenger_test_cases() -> list:
    """Parameterized test cases for oxygen scavenger calculations."""
    return [
        # (do_ppb, scavenger_type, expected_dose_ppm)
        (5.0, ChemicalType.SULFITE, 0.06),  # 5 ppb * 7.88 * 1.5 / 1000
        (10.0, ChemicalType.SULFITE, 0.12),
        (7.0, ChemicalType.HYDRAZINE, 0.02),  # 7 ppb * 1.0 * 2.0 / 1000
    ]


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_historian():
    """Mock historian interface."""
    mock = MagicMock()
    mock.read_tag.return_value = 100.0
    mock.write_tag.return_value = True
    mock.query_history.return_value = []
    return mock


@pytest.fixture
def mock_opc_client():
    """Mock OPC-UA client."""
    mock = MagicMock()
    mock.connect.return_value = True
    mock.disconnect.return_value = True
    mock.read_node.return_value = 100.0
    mock.write_node.return_value = True
    return mock


@pytest.fixture
def mock_intelligence_mixin():
    """Mock intelligence mixin for testing."""
    mock = MagicMock()
    mock.explain_result = AsyncMock(return_value="Test explanation")
    mock.detect_anomalies = AsyncMock(return_value=[])
    mock.generate_recommendations = AsyncMock(return_value=["Test recommendation"])
    return mock


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for combined components"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "compliance: Regulatory compliance tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer than 1 second"
    )


@pytest.fixture(scope="session")
def benchmark_baseline():
    """Baseline performance metrics for benchmarking."""
    return {
        "boiler_water_analysis_ms": 5.0,
        "feedwater_analysis_ms": 3.0,
        "condensate_analysis_ms": 3.0,
        "blowdown_optimization_ms": 5.0,
        "chemical_dosing_optimization_ms": 5.0,
        "deaeration_analysis_ms": 3.0,
        "full_analysis_ms": 20.0,
    }
