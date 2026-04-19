"""
GL-003 Unified Steam System Optimizer - Test Fixtures

Shared pytest fixtures for GL-003 test suite.
Provides reusable test data, configurations, and mock objects.
"""

import pytest
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

# Import configurations
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    UnifiedSteamConfig,
    SteamHeaderConfig,
    SteamHeaderLevel,
    PRVConfig,
    PRVSizingMethod,
    DesuperheaterConfig,
    DesuperheaterType,
    QualityMonitoringConfig,
    SteamQualityStandard,
    CondensateConfig,
    CondensateFlashMethod,
    FlashRecoveryConfig,
    SteamTrapSurveyConfig,
    ExergyOptimizationConfig,
    create_default_config,
)

from schemas import (
    SteamPhase,
    ValidationStatus,
    OptimizationStatus,
    TrapStatus,
    SteamProperties,
    SteamFlowMeasurement,
    HeaderReading,
    HeaderBalanceInput,
    HeaderBalanceOutput,
    SteamQualityReading,
    SteamQualityAnalysis,
    PRVOperatingPoint,
    PRVSizingInput,
    PRVSizingOutput,
    CondensateReading,
    CondensateReturnAnalysis,
    FlashSteamInput,
    FlashSteamOutput,
    SteamTrapReading,
    TrapSurveyAnalysis,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_config() -> UnifiedSteamConfig:
    """Create default steam system configuration."""
    return create_default_config()


@pytest.fixture
def hp_header_config() -> SteamHeaderConfig:
    """Create high-pressure header configuration."""
    return SteamHeaderConfig(
        name="HP-MAIN",
        level=SteamHeaderLevel.HIGH_PRESSURE,
        design_pressure_psig=600.0,
        min_pressure_psig=580.0,
        max_pressure_psig=620.0,
        design_flow_lb_hr=100000.0,
        max_flow_lb_hr=120000.0,
        design_temperature_f=750.0,
        connected_consumers=["TURB-001", "TURB-002"],
        connected_suppliers=["BLR-001", "BLR-002"],
    )


@pytest.fixture
def mp_header_config() -> SteamHeaderConfig:
    """Create medium-pressure header configuration."""
    return SteamHeaderConfig(
        name="MP-MAIN",
        level=SteamHeaderLevel.MEDIUM_PRESSURE,
        design_pressure_psig=150.0,
        min_pressure_psig=140.0,
        max_pressure_psig=160.0,
        design_flow_lb_hr=50000.0,
        max_flow_lb_hr=65000.0,
    )


@pytest.fixture
def lp_header_config() -> SteamHeaderConfig:
    """Create low-pressure header configuration."""
    return SteamHeaderConfig(
        name="LP-MAIN",
        level=SteamHeaderLevel.LOW_PRESSURE,
        design_pressure_psig=15.0,
        min_pressure_psig=10.0,
        max_pressure_psig=20.0,
        design_flow_lb_hr=25000.0,
        max_flow_lb_hr=35000.0,
    )


@pytest.fixture
def prv_config() -> PRVConfig:
    """Create PRV configuration."""
    return PRVConfig(
        prv_id="PRV-HP-MP",
        sizing_method=PRVSizingMethod.ASME_B31_1,
        inlet_pressure_psig=600.0,
        outlet_pressure_psig=150.0,
        min_pressure_drop_psi=10.0,
        design_flow_lb_hr=30000.0,
        min_flow_lb_hr=5000.0,
        max_flow_lb_hr=40000.0,
        cv_rated=150.0,
        rangeability=50.0,
        target_opening_min_pct=50.0,
        target_opening_max_pct=70.0,
        desuperheater_enabled=True,
        desuperheater_type=DesuperheaterType.WATER_SPRAY,
        target_superheat_f=50.0,
    )


@pytest.fixture
def quality_config() -> QualityMonitoringConfig:
    """Create quality monitoring configuration."""
    return QualityMonitoringConfig(
        standard=SteamQualityStandard.ASME,
        min_dryness_fraction=0.95,
        target_dryness_fraction=0.98,
        max_tds_ppm_lp=3500.0,
        max_tds_ppm_mp=3000.0,
        max_tds_ppm_hp=2500.0,
        max_cation_conductivity_us_cm=0.3,
        max_silica_ppm=0.02,
        warning_threshold_pct=80.0,
        critical_threshold_pct=95.0,
    )


@pytest.fixture
def condensate_config() -> CondensateConfig:
    """Create condensate configuration."""
    return CondensateConfig(
        target_return_rate_pct=85.0,
        min_acceptable_return_pct=70.0,
        target_return_temp_f=180.0,
        min_return_temp_f=140.0,
        max_contamination_tds_ppm=50.0,
        max_oil_ppm=1.0,
        max_iron_ppb=100.0,
        trap_survey_enabled=True,
        flash_recovery_enabled=True,
    )


@pytest.fixture
def flash_recovery_config() -> FlashRecoveryConfig:
    """Create flash recovery configuration."""
    return FlashRecoveryConfig(
        flash_tank_id="FT-HP",
        condensate_pressure_psig=150.0,
        flash_pressure_psig=15.0,
        condensate_flow_lb_hr=5000.0,
        min_recovery_efficiency_pct=90.0,
        flash_steam_destination="LP-MAIN",
        fuel_cost_per_mmbtu=5.0,
        operating_hours_per_year=8000,
    )


@pytest.fixture
def exergy_config() -> ExergyOptimizationConfig:
    """Create exergy optimization configuration."""
    return ExergyOptimizationConfig(
        enabled=True,
        reference_temperature_f=77.0,
        reference_pressure_psia=14.696,
        exergy_weight=0.6,
        cost_weight=0.3,
        reliability_weight=0.1,
        min_exergy_efficiency_pct=40.0,
    )


@pytest.fixture
def trap_survey_config() -> SteamTrapSurveyConfig:
    """Create steam trap survey configuration."""
    return SteamTrapSurveyConfig(
        survey_enabled=True,
        survey_frequency_days=90,
        failed_open_threshold_pct=5.0,
        failed_closed_threshold_pct=10.0,
        steam_cost_per_mlb=10.0,
    )


# =============================================================================
# DATA MODEL FIXTURES
# =============================================================================

@pytest.fixture
def steam_properties_saturated() -> SteamProperties:
    """Create saturated steam properties at 150 psig."""
    return SteamProperties(
        pressure_psig=150.0,
        temperature_f=365.9,
        saturation_temperature_f=365.9,
        phase=SteamPhase.SATURATED_VAPOR,
        dryness_fraction=1.0,
        enthalpy_btu_lb=1196.0,
        entropy_btu_lb_r=1.6375,
    )


@pytest.fixture
def steam_properties_superheated() -> SteamProperties:
    """Create superheated steam properties at 600 psig."""
    return SteamProperties(
        pressure_psig=600.0,
        temperature_f=750.0,
        saturation_temperature_f=489.0,
        phase=SteamPhase.SUPERHEATED_VAPOR,
        dryness_fraction=1.0,
        superheat_f=261.0,
        enthalpy_btu_lb=1315.0,
        entropy_btu_lb_r=1.58,
    )


@pytest.fixture
def header_balance_input() -> HeaderBalanceInput:
    """Create header balance input data."""
    return HeaderBalanceInput(
        header_id="HP-MAIN",
        timestamp=datetime.now(timezone.utc),
        current_pressure_psig=598.0,
        current_temperature_f=745.0,
        pressure_setpoint_psig=600.0,
        temperature_setpoint_f=750.0,
        supplies=[
            {"id": "BLR-001", "flow_lb_hr": 60000, "controllable": True},
            {"id": "BLR-002", "flow_lb_hr": 40000, "controllable": True},
        ],
        demands=[
            {"id": "TURB-001", "flow_lb_hr": 50000},
            {"id": "TURB-002", "flow_lb_hr": 45000},
        ],
        pressure_deadband_psi=2.0,
    )


@pytest.fixture
def steam_quality_reading_good() -> SteamQualityReading:
    """Create good quality steam reading."""
    return SteamQualityReading(
        reading_id="QR-001",
        location_id="BLR-001-STEAM",
        timestamp=datetime.now(timezone.utc),
        pressure_psig=150.0,
        temperature_f=366.0,
        dryness_fraction=0.98,
        tds_ppm=10.0,
        cation_conductivity_us_cm=0.15,
        silica_ppm=0.01,
        dissolved_o2_ppb=5.0,
    )


@pytest.fixture
def steam_quality_reading_poor() -> SteamQualityReading:
    """Create poor quality steam reading."""
    return SteamQualityReading(
        reading_id="QR-002",
        location_id="BLR-002-STEAM",
        timestamp=datetime.now(timezone.utc),
        pressure_psig=150.0,
        temperature_f=364.0,
        dryness_fraction=0.92,
        tds_ppm=50.0,
        cation_conductivity_us_cm=0.4,
        silica_ppm=0.03,
        dissolved_o2_ppb=10.0,
    )


@pytest.fixture
def prv_operating_point() -> PRVOperatingPoint:
    """Create PRV operating point data."""
    return PRVOperatingPoint(
        prv_id="PRV-HP-MP",
        timestamp=datetime.now(timezone.utc),
        inlet_pressure_psig=600.0,
        outlet_pressure_psig=150.0,
        flow_rate_lb_hr=30000.0,
        opening_pct=60.0,
        inlet_temperature_f=750.0,
        outlet_temperature_f=400.0,
    )


@pytest.fixture
def condensate_reading() -> CondensateReading:
    """Create condensate reading."""
    return CondensateReading(
        location_id="COND-RETURN-001",
        timestamp=datetime.now(timezone.utc),
        flow_rate_lb_hr=40000.0,
        temperature_f=185.0,
        pressure_psig=5.0,
        tds_ppm=30.0,
        oil_ppm=0.1,
        iron_ppb=40.0,
        ph=9.0,
        is_contaminated=False,
    )


@pytest.fixture
def flash_steam_input() -> FlashSteamInput:
    """Create flash steam calculation input."""
    return FlashSteamInput(
        condensate_flow_lb_hr=5000.0,
        condensate_pressure_psig=150.0,
        flash_pressure_psig=15.0,
    )


@pytest.fixture
def steam_trap_reading_good() -> SteamTrapReading:
    """Create good steam trap reading."""
    return SteamTrapReading(
        trap_id="TRAP-001",
        location="Heat Exchanger HX-001",
        timestamp=datetime.now(timezone.utc),
        trap_type="thermodynamic",
        size_inches=0.75,
        design_capacity_lb_hr=500.0,
        inlet_pressure_psig=150.0,
        differential_pressure_psi=145.0,
        status=TrapStatus.OPERATING,
        temperature_f=365.0,
    )


@pytest.fixture
def steam_trap_reading_failed() -> SteamTrapReading:
    """Create failed steam trap reading."""
    return SteamTrapReading(
        trap_id="TRAP-002",
        location="Heat Exchanger HX-002",
        timestamp=datetime.now(timezone.utc),
        trap_type="thermodynamic",
        size_inches=0.75,
        design_capacity_lb_hr=500.0,
        inlet_pressure_psig=150.0,
        differential_pressure_psi=10.0,
        status=TrapStatus.FAILED_OPEN,
        temperature_f=350.0,
        steam_loss_lb_hr=50.0,
    )


# =============================================================================
# DATA GENERATOR FIXTURES
# =============================================================================

@pytest.fixture
def generate_trap_readings():
    """Factory fixture for generating trap readings."""
    def _generate(count: int, failure_rate: float = 0.1) -> List[SteamTrapReading]:
        import random
        readings = []
        for i in range(count):
            if random.random() < failure_rate:
                status = random.choice([TrapStatus.FAILED_OPEN, TrapStatus.FAILED_CLOSED])
                steam_loss = 50.0 if status == TrapStatus.FAILED_OPEN else 0.0
            else:
                status = TrapStatus.OPERATING
                steam_loss = 0.0

            readings.append(SteamTrapReading(
                trap_id=f"TRAP-{i:03d}",
                location=f"Location-{i}",
                timestamp=datetime.now(timezone.utc),
                trap_type="thermodynamic",
                size_inches=0.75,
                inlet_pressure_psig=150.0,
                status=status,
                temperature_f=365.0,
                steam_loss_lb_hr=steam_loss,
            ))
        return readings
    return _generate


@pytest.fixture
def generate_condensate_readings():
    """Factory fixture for generating condensate readings."""
    def _generate(count: int) -> List[CondensateReading]:
        readings = []
        for i in range(count):
            readings.append(CondensateReading(
                location_id=f"COND-{i:03d}",
                timestamp=datetime.now(timezone.utc),
                flow_rate_lb_hr=5000.0 + i * 1000,
                temperature_f=180.0 + i * 2,
                pressure_psig=5.0,
                tds_ppm=20.0 + i,
                ph=9.0,
            ))
        return readings
    return _generate


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_intelligence_mixin():
    """Mock for IntelligenceMixin."""
    mock = MagicMock()
    mock.generate_explanation.return_value = "Test explanation"
    mock.get_intelligence_level.return_value = "ADVANCED"
    return mock


# =============================================================================
# IAPWS-IF97 REFERENCE DATA FIXTURES
# =============================================================================

@pytest.fixture
def iapws_saturation_reference():
    """Reference saturation data for validation (IAPWS-IF97)."""
    return {
        # psig: (T_sat_F, h_f, h_fg, h_g)
        0: (212.0, 180.2, 970.3, 1150.5),
        15: (250.3, 218.9, 945.4, 1164.3),
        50: (298.0, 267.6, 911.0, 1178.6),
        100: (337.9, 309.0, 879.5, 1188.5),
        150: (365.9, 339.2, 856.8, 1196.0),
        200: (387.9, 362.2, 837.4, 1199.6),
        300: (421.7, 397.0, 804.3, 1201.3),
        400: (448.0, 424.2, 774.4, 1198.6),
        500: (470.0, 447.7, 747.1, 1194.8),
        600: (489.0, 468.4, 721.4, 1189.8),
    }


@pytest.fixture
def asme_prv_opening_targets():
    """ASME B31.1 PRV opening percentage targets."""
    return {
        "minimum": 50.0,
        "maximum": 70.0,
        "optimal": 60.0,
        "critical_high": 90.0,
        "critical_low": 20.0,
    }


@pytest.fixture
def asme_quality_limits():
    """ASME quality limits for validation."""
    return {
        "min_dryness_fraction": 0.95,
        "max_tds_lp": 3500.0,
        "max_tds_mp": 3000.0,
        "max_tds_hp": 2500.0,
        "max_cation_conductivity": 0.3,
        "max_silica": 0.02,
        "max_dissolved_o2": 7.0,
    }
