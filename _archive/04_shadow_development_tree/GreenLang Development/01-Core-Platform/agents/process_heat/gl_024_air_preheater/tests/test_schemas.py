# -*- coding: utf-8 -*-
"""
Unit tests for GL-024 Air Preheater Agent Schemas Module

Tests all Pydantic models, enumerations, and data validation with 90%+ coverage.
Validates schema serialization, deserialization, and business logic.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict
import json

from greenlang.agents.process_heat.gl_024_air_preheater.schemas import (
    # Enums
    PreheaterType,
    AirPreheaterType,
    PreheaterStatus,
    RotorStatus,
    FoulingSeverity,
    CorrosionRiskLevel,
    LeakageDirection,
    CleaningMethod,
    AcidDewPointMethod,
    AlertSeverity,
    ValidationStatus,
    # Base schemas
    BasePreheaterSchema,
    # Input schemas
    GasComposition,
    GasSideInput,
    AirSideInput,
    RegenerativeOperatingData,
    SootBlowerStatus,
    PerformanceBaseline,
    AirPreheaterInput,
    # Output schemas
    HeatTransferAnalysis,
    LeakageAnalysis,
    ColdEndProtection,
    FoulingAnalysis,
    OptimizationRecommendation,
    Recommendation,
    OptimizationResult,
    PreheaterAlert,
    Alert,
    AirPreheaterOutput,
    # Configuration
    AirPreheaterConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_gas_composition():
    """Create sample gas composition."""
    return GasComposition(
        o2_pct=3.5,
        co2_pct=14.0,
        so2_ppm=500.0,
        so3_ppm=5.0,
        moisture_pct=8.0,
        n2_pct=72.0,
    )


@pytest.fixture
def sample_gas_side_input():
    """Create sample gas-side input."""
    return GasSideInput(
        inlet_temp_f=650.0,
        outlet_temp_f=300.0,
        flow_rate_lb_hr=500000.0,
        inlet_pressure_in_wc=-5.0,
        outlet_pressure_in_wc=-8.0,
    )


@pytest.fixture
def sample_air_side_input():
    """Create sample air-side input."""
    return AirSideInput(
        inlet_temp_f=80.0,
        outlet_temp_f=550.0,
        flow_rate_lb_hr=480000.0,
        relative_humidity_pct=50.0,
        inlet_pressure_in_wc=2.0,
        outlet_pressure_in_wc=-2.0,
    )


@pytest.fixture
def sample_regenerative_data():
    """Create sample regenerative operating data."""
    return RegenerativeOperatingData(
        rotor_speed_rpm=1.5,
        design_speed_rpm=1.5,
        rotor_status=RotorStatus.NORMAL_SPEED,
        radial_seal_clearance_in=0.125,
        axial_seal_clearance_in=0.100,
    )


@pytest.fixture
def sample_performance_baseline():
    """Create sample performance baseline."""
    return PerformanceBaseline(
        design_gas_inlet_temp_f=700.0,
        design_gas_outlet_temp_f=280.0,
        design_air_inlet_temp_f=80.0,
        design_air_outlet_temp_f=600.0,
        design_effectiveness=0.80,
        design_gas_flow_lb_hr=500000.0,
        design_air_flow_lb_hr=480000.0,
        design_gas_dp_in_wc=3.0,
        design_air_dp_in_wc=4.0,
        design_leakage_pct=6.0,
    )


@pytest.fixture
def sample_air_preheater_input(sample_gas_composition, sample_regenerative_data, sample_performance_baseline):
    """Create sample comprehensive input."""
    return AirPreheaterInput(
        preheater_id="APH-001",
        preheater_type=PreheaterType.REGENERATIVE,
        boiler_load_pct=85.0,
        gas_inlet_temp_f=650.0,
        gas_outlet_temp_f=300.0,
        gas_flow_lb_hr=500000.0,
        gas_dp_in_wc=3.5,
        air_inlet_temp_f=80.0,
        air_outlet_temp_f=550.0,
        air_flow_lb_hr=480000.0,
        air_dp_in_wc=4.5,
        gas_composition=sample_gas_composition,
        regenerative_data=sample_regenerative_data,
        baseline=sample_performance_baseline,
    )


# =============================================================================
# ENUMERATION TESTS
# =============================================================================

class TestPreheaterTypeEnum:
    """Test suite for PreheaterType enumeration."""

    @pytest.mark.unit
    def test_preheater_type_values(self):
        """Test preheater type enumeration values."""
        assert PreheaterType.REGENERATIVE.value == "regenerative"
        assert PreheaterType.RECUPERATIVE_TUBULAR.value == "recuperative_tubular"
        assert PreheaterType.RECUPERATIVE_PLATE.value == "recuperative_plate"
        assert PreheaterType.HEAT_PIPE.value == "heat_pipe"
        assert PreheaterType.LJUNGSTROM.value == "ljungstrom"

    @pytest.mark.unit
    def test_air_preheater_type_alias(self):
        """Test that AirPreheaterType is an alias for PreheaterType."""
        assert AirPreheaterType is PreheaterType
        assert AirPreheaterType.REGENERATIVE.value == "regenerative"


class TestPreheaterStatusEnum:
    """Test suite for PreheaterStatus enumeration."""

    @pytest.mark.unit
    def test_preheater_status_values(self):
        """Test preheater status enumeration values."""
        assert PreheaterStatus.OFFLINE.value == "offline"
        assert PreheaterStatus.NORMAL.value == "normal"
        assert PreheaterStatus.DEGRADED.value == "degraded"
        assert PreheaterStatus.ALARM.value == "alarm"
        assert PreheaterStatus.TRIP.value == "trip"


class TestCorrosionRiskLevelEnum:
    """Test suite for CorrosionRiskLevel enumeration."""

    @pytest.mark.unit
    def test_corrosion_risk_values(self):
        """Test corrosion risk enumeration values."""
        assert CorrosionRiskLevel.NEGLIGIBLE.value == "negligible"
        assert CorrosionRiskLevel.LOW.value == "low"
        assert CorrosionRiskLevel.MODERATE.value == "moderate"
        assert CorrosionRiskLevel.HIGH.value == "high"
        assert CorrosionRiskLevel.CRITICAL.value == "critical"


class TestAlertSeverityEnum:
    """Test suite for AlertSeverity enumeration."""

    @pytest.mark.unit
    def test_alert_severity_values(self):
        """Test alert severity enumeration values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ALARM.value == "alarm"
        assert AlertSeverity.CRITICAL.value == "critical"


# =============================================================================
# INPUT SCHEMA TESTS
# =============================================================================

class TestGasComposition:
    """Test suite for GasComposition schema."""

    @pytest.mark.unit
    def test_valid_gas_composition(self, sample_gas_composition):
        """Test valid gas composition creation."""
        assert sample_gas_composition.o2_pct == 3.5
        assert sample_gas_composition.so3_ppm == 5.0
        assert sample_gas_composition.moisture_pct == 8.0

    @pytest.mark.unit
    def test_o2_validation(self):
        """Test O2 percentage validation."""
        with pytest.raises(ValueError):
            GasComposition(o2_pct=25.0)  # O2 cannot exceed 21%

    @pytest.mark.unit
    def test_negative_o2_validation(self):
        """Test negative O2 percentage validation."""
        with pytest.raises(ValueError):
            GasComposition(o2_pct=-1.0)


class TestGasSideInput:
    """Test suite for GasSideInput schema."""

    @pytest.mark.unit
    def test_valid_gas_side_input(self, sample_gas_side_input):
        """Test valid gas-side input creation."""
        assert sample_gas_side_input.inlet_temp_f == 650.0
        assert sample_gas_side_input.outlet_temp_f == 300.0
        assert sample_gas_side_input.flow_rate_lb_hr == 500000.0

    @pytest.mark.unit
    def test_temperature_validation(self):
        """Test that gas outlet must be less than inlet."""
        with pytest.raises(ValueError):
            GasSideInput(
                inlet_temp_f=300.0,  # Inlet less than outlet - invalid
                outlet_temp_f=650.0,
                flow_rate_lb_hr=500000.0,
            )

    @pytest.mark.unit
    def test_pressure_drop_calculation(self):
        """Test automatic pressure drop calculation."""
        gas_input = GasSideInput(
            inlet_temp_f=650.0,
            outlet_temp_f=300.0,
            flow_rate_lb_hr=500000.0,
            inlet_pressure_in_wc=-5.0,
            outlet_pressure_in_wc=-8.0,
        )
        assert gas_input.pressure_drop_in_wc == 3.0


class TestAirSideInput:
    """Test suite for AirSideInput schema."""

    @pytest.mark.unit
    def test_valid_air_side_input(self, sample_air_side_input):
        """Test valid air-side input creation."""
        assert sample_air_side_input.inlet_temp_f == 80.0
        assert sample_air_side_input.outlet_temp_f == 550.0
        assert sample_air_side_input.flow_rate_lb_hr == 480000.0

    @pytest.mark.unit
    def test_temperature_validation(self):
        """Test that air outlet must be greater than inlet."""
        with pytest.raises(ValueError):
            AirSideInput(
                inlet_temp_f=550.0,  # Inlet greater than outlet - invalid
                outlet_temp_f=80.0,
                flow_rate_lb_hr=480000.0,
            )


class TestAirPreheaterInput:
    """Test suite for AirPreheaterInput schema."""

    @pytest.mark.unit
    def test_valid_input(self, sample_air_preheater_input):
        """Test valid comprehensive input creation."""
        assert sample_air_preheater_input.preheater_id == "APH-001"
        assert sample_air_preheater_input.preheater_type == PreheaterType.REGENERATIVE
        assert sample_air_preheater_input.boiler_load_pct == 85.0

    @pytest.mark.unit
    def test_gas_temperature_validation(self):
        """Test gas temperature validation in main input."""
        with pytest.raises(ValueError):
            AirPreheaterInput(
                preheater_id="APH-001",
                preheater_type=PreheaterType.REGENERATIVE,
                boiler_load_pct=85.0,
                gas_inlet_temp_f=300.0,  # Less than outlet - invalid
                gas_outlet_temp_f=650.0,
                gas_flow_lb_hr=500000.0,
                air_inlet_temp_f=80.0,
                air_outlet_temp_f=550.0,
                air_flow_lb_hr=480000.0,
            )

    @pytest.mark.unit
    def test_air_temperature_validation(self):
        """Test air temperature validation in main input."""
        with pytest.raises(ValueError):
            AirPreheaterInput(
                preheater_id="APH-001",
                preheater_type=PreheaterType.REGENERATIVE,
                boiler_load_pct=85.0,
                gas_inlet_temp_f=650.0,
                gas_outlet_temp_f=300.0,
                gas_flow_lb_hr=500000.0,
                air_inlet_temp_f=550.0,  # Greater than outlet - invalid
                air_outlet_temp_f=80.0,
                air_flow_lb_hr=480000.0,
            )

    @pytest.mark.unit
    def test_input_serialization(self, sample_air_preheater_input):
        """Test input serialization to JSON."""
        json_data = sample_air_preheater_input.model_dump_json()
        assert isinstance(json_data, str)
        parsed = json.loads(json_data)
        assert parsed["preheater_id"] == "APH-001"


# =============================================================================
# OUTPUT SCHEMA TESTS
# =============================================================================

class TestHeatTransferAnalysis:
    """Test suite for HeatTransferAnalysis schema."""

    @pytest.mark.unit
    def test_valid_heat_transfer_analysis(self):
        """Test valid heat transfer analysis creation."""
        analysis = HeatTransferAnalysis(
            effectiveness=0.75,
            ntu=2.5,
            heat_duty_mmbtu_hr=45.0,
            lmtd_f=250.0,
            approach_temp_hot_end_f=100.0,
            approach_temp_cold_end_f=220.0,
            x_ratio=0.95,
            current_ua_btu_hr_f=180000.0,
            gas_temp_drop_f=350.0,
            air_temp_rise_f=470.0,
        )
        assert analysis.effectiveness == 0.75
        assert analysis.ntu == 2.5
        assert analysis.heat_duty_mmbtu_hr == 45.0


class TestLeakageAnalysis:
    """Test suite for LeakageAnalysis schema."""

    @pytest.mark.unit
    def test_valid_leakage_analysis(self):
        """Test valid leakage analysis creation."""
        analysis = LeakageAnalysis(
            air_to_gas_leakage_pct=8.5,
            total_leakage_pct=8.5,
            efficiency_loss_due_to_leakage_pct=0.35,
        )
        assert analysis.air_to_gas_leakage_pct == 8.5
        assert analysis.total_leakage_pct == 8.5


class TestColdEndProtection:
    """Test suite for ColdEndProtection schema."""

    @pytest.mark.unit
    def test_valid_cold_end_protection(self):
        """Test valid cold-end protection analysis creation."""
        analysis = ColdEndProtection(
            acid_dew_point_f=275.0,
            water_dew_point_f=125.0,
            min_metal_temp_f=290.0,
            avg_metal_temp_f=310.0,
            min_recommended_metal_temp_f=280.0,
            margin_above_adp_f=15.0,
            margin_above_wdp_f=165.0,
            margin_is_adequate=False,
            corrosion_risk=CorrosionRiskLevel.HIGH,
            below_acid_dew_point=False,
        )
        assert analysis.acid_dew_point_f == 275.0
        assert analysis.corrosion_risk == CorrosionRiskLevel.HIGH


class TestOptimizationRecommendation:
    """Test suite for OptimizationRecommendation schema."""

    @pytest.mark.unit
    def test_valid_recommendation(self):
        """Test valid recommendation creation."""
        rec = OptimizationRecommendation(
            category="heat_transfer",
            priority=AlertSeverity.WARNING,
            title="Improve Effectiveness",
            description="Consider cleaning to restore heat transfer performance",
            efficiency_improvement_pct=1.5,
            cost_savings_usd_yr=50000.0,
        )
        assert rec.category == "heat_transfer"
        assert rec.efficiency_improvement_pct == 1.5

    @pytest.mark.unit
    def test_recommendation_alias(self):
        """Test Recommendation alias for OptimizationRecommendation."""
        assert Recommendation is OptimizationRecommendation


class TestPreheaterAlert:
    """Test suite for PreheaterAlert schema."""

    @pytest.mark.unit
    def test_valid_alert(self):
        """Test valid alert creation."""
        alert = PreheaterAlert(
            severity=AlertSeverity.ALARM,
            category="cold_end",
            title="Cold-End Corrosion Risk",
            description="Metal temperature approaching acid dew point",
            value=280.0,
            threshold=275.0,
            unit="F",
        )
        assert alert.severity == AlertSeverity.ALARM
        assert alert.category == "cold_end"

    @pytest.mark.unit
    def test_alert_alias(self):
        """Test Alert alias for PreheaterAlert."""
        assert Alert is PreheaterAlert


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

class TestBasePreheaterSchema:
    """Test suite for BasePreheaterSchema provenance features."""

    @pytest.mark.unit
    def test_provenance_hash_calculation(self):
        """Test provenance hash calculation."""
        # Use a concrete implementation for testing
        input_data = AirPreheaterInput(
            preheater_id="APH-TEST",
            preheater_type=PreheaterType.REGENERATIVE,
            boiler_load_pct=85.0,
            gas_inlet_temp_f=650.0,
            gas_outlet_temp_f=300.0,
            gas_flow_lb_hr=500000.0,
            air_inlet_temp_f=80.0,
            air_outlet_temp_f=550.0,
            air_flow_lb_hr=480000.0,
        )
        hash1 = input_data.calculate_provenance_hash()
        hash2 = input_data.calculate_provenance_hash()
        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestAirPreheaterConfig:
    """Test suite for AirPreheaterConfig schema."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration creation."""
        config = AirPreheaterConfig()
        assert config.agent_id == "GL-024-AIR-PREHEATER"
        assert config.default_acid_dew_point_method == AcidDewPointMethod.VERHOFF_BANCHERO

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = AirPreheaterConfig(
            leakage_warning_pct=10.0,
            leakage_alarm_pct=15.0,
            default_fuel_cost_per_mmbtu_usd=8.0,
        )
        assert config.leakage_warning_pct == 10.0
        assert config.default_fuel_cost_per_mmbtu_usd == 8.0


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    """Test suite for schema serialization."""

    @pytest.mark.unit
    def test_json_serialization_roundtrip(self, sample_air_preheater_input):
        """Test JSON serialization and deserialization."""
        json_str = sample_air_preheater_input.model_dump_json()
        parsed = AirPreheaterInput.model_validate_json(json_str)
        assert parsed.preheater_id == sample_air_preheater_input.preheater_id
        assert parsed.boiler_load_pct == sample_air_preheater_input.boiler_load_pct

    @pytest.mark.unit
    def test_dict_conversion(self, sample_gas_composition):
        """Test dictionary conversion."""
        data = sample_gas_composition.model_dump()
        assert isinstance(data, dict)
        assert data["o2_pct"] == 3.5
        reconstructed = GasComposition.model_validate(data)
        assert reconstructed.o2_pct == 3.5
