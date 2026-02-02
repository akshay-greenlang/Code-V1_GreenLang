"""
Unit tests for GL-020 ECONOPULSE Schema Definitions

Tests all Pydantic models for input, output, and result schemas.
Target coverage: 85%+

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
"""

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from ..schemas import (
    # Enums
    EconomizerStatus,
    FoulingType,
    FoulingSeverity,
    AlertSeverity,
    CleaningStatus,
    SootBlowingStatus,
    # Input schemas
    EconomizerInput,
    SootBlowerInput,
    # Output schemas
    GasSideFoulingResult,
    WaterSideFoulingResult,
    SootBlowerResult,
    AcidDewPointResult,
    EffectivenessResult,
    SteamingResult,
    OptimizationRecommendation,
    Alert,
    EconomizerOutput,
)


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEconomizerStatus:
    """Test EconomizerStatus enum."""

    def test_all_statuses_exist(self):
        """Test all economizer statuses are defined."""
        assert EconomizerStatus.OFFLINE.value == "offline"
        assert EconomizerStatus.STANDBY.value == "standby"
        assert EconomizerStatus.WARMING.value == "warming"
        assert EconomizerStatus.NORMAL.value == "normal"
        assert EconomizerStatus.DEGRADED.value == "degraded"
        assert EconomizerStatus.STEAMING_RISK.value == "steaming_risk"
        assert EconomizerStatus.ALARM.value == "alarm"
        assert EconomizerStatus.TRIP.value == "trip"

    def test_enum_count(self):
        """Test correct number of statuses."""
        assert len(EconomizerStatus) == 8


class TestFoulingType:
    """Test FoulingType enum."""

    def test_all_types_exist(self):
        """Test all fouling types are defined."""
        assert FoulingType.GAS_SIDE.value == "gas_side"
        assert FoulingType.WATER_SIDE.value == "water_side"
        assert FoulingType.COMBINED.value == "combined"
        assert FoulingType.NONE.value == "none"


class TestFoulingSeverity:
    """Test FoulingSeverity enum."""

    def test_all_severities_exist(self):
        """Test all fouling severities are defined."""
        assert FoulingSeverity.NONE.value == "none"
        assert FoulingSeverity.LIGHT.value == "light"
        assert FoulingSeverity.MODERATE.value == "moderate"
        assert FoulingSeverity.SEVERE.value == "severe"
        assert FoulingSeverity.CRITICAL.value == "critical"


class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_all_severities_exist(self):
        """Test all alert severities are defined."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ALARM.value == "alarm"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestCleaningStatus:
    """Test CleaningStatus enum."""

    def test_all_statuses_exist(self):
        """Test all cleaning statuses are defined."""
        assert CleaningStatus.NOT_REQUIRED.value == "not_required"
        assert CleaningStatus.MONITOR.value == "monitor"
        assert CleaningStatus.RECOMMENDED.value == "recommended"
        assert CleaningStatus.REQUIRED.value == "required"
        assert CleaningStatus.URGENT.value == "urgent"


class TestSootBlowingStatus:
    """Test SootBlowingStatus enum."""

    def test_all_statuses_exist(self):
        """Test all soot blowing statuses are defined."""
        assert SootBlowingStatus.IDLE.value == "idle"
        assert SootBlowingStatus.IN_PROGRESS.value == "in_progress"
        assert SootBlowingStatus.SCHEDULED.value == "scheduled"
        assert SootBlowingStatus.COMPLETED.value == "completed"
        assert SootBlowingStatus.BYPASSED.value == "bypassed"


# =============================================================================
# INPUT SCHEMA TESTS
# =============================================================================

class TestEconomizerInput:
    """Test EconomizerInput schema."""

    @pytest.fixture
    def valid_input_data(self):
        """Provide valid input data fixture."""
        return {
            "economizer_id": "ECO-001",
            "load_pct": 75.0,
            "gas_inlet_temp_f": 600.0,
            "gas_inlet_flow_lb_hr": 100000.0,
            "gas_outlet_temp_f": 350.0,
            "water_inlet_temp_f": 250.0,
            "water_inlet_flow_lb_hr": 80000.0,
            "water_inlet_pressure_psig": 550.0,
            "water_outlet_temp_f": 330.0,
            "water_outlet_pressure_psig": 540.0,
        }

    def test_valid_input(self, valid_input_data):
        """Test valid input creates successfully."""
        input_schema = EconomizerInput(**valid_input_data)

        assert input_schema.economizer_id == "ECO-001"
        assert input_schema.load_pct == 75.0
        assert input_schema.gas_inlet_temp_f == 600.0
        assert input_schema.gas_outlet_temp_f == 350.0
        assert input_schema.water_inlet_temp_f == 250.0
        assert input_schema.water_outlet_temp_f == 330.0

    def test_default_values(self, valid_input_data):
        """Test default values are applied."""
        input_schema = EconomizerInput(**valid_input_data)

        assert input_schema.operating_status == EconomizerStatus.NORMAL
        assert input_schema.gas_inlet_pressure_in_wc == 0.0
        assert input_schema.gas_outlet_pressure_in_wc == 0.0
        assert input_schema.flue_gas_o2_pct == 3.0
        assert input_schema.flue_gas_moisture_pct == 10.0
        assert input_schema.drum_pressure_psig == 500.0
        assert input_schema.ambient_temp_f == 70.0
        assert input_schema.barometric_pressure_inhg == 29.92
        assert input_schema.soot_blower_active is False

    def test_timestamp_auto_generated(self, valid_input_data):
        """Test timestamp is auto-generated."""
        input_schema = EconomizerInput(**valid_input_data)

        assert input_schema.timestamp is not None
        assert isinstance(input_schema.timestamp, datetime)

    def test_gas_dp_calculation(self, valid_input_data):
        """Test gas-side DP auto-calculation."""
        valid_input_data["gas_inlet_pressure_in_wc"] = 5.0
        valid_input_data["gas_outlet_pressure_in_wc"] = 3.0

        input_schema = EconomizerInput(**valid_input_data)

        assert input_schema.gas_side_dp_in_wc == 2.0

    def test_water_dp_calculation(self, valid_input_data):
        """Test water-side DP auto-calculation."""
        input_schema = EconomizerInput(**valid_input_data)

        expected_dp = abs(550.0 - 540.0)
        assert input_schema.water_side_dp_psi == expected_dp

    def test_load_bounds_validation(self, valid_input_data):
        """Test load percentage bounds validation."""
        # Valid range
        valid_input_data["load_pct"] = 100.0
        input_schema = EconomizerInput(**valid_input_data)
        assert input_schema.load_pct == 100.0

        # Above maximum (allows 120% for overload conditions)
        valid_input_data["load_pct"] = 115.0
        input_schema = EconomizerInput(**valid_input_data)
        assert input_schema.load_pct == 115.0

        # Too high
        valid_input_data["load_pct"] = 125.0
        with pytest.raises(ValidationError):
            EconomizerInput(**valid_input_data)

        # Negative
        valid_input_data["load_pct"] = -5.0
        with pytest.raises(ValidationError):
            EconomizerInput(**valid_input_data)

    def test_temperature_bounds_validation(self, valid_input_data):
        """Test temperature bounds validation."""
        # Gas inlet temp too low
        valid_input_data["gas_inlet_temp_f"] = 150.0
        with pytest.raises(ValidationError):
            EconomizerInput(**valid_input_data)

        # Gas inlet temp too high
        valid_input_data["gas_inlet_temp_f"] = 1500.0
        with pytest.raises(ValidationError):
            EconomizerInput(**valid_input_data)

    def test_optional_fields(self, valid_input_data):
        """Test optional fields can be None."""
        input_schema = EconomizerInput(**valid_input_data)

        assert input_schema.flue_gas_co2_pct is None
        assert input_schema.flue_gas_so2_ppm is None
        assert input_schema.saturation_temp_f is None
        assert input_schema.cold_end_metal_temp_f is None
        assert input_schema.feedwater_ph is None

    def test_optional_field_values(self, valid_input_data):
        """Test optional fields accept valid values."""
        valid_input_data["flue_gas_co2_pct"] = 12.0
        valid_input_data["flue_gas_so2_ppm"] = 200.0
        valid_input_data["feedwater_ph"] = 9.2

        input_schema = EconomizerInput(**valid_input_data)

        assert input_schema.flue_gas_co2_pct == 12.0
        assert input_schema.flue_gas_so2_ppm == 200.0
        assert input_schema.feedwater_ph == 9.2


class TestSootBlowerInput:
    """Test SootBlowerInput schema."""

    @pytest.fixture
    def valid_soot_blower_input(self):
        """Provide valid soot blower input fixture."""
        return {
            "timestamp": datetime.now(timezone.utc),
            "gas_side_dp_ratio": 1.2,
            "effectiveness_ratio": 0.92,
            "boiler_load_pct": 80.0,
        }

    def test_valid_input(self, valid_soot_blower_input):
        """Test valid soot blower input."""
        input_schema = SootBlowerInput(**valid_soot_blower_input)

        assert input_schema.gas_side_dp_ratio == 1.2
        assert input_schema.effectiveness_ratio == 0.92
        assert input_schema.boiler_load_pct == 80.0

    def test_default_values(self, valid_soot_blower_input):
        """Test default values."""
        input_schema = SootBlowerInput(**valid_soot_blower_input)

        assert input_schema.gas_outlet_temp_deviation_f == 0.0
        assert input_schema.steam_available is True
        assert input_schema.hours_since_last_blow == 0.0

    def test_dp_ratio_bounds(self, valid_soot_blower_input):
        """Test DP ratio bounds validation."""
        # Too low
        valid_soot_blower_input["gas_side_dp_ratio"] = 0.4
        with pytest.raises(ValidationError):
            SootBlowerInput(**valid_soot_blower_input)

        # Too high
        valid_soot_blower_input["gas_side_dp_ratio"] = 6.0
        with pytest.raises(ValidationError):
            SootBlowerInput(**valid_soot_blower_input)


# =============================================================================
# OUTPUT SCHEMA TESTS
# =============================================================================

class TestGasSideFoulingResult:
    """Test GasSideFoulingResult schema."""

    @pytest.fixture
    def valid_gas_side_result(self):
        """Provide valid gas-side fouling result fixture."""
        return {
            "fouling_detected": True,
            "fouling_severity": FoulingSeverity.MODERATE,
            "fouling_trend": "degrading",
            "current_dp_in_wc": 2.5,
            "design_dp_in_wc": 2.0,
            "corrected_dp_in_wc": 2.6,
            "dp_ratio": 1.3,
            "dp_deviation_pct": 30.0,
            "u_actual_btu_hr_ft2_f": 8.5,
            "u_clean_btu_hr_ft2_f": 10.0,
            "u_degradation_pct": 15.0,
            "fouling_resistance_hr_ft2_f_btu": 0.0015,
            "cleaning_status": CleaningStatus.RECOMMENDED,
        }

    def test_valid_result(self, valid_gas_side_result):
        """Test valid gas-side fouling result."""
        result = GasSideFoulingResult(**valid_gas_side_result)

        assert result.fouling_detected is True
        assert result.fouling_severity == FoulingSeverity.MODERATE
        assert result.dp_ratio == 1.3
        assert result.u_degradation_pct == 15.0

    def test_default_values(self, valid_gas_side_result):
        """Test default values in result."""
        result = GasSideFoulingResult(**valid_gas_side_result)

        assert result.estimated_fouling_thickness_in is None
        assert result.efficiency_loss_pct == 0.0
        assert result.fuel_waste_pct == 0.0
        assert result.soot_blow_recommended is False
        assert result.estimated_hours_to_cleaning is None
        assert result.calculation_method == "ASME_PTC_4.3"


class TestAcidDewPointResult:
    """Test AcidDewPointResult schema."""

    @pytest.fixture
    def valid_acid_dew_point_result(self):
        """Provide valid acid dew point result fixture."""
        return {
            "sulfuric_acid_dew_point_f": 275.0,
            "water_dew_point_f": 125.0,
            "effective_dew_point_f": 275.0,
            "min_metal_temp_f": 310.0,
            "avg_metal_temp_f": 320.0,
            "margin_above_dew_point_f": 35.0,
            "so3_concentration_ppm": 5.0,
            "h2o_concentration_pct": 10.0,
            "excess_air_pct": 16.7,
            "min_recommended_metal_temp_f": 305.0,
        }

    def test_valid_result(self, valid_acid_dew_point_result):
        """Test valid acid dew point result."""
        result = AcidDewPointResult(**valid_acid_dew_point_result)

        assert result.sulfuric_acid_dew_point_f == 275.0
        assert result.water_dew_point_f == 125.0
        assert result.margin_above_dew_point_f == 35.0

    def test_default_values(self, valid_acid_dew_point_result):
        """Test default values in result."""
        result = AcidDewPointResult(**valid_acid_dew_point_result)

        assert result.corrosion_risk == "low"
        assert result.below_dew_point is False
        assert result.margin_adequate is True
        assert result.action_required is False
        assert result.calculation_method == "VERHOFF_BANCHERO"
        assert "Verhoff & Banchero" in result.formula_reference


class TestEffectivenessResult:
    """Test EffectivenessResult schema."""

    @pytest.fixture
    def valid_effectiveness_result(self):
        """Provide valid effectiveness result fixture."""
        return {
            "current_effectiveness": 0.75,
            "design_effectiveness": 0.80,
            "effectiveness_ratio": 0.9375,
            "effectiveness_deviation_pct": 6.25,
            "current_ntu": 1.8,
            "design_ntu": 2.0,
            "current_ua_btu_hr_f": 90000.0,
            "design_ua_btu_hr_f": 100000.0,
            "clean_ua_btu_hr_f": 120000.0,
            "ua_degradation_pct": 25.0,
            "actual_duty_btu_hr": 18000000.0,
            "expected_duty_btu_hr": 20000000.0,
            "duty_deficit_btu_hr": 2000000.0,
            "lmtd_f": 150.0,
            "approach_temp_f": 270.0,
            "gas_temp_drop_f": 250.0,
            "water_temp_rise_f": 80.0,
            "c_min_btu_hr_f": 26000.0,
            "c_max_btu_hr_f": 80000.0,
            "capacity_ratio": 0.325,
        }

    def test_valid_result(self, valid_effectiveness_result):
        """Test valid effectiveness result."""
        result = EffectivenessResult(**valid_effectiveness_result)

        assert result.current_effectiveness == 0.75
        assert result.design_effectiveness == 0.80
        assert result.effectiveness_ratio == 0.9375

    def test_default_values(self, valid_effectiveness_result):
        """Test default values."""
        result = EffectivenessResult(**valid_effectiveness_result)

        assert result.performance_status == "normal"
        assert result.primary_degradation_source == "none"
        assert result.calculation_method == "NTU_EPSILON"


class TestSteamingResult:
    """Test SteamingResult schema."""

    @pytest.fixture
    def valid_steaming_result(self):
        """Provide valid steaming result fixture."""
        return {
            "approach_temp_f": 25.0,
            "design_approach_f": 30.0,
            "approach_margin_f": -5.0,
            "water_outlet_temp_f": 445.0,
            "saturation_temp_f": 470.0,
            "subcooling_f": 25.0,
            "current_load_pct": 75.0,
            "water_flow_pct": 80.0,
            "min_safe_load_pct": 30.0,
            "current_min_load_margin_pct": 45.0,
        }

    def test_valid_result(self, valid_steaming_result):
        """Test valid steaming result."""
        result = SteamingResult(**valid_steaming_result)

        assert result.approach_temp_f == 25.0
        assert result.saturation_temp_f == 470.0
        assert result.subcooling_f == 25.0

    def test_default_values(self, valid_steaming_result):
        """Test default values."""
        result = SteamingResult(**valid_steaming_result)

        assert result.steaming_detected is False
        assert result.steaming_risk == "low"
        assert result.steaming_risk_score == 0.0
        assert result.low_load_risk is False
        assert result.dp_fluctuation_detected is False
        assert result.temp_fluctuation_detected is False
        assert result.action_required is False


class TestAlert:
    """Test Alert schema."""

    def test_valid_alert(self):
        """Test valid alert creation."""
        alert = Alert(
            severity=AlertSeverity.ALARM,
            category="gas_fouling",
            title="High Gas-Side Pressure Drop",
            description="Pressure drop ratio exceeds alarm threshold.",
            value=1.6,
            threshold=1.5,
            unit="ratio",
        )

        assert alert.severity == AlertSeverity.ALARM
        assert alert.category == "gas_fouling"
        assert alert.title == "High Gas-Side Pressure Drop"
        assert alert.value == 1.6
        assert alert.threshold == 1.5

    def test_alert_id_auto_generated(self):
        """Test alert ID is auto-generated."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            category="test",
            title="Test Alert",
            description="Test description",
        )

        assert alert.alert_id is not None
        assert len(alert.alert_id) == 8

    def test_alert_timestamp_auto_generated(self):
        """Test alert timestamp is auto-generated."""
        alert = Alert(
            severity=AlertSeverity.INFO,
            category="test",
            title="Test",
            description="Test",
        )

        assert alert.timestamp is not None
        assert isinstance(alert.timestamp, datetime)


class TestOptimizationRecommendation:
    """Test OptimizationRecommendation schema."""

    def test_valid_recommendation(self):
        """Test valid recommendation creation."""
        rec = OptimizationRecommendation(
            category="soot_blowing",
            priority=AlertSeverity.WARNING,
            title="Initiate Soot Blowing",
            description="Gas-side fouling detected.",
            estimated_efficiency_gain_pct=0.5,
        )

        assert rec.category == "soot_blowing"
        assert rec.priority == AlertSeverity.WARNING
        assert rec.title == "Initiate Soot Blowing"
        assert rec.estimated_efficiency_gain_pct == 0.5

    def test_recommendation_id_auto_generated(self):
        """Test recommendation ID is auto-generated."""
        rec = OptimizationRecommendation(
            category="test",
            title="Test",
            description="Test",
        )

        assert rec.recommendation_id is not None
        assert len(rec.recommendation_id) == 8

    def test_default_values(self):
        """Test default values."""
        rec = OptimizationRecommendation(
            category="test",
            title="Test",
            description="Test",
        )

        assert rec.priority == AlertSeverity.INFO
        assert rec.current_value is None
        assert rec.target_value is None
        assert rec.implementation_difficulty == "low"
        assert rec.requires_outage is False


# =============================================================================
# COMPLETE OUTPUT SCHEMA TESTS
# =============================================================================

class TestEconomizerOutput:
    """Test EconomizerOutput schema."""

    @pytest.fixture
    def complete_output_components(self):
        """Provide complete output component fixtures."""
        gas_side = GasSideFoulingResult(
            fouling_detected=False,
            fouling_severity=FoulingSeverity.NONE,
            fouling_trend="stable",
            current_dp_in_wc=2.0,
            design_dp_in_wc=2.0,
            corrected_dp_in_wc=2.0,
            dp_ratio=1.0,
            dp_deviation_pct=0.0,
            u_actual_btu_hr_ft2_f=10.0,
            u_clean_btu_hr_ft2_f=10.0,
            u_degradation_pct=0.0,
            fouling_resistance_hr_ft2_f_btu=0.0,
            cleaning_status=CleaningStatus.NOT_REQUIRED,
        )

        water_side = WaterSideFoulingResult(
            fouling_detected=False,
            fouling_severity=FoulingSeverity.NONE,
            fouling_type="none",
            current_dp_psi=5.0,
            design_dp_psi=5.0,
            corrected_dp_psi=5.0,
            dp_ratio=1.0,
            fouling_factor_hr_ft2_f_btu=0.001,
            design_fouling_factor=0.001,
            fouling_factor_ratio=1.0,
            chemistry_compliant=True,
            chemistry_deviations=[],
            cleaning_status=CleaningStatus.NOT_REQUIRED,
        )

        soot_blower = SootBlowerResult(
            blowing_recommended=False,
            blowing_status=SootBlowingStatus.IDLE,
            hours_since_last_blow=4.0,
            recommended_next_blow_hours=4.0,
            optimal_blow_interval_hours=8.0,
            dp_trigger_active=False,
            effectiveness_trigger_active=False,
            time_trigger_active=False,
            trigger_reason="",
            estimated_steam_per_cycle_lb=2000.0,
            blowing_efficiency_score=0.9,
        )

        acid_dew_point = AcidDewPointResult(
            sulfuric_acid_dew_point_f=270.0,
            water_dew_point_f=120.0,
            effective_dew_point_f=270.0,
            min_metal_temp_f=310.0,
            avg_metal_temp_f=315.0,
            margin_above_dew_point_f=40.0,
            so3_concentration_ppm=5.0,
            h2o_concentration_pct=10.0,
            excess_air_pct=15.0,
            min_recommended_metal_temp_f=300.0,
        )

        effectiveness = EffectivenessResult(
            current_effectiveness=0.78,
            design_effectiveness=0.80,
            effectiveness_ratio=0.975,
            effectiveness_deviation_pct=2.5,
            current_ntu=1.9,
            design_ntu=2.0,
            current_ua_btu_hr_f=95000.0,
            design_ua_btu_hr_f=100000.0,
            clean_ua_btu_hr_f=120000.0,
            ua_degradation_pct=20.8,
            actual_duty_btu_hr=19500000.0,
            expected_duty_btu_hr=20000000.0,
            duty_deficit_btu_hr=500000.0,
            lmtd_f=145.0,
            approach_temp_f=265.0,
            gas_temp_drop_f=245.0,
            water_temp_rise_f=78.0,
            c_min_btu_hr_f=25000.0,
            c_max_btu_hr_f=78000.0,
            capacity_ratio=0.32,
        )

        steaming = SteamingResult(
            approach_temp_f=28.0,
            design_approach_f=30.0,
            approach_margin_f=-2.0,
            water_outlet_temp_f=442.0,
            saturation_temp_f=470.0,
            subcooling_f=28.0,
            current_load_pct=75.0,
            water_flow_pct=80.0,
            min_safe_load_pct=30.0,
            current_min_load_margin_pct=45.0,
        )

        return {
            "gas_side_fouling": gas_side,
            "water_side_fouling": water_side,
            "soot_blower": soot_blower,
            "acid_dew_point": acid_dew_point,
            "effectiveness": effectiveness,
            "steaming": steaming,
        }

    def test_valid_complete_output(self, complete_output_components):
        """Test valid complete output creation."""
        output = EconomizerOutput(
            economizer_id="ECO-001",
            **complete_output_components,
        )

        assert output.economizer_id == "ECO-001"
        assert output.status == "success"
        assert output.operating_status == EconomizerStatus.NORMAL
        assert output.gas_side_fouling.fouling_detected is False
        assert output.effectiveness.current_effectiveness == 0.78

    def test_request_id_auto_generated(self, complete_output_components):
        """Test request ID is auto-generated."""
        output = EconomizerOutput(
            economizer_id="ECO-001",
            **complete_output_components,
        )

        assert output.request_id is not None
        # UUID format
        assert len(output.request_id) == 36

    def test_timestamp_auto_generated(self, complete_output_components):
        """Test timestamp is auto-generated."""
        output = EconomizerOutput(
            economizer_id="ECO-001",
            **complete_output_components,
        )

        assert output.timestamp is not None
        assert isinstance(output.timestamp, datetime)

    def test_default_collections(self, complete_output_components):
        """Test default collections are empty."""
        output = EconomizerOutput(
            economizer_id="ECO-001",
            **complete_output_components,
        )

        assert output.recommendations == []
        assert output.alerts == []
        assert output.kpis == {}
        assert output.metadata == {}

    def test_with_recommendations_and_alerts(self, complete_output_components):
        """Test output with recommendations and alerts."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            category="test",
            title="Test Alert",
            description="Test",
        )

        rec = OptimizationRecommendation(
            category="test",
            title="Test Rec",
            description="Test",
        )

        output = EconomizerOutput(
            economizer_id="ECO-001",
            recommendations=[rec],
            alerts=[alert],
            **complete_output_components,
        )

        assert len(output.recommendations) == 1
        assert len(output.alerts) == 1

    def test_with_kpis(self, complete_output_components):
        """Test output with KPIs."""
        kpis = {
            "effectiveness_pct": 78.0,
            "dp_ratio": 1.0,
            "health_score": 95.0,
        }

        output = EconomizerOutput(
            economizer_id="ECO-001",
            kpis=kpis,
            **complete_output_components,
        )

        assert output.kpis["effectiveness_pct"] == 78.0
        assert output.kpis["health_score"] == 95.0

    def test_with_metadata(self, complete_output_components):
        """Test output with metadata."""
        metadata = {
            "agent_id": "GL-020",
            "version": "1.0.0",
        }

        output = EconomizerOutput(
            economizer_id="ECO-001",
            metadata=metadata,
            **complete_output_components,
        )

        assert output.metadata["agent_id"] == "GL-020"
        assert output.metadata["version"] == "1.0.0"

    def test_with_provenance(self, complete_output_components):
        """Test output with provenance hashes."""
        output = EconomizerOutput(
            economizer_id="ECO-001",
            provenance_hash="abc123def456",
            input_hash="xyz789",
            **complete_output_components,
        )

        assert output.provenance_hash == "abc123def456"
        assert output.input_hash == "xyz789"


# =============================================================================
# PARAMETERIZED TESTS
# =============================================================================

class TestSchemasParameterized:
    """Parameterized tests for schema validation."""

    @pytest.mark.parametrize("severity", list(FoulingSeverity))
    def test_all_fouling_severities_valid(self, severity):
        """Test all fouling severities work in results."""
        result = GasSideFoulingResult(
            fouling_detected=severity != FoulingSeverity.NONE,
            fouling_severity=severity,
            fouling_trend="stable",
            current_dp_in_wc=2.0,
            design_dp_in_wc=2.0,
            corrected_dp_in_wc=2.0,
            dp_ratio=1.0,
            dp_deviation_pct=0.0,
            u_actual_btu_hr_ft2_f=10.0,
            u_clean_btu_hr_ft2_f=10.0,
            u_degradation_pct=0.0,
            fouling_resistance_hr_ft2_f_btu=0.0,
            cleaning_status=CleaningStatus.NOT_REQUIRED,
        )
        assert result.fouling_severity == severity

    @pytest.mark.parametrize("status", list(EconomizerStatus))
    def test_all_operating_statuses_valid(self, status):
        """Test all operating statuses work in input."""
        input_data = EconomizerInput(
            economizer_id="ECO-001",
            operating_status=status,
            load_pct=75.0,
            gas_inlet_temp_f=600.0,
            gas_inlet_flow_lb_hr=100000.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_inlet_flow_lb_hr=80000.0,
            water_inlet_pressure_psig=550.0,
            water_outlet_temp_f=330.0,
            water_outlet_pressure_psig=540.0,
        )
        assert input_data.operating_status == status

    @pytest.mark.parametrize("load_pct", [0.0, 25.0, 50.0, 75.0, 100.0, 110.0, 120.0])
    def test_valid_load_percentages(self, load_pct):
        """Test valid load percentages."""
        input_data = EconomizerInput(
            economizer_id="ECO-001",
            load_pct=load_pct,
            gas_inlet_temp_f=600.0,
            gas_inlet_flow_lb_hr=100000.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_inlet_flow_lb_hr=80000.0,
            water_inlet_pressure_psig=550.0,
            water_outlet_temp_f=330.0,
            water_outlet_pressure_psig=540.0,
        )
        assert input_data.load_pct == load_pct
