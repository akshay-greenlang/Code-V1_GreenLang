"""
GL-003 Unified Steam System Optimizer - Schema Tests

Unit tests for Pydantic data model schemas.
Target: 85%+ coverage of schemas.py

Tests:
    - Steam property model validation
    - Header balance input/output models
    - Quality reading models
    - PRV models
    - Condensate models
    - Flash steam models
    - Trap models
    - Validator functions
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    OptimizationRecommendation,
    UnifiedSteamOptimizerOutput,
)


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEnums:
    """Test suite for schema enums."""

    def test_steam_phase_values(self):
        """Test SteamPhase enum values."""
        assert SteamPhase.SUBCOOLED_LIQUID.value == "subcooled_liquid"
        assert SteamPhase.SATURATED_LIQUID.value == "saturated_liquid"
        assert SteamPhase.WET_STEAM.value == "wet_steam"
        assert SteamPhase.SATURATED_VAPOR.value == "saturated_vapor"
        assert SteamPhase.SUPERHEATED_VAPOR.value == "superheated_vapor"

    def test_validation_status_values(self):
        """Test ValidationStatus enum values."""
        assert ValidationStatus.VALID.value == "valid"
        assert ValidationStatus.WARNING.value == "warning"
        assert ValidationStatus.INVALID.value == "invalid"
        assert ValidationStatus.UNCHECKED.value == "unchecked"

    def test_optimization_status_values(self):
        """Test OptimizationStatus enum values."""
        assert OptimizationStatus.OPTIMAL.value == "optimal"
        assert OptimizationStatus.SUBOPTIMAL.value == "suboptimal"
        assert OptimizationStatus.CRITICAL.value == "critical"
        assert OptimizationStatus.FAILED.value == "failed"

    def test_trap_status_values(self):
        """Test TrapStatus enum values."""
        assert TrapStatus.OPERATING.value == "operating"
        assert TrapStatus.FAILED_OPEN.value == "failed_open"
        assert TrapStatus.FAILED_CLOSED.value == "failed_closed"
        assert TrapStatus.LEAKING.value == "leaking"
        assert TrapStatus.UNKNOWN.value == "unknown"


# =============================================================================
# STEAM PROPERTIES TESTS
# =============================================================================

class TestSteamProperties:
    """Test suite for SteamProperties model."""

    def test_valid_steam_properties(self, steam_properties_saturated):
        """Test valid steam properties."""
        assert steam_properties_saturated.pressure_psig == 150.0
        assert steam_properties_saturated.temperature_f == 365.9
        assert steam_properties_saturated.phase == SteamPhase.SATURATED_VAPOR

    def test_psia_calculation(self):
        """Test automatic pressure_psia calculation."""
        props = SteamProperties(
            pressure_psig=150.0,
            temperature_f=365.9,
            enthalpy_btu_lb=1196.0,
        )
        assert props.pressure_psia == pytest.approx(164.696, rel=1e-3)

    def test_density_calculation(self):
        """Test automatic density calculation from specific volume."""
        props = SteamProperties(
            pressure_psig=150.0,
            temperature_f=365.9,
            enthalpy_btu_lb=1196.0,
            specific_volume_ft3_lb=2.75,
        )
        assert props.density_lb_ft3 == pytest.approx(0.3636, rel=1e-3)

    def test_dryness_fraction_bounds(self):
        """Test dryness fraction bounds (0-1)."""
        with pytest.raises(ValidationError):
            SteamProperties(
                pressure_psig=150.0,
                temperature_f=365.9,
                enthalpy_btu_lb=1196.0,
                dryness_fraction=1.5,  # Above 1.0
            )

    def test_temperature_bounds(self):
        """Test temperature bounds (32-1500F)."""
        with pytest.raises(ValidationError):
            SteamProperties(
                pressure_psig=150.0,
                temperature_f=20.0,  # Below 32F
                enthalpy_btu_lb=1196.0,
            )

    def test_superheated_properties(self, steam_properties_superheated):
        """Test superheated steam properties."""
        assert steam_properties_superheated.phase == SteamPhase.SUPERHEATED_VAPOR
        assert steam_properties_superheated.superheat_f == 261.0
        assert steam_properties_superheated.dryness_fraction == 1.0


# =============================================================================
# STEAM FLOW MEASUREMENT TESTS
# =============================================================================

class TestSteamFlowMeasurement:
    """Test suite for SteamFlowMeasurement model."""

    def test_valid_flow_measurement(self):
        """Test valid flow measurement."""
        flow = SteamFlowMeasurement(
            flow_rate_lb_hr=50000.0,
            measurement_type="orifice",
            uncertainty_pct=2.0,
        )
        assert flow.flow_rate_lb_hr == 50000.0
        assert flow.flow_rate_klb_hr == 50.0

    def test_klb_calculation(self):
        """Test automatic klb/hr calculation."""
        flow = SteamFlowMeasurement(flow_rate_lb_hr=100000.0)
        assert flow.flow_rate_klb_hr == 100.0

    def test_negative_flow_rejection(self):
        """Test negative flow rate is rejected."""
        with pytest.raises(ValidationError):
            SteamFlowMeasurement(flow_rate_lb_hr=-1000.0)

    def test_uncertainty_bounds(self):
        """Test uncertainty bounds (0-20%)."""
        with pytest.raises(ValidationError):
            SteamFlowMeasurement(
                flow_rate_lb_hr=50000.0,
                uncertainty_pct=25.0,  # Above 20%
            )

    def test_timestamp_default(self):
        """Test timestamp defaults to current time."""
        flow = SteamFlowMeasurement(flow_rate_lb_hr=50000.0)
        assert flow.timestamp is not None
        assert isinstance(flow.timestamp, datetime)


# =============================================================================
# HEADER READING TESTS
# =============================================================================

class TestHeaderReading:
    """Test suite for HeaderReading model."""

    def test_valid_header_reading(self):
        """Test valid header reading."""
        reading = HeaderReading(
            header_id="HP-MAIN",
            pressure_psig=598.0,
            pressure_setpoint_psig=600.0,
            temperature_f=745.0,
        )
        assert reading.header_id == "HP-MAIN"
        assert reading.pressure_deviation_psi == pytest.approx(-2.0)

    def test_pressure_deviation_calculation(self):
        """Test automatic pressure deviation calculation."""
        reading = HeaderReading(
            header_id="TEST",
            pressure_psig=605.0,
            pressure_setpoint_psig=600.0,
            temperature_f=750.0,
        )
        assert reading.pressure_deviation_psi == pytest.approx(5.0)

    def test_imbalance_calculation(self):
        """Test automatic imbalance calculation."""
        reading = HeaderReading(
            header_id="TEST",
            pressure_psig=600.0,
            pressure_setpoint_psig=600.0,
            temperature_f=750.0,
            total_supply_lb_hr=100000.0,
            total_demand_lb_hr=95000.0,
        )
        assert reading.imbalance_lb_hr == pytest.approx(5000.0)


# =============================================================================
# HEADER BALANCE INPUT/OUTPUT TESTS
# =============================================================================

class TestHeaderBalanceInput:
    """Test suite for HeaderBalanceInput model."""

    def test_valid_header_balance_input(self, header_balance_input):
        """Test valid header balance input."""
        assert header_balance_input.header_id == "HP-MAIN"
        assert len(header_balance_input.supplies) == 2
        assert len(header_balance_input.demands) == 2

    def test_supplies_and_demands_format(self, header_balance_input):
        """Test supplies and demands data structure."""
        supply = header_balance_input.supplies[0]
        assert "id" in supply
        assert "flow_lb_hr" in supply


class TestHeaderBalanceOutput:
    """Test suite for HeaderBalanceOutput model."""

    def test_valid_header_balance_output(self):
        """Test valid header balance output."""
        output = HeaderBalanceOutput(
            header_id="HP-MAIN",
            status=OptimizationStatus.OPTIMAL,
            total_supply_lb_hr=100000.0,
            total_demand_lb_hr=95000.0,
            imbalance_lb_hr=5000.0,
            imbalance_pct=5.26,
            pressure_psig=600.0,
            pressure_deviation_psi=0.0,
            provenance_hash="abc123" * 10 + "abcd",
        )
        assert output.status == OptimizationStatus.OPTIMAL
        assert output.imbalance_lb_hr == 5000.0

    def test_provenance_hash_required(self):
        """Test provenance hash is required."""
        with pytest.raises(ValidationError):
            HeaderBalanceOutput(
                header_id="TEST",
                status=OptimizationStatus.OPTIMAL,
                total_supply_lb_hr=100000.0,
                total_demand_lb_hr=95000.0,
                imbalance_lb_hr=5000.0,
                imbalance_pct=5.0,
                pressure_psig=600.0,
                pressure_deviation_psi=0.0,
                # Missing provenance_hash
            )


# =============================================================================
# STEAM QUALITY READING TESTS
# =============================================================================

class TestSteamQualityReading:
    """Test suite for SteamQualityReading model."""

    def test_valid_quality_reading(self, steam_quality_reading_good):
        """Test valid quality reading."""
        assert steam_quality_reading_good.dryness_fraction == 0.98
        assert steam_quality_reading_good.tds_ppm == 10.0

    def test_moisture_content_calculation(self):
        """Test automatic moisture content calculation."""
        reading = SteamQualityReading(
            location_id="TEST",
            pressure_psig=150.0,
            temperature_f=366.0,
            dryness_fraction=0.95,
        )
        assert reading.moisture_content_pct == pytest.approx(5.0)

    def test_dryness_fraction_bounds(self):
        """Test dryness fraction bounds (0-1)."""
        with pytest.raises(ValidationError):
            SteamQualityReading(
                location_id="TEST",
                pressure_psig=150.0,
                temperature_f=366.0,
                dryness_fraction=1.2,  # Above 1.0
            )

    def test_ph_bounds(self):
        """Test pH bounds (0-14)."""
        with pytest.raises(ValidationError):
            SteamQualityReading(
                location_id="TEST",
                pressure_psig=150.0,
                temperature_f=366.0,
                ph=15.0,  # Above 14
            )


# =============================================================================
# STEAM QUALITY ANALYSIS TESTS
# =============================================================================

class TestSteamQualityAnalysis:
    """Test suite for SteamQualityAnalysis model."""

    def test_valid_quality_analysis(self, steam_quality_reading_good):
        """Test valid quality analysis."""
        analysis = SteamQualityAnalysis(
            reading=steam_quality_reading_good,
            overall_status=ValidationStatus.VALID,
            dryness_status=ValidationStatus.VALID,
            tds_status=ValidationStatus.VALID,
            conductivity_status=ValidationStatus.VALID,
        )
        assert analysis.overall_status == ValidationStatus.VALID

    def test_limits_exceeded_tracking(self, steam_quality_reading_poor):
        """Test limits exceeded tracking."""
        analysis = SteamQualityAnalysis(
            reading=steam_quality_reading_poor,
            overall_status=ValidationStatus.INVALID,
            limits_exceeded=["Dryness fraction below minimum"],
        )
        assert len(analysis.limits_exceeded) == 1


# =============================================================================
# PRV OPERATING POINT TESTS
# =============================================================================

class TestPRVOperatingPoint:
    """Test suite for PRVOperatingPoint model."""

    def test_valid_prv_operating_point(self, prv_operating_point):
        """Test valid PRV operating point."""
        assert prv_operating_point.prv_id == "PRV-HP-MP"
        assert prv_operating_point.opening_pct == 60.0

    def test_pressure_drop_calculation(self):
        """Test automatic pressure drop calculation."""
        point = PRVOperatingPoint(
            prv_id="TEST",
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=150.0,
            flow_rate_lb_hr=30000.0,
            opening_pct=60.0,
        )
        assert point.pressure_drop_psi == pytest.approx(450.0)

    def test_opening_pct_bounds(self):
        """Test opening percentage bounds (0-100)."""
        with pytest.raises(ValidationError):
            PRVOperatingPoint(
                prv_id="TEST",
                inlet_pressure_psig=600.0,
                outlet_pressure_psig=150.0,
                flow_rate_lb_hr=30000.0,
                opening_pct=110.0,  # Above 100
            )


# =============================================================================
# PRV SIZING INPUT/OUTPUT TESTS
# =============================================================================

class TestPRVSizingInput:
    """Test suite for PRVSizingInput model."""

    def test_valid_prv_sizing_input(self):
        """Test valid PRV sizing input."""
        sizing_input = PRVSizingInput(
            prv_id="TEST",
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=150.0,
            design_flow_lb_hr=30000.0,
            max_flow_lb_hr=40000.0,
        )
        assert sizing_input.inlet_pressure_psig == 600.0

    def test_outlet_less_than_inlet(self):
        """Test outlet pressure must be less than inlet."""
        with pytest.raises(ValidationError):
            PRVSizingInput(
                prv_id="TEST",
                inlet_pressure_psig=150.0,
                outlet_pressure_psig=600.0,  # Greater than inlet
                design_flow_lb_hr=30000.0,
                max_flow_lb_hr=40000.0,
            )


class TestPRVSizingOutput:
    """Test suite for PRVSizingOutput model."""

    def test_valid_prv_sizing_output(self):
        """Test valid PRV sizing output."""
        output = PRVSizingOutput(
            prv_id="TEST",
            cv_required=100.0,
            cv_recommended=115.0,
            opening_at_design_pct=60.0,
            opening_at_min_pct=30.0,
            opening_at_max_pct=80.0,
            meets_opening_targets=True,
            opening_target_status="Within 50-70% target",
            max_flow_capacity_lb_hr=50000.0,
            rangeability=50.0,
            provenance_hash="abc123" * 10 + "abcd",
        )
        assert output.meets_opening_targets is True
        assert output.cv_margin_pct == 15.0


# =============================================================================
# CONDENSATE READING TESTS
# =============================================================================

class TestCondensateReading:
    """Test suite for CondensateReading model."""

    def test_valid_condensate_reading(self, condensate_reading):
        """Test valid condensate reading."""
        assert condensate_reading.flow_rate_lb_hr == 40000.0
        assert condensate_reading.temperature_f == 185.0
        assert condensate_reading.is_contaminated is False

    def test_negative_flow_rejection(self):
        """Test negative flow rate is rejected."""
        with pytest.raises(ValidationError):
            CondensateReading(
                location_id="TEST",
                flow_rate_lb_hr=-1000.0,
                temperature_f=180.0,
            )


# =============================================================================
# CONDENSATE RETURN ANALYSIS TESTS
# =============================================================================

class TestCondensateReturnAnalysis:
    """Test suite for CondensateReturnAnalysis model."""

    def test_valid_return_analysis(self):
        """Test valid condensate return analysis."""
        analysis = CondensateReturnAnalysis(
            total_steam_flow_lb_hr=100000.0,
            condensate_return_lb_hr=85000.0,
            return_rate_pct=85.0,
            avg_return_temperature_f=180.0,
            heat_recovered_btu_hr=10200000.0,
            makeup_water_required_lb_hr=15000.0,
        )
        assert analysis.return_rate_pct == 85.0


# =============================================================================
# FLASH STEAM INPUT/OUTPUT TESTS
# =============================================================================

class TestFlashSteamInput:
    """Test suite for FlashSteamInput model."""

    def test_valid_flash_input(self, flash_steam_input):
        """Test valid flash steam input."""
        assert flash_steam_input.condensate_flow_lb_hr == 5000.0
        assert flash_steam_input.condensate_pressure_psig == 150.0

    def test_flash_less_than_condensate(self):
        """Test flash pressure must be less than condensate pressure."""
        with pytest.raises(ValidationError):
            FlashSteamInput(
                condensate_flow_lb_hr=5000.0,
                condensate_pressure_psig=15.0,
                flash_pressure_psig=150.0,  # Greater than condensate
            )


class TestFlashSteamOutput:
    """Test suite for FlashSteamOutput model."""

    def test_valid_flash_output(self):
        """Test valid flash steam output."""
        output = FlashSteamOutput(
            flash_fraction_pct=13.5,
            flash_steam_lb_hr=675.0,
            residual_condensate_lb_hr=4325.0,
            condensate_enthalpy_in_btu_lb=339.2,
            flash_steam_enthalpy_btu_lb=1164.3,
            residual_enthalpy_btu_lb=218.9,
            energy_recovered_btu_hr=785900.0,
            recovery_efficiency_pct=95.0,
            provenance_hash="abc123" * 10 + "abcd",
        )
        assert output.flash_fraction_pct == 13.5


# =============================================================================
# STEAM TRAP READING TESTS
# =============================================================================

class TestSteamTrapReading:
    """Test suite for SteamTrapReading model."""

    def test_valid_trap_reading(self, steam_trap_reading_good):
        """Test valid steam trap reading."""
        assert steam_trap_reading_good.trap_id == "TRAP-001"
        assert steam_trap_reading_good.status == TrapStatus.OPERATING

    def test_failed_trap_reading(self, steam_trap_reading_failed):
        """Test failed steam trap reading."""
        assert steam_trap_reading_failed.status == TrapStatus.FAILED_OPEN
        assert steam_trap_reading_failed.steam_loss_lb_hr == 50.0


# =============================================================================
# TRAP SURVEY ANALYSIS TESTS
# =============================================================================

class TestTrapSurveyAnalysis:
    """Test suite for TrapSurveyAnalysis model."""

    def test_valid_survey_analysis(self):
        """Test valid trap survey analysis."""
        analysis = TrapSurveyAnalysis(
            total_traps=100,
            operating_count=90,
            failed_open_count=5,
            failed_closed_count=3,
            unknown_count=2,
            failure_rate_pct=8.0,
            total_steam_loss_lb_hr=250.0,
        )
        assert analysis.failure_rate_pct == 8.0


# =============================================================================
# OPTIMIZATION RECOMMENDATION TESTS
# =============================================================================

class TestOptimizationRecommendation:
    """Test suite for OptimizationRecommendation model."""

    def test_valid_recommendation(self):
        """Test valid optimization recommendation."""
        rec = OptimizationRecommendation(
            category="header_balance",
            priority=1,
            description="Adjust boiler output",
            action="Increase BLR-001 output by 5000 lb/hr",
            energy_savings_pct=2.5,
            cost_savings_usd_year=25000.0,
        )
        assert rec.priority == 1
        assert rec.category == "header_balance"

    def test_priority_bounds(self):
        """Test priority bounds (1-5)."""
        with pytest.raises(ValidationError):
            OptimizationRecommendation(
                category="test",
                priority=0,  # Below 1
                description="Test",
                action="Test action",
            )

        with pytest.raises(ValidationError):
            OptimizationRecommendation(
                category="test",
                priority=6,  # Above 5
                description="Test",
                action="Test action",
            )


# =============================================================================
# UNIFIED STEAM OPTIMIZER OUTPUT TESTS
# =============================================================================

class TestUnifiedSteamOptimizerOutput:
    """Test suite for UnifiedSteamOptimizerOutput model."""

    def test_valid_optimizer_output(self):
        """Test valid optimizer output."""
        output = UnifiedSteamOptimizerOutput(
            overall_status=OptimizationStatus.OPTIMAL,
            system_efficiency_pct=92.5,
            exergy_efficiency_pct=45.0,
            provenance_hash="abc123" * 10 + "abcd",
        )
        assert output.overall_status == OptimizationStatus.OPTIMAL
        assert output.system_efficiency_pct == 92.5

    def test_provenance_hash_required(self):
        """Test provenance hash is required."""
        with pytest.raises(ValidationError):
            UnifiedSteamOptimizerOutput(
                overall_status=OptimizationStatus.OPTIMAL,
                system_efficiency_pct=92.5,
                # Missing provenance_hash
            )

    def test_default_lists_empty(self):
        """Test default lists are empty."""
        output = UnifiedSteamOptimizerOutput(
            overall_status=OptimizationStatus.OPTIMAL,
            system_efficiency_pct=92.5,
            provenance_hash="abc123" * 10 + "abcd",
        )
        assert output.header_analyses == []
        assert output.quality_analyses == []
        assert output.recommendations == []


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSchemaSerialization:
    """Test suite for schema serialization."""

    def test_steam_properties_json(self, steam_properties_saturated):
        """Test SteamProperties JSON serialization."""
        json_str = steam_properties_saturated.json()
        assert "150.0" in json_str
        assert "saturated_vapor" in json_str

    def test_quality_reading_json(self, steam_quality_reading_good):
        """Test SteamQualityReading JSON serialization."""
        json_str = steam_quality_reading_good.json()
        assert "BLR-001-STEAM" in json_str

    def test_optimizer_output_dict(self):
        """Test optimizer output dict conversion."""
        output = UnifiedSteamOptimizerOutput(
            overall_status=OptimizationStatus.OPTIMAL,
            system_efficiency_pct=92.5,
            provenance_hash="abc123" * 10 + "abcd",
        )
        output_dict = output.dict()
        assert "overall_status" in output_dict
        assert "system_efficiency_pct" in output_dict
