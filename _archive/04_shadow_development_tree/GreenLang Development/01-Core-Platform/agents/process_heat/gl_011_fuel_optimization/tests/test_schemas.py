"""
GL-011 FUELCRAFT - Schema Tests

Unit tests for all Pydantic schema models in schemas.py.
Validates data structures, validation rules, and serialization.
"""

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError as PydanticValidationError
import uuid

from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelStatus,
    BlendStatus,
    InventoryAlertType,
    FuelProperties,
    FuelPrice,
    BlendRecommendation,
    SwitchingRecommendation,
    InventoryStatus,
    CostAnalysis,
    OptimizationResult,
    FuelOptimizationInput,
    FuelOptimizationOutput,
)


class TestFuelStatus:
    """Tests for FuelStatus enum."""

    def test_fuel_status_values(self):
        """Test all fuel status values."""
        assert FuelStatus.ONLINE.value == "online"
        assert FuelStatus.STANDBY.value == "standby"
        assert FuelStatus.TRANSITIONING.value == "transitioning"
        assert FuelStatus.OFFLINE.value == "offline"
        assert FuelStatus.MAINTENANCE.value == "maintenance"
        assert FuelStatus.EMERGENCY.value == "emergency"


class TestBlendStatus:
    """Tests for BlendStatus enum."""

    def test_blend_status_values(self):
        """Test all blend status values."""
        assert BlendStatus.OPTIMAL.value == "optimal"
        assert BlendStatus.SUB_OPTIMAL.value == "sub_optimal"
        assert BlendStatus.INFEASIBLE.value == "infeasible"
        assert BlendStatus.MANUAL_OVERRIDE.value == "manual_override"


class TestInventoryAlertType:
    """Tests for InventoryAlertType enum."""

    def test_alert_type_values(self):
        """Test all inventory alert type values."""
        assert InventoryAlertType.LOW_LEVEL.value == "low_level"
        assert InventoryAlertType.CRITICAL_LEVEL.value == "critical_level"
        assert InventoryAlertType.HIGH_LEVEL.value == "high_level"
        assert InventoryAlertType.DELIVERY_REQUIRED.value == "delivery_required"
        assert InventoryAlertType.QUALITY_ISSUE.value == "quality_issue"


class TestFuelProperties:
    """Tests for FuelProperties schema."""

    def test_required_fuel_type(self):
        """Test fuel_type is required."""
        props = FuelProperties(fuel_type="natural_gas")
        assert props.fuel_type == "natural_gas"

        with pytest.raises(PydanticValidationError):
            FuelProperties()

    def test_gas_properties(self, natural_gas_properties):
        """Test gas fuel properties."""
        props = natural_gas_properties

        assert props.hhv_btu_scf == 1020.0
        assert props.lhv_btu_scf == 918.0
        assert props.specific_gravity == 0.60
        assert props.wobbe_index == 1317.0
        assert props.co2_kg_mmbtu == 53.06
        assert props.methane_pct == 95.0

    def test_liquid_properties(self, fuel_oil_properties):
        """Test liquid fuel properties."""
        props = fuel_oil_properties

        assert props.hhv_btu_lb == 19580.0
        assert props.lhv_btu_lb == 18410.0
        assert props.density_lb_gal == 7.21
        assert props.co2_kg_mmbtu == 73.16

    def test_composition_validation(self):
        """Test composition percentage constraints."""
        # Valid composition
        props = FuelProperties(
            fuel_type="test_gas",
            methane_pct=95.0,
            ethane_pct=5.0,
        )
        assert props.methane_pct == 95.0

        # Invalid: percentage > 100
        with pytest.raises(PydanticValidationError):
            FuelProperties(
                fuel_type="test_gas",
                methane_pct=101.0,
            )

        # Invalid: negative percentage
        with pytest.raises(PydanticValidationError):
            FuelProperties(
                fuel_type="test_gas",
                methane_pct=-1.0,
            )

    def test_heating_value_constraints(self):
        """Test heating value must be positive."""
        with pytest.raises(PydanticValidationError):
            FuelProperties(
                fuel_type="test",
                hhv_btu_scf=0,
            )

        with pytest.raises(PydanticValidationError):
            FuelProperties(
                fuel_type="test",
                hhv_btu_scf=-100,
            )


class TestFuelPrice:
    """Tests for FuelPrice schema."""

    def test_required_fields(self):
        """Test required fields validation."""
        now = datetime.now(timezone.utc)
        price = FuelPrice(
            fuel_type="natural_gas",
            price=3.50,
            source="henry_hub",
            commodity_price=3.00,
        )
        assert price.fuel_type == "natural_gas"
        assert price.price == 3.50

    def test_price_breakdown(self, natural_gas_price):
        """Test price breakdown fields."""
        price = natural_gas_price

        assert price.commodity_price == 3.00
        assert price.transport_cost == 0.15
        assert price.basis_differential == 0.20
        assert price.taxes == 0.15
        assert price.confidence == 0.95

    def test_price_constraints(self):
        """Test price must be non-negative."""
        with pytest.raises(PydanticValidationError):
            FuelPrice(
                fuel_type="test",
                price=-1.0,
                source="test",
                commodity_price=0,
            )

    def test_default_unit(self):
        """Test default unit is USD/MMBTU."""
        price = FuelPrice(
            fuel_type="test",
            price=5.0,
            source="test",
            commodity_price=4.0,
        )
        assert price.unit == "USD/MMBTU"

    def test_timestamp_default(self):
        """Test timestamp defaults to now."""
        price = FuelPrice(
            fuel_type="test",
            price=5.0,
            source="test",
            commodity_price=4.0,
        )
        assert price.timestamp is not None
        assert price.timestamp.tzinfo is not None


class TestBlendRecommendation:
    """Tests for BlendRecommendation schema."""

    def test_blend_ratios_validation(self):
        """Test blend ratios must sum to 100%."""
        # Valid blend
        blend = BlendRecommendation(
            blend_ratios={"natural_gas": 70.0, "propane": 30.0},
            primary_fuel="natural_gas",
            blended_hhv=1050.0,
            blended_co2_factor=55.0,
            blended_cost_usd_mmbtu=4.50,
        )
        assert sum(blend.blend_ratios.values()) == pytest.approx(100.0)

        # Invalid: doesn't sum to 100%
        with pytest.raises(PydanticValidationError):
            BlendRecommendation(
                blend_ratios={"natural_gas": 70.0, "propane": 20.0},
                primary_fuel="natural_gas",
                blended_hhv=1050.0,
                blended_co2_factor=55.0,
                blended_cost_usd_mmbtu=4.50,
            )

    def test_blend_status(self):
        """Test blend status values."""
        blend = BlendRecommendation(
            status=BlendStatus.OPTIMAL,
            blend_ratios={"natural_gas": 100.0},
            primary_fuel="natural_gas",
            blended_hhv=1020.0,
            blended_co2_factor=53.06,
            blended_cost_usd_mmbtu=3.50,
        )
        assert blend.status == BlendStatus.OPTIMAL

    def test_constraint_satisfaction_flags(self):
        """Test constraint satisfaction flags."""
        blend = BlendRecommendation(
            blend_ratios={"natural_gas": 100.0},
            primary_fuel="natural_gas",
            blended_hhv=1020.0,
            blended_co2_factor=53.06,
            blended_cost_usd_mmbtu=3.50,
            wobbe_in_range=True,
            hhv_in_range=True,
            emissions_in_range=False,
        )
        assert blend.wobbe_in_range is True
        assert blend.hhv_in_range is True
        assert blend.emissions_in_range is False

    def test_recommendation_id_generation(self):
        """Test recommendation ID is auto-generated."""
        blend = BlendRecommendation(
            blend_ratios={"natural_gas": 100.0},
            primary_fuel="natural_gas",
            blended_hhv=1020.0,
            blended_co2_factor=53.06,
            blended_cost_usd_mmbtu=3.50,
        )
        assert blend.recommendation_id is not None
        assert len(blend.recommendation_id) == 8


class TestSwitchingRecommendation:
    """Tests for SwitchingRecommendation schema."""

    def test_required_fields(self):
        """Test required fields for switching recommendation."""
        now = datetime.now(timezone.utc)
        rec = SwitchingRecommendation(
            recommended=True,
            current_fuel="natural_gas",
            recommended_fuel="no2_fuel_oil",
            trigger_reason="Price savings",
            current_cost_usd_hr=175.0,
            recommended_cost_usd_hr=150.0,
            savings_usd_hr=25.0,
            transition_time_minutes=15,
            valid_until=now + timedelta(hours=1),
        )
        assert rec.recommended is True
        assert rec.savings_usd_hr == 25.0

    def test_safety_checks(self):
        """Test safety check fields."""
        now = datetime.now(timezone.utc)
        rec = SwitchingRecommendation(
            recommended=True,
            current_fuel="natural_gas",
            recommended_fuel="no2_fuel_oil",
            trigger_reason="Price savings",
            current_cost_usd_hr=175.0,
            recommended_cost_usd_hr=150.0,
            savings_usd_hr=25.0,
            transition_time_minutes=15,
            valid_until=now + timedelta(hours=1),
            safety_checks_passed=True,
            safety_warnings=["Check flame sensor"],
            requires_purge=True,
        )
        assert rec.safety_checks_passed is True
        assert len(rec.safety_warnings) == 1
        assert rec.requires_purge is True


class TestInventoryStatus:
    """Tests for InventoryStatus schema."""

    def test_required_fields(self):
        """Test required fields for inventory status."""
        status = InventoryStatus(
            tank_id="TANK-001",
            fuel_type="natural_gas",
            current_level_gal=5000.0,
            current_level_pct=50.0,
            capacity_gal=10000.0,
            usable_capacity_gal=9500.0,
            reorder_point_gal=3000.0,
            safety_stock_gal=1500.0,
            critical_level_gal=1000.0,
        )
        assert status.tank_id == "TANK-001"
        assert status.current_level_pct == 50.0

    def test_level_constraints(self):
        """Test level percentage constraints."""
        # Valid levels
        status = InventoryStatus(
            tank_id="TANK-001",
            fuel_type="test",
            current_level_gal=0,
            current_level_pct=0.0,
            capacity_gal=10000.0,
            usable_capacity_gal=9500.0,
            reorder_point_gal=3000.0,
            safety_stock_gal=1500.0,
            critical_level_gal=1000.0,
        )
        assert status.current_level_pct == 0.0

        # Invalid: > 100%
        with pytest.raises(PydanticValidationError):
            InventoryStatus(
                tank_id="TANK-001",
                fuel_type="test",
                current_level_gal=11000,
                current_level_pct=110.0,
                capacity_gal=10000.0,
                usable_capacity_gal=9500.0,
                reorder_point_gal=3000.0,
                safety_stock_gal=1500.0,
                critical_level_gal=1000.0,
            )

    def test_consumption_fields(self):
        """Test consumption tracking fields."""
        status = InventoryStatus(
            tank_id="TANK-001",
            fuel_type="natural_gas",
            current_level_gal=5000.0,
            current_level_pct=50.0,
            capacity_gal=10000.0,
            usable_capacity_gal=9500.0,
            reorder_point_gal=3000.0,
            safety_stock_gal=1500.0,
            critical_level_gal=1000.0,
            consumption_rate_gal_hr=100.0,
            avg_daily_consumption_gal=2400.0,
            days_of_supply=2.0,
        )
        assert status.consumption_rate_gal_hr == 100.0
        assert status.days_of_supply == 2.0


class TestCostAnalysis:
    """Tests for CostAnalysis schema."""

    def test_required_fields(self):
        """Test required fields for cost analysis."""
        analysis = CostAnalysis(
            period_hours=8760.0,
            fuel_cost_usd=1000000.0,
            total_cost_usd=1200000.0,
            cost_per_mmbtu=3.50,
        )
        assert analysis.period_hours == 8760.0
        assert analysis.total_cost_usd == 1200000.0

    def test_cost_breakdown(self):
        """Test cost breakdown fields."""
        analysis = CostAnalysis(
            period_hours=8760.0,
            fuel_cost_usd=1000000.0,
            transport_cost_usd=50000.0,
            storage_cost_usd=20000.0,
            carbon_cost_usd=100000.0,
            maintenance_cost_usd=30000.0,
            total_cost_usd=1200000.0,
            cost_per_mmbtu=3.50,
        )
        assert analysis.transport_cost_usd == 50000.0
        assert analysis.carbon_cost_usd == 100000.0

    def test_emissions_fields(self):
        """Test emissions tracking fields."""
        analysis = CostAnalysis(
            period_hours=8760.0,
            fuel_cost_usd=1000000.0,
            total_cost_usd=1200000.0,
            cost_per_mmbtu=3.50,
            total_co2_kg=5000000.0,
            total_co2_cost_usd=100000.0,
            co2_intensity_kg_mmbtu=53.06,
        )
        assert analysis.total_co2_kg == 5000000.0
        assert analysis.co2_intensity_kg_mmbtu == 53.06


class TestOptimizationResult:
    """Tests for OptimizationResult schema."""

    def test_required_fields(self):
        """Test required fields for optimization result."""
        result = OptimizationResult(
            recommended_fuel_cost_usd_hr=175.0,
            current_fuel_cost_usd_hr=200.0,
        )
        assert result.recommended_fuel_cost_usd_hr == 175.0
        assert result.potential_savings_usd_hr == 0.0  # Default

    def test_savings_calculation(self):
        """Test savings fields."""
        result = OptimizationResult(
            recommended_fuel_cost_usd_hr=175.0,
            current_fuel_cost_usd_hr=200.0,
            potential_savings_usd_hr=25.0,
            potential_savings_usd_year=219000.0,
        )
        assert result.potential_savings_usd_hr == 25.0
        assert result.potential_savings_usd_year == 219000.0

    def test_emissions_fields(self):
        """Test emissions fields in result."""
        result = OptimizationResult(
            recommended_fuel_cost_usd_hr=175.0,
            current_fuel_cost_usd_hr=200.0,
            current_co2_kg_hr=2653.0,
            recommended_co2_kg_hr=2500.0,
            co2_reduction_kg_hr=153.0,
        )
        assert result.co2_reduction_kg_hr == 153.0

    def test_confidence_score(self):
        """Test confidence score constraints."""
        result = OptimizationResult(
            recommended_fuel_cost_usd_hr=175.0,
            current_fuel_cost_usd_hr=200.0,
            confidence_score=0.95,
        )
        assert result.confidence_score == 0.95

        # Invalid: > 1.0
        with pytest.raises(PydanticValidationError):
            OptimizationResult(
                recommended_fuel_cost_usd_hr=175.0,
                current_fuel_cost_usd_hr=200.0,
                confidence_score=1.5,
            )

    def test_result_id_generation(self):
        """Test result ID is auto-generated."""
        result = OptimizationResult(
            recommended_fuel_cost_usd_hr=175.0,
            current_fuel_cost_usd_hr=200.0,
        )
        assert result.result_id is not None
        # UUID format
        uuid.UUID(result.result_id)


class TestFuelOptimizationInput:
    """Tests for FuelOptimizationInput schema."""

    def test_required_fields(self):
        """Test required fields for input."""
        input_data = FuelOptimizationInput(
            facility_id="PLANT-001",
            current_fuel="natural_gas",
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=50.0,
        )
        assert input_data.facility_id == "PLANT-001"
        assert input_data.current_heat_input_mmbtu_hr == 50.0

    def test_load_constraints(self):
        """Test load percentage constraints."""
        input_data = FuelOptimizationInput(
            facility_id="PLANT-001",
            current_fuel="natural_gas",
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=50.0,
            current_load_pct=80.0,
        )
        assert input_data.current_load_pct == 80.0

        # Invalid: > 120%
        with pytest.raises(PydanticValidationError):
            FuelOptimizationInput(
                facility_id="PLANT-001",
                current_fuel="natural_gas",
                current_fuel_flow_rate=50000.0,
                current_heat_input_mmbtu_hr=50.0,
                current_load_pct=130.0,
            )

    def test_request_id_generation(self):
        """Test request ID is auto-generated."""
        input_data = FuelOptimizationInput(
            facility_id="PLANT-001",
            current_fuel="natural_gas",
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=50.0,
        )
        assert input_data.request_id is not None

    def test_optional_constraints(self):
        """Test optional constraint fields."""
        input_data = FuelOptimizationInput(
            facility_id="PLANT-001",
            current_fuel="natural_gas",
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=50.0,
            max_emissions_kg_hr=3000.0,
            min_efficiency_pct=80.0,
            excluded_fuels=["coal_bituminous"],
        )
        assert input_data.max_emissions_kg_hr == 3000.0
        assert input_data.min_efficiency_pct == 80.0
        assert "coal_bituminous" in input_data.excluded_fuels


class TestFuelOptimizationOutput:
    """Tests for FuelOptimizationOutput schema."""

    def test_required_fields(self):
        """Test required fields for output."""
        result = OptimizationResult(
            recommended_fuel_cost_usd_hr=175.0,
            current_fuel_cost_usd_hr=200.0,
        )
        output = FuelOptimizationOutput(
            facility_id="PLANT-001",
            request_id="test-123",
            optimization_result=result,
        )
        assert output.facility_id == "PLANT-001"
        assert output.status == "success"  # Default

    def test_processing_time(self):
        """Test processing time field."""
        result = OptimizationResult(
            recommended_fuel_cost_usd_hr=175.0,
            current_fuel_cost_usd_hr=200.0,
        )
        output = FuelOptimizationOutput(
            facility_id="PLANT-001",
            request_id="test-123",
            optimization_result=result,
            processing_time_ms=45.5,
        )
        assert output.processing_time_ms == 45.5

    def test_provenance_hash(self):
        """Test provenance hash field."""
        result = OptimizationResult(
            recommended_fuel_cost_usd_hr=175.0,
            current_fuel_cost_usd_hr=200.0,
        )
        output = FuelOptimizationOutput(
            facility_id="PLANT-001",
            request_id="test-123",
            optimization_result=result,
            provenance_hash="abc123def456",
        )
        assert output.provenance_hash == "abc123def456"

    def test_kpis_dictionary(self):
        """Test KPIs dictionary field."""
        result = OptimizationResult(
            recommended_fuel_cost_usd_hr=175.0,
            current_fuel_cost_usd_hr=200.0,
        )
        output = FuelOptimizationOutput(
            facility_id="PLANT-001",
            request_id="test-123",
            optimization_result=result,
            kpis={
                "fuel_cost_usd_hr": 175.0,
                "co2_emissions_kg_hr": 2653.0,
                "efficiency_pct": 82.0,
            },
        )
        assert output.kpis["fuel_cost_usd_hr"] == 175.0
        assert len(output.kpis) == 3
