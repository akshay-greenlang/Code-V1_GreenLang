# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft - Fuel Mix Optimizer Workflow Unit Tests

Comprehensive unit tests for the FuelMixOptimizer class (optimization/fuel_mix_optimizer.py).
Tests the complete optimization workflow orchestration, result data structures,
and provenance tracking.

Validates:
- OptimizerConfig initialization and serialization
- FuelMixEntry, ProcurementSchedule, InventoryProjection data classes
- BlendQuality validation
- OptimizationResult structure and provenance
- FuelMixOptimizer initialization
- Result summary generation

Author: GL-TestEngineer
Date: 2025-01-01
Version: 1.0.0
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.fuel_mix_optimizer import (
    FuelMixOptimizer,
    OptimizerConfig,
    OptimizationResult,
    OptimizationStatus,
    FuelMixEntry,
    ProcurementSchedule,
    InventoryProjection,
    BlendQuality,
    BlendQualityStatus,
)
from optimization.cost_model import (
    CostBreakdown,
    CostCategory,
)
from optimization.solver import (
    SolverType,
    SolverStatus,
)


@pytest.mark.unit
class TestOptimizerConfigInitialization:
    """Tests for OptimizerConfig initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizerConfig()

        assert config.solver_type == SolverType.HIGHS
        assert config.time_limit_seconds == 300.0
        assert config.mip_gap == 0.01
        assert config.threads == 1
        assert config.random_seed == 42
        assert config.include_carbon_cost is True
        assert config.carbon_price_per_kg_co2e == Decimal("0.08")
        assert config.validate_blend_quality is True
        assert config.enable_provenance_tracking is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptimizerConfig(
            solver_type=SolverType.CBC,
            time_limit_seconds=600.0,
            mip_gap=0.005,
            threads=4,
            carbon_price_per_kg_co2e=Decimal("0.10"),
            include_risk_premium=True,
            risk_premium_pct=Decimal("3.0")
        )

        assert config.solver_type == SolverType.CBC
        assert config.time_limit_seconds == 600.0
        assert config.mip_gap == 0.005
        assert config.threads == 4
        assert config.carbon_price_per_kg_co2e == Decimal("0.10")
        assert config.include_risk_premium is True
        assert config.risk_premium_pct == Decimal("3.0")

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = OptimizerConfig(
            solver_type=SolverType.CPLEX,
            time_limit_seconds=120.0
        )
        result = config.to_dict()

        assert "solver_type" in result
        assert "time_limit_seconds" in result
        assert "carbon_price_per_kg_co2e" in result
        assert result["solver_type"] == "cplex"
        assert result["time_limit_seconds"] == 120.0

    def test_config_immutable_decimal(self):
        """Test that Decimal values are preserved."""
        config = OptimizerConfig(
            carbon_price_per_kg_co2e=Decimal("0.0741234567890123456789")
        )

        # Precision should be preserved
        assert config.carbon_price_per_kg_co2e == Decimal("0.0741234567890123456789")


@pytest.mark.unit
class TestFuelMixEntry:
    """Tests for FuelMixEntry data class."""

    def test_fuel_mix_entry_creation(self):
        """Test FuelMixEntry creation."""
        entry = FuelMixEntry(
            fuel_id="diesel",
            period=1,
            procurement_mj=Decimal("50000.00"),
            consumption_mj=Decimal("45000.00"),
            blend_fraction=Decimal("0.6000"),
            contract_allocation_mj=Decimal("30000.00"),
            spot_allocation_mj=Decimal("20000.00"),
            unit_cost_per_mj=Decimal("0.025000"),
            total_cost=Decimal("1125.00"),
            carbon_intensity_kg_co2e_mj=Decimal("0.074100"),
            emissions_kg_co2e=Decimal("3334.50")
        )

        assert entry.fuel_id == "diesel"
        assert entry.period == 1
        assert entry.blend_fraction == Decimal("0.6000")
        assert entry.total_cost == Decimal("1125.00")

    def test_fuel_mix_entry_to_dict(self):
        """Test FuelMixEntry serialization."""
        entry = FuelMixEntry(
            fuel_id="natural_gas",
            period=2,
            procurement_mj=Decimal("100000"),
            consumption_mj=Decimal("95000"),
            blend_fraction=Decimal("0.4"),
            contract_allocation_mj=Decimal("80000"),
            spot_allocation_mj=Decimal("20000"),
            unit_cost_per_mj=Decimal("0.0035"),
            total_cost=Decimal("332.50"),
            carbon_intensity_kg_co2e_mj=Decimal("0.0561"),
            emissions_kg_co2e=Decimal("5329.50")
        )

        result = entry.to_dict()

        assert result["fuel_id"] == "natural_gas"
        assert result["period"] == 2
        assert result["blend_fraction"] == "0.4"
        assert result["procurement_mj"] == "100000"
        assert isinstance(result["procurement_mj"], str)

    def test_fuel_mix_entry_allocations_sum(self):
        """Test that contract + spot allocations sum to procurement."""
        contract = Decimal("60000")
        spot = Decimal("40000")
        procurement = contract + spot

        entry = FuelMixEntry(
            fuel_id="hfo",
            period=1,
            procurement_mj=procurement,
            consumption_mj=Decimal("95000"),
            blend_fraction=Decimal("0.3"),
            contract_allocation_mj=contract,
            spot_allocation_mj=spot,
            unit_cost_per_mj=Decimal("0.028"),
            total_cost=Decimal("2660"),
            carbon_intensity_kg_co2e_mj=Decimal("0.0771"),
            emissions_kg_co2e=Decimal("7324.50")
        )

        assert entry.contract_allocation_mj + entry.spot_allocation_mj == entry.procurement_mj


@pytest.mark.unit
class TestInventoryProjection:
    """Tests for InventoryProjection data class."""

    def test_inventory_projection_creation(self):
        """Test InventoryProjection creation."""
        projection = InventoryProjection(
            tank_id="TANK-001",
            period=1,
            opening_level_mj=Decimal("100000"),
            inflow_mj=Decimal("50000"),
            outflow_mj=Decimal("45000"),
            losses_mj=Decimal("100"),
            closing_level_mj=Decimal("104900"),
            utilization_pct=Decimal("52.45"),
            is_at_minimum=False,
            is_near_maximum=False
        )

        assert projection.tank_id == "TANK-001"
        assert projection.closing_level_mj == Decimal("104900")
        assert projection.is_at_minimum is False

    def test_inventory_projection_balance_equation(self):
        """Test inventory balance: closing = opening + inflow - outflow - losses."""
        opening = Decimal("100000")
        inflow = Decimal("50000")
        outflow = Decimal("45000")
        losses = Decimal("100")
        closing = opening + inflow - outflow - losses

        projection = InventoryProjection(
            tank_id="TANK-002",
            period=3,
            opening_level_mj=opening,
            inflow_mj=inflow,
            outflow_mj=outflow,
            losses_mj=losses,
            closing_level_mj=closing,
            utilization_pct=Decimal("50.00"),
            is_at_minimum=False,
            is_near_maximum=False
        )

        calculated_closing = (
            projection.opening_level_mj +
            projection.inflow_mj -
            projection.outflow_mj -
            projection.losses_mj
        )
        assert calculated_closing == projection.closing_level_mj

    def test_inventory_projection_to_dict(self):
        """Test InventoryProjection serialization."""
        projection = InventoryProjection(
            tank_id="TANK-003",
            period=5,
            opening_level_mj=Decimal("250000"),
            inflow_mj=Decimal("75000"),
            outflow_mj=Decimal("80000"),
            losses_mj=Decimal("250"),
            closing_level_mj=Decimal("244750"),
            utilization_pct=Decimal("48.95"),
            is_at_minimum=True,
            is_near_maximum=False
        )

        result = projection.to_dict()

        assert result["tank_id"] == "TANK-003"
        assert result["period"] == 5
        assert result["is_at_minimum"] is True
        assert result["is_near_maximum"] is False


@pytest.mark.unit
class TestBlendQuality:
    """Tests for BlendQuality data class."""

    def test_blend_quality_compliant(self):
        """Test compliant blend quality."""
        quality = BlendQuality(
            period=1,
            sulfur_wt_pct=Decimal("0.35"),
            ash_wt_pct=Decimal("0.05"),
            water_vol_pct=Decimal("0.10"),
            viscosity_cst=Decimal("180"),
            flash_point_c=Decimal("72"),
            carbon_intensity_kg_co2e_mj=Decimal("0.068"),
            status=BlendQualityStatus.COMPLIANT,
            violations=[]
        )

        assert quality.status == BlendQualityStatus.COMPLIANT
        assert len(quality.violations) == 0
        assert quality.sulfur_wt_pct < Decimal("0.50")

    def test_blend_quality_violation(self):
        """Test blend quality with violations."""
        quality = BlendQuality(
            period=1,
            sulfur_wt_pct=Decimal("0.55"),
            ash_wt_pct=Decimal("0.05"),
            water_vol_pct=Decimal("0.10"),
            viscosity_cst=Decimal("180"),
            flash_point_c=Decimal("55"),
            carbon_intensity_kg_co2e_mj=Decimal("0.068"),
            status=BlendQualityStatus.VIOLATION,
            violations=[
                "Sulfur 0.55% exceeds limit 0.50%",
                "Flash point 55C below minimum 60C"
            ]
        )

        assert quality.status == BlendQualityStatus.VIOLATION
        assert len(quality.violations) == 2
        assert "Sulfur" in quality.violations[0]

    def test_blend_quality_warning(self):
        """Test blend quality with warning status."""
        quality = BlendQuality(
            period=3,
            sulfur_wt_pct=Decimal("0.48"),
            ash_wt_pct=Decimal("0.09"),
            water_vol_pct=Decimal("0.45"),
            viscosity_cst=Decimal("650"),
            flash_point_c=Decimal("62"),
            carbon_intensity_kg_co2e_mj=Decimal("0.072"),
            status=BlendQualityStatus.WARNING,
            violations=[]
        )

        assert quality.status == BlendQualityStatus.WARNING

    def test_blend_quality_to_dict(self):
        """Test BlendQuality serialization."""
        quality = BlendQuality(
            period=2,
            sulfur_wt_pct=Decimal("0.40"),
            ash_wt_pct=Decimal("0.06"),
            water_vol_pct=Decimal("0.15"),
            viscosity_cst=Decimal("200"),
            flash_point_c=Decimal("68"),
            carbon_intensity_kg_co2e_mj=Decimal("0.065"),
            status=BlendQualityStatus.COMPLIANT,
            violations=[]
        )

        result = quality.to_dict()

        assert result["period"] == 2
        assert result["status"] == "compliant"
        assert result["violations"] == []


@pytest.mark.unit
class TestProcurementSchedule:
    """Tests for ProcurementSchedule data class."""

    def test_procurement_schedule_contract(self):
        """Test contract procurement schedule entry."""
        schedule = ProcurementSchedule(
            fuel_id="diesel",
            period=1,
            quantity_mj=Decimal("50000"),
            source="contract",
            contract_id="CONTRACT-TOP-001",
            delivery_date=None,
            price_per_mj=Decimal("0.0238"),
            total_cost=Decimal("1190")
        )

        assert schedule.source == "contract"
        assert schedule.contract_id == "CONTRACT-TOP-001"
        assert schedule.fuel_id == "diesel"

    def test_procurement_schedule_spot(self):
        """Test spot procurement schedule entry."""
        schedule = ProcurementSchedule(
            fuel_id="hfo",
            period=2,
            quantity_mj=Decimal("20000"),
            source="spot",
            contract_id=None,
            delivery_date=None,
            price_per_mj=Decimal("0.028"),
            total_cost=Decimal("560")
        )

        assert schedule.source == "spot"
        assert schedule.contract_id is None

    def test_procurement_schedule_to_dict(self):
        """Test ProcurementSchedule serialization."""
        delivery = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        schedule = ProcurementSchedule(
            fuel_id="natural_gas",
            period=1,
            quantity_mj=Decimal("80000"),
            source="contract",
            contract_id="NG-CONTRACT-001",
            delivery_date=delivery,
            price_per_mj=Decimal("0.0035"),
            total_cost=Decimal("280")
        )

        result = schedule.to_dict()

        assert result["fuel_id"] == "natural_gas"
        assert result["source"] == "contract"
        assert result["contract_id"] == "NG-CONTRACT-001"
        assert result["delivery_date"] == delivery.isoformat()


@pytest.mark.unit
class TestOptimizationResult:
    """Tests for OptimizationResult data class."""

    @pytest.fixture
    def sample_cost_breakdown(self):
        """Create sample cost breakdown."""
        return CostBreakdown(
            purchase_cost=Decimal("10000"),
            logistics_cost=Decimal("500"),
            storage_cost=Decimal("100"),
            loss_cost=Decimal("50"),
            penalty_cost=Decimal("0"),
            carbon_cost=Decimal("800"),
            risk_cost=Decimal("200"),
            total_cost=Decimal("11650"),
            components=[],
            cost_by_fuel={},
            cost_by_period={}
        )

    def test_successful_result(self, sample_cost_breakdown):
        """Test successful optimization result."""
        result = OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            solver_status=SolverStatus.OPTIMAL,
            objective_value=Decimal("11650"),
            mip_gap=0.0,
            is_optimal=True,
            is_feasible=True,
            fuel_mix=[],
            cost_breakdown=sample_cost_breakdown,
            procurement_schedule=[],
            inventory_projections=[],
            blend_quality=[],
            total_procurement_mj=Decimal("500000"),
            total_consumption_mj=Decimal("480000"),
            total_cost=Decimal("11650"),
            average_cost_per_mj=Decimal("0.02427"),
            total_emissions_kg_co2e=Decimal("35568"),
            average_carbon_intensity=Decimal("0.0741"),
            fuel_summary={},
            solve_time_seconds=1.5,
            model_build_time_seconds=0.2,
            total_time_seconds=1.7,
            run_id="OPT-TEST123ABC",
            run_bundle_hash="",
            model_hash="abc123def456",
            solver_config_hash="ghi789jkl012",
            input_data_hash="mno345pqr678"
        )

        assert result.status == OptimizationStatus.SUCCESS
        assert result.is_optimal is True
        assert result.is_feasible is True
        assert result.run_bundle_hash != ""  # Computed in __post_init__

    def test_infeasible_result(self, sample_cost_breakdown):
        """Test infeasible optimization result."""
        zero_breakdown = CostBreakdown(
            purchase_cost=Decimal("0"),
            logistics_cost=Decimal("0"),
            storage_cost=Decimal("0"),
            loss_cost=Decimal("0"),
            penalty_cost=Decimal("0"),
            carbon_cost=Decimal("0"),
            risk_cost=Decimal("0"),
            total_cost=Decimal("0"),
            components=[],
            cost_by_fuel={},
            cost_by_period={}
        )

        result = OptimizationResult(
            status=OptimizationStatus.INFEASIBLE,
            solver_status=SolverStatus.INFEASIBLE,
            objective_value=Decimal("0"),
            mip_gap=None,
            is_optimal=False,
            is_feasible=False,
            fuel_mix=[],
            cost_breakdown=zero_breakdown,
            procurement_schedule=[],
            inventory_projections=[],
            blend_quality=[],
            total_procurement_mj=Decimal("0"),
            total_consumption_mj=Decimal("0"),
            total_cost=Decimal("0"),
            average_cost_per_mj=Decimal("0"),
            total_emissions_kg_co2e=Decimal("0"),
            average_carbon_intensity=Decimal("0"),
            fuel_summary={},
            solve_time_seconds=0.5,
            model_build_time_seconds=0.2,
            total_time_seconds=0.7,
            run_id="OPT-FAIL123",
            run_bundle_hash="",
            model_hash="abc123",
            solver_config_hash="def456",
            input_data_hash="ghi789"
        )

        assert result.status == OptimizationStatus.INFEASIBLE
        assert result.is_feasible is False
        assert result.is_optimal is False

    def test_result_to_dict(self, sample_cost_breakdown):
        """Test result serialization."""
        result = OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            solver_status=SolverStatus.OPTIMAL,
            objective_value=Decimal("11650"),
            mip_gap=0.001,
            is_optimal=True,
            is_feasible=True,
            fuel_mix=[],
            cost_breakdown=sample_cost_breakdown,
            procurement_schedule=[],
            inventory_projections=[],
            blend_quality=[],
            total_procurement_mj=Decimal("500000"),
            total_consumption_mj=Decimal("480000"),
            total_cost=Decimal("11650"),
            average_cost_per_mj=Decimal("0.02427"),
            total_emissions_kg_co2e=Decimal("35568"),
            average_carbon_intensity=Decimal("0.0741"),
            fuel_summary={},
            solve_time_seconds=1.5,
            model_build_time_seconds=0.2,
            total_time_seconds=1.7,
            run_id="OPT-TEST123",
            run_bundle_hash="",
            model_hash="abc123",
            solver_config_hash="def456",
            input_data_hash="ghi789"
        )

        data = result.to_dict()

        assert data["status"] == "success"
        assert data["solver_status"] == "optimal"
        assert "run_bundle_hash" in data
        assert data["run_id"] == "OPT-TEST123"
        assert "cost_breakdown" in data

    def test_result_summary(self, sample_cost_breakdown):
        """Test human-readable summary generation."""
        result = OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            solver_status=SolverStatus.OPTIMAL,
            objective_value=Decimal("11650"),
            mip_gap=0.001,
            is_optimal=True,
            is_feasible=True,
            fuel_mix=[],
            cost_breakdown=sample_cost_breakdown,
            procurement_schedule=[],
            inventory_projections=[],
            blend_quality=[],
            total_procurement_mj=Decimal("500000"),
            total_consumption_mj=Decimal("480000"),
            total_cost=Decimal("11650"),
            average_cost_per_mj=Decimal("0.02427"),
            total_emissions_kg_co2e=Decimal("35568"),
            average_carbon_intensity=Decimal("0.0741"),
            fuel_summary={"diesel": {"blend_fraction_avg": "0.6"}},
            solve_time_seconds=1.5,
            model_build_time_seconds=0.2,
            total_time_seconds=1.7,
            run_id="OPT-TEST123",
            run_bundle_hash="",
            model_hash="abc123",
            solver_config_hash="def456",
            input_data_hash="ghi789"
        )

        summary = result.get_summary()

        assert "FUEL MIX OPTIMIZATION RESULTS" in summary
        assert "SUCCESS" in summary
        assert "Total Cost" in summary
        assert "PROVENANCE" in summary
        assert "diesel" in summary


@pytest.mark.unit
class TestFuelMixOptimizerInit:
    """Tests for FuelMixOptimizer initialization."""

    def test_default_initialization(self):
        """Test default optimizer initialization."""
        config = OptimizerConfig()
        optimizer = FuelMixOptimizer(config)

        assert optimizer._config == config
        assert optimizer._cost_model is not None
        assert optimizer._run_id is None
        assert optimizer._model is None
        assert optimizer._solver is None

    def test_custom_initialization(self):
        """Test optimizer with custom config."""
        config = OptimizerConfig(
            solver_type=SolverType.CBC,
            time_limit_seconds=600.0,
            include_carbon_cost=False
        )
        optimizer = FuelMixOptimizer(config)

        assert optimizer._config.solver_type == SolverType.CBC
        assert optimizer._config.include_carbon_cost is False


@pytest.mark.unit
class TestProvenanceTracking:
    """Tests for provenance and reproducibility features."""

    @pytest.fixture
    def sample_cost_breakdown(self):
        """Create sample cost breakdown."""
        return CostBreakdown(
            purchase_cost=Decimal("10000"),
            logistics_cost=Decimal("500"),
            storage_cost=Decimal("100"),
            loss_cost=Decimal("50"),
            penalty_cost=Decimal("0"),
            carbon_cost=Decimal("800"),
            risk_cost=Decimal("200"),
            total_cost=Decimal("11650"),
            components=[],
            cost_by_fuel={},
            cost_by_period={}
        )

    def test_run_bundle_hash_uniqueness(self, sample_cost_breakdown):
        """Test that different inputs produce different hashes."""
        result1 = OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            solver_status=SolverStatus.OPTIMAL,
            objective_value=Decimal("11650"),
            mip_gap=0.001,
            is_optimal=True,
            is_feasible=True,
            fuel_mix=[],
            cost_breakdown=sample_cost_breakdown,
            procurement_schedule=[],
            inventory_projections=[],
            blend_quality=[],
            total_procurement_mj=Decimal("500000"),
            total_consumption_mj=Decimal("480000"),
            total_cost=Decimal("11650"),
            average_cost_per_mj=Decimal("0.02427"),
            total_emissions_kg_co2e=Decimal("35568"),
            average_carbon_intensity=Decimal("0.0741"),
            fuel_summary={},
            solve_time_seconds=1.5,
            model_build_time_seconds=0.2,
            total_time_seconds=1.7,
            run_id="OPT-TEST001",
            run_bundle_hash="",
            model_hash="abc123",
            solver_config_hash="def456",
            input_data_hash="ghi789"
        )

        result2 = OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            solver_status=SolverStatus.OPTIMAL,
            objective_value=Decimal("12000"),  # Different value
            mip_gap=0.001,
            is_optimal=True,
            is_feasible=True,
            fuel_mix=[],
            cost_breakdown=sample_cost_breakdown,
            procurement_schedule=[],
            inventory_projections=[],
            blend_quality=[],
            total_procurement_mj=Decimal("500000"),
            total_consumption_mj=Decimal("480000"),
            total_cost=Decimal("12000"),
            average_cost_per_mj=Decimal("0.025"),
            total_emissions_kg_co2e=Decimal("35568"),
            average_carbon_intensity=Decimal("0.0741"),
            fuel_summary={},
            solve_time_seconds=1.5,
            model_build_time_seconds=0.2,
            total_time_seconds=1.7,
            run_id="OPT-TEST002",
            run_bundle_hash="",
            model_hash="abc123",
            solver_config_hash="def456",
            input_data_hash="ghi789"
        )

        # Hashes should be different
        assert result1.run_bundle_hash != result2.run_bundle_hash

    def test_run_bundle_hash_format(self, sample_cost_breakdown):
        """Test that run bundle hash is valid SHA-256."""
        result = OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            solver_status=SolverStatus.OPTIMAL,
            objective_value=Decimal("0"),
            mip_gap=None,
            is_optimal=True,
            is_feasible=True,
            fuel_mix=[],
            cost_breakdown=sample_cost_breakdown,
            procurement_schedule=[],
            inventory_projections=[],
            blend_quality=[],
            total_procurement_mj=Decimal("0"),
            total_consumption_mj=Decimal("0"),
            total_cost=Decimal("0"),
            average_cost_per_mj=Decimal("0"),
            total_emissions_kg_co2e=Decimal("0"),
            average_carbon_intensity=Decimal("0"),
            fuel_summary={},
            solve_time_seconds=0.0,
            model_build_time_seconds=0.0,
            total_time_seconds=0.0,
            run_id="OPT-TEST",
            run_bundle_hash="",
            model_hash="abc",
            solver_config_hash="def",
            input_data_hash="ghi"
        )

        # SHA-256 produces 64 hex characters
        assert len(result.run_bundle_hash) == 64
        # Should be valid hex
        assert all(c in '0123456789abcdef' for c in result.run_bundle_hash)


@pytest.mark.unit
class TestOptimizationStatus:
    """Tests for OptimizationStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert OptimizationStatus.SUCCESS.value == "success"
        assert OptimizationStatus.INFEASIBLE.value == "infeasible"
        assert OptimizationStatus.UNBOUNDED.value == "unbounded"
        assert OptimizationStatus.TIME_LIMIT.value == "time_limit"
        assert OptimizationStatus.ERROR.value == "error"

    def test_status_comparison(self):
        """Test status enum comparison."""
        assert OptimizationStatus.SUCCESS != OptimizationStatus.INFEASIBLE
        assert OptimizationStatus.SUCCESS == OptimizationStatus.SUCCESS


@pytest.mark.unit
class TestBlendQualityStatus:
    """Tests for BlendQualityStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert BlendQualityStatus.COMPLIANT.value == "compliant"
        assert BlendQualityStatus.VIOLATION.value == "violation"
        assert BlendQualityStatus.WARNING.value == "warning"


@pytest.mark.unit
class TestOptimizerGetters:
    """Tests for FuelMixOptimizer getter methods."""

    def test_get_run_id_before_optimization(self):
        """Test get_run_id returns None before optimization."""
        config = OptimizerConfig()
        optimizer = FuelMixOptimizer(config)

        assert optimizer.get_run_id() is None

    def test_get_solution_before_optimization(self):
        """Test get_solution returns None before optimization."""
        config = OptimizerConfig()
        optimizer = FuelMixOptimizer(config)

        assert optimizer.get_solution() is None

    def test_get_model_statistics_before_optimization(self):
        """Test get_model_statistics returns None before optimization."""
        config = OptimizerConfig()
        optimizer = FuelMixOptimizer(config)

        assert optimizer.get_model_statistics() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
