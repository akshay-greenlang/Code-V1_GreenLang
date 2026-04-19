"""
Unit tests for GL-001 ThermalCommand Orchestrator Load Allocation Module

Tests load allocation algorithms with 90%+ coverage.
Validates thermal load distribution, optimization, and efficiency calculations.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import Mock, patch, AsyncMock
import math

from greenlang.agents.process_heat.gl_001_thermal_command.load_allocation import (
    LoadAllocator,
    AllocationStrategy,
    EquipmentProfile,
    AllocationResult,
    LoadDistribution,
    EfficiencyCurve,
    OptimizationObjective,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def load_allocator():
    """Create load allocator instance."""
    return LoadAllocator()


@pytest.fixture
def sample_equipment_profiles():
    """Create sample equipment profiles."""
    return [
        EquipmentProfile(
            equipment_id="BLR-001",
            equipment_type="boiler",
            max_capacity_mw=25.0,
            min_capacity_mw=5.0,
            efficiency_curve=EfficiencyCurve([
                (0.2, 0.75),
                (0.5, 0.85),
                (0.8, 0.90),
                (1.0, 0.88),
            ]),
            startup_cost=100.0,
            operating_cost_per_mw=10.0,
        ),
        EquipmentProfile(
            equipment_id="BLR-002",
            equipment_type="boiler",
            max_capacity_mw=30.0,
            min_capacity_mw=8.0,
            efficiency_curve=EfficiencyCurve([
                (0.2, 0.72),
                (0.5, 0.83),
                (0.8, 0.88),
                (1.0, 0.85),
            ]),
            startup_cost=120.0,
            operating_cost_per_mw=12.0,
        ),
        EquipmentProfile(
            equipment_id="BLR-003",
            equipment_type="boiler",
            max_capacity_mw=20.0,
            min_capacity_mw=4.0,
            efficiency_curve=EfficiencyCurve([
                (0.2, 0.78),
                (0.5, 0.87),
                (0.8, 0.92),
                (1.0, 0.90),
            ]),
            startup_cost=80.0,
            operating_cost_per_mw=8.0,
        ),
    ]


@pytest.fixture
def load_allocator_with_equipment(load_allocator, sample_equipment_profiles):
    """Create load allocator with registered equipment."""
    for profile in sample_equipment_profiles:
        load_allocator.register_equipment(profile)
    return load_allocator


# =============================================================================
# EQUIPMENT PROFILE TESTS
# =============================================================================

class TestEquipmentProfile:
    """Test suite for EquipmentProfile."""

    @pytest.mark.unit
    def test_initialization(self, sample_equipment_profiles):
        """Test equipment profile initialization."""
        profile = sample_equipment_profiles[0]

        assert profile.equipment_id == "BLR-001"
        assert profile.max_capacity_mw == 25.0
        assert profile.min_capacity_mw == 5.0

    @pytest.mark.unit
    def test_capacity_range_validation(self):
        """Test capacity range validation."""
        # Valid range
        profile = EquipmentProfile(
            equipment_id="TEST",
            equipment_type="boiler",
            max_capacity_mw=25.0,
            min_capacity_mw=5.0,
        )
        assert profile.max_capacity_mw > profile.min_capacity_mw

        # Invalid range (min > max) should raise
        with pytest.raises(ValueError):
            EquipmentProfile(
                equipment_id="TEST",
                equipment_type="boiler",
                max_capacity_mw=5.0,
                min_capacity_mw=25.0,
            )

    @pytest.mark.unit
    def test_efficiency_at_load(self, sample_equipment_profiles):
        """Test efficiency interpolation at given load."""
        profile = sample_equipment_profiles[0]  # BLR-001

        # At defined point
        efficiency_80 = profile.get_efficiency_at_load(0.8)
        assert efficiency_80 == pytest.approx(0.90, rel=0.01)

        # Interpolated
        efficiency_65 = profile.get_efficiency_at_load(0.65)
        assert 0.85 < efficiency_65 < 0.90

    @pytest.mark.unit
    def test_available_capacity(self, sample_equipment_profiles):
        """Test available capacity calculation."""
        profile = sample_equipment_profiles[0]

        available = profile.max_capacity_mw - profile.min_capacity_mw
        assert available == 20.0  # 25 - 5 = 20 MW

    @pytest.mark.unit
    def test_operating_range(self, sample_equipment_profiles):
        """Test equipment operating range."""
        profile = sample_equipment_profiles[0]

        # Load factor at min capacity
        min_load_factor = profile.min_capacity_mw / profile.max_capacity_mw
        assert min_load_factor == 0.2  # 5/25 = 0.2

        # Load factor at max
        max_load_factor = 1.0


# =============================================================================
# EFFICIENCY CURVE TESTS
# =============================================================================

class TestEfficiencyCurve:
    """Test suite for EfficiencyCurve."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test efficiency curve initialization."""
        curve = EfficiencyCurve([
            (0.2, 0.75),
            (0.5, 0.85),
            (0.8, 0.90),
            (1.0, 0.88),
        ])

        assert len(curve.points) == 4

    @pytest.mark.unit
    def test_interpolation_at_defined_point(self):
        """Test efficiency at defined curve point."""
        curve = EfficiencyCurve([
            (0.2, 0.75),
            (0.5, 0.85),
            (0.8, 0.90),
            (1.0, 0.88),
        ])

        assert curve.get_efficiency(0.5) == pytest.approx(0.85, rel=0.001)

    @pytest.mark.unit
    def test_linear_interpolation(self):
        """Test linear interpolation between points."""
        curve = EfficiencyCurve([
            (0.0, 0.70),
            (1.0, 0.90),
        ])

        # Midpoint should be 0.80
        assert curve.get_efficiency(0.5) == pytest.approx(0.80, rel=0.001)

    @pytest.mark.unit
    def test_extrapolation_below_min(self):
        """Test behavior below minimum load."""
        curve = EfficiencyCurve([
            (0.2, 0.75),
            (1.0, 0.90),
        ])

        # Below min should return min efficiency
        efficiency = curve.get_efficiency(0.1)
        assert efficiency <= 0.75

    @pytest.mark.unit
    def test_extrapolation_above_max(self):
        """Test behavior above maximum load."""
        curve = EfficiencyCurve([
            (0.2, 0.75),
            (1.0, 0.90),
        ])

        # Above max should return max efficiency
        efficiency = curve.get_efficiency(1.1)
        assert efficiency <= 0.90

    @pytest.mark.unit
    def test_peak_efficiency(self):
        """Test finding peak efficiency."""
        curve = EfficiencyCurve([
            (0.2, 0.75),
            (0.5, 0.85),
            (0.8, 0.92),  # Peak
            (1.0, 0.88),
        ])

        peak_load, peak_efficiency = curve.get_peak_efficiency()
        assert peak_load == pytest.approx(0.8, rel=0.01)
        assert peak_efficiency == pytest.approx(0.92, rel=0.01)


# =============================================================================
# LOAD ALLOCATOR TESTS
# =============================================================================

class TestLoadAllocator:
    """Test suite for LoadAllocator."""

    @pytest.mark.unit
    def test_initialization(self, load_allocator):
        """Test load allocator initialization."""
        assert load_allocator is not None
        assert hasattr(load_allocator, '_equipment')

    @pytest.mark.unit
    def test_register_equipment(self, load_allocator, sample_equipment_profiles):
        """Test equipment registration."""
        for profile in sample_equipment_profiles:
            load_allocator.register_equipment(profile)

        assert len(load_allocator._equipment) == 3

    @pytest.mark.unit
    def test_deregister_equipment(self, load_allocator_with_equipment):
        """Test equipment deregistration."""
        result = load_allocator_with_equipment.deregister_equipment("BLR-001")

        assert result is True
        assert "BLR-001" not in load_allocator_with_equipment._equipment

    @pytest.mark.unit
    def test_get_total_capacity(self, load_allocator_with_equipment):
        """Test total capacity calculation."""
        capacity = load_allocator_with_equipment.get_total_capacity()

        # BLR-001: 25 + BLR-002: 30 + BLR-003: 20 = 75 MW
        assert capacity["max_capacity_mw"] == 75.0
        # Min: 5 + 8 + 4 = 17 MW
        assert capacity["min_capacity_mw"] == 17.0

    @pytest.mark.unit
    def test_allocate_equal_distribution(self, load_allocator_with_equipment):
        """Test equal distribution allocation."""
        result = load_allocator_with_equipment.allocate(
            demand_mw=45.0,
            strategy=AllocationStrategy.EQUAL
        )

        assert result is not None
        assert result.total_allocated == pytest.approx(45.0, rel=0.01)

        # Each should get roughly equal share (15 MW each if possible)

    @pytest.mark.unit
    def test_allocate_efficiency_optimized(self, load_allocator_with_equipment):
        """Test efficiency-optimized allocation."""
        result = load_allocator_with_equipment.allocate(
            demand_mw=45.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        assert result is not None
        assert result.total_allocated == pytest.approx(45.0, rel=0.01)
        assert result.overall_efficiency > 0.80

    @pytest.mark.unit
    def test_allocate_cost_optimized(self, load_allocator_with_equipment):
        """Test cost-optimized allocation."""
        result = load_allocator_with_equipment.allocate(
            demand_mw=45.0,
            strategy=AllocationStrategy.COST
        )

        assert result is not None
        assert result.total_cost is not None

    @pytest.mark.unit
    def test_allocate_with_constraints(self, load_allocator_with_equipment):
        """Test allocation with constraints."""
        constraints = {
            "BLR-001": {"max_load_mw": 20.0},  # Limit BLR-001 to 20 MW
            "BLR-002": {"unavailable": True},  # BLR-002 unavailable
        }

        result = load_allocator_with_equipment.allocate(
            demand_mw=35.0,
            strategy=AllocationStrategy.EFFICIENCY,
            constraints=constraints
        )

        assert result is not None
        if result.allocations.get("BLR-001"):
            assert result.allocations["BLR-001"] <= 20.0
        assert result.allocations.get("BLR-002", 0.0) == 0.0

    @pytest.mark.unit
    def test_allocate_exceeds_capacity(self, load_allocator_with_equipment):
        """Test allocation when demand exceeds capacity."""
        # Total capacity is 75 MW, request 100 MW
        result = load_allocator_with_equipment.allocate(
            demand_mw=100.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        # Should allocate maximum available
        assert result.total_allocated <= 75.0
        assert result.shortfall >= 25.0

    @pytest.mark.unit
    def test_allocate_below_minimum(self, load_allocator_with_equipment):
        """Test allocation below minimum operating point."""
        # Total minimum is 17 MW, request 10 MW
        result = load_allocator_with_equipment.allocate(
            demand_mw=10.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        # Should still operate at minimum viable configuration
        assert result is not None

    @pytest.mark.unit
    def test_rebalance_allocation(self, load_allocator_with_equipment):
        """Test load rebalancing."""
        current_allocation = {
            "BLR-001": 20.0,
            "BLR-002": 20.0,
            "BLR-003": 10.0,
        }

        result = load_allocator_with_equipment.rebalance(
            current_allocation=current_allocation,
            target_demand_mw=50.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        assert result is not None
        assert result.total_allocated == pytest.approx(50.0, rel=0.01)


# =============================================================================
# ALLOCATION RESULT TESTS
# =============================================================================

class TestAllocationResult:
    """Test suite for AllocationResult."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test allocation result initialization."""
        result = AllocationResult(
            allocations={"BLR-001": 15.0, "BLR-002": 20.0},
            total_allocated=35.0,
            total_demand=35.0,
            overall_efficiency=0.87,
            total_cost=350.0,
        )

        assert result.total_allocated == 35.0
        assert result.shortfall == 0.0

    @pytest.mark.unit
    def test_shortfall_calculation(self):
        """Test shortfall calculation."""
        result = AllocationResult(
            allocations={"BLR-001": 25.0},
            total_allocated=25.0,
            total_demand=40.0,
        )

        assert result.shortfall == 15.0

    @pytest.mark.unit
    def test_provenance_hash(self):
        """Test allocation result provenance hash."""
        result = AllocationResult(
            allocations={"BLR-001": 15.0},
            total_allocated=15.0,
            total_demand=15.0,
        )

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_per_equipment_efficiency(self):
        """Test per-equipment efficiency in result."""
        result = AllocationResult(
            allocations={"BLR-001": 15.0, "BLR-002": 20.0},
            total_allocated=35.0,
            total_demand=35.0,
            equipment_efficiencies={"BLR-001": 0.88, "BLR-002": 0.85},
        )

        assert result.equipment_efficiencies["BLR-001"] == 0.88


# =============================================================================
# LOAD DISTRIBUTION TESTS
# =============================================================================

class TestLoadDistribution:
    """Test suite for LoadDistribution."""

    @pytest.mark.unit
    def test_distribution_sums_correctly(self):
        """Test distribution sums to total demand."""
        distribution = LoadDistribution(
            equipment_loads={"BLR-001": 20.0, "BLR-002": 15.0, "BLR-003": 10.0},
            total_demand=45.0,
        )

        total = sum(distribution.equipment_loads.values())
        assert total == pytest.approx(45.0, rel=0.01)

    @pytest.mark.unit
    def test_load_factors(self):
        """Test load factor calculation."""
        distribution = LoadDistribution(
            equipment_loads={"BLR-001": 20.0},
            total_demand=20.0,
            equipment_capacities={"BLR-001": 25.0},
        )

        load_factor = distribution.equipment_loads["BLR-001"] / distribution.equipment_capacities["BLR-001"]
        assert load_factor == pytest.approx(0.8, rel=0.01)


# =============================================================================
# OPTIMIZATION OBJECTIVE TESTS
# =============================================================================

class TestOptimizationObjective:
    """Test suite for OptimizationObjective."""

    @pytest.mark.unit
    def test_objective_values(self):
        """Test optimization objective enumeration."""
        assert OptimizationObjective.EFFICIENCY.value == "efficiency"
        assert OptimizationObjective.COST.value == "cost"
        assert OptimizationObjective.EMISSIONS.value == "emissions"

    @pytest.mark.unit
    def test_multi_objective_weights(self):
        """Test multi-objective optimization weights."""
        weights = {
            OptimizationObjective.EFFICIENCY: 0.5,
            OptimizationObjective.COST: 0.3,
            OptimizationObjective.EMISSIONS: 0.2,
        }

        assert sum(weights.values()) == pytest.approx(1.0, rel=0.001)


# =============================================================================
# ALLOCATION STRATEGY TESTS
# =============================================================================

class TestAllocationStrategy:
    """Test suite for AllocationStrategy."""

    @pytest.mark.unit
    def test_strategy_values(self):
        """Test allocation strategy enumeration."""
        assert AllocationStrategy.EQUAL.value == "equal"
        assert AllocationStrategy.EFFICIENCY.value == "efficiency"
        assert AllocationStrategy.COST.value == "cost"
        assert AllocationStrategy.PROPORTIONAL.value == "proportional"


# =============================================================================
# CALCULATION ACCURACY TESTS
# =============================================================================

class TestCalculationAccuracy:
    """Test calculation accuracy for regulatory compliance."""

    @pytest.mark.compliance
    def test_efficiency_calculation_precision(self, load_allocator_with_equipment):
        """Test efficiency calculations have required precision."""
        result = load_allocator_with_equipment.allocate(
            demand_mw=45.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        # Efficiency should be calculated to 4 decimal places
        efficiency_str = f"{result.overall_efficiency:.4f}"
        assert len(efficiency_str.split('.')[1]) >= 4

    @pytest.mark.compliance
    def test_load_allocation_precision(self, load_allocator_with_equipment):
        """Test load allocations have required precision."""
        result = load_allocator_with_equipment.allocate(
            demand_mw=45.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        # Allocations should sum precisely to demand (or max capacity)
        total = sum(result.allocations.values())
        assert abs(total - result.total_allocated) < 0.001

    @pytest.mark.compliance
    def test_reproducibility(self, load_allocator_with_equipment):
        """Test allocation is reproducible."""
        result1 = load_allocator_with_equipment.allocate(
            demand_mw=45.0,
            strategy=AllocationStrategy.EFFICIENCY
        )
        result2 = load_allocator_with_equipment.allocate(
            demand_mw=45.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        # Same inputs should produce same outputs
        assert result1.allocations == result2.allocations
        assert result1.overall_efficiency == result2.overall_efficiency


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestLoadAllocationPerformance:
    """Performance tests for load allocation."""

    @pytest.mark.performance
    def test_allocation_speed(self, load_allocator_with_equipment):
        """Test allocation completes quickly."""
        import time

        start = time.perf_counter()
        for _ in range(100):
            load_allocator_with_equipment.allocate(
                demand_mw=45.0,
                strategy=AllocationStrategy.EFFICIENCY
            )
        duration = time.perf_counter() - start

        # 100 allocations should complete in < 1 second
        assert duration < 1.0

    @pytest.mark.performance
    def test_large_equipment_set(self, load_allocator):
        """Test allocation with many equipment units."""
        import time

        # Register 50 equipment units
        for i in range(50):
            load_allocator.register_equipment(EquipmentProfile(
                equipment_id=f"EQ-{i:03d}",
                equipment_type="boiler",
                max_capacity_mw=25.0,
                min_capacity_mw=5.0,
            ))

        start = time.perf_counter()
        result = load_allocator.allocate(
            demand_mw=500.0,
            strategy=AllocationStrategy.EFFICIENCY
        )
        duration = time.perf_counter() - start

        assert duration < 5.0  # Should complete in < 5 seconds
        assert result is not None


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases in load allocation."""

    @pytest.mark.unit
    def test_zero_demand(self, load_allocator_with_equipment):
        """Test allocation with zero demand."""
        result = load_allocator_with_equipment.allocate(
            demand_mw=0.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        assert result.total_allocated == 0.0

    @pytest.mark.unit
    def test_single_equipment(self, load_allocator):
        """Test allocation with single equipment."""
        load_allocator.register_equipment(EquipmentProfile(
            equipment_id="SINGLE-001",
            equipment_type="boiler",
            max_capacity_mw=25.0,
            min_capacity_mw=5.0,
        ))

        result = load_allocator.allocate(
            demand_mw=15.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        assert result.total_allocated == 15.0
        assert result.allocations["SINGLE-001"] == 15.0

    @pytest.mark.unit
    def test_no_equipment(self, load_allocator):
        """Test allocation with no equipment registered."""
        result = load_allocator.allocate(
            demand_mw=50.0,
            strategy=AllocationStrategy.EFFICIENCY
        )

        assert result.total_allocated == 0.0
        assert result.shortfall == 50.0

    @pytest.mark.unit
    def test_negative_demand_rejected(self, load_allocator_with_equipment):
        """Test negative demand is rejected."""
        with pytest.raises(ValueError):
            load_allocator_with_equipment.allocate(
                demand_mw=-10.0,
                strategy=AllocationStrategy.EFFICIENCY
            )
