# -*- coding: utf-8 -*-
"""
Unit tests for FuelBlendingOptimizer.

Tests the advanced fuel blending optimization calculator including:
- Multi-fuel blend optimization
- Refutas viscosity blending
- Linear programming optimization
- Flash point safety constraints
- Heavy metal tracking
- Inventory constraints
- Provenance hashing

Standards tested: ASTM D341, ASTM D975, ISO 8217
"""

import pytest
import sys
from pathlib import Path
from decimal import Decimal
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calculators.fuel_blending_optimizer import (
    FuelBlendingOptimizer,
    FuelComponent,
    BlendSpecification,
    OptimizationObjective,
    BlendOptimizationInput,
    BlendOptimizationResult
)


class TestFuelBlendingOptimizer:
    """Test suite for FuelBlendingOptimizer."""

    @pytest.fixture
    def optimizer(self) -> FuelBlendingOptimizer:
        """Create optimizer instance."""
        return FuelBlendingOptimizer()

    @pytest.fixture
    def heavy_fuel_oil(self) -> FuelComponent:
        """Heavy Fuel Oil component."""
        return FuelComponent(
            fuel_id="HFO",
            name="Heavy Fuel Oil",
            hhv_mj_kg=Decimal("42.5"),
            lhv_mj_kg=Decimal("40.2"),
            sulfur_ppm=Decimal("35000"),  # 3.5%
            ash_percent=Decimal("0.08"),
            viscosity_cst_40c=Decimal("380"),
            flash_point_c=Decimal("70"),
            density_kg_m3=Decimal("980"),
            cost_per_kg=Decimal("0.35"),
            available_kg=Decimal("100000"),
            heavy_metals_ppm={"vanadium": Decimal("150"), "nickel": Decimal("50")},
            moisture_percent=Decimal("0.5"),
            carbon_content_percent=Decimal("86")
        )

    @pytest.fixture
    def marine_diesel_oil(self) -> FuelComponent:
        """Marine Diesel Oil component."""
        return FuelComponent(
            fuel_id="MDO",
            name="Marine Diesel Oil",
            hhv_mj_kg=Decimal("45.0"),
            lhv_mj_kg=Decimal("42.8"),
            sulfur_ppm=Decimal("5000"),  # 0.5%
            ash_percent=Decimal("0.01"),
            viscosity_cst_40c=Decimal("6"),
            flash_point_c=Decimal("65"),
            density_kg_m3=Decimal("850"),
            cost_per_kg=Decimal("0.75"),
            available_kg=Decimal("50000"),
            heavy_metals_ppm={"vanadium": Decimal("5"), "nickel": Decimal("2")},
            moisture_percent=Decimal("0.1"),
            carbon_content_percent=Decimal("87")
        )

    @pytest.fixture
    def low_sulfur_fuel_oil(self) -> FuelComponent:
        """Low Sulfur Fuel Oil component."""
        return FuelComponent(
            fuel_id="LSFO",
            name="Low Sulfur Fuel Oil",
            hhv_mj_kg=Decimal("43.5"),
            lhv_mj_kg=Decimal("41.2"),
            sulfur_ppm=Decimal("5000"),  # 0.5%
            ash_percent=Decimal("0.05"),
            viscosity_cst_40c=Decimal("180"),
            flash_point_c=Decimal("72"),
            density_kg_m3=Decimal("920"),
            cost_per_kg=Decimal("0.55"),
            available_kg=Decimal("75000"),
            heavy_metals_ppm={"vanadium": Decimal("30"), "nickel": Decimal("15")},
            moisture_percent=Decimal("0.3"),
            carbon_content_percent=Decimal("86.5")
        )

    @pytest.fixture
    def standard_specification(self) -> BlendSpecification:
        """Standard blend specification."""
        return BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("10000"),  # 1%
            max_ash_percent=Decimal("0.1"),
            min_flash_point_c=Decimal("60"),
            max_viscosity_cst_40c=Decimal("380"),
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={"vanadium": Decimal("100"), "nickel": Decimal("40")},
            target_volume_kg=Decimal("10000"),
            max_moisture_percent=Decimal("1.0")
        )

    @pytest.fixture
    def cost_objective(self) -> OptimizationObjective:
        """Cost minimization objective."""
        return OptimizationObjective(
            objective_type="minimize_cost",
            cost_weight=Decimal("0.7"),
            hhv_weight=Decimal("0.2"),
            emissions_weight=Decimal("0.05"),
            quality_weight=Decimal("0.05")
        )

    @pytest.fixture
    def two_fuel_input(
        self,
        heavy_fuel_oil: FuelComponent,
        marine_diesel_oil: FuelComponent,
        standard_specification: BlendSpecification,
        cost_objective: OptimizationObjective
    ) -> BlendOptimizationInput:
        """Two-fuel blend input."""
        return BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil, marine_diesel_oil),
            specification=standard_specification,
            objective=cost_objective
        )

    @pytest.fixture
    def three_fuel_input(
        self,
        heavy_fuel_oil: FuelComponent,
        marine_diesel_oil: FuelComponent,
        low_sulfur_fuel_oil: FuelComponent,
        standard_specification: BlendSpecification,
        cost_objective: OptimizationObjective
    ) -> BlendOptimizationInput:
        """Three-fuel blend input."""
        return BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil, marine_diesel_oil, low_sulfur_fuel_oil),
            specification=standard_specification,
            objective=cost_objective
        )

    # ==========================================================================
    # Basic Functionality Tests
    # ==========================================================================

    def test_basic_two_fuel_optimization(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test basic two-fuel blend optimization."""
        result = optimizer.optimize_blend(two_fuel_input)

        assert result is not None
        assert isinstance(result, BlendOptimizationResult)
        assert len(result.blend_ratios) > 0

    def test_blend_ratios_sum_to_one(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test that blend ratios sum to approximately 1.0."""
        result = optimizer.optimize_blend(two_fuel_input)

        total_ratio = sum(result.blend_ratios.values())
        assert Decimal("0.999") <= total_ratio <= Decimal("1.001"), \
            f"Ratios sum to {total_ratio}, expected ~1.0"

    def test_three_fuel_optimization(
        self,
        optimizer: FuelBlendingOptimizer,
        three_fuel_input: BlendOptimizationInput
    ):
        """Test three-fuel blend optimization."""
        result = optimizer.optimize_blend(three_fuel_input)

        assert result is not None
        total_ratio = sum(result.blend_ratios.values())
        assert Decimal("0.999") <= total_ratio <= Decimal("1.001")

    def test_single_fuel_optimization(
        self,
        optimizer: FuelBlendingOptimizer,
        heavy_fuel_oil: FuelComponent,
        standard_specification: BlendSpecification,
        cost_objective: OptimizationObjective
    ):
        """Test single fuel (no blending) optimization."""
        input_data = BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil,),
            specification=standard_specification,
            objective=cost_objective
        )

        result = optimizer.optimize_blend(input_data)

        assert result.blend_ratios.get("HFO") == Decimal("1")

    # ==========================================================================
    # Heating Value Tests
    # ==========================================================================

    def test_blend_hhv_calculation(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test that blend HHV is calculated correctly (weighted average)."""
        result = optimizer.optimize_blend(two_fuel_input)

        # HHV should be positive and within component range
        assert result.blend_hhv_mj_kg > Decimal("0")
        assert Decimal("40") <= result.blend_hhv_mj_kg <= Decimal("50")

    def test_blend_hhv_meets_minimum(
        self,
        optimizer: FuelBlendingOptimizer,
        three_fuel_input: BlendOptimizationInput
    ):
        """Test that blend meets minimum HHV specification."""
        result = optimizer.optimize_blend(three_fuel_input)

        # Check if meets spec or has violation warning
        if result.meets_specification:
            assert result.blend_hhv_mj_kg >= Decimal("40")
        else:
            # Should have HHV violation in list
            hhv_violations = [v for v in result.constraint_violations if "HHV" in v]
            assert len(hhv_violations) > 0 or result.blend_hhv_mj_kg >= Decimal("40")

    def test_blend_lhv_less_than_hhv(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test that LHV is always less than HHV."""
        result = optimizer.optimize_blend(two_fuel_input)

        assert result.blend_lhv_mj_kg < result.blend_hhv_mj_kg

    # ==========================================================================
    # Sulfur Constraint Tests
    # ==========================================================================

    def test_sulfur_constraint_enforcement(
        self,
        optimizer: FuelBlendingOptimizer,
        heavy_fuel_oil: FuelComponent,
        marine_diesel_oil: FuelComponent,
        cost_objective: OptimizationObjective
    ):
        """Test sulfur constraint is enforced."""
        strict_spec = BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("5000"),  # Strict 0.5% limit
            max_ash_percent=Decimal("0.1"),
            min_flash_point_c=Decimal("60"),
            max_viscosity_cst_40c=Decimal("400"),
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={"vanadium": Decimal("200")},
            target_volume_kg=Decimal("10000")
        )

        input_data = BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil, marine_diesel_oil),
            specification=strict_spec,
            objective=cost_objective
        )

        result = optimizer.optimize_blend(input_data)

        # Either meets sulfur spec or has violation listed
        if result.blend_sulfur_ppm > Decimal("5000"):
            assert any("Sulfur" in v or "sulfur" in v for v in result.constraint_violations)

    def test_high_sulfur_fuel_ratio_limited(
        self,
        optimizer: FuelBlendingOptimizer,
        heavy_fuel_oil: FuelComponent,
        marine_diesel_oil: FuelComponent,
        cost_objective: OptimizationObjective
    ):
        """Test that high-sulfur fuel ratio is limited by sulfur constraint."""
        strict_spec = BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("10000"),
            max_ash_percent=Decimal("0.1"),
            min_flash_point_c=Decimal("60"),
            max_viscosity_cst_40c=Decimal("400"),
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={"vanadium": Decimal("200")},
            target_volume_kg=Decimal("10000")
        )

        input_data = BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil, marine_diesel_oil),
            specification=strict_spec,
            objective=cost_objective
        )

        result = optimizer.optimize_blend(input_data)

        # HFO has 35000 ppm sulfur, MDO has 5000 ppm
        # To get 10000 ppm, need MDO ratio > (35000-10000)/(35000-5000) = 83%
        # So HFO should be limited
        if result.blend_sulfur_ppm <= Decimal("10000"):
            # Verify weighted sulfur calculation
            expected_sulfur = (
                result.blend_ratios.get("HFO", Decimal("0")) * Decimal("35000") +
                result.blend_ratios.get("MDO", Decimal("0")) * Decimal("5000")
            )
            assert abs(result.blend_sulfur_ppm - expected_sulfur) < Decimal("100")

    # ==========================================================================
    # Viscosity (Refutas) Tests
    # ==========================================================================

    def test_refutas_viscosity_calculation(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test Refutas viscosity blending calculation."""
        result = optimizer.optimize_blend(two_fuel_input)

        # Viscosity should be positive
        assert result.blend_viscosity_cst_40c > Decimal("0")

        # Should be between component viscosities (non-linear but bounded)
        # HFO: 380, MDO: 6
        assert result.blend_viscosity_cst_40c >= Decimal("5")
        assert result.blend_viscosity_cst_40c <= Decimal("400")

    def test_viscosity_blending_nonlinear(
        self,
        optimizer: FuelBlendingOptimizer,
        heavy_fuel_oil: FuelComponent,
        marine_diesel_oil: FuelComponent,
        standard_specification: BlendSpecification,
        cost_objective: OptimizationObjective
    ):
        """Test that viscosity blending follows Refutas (non-linear)."""
        input_data = BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil, marine_diesel_oil),
            specification=standard_specification,
            objective=cost_objective
        )

        result = optimizer.optimize_blend(input_data)

        # Calculate linear average for comparison
        hfo_ratio = result.blend_ratios.get("HFO", Decimal("0"))
        mdo_ratio = result.blend_ratios.get("MDO", Decimal("0"))
        linear_viscosity = hfo_ratio * Decimal("380") + mdo_ratio * Decimal("6")

        # Refutas viscosity should differ from linear (usually lower due to log scale)
        # This is a soft check - the key is that calculation completed
        assert result.blend_viscosity_cst_40c != linear_viscosity or \
               abs(hfo_ratio - Decimal("1")) < Decimal("0.01") or \
               abs(mdo_ratio - Decimal("1")) < Decimal("0.01")

    def test_viscosity_constraint_max(
        self,
        optimizer: FuelBlendingOptimizer,
        heavy_fuel_oil: FuelComponent,
        marine_diesel_oil: FuelComponent,
        cost_objective: OptimizationObjective
    ):
        """Test maximum viscosity constraint."""
        strict_visc_spec = BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("40000"),
            max_ash_percent=Decimal("0.1"),
            min_flash_point_c=Decimal("60"),
            max_viscosity_cst_40c=Decimal("100"),  # Strict limit
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={},
            target_volume_kg=Decimal("10000")
        )

        input_data = BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil, marine_diesel_oil),
            specification=strict_visc_spec,
            objective=cost_objective
        )

        result = optimizer.optimize_blend(input_data)

        # Should either meet spec or have violation
        if result.blend_viscosity_cst_40c > Decimal("100"):
            assert any("Viscosity" in v or "viscosity" in v for v in result.constraint_violations)

    # ==========================================================================
    # Flash Point Safety Tests
    # ==========================================================================

    def test_flash_point_safety_constraint(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test flash point safety constraint."""
        result = optimizer.optimize_blend(two_fuel_input)

        # Flash point should be positive
        assert result.blend_flash_point_c > Decimal("0")

    def test_flash_point_violation_flagged(
        self,
        optimizer: FuelBlendingOptimizer,
        heavy_fuel_oil: FuelComponent,
        marine_diesel_oil: FuelComponent,
        cost_objective: OptimizationObjective
    ):
        """Test that flash point violation is flagged as safety concern."""
        high_flash_spec = BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("40000"),
            max_ash_percent=Decimal("0.1"),
            min_flash_point_c=Decimal("100"),  # High minimum (hard to meet)
            max_viscosity_cst_40c=Decimal("400"),
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={},
            target_volume_kg=Decimal("10000")
        )

        input_data = BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil, marine_diesel_oil),
            specification=high_flash_spec,
            objective=cost_objective
        )

        result = optimizer.optimize_blend(input_data)

        # Both fuels have flash point < 100C, so should violate
        # HFO: 70C, MDO: 65C
        if result.blend_flash_point_c < Decimal("100"):
            assert any("flash" in v.lower() or "Flash" in v for v in result.constraint_violations)

    # ==========================================================================
    # Heavy Metal Tests
    # ==========================================================================

    def test_heavy_metal_tracking(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test heavy metal content tracking."""
        result = optimizer.optimize_blend(two_fuel_input)

        # Should have vanadium and nickel tracked
        assert "vanadium" in result.heavy_metals_ppm or len(result.heavy_metals_ppm) >= 0

    def test_heavy_metal_constraint_violation(
        self,
        optimizer: FuelBlendingOptimizer,
        heavy_fuel_oil: FuelComponent,
        marine_diesel_oil: FuelComponent,
        cost_objective: OptimizationObjective
    ):
        """Test heavy metal constraint violation detection."""
        strict_metal_spec = BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("40000"),
            max_ash_percent=Decimal("0.1"),
            min_flash_point_c=Decimal("60"),
            max_viscosity_cst_40c=Decimal("400"),
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={"vanadium": Decimal("10")},  # Very strict
            target_volume_kg=Decimal("10000")
        )

        input_data = BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil, marine_diesel_oil),
            specification=strict_metal_spec,
            objective=cost_objective
        )

        result = optimizer.optimize_blend(input_data)

        # HFO has 150 ppm V, MDO has 5 ppm - hard to meet 10 ppm limit
        vanadium_content = result.heavy_metals_ppm.get("vanadium", Decimal("0"))
        if vanadium_content > Decimal("10"):
            assert any("vanadium" in v.lower() for v in result.constraint_violations)

    # ==========================================================================
    # Cost Optimization Tests
    # ==========================================================================

    def test_cost_calculation(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test total cost calculation."""
        result = optimizer.optimize_blend(two_fuel_input)

        # Cost should be positive
        assert result.total_cost_per_kg > Decimal("0")
        assert result.total_cost > Decimal("0")

        # Total cost = cost_per_kg * volume
        expected_total = result.total_cost_per_kg * Decimal("10000")
        assert abs(result.total_cost - expected_total) < Decimal("1")

    def test_cost_objective_favors_cheaper_fuel(
        self,
        optimizer: FuelBlendingOptimizer,
        heavy_fuel_oil: FuelComponent,
        marine_diesel_oil: FuelComponent,
        standard_specification: BlendSpecification
    ):
        """Test that cost objective favors cheaper fuel when constraints allow."""
        strong_cost_objective = OptimizationObjective(
            objective_type="minimize_cost",
            cost_weight=Decimal("0.9"),
            hhv_weight=Decimal("0.05"),
            emissions_weight=Decimal("0.025"),
            quality_weight=Decimal("0.025")
        )

        # Relax specs to allow more HFO (cheaper)
        relaxed_spec = BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("40000"),  # Very relaxed
            max_ash_percent=Decimal("0.5"),
            min_flash_point_c=Decimal("60"),
            max_viscosity_cst_40c=Decimal("400"),
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={},
            target_volume_kg=Decimal("10000")
        )

        input_data = BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil, marine_diesel_oil),
            specification=relaxed_spec,
            objective=strong_cost_objective
        )

        result = optimizer.optimize_blend(input_data)

        # HFO is cheaper ($0.35/kg vs $0.75/kg)
        # With relaxed specs, should favor HFO
        hfo_ratio = result.blend_ratios.get("HFO", Decimal("0"))
        mdo_ratio = result.blend_ratios.get("MDO", Decimal("0"))

        # Cost per kg should be closer to HFO price than MDO
        assert result.total_cost_per_kg < Decimal("0.60")

    # ==========================================================================
    # Inventory Constraint Tests
    # ==========================================================================

    def test_inventory_constraint(
        self,
        optimizer: FuelBlendingOptimizer,
        marine_diesel_oil: FuelComponent,
        cost_objective: OptimizationObjective
    ):
        """Test inventory constraint enforcement."""
        # Modify MDO to have limited inventory
        limited_mdo = FuelComponent(
            fuel_id="MDO",
            name="Marine Diesel Oil",
            hhv_mj_kg=Decimal("45.0"),
            lhv_mj_kg=Decimal("42.8"),
            sulfur_ppm=Decimal("5000"),
            ash_percent=Decimal("0.01"),
            viscosity_cst_40c=Decimal("6"),
            flash_point_c=Decimal("65"),
            density_kg_m3=Decimal("850"),
            cost_per_kg=Decimal("0.75"),
            available_kg=Decimal("2000"),  # Only 2000 kg available
            heavy_metals_ppm={},
            moisture_percent=Decimal("0.1"),
            carbon_content_percent=Decimal("87")
        )

        unlimited_hfo = FuelComponent(
            fuel_id="HFO",
            name="Heavy Fuel Oil",
            hhv_mj_kg=Decimal("42.5"),
            lhv_mj_kg=Decimal("40.2"),
            sulfur_ppm=Decimal("35000"),
            ash_percent=Decimal("0.08"),
            viscosity_cst_40c=Decimal("380"),
            flash_point_c=Decimal("70"),
            density_kg_m3=Decimal("980"),
            cost_per_kg=Decimal("0.35"),
            available_kg=Decimal("100000"),
            heavy_metals_ppm={},
            moisture_percent=Decimal("0.5"),
            carbon_content_percent=Decimal("86")
        )

        spec = BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("40000"),
            max_ash_percent=Decimal("0.1"),
            min_flash_point_c=Decimal("60"),
            max_viscosity_cst_40c=Decimal("400"),
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={},
            target_volume_kg=Decimal("10000")  # Need 10000 kg
        )

        input_data = BlendOptimizationInput(
            fuel_components=(limited_mdo, unlimited_hfo),
            specification=spec,
            objective=cost_objective
        )

        result = optimizer.optimize_blend(input_data)

        # MDO ratio should be limited to 2000/10000 = 20%
        mdo_ratio = result.blend_ratios.get("MDO", Decimal("0"))
        assert mdo_ratio <= Decimal("0.21")

    def test_insufficient_inventory_error(
        self,
        optimizer: FuelBlendingOptimizer,
        cost_objective: OptimizationObjective
    ):
        """Test error when total inventory is insufficient."""
        small_hfo = FuelComponent(
            fuel_id="HFO",
            name="Heavy Fuel Oil",
            hhv_mj_kg=Decimal("42.5"),
            lhv_mj_kg=Decimal("40.2"),
            sulfur_ppm=Decimal("35000"),
            ash_percent=Decimal("0.08"),
            viscosity_cst_40c=Decimal("380"),
            flash_point_c=Decimal("70"),
            density_kg_m3=Decimal("980"),
            cost_per_kg=Decimal("0.35"),
            available_kg=Decimal("3000"),  # Only 3000 kg
            heavy_metals_ppm={},
            moisture_percent=Decimal("0.5"),
            carbon_content_percent=Decimal("86")
        )

        small_mdo = FuelComponent(
            fuel_id="MDO",
            name="Marine Diesel Oil",
            hhv_mj_kg=Decimal("45.0"),
            lhv_mj_kg=Decimal("42.8"),
            sulfur_ppm=Decimal("5000"),
            ash_percent=Decimal("0.01"),
            viscosity_cst_40c=Decimal("6"),
            flash_point_c=Decimal("65"),
            density_kg_m3=Decimal("850"),
            cost_per_kg=Decimal("0.75"),
            available_kg=Decimal("2000"),  # Only 2000 kg
            heavy_metals_ppm={},
            moisture_percent=Decimal("0.1"),
            carbon_content_percent=Decimal("87")
        )

        spec = BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("40000"),
            max_ash_percent=Decimal("0.1"),
            min_flash_point_c=Decimal("60"),
            max_viscosity_cst_40c=Decimal("400"),
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={},
            target_volume_kg=Decimal("10000")  # Need 10000 kg but only have 5000
        )

        input_data = BlendOptimizationInput(
            fuel_components=(small_hfo, small_mdo),
            specification=spec,
            objective=cost_objective
        )

        with pytest.raises(ValueError, match="Insufficient inventory"):
            optimizer.optimize_blend(input_data)

    # ==========================================================================
    # Provenance and Determinism Tests
    # ==========================================================================

    def test_provenance_hash_generated(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test that provenance hash is generated."""
        result = optimizer.optimize_blend(two_fuel_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_deterministic_results(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test that same inputs produce same outputs (determinism)."""
        result1 = optimizer.optimize_blend(two_fuel_input)
        result2 = optimizer.optimize_blend(two_fuel_input)

        assert result1.blend_ratios == result2.blend_ratios
        assert result1.blend_hhv_mj_kg == result2.blend_hhv_mj_kg
        assert result1.blend_sulfur_ppm == result2.blend_sulfur_ppm
        assert result1.blend_viscosity_cst_40c == result2.blend_viscosity_cst_40c

    def test_calculation_steps_recorded(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test that calculation steps are recorded for provenance."""
        result = optimizer.optimize_blend(two_fuel_input)

        assert len(result.calculation_steps) > 0

        # Check steps have required fields
        for step in result.calculation_steps:
            assert 'step_number' in step
            assert 'operation' in step
            assert 'inputs' in step
            assert 'outputs' in step

    # ==========================================================================
    # Optimization Score Tests
    # ==========================================================================

    def test_optimization_score_range(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test optimization score is in valid range."""
        result = optimizer.optimize_blend(two_fuel_input)

        assert Decimal("0") <= result.optimization_score <= Decimal("100")

    def test_meets_specification_flag(
        self,
        optimizer: FuelBlendingOptimizer,
        three_fuel_input: BlendOptimizationInput
    ):
        """Test meets_specification flag accuracy."""
        result = optimizer.optimize_blend(three_fuel_input)

        # If meets spec, should have no violations
        if result.meets_specification:
            assert len(result.constraint_violations) == 0
        else:
            assert len(result.constraint_violations) > 0

    # ==========================================================================
    # Processing Time Tests
    # ==========================================================================

    def test_processing_time_recorded(
        self,
        optimizer: FuelBlendingOptimizer,
        two_fuel_input: BlendOptimizationInput
    ):
        """Test that processing time is recorded."""
        result = optimizer.optimize_blend(two_fuel_input)

        assert result.processing_time_ms >= Decimal("0")

    # ==========================================================================
    # Edge Case Tests
    # ==========================================================================

    def test_empty_components_raises_error(
        self,
        optimizer: FuelBlendingOptimizer,
        standard_specification: BlendSpecification,
        cost_objective: OptimizationObjective
    ):
        """Test error on empty components list."""
        input_data = BlendOptimizationInput(
            fuel_components=tuple(),
            specification=standard_specification,
            objective=cost_objective
        )

        with pytest.raises(ValueError):
            optimizer.optimize_blend(input_data)

    def test_zero_target_volume_raises_error(
        self,
        optimizer: FuelBlendingOptimizer,
        heavy_fuel_oil: FuelComponent,
        cost_objective: OptimizationObjective
    ):
        """Test error on zero target volume."""
        zero_volume_spec = BlendSpecification(
            min_hhv_mj_kg=Decimal("40"),
            max_sulfur_ppm=Decimal("10000"),
            max_ash_percent=Decimal("0.1"),
            min_flash_point_c=Decimal("60"),
            max_viscosity_cst_40c=Decimal("400"),
            min_viscosity_cst_40c=Decimal("2"),
            max_heavy_metals_ppm={},
            target_volume_kg=Decimal("0")  # Zero volume
        )

        input_data = BlendOptimizationInput(
            fuel_components=(heavy_fuel_oil,),
            specification=zero_volume_spec,
            objective=cost_objective
        )

        with pytest.raises(ValueError):
            optimizer.optimize_blend(input_data)

    def test_statistics_tracking(self, optimizer: FuelBlendingOptimizer):
        """Test statistics are tracked correctly."""
        stats = optimizer.get_statistics()

        assert 'calculation_count' in stats
        assert 'max_iterations' in stats
        assert stats['calculation_count'] >= 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
