"""
GL-016 WATERGUARD Agent - Chemical Dosing Optimizer Tests

Unit tests for ChemicalDosingOptimizer covering:
- Oxygen scavenger dosing calculations
- Phosphate dosing calculations
- Amine dosing calculations
- Cost optimization
- Chemical feed rate calculations
- Provenance tracking

Author: GL-TestEngineer
Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timezone

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    ChemicalDosingInput,
    ChemicalDosingOutput,
    TreatmentProgram,
    ChemicalType,
)

from greenlang.agents.process_heat.gl_016_water_treatment.chemical_dosing import (
    ChemicalDosingOptimizer,
    DosingConstants,
    calculate_chemical_feed_rate,
)


class TestChemicalDosingOptimizerInitialization:
    """Test ChemicalDosingOptimizer initialization."""

    def test_default_initialization(self):
        """Test optimizer initializes with defaults."""
        optimizer = ChemicalDosingOptimizer()
        assert optimizer is not None


class TestChemicalDosingOptimization:
    """Test chemical dosing optimization."""

    @pytest.fixture
    def optimizer(self):
        return ChemicalDosingOptimizer()

    def test_optimize_returns_output(self, optimizer, chemical_dosing_input_standard):
        """Test optimize returns ChemicalDosingOutput."""
        result = optimizer.optimize(chemical_dosing_input_standard)
        assert isinstance(result, ChemicalDosingOutput)

    def test_optimize_all_fields_populated(self, optimizer, chemical_dosing_input_standard):
        """Test optimize populates all output fields."""
        result = optimizer.optimize(chemical_dosing_input_standard)

        assert result.scavenger_dose_recommended_ppm is not None
        assert result.scavenger_dose_change_ppm is not None
        assert result.scavenger_feed_rate_lb_hr is not None
        assert result.scavenger_ratio_to_o2 is not None
        assert result.phosphate_dose_recommended_ppm is not None
        assert result.phosphate_feed_rate_lb_hr is not None
        assert result.provenance_hash is not None
        assert result.processing_time_ms > 0

    def test_optimize_with_high_do(self, optimizer, chemical_dosing_input_high_do):
        """Test optimization with high dissolved oxygen."""
        result = optimizer.optimize(chemical_dosing_input_high_do)

        # Should recommend higher scavenger dose
        assert result.scavenger_dose_change_ppm > 0
        assert len(result.recommendations) > 0


class TestOxygenScavengerDosing:
    """Test oxygen scavenger dosing calculations."""

    @pytest.fixture
    def optimizer(self):
        return ChemicalDosingOptimizer()

    def test_scavenger_dose_calculation(self, optimizer):
        """Test scavenger dose calculation."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            current_scavenger_type=ChemicalType.SULFITE,
            current_scavenger_dose_ppm=25.0,
            target_scavenger_residual_ppm=30.0,
            operating_pressure_psig=450.0,
        )
        dose, rate, ratio = optimizer._calculate_scavenger_dosing(input_data)

        assert dose > 0
        assert rate > 0
        assert ratio > 0

    @pytest.mark.parametrize("do_ppb,expected_min_dose", [
        (5.0, 0.05),   # 5 ppb * 7.9 * 1.5 / 1000 ~ 0.06
        (10.0, 0.10),  # Higher DO = higher dose
        (3.0, 0.03),   # Lower DO = lower dose
    ])
    def test_scavenger_dose_scales_with_do(self, optimizer, do_ppb, expected_min_dose):
        """Test scavenger dose scales with dissolved oxygen."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=do_ppb,
            current_scavenger_type=ChemicalType.SULFITE,
            current_scavenger_dose_ppm=25.0,
            target_scavenger_residual_ppm=30.0,
            operating_pressure_psig=450.0,
        )
        dose, _, _ = optimizer._calculate_scavenger_dosing(input_data)

        # Dose should be at least stoichiometric
        assert dose >= expected_min_dose

    def test_sulfite_stoichiometry(self, optimizer):
        """Test sulfite stoichiometric ratio."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            current_scavenger_type=ChemicalType.SULFITE,
            current_scavenger_dose_ppm=25.0,
            target_scavenger_residual_ppm=30.0,
            operating_pressure_psig=450.0,
        )
        _, _, ratio = optimizer._calculate_scavenger_dosing(input_data)

        # Sulfite ratio should be around 7.9 * excess factor + residual contribution
        assert ratio > 7.0

    def test_hydrazine_stoichiometry(self, optimizer):
        """Test hydrazine stoichiometric ratio."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            current_scavenger_type=ChemicalType.HYDRAZINE,
            current_scavenger_dose_ppm=0.1,
            target_scavenger_residual_ppm=0.1,
            operating_pressure_psig=450.0,
        )
        _, _, ratio = optimizer._calculate_scavenger_dosing(input_data)

        # Hydrazine ratio should be around 1.0 * 2.0 (100% excess)
        assert ratio > 1.0


class TestPhosphateDosing:
    """Test phosphate dosing calculations."""

    @pytest.fixture
    def optimizer(self):
        return ChemicalDosingOptimizer()

    def test_phosphate_dose_calculation(self, optimizer):
        """Test phosphate dose calculation."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            boiler_phosphate_ppm=8.0,
            target_phosphate_ppm=10.0,
            current_phosphate_dose_ppm=0.3,
            blowdown_rate_pct=5.0,
            operating_pressure_psig=450.0,
            treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
        )
        dose, rate = optimizer._calculate_phosphate_dosing(input_data)

        assert dose > 0
        assert rate > 0

    def test_phosphate_dose_increases_with_blowdown(self, optimizer):
        """Test phosphate dose increases with higher blowdown."""
        base_input = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            target_phosphate_ppm=10.0,
            blowdown_rate_pct=3.0,
            operating_pressure_psig=450.0,
            treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
        )
        high_bd_input = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            target_phosphate_ppm=10.0,
            blowdown_rate_pct=8.0,  # Higher blowdown
            operating_pressure_psig=450.0,
            treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
        )

        dose_low_bd, _ = optimizer._calculate_phosphate_dosing(base_input)
        dose_high_bd, _ = optimizer._calculate_phosphate_dosing(high_bd_input)

        assert dose_high_bd > dose_low_bd


class TestAmineDosing:
    """Test amine dosing calculations."""

    @pytest.fixture
    def optimizer(self):
        return ChemicalDosingOptimizer()

    def test_amine_dose_calculation(self, optimizer):
        """Test amine dose calculation."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            condensate_ph=8.2,
            target_condensate_ph=8.7,
            current_amine_type=ChemicalType.MORPHOLINE,
            current_amine_dose_ppm=3.0,
            operating_pressure_psig=450.0,
        )
        dose, rate, change = optimizer._calculate_amine_dosing(input_data)

        assert dose is not None
        assert rate is not None
        # Should recommend increase since current_ph < target_ph
        assert change > 0

    def test_amine_dose_increase_for_low_ph(self, optimizer):
        """Test amine dose increase when pH is low."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            condensate_ph=7.5,  # Low
            target_condensate_ph=8.7,
            current_amine_type=ChemicalType.MORPHOLINE,
            current_amine_dose_ppm=3.0,
            operating_pressure_psig=450.0,
        )
        dose, _, change = optimizer._calculate_amine_dosing(input_data)

        assert change > 0  # Should increase

    def test_amine_dose_decrease_for_high_ph(self, optimizer):
        """Test amine dose decrease when pH is high."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            condensate_ph=9.2,  # High
            target_condensate_ph=8.7,
            current_amine_type=ChemicalType.MORPHOLINE,
            current_amine_dose_ppm=8.0,
            operating_pressure_psig=450.0,
        )
        dose, _, change = optimizer._calculate_amine_dosing(input_data)

        assert change < 0  # Should decrease

    def test_no_amine_when_not_configured(self, optimizer):
        """Test no amine calculation when not configured."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            operating_pressure_psig=450.0,
            # No amine configuration
        )
        dose, rate, change = optimizer._calculate_amine_dosing(input_data)

        assert dose is None
        assert rate is None
        assert change is None


class TestCostCalculations:
    """Test chemical cost calculations."""

    @pytest.fixture
    def optimizer(self):
        return ChemicalDosingOptimizer()

    def test_cost_calculation(self, optimizer, chemical_dosing_input_standard):
        """Test cost calculation."""
        result = optimizer.optimize(chemical_dosing_input_standard)

        assert result.current_chemical_cost_per_day >= 0
        assert result.optimized_chemical_cost_per_day >= 0
        assert result.cost_savings_per_day is not None
        assert result.annual_savings_usd is not None

    def test_annual_savings_is_365x_daily(self, optimizer, chemical_dosing_input_standard):
        """Test annual savings is 365x daily savings."""
        result = optimizer.optimize(chemical_dosing_input_standard)

        expected_annual = result.cost_savings_per_day * 365
        assert result.annual_savings_usd == pytest.approx(expected_annual, rel=0.01)


class TestDosingRangeValidation:
    """Test dosing range validation."""

    @pytest.fixture
    def optimizer(self):
        return ChemicalDosingOptimizer()

    def test_within_recommended_ranges(self, optimizer, chemical_dosing_input_standard):
        """Test within_recommended_ranges flag."""
        result = optimizer.optimize(chemical_dosing_input_standard)
        assert isinstance(result.within_recommended_ranges, bool)

    def test_out_of_range_detection(self, optimizer):
        """Test out of range detection."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=50.0,  # Very high DO
            current_scavenger_type=ChemicalType.SULFITE,
            current_scavenger_dose_ppm=5.0,  # Insufficient
            target_scavenger_residual_ppm=30.0,
            operating_pressure_psig=450.0,
        )
        result = optimizer.optimize(input_data)

        # Should flag as out of range
        assert result.within_recommended_ranges == False or len(result.recommendations) > 0


class TestRecommendationGeneration:
    """Test recommendation generation."""

    @pytest.fixture
    def optimizer(self):
        return ChemicalDosingOptimizer()

    def test_recommendations_for_high_do(self, optimizer, chemical_dosing_input_high_do):
        """Test recommendations for high dissolved oxygen."""
        result = optimizer.optimize(chemical_dosing_input_high_do)

        rec_text = " ".join(result.recommendations).lower()
        assert "deaerator" in rec_text or "scavenger" in rec_text

    def test_recommendations_for_dose_increase(self, optimizer):
        """Test recommendations for required dose increase."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=10.0,
            current_scavenger_type=ChemicalType.SULFITE,
            current_scavenger_dose_ppm=10.0,  # Low
            target_scavenger_residual_ppm=30.0,
            operating_pressure_psig=450.0,
        )
        result = optimizer.optimize(input_data)

        # Should have recommendation to increase scavenger
        rec_text = " ".join(result.recommendations).lower()
        assert "increase" in rec_text or "scavenger" in rec_text

    def test_optimal_dosing_no_changes(self, optimizer):
        """Test optimal dosing produces minimal recommendations."""
        input_data = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            current_scavenger_type=ChemicalType.SULFITE,
            current_scavenger_dose_ppm=30.0,  # Optimal
            target_scavenger_residual_ppm=30.0,
            boiler_phosphate_ppm=10.0,
            target_phosphate_ppm=10.0,
            current_phosphate_dose_ppm=0.5,
            blowdown_rate_pct=5.0,
            operating_pressure_psig=450.0,
            treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
        )
        result = optimizer.optimize(input_data)

        # Changes should be small
        assert abs(result.scavenger_dose_change_ppm) < 10
        assert abs(result.phosphate_dose_change_ppm) < 5


class TestCalculateChemicalFeedRate:
    """Test chemical feed rate calculation function."""

    def test_feed_rate_calculation(self):
        """Test basic feed rate calculation."""
        rate = calculate_chemical_feed_rate(
            dose_ppm=30.0,
            flow_rate_lb_hr=50000.0,
            product_strength_pct=100.0,
        )
        # Expected: 30 * 50000 / 1,000,000 = 1.5 lb/hr
        assert rate == pytest.approx(1.5, rel=0.01)

    def test_feed_rate_with_dilute_product(self):
        """Test feed rate with dilute product."""
        rate = calculate_chemical_feed_rate(
            dose_ppm=30.0,
            flow_rate_lb_hr=50000.0,
            product_strength_pct=50.0,  # 50% product
        )
        # Should be double the 100% product rate
        assert rate == pytest.approx(3.0, rel=0.01)

    @pytest.mark.parametrize("dose,flow,strength,expected", [
        (30.0, 50000.0, 100.0, 1.5),
        (30.0, 50000.0, 50.0, 3.0),
        (15.0, 100000.0, 100.0, 1.5),
        (60.0, 25000.0, 100.0, 1.5),
    ])
    def test_feed_rate_parameterized(self, dose, flow, strength, expected):
        """Test feed rate calculation with various parameters."""
        rate = calculate_chemical_feed_rate(dose, flow, strength)
        assert rate == pytest.approx(expected, rel=0.01)


class TestDosingConstants:
    """Test dosing constants values."""

    def test_ppm_conversion(self):
        """Test PPM conversion constant."""
        assert DosingConstants.PPM_TO_LB_PER_KGAL == 0.00834

    def test_water_density(self):
        """Test water density constant."""
        assert DosingConstants.LB_PER_GAL_WATER == 8.34

    def test_scavenger_stoichiometry(self):
        """Test scavenger stoichiometry values."""
        assert ChemicalType.SULFITE in DosingConstants.SCAVENGER_STOICH
        assert ChemicalType.HYDRAZINE in DosingConstants.SCAVENGER_STOICH
        assert ChemicalType.CARBOHYDRAZIDE in DosingConstants.SCAVENGER_STOICH

        # Sulfite ratio should be ~7.88
        assert DosingConstants.SCAVENGER_STOICH[ChemicalType.SULFITE] == pytest.approx(7.88, rel=0.02)

        # Hydrazine ratio should be 1.0
        assert DosingConstants.SCAVENGER_STOICH[ChemicalType.HYDRAZINE] == 1.0

    def test_scavenger_excess_factors(self):
        """Test scavenger excess factors."""
        assert ChemicalType.SULFITE in DosingConstants.SCAVENGER_EXCESS
        assert DosingConstants.SCAVENGER_EXCESS[ChemicalType.SULFITE] == 1.5  # 50% excess
        assert DosingConstants.SCAVENGER_EXCESS[ChemicalType.HYDRAZINE] == 2.0  # 100% excess


class TestProvenanceTracking:
    """Test provenance hash calculation."""

    @pytest.fixture
    def optimizer(self):
        return ChemicalDosingOptimizer()

    def test_provenance_hash_generated(self, optimizer, chemical_dosing_input_standard):
        """Test provenance hash is generated."""
        result = optimizer.optimize(chemical_dosing_input_standard)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_reproducible(self, optimizer, chemical_dosing_input_standard):
        """Test same input produces same provenance hash."""
        hash1 = optimizer._calculate_provenance_hash(chemical_dosing_input_standard)
        hash2 = optimizer._calculate_provenance_hash(chemical_dosing_input_standard)
        assert hash1 == hash2


class TestPerformance:
    """Performance tests for chemical dosing optimizer."""

    @pytest.fixture
    def optimizer(self):
        return ChemicalDosingOptimizer()

    @pytest.mark.performance
    def test_optimization_performance(self, optimizer, chemical_dosing_input_standard):
        """Test optimization completes within performance target."""
        import time
        start = time.perf_counter()
        result = optimizer.optimize(chemical_dosing_input_standard)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50  # Should complete in < 50ms
        assert result.processing_time_ms < 50

    @pytest.mark.performance
    def test_batch_optimization_performance(self, optimizer, chemical_dosing_input_standard):
        """Test batch optimization maintains throughput."""
        import time
        num_optimizations = 100

        start = time.perf_counter()
        for _ in range(num_optimizations):
            optimizer.optimize(chemical_dosing_input_standard)
        elapsed_s = time.perf_counter() - start

        throughput = num_optimizations / elapsed_s
        assert throughput > 50  # At least 50 optimizations/second
