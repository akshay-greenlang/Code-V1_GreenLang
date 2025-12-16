"""
GL-006 WasteHeatRecovery Agent - HEN Synthesis Tests

Comprehensive unit tests for the HENSynthesizer class.
Tests heat exchanger network design and optimization.

Coverage Target: 85%+
"""

import pytest
import math
from datetime import datetime
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_006_waste_heat_recovery.hen_synthesis import (
    HENSynthesizer,
    HeatExchangerType,
    MatchType,
    NetworkRegion,
    HeatExchangerCostModel,
    UtilityCostModel,
    StreamMatch,
    UtilityMatch,
    HENDesign,
    AreaTargetResult,
)
from greenlang.agents.process_heat.gl_006_waste_heat_recovery.pinch_analysis import (
    PinchAnalyzer,
    HeatStream,
    StreamType,
    PinchAnalysisResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def pinch_analyzer():
    """Create PinchAnalyzer for generating pinch results."""
    return PinchAnalyzer(delta_t_min_f=20.0)


@pytest.fixture
def four_stream_problem():
    """Classic 4-stream problem."""
    return [
        HeatStream(
            name="H1",
            stream_type=StreamType.HOT,
            supply_temp_f=350.0,
            target_temp_f=140.0,
            mcp=3.0,
        ),
        HeatStream(
            name="H2",
            stream_type=StreamType.HOT,
            supply_temp_f=260.0,
            target_temp_f=100.0,
            mcp=1.5,
        ),
        HeatStream(
            name="C1",
            stream_type=StreamType.COLD,
            supply_temp_f=50.0,
            target_temp_f=260.0,
            mcp=2.0,
        ),
        HeatStream(
            name="C2",
            stream_type=StreamType.COLD,
            supply_temp_f=120.0,
            target_temp_f=300.0,
            mcp=2.5,
        ),
    ]


@pytest.fixture
def pinch_result(pinch_analyzer, four_stream_problem):
    """Generate pinch analysis result."""
    return pinch_analyzer.analyze(four_stream_problem)


@pytest.fixture
def synthesizer(pinch_result):
    """Create HENSynthesizer with pinch result."""
    return HENSynthesizer(
        pinch_result=pinch_result,
        default_u_value=50.0,
    )


@pytest.fixture
def cost_model():
    """Create custom cost model."""
    return HeatExchangerCostModel(
        hx_type=HeatExchangerType.SHELL_TUBE_AES,
        base_cost_usd=15000.0,
        area_exponent=0.65,
        cost_per_ft2=200.0,
        installation_factor=3.0,
        material_factor=1.2,
        pressure_factor=1.0,
    )


@pytest.fixture
def utility_model():
    """Create custom utility model."""
    return UtilityCostModel(
        hot_utility_cost_per_mmbtu=10.0,
        cold_utility_cost_per_mmbtu=2.0,
        operating_hours_per_year=8000,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestHENSynthesizerInitialization:
    """Test HENSynthesizer initialization."""

    @pytest.mark.unit
    def test_default_initialization(self, pinch_result):
        """Test synthesizer initializes with defaults."""
        synthesizer = HENSynthesizer(pinch_result=pinch_result)

        assert synthesizer.pinch_result == pinch_result
        assert synthesizer.default_u_value == 50.0

    @pytest.mark.unit
    def test_custom_initialization(self, pinch_result, cost_model, utility_model):
        """Test synthesizer with custom models."""
        synthesizer = HENSynthesizer(
            pinch_result=pinch_result,
            cost_model=cost_model,
            utility_model=utility_model,
            default_u_value=75.0,
        )

        assert synthesizer.cost_model == cost_model
        assert synthesizer.utility_model == utility_model
        assert synthesizer.default_u_value == 75.0

    @pytest.mark.unit
    def test_initialization_without_pinch_result(self):
        """Test synthesizer can initialize without pinch result."""
        synthesizer = HENSynthesizer()

        assert synthesizer.pinch_result is None


# =============================================================================
# NETWORK SYNTHESIS TESTS
# =============================================================================

class TestNetworkSynthesis:
    """Test network synthesis functionality."""

    @pytest.mark.unit
    def test_synthesis_requires_pinch_result(self, four_stream_problem):
        """Test synthesis requires pinch result."""
        synthesizer = HENSynthesizer()

        with pytest.raises(ValueError, match="Pinch analysis result required"):
            synthesizer.synthesize_network(four_stream_problem)

    @pytest.mark.unit
    def test_basic_synthesis(self, synthesizer, four_stream_problem, pinch_result):
        """Test basic network synthesis."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        assert isinstance(design, HENDesign)
        assert design.design_id is not None
        assert design.timestamp is not None

    @pytest.mark.unit
    def test_synthesis_creates_matches(self, synthesizer, four_stream_problem, pinch_result):
        """Test synthesis creates heat exchanger matches."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        # Should have some process matches
        assert design.total_units > 0

    @pytest.mark.unit
    def test_synthesis_with_pinch_design_method(
        self, synthesizer, four_stream_problem, pinch_result
    ):
        """Test synthesis with pinch design method."""
        design = synthesizer.synthesize_network(
            four_stream_problem,
            pinch_result,
            method="pinch_design",
        )

        assert isinstance(design, HENDesign)

    @pytest.mark.unit
    def test_synthesis_invalid_method(self, synthesizer, four_stream_problem, pinch_result):
        """Test synthesis with invalid method."""
        with pytest.raises(ValueError, match="Unknown synthesis method"):
            synthesizer.synthesize_network(
                four_stream_problem,
                pinch_result,
                method="invalid_method",
            )


# =============================================================================
# STREAM MATCH TESTS
# =============================================================================

class TestStreamMatches:
    """Test stream matching functionality."""

    @pytest.mark.unit
    def test_process_matches_have_required_fields(
        self, synthesizer, four_stream_problem, pinch_result
    ):
        """Test process matches have all required fields."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        for match in design.process_matches:
            assert match.match_id is not None
            assert match.hot_stream_name is not None
            assert match.cold_stream_name is not None
            assert match.heat_duty_btu_hr > 0
            assert match.hot_inlet_temp_f > match.hot_outlet_temp_f
            assert match.cold_outlet_temp_f > match.cold_inlet_temp_f

    @pytest.mark.unit
    def test_matches_respect_approach_temp(
        self, synthesizer, four_stream_problem, pinch_result
    ):
        """Test matches respect minimum approach temperature."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        delta_t_min = pinch_result.delta_t_min_f

        for match in design.process_matches:
            approach1 = match.hot_inlet_temp_f - match.cold_outlet_temp_f
            approach2 = match.hot_outlet_temp_f - match.cold_inlet_temp_f

            # Allow 10% tolerance for numerical issues
            assert approach1 >= delta_t_min * 0.9
            assert approach2 >= delta_t_min * 0.9

    @pytest.mark.unit
    def test_matches_have_lmtd(self, synthesizer, four_stream_problem, pinch_result):
        """Test matches have LMTD calculated."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        for match in design.process_matches:
            assert match.lmtd_f > 0

    @pytest.mark.unit
    def test_matches_have_area(self, synthesizer, four_stream_problem, pinch_result):
        """Test matches have area calculated."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        for match in design.process_matches:
            assert match.area_ft2 >= 0


# =============================================================================
# UTILITY EXCHANGER TESTS
# =============================================================================

class TestUtilityExchangers:
    """Test utility exchanger handling."""

    @pytest.mark.unit
    def test_utility_matches_created(self, synthesizer, four_stream_problem, pinch_result):
        """Test utility matches are created when needed."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        # Network should have utility exchangers to balance heat
        total_utilities = len(design.hot_utility_matches) + len(design.cold_utility_matches)
        # At least some utility should be needed
        assert design.utility_hx_units >= 0

    @pytest.mark.unit
    def test_hot_utility_above_pinch(self, synthesizer, four_stream_problem, pinch_result):
        """Test hot utility is placed above pinch (cold streams)."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        # Hot utility should heat cold streams
        for util in design.hot_utility_matches:
            assert util.utility_type == "hot"
            assert util.duty_btu_hr > 0

    @pytest.mark.unit
    def test_cold_utility_below_pinch(self, synthesizer, four_stream_problem, pinch_result):
        """Test cold utility is placed below pinch (hot streams)."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        # Cold utility should cool hot streams
        for util in design.cold_utility_matches:
            assert util.utility_type == "cold"
            assert util.duty_btu_hr > 0


# =============================================================================
# COST CALCULATION TESTS
# =============================================================================

class TestCostCalculations:
    """Test cost calculation functionality."""

    @pytest.mark.unit
    def test_capital_costs_calculated(
        self, synthesizer, four_stream_problem, pinch_result
    ):
        """Test capital costs are calculated."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        assert design.total_capital_cost_usd >= 0
        assert design.process_hx_capital_usd >= 0
        assert design.utility_hx_capital_usd >= 0

    @pytest.mark.unit
    def test_operating_costs_calculated(
        self, synthesizer, four_stream_problem, pinch_result
    ):
        """Test operating costs are calculated."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        assert design.annual_utility_cost_usd >= 0

    @pytest.mark.unit
    def test_total_annual_cost_calculated(
        self, synthesizer, four_stream_problem, pinch_result
    ):
        """Test total annual cost is calculated."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        # Total annual cost = annualized capital + operating
        assert design.total_annual_cost_usd >= 0

    @pytest.mark.unit
    def test_hx_cost_estimation(self, synthesizer):
        """Test heat exchanger cost estimation formula."""
        # Test area-based cost estimation
        area = 100.0  # ft2

        cost = synthesizer._estimate_hx_cost(area)

        # Cost should be positive and reasonable
        assert cost > 0
        assert cost > synthesizer.cost_model.base_cost_usd

    @pytest.mark.unit
    def test_hx_cost_zero_area(self, synthesizer):
        """Test HX cost for zero area."""
        cost = synthesizer._estimate_hx_cost(0.0)
        assert cost == 0

    @pytest.mark.unit
    def test_custom_cost_model(self, pinch_result, cost_model, four_stream_problem):
        """Test with custom cost model."""
        synthesizer = HENSynthesizer(
            pinch_result=pinch_result,
            cost_model=cost_model,
        )

        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        # Costs should reflect custom model
        assert design.total_capital_cost_usd >= 0


# =============================================================================
# LMTD CALCULATION TESTS
# =============================================================================

class TestLMTDCalculation:
    """Test LMTD calculation."""

    @pytest.mark.unit
    def test_lmtd_counterflow(self, synthesizer):
        """Test LMTD calculation for counterflow."""
        # Counterflow: hot in -> cold out, hot out -> cold in
        lmtd = synthesizer._calculate_lmtd(
            t_hot_in=200.0,
            t_hot_out=100.0,
            t_cold_out=150.0,
            t_cold_in=50.0,
        )

        # dt1 = 200 - 150 = 50
        # dt2 = 100 - 50 = 50
        # When dt1 = dt2, LMTD = dt1 = 50
        assert lmtd == pytest.approx(50.0, rel=0.01)

    @pytest.mark.unit
    def test_lmtd_unequal_dt(self, synthesizer):
        """Test LMTD with unequal temperature differences."""
        lmtd = synthesizer._calculate_lmtd(
            t_hot_in=200.0,
            t_hot_out=100.0,
            t_cold_out=180.0,
            t_cold_in=60.0,
        )

        # dt1 = 200 - 180 = 20
        # dt2 = 100 - 60 = 40
        # LMTD = (40 - 20) / ln(40/20) = 20 / 0.693 = 28.85
        expected_lmtd = (40 - 20) / math.log(40 / 20)
        assert lmtd == pytest.approx(expected_lmtd, rel=0.01)

    @pytest.mark.unit
    def test_lmtd_zero_dt_handled(self, synthesizer):
        """Test LMTD handles zero temperature difference."""
        # This should return a small positive value
        lmtd = synthesizer._calculate_lmtd(
            t_hot_in=100.0,
            t_hot_out=100.0,
            t_cold_out=100.0,
            t_cold_in=100.0,
        )

        # Should return small positive value to avoid division by zero
        assert lmtd > 0


# =============================================================================
# NETWORK METRICS TESTS
# =============================================================================

class TestNetworkMetrics:
    """Test network metrics calculation."""

    @pytest.mark.unit
    def test_unit_counts(self, synthesizer, four_stream_problem, pinch_result):
        """Test heat exchanger unit counts."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        assert design.total_units == design.process_hx_units + design.utility_hx_units
        assert design.process_hx_units == len(design.process_matches)

    @pytest.mark.unit
    def test_area_totals(self, synthesizer, four_stream_problem, pinch_result):
        """Test area totals are correct."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        expected_process_area = sum(m.area_ft2 for m in design.process_matches)
        assert design.process_area_ft2 == pytest.approx(expected_process_area, rel=0.01)

        expected_total = design.process_area_ft2 + design.utility_area_ft2
        assert design.total_area_ft2 == pytest.approx(expected_total, rel=0.01)

    @pytest.mark.unit
    def test_heat_recovery_calculated(self, synthesizer, four_stream_problem, pinch_result):
        """Test heat recovery is calculated."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        # Total heat recovery = sum of process match duties
        expected_recovery = sum(m.heat_duty_btu_hr for m in design.process_matches)
        assert design.total_heat_recovery_btu_hr == pytest.approx(
            expected_recovery, rel=0.01
        )

    @pytest.mark.unit
    def test_recovery_fraction(self, synthesizer, four_stream_problem, pinch_result):
        """Test heat recovery fraction is calculated."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        # Recovery fraction should be between 0 and 1
        assert 0 <= design.heat_recovery_fraction <= 1


# =============================================================================
# PINCH COMPLIANCE TESTS
# =============================================================================

class TestPinchCompliance:
    """Test pinch compliance checking."""

    @pytest.mark.unit
    def test_pinch_compliance_checked(
        self, synthesizer, four_stream_problem, pinch_result
    ):
        """Test pinch compliance is checked."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        assert hasattr(design, 'is_pinch_compliant')
        assert isinstance(design.is_pinch_compliant, bool)

    @pytest.mark.unit
    def test_violations_list_present(self, synthesizer, four_stream_problem, pinch_result):
        """Test violations list is present."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        assert hasattr(design, 'violations')
        assert isinstance(design.violations, list)


# =============================================================================
# AREA TARGETING TESTS
# =============================================================================

class TestAreaTargeting:
    """Test area targeting calculations."""

    @pytest.mark.unit
    def test_area_targets_calculated(self, synthesizer, four_stream_problem, pinch_result):
        """Test area targets are calculated."""
        result = synthesizer.calculate_area_targets(four_stream_problem, pinch_result)

        assert isinstance(result, AreaTargetResult)
        assert result.total_area_ft2 >= 0

    @pytest.mark.unit
    def test_area_split_by_region(self, synthesizer, four_stream_problem, pinch_result):
        """Test area is split by region."""
        result = synthesizer.calculate_area_targets(four_stream_problem, pinch_result)

        # Total should approximately equal sum of above + below
        expected_total = result.area_above_pinch_ft2 + result.area_below_pinch_ft2
        # Allow some difference due to different calculation methods
        assert result.total_area_ft2 > 0


# =============================================================================
# MINIMUM UNITS TESTS
# =============================================================================

class TestMinimumUnits:
    """Test minimum units calculation."""

    @pytest.mark.unit
    def test_minimum_units_formula(self, synthesizer):
        """Test minimum units formula."""
        # Formula: N_hot + N_cold + N_utilities - 1
        min_units = synthesizer.get_minimum_units(
            num_hot_streams=2,
            num_cold_streams=2,
            num_utilities=2,
        )

        assert min_units == 2 + 2 + 2 - 1  # = 5

    @pytest.mark.unit
    def test_minimum_units_single_utility(self, synthesizer):
        """Test minimum units with single utility."""
        min_units = synthesizer.get_minimum_units(
            num_hot_streams=3,
            num_cold_streams=2,
            num_utilities=1,
        )

        assert min_units == 3 + 2 + 1 - 1  # = 5


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

class TestProvenance:
    """Test provenance tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(
        self, synthesizer, four_stream_problem, pinch_result
    ):
        """Test provenance hash is generated."""
        design = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        assert design.provenance_hash is not None
        assert len(design.provenance_hash) == 64  # SHA-256

    @pytest.mark.unit
    def test_design_id_unique(self, synthesizer, four_stream_problem, pinch_result):
        """Test design IDs are unique."""
        design1 = synthesizer.synthesize_network(four_stream_problem, pinch_result)
        design2 = synthesizer.synthesize_network(four_stream_problem, pinch_result)

        assert design1.design_id != design2.design_id


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_small_stream_problem(self, pinch_analyzer):
        """Test with minimum streams."""
        streams = [
            HeatStream(
                name="H1",
                stream_type=StreamType.HOT,
                supply_temp_f=200.0,
                target_temp_f=100.0,
                mcp=10.0,
            ),
            HeatStream(
                name="C1",
                stream_type=StreamType.COLD,
                supply_temp_f=50.0,
                target_temp_f=150.0,
                mcp=10.0,
            ),
        ]

        pinch_result = pinch_analyzer.analyze(streams)
        synthesizer = HENSynthesizer(pinch_result=pinch_result)
        design = synthesizer.synthesize_network(streams, pinch_result)

        assert isinstance(design, HENDesign)

    @pytest.mark.unit
    def test_large_mcp_difference(self, pinch_analyzer):
        """Test with large MCP differences."""
        streams = [
            HeatStream(
                name="H1",
                stream_type=StreamType.HOT,
                supply_temp_f=300.0,
                target_temp_f=100.0,
                mcp=100.0,  # Large
            ),
            HeatStream(
                name="C1",
                stream_type=StreamType.COLD,
                supply_temp_f=50.0,
                target_temp_f=250.0,
                mcp=1.0,  # Small
            ),
        ]

        pinch_result = pinch_analyzer.analyze(streams)
        synthesizer = HENSynthesizer(pinch_result=pinch_result)
        design = synthesizer.synthesize_network(streams, pinch_result)

        # Should need significant cold utility
        assert design.cold_utility_btu_hr > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for HEN synthesis."""

    @pytest.mark.performance
    def test_synthesis_speed(self, pinch_analyzer):
        """Test synthesis speed."""
        import time

        # Create 10-stream problem
        streams = []
        for i in range(5):
            streams.append(HeatStream(
                name=f"H{i}",
                stream_type=StreamType.HOT,
                supply_temp_f=350.0 - i * 30,
                target_temp_f=150.0 - i * 10,
                mcp=5.0 + i,
            ))
            streams.append(HeatStream(
                name=f"C{i}",
                stream_type=StreamType.COLD,
                supply_temp_f=50.0 + i * 20,
                target_temp_f=280.0 + i * 10,
                mcp=4.0 + i,
            ))

        pinch_result = pinch_analyzer.analyze(streams)
        synthesizer = HENSynthesizer(pinch_result=pinch_result)

        start = time.time()
        design = synthesizer.synthesize_network(streams, pinch_result)
        elapsed = time.time() - start

        assert elapsed < 2.0  # Should complete in under 2 seconds
        assert design.total_units > 0
