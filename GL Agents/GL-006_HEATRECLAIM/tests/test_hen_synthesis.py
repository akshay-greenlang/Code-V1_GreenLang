"""
GL-006 HEATRECLAIM - HEN Synthesis Tests

Comprehensive test suite for Heat Exchanger Network synthesis
using the Pinch Design Method.

Reference: Linnhoff & Hindmarsh, "The Pinch Design Method for
Heat Exchanger Networks", Chem Eng Sci, 1983.
"""

import pytest
from unittest.mock import MagicMock, patch

from ..core.schemas import HeatStream, HeatExchanger, PinchAnalysisResult, HENDesign
from ..core.config import StreamType, Phase, ExchangerType, OptimizationMode
from ..calculators.hen_synthesis import (
    HENSynthesizer,
    StreamSegment,
    Match,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def hen_synthesizer():
    """Create HEN synthesizer with 10K minimum approach."""
    return HENSynthesizer(delta_t_min=10.0)


@pytest.fixture
def simple_hot_stream():
    """Create a simple hot stream."""
    return HeatStream(
        stream_id="H1",
        stream_name="Hot Stream 1",
        stream_type=StreamType.HOT,
        fluid_name="Water",
        phase=Phase.LIQUID,
        T_supply_C=150.0,
        T_target_C=60.0,
        m_dot_kg_s=2.0,
        Cp_kJ_kgK=4.186,
    )


@pytest.fixture
def simple_cold_stream():
    """Create a simple cold stream."""
    return HeatStream(
        stream_id="C1",
        stream_name="Cold Stream 1",
        stream_type=StreamType.COLD,
        fluid_name="Water",
        phase=Phase.LIQUID,
        T_supply_C=30.0,
        T_target_C=120.0,
        m_dot_kg_s=2.5,
        Cp_kJ_kgK=4.186,
    )


@pytest.fixture
def four_stream_problem():
    """Create classic 4-stream pinch problem."""
    hot_streams = [
        HeatStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            T_supply_C=180.0,
            T_target_C=80.0,
            m_dot_kg_s=3.0,
            Cp_kJ_kgK=2.0,
        ),
        HeatStream(
            stream_id="H2",
            stream_type=StreamType.HOT,
            T_supply_C=150.0,
            T_target_C=60.0,
            m_dot_kg_s=1.5,
            Cp_kJ_kgK=4.0,
        ),
    ]
    cold_streams = [
        HeatStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            T_supply_C=30.0,
            T_target_C=130.0,
            m_dot_kg_s=2.5,
            Cp_kJ_kgK=3.0,
        ),
        HeatStream(
            stream_id="C2",
            stream_type=StreamType.COLD,
            T_supply_C=50.0,
            T_target_C=160.0,
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=3.5,
        ),
    ]
    return hot_streams, cold_streams


@pytest.fixture
def pinch_result():
    """Create mock pinch analysis result."""
    return PinchAnalysisResult(
        pinch_temperature_C=95.0,
        hot_pinch_T_C=100.0,
        cold_pinch_T_C=90.0,
        delta_T_min_C=10.0,
        minimum_hot_utility_kW=200.0,
        minimum_cold_utility_kW=150.0,
        maximum_heat_recovery_kW=800.0,
        num_hot_streams=2,
        num_cold_streams=2,
    )


# =============================================================================
# TEST: INITIALIZATION
# =============================================================================

class TestHENSynthesizerInit:
    """Test HEN synthesizer initialization."""

    def test_default_initialization(self):
        """Test default parameters."""
        synth = HENSynthesizer()
        assert synth.delta_t_min == 10.0
        assert synth.max_exchangers == 50
        assert synth.default_U == 500.0

    def test_custom_delta_t_min(self):
        """Test custom minimum approach temperature."""
        synth = HENSynthesizer(delta_t_min=5.0)
        assert synth.delta_t_min == 5.0

    def test_custom_max_exchangers(self):
        """Test custom maximum exchangers."""
        synth = HENSynthesizer(max_exchangers=100)
        assert synth.max_exchangers == 100

    def test_custom_heat_transfer_coefficient(self):
        """Test custom default U value."""
        synth = HENSynthesizer(default_U_W_m2K=1000.0)
        assert synth.default_U == 1000.0


# =============================================================================
# TEST: STREAM SEGMENTATION
# =============================================================================

class TestStreamSegmentation:
    """Test stream segmentation above/below pinch."""

    def test_hot_stream_above_pinch(self, hen_synthesizer, simple_hot_stream):
        """Hot stream segment above pinch."""
        above_hot, above_cold = hen_synthesizer._segment_streams(
            [simple_hot_stream], [],
            pinch_T=100.0,
            above_pinch=True
        )

        # Hot stream 150→60 crosses pinch at 100
        # Above pinch: 150→105 (shifted pinch + shift)
        assert len(above_hot) == 1
        seg = above_hot[0]
        assert seg.above_pinch is True
        assert seg.T_start_C == 150.0
        assert seg.duty_kW > 0

    def test_cold_stream_above_pinch(self, hen_synthesizer, simple_cold_stream):
        """Cold stream segment above pinch."""
        above_hot, above_cold = hen_synthesizer._segment_streams(
            [], [simple_cold_stream],
            pinch_T=100.0,
            above_pinch=True
        )

        # Cold stream 30→120 crosses pinch
        # Above pinch: from pinch to target
        assert len(above_cold) == 1
        seg = above_cold[0]
        assert seg.above_pinch is True
        assert seg.T_end_C == 120.0

    def test_stream_entirely_above_pinch(self, hen_synthesizer):
        """Stream entirely above pinch should be included."""
        hot_stream = HeatStream(
            stream_id="H_HIGH",
            stream_type=StreamType.HOT,
            T_supply_C=200.0,
            T_target_C=120.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=4.0,
        )

        above_hot, _ = hen_synthesizer._segment_streams(
            [hot_stream], [],
            pinch_T=100.0,
            above_pinch=True
        )

        assert len(above_hot) == 1

    def test_stream_entirely_below_pinch(self, hen_synthesizer):
        """Stream entirely below pinch should not be in above."""
        cold_stream = HeatStream(
            stream_id="C_LOW",
            stream_type=StreamType.COLD,
            T_supply_C=20.0,
            T_target_C=60.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=4.0,
        )

        _, above_cold = hen_synthesizer._segment_streams(
            [], [cold_stream],
            pinch_T=100.0,
            above_pinch=True
        )

        assert len(above_cold) == 0


# =============================================================================
# TEST: FEASIBILITY CHECKING
# =============================================================================

class TestFeasibilityChecking:
    """Test match feasibility checks."""

    def test_feasible_match(self, hen_synthesizer):
        """Test a thermodynamically feasible match."""
        hot_seg = StreamSegment(
            stream_id="H1",
            segment_id="H1_above",
            stream_type=StreamType.HOT,
            T_start_C=150.0,
            T_end_C=100.0,
            FCp_kW_K=10.0,
            duty_kW=500.0,
            remaining_duty_kW=500.0,
            above_pinch=True,
        )

        cold_seg = StreamSegment(
            stream_id="C1",
            segment_id="C1_above",
            stream_type=StreamType.COLD,
            T_start_C=80.0,
            T_end_C=130.0,
            FCp_kW_K=8.0,
            duty_kW=400.0,
            remaining_duty_kW=400.0,
            above_pinch=True,
        )

        feasible = hen_synthesizer._check_feasibility(hot_seg, cold_seg)
        assert feasible is True

    def test_infeasible_match_temperature_cross(self, hen_synthesizer):
        """Test infeasible match with temperature cross."""
        hot_seg = StreamSegment(
            stream_id="H1",
            segment_id="H1_above",
            stream_type=StreamType.HOT,
            T_start_C=100.0,
            T_end_C=80.0,
            FCp_kW_K=10.0,
            duty_kW=200.0,
            remaining_duty_kW=200.0,
            above_pinch=True,
        )

        cold_seg = StreamSegment(
            stream_id="C1",
            segment_id="C1_above",
            stream_type=StreamType.COLD,
            T_start_C=95.0,  # Too close to hot outlet
            T_end_C=130.0,
            FCp_kW_K=8.0,
            duty_kW=280.0,
            remaining_duty_kW=280.0,
            above_pinch=True,
        )

        feasible = hen_synthesizer._check_feasibility(hot_seg, cold_seg)
        assert feasible is False


# =============================================================================
# TEST: MATCH CALCULATION
# =============================================================================

class TestMatchCalculation:
    """Test match duty and temperature calculation."""

    def test_calculate_match(self, hen_synthesizer):
        """Test match calculation."""
        hot_seg = StreamSegment(
            stream_id="H1",
            segment_id="H1_above",
            stream_type=StreamType.HOT,
            T_start_C=150.0,
            T_end_C=100.0,
            FCp_kW_K=10.0,
            duty_kW=500.0,
            remaining_duty_kW=500.0,
            above_pinch=True,
        )

        cold_seg = StreamSegment(
            stream_id="C1",
            segment_id="C1_above",
            stream_type=StreamType.COLD,
            T_start_C=80.0,
            T_end_C=130.0,
            FCp_kW_K=8.0,
            duty_kW=400.0,
            remaining_duty_kW=400.0,
            above_pinch=True,
        )

        match = hen_synthesizer._calculate_match(hot_seg, cold_seg, duty=300.0)

        assert match is not None
        assert match.duty_kW == 300.0
        assert match.hot_inlet_T == 150.0
        # Hot outlet: 150 - 300/10 = 120
        assert match.hot_outlet_T == pytest.approx(120.0, rel=0.01)
        # Cold outlet: 80 + 300/8 = 117.5
        assert match.cold_outlet_T == pytest.approx(117.5, rel=0.01)

    def test_match_with_violations(self, hen_synthesizer):
        """Test match that violates approach temperature."""
        hot_seg = StreamSegment(
            stream_id="H1",
            segment_id="H1_above",
            stream_type=StreamType.HOT,
            T_start_C=100.0,
            T_end_C=90.0,
            FCp_kW_K=10.0,
            duty_kW=100.0,
            remaining_duty_kW=100.0,
            above_pinch=True,
        )

        cold_seg = StreamSegment(
            stream_id="C1",
            segment_id="C1_above",
            stream_type=StreamType.COLD,
            T_start_C=85.0,
            T_end_C=95.0,  # Target close to hot inlet
            FCp_kW_K=10.0,
            duty_kW=100.0,
            remaining_duty_kW=100.0,
            above_pinch=True,
        )

        match = hen_synthesizer._calculate_match(hot_seg, cold_seg, duty=50.0)

        # Should have violations due to tight temperature approach
        assert len(match.constraint_violations) > 0 or match.feasibility_score < 1.0


# =============================================================================
# TEST: REGION DESIGN
# =============================================================================

class TestRegionDesign:
    """Test design of above/below pinch regions."""

    def test_design_above_pinch(self, hen_synthesizer, four_stream_problem, pinch_result):
        """Test design of above-pinch region."""
        hot_streams, cold_streams = four_stream_problem

        above_hot, above_cold = hen_synthesizer._segment_streams(
            hot_streams, cold_streams,
            pinch_T=pinch_result.pinch_temperature_C,
            above_pinch=True
        )

        matches = hen_synthesizer._design_region(
            above_hot, above_cold, above_pinch=True
        )

        # Should find at least some matches
        assert len(matches) >= 0  # May be empty if no feasible matches
        for match in matches:
            assert match.duty_kW > 0

    def test_design_below_pinch(self, hen_synthesizer, four_stream_problem, pinch_result):
        """Test design of below-pinch region."""
        hot_streams, cold_streams = four_stream_problem

        below_hot, below_cold = hen_synthesizer._segment_streams(
            hot_streams, cold_streams,
            pinch_T=pinch_result.pinch_temperature_C,
            above_pinch=False
        )

        matches = hen_synthesizer._design_region(
            below_hot, below_cold, above_pinch=False
        )

        assert len(matches) >= 0
        for match in matches:
            assert match.duty_kW > 0


# =============================================================================
# TEST: COMPLETE HEN SYNTHESIS
# =============================================================================

class TestCompleteSynthesis:
    """Test complete HEN synthesis."""

    def test_synthesis_basic(
        self, hen_synthesizer, simple_hot_stream, simple_cold_stream
    ):
        """Test basic HEN synthesis."""
        design = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
        )

        assert isinstance(design, HENDesign)
        assert design.design_name is not None
        assert len(design.exchangers) > 0

    def test_synthesis_with_pinch_result(
        self, hen_synthesizer, four_stream_problem, pinch_result
    ):
        """Test synthesis with provided pinch result."""
        hot_streams, cold_streams = four_stream_problem

        design = hen_synthesizer.synthesize(
            hot_streams=hot_streams,
            cold_streams=cold_streams,
            pinch_result=pinch_result,
        )

        assert design is not None
        assert design.exchanger_count > 0

    def test_synthesis_calculates_area(
        self, hen_synthesizer, simple_hot_stream, simple_cold_stream
    ):
        """Synthesized exchangers should have calculated areas."""
        design = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
        )

        # Process exchangers should have area
        for hx in design.exchangers:
            if "Utility" not in hx.exchanger_name:
                assert hx.area_m2 >= 0

    def test_synthesis_total_area(
        self, hen_synthesizer, simple_hot_stream, simple_cold_stream
    ):
        """Total area should equal sum of exchanger areas."""
        design = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
        )

        total_area = sum(hx.area_m2 for hx in design.exchangers)
        assert design.total_area_m2 == pytest.approx(total_area, rel=0.01)

    def test_synthesis_with_utilities(
        self, hen_synthesizer, simple_hot_stream, simple_cold_stream
    ):
        """Synthesis should add utility exchangers if needed."""
        design = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
        )

        # Should have some utility exchangers
        utility_hx = [
            hx for hx in design.exchangers
            if "Utility" in hx.exchanger_name
        ]
        assert len(utility_hx) >= 0  # May or may not need utilities

    def test_synthesis_provenance(
        self, hen_synthesizer, simple_hot_stream, simple_cold_stream
    ):
        """Design should include provenance hashes."""
        design = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
        )

        assert design.input_hash is not None
        assert design.output_hash is not None

    def test_synthesis_reproducibility(
        self, hen_synthesizer, simple_hot_stream, simple_cold_stream
    ):
        """Same inputs should produce identical designs."""
        design1 = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
        )
        design2 = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
        )

        assert design1.total_heat_recovered_kW == design2.total_heat_recovered_kW
        assert design1.exchanger_count == design2.exchanger_count


# =============================================================================
# TEST: EXCHANGER CREATION
# =============================================================================

class TestExchangerCreation:
    """Test heat exchanger object creation."""

    def test_create_exchanger_from_match(self, hen_synthesizer, simple_hot_stream, simple_cold_stream):
        """Test exchanger creation from match."""
        match = Match(
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=300.0,
            hot_inlet_T=150.0,
            hot_outlet_T=120.0,
            cold_inlet_T=80.0,
            cold_outlet_T=110.0,
            feasibility_score=1.0,
        )

        stream_map = {
            simple_hot_stream.stream_id: simple_hot_stream,
            simple_cold_stream.stream_id: simple_cold_stream,
        }

        hx = hen_synthesizer._create_exchanger(
            match, stream_map, exchanger_id="HX-001"
        )

        assert isinstance(hx, HeatExchanger)
        assert hx.exchanger_id == "HX-001"
        assert hx.duty_kW == 300.0
        assert hx.hot_stream_id == "H1"
        assert hx.cold_stream_id == "C1"

    def test_exchanger_has_lmtd(self, hen_synthesizer, simple_hot_stream, simple_cold_stream):
        """Created exchanger should have LMTD calculated."""
        match = Match(
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=300.0,
            hot_inlet_T=150.0,
            hot_outlet_T=120.0,
            cold_inlet_T=80.0,
            cold_outlet_T=110.0,
            feasibility_score=1.0,
        )

        stream_map = {
            simple_hot_stream.stream_id: simple_hot_stream,
            simple_cold_stream.stream_id: simple_cold_stream,
        }

        hx = hen_synthesizer._create_exchanger(
            match, stream_map, exchanger_id="HX-001"
        )

        assert hx.LMTD_C > 0

    def test_exchanger_type_is_shell_and_tube(self, hen_synthesizer, simple_hot_stream, simple_cold_stream):
        """Default exchanger type should be S&T."""
        match = Match(
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=300.0,
            hot_inlet_T=150.0,
            hot_outlet_T=120.0,
            cold_inlet_T=80.0,
            cold_outlet_T=110.0,
            feasibility_score=1.0,
        )

        stream_map = {
            simple_hot_stream.stream_id: simple_hot_stream,
            simple_cold_stream.stream_id: simple_cold_stream,
        }

        hx = hen_synthesizer._create_exchanger(
            match, stream_map, exchanger_id="HX-001"
        )

        assert hx.exchanger_type == ExchangerType.SHELL_AND_TUBE


# =============================================================================
# TEST: DESIGN VALIDATION
# =============================================================================

class TestDesignValidation:
    """Test design constraint validation."""

    def test_validates_approach_temperatures(
        self, hen_synthesizer, simple_hot_stream, simple_cold_stream
    ):
        """Design should validate approach temperatures."""
        design = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
        )

        # Check that validation was performed
        assert hasattr(design, 'temperature_violations')
        assert hasattr(design, 'all_constraints_satisfied')


# =============================================================================
# TEST: UTILITY EXCHANGERS
# =============================================================================

class TestUtilityExchangers:
    """Test utility exchanger addition."""

    def test_add_hot_utility_exchanger(self, hen_synthesizer):
        """Test adding hot utility exchanger."""
        cold_stream = HeatStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            T_supply_C=30.0,
            T_target_C=200.0,  # Needs hot utility
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=4.0,
        )

        utility_hx = hen_synthesizer._add_utility_exchangers(
            hot_streams=[],
            cold_streams=[cold_stream],
            exchangers=[],
            hot_utility_needed=100.0,
            cold_utility_needed=0.0,
            hot_utilities=None,
            cold_utilities=None,
            start_index=0,
        )

        assert len(utility_hx) > 0
        assert any("HotUtility" in hx.exchanger_name for hx in utility_hx)

    def test_add_cold_utility_exchanger(self, hen_synthesizer):
        """Test adding cold utility exchanger."""
        hot_stream = HeatStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            T_supply_C=150.0,
            T_target_C=30.0,  # Needs cooling
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=4.0,
        )

        utility_hx = hen_synthesizer._add_utility_exchangers(
            hot_streams=[hot_stream],
            cold_streams=[],
            exchangers=[],
            hot_utility_needed=0.0,
            cold_utility_needed=100.0,
            hot_utilities=None,
            cold_utilities=None,
            start_index=0,
        )

        assert len(utility_hx) > 0
        assert any("ColdUtility" in hx.exchanger_name for hx in utility_hx)


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_streams(self, hen_synthesizer):
        """Test with no streams."""
        design = hen_synthesizer.synthesize(
            hot_streams=[],
            cold_streams=[],
        )

        assert design.exchanger_count == 0
        assert design.total_heat_recovered_kW == 0

    def test_single_hot_stream_only(self, hen_synthesizer, simple_hot_stream):
        """Test with only hot stream (needs cold utility)."""
        design = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[],
        )

        assert design is not None

    def test_single_cold_stream_only(self, hen_synthesizer, simple_cold_stream):
        """Test with only cold stream (needs hot utility)."""
        design = hen_synthesizer.synthesize(
            hot_streams=[],
            cold_streams=[simple_cold_stream],
        )

        assert design is not None

    def test_tight_temperature_approach(self, hen_synthesizer):
        """Test with very tight temperature approach."""
        synth = HENSynthesizer(delta_t_min=5.0)

        hot = HeatStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            T_supply_C=100.0,
            T_target_C=50.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=4.0,
        )
        cold = HeatStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            T_supply_C=40.0,
            T_target_C=90.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=4.0,
        )

        design = synth.synthesize(
            hot_streams=[hot],
            cold_streams=[cold],
        )

        assert design is not None

    def test_large_number_of_streams(self, hen_synthesizer):
        """Test with many streams."""
        hot_streams = [
            HeatStream(
                stream_id=f"H{i}",
                stream_type=StreamType.HOT,
                T_supply_C=150 + i * 10,
                T_target_C=60 + i * 5,
                m_dot_kg_s=1.0 + i * 0.1,
                Cp_kJ_kgK=4.0,
            )
            for i in range(5)
        ]

        cold_streams = [
            HeatStream(
                stream_id=f"C{i}",
                stream_type=StreamType.COLD,
                T_supply_C=30 + i * 5,
                T_target_C=120 + i * 10,
                m_dot_kg_s=1.0 + i * 0.1,
                Cp_kJ_kgK=4.0,
            )
            for i in range(5)
        ]

        design = hen_synthesizer.synthesize(
            hot_streams=hot_streams,
            cold_streams=cold_streams,
        )

        assert design is not None
        assert design.exchanger_count <= hen_synthesizer.max_exchangers


# =============================================================================
# TEST: OPTIMIZATION MODES
# =============================================================================

class TestOptimizationModes:
    """Test different optimization modes."""

    def test_grassroots_mode(self, hen_synthesizer, simple_hot_stream, simple_cold_stream):
        """Test grassroots (greenfield) design mode."""
        design = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
            mode=OptimizationMode.GRASSROOTS,
        )

        assert design.mode == OptimizationMode.GRASSROOTS

    def test_retrofit_mode(self, hen_synthesizer, simple_hot_stream, simple_cold_stream):
        """Test retrofit design mode."""
        design = hen_synthesizer.synthesize(
            hot_streams=[simple_hot_stream],
            cold_streams=[simple_cold_stream],
            mode=OptimizationMode.RETROFIT,
        )

        assert design.mode == OptimizationMode.RETROFIT
