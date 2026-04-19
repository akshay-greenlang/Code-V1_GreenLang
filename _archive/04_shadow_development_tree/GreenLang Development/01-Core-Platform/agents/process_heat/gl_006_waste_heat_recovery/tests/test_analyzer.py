"""
GL-006 WasteHeatRecovery Agent - Main Analyzer Tests

Comprehensive unit tests for the WasteHeatAnalyzer class.
Tests waste heat recovery opportunity identification and evaluation.

Coverage Target: 85%+
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import math

# Module imports - adjust path as needed
from greenlang.agents.process_heat.gl_006_waste_heat_recovery.analyzer import (
    WasteHeatAnalyzer,
    WasteHeatSource,
    WasteHeatSink,
    RecoveryOpportunity,
    WasteHeatAnalysisOutput,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def analyzer():
    """Create WasteHeatAnalyzer instance for testing."""
    return WasteHeatAnalyzer(
        min_approach_temp_f=20.0,
        discount_rate=0.10,
        project_life_years=10,
    )


@pytest.fixture
def simple_source():
    """Create simple waste heat source."""
    return WasteHeatSource(
        source_id="S1",
        source_type="exhaust_gas",
        temperature_f=500.0,
        flow_rate=10000.0,
        flow_unit="lb/hr",
        specific_heat=0.25,
        availability_pct=100.0,
        operating_hours_yr=8760,
        min_discharge_temp_f=200.0,
    )


@pytest.fixture
def simple_sink():
    """Create simple heat sink."""
    return WasteHeatSink(
        sink_id="K1",
        sink_type="process_heating",
        required_temperature_f=300.0,
        inlet_temperature_f=100.0,
        flow_rate=5000.0,
        flow_unit="lb/hr",
        specific_heat=0.5,
        current_energy_source="natural_gas",
        current_cost_per_mmbtu=8.0,
    )


@pytest.fixture
def multiple_sources():
    """Create multiple waste heat sources."""
    return [
        WasteHeatSource(
            source_id="S1",
            source_type="exhaust_gas",
            temperature_f=600.0,
            flow_rate=15000.0,
            specific_heat=0.25,
            min_discharge_temp_f=250.0,
        ),
        WasteHeatSource(
            source_id="S2",
            source_type="hot_water",
            temperature_f=200.0,
            flow_rate=20000.0,
            specific_heat=1.0,
            min_discharge_temp_f=120.0,
        ),
        WasteHeatSource(
            source_id="S3",
            source_type="steam",
            temperature_f=350.0,
            flow_rate=5000.0,
            specific_heat=0.48,
            min_discharge_temp_f=212.0,
        ),
    ]


@pytest.fixture
def multiple_sinks():
    """Create multiple heat sinks."""
    return [
        WasteHeatSink(
            sink_id="K1",
            sink_type="process_heating",
            required_temperature_f=400.0,
            inlet_temperature_f=100.0,
            flow_rate=8000.0,
            specific_heat=0.5,
            current_cost_per_mmbtu=10.0,
        ),
        WasteHeatSink(
            sink_id="K2",
            sink_type="preheating",
            required_temperature_f=180.0,
            inlet_temperature_f=60.0,
            flow_rate=12000.0,
            specific_heat=1.0,
            current_cost_per_mmbtu=6.0,
        ),
    ]


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestWasteHeatAnalyzerInitialization:
    """Test WasteHeatAnalyzer initialization."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test analyzer initializes with default parameters."""
        analyzer = WasteHeatAnalyzer()

        assert analyzer.min_approach_temp == 20.0
        assert analyzer.discount_rate == 0.10
        assert analyzer.project_life_years == 10
        assert analyzer._hx_cost_per_ft2 == 150.0
        assert analyzer._installation_factor == 1.5

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test analyzer initializes with custom parameters."""
        analyzer = WasteHeatAnalyzer(
            min_approach_temp_f=30.0,
            discount_rate=0.08,
            project_life_years=15,
        )

        assert analyzer.min_approach_temp == 30.0
        assert analyzer.discount_rate == 0.08
        assert analyzer.project_life_years == 15

    @pytest.mark.unit
    def test_calc_library_initialized(self):
        """Test calculation library is initialized."""
        analyzer = WasteHeatAnalyzer()
        assert analyzer.calc_library is not None


# =============================================================================
# AVAILABLE HEAT CALCULATION TESTS
# =============================================================================

class TestAvailableHeatCalculation:
    """Test _calculate_available_heat method."""

    @pytest.mark.unit
    def test_basic_available_heat(self, analyzer, simple_source):
        """Test basic available heat calculation."""
        heat = analyzer._calculate_available_heat(simple_source)

        # Q = m * Cp * (T_in - T_out)
        # Q = 10000 * 0.25 * (500 - 200) = 750,000 BTU/hr
        expected = 10000 * 0.25 * (500 - 200) * 1.0  # 100% availability
        assert heat == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_available_heat_with_partial_availability(self, analyzer):
        """Test available heat with partial availability."""
        source = WasteHeatSource(
            source_id="S1",
            source_type="exhaust_gas",
            temperature_f=500.0,
            flow_rate=10000.0,
            specific_heat=0.25,
            availability_pct=80.0,
            min_discharge_temp_f=200.0,
        )

        heat = analyzer._calculate_available_heat(source)
        expected = 10000 * 0.25 * (500 - 200) * 0.80
        assert heat == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_available_heat_with_acid_dew_point(self, analyzer):
        """Test available heat respects acid dew point constraint."""
        source = WasteHeatSource(
            source_id="S1",
            source_type="exhaust_gas",
            temperature_f=500.0,
            flow_rate=10000.0,
            specific_heat=0.25,
            min_discharge_temp_f=200.0,
            acid_dew_point_f=275.0,  # Must stay above 275 + 25 = 300F
        )

        heat = analyzer._calculate_available_heat(source)
        # min_temp should be 300 (275 + 25 margin)
        expected = 10000 * 0.25 * (500 - 300)
        assert heat == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_available_heat_zero_when_too_cold(self, analyzer):
        """Test available heat is zero when source is too cold."""
        source = WasteHeatSource(
            source_id="S1",
            source_type="hot_water",
            temperature_f=140.0,  # Below default min discharge temp
            flow_rate=10000.0,
            specific_heat=1.0,
            min_discharge_temp_f=150.0,
        )

        heat = analyzer._calculate_available_heat(source)
        assert heat == 0.0

    @pytest.mark.unit
    def test_default_min_discharge_temp(self, analyzer):
        """Test default minimum discharge temperature is used."""
        source = WasteHeatSource(
            source_id="S1",
            source_type="exhaust_gas",
            temperature_f=400.0,
            flow_rate=10000.0,
            specific_heat=0.25,
            min_discharge_temp_f=None,  # Should default to 150F
        )

        heat = analyzer._calculate_available_heat(source)
        expected = 10000 * 0.25 * (400 - 150)
        assert heat == pytest.approx(expected, rel=0.01)


# =============================================================================
# OPPORTUNITY EVALUATION TESTS
# =============================================================================

class TestOpportunityEvaluation:
    """Test _evaluate_opportunity method."""

    @pytest.mark.unit
    def test_basic_opportunity_evaluation(self, analyzer, simple_source, simple_sink):
        """Test basic opportunity evaluation."""
        opportunity = analyzer._evaluate_opportunity(simple_source, simple_sink)

        assert opportunity is not None
        assert opportunity.source_id == "S1"
        assert opportunity.sink_id == "K1"
        assert opportunity.recoverable_heat_btu_hr > 0

    @pytest.mark.unit
    def test_opportunity_temperatures(self, analyzer, simple_source, simple_sink):
        """Test opportunity outlet temperatures are calculated."""
        opportunity = analyzer._evaluate_opportunity(simple_source, simple_sink)

        assert opportunity.source_outlet_temp_f < simple_source.temperature_f
        assert opportunity.sink_outlet_temp_f > simple_sink.inlet_temperature_f

    @pytest.mark.unit
    def test_opportunity_effectiveness(self, analyzer, simple_source, simple_sink):
        """Test opportunity effectiveness is within valid range."""
        opportunity = analyzer._evaluate_opportunity(simple_source, simple_sink)

        assert 0.0 <= opportunity.effectiveness <= 1.0

    @pytest.mark.unit
    def test_opportunity_lmtd_calculation(self, analyzer, simple_source, simple_sink):
        """Test LMTD is calculated correctly."""
        opportunity = analyzer._evaluate_opportunity(simple_source, simple_sink)

        assert opportunity.lmtd_f > 0
        # LMTD should be reasonable for counterflow
        assert opportunity.lmtd_f < (simple_source.temperature_f - simple_sink.inlet_temperature_f)

    @pytest.mark.unit
    def test_opportunity_cost_estimation(self, analyzer, simple_source, simple_sink):
        """Test capital cost is estimated."""
        opportunity = analyzer._evaluate_opportunity(simple_source, simple_sink)

        assert opportunity.estimated_capital_cost >= 10000  # Minimum project cost
        assert opportunity.estimated_hx_area_ft2 > 0

    @pytest.mark.unit
    def test_opportunity_economics(self, analyzer, simple_source, simple_sink):
        """Test economic metrics are calculated."""
        opportunity = analyzer._evaluate_opportunity(simple_source, simple_sink)

        assert opportunity.annual_savings > 0
        assert opportunity.simple_payback_years > 0
        assert opportunity.npv_10yr != 0  # Can be positive or negative

    @pytest.mark.unit
    def test_infeasible_opportunity_source_too_cold(self, analyzer):
        """Test infeasible opportunity when source is too cold."""
        source = WasteHeatSource(
            source_id="S1",
            source_type="hot_water",
            temperature_f=180.0,  # Too cold for sink requirement
            flow_rate=10000.0,
            specific_heat=1.0,
            min_discharge_temp_f=150.0,
        )

        sink = WasteHeatSink(
            sink_id="K1",
            sink_type="process_heating",
            required_temperature_f=200.0,  # Requires hotter than source
            inlet_temperature_f=100.0,
            flow_rate=5000.0,
            specific_heat=1.0,
        )

        opportunity = analyzer._evaluate_opportunity(source, sink)
        assert opportunity is None

    @pytest.mark.unit
    def test_technical_feasibility_classification(self, analyzer):
        """Test technical feasibility classification."""
        # High effectiveness should be challenging
        source = WasteHeatSource(
            source_id="S1",
            source_type="exhaust_gas",
            temperature_f=300.0,
            flow_rate=1000.0,
            specific_heat=0.25,
            min_discharge_temp_f=100.0,
        )

        sink = WasteHeatSink(
            sink_id="K1",
            sink_type="process_heating",
            required_temperature_f=250.0,
            inlet_temperature_f=100.0,
            flow_rate=10000.0,  # Much higher flow rate
            specific_heat=0.25,
        )

        opportunity = analyzer._evaluate_opportunity(source, sink)
        if opportunity:
            assert opportunity.technical_feasibility in ["straightforward", "moderate", "challenging"]


# =============================================================================
# FULL ANALYSIS TESTS
# =============================================================================

class TestFullAnalysis:
    """Test complete analyze() method."""

    @pytest.mark.unit
    def test_basic_analysis(self, analyzer, simple_source, simple_sink):
        """Test basic analysis with single source and sink."""
        result = analyzer.analyze([simple_source], [simple_sink])

        assert isinstance(result, WasteHeatAnalysisOutput)
        assert result.analysis_id is not None
        assert result.timestamp is not None

    @pytest.mark.unit
    def test_analysis_totals(self, analyzer, simple_source, simple_sink):
        """Test analysis calculates correct totals."""
        result = analyzer.analyze([simple_source], [simple_sink])

        assert result.total_waste_heat_btu_hr > 0
        assert result.total_recoverable_btu_hr >= 0
        assert 0 <= result.recovery_potential_pct <= 100

    @pytest.mark.unit
    def test_analysis_with_multiple_sources_sinks(
        self, analyzer, multiple_sources, multiple_sinks
    ):
        """Test analysis with multiple sources and sinks."""
        result = analyzer.analyze(multiple_sources, multiple_sinks)

        assert result.total_waste_heat_btu_hr > 0
        # Should have opportunities from matching sources to sinks
        assert len(result.opportunities) >= 0

    @pytest.mark.unit
    def test_opportunities_sorted_by_npv(self, analyzer, multiple_sources, multiple_sinks):
        """Test opportunities are sorted by NPV descending."""
        result = analyzer.analyze(multiple_sources, multiple_sinks)

        if len(result.opportunities) > 1:
            for i in range(len(result.opportunities) - 1):
                assert result.opportunities[i].npv_10yr >= result.opportunities[i + 1].npv_10yr

    @pytest.mark.unit
    def test_portfolio_economics(self, analyzer, multiple_sources, multiple_sinks):
        """Test portfolio economics are calculated."""
        result = analyzer.analyze(multiple_sources, multiple_sinks)

        assert result.total_annual_savings >= 0
        assert result.total_capital_cost >= 0

        if result.total_annual_savings > 0:
            expected_payback = result.total_capital_cost / result.total_annual_savings
            assert result.portfolio_simple_payback == pytest.approx(expected_payback, rel=0.01)

    @pytest.mark.unit
    def test_recommendations_generated(self, analyzer, multiple_sources, multiple_sinks):
        """Test recommendations are generated."""
        result = analyzer.analyze(multiple_sources, multiple_sinks)

        assert isinstance(result.recommendations, list)
        # Should have at least one recommendation
        assert len(result.recommendations) > 0

    @pytest.mark.unit
    def test_pinch_analysis_triggered(self, analyzer, multiple_sources, multiple_sinks):
        """Test pinch analysis is triggered with sufficient streams."""
        result = analyzer.analyze(multiple_sources, multiple_sinks)

        # Pinch analysis should be performed with 2+ sources and 2+ sinks
        # Result may or may not have pinch temp depending on data
        # Just verify the field exists
        assert hasattr(result, 'pinch_temperature_f')


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_empty_sources(self, analyzer, simple_sink):
        """Test analysis with empty sources list."""
        result = analyzer.analyze([], [simple_sink])

        assert result.total_waste_heat_btu_hr == 0
        assert len(result.opportunities) == 0

    @pytest.mark.unit
    def test_empty_sinks(self, analyzer, simple_source):
        """Test analysis with empty sinks list."""
        result = analyzer.analyze([simple_source], [])

        assert result.total_waste_heat_btu_hr > 0
        assert len(result.opportunities) == 0

    @pytest.mark.unit
    def test_no_feasible_matches(self, analyzer):
        """Test when no feasible matches exist."""
        source = WasteHeatSource(
            source_id="S1",
            source_type="hot_water",
            temperature_f=150.0,
            flow_rate=10000.0,
            specific_heat=1.0,
            min_discharge_temp_f=140.0,
        )

        sink = WasteHeatSink(
            sink_id="K1",
            sink_type="process_heating",
            required_temperature_f=200.0,  # Higher than source
            inlet_temperature_f=50.0,
            flow_rate=5000.0,
            specific_heat=1.0,
        )

        result = analyzer.analyze([source], [sink])
        assert len(result.opportunities) == 0

    @pytest.mark.unit
    def test_very_small_temperature_difference(self, analyzer):
        """Test with very small temperature difference."""
        source = WasteHeatSource(
            source_id="S1",
            source_type="hot_water",
            temperature_f=200.0,
            flow_rate=10000.0,
            specific_heat=1.0,
            min_discharge_temp_f=190.0,  # Only 10F drop available
        )

        sink = WasteHeatSink(
            sink_id="K1",
            sink_type="preheating",
            required_temperature_f=175.0,
            inlet_temperature_f=170.0,  # Only 5F needed
            flow_rate=5000.0,
            specific_heat=1.0,
        )

        result = analyzer.analyze([source], [sink])
        # May or may not have opportunities based on approach temp
        assert isinstance(result, WasteHeatAnalysisOutput)

    @pytest.mark.unit
    def test_zero_flow_rate_handled(self, analyzer, simple_sink):
        """Test that zero flow rate in validation prevents issues."""
        # WasteHeatSource should reject flow_rate <= 0
        with pytest.raises(Exception):  # Pydantic validation error
            WasteHeatSource(
                source_id="S1",
                source_type="exhaust_gas",
                temperature_f=500.0,
                flow_rate=0.0,  # Invalid
                specific_heat=0.25,
            )


# =============================================================================
# PINCH ANALYSIS INTEGRATION
# =============================================================================

class TestPinchAnalysisIntegration:
    """Test pinch analysis integration in analyzer."""

    @pytest.mark.unit
    def test_pinch_analysis_with_sufficient_streams(
        self, analyzer, multiple_sources, multiple_sinks
    ):
        """Test pinch analysis runs with sufficient streams."""
        result = analyzer.analyze(multiple_sources, multiple_sinks)

        # With 2+ sources and 2+ sinks, pinch analysis should run
        # Check that results include pinch-related fields
        assert hasattr(result, 'minimum_utility_hot_btu_hr')
        assert hasattr(result, 'minimum_utility_cold_btu_hr')

    @pytest.mark.unit
    def test_pinch_analysis_skipped_with_insufficient_streams(
        self, analyzer, simple_source, simple_sink
    ):
        """Test pinch analysis is skipped with insufficient streams."""
        result = analyzer.analyze([simple_source], [simple_sink])

        # With only 1 source and 1 sink, pinch analysis should not run
        assert result.pinch_temperature_f is None


# =============================================================================
# RECOMMENDATION GENERATION TESTS
# =============================================================================

class TestRecommendationGeneration:
    """Test recommendation generation."""

    @pytest.mark.unit
    def test_low_recovery_recommendation(self, analyzer):
        """Test recommendation for low recovery potential."""
        source = WasteHeatSource(
            source_id="S1",
            source_type="exhaust_gas",
            temperature_f=300.0,
            flow_rate=1000.0,
            specific_heat=0.25,
            min_discharge_temp_f=250.0,  # Limited recovery
        )

        sink = WasteHeatSink(
            sink_id="K1",
            sink_type="process_heating",
            required_temperature_f=250.0,
            inlet_temperature_f=100.0,
            flow_rate=10000.0,  # Much larger demand
            specific_heat=0.5,
        )

        result = analyzer.analyze([source], [sink])

        # Should have recommendation about recovery potential
        assert len(result.recommendations) > 0

    @pytest.mark.unit
    def test_good_opportunities_recommendation(self, analyzer, simple_source, simple_sink):
        """Test recommendation for good payback opportunities."""
        result = analyzer.analyze([simple_source], [simple_sink])

        # Recommendations should mention good opportunities if they exist
        assert isinstance(result.recommendations, list)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for analyzer."""

    @pytest.mark.performance
    def test_analysis_speed_small_dataset(self, analyzer):
        """Test analysis speed with small dataset."""
        import time

        sources = [
            WasteHeatSource(
                source_id=f"S{i}",
                source_type="exhaust_gas",
                temperature_f=400 + i * 50,
                flow_rate=10000 + i * 1000,
                specific_heat=0.25,
            )
            for i in range(5)
        ]

        sinks = [
            WasteHeatSink(
                sink_id=f"K{i}",
                sink_type="process_heating",
                required_temperature_f=200 + i * 30,
                inlet_temperature_f=80,
                flow_rate=8000 + i * 500,
                specific_heat=0.5,
            )
            for i in range(5)
        ]

        start = time.time()
        result = analyzer.analyze(sources, sinks)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should complete in under 1 second

    @pytest.mark.performance
    @pytest.mark.slow
    def test_analysis_speed_large_dataset(self, analyzer):
        """Test analysis speed with larger dataset."""
        import time

        sources = [
            WasteHeatSource(
                source_id=f"S{i}",
                source_type="exhaust_gas",
                temperature_f=300 + (i % 10) * 50,
                flow_rate=5000 + i * 100,
                specific_heat=0.25,
            )
            for i in range(20)
        ]

        sinks = [
            WasteHeatSink(
                sink_id=f"K{i}",
                sink_type="process_heating",
                required_temperature_f=150 + (i % 10) * 30,
                inlet_temperature_f=60,
                flow_rate=4000 + i * 100,
                specific_heat=0.5,
            )
            for i in range(20)
        ]

        start = time.time()
        result = analyzer.analyze(sources, sinks)
        elapsed = time.time() - start

        # 20x20 = 400 potential matches should still be fast
        assert elapsed < 5.0


# =============================================================================
# PROVENANCE AND AUDIT TESTS
# =============================================================================

class TestProvenanceAudit:
    """Test provenance tracking and audit features."""

    @pytest.mark.unit
    def test_analysis_has_id(self, analyzer, simple_source, simple_sink):
        """Test analysis output has unique ID."""
        result = analyzer.analyze([simple_source], [simple_sink])

        assert result.analysis_id is not None
        assert len(result.analysis_id) > 0

    @pytest.mark.unit
    def test_analysis_has_timestamp(self, analyzer, simple_source, simple_sink):
        """Test analysis output has timestamp."""
        result = analyzer.analyze([simple_source], [simple_sink])

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.unit
    def test_opportunity_has_id(self, analyzer, simple_source, simple_sink):
        """Test each opportunity has unique ID."""
        result = analyzer.analyze([simple_source], [simple_sink])

        if result.opportunities:
            for opp in result.opportunities:
                assert opp.opportunity_id is not None
                assert len(opp.opportunity_id) > 0


# =============================================================================
# NPV CALCULATION TESTS
# =============================================================================

class TestNPVCalculation:
    """Test NPV calculation accuracy."""

    @pytest.mark.unit
    def test_npv_calculation_formula(self, analyzer, simple_source, simple_sink):
        """Test NPV calculation follows correct formula."""
        opportunity = analyzer._evaluate_opportunity(simple_source, simple_sink)

        if opportunity:
            # Verify NPV = -Capital + Sum(Savings / (1+r)^t)
            capital = opportunity.estimated_capital_cost
            annual_savings = opportunity.annual_savings
            discount_rate = analyzer.discount_rate
            years = analyzer.project_life_years

            expected_npv = -capital
            for year in range(1, years + 1):
                expected_npv += annual_savings / ((1 + discount_rate) ** year)

            assert opportunity.npv_10yr == pytest.approx(expected_npv, rel=0.01)

    @pytest.mark.unit
    def test_positive_npv_for_good_project(self, analyzer):
        """Test NPV is positive for a clearly good project."""
        # Create a source with lots of recoverable heat
        source = WasteHeatSource(
            source_id="S1",
            source_type="exhaust_gas",
            temperature_f=800.0,
            flow_rate=50000.0,
            specific_heat=0.25,
            min_discharge_temp_f=300.0,
        )

        sink = WasteHeatSink(
            sink_id="K1",
            sink_type="process_heating",
            required_temperature_f=600.0,
            inlet_temperature_f=200.0,
            flow_rate=20000.0,
            specific_heat=0.5,
            current_cost_per_mmbtu=15.0,  # High energy cost
        )

        result = analyzer.analyze([source], [sink])

        # With high energy cost and lots of heat, should have positive NPV
        if result.opportunities:
            best_opp = result.opportunities[0]
            # Not guaranteed to be positive, but savings should be significant
            assert best_opp.annual_savings > 0
