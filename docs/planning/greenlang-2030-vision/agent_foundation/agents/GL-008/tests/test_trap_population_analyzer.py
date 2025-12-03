# -*- coding: utf-8 -*-
"""
Unit Tests for TrapPopulationAnalyzer

Comprehensive test suite covering all analyzer functionality including:
- Fleet health metrics calculation
- Failure rate trending
- Priority ranking
- Survey frequency optimization
- Spare parts inventory optimization
- Total cost of ownership analysis
- Pareto analysis
- Provenance tracking

Test Coverage Target: 90%+

Author: GL-BackendDeveloper
Date: December 2025
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import hashlib
import json

import sys
import os

# Add the calculators directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculators.trap_population_analyzer import (
    TrapPopulationAnalyzer,
    PopulationAnalysisConfig,
    TrapStatus,
    TrapType,
    PriorityLevel,
    TrendDirection,
    SurveyMethod,
    TrapRecord,
    FleetHealthMetrics,
    FailureRateTrend,
    PriorityRanking,
    SurveyFrequencyRecommendation,
    SparePartsRecommendation,
    TotalCostOfOwnership,
    PopulationAnalysisResult,
    ProvenanceTracker,
    create_trap_record,
    BASE_FAILURE_RATES,
    REPLACEMENT_COSTS,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def analyzer():
    """Create a default analyzer instance."""
    return TrapPopulationAnalyzer()


@pytest.fixture
def custom_config():
    """Create a custom configuration for testing."""
    return PopulationAnalysisConfig(
        failure_rate_threshold_percent=Decimal("12.0"),
        critical_loss_threshold_usd=Decimal("4000.0"),
        high_loss_threshold_usd=Decimal("1500.0"),
        survey_interval_base_days=180,
        spare_parts_safety_factor=Decimal("2.0"),
        typical_trap_lifetime_years=10
    )


@pytest.fixture
def analyzer_with_config(custom_config):
    """Create an analyzer with custom configuration."""
    return TrapPopulationAnalyzer(config=custom_config)


@pytest.fixture
def sample_trap_population():
    """Create a sample trap population for testing."""
    traps = []

    # Operating traps (healthy fleet)
    for i in range(50):
        traps.append(create_trap_record(
            trap_id=f"OPER-{i:03d}",
            trap_type="thermodynamic",
            status="operating",
            pressure_bar=10.0,
            annual_cost_usd=0.0,
            age_years=float(i % 8),
            manufacturer="Armstrong",
            model="TD52",
            system="steam_main"
        ))

    # Failed open traps (high cost)
    for i in range(10):
        traps.append(create_trap_record(
            trap_id=f"FAIL-OPEN-{i:03d}",
            trap_type="mechanical",
            status="failed_open",
            pressure_bar=15.0,
            annual_cost_usd=3000.0 + i * 500,
            age_years=7.0 + i * 0.5,
            manufacturer="Spirax",
            model="FT44",
            system="process_1"
        ))

    # Failed closed traps
    for i in range(5):
        traps.append(create_trap_record(
            trap_id=f"FAIL-CLOSED-{i:03d}",
            trap_type="thermostatic",
            status="failed_closed",
            pressure_bar=8.0,
            annual_cost_usd=500.0,
            age_years=5.0 + i,
            manufacturer="TLV",
            model="JH5",
            system="hvac"
        ))

    # Leaking traps
    for i in range(15):
        traps.append(create_trap_record(
            trap_id=f"LEAK-{i:03d}",
            trap_type="thermodynamic",
            status="leaking",
            pressure_bar=12.0,
            annual_cost_usd=1000.0 + i * 200,
            age_years=4.0 + i * 0.3,
            manufacturer="Armstrong",
            model="TD52",
            system="steam_main"
        ))

    # Unknown status traps
    for i in range(5):
        traps.append(create_trap_record(
            trap_id=f"UNK-{i:03d}",
            trap_type="venturi",
            status="unknown",
            pressure_bar=6.0,
            annual_cost_usd=0.0,
            age_years=2.0,
            manufacturer="Nicholson",
            model="V7",
            system="condensate"
        ))

    return traps


@pytest.fixture
def small_trap_population():
    """Create a small trap population for edge case testing."""
    return [
        create_trap_record(
            trap_id="SMALL-001",
            trap_type="thermodynamic",
            status="operating",
            pressure_bar=10.0,
            annual_cost_usd=0.0,
            age_years=3.0
        ),
        create_trap_record(
            trap_id="SMALL-002",
            trap_type="thermodynamic",
            status="failed_open",
            pressure_bar=10.0,
            annual_cost_usd=5000.0,
            age_years=8.0
        ),
    ]


# ============================================================================
# BASIC INITIALIZATION TESTS
# ============================================================================

class TestAnalyzerInitialization:
    """Test analyzer initialization."""

    def test_default_initialization(self, analyzer):
        """Test default analyzer initialization."""
        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.config.typical_trap_lifetime_years == 7

    def test_custom_config_initialization(self, analyzer_with_config, custom_config):
        """Test initialization with custom config."""
        assert analyzer_with_config.config.typical_trap_lifetime_years == 10
        assert analyzer_with_config.config.spare_parts_safety_factor == Decimal("2.0")

    def test_statistics_initialization(self, analyzer):
        """Test statistics are properly initialized."""
        stats = analyzer.get_statistics()
        assert stats["analysis_count"] == 0
        assert "supported_trap_types" in stats
        assert "supported_statuses" in stats


# ============================================================================
# FLEET HEALTH METRICS TESTS
# ============================================================================

class TestFleetHealthMetrics:
    """Test fleet health metrics calculations."""

    def test_health_score_calculation(self, analyzer, sample_trap_population):
        """Test fleet health score is calculated correctly."""
        result = analyzer.analyze_population(sample_trap_population)

        # 50 operating out of 85 total = ~58.8%
        expected_health = (50 / 85) * 100
        assert abs(float(result.fleet_metrics.health_score_percent) - expected_health) < 1.0

    def test_failure_rate_calculation(self, analyzer, sample_trap_population):
        """Test failure rate is calculated correctly."""
        result = analyzer.analyze_population(sample_trap_population)

        # 10 failed open + 5 failed closed + 15 leaking = 30 failed
        # 30 / 85 = ~35.3%
        expected_failure = (30 / 85) * 100
        assert abs(float(result.fleet_metrics.failure_rate_percent) - expected_failure) < 1.0

    def test_total_annual_loss(self, analyzer, sample_trap_population):
        """Test total annual loss is calculated correctly."""
        result = analyzer.analyze_population(sample_trap_population)

        # Calculate expected total
        expected_total = sum(float(t.annual_cost_usd) for t in sample_trap_population)
        assert abs(float(result.fleet_metrics.total_annual_loss_usd) - expected_total) < 1.0

    def test_trap_counts(self, analyzer, sample_trap_population):
        """Test trap counts are accurate."""
        result = analyzer.analyze_population(sample_trap_population)

        assert result.fleet_metrics.total_traps == 85
        assert result.fleet_metrics.operating_count == 50
        assert result.fleet_metrics.failed_count == 15  # 10 open + 5 closed
        assert result.fleet_metrics.leaking_count == 15
        assert result.fleet_metrics.unknown_count == 5

    def test_average_trap_age(self, analyzer, sample_trap_population):
        """Test average trap age calculation."""
        result = analyzer.analyze_population(sample_trap_population)

        ages = [float(t.age_years) for t in sample_trap_population if float(t.age_years) > 0]
        expected_avg = sum(ages) / len(ages) if ages else 0
        assert abs(float(result.fleet_metrics.average_trap_age_years) - expected_avg) < 0.5


# ============================================================================
# FAILURE TREND ANALYSIS TESTS
# ============================================================================

class TestFailureTrendAnalysis:
    """Test failure rate trend analysis."""

    def test_trends_by_trap_type(self, analyzer, sample_trap_population):
        """Test trends are generated for trap types."""
        result = analyzer.analyze_population(sample_trap_population)

        trap_type_trends = [
            t for t in result.failure_trends if t.category == "trap_type"
        ]
        assert len(trap_type_trends) > 0

    def test_trends_by_manufacturer(self, analyzer, sample_trap_population):
        """Test trends are generated for manufacturers."""
        result = analyzer.analyze_population(sample_trap_population)

        mfg_trends = [
            t for t in result.failure_trends if t.category == "manufacturer"
        ]
        # Should have trends for Armstrong, Spirax, TLV
        assert len(mfg_trends) >= 2

    def test_trends_by_age_group(self, analyzer, sample_trap_population):
        """Test trends are generated for age groups."""
        result = analyzer.analyze_population(sample_trap_population)

        age_trends = [
            t for t in result.failure_trends if t.category == "age_group"
        ]
        assert len(age_trends) > 0

    def test_trend_direction_classification(self, analyzer, sample_trap_population):
        """Test trend direction is properly classified."""
        result = analyzer.analyze_population(sample_trap_population)

        for trend in result.failure_trends:
            assert trend.trend_direction in [
                TrendDirection.IMPROVING,
                TrendDirection.STABLE,
                TrendDirection.DEGRADING,
                TrendDirection.CRITICAL
            ]

    def test_confidence_based_on_sample_size(self, analyzer, sample_trap_population):
        """Test confidence level is based on sample size."""
        result = analyzer.analyze_population(sample_trap_population)

        for trend in result.failure_trends:
            if trend.sample_size >= 30:
                assert trend.confidence_level >= Decimal("0.90")
            elif trend.sample_size >= 10:
                assert trend.confidence_level >= Decimal("0.75")


# ============================================================================
# PRIORITY RANKING TESTS
# ============================================================================

class TestPriorityRanking:
    """Test priority ranking generation."""

    def test_priority_ranking_generated(self, analyzer, sample_trap_population):
        """Test priority ranking is generated for all traps."""
        result = analyzer.analyze_population(sample_trap_population)

        assert len(result.priority_ranking) == len(sample_trap_population)

    def test_priority_ranking_sorted_by_score(self, analyzer, sample_trap_population):
        """Test priority ranking is sorted by score descending."""
        result = analyzer.analyze_population(sample_trap_population)

        scores = [float(r.priority_score) for r in result.priority_ranking]
        assert scores == sorted(scores, reverse=True)

    def test_failed_traps_higher_priority(self, analyzer, sample_trap_population):
        """Test failed traps have higher priority than operating."""
        result = analyzer.analyze_population(sample_trap_population)

        # First few should be failed traps with high costs
        top_5 = result.priority_ranking[:5]
        for ranking in top_5:
            assert ranking.priority != PriorityLevel.NONE

    def test_operating_traps_no_priority(self, analyzer, sample_trap_population):
        """Test operating traps have no priority."""
        result = analyzer.analyze_population(sample_trap_population)

        operating_rankings = [
            r for r in result.priority_ranking
            if r.trap_id.startswith("OPER-")
        ]

        for ranking in operating_rankings:
            assert ranking.priority == PriorityLevel.NONE
            assert float(ranking.priority_score) == 0

    def test_critical_priority_for_high_loss(self, analyzer, sample_trap_population):
        """Test critical priority for high-loss traps."""
        result = analyzer.analyze_population(sample_trap_population)

        # FAIL-OPEN traps have annual cost > 3000
        fail_open_rankings = [
            r for r in result.priority_ranking
            if r.trap_id.startswith("FAIL-OPEN-")
        ]

        # At least some should be critical
        critical_count = sum(
            1 for r in fail_open_rankings
            if r.priority == PriorityLevel.CRITICAL
        )
        assert critical_count > 0

    def test_deadline_assigned_for_action_items(self, analyzer, sample_trap_population):
        """Test deadlines are assigned for action items."""
        result = analyzer.analyze_population(sample_trap_population)

        for ranking in result.priority_ranking:
            if ranking.priority in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]:
                assert ranking.deadline is not None


# ============================================================================
# SURVEY FREQUENCY OPTIMIZATION TESTS
# ============================================================================

class TestSurveyFrequencyOptimization:
    """Test survey frequency recommendations."""

    def test_survey_recommendations_generated(self, analyzer, sample_trap_population):
        """Test survey recommendations are generated for each system."""
        result = analyzer.analyze_population(sample_trap_population)

        assert len(result.survey_recommendations) > 0

    def test_high_failure_rate_shorter_interval(self, analyzer, sample_trap_population):
        """Test high failure rate results in shorter survey interval."""
        result = analyzer.analyze_population(sample_trap_population)

        for rec in result.survey_recommendations:
            if float(rec.current_failure_rate) > 20:
                assert rec.recommended_interval_days <= 90

    def test_survey_method_appropriate(self, analyzer, sample_trap_population):
        """Test survey method is appropriate for failure rate."""
        result = analyzer.analyze_population(sample_trap_population)

        for rec in result.survey_recommendations:
            if float(rec.current_failure_rate) > 20:
                assert rec.method == SurveyMethod.COMBINED
            elif float(rec.current_failure_rate) > 15:
                assert rec.method == SurveyMethod.ULTRASONIC

    def test_net_benefit_calculated(self, analyzer, sample_trap_population):
        """Test net benefit is calculated for recommendations."""
        result = analyzer.analyze_population(sample_trap_population)

        for rec in result.survey_recommendations:
            assert rec.net_benefit_usd is not None
            # Net benefit = expected savings - survey cost
            expected = rec.expected_savings_usd - rec.annual_survey_cost_usd
            assert rec.net_benefit_usd == expected


# ============================================================================
# SPARE PARTS OPTIMIZATION TESTS
# ============================================================================

class TestSparePartsOptimization:
    """Test spare parts inventory recommendations."""

    def test_spare_parts_recommendations_generated(self, analyzer, sample_trap_population):
        """Test spare parts recommendations are generated."""
        result = analyzer.analyze_population(sample_trap_population)

        assert len(result.spare_parts) > 0

    def test_recommended_stock_includes_safety(self, analyzer, sample_trap_population):
        """Test recommended stock includes safety stock."""
        result = analyzer.analyze_population(sample_trap_population)

        for rec in result.spare_parts:
            assert rec.recommended_stock >= rec.safety_stock

    def test_reorder_point_set(self, analyzer, sample_trap_population):
        """Test reorder point is set appropriately."""
        result = analyzer.analyze_population(sample_trap_population)

        for rec in result.spare_parts:
            assert rec.reorder_point > 0
            assert rec.reorder_point <= rec.recommended_stock

    def test_estimated_cost_calculated(self, analyzer, sample_trap_population):
        """Test estimated inventory cost is calculated."""
        result = analyzer.analyze_population(sample_trap_population)

        for rec in result.spare_parts:
            assert float(rec.estimated_cost_usd) >= 0


# ============================================================================
# TOTAL COST OF OWNERSHIP TESTS
# ============================================================================

class TestTotalCostOfOwnership:
    """Test TCO analysis."""

    def test_tco_calculated_by_trap_type(self, analyzer, sample_trap_population):
        """Test TCO is calculated for each trap type present."""
        result = analyzer.analyze_population(sample_trap_population)

        assert len(result.tco_analysis) > 0

    def test_tco_includes_all_cost_components(self, analyzer, sample_trap_population):
        """Test TCO includes all cost components."""
        result = analyzer.analyze_population(sample_trap_population)

        for tco in result.tco_analysis:
            # Total should be sum of all components
            components = (
                float(tco.average_purchase_cost_usd) +
                float(tco.average_installation_cost_usd) +
                float(tco.annual_inspection_cost_usd) * 7 +  # lifetime
                float(tco.average_repair_cost_usd) * float(tco.expected_repairs_over_lifetime) +
                float(tco.steam_loss_cost_over_lifetime_usd)
            )
            # Allow for rounding differences
            assert abs(float(tco.total_tco_usd) - components) < 10

    def test_tco_per_year_calculated(self, analyzer, sample_trap_population):
        """Test annualized TCO is calculated."""
        result = analyzer.analyze_population(sample_trap_population)

        lifetime = analyzer.config.typical_trap_lifetime_years
        for tco in result.tco_analysis:
            expected_per_year = float(tco.total_tco_usd) / lifetime
            assert abs(float(tco.tco_per_year_usd) - expected_per_year) < 0.1

    def test_tco_recommendations_provided(self, analyzer, sample_trap_population):
        """Test TCO includes recommendations."""
        result = analyzer.analyze_population(sample_trap_population)

        for tco in result.tco_analysis:
            assert tco.recommendation is not None
            assert len(tco.recommendation) > 0


# ============================================================================
# PARETO ANALYSIS TESTS
# ============================================================================

class TestParetoAnalysis:
    """Test Pareto (80/20) analysis."""

    def test_pareto_analysis_generated(self, analyzer, sample_trap_population):
        """Test Pareto analysis is generated."""
        result = analyzer.analyze_population(sample_trap_population)

        assert result.pareto_analysis is not None
        assert "pareto_ratio" in result.pareto_analysis

    def test_pareto_identifies_top_contributors(self, analyzer, sample_trap_population):
        """Test Pareto identifies top loss contributors."""
        result = analyzer.analyze_population(sample_trap_population)

        assert "trap_ids_for_80_percent" in result.pareto_analysis
        assert len(result.pareto_analysis["trap_ids_for_80_percent"]) > 0

    def test_pareto_ratio_reasonable(self, analyzer, sample_trap_population):
        """Test Pareto ratio is reasonable (less than 100%)."""
        result = analyzer.analyze_population(sample_trap_population)

        ratio = result.pareto_analysis["pareto_ratio"]
        assert 0 <= ratio <= 100

    def test_pareto_insight_provided(self, analyzer, sample_trap_population):
        """Test Pareto provides insight string."""
        result = analyzer.analyze_population(sample_trap_population)

        assert "insight" in result.pareto_analysis
        assert "%" in result.pareto_analysis["insight"]


# ============================================================================
# PROVENANCE TRACKING TESTS
# ============================================================================

class TestProvenanceTracking:
    """Test provenance tracking and audit trail."""

    def test_provenance_hash_generated(self, analyzer, sample_trap_population):
        """Test provenance hash is generated."""
        result = analyzer.analyze_population(sample_trap_population)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_deterministic_results(self, analyzer, sample_trap_population):
        """Test same inputs produce same outputs (determinism)."""
        result1 = analyzer.analyze_population(sample_trap_population)
        result2 = analyzer.analyze_population(sample_trap_population)

        assert result1.fleet_metrics.total_traps == result2.fleet_metrics.total_traps
        assert result1.fleet_metrics.health_score_percent == result2.fleet_metrics.health_score_percent

    def test_timestamp_recorded(self, analyzer, sample_trap_population):
        """Test analysis timestamp is recorded."""
        before = datetime.now(timezone.utc)
        result = analyzer.analyze_population(sample_trap_population)
        after = datetime.now(timezone.utc)

        assert before <= result.analysis_timestamp <= after


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Test input validation and error handling."""

    def test_empty_population_raises_error(self, analyzer):
        """Test empty population raises error."""
        with pytest.raises(ValueError, match="Cannot analyze empty"):
            analyzer.analyze_population([])

    def test_single_trap_population(self, analyzer):
        """Test single trap population works."""
        single_trap = [create_trap_record(
            trap_id="SINGLE-001",
            trap_type="thermodynamic",
            status="operating",
            pressure_bar=10.0,
            annual_cost_usd=0.0
        )]

        result = analyzer.analyze_population(single_trap)
        assert result.fleet_metrics.total_traps == 1


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_operating_fleet(self, analyzer):
        """Test fleet with all operating traps."""
        traps = [
            create_trap_record(
                trap_id=f"HEALTHY-{i:03d}",
                trap_type="thermodynamic",
                status="operating",
                pressure_bar=10.0,
                annual_cost_usd=0.0
            )
            for i in range(20)
        ]

        result = analyzer.analyze_population(traps)

        assert result.fleet_metrics.health_score_percent == Decimal("100.0")
        assert result.fleet_metrics.failure_rate_percent == Decimal("0.0")

    def test_all_failed_fleet(self, analyzer):
        """Test fleet with all failed traps."""
        traps = [
            create_trap_record(
                trap_id=f"FAILED-{i:03d}",
                trap_type="thermodynamic",
                status="failed_open",
                pressure_bar=10.0,
                annual_cost_usd=1000.0 * i
            )
            for i in range(20)
        ]

        result = analyzer.analyze_population(traps)

        assert result.fleet_metrics.health_score_percent == Decimal("0.0")
        assert result.fleet_metrics.failure_rate_percent == Decimal("100.0")

    def test_mixed_trap_types(self, analyzer):
        """Test fleet with all trap types represented."""
        traps = []
        for trap_type in TrapType:
            for i in range(5):
                traps.append(create_trap_record(
                    trap_id=f"{trap_type.value}-{i:03d}",
                    trap_type=trap_type.value,
                    status="operating",
                    pressure_bar=10.0,
                    annual_cost_usd=0.0
                ))

        result = analyzer.analyze_population(traps)

        # Should have trends for each type
        type_trends = [
            t for t in result.failure_trends if t.category == "trap_type"
        ]
        assert len(type_trends) == len(TrapType)

    def test_zero_cost_fleet(self, analyzer):
        """Test fleet with zero annual costs."""
        traps = [
            create_trap_record(
                trap_id=f"ZERO-{i:03d}",
                trap_type="thermodynamic",
                status="operating",
                pressure_bar=10.0,
                annual_cost_usd=0.0
            )
            for i in range(10)
        ]

        result = analyzer.analyze_population(traps)

        assert result.fleet_metrics.total_annual_loss_usd == Decimal("0.00")
        assert result.pareto_analysis["pareto_ratio"] == 0


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization:
    """Test result serialization."""

    def test_to_dict_method(self, analyzer, sample_trap_population):
        """Test to_dict produces valid dictionary."""
        result = analyzer.analyze_population(sample_trap_population)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "fleet_metrics" in result_dict
        assert "top_5_priorities" in result_dict
        assert "pareto_analysis" in result_dict

    def test_json_serializable(self, analyzer, sample_trap_population):
        """Test result is JSON serializable."""
        result = analyzer.analyze_population(sample_trap_population)

        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert "fleet_metrics" in parsed


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

class TestHelperFunctions:
    """Test helper functions."""

    def test_create_trap_record(self):
        """Test create_trap_record helper function."""
        trap = create_trap_record(
            trap_id="HELPER-001",
            trap_type="thermodynamic",
            status="operating",
            pressure_bar=10.0,
            annual_cost_usd=500.0,
            age_years=5.0,
            manufacturer="Armstrong",
            model="TD52",
            location="Building A",
            system="steam_main"
        )

        assert trap.trap_id == "HELPER-001"
        assert trap.trap_type == TrapType.THERMODYNAMIC
        assert trap.status == TrapStatus.OPERATING
        assert trap.pressure_bar == Decimal("10.0")
        assert trap.annual_cost_usd == Decimal("500.0")
        assert trap.manufacturer == "Armstrong"

    def test_create_trap_record_minimal(self):
        """Test create_trap_record with minimal parameters."""
        trap = create_trap_record(
            trap_id="MINIMAL-001",
            trap_type="mechanical",
            status="failed_open",
            pressure_bar=5.0,
            annual_cost_usd=1000.0
        )

        assert trap.trap_id == "MINIMAL-001"
        assert trap.manufacturer == ""
        assert trap.age_years == Decimal("0")


# ============================================================================
# THREAD SAFETY TESTS
# ============================================================================

class TestThreadSafety:
    """Test thread safety of analyzer."""

    def test_analysis_count_thread_safe(self, analyzer, small_trap_population):
        """Test analysis count is thread-safe."""
        import threading

        def run_analysis():
            analyzer.analyze_population(small_trap_population)

        threads = [threading.Thread(target=run_analysis) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = analyzer.get_statistics()
        assert stats["analysis_count"] == 5


# ============================================================================
# CONSTANTS VALIDATION TESTS
# ============================================================================

class TestConstants:
    """Test constant values and reference data."""

    def test_base_failure_rates_defined(self):
        """Test base failure rates are defined for all trap types."""
        for trap_type in TrapType:
            assert trap_type in BASE_FAILURE_RATES

    def test_replacement_costs_defined(self):
        """Test replacement costs are defined for all trap types."""
        for trap_type in TrapType:
            assert trap_type in REPLACEMENT_COSTS

    def test_base_failure_rates_reasonable(self):
        """Test base failure rates are reasonable (5-20%)."""
        for trap_type, rate in BASE_FAILURE_RATES.items():
            assert Decimal("5") <= rate <= Decimal("20")

    def test_replacement_costs_reasonable(self):
        """Test replacement costs are reasonable ($100-$500)."""
        for trap_type, cost in REPLACEMENT_COSTS.items():
            assert Decimal("100") <= cost <= Decimal("500")


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance with large populations."""

    def test_large_population_analysis(self, analyzer):
        """Test analysis of large population completes in reasonable time."""
        import time

        # Create large population (1000 traps)
        traps = [
            create_trap_record(
                trap_id=f"PERF-{i:05d}",
                trap_type=list(TrapType)[i % 4].value,
                status=list(TrapStatus)[i % 5].value,
                pressure_bar=float(5 + i % 20),
                annual_cost_usd=float(i * 10) if i % 3 == 0 else 0.0,
                age_years=float(i % 10)
            )
            for i in range(1000)
        ]

        start = time.time()
        result = analyzer.analyze_population(traps)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds
        assert result.fleet_metrics.total_traps == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
