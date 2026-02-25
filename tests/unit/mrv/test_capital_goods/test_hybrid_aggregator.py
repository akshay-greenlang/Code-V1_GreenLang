"""
Unit tests for HybridAggregatorEngine.

Tests hybrid aggregation, method prioritization, coverage analysis,
double-counting detection, hot-spot analysis, and intensity metrics.
"""

import pytest
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from greenlang.mrv.capital_goods.engines.hybrid_aggregator import (
    HybridAggregatorEngine,
    CalculationResult,
    MethodPriority,
    MaterialityQuadrant,
    AggregationInput,
    AggregatedResult,
    CoverageAnalysis,
    DoubleCountingIssue,
    HotSpot,
    IntensityMetrics,
)


class TestHybridAggregatorEngineSingleton:
    """Test singleton pattern."""

    def test_singleton_same_instance(self):
        """Test that multiple calls return same instance."""
        engine1 = HybridAggregatorEngine()
        engine2 = HybridAggregatorEngine()
        assert engine1 is engine2

    def test_singleton_with_reset(self):
        """Test singleton reset for testing."""
        engine1 = HybridAggregatorEngine()
        HybridAggregatorEngine._instance = None
        engine2 = HybridAggregatorEngine()
        assert engine1 is not engine2


class TestAggregate:
    """Test aggregate() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    @pytest.fixture
    def sample_results(self):
        """Create sample calculation results."""
        return [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="spend_based" if i < 3 else "supplier_specific",
                emissions=Decimal("100.0") if i < 3 else Decimal("80.0"),
                data_quality_score=3.0 if i < 3 else 4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Server",
                supplier_id=f"SUP{i % 2 + 1:03d}",
            )
            for i in range(6)
        ]

    def test_aggregate_success(self, engine, sample_results):
        """Test successful aggregation."""
        input_data = AggregationInput(
            calculation_results=sample_results,
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            organization_id="ORG001",
        )

        result = engine.aggregate(input_data)

        assert isinstance(result, AggregatedResult)
        assert result.total_emissions > 0
        assert result.asset_count == 6
        assert result.coverage_analysis is not None
        assert result.provenance_hash is not None

    def test_aggregate_prioritizes_high_quality(self, engine):
        """Test aggregation prioritizes high-quality methods."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="spend_based",
                emissions=Decimal("100.0"),
                data_quality_score=2.0,
                purchase_value=Decimal("10000.00"),
                asset_type="Server",
            ),
            CalculationResult(
                asset_id="A001",  # Same asset
                method="supplier_specific",
                emissions=Decimal("80.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Server",
            ),
        ]

        input_data = AggregationInput(
            calculation_results=results,
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            organization_id="ORG001",
        )

        result = engine.aggregate(input_data)

        # Should use supplier_specific (higher quality)
        assert result.total_emissions == Decimal("80.0")

    def test_aggregate_empty_results(self, engine):
        """Test aggregation with empty results."""
        input_data = AggregationInput(
            calculation_results=[],
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            organization_id="ORG001",
        )

        result = engine.aggregate(input_data)

        assert result.total_emissions == Decimal("0.0")
        assert result.asset_count == 0


class TestAggregateBatch:
    """Test aggregate_batch() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_aggregate_batch_multiple_periods(self, engine):
        """Test batch aggregation for multiple periods."""
        inputs = [
            AggregationInput(
                calculation_results=[
                    CalculationResult(
                        asset_id=f"A{j:03d}",
                        method="spend_based",
                        emissions=Decimal("50.0"),
                        data_quality_score=3.0,
                        purchase_value=Decimal("5000.00"),
                        asset_type="Equipment",
                    )
                    for j in range(5)
                ],
                reporting_period_start=date(2024, i, 1),
                reporting_period_end=date(2024, i, 28),
                organization_id="ORG001",
            )
            for i in range(1, 4)
        ]

        results = engine.aggregate_batch(inputs)

        assert len(results) == 3
        assert all(isinstance(r, AggregatedResult) for r in results)

    def test_aggregate_batch_empty_list(self, engine):
        """Test batch aggregation with empty list."""
        results = engine.aggregate_batch([])
        assert results == []


class TestPrioritizeMethods:
    """Test prioritize_methods() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_prioritize_single_method_per_asset(self, engine):
        """Test prioritization selects single highest-quality method per asset."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="spend_based",
                emissions=Decimal("100.0"),
                data_quality_score=2.0,
                purchase_value=Decimal("10000.00"),
                asset_type="Server",
            ),
            CalculationResult(
                asset_id="A001",
                method="average_data",
                emissions=Decimal("90.0"),
                data_quality_score=3.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Server",
            ),
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("85.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Server",
            ),
        ]

        prioritized = engine.prioritize_methods(results)

        assert len(prioritized) == 1
        assert prioritized[0].method == "supplier_specific"
        assert prioritized[0].emissions == Decimal("85.0")

    def test_prioritize_multiple_assets(self, engine):
        """Test prioritization works across multiple assets."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="spend_based" if i % 2 == 0 else "supplier_specific",
                emissions=Decimal("50.0"),
                data_quality_score=2.0 if i % 2 == 0 else 4.5,
                purchase_value=Decimal("5000.00"),
                asset_type="Equipment",
            )
            for i in range(10)
        ]

        prioritized = engine.prioritize_methods(results)

        assert len(prioritized) == 10
        # Odd indices should have supplier_specific
        supplier_specific = [r for r in prioritized if r.method == "supplier_specific"]
        assert len(supplier_specific) == 5

    def test_prioritize_tie_breaking(self, engine):
        """Test tie-breaking when methods have same DQI score."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="average_data",
                emissions=Decimal("100.0"),
                data_quality_score=3.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Server",
            ),
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("95.0"),
                data_quality_score=3.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Server",
            ),
        ]

        prioritized = engine.prioritize_methods(results)

        # Should prefer supplier_specific in tie (method hierarchy)
        assert len(prioritized) == 1
        assert prioritized[0].method == "supplier_specific"


class TestAnalyzeCoverage:
    """Test analyze_coverage() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_analyze_coverage_percentages(self, engine):
        """Test coverage analysis calculates correct percentages."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method=(
                    "supplier_specific" if i < 3
                    else "average_data" if i < 6
                    else "spend_based"
                ),
                emissions=Decimal("100.0"),
                data_quality_score=4.5 if i < 3 else 3.5 if i < 6 else 2.0,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            )
            for i in range(10)
        ]

        coverage = engine.analyze_coverage(results)

        assert isinstance(coverage, CoverageAnalysis)
        assert coverage.total_assets == 10
        assert coverage.supplier_specific_count == 3
        assert coverage.average_data_count == 3
        assert coverage.spend_based_count == 4
        assert coverage.supplier_specific_percentage == 30.0
        assert coverage.average_data_percentage == 30.0
        assert coverage.spend_based_percentage == 40.0

    def test_analyze_coverage_emissions_percentages(self, engine):
        """Test coverage analysis includes emissions percentages."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("500.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("50000.00"),
                asset_type="Equipment",
            ),
            CalculationResult(
                asset_id="A002",
                method="spend_based",
                emissions=Decimal("100.0"),
                data_quality_score=2.0,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            ),
        ]

        coverage = engine.analyze_coverage(results)

        # 500 / 600 = 83.33%
        assert abs(coverage.supplier_specific_emissions_pct - 83.33) < 0.01

    def test_analyze_coverage_empty_results(self, engine):
        """Test coverage analysis with empty results."""
        coverage = engine.analyze_coverage([])

        assert coverage.total_assets == 0
        assert coverage.supplier_specific_percentage == 0.0


class TestDetectDoubleCounting:
    """Test detect_double_counting() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_detect_category1_overlap(self, engine):
        """Test detection of Category 1 overlap."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Raw Material",  # Category 1 overlap
                category="capital_goods",
            ),
        ]

        issues = engine.detect_double_counting(results)

        assert len(issues) > 0
        assert any("Category 1" in issue.description for issue in issues)

    def test_detect_scope12_overlap(self, engine):
        """Test detection of Scope 1/2 overlap."""
        results = [
            CalculationResult(
                asset_id="A002",
                method="supplier_specific",
                emissions=Decimal("200.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("20000.00"),
                asset_type="Owned Equipment",  # Scope 1/2 overlap
                owned_operated=True,
            ),
        ]

        issues = engine.detect_double_counting(results)

        assert len(issues) > 0
        assert any("Scope 1" in issue.description or "Scope 2" in issue.description for issue in issues)

    def test_detect_no_overlap(self, engine):
        """Test no issues when no overlap exists."""
        results = [
            CalculationResult(
                asset_id="A003",
                method="supplier_specific",
                emissions=Decimal("150.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("15000.00"),
                asset_type="Third-Party Equipment",
                owned_operated=False,
                category="capital_goods",
            ),
        ]

        issues = engine.detect_double_counting(results)

        assert len(issues) == 0


class TestPreventDoubleCounting:
    """Test prevent_double_counting() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_prevent_removes_overlapping_assets(self, engine):
        """Test prevention removes overlapping assets."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Third-Party Equipment",
                owned_operated=False,
            ),
            CalculationResult(
                asset_id="A002",
                method="supplier_specific",
                emissions=Decimal("200.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("20000.00"),
                asset_type="Owned Equipment",
                owned_operated=True,  # Should be removed
            ),
        ]

        filtered = engine.prevent_double_counting(results)

        assert len(filtered) == 1
        assert filtered[0].asset_id == "A001"

    def test_prevent_keeps_valid_assets(self, engine):
        """Test prevention keeps valid assets."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
                owned_operated=False,
            )
            for i in range(5)
        ]

        filtered = engine.prevent_double_counting(results)

        assert len(filtered) == 5


class TestHotSpotAnalysis:
    """Test hot_spot_analysis() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_hot_spot_analysis_identifies_top_emitters(self, engine):
        """Test hot-spot analysis identifies top emitters."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("500.0"),  # High emitter
                data_quality_score=4.5,
                purchase_value=Decimal("50000.00"),
                asset_type="Server",
            ),
            CalculationResult(
                asset_id="A002",
                method="supplier_specific",
                emissions=Decimal("10.0"),  # Low emitter
                data_quality_score=4.5,
                purchase_value=Decimal("1000.00"),
                asset_type="Laptop",
            ),
        ]

        hot_spots = engine.hot_spot_analysis(results, top_n=1)

        assert len(hot_spots) == 1
        assert hot_spots[0].asset_id == "A001"
        assert hot_spots[0].emissions == Decimal("500.0")

    def test_hot_spot_analysis_top_n(self, engine):
        """Test hot-spot analysis respects top_n parameter."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal(str(100.0 - i * 10)),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            )
            for i in range(10)
        ]

        hot_spots = engine.hot_spot_analysis(results, top_n=5)

        assert len(hot_spots) == 5
        assert hot_spots[0].emissions > hot_spots[4].emissions


class TestParetoAnalysis:
    """Test pareto_analysis() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_pareto_analysis_80_20(self, engine):
        """Test Pareto analysis identifies 80/20 split."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("800.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("80000.00"),
                asset_type="Equipment",
            ),
        ] + [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="spend_based",
                emissions=Decimal("5.0"),
                data_quality_score=2.0,
                purchase_value=Decimal("500.00"),
                asset_type="Equipment",
            )
            for i in range(2, 42)  # 40 small emitters @ 5 each = 200
        ]

        pareto = engine.pareto_analysis(results)

        assert pareto["threshold_percentage"] == 80.0
        # 1 asset contributes 800/(800+200) = 80%
        assert pareto["assets_in_threshold"] == 1
        assert abs(pareto["cumulative_percentage_at_threshold"] - 80.0) < 1.0

    def test_pareto_analysis_cumulative_percentages(self, engine):
        """Test Pareto analysis calculates cumulative percentages."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal(str(100.0 - i * 10)),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            )
            for i in range(10)
        ]

        pareto = engine.pareto_analysis(results)

        assert "cumulative_percentages" in pareto
        assert len(pareto["cumulative_percentages"]) == 10
        assert pareto["cumulative_percentages"][-1] == 100.0  # Last should be 100%


class TestClassifyMateriality:
    """Test classify_materiality() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_classify_q1_high_emissions_high_quality(self, engine):
        """Test Q1 classification (high emissions, high quality)."""
        result = CalculationResult(
            asset_id="A001",
            method="supplier_specific",
            emissions=Decimal("1000.0"),  # High
            data_quality_score=4.5,  # High
            purchase_value=Decimal("100000.00"),
            asset_type="Equipment",
        )

        quadrant = engine.classify_materiality(result, median_emissions=Decimal("100.0"))

        assert quadrant == MaterialityQuadrant.Q1

    def test_classify_q2_high_emissions_low_quality(self, engine):
        """Test Q2 classification (high emissions, low quality)."""
        result = CalculationResult(
            asset_id="A002",
            method="spend_based",
            emissions=Decimal("1000.0"),  # High
            data_quality_score=2.0,  # Low
            purchase_value=Decimal("100000.00"),
            asset_type="Equipment",
        )

        quadrant = engine.classify_materiality(result, median_emissions=Decimal("100.0"))

        assert quadrant == MaterialityQuadrant.Q2

    def test_classify_q3_low_emissions_low_quality(self, engine):
        """Test Q3 classification (low emissions, low quality)."""
        result = CalculationResult(
            asset_id="A003",
            method="spend_based",
            emissions=Decimal("50.0"),  # Low
            data_quality_score=2.0,  # Low
            purchase_value=Decimal("5000.00"),
            asset_type="Equipment",
        )

        quadrant = engine.classify_materiality(result, median_emissions=Decimal("100.0"))

        assert quadrant == MaterialityQuadrant.Q3

    def test_classify_q4_low_emissions_high_quality(self, engine):
        """Test Q4 classification (low emissions, high quality)."""
        result = CalculationResult(
            asset_id="A004",
            method="supplier_specific",
            emissions=Decimal("50.0"),  # Low
            data_quality_score=4.5,  # High
            purchase_value=Decimal("5000.00"),
            asset_type="Equipment",
        )

        quadrant = engine.classify_materiality(result, median_emissions=Decimal("100.0"))

        assert quadrant == MaterialityQuadrant.Q4


class TestCalculateCapexVolatility:
    """Test calculate_capex_volatility() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_calculate_volatility_with_history(self, engine):
        """Test volatility calculation with historical data."""
        current_period_results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            )
            for i in range(10)
        ]

        historical_data = [
            {"period": "2023-Q1", "total_emissions": 800.0, "total_capex": 80000.0},
            {"period": "2023-Q2", "total_emissions": 900.0, "total_capex": 90000.0},
            {"period": "2023-Q3", "total_emissions": 850.0, "total_capex": 85000.0},
        ]

        volatility = engine.calculate_capex_volatility(
            current_period_results, historical_data
        )

        assert "current_total_emissions" in volatility
        assert "current_total_capex" in volatility
        assert "historical_average_emissions" in volatility
        assert "volatility_ratio" in volatility

    def test_calculate_volatility_no_history(self, engine):
        """Test volatility calculation without historical data."""
        current_period_results = [
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            ),
        ]

        volatility = engine.calculate_capex_volatility(current_period_results, [])

        assert volatility["historical_average_emissions"] == 0.0
        assert volatility["volatility_ratio"] is None


class TestYoYDecomposition:
    """Test yoy_decomposition() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_yoy_decomposition_with_two_periods(self, engine):
        """Test YoY decomposition with two periods."""
        current_results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("120.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("12000.00"),
                asset_type="Equipment",
            )
            for i in range(10)
        ]

        previous_results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            )
            for i in range(10)
        ]

        decomposition = engine.yoy_decomposition(current_results, previous_results)

        assert "total_change" in decomposition
        assert "activity_effect" in decomposition
        assert "intensity_effect" in decomposition
        assert decomposition["total_change"] == 200.0  # 1200 - 1000

    def test_yoy_decomposition_no_previous_period(self, engine):
        """Test YoY decomposition with no previous period."""
        current_results = [
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            ),
        ]

        decomposition = engine.yoy_decomposition(current_results, [])

        assert decomposition["total_change"] is None
        assert decomposition["activity_effect"] is None


class TestCalculateIntensityMetrics:
    """Test calculate_intensity_metrics() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_calculate_intensity_metrics(self, engine):
        """Test intensity metrics calculation."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            )
            for i in range(10)
        ]

        revenue = Decimal("1000000.00")
        employee_count = 100

        metrics = engine.calculate_intensity_metrics(
            results, revenue=revenue, employee_count=employee_count
        )

        assert isinstance(metrics, IntensityMetrics)
        assert metrics.emissions_per_dollar > 0
        assert metrics.emissions_per_employee > 0
        # 1000 / 1000000 = 0.001
        assert metrics.emissions_per_dollar == Decimal("0.001")
        # 1000 / 100 = 10
        assert metrics.emissions_per_employee == Decimal("10.0")

    def test_calculate_intensity_metrics_no_revenue(self, engine):
        """Test intensity metrics without revenue."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            ),
        ]

        metrics = engine.calculate_intensity_metrics(results)

        assert metrics.emissions_per_dollar is None
        assert metrics.emissions_per_employee is None


class TestAggregateByCategory:
    """Test aggregate_by_category() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_aggregate_by_category(self, engine):
        """Test aggregation by asset category."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Server" if i < 5 else "Network Equipment",
            )
            for i in range(10)
        ]

        aggregated = engine.aggregate_by_category(results)

        assert "Server" in aggregated
        assert "Network Equipment" in aggregated
        assert aggregated["Server"]["asset_count"] == 5
        assert aggregated["Network Equipment"]["asset_count"] == 5
        assert aggregated["Server"]["total_emissions"] == Decimal("500.0")


class TestAggregateByMethod:
    """Test aggregate_by_method() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_aggregate_by_method(self, engine):
        """Test aggregation by calculation method."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific" if i < 3 else "average_data" if i < 7 else "spend_based",
                emissions=Decimal("100.0"),
                data_quality_score=4.5 if i < 3 else 3.5 if i < 7 else 2.0,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
            )
            for i in range(10)
        ]

        aggregated = engine.aggregate_by_method(results)

        assert "supplier_specific" in aggregated
        assert "average_data" in aggregated
        assert "spend_based" in aggregated
        assert aggregated["supplier_specific"]["asset_count"] == 3
        assert aggregated["average_data"]["asset_count"] == 4
        assert aggregated["spend_based"]["asset_count"] == 3


class TestAggregateBySupplier:
    """Test aggregate_by_supplier() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_aggregate_by_supplier(self, engine):
        """Test aggregation by supplier."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
                supplier_id=f"SUP{(i % 3) + 1:03d}",
            )
            for i in range(9)
        ]

        aggregated = engine.aggregate_by_supplier(results)

        assert len(aggregated) == 3
        assert all(f"SUP{i:03d}" in aggregated for i in range(1, 4))
        assert all(agg["asset_count"] == 3 for agg in aggregated.values())


class TestAggregateByPeriod:
    """Test aggregate_by_period() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_aggregate_by_period_quarterly(self, engine):
        """Test aggregation by quarterly period."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
                purchase_date=date(2024, (i % 4) + 1, 1),
            )
            for i in range(12)
        ]

        aggregated = engine.aggregate_by_period(results, period_type="quarter")

        assert len(aggregated) == 4  # 4 quarters
        assert all("Q" in period for period in aggregated.keys())

    def test_aggregate_by_period_monthly(self, engine):
        """Test aggregation by monthly period."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
                purchase_date=date(2024, (i % 12) + 1, 1),
            )
            for i in range(24)
        ]

        aggregated = engine.aggregate_by_period(results, period_type="month")

        assert len(aggregated) == 12  # 12 months


class TestCalculateCombinedUncertainty:
    """Test calculate_combined_uncertainty() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        HybridAggregatorEngine._instance = None
        return HybridAggregatorEngine()

    def test_calculate_combined_uncertainty_low_quality_mix(self, engine):
        """Test combined uncertainty with mixed quality data."""
        results = [
            CalculationResult(
                asset_id="A001",
                method="supplier_specific",
                emissions=Decimal("500.0"),
                data_quality_score=4.5,  # High quality
                purchase_value=Decimal("50000.00"),
                asset_type="Equipment",
                uncertainty=5.0,  # 5%
            ),
            CalculationResult(
                asset_id="A002",
                method="spend_based",
                emissions=Decimal("500.0"),
                data_quality_score=2.0,  # Low quality
                purchase_value=Decimal("50000.00"),
                asset_type="Equipment",
                uncertainty=50.0,  # 50%
            ),
        ]

        combined_uncertainty = engine.calculate_combined_uncertainty(results)

        # Combined should be between 5% and 50%
        assert 5.0 < combined_uncertainty < 50.0

    def test_calculate_combined_uncertainty_uniform_quality(self, engine):
        """Test combined uncertainty with uniform quality data."""
        results = [
            CalculationResult(
                asset_id=f"A{i:03d}",
                method="supplier_specific",
                emissions=Decimal("100.0"),
                data_quality_score=4.5,
                purchase_value=Decimal("10000.00"),
                asset_type="Equipment",
                uncertainty=10.0,
            )
            for i in range(10)
        ]

        combined_uncertainty = engine.calculate_combined_uncertainty(results)

        # With uniform uncertainty, combined should be similar
        assert 8.0 < combined_uncertainty < 12.0

    def test_calculate_combined_uncertainty_empty_results(self, engine):
        """Test combined uncertainty with empty results."""
        combined_uncertainty = engine.calculate_combined_uncertainty([])

        assert combined_uncertainty == 0.0
