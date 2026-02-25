"""
Unit tests for HybridAggregatorEngine (AGENT-MRV-014).

Tests cover:
- Singleton pattern
- Method selection logic
- Coverage analysis
- Hot-spot analysis
- Boundary enforcement
- Overlap detection
- Gap filling
- Weighted DQI
- YoY decomposition
- Intensity metrics
- Aggregation
- Health checks
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

try:
    from greenlang.purchased_goods_services.hybrid_aggregator import (
        HybridAggregatorEngine,
        HybridInput,
        HybridOutput,
        MethodResult,
        CoverageLevel,
        MaterialityQuadrant,
        DecompositionAnalysis,
    )
except ImportError:
    pytest.skip("HybridAggregatorEngine not available", allow_module_level=True)


class TestHybridAggregatorSingleton:
    """Test singleton pattern for HybridAggregatorEngine."""

    def test_singleton_same_instance(self):
        """Test that get_instance returns same instance."""
        engine1 = HybridAggregatorEngine.get_instance()
        engine2 = HybridAggregatorEngine.get_instance()
        assert engine1 is engine2

    def test_singleton_thread_safe(self):
        """Test thread-safe singleton creation."""
        import threading
        instances = []

        def get_instance():
            instances.append(HybridAggregatorEngine.get_instance())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

    def test_singleton_reset(self):
        """Test singleton reset functionality."""
        engine1 = HybridAggregatorEngine.get_instance()
        HybridAggregatorEngine.reset_instance()
        engine2 = HybridAggregatorEngine.get_instance()
        assert engine1 is not engine2


class TestMethodSelection:
    """Test method selection logic (supplier > avgdata > spend)."""

    def test_supplier_specific_preferred(self):
        """Test supplier-specific method is preferred when available."""
        engine = HybridAggregatorEngine.get_instance()

        methods_available = ["supplier", "avgdata", "spend"]
        selected = engine.select_method(methods_available)

        assert selected == "supplier"

    def test_avgdata_when_supplier_unavailable(self):
        """Test avgdata method selected when supplier unavailable."""
        engine = HybridAggregatorEngine.get_instance()

        methods_available = ["avgdata", "spend"]
        selected = engine.select_method(methods_available)

        assert selected == "avgdata"

    def test_spend_only_fallback(self):
        """Test spend-based method as last fallback."""
        engine = HybridAggregatorEngine.get_instance()

        methods_available = ["spend"]
        selected = engine.select_method(methods_available)

        assert selected == "spend"

    def test_all_three_methods_available(self):
        """Test method selection when all three available."""
        engine = HybridAggregatorEngine.get_instance()

        methods_available = ["supplier", "avgdata", "spend"]
        selected = engine.select_method(methods_available)

        assert selected == "supplier"

    def test_no_methods_available_raises_error(self):
        """Test error when no methods available."""
        engine = HybridAggregatorEngine.get_instance()

        with pytest.raises(ValueError, match="No calculation methods available"):
            engine.select_method([])

    def test_method_priority_ordering(self):
        """Test method priority is correctly ordered."""
        engine = HybridAggregatorEngine.get_instance()

        priority = engine.get_method_priority()

        assert priority[0] == "supplier"
        assert priority[1] == "avgdata"
        assert priority[2] == "spend"


class TestCoverageAnalysis:
    """Test coverage analysis and classification."""

    def test_full_coverage_classification(self):
        """Test FULL coverage classification (>=95%)."""
        engine = HybridAggregatorEngine.get_instance()

        level = engine.classify_coverage(Decimal("97.5"))

        assert level == CoverageLevel.FULL

    def test_high_coverage_classification(self):
        """Test HIGH coverage classification (67-95%)."""
        engine = HybridAggregatorEngine.get_instance()

        level = engine.classify_coverage(Decimal("80.0"))

        assert level == CoverageLevel.HIGH

    def test_medium_coverage_classification(self):
        """Test MEDIUM coverage classification (40-67%)."""
        engine = HybridAggregatorEngine.get_instance()

        level = engine.classify_coverage(Decimal("55.0"))

        assert level == CoverageLevel.MEDIUM

    def test_low_coverage_classification(self):
        """Test LOW coverage classification (20-40%)."""
        engine = HybridAggregatorEngine.get_instance()

        level = engine.classify_coverage(Decimal("30.0"))

        assert level == CoverageLevel.LOW

    def test_minimal_coverage_classification(self):
        """Test MINIMAL coverage classification (<20%)."""
        engine = HybridAggregatorEngine.get_instance()

        level = engine.classify_coverage(Decimal("15.0"))

        assert level == CoverageLevel.MINIMAL

    def test_compute_coverage_percentage(self):
        """Test coverage percentage computation."""
        engine = HybridAggregatorEngine.get_instance()

        total_spend = Decimal("1000000")
        covered_spend = Decimal("750000")

        coverage = engine.compute_coverage(
            covered_spend=covered_spend,
            total_spend=total_spend,
        )

        assert coverage == Decimal("75.0")


class TestHotSpotAnalysis:
    """Test hot-spot and materiality analysis."""

    def test_pareto_ranking_80_20(self):
        """Test Pareto ranking (80/20 rule)."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "emissions": Decimal("500")},
            {"id": "B", "emissions": Decimal("300")},
            {"id": "C", "emissions": Decimal("150")},
            {"id": "D", "emissions": Decimal("40")},
            {"id": "E", "emissions": Decimal("10")},
        ]

        ranked = engine.pareto_rank(items, emission_key="emissions")

        # Top items should contribute ~80% of total (1000 total)
        top_items = [r for r in ranked if r["pareto_group"] == "top_20"]
        top_emissions = sum(item["emissions"] for item in top_items)

        assert top_emissions >= Decimal("800")  # At least 80% of total

    def test_materiality_quadrant_q1_high_high(self):
        """Test Quadrant 1 (high emissions, high spend)."""
        engine = HybridAggregatorEngine.get_instance()

        quadrant = engine.classify_materiality_quadrant(
            emissions=Decimal("1000"),
            spend=Decimal("500000"),
            emissions_threshold=Decimal("500"),
            spend_threshold=Decimal("250000"),
        )

        assert quadrant == MaterialityQuadrant.Q1_HIGH_HIGH

    def test_materiality_quadrant_q2_high_low(self):
        """Test Quadrant 2 (high emissions, low spend)."""
        engine = HybridAggregatorEngine.get_instance()

        quadrant = engine.classify_materiality_quadrant(
            emissions=Decimal("1000"),
            spend=Decimal("100000"),
            emissions_threshold=Decimal("500"),
            spend_threshold=Decimal("250000"),
        )

        assert quadrant == MaterialityQuadrant.Q2_HIGH_LOW

    def test_materiality_quadrant_q3_low_high(self):
        """Test Quadrant 3 (low emissions, high spend)."""
        engine = HybridAggregatorEngine.get_instance()

        quadrant = engine.classify_materiality_quadrant(
            emissions=Decimal("300"),
            spend=Decimal("500000"),
            emissions_threshold=Decimal("500"),
            spend_threshold=Decimal("250000"),
        )

        assert quadrant == MaterialityQuadrant.Q3_LOW_HIGH

    def test_materiality_quadrant_q4_low_low(self):
        """Test Quadrant 4 (low emissions, low spend)."""
        engine = HybridAggregatorEngine.get_instance()

        quadrant = engine.classify_materiality_quadrant(
            emissions=Decimal("300"),
            spend=Decimal("100000"),
            emissions_threshold=Decimal("500"),
            spend_threshold=Decimal("250000"),
        )

        assert quadrant == MaterialityQuadrant.Q4_LOW_LOW

    def test_identify_hotspots(self):
        """Test hotspot identification."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "emissions": Decimal("5000"), "spend": Decimal("100000")},
            {"id": "B", "emissions": Decimal("3000"), "spend": Decimal("80000")},
            {"id": "C", "emissions": Decimal("500"), "spend": Decimal("20000")},
        ]

        hotspots = engine.identify_hotspots(items, top_n=2)

        assert len(hotspots) == 2
        assert hotspots[0]["id"] == "A"
        assert hotspots[1]["id"] == "B"

    def test_cumulative_percentage_calculation(self):
        """Test cumulative percentage calculation for Pareto."""
        engine = HybridAggregatorEngine.get_instance()

        emissions = [Decimal("500"), Decimal("300"), Decimal("150"), Decimal("50")]

        cumulative = engine.calculate_cumulative_percentages(emissions)

        assert cumulative[0] == Decimal("50.0")  # 500/1000
        assert cumulative[1] == Decimal("80.0")  # 800/1000
        assert cumulative[2] == Decimal("95.0")  # 950/1000
        assert cumulative[3] == Decimal("100.0")  # 1000/1000


class TestBoundaryEnforcement:
    """Test boundary enforcement (exclusions)."""

    def test_capital_goods_excluded(self):
        """Test capital goods are excluded."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "category": "raw_materials", "emissions": Decimal("1000")},
            {"id": "B", "category": "capital_goods", "emissions": Decimal("500")},
        ]

        filtered = engine.filter_excluded_items(items)

        assert len(filtered) == 1
        assert filtered[0]["id"] == "A"

    def test_fuel_energy_excluded(self):
        """Test fuel and energy are excluded."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "category": "packaging", "emissions": Decimal("200")},
            {"id": "B", "category": "fuel_energy", "emissions": Decimal("1000")},
        ]

        filtered = engine.filter_excluded_items(items)

        assert len(filtered) == 1
        assert filtered[0]["id"] == "A"

    def test_transport_excluded(self):
        """Test transport/distribution are excluded."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "category": "components", "emissions": Decimal("800")},
            {"id": "B", "category": "upstream_transport", "emissions": Decimal("300")},
        ]

        filtered = engine.filter_excluded_items(items)

        assert len(filtered) == 1
        assert filtered[0]["id"] == "A"

    def test_business_travel_excluded(self):
        """Test business travel is excluded."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "category": "services", "emissions": Decimal("500")},
            {"id": "B", "category": "business_travel", "emissions": Decimal("100")},
        ]

        filtered = engine.filter_excluded_items(items)

        assert len(filtered) == 1
        assert filtered[0]["id"] == "A"

    def test_intercompany_transactions_excluded(self):
        """Test intercompany transactions are excluded."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "is_intercompany": False, "emissions": Decimal("1000")},
            {"id": "B", "is_intercompany": True, "emissions": Decimal("500")},
        ]

        filtered = engine.filter_excluded_items(items)

        assert len(filtered) == 1
        assert filtered[0]["id"] == "A"

    def test_emissions_credits_excluded(self):
        """Test emissions credits/offsets are excluded."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "category": "materials", "emissions": Decimal("800")},
            {"id": "B", "category": "carbon_credits", "emissions": Decimal("-200")},
        ]

        filtered = engine.filter_excluded_items(items)

        assert len(filtered) == 1
        assert filtered[0]["id"] == "A"

    def test_filter_excluded_items_comprehensive(self):
        """Test comprehensive exclusion filtering."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "category": "raw_materials", "emissions": Decimal("1000")},
            {"id": "B", "category": "capital_goods", "emissions": Decimal("500")},
            {"id": "C", "category": "packaging", "emissions": Decimal("200")},
            {"id": "D", "category": "fuel_energy", "emissions": Decimal("300")},
            {"id": "E", "category": "components", "emissions": Decimal("600")},
        ]

        filtered = engine.filter_excluded_items(items)

        assert len(filtered) == 3  # A, C, E
        assert all(item["id"] in ["A", "C", "E"] for item in filtered)

    def test_no_exclusions_needed(self):
        """Test when no exclusions are needed."""
        engine = HybridAggregatorEngine.get_instance()

        items = [
            {"id": "A", "category": "raw_materials", "emissions": Decimal("1000")},
            {"id": "B", "category": "packaging", "emissions": Decimal("500")},
        ]

        filtered = engine.filter_excluded_items(items)

        assert len(filtered) == 2


class TestOverlapDetection:
    """Test overlap detection between methods."""

    def test_no_overlap_between_methods(self):
        """Test detection when no overlap exists."""
        engine = HybridAggregatorEngine.get_instance()

        supplier_items = ["P001", "P002", "P003"]
        avgdata_items = ["P004", "P005"]
        spend_items = ["P006", "P007"]

        overlaps = engine.detect_overlaps(
            supplier_items=supplier_items,
            avgdata_items=avgdata_items,
            spend_items=spend_items,
        )

        assert len(overlaps) == 0

    def test_overlap_supplier_avgdata(self):
        """Test overlap between supplier and avgdata methods."""
        engine = HybridAggregatorEngine.get_instance()

        supplier_items = ["P001", "P002", "P003"]
        avgdata_items = ["P003", "P004"]
        spend_items = ["P005"]

        overlaps = engine.detect_overlaps(
            supplier_items=supplier_items,
            avgdata_items=avgdata_items,
            spend_items=spend_items,
        )

        assert len(overlaps) > 0
        assert "P003" in overlaps

    def test_overlap_all_three_methods(self):
        """Test overlap across all three methods."""
        engine = HybridAggregatorEngine.get_instance()

        supplier_items = ["P001", "P002"]
        avgdata_items = ["P002", "P003"]
        spend_items = ["P002", "P004"]

        overlaps = engine.detect_overlaps(
            supplier_items=supplier_items,
            avgdata_items=avgdata_items,
            spend_items=spend_items,
        )

        assert "P002" in overlaps

    def test_resolve_overlaps_prioritizes_supplier(self):
        """Test overlap resolution prioritizes supplier method."""
        engine = HybridAggregatorEngine.get_instance()

        method_results = {
            "supplier": [{"id": "P001", "emissions": Decimal("100")}],
            "avgdata": [{"id": "P001", "emissions": Decimal("120")}],
            "spend": [{"id": "P001", "emissions": Decimal("150")}],
        }

        resolved = engine.resolve_overlaps(method_results)

        assert len(resolved) == 1
        assert resolved[0]["emissions"] == Decimal("100")  # Supplier value


class TestGapFilling:
    """Test gap filling with spend-based method."""

    def test_fill_uncovered_items(self):
        """Test filling uncovered items with spend-based."""
        engine = HybridAggregatorEngine.get_instance()

        all_items = ["P001", "P002", "P003", "P004"]
        covered_items = ["P001", "P002"]

        gaps = engine.identify_gaps(all_items, covered_items)

        assert len(gaps) == 2
        assert "P003" in gaps
        assert "P004" in gaps

    def test_gap_filling_with_spend_method(self):
        """Test gap filling uses spend-based method."""
        engine = HybridAggregatorEngine.get_instance()

        gaps = ["P003", "P004"]
        spend_data = {
            "P003": {"spend": Decimal("10000"), "category": "raw_materials"},
            "P004": {"spend": Decimal("5000"), "category": "packaging"},
        }

        filled = engine.fill_gaps_with_spend(gaps, spend_data)

        assert len(filled) == 2
        assert all(item["method"] == "spend" for item in filled)

    def test_no_gaps_to_fill(self):
        """Test when no gaps exist."""
        engine = HybridAggregatorEngine.get_instance()

        all_items = ["P001", "P002"]
        covered_items = ["P001", "P002"]

        gaps = engine.identify_gaps(all_items, covered_items)

        assert len(gaps) == 0


class TestWeightedDQI:
    """Test weighted DQI calculation."""

    def test_weighted_dqi_single_method(self):
        """Test weighted DQI with single method."""
        engine = HybridAggregatorEngine.get_instance()

        results = [
            {"emissions": Decimal("1000"), "dqi": Decimal("1.5"), "method": "supplier"},
        ]

        weighted_dqi = engine.calculate_weighted_dqi(results)

        assert weighted_dqi == Decimal("1.5")

    def test_weighted_dqi_mixed_methods(self):
        """Test weighted DQI with mixed methods."""
        engine = HybridAggregatorEngine.get_instance()

        results = [
            {"emissions": Decimal("1000"), "dqi": Decimal("1.5"), "method": "supplier"},
            {"emissions": Decimal("500"), "dqi": Decimal("2.5"), "method": "avgdata"},
            {"emissions": Decimal("500"), "dqi": Decimal("3.5"), "method": "spend"},
        ]

        weighted_dqi = engine.calculate_weighted_dqi(results)

        # (1000*1.5 + 500*2.5 + 500*3.5) / 2000 = (1500+1250+1750) / 2000 = 2.25
        assert weighted_dqi == Decimal("2.25")

    def test_weighted_dqi_zero_emissions(self):
        """Test weighted DQI with zero emissions."""
        engine = HybridAggregatorEngine.get_instance()

        results = [
            {"emissions": Decimal("0"), "dqi": Decimal("1.5"), "method": "supplier"},
        ]

        weighted_dqi = engine.calculate_weighted_dqi(results)

        assert weighted_dqi == Decimal("1.5")  # Default to DQI when no weight

    def test_weighted_dqi_by_method_breakdown(self):
        """Test DQI breakdown by method."""
        engine = HybridAggregatorEngine.get_instance()

        results = [
            {"emissions": Decimal("1000"), "dqi": Decimal("1.5"), "method": "supplier"},
            {"emissions": Decimal("500"), "dqi": Decimal("2.5"), "method": "avgdata"},
        ]

        breakdown = engine.calculate_dqi_by_method(results)

        assert breakdown["supplier"] == Decimal("1.5")
        assert breakdown["avgdata"] == Decimal("2.5")


class TestYoYDecomposition:
    """Test year-over-year decomposition analysis."""

    def test_activity_effect_decomposition(self):
        """Test activity effect decomposition."""
        engine = HybridAggregatorEngine.get_instance()

        baseline = {"quantity": Decimal("1000"), "ef": Decimal("2.0")}
        current = {"quantity": Decimal("1200"), "ef": Decimal("2.0")}

        activity_effect = engine.decompose_activity_effect(baseline, current)

        # (1200 - 1000) × 2.0 = 400
        assert activity_effect == Decimal("400")

    def test_ef_effect_decomposition(self):
        """Test emission factor effect decomposition."""
        engine = HybridAggregatorEngine.get_instance()

        baseline = {"quantity": Decimal("1000"), "ef": Decimal("2.0")}
        current = {"quantity": Decimal("1000"), "ef": Decimal("1.8")}

        ef_effect = engine.decompose_ef_effect(baseline, current)

        # 1000 × (1.8 - 2.0) = -200
        assert ef_effect == Decimal("-200")

    def test_method_effect_decomposition(self):
        """Test method change effect decomposition."""
        engine = HybridAggregatorEngine.get_instance()

        baseline_method = "spend"
        current_method = "supplier"
        baseline_emissions = Decimal("1000")
        current_emissions = Decimal("950")

        method_effect = engine.decompose_method_effect(
            baseline_method, current_method,
            baseline_emissions, current_emissions
        )

        # Improvement from better method = -50
        assert method_effect == Decimal("-50")

    def test_scope_change_decomposition(self):
        """Test scope change decomposition."""
        engine = HybridAggregatorEngine.get_instance()

        baseline_items = ["P001", "P002", "P003"]
        current_items = ["P001", "P002", "P003", "P004"]

        scope_change = engine.decompose_scope_change(baseline_items, current_items)

        assert scope_change["added_items"] == 1
        assert scope_change["removed_items"] == 0


class TestIntensityMetrics:
    """Test intensity metrics calculation."""

    def test_revenue_intensity(self):
        """Test emissions per revenue intensity."""
        engine = HybridAggregatorEngine.get_instance()

        total_emissions = Decimal("10000")
        total_revenue = Decimal("1000000")

        intensity = engine.calculate_revenue_intensity(total_emissions, total_revenue)

        # 10000 / 1000000 = 0.01 kg CO2e per $
        assert intensity == Decimal("0.01")

    def test_fte_intensity(self):
        """Test emissions per FTE intensity."""
        engine = HybridAggregatorEngine.get_instance()

        total_emissions = Decimal("50000")
        total_fte = Decimal("100")

        intensity = engine.calculate_fte_intensity(total_emissions, total_fte)

        # 50000 / 100 = 500 kg CO2e per FTE
        assert intensity == Decimal("500")

    def test_spend_intensity(self):
        """Test emissions per spend intensity."""
        engine = HybridAggregatorEngine.get_instance()

        total_emissions = Decimal("20000")
        total_spend = Decimal("500000")

        intensity = engine.calculate_spend_intensity(total_emissions, total_spend)

        # 20000 / 500000 = 0.04 kg CO2e per $
        assert intensity == Decimal("0.04")

    def test_intensity_zero_denominator(self):
        """Test intensity with zero denominator."""
        engine = HybridAggregatorEngine.get_instance()

        with pytest.raises(ValueError, match="Denominator cannot be zero"):
            engine.calculate_revenue_intensity(Decimal("1000"), Decimal("0"))


class TestAggregation:
    """Test aggregation of all three methods."""

    def test_combine_supplier_avgdata_spend(self):
        """Test combining results from all three methods."""
        engine = HybridAggregatorEngine.get_instance()

        supplier_results = [
            {"id": "P001", "emissions": Decimal("1000"), "method": "supplier"},
        ]
        avgdata_results = [
            {"id": "P002", "emissions": Decimal("500"), "method": "avgdata"},
        ]
        spend_results = [
            {"id": "P003", "emissions": Decimal("300"), "method": "spend"},
        ]

        combined = engine.combine_results(
            supplier_results, avgdata_results, spend_results
        )

        assert len(combined) == 3
        assert sum(item["emissions"] for item in combined) == Decimal("1800")

    def test_aggregation_with_overlaps(self):
        """Test aggregation handles overlaps correctly."""
        engine = HybridAggregatorEngine.get_instance()

        supplier_results = [
            {"id": "P001", "emissions": Decimal("1000"), "method": "supplier"},
        ]
        avgdata_results = [
            {"id": "P001", "emissions": Decimal("1200"), "method": "avgdata"},
            {"id": "P002", "emissions": Decimal("500"), "method": "avgdata"},
        ]
        spend_results = [
            {"id": "P003", "emissions": Decimal("300"), "method": "spend"},
        ]

        combined = engine.combine_results(
            supplier_results, avgdata_results, spend_results,
            resolve_overlaps=True,
        )

        # Should use supplier value for P001 (1000), not avgdata (1200)
        p001_item = [item for item in combined if item["id"] == "P001"][0]
        assert p001_item["emissions"] == Decimal("1000")

    def test_total_emissions_calculation(self):
        """Test total emissions calculation."""
        engine = HybridAggregatorEngine.get_instance()

        results = [
            {"id": "P001", "emissions": Decimal("1000")},
            {"id": "P002", "emissions": Decimal("500")},
            {"id": "P003", "emissions": Decimal("300")},
        ]

        total = engine.calculate_total_emissions(results)

        assert total == Decimal("1800")


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_healthy(self):
        """Test health check returns healthy status."""
        engine = HybridAggregatorEngine.get_instance()

        health = engine.health_check()

        assert health["status"] == "healthy"
        assert "engine" in health
        assert health["engine"] == "HybridAggregatorEngine"

    def test_health_check_includes_stats(self):
        """Test health check includes statistics."""
        engine = HybridAggregatorEngine.get_instance()

        # Perform some operations
        engine.select_method(["supplier", "avgdata", "spend"])

        health = engine.health_check()

        assert "aggregations_performed" in health
        assert health["aggregations_performed"] >= 0
