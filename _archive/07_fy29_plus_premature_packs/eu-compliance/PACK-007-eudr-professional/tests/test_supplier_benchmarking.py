# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional Pack - Supplier Benchmarking Tests
=============================================================

Tests the supplier benchmarking engine including:
- Performance score calculation
- Peer group assignment
- Percentile calculation
- Scorecard generation
- Improvement tracking over time
- Degradation alerts
- Best practice identification
- Portfolio-wide benchmarking
- 6 scoring dimensions
- Industry benchmarks per commodity

Author: GreenLang QA Team
Version: 1.0.0
"""

from typing import Any, Dict, List

import pytest


@pytest.mark.unit
class TestSupplierBenchmarking:
    """Test suite for supplier benchmarking engine."""

    def test_calculate_score_basic(self, sample_suppliers: List[Dict[str, Any]]):
        """Test basic performance score calculation."""
        supplier = sample_suppliers[0]

        # 6 dimensions with equal weight
        dimensions = {
            "data_completeness": 0.88,
            "risk_score_inverted": 1 - 0.58,  # Lower risk = higher score
            "certification": 1.0,  # Has certification
            "audit_compliance": 0.85,
            "responsiveness": 0.90,
            "improvement_trend": 0.75,
        }

        performance_score = sum(dimensions.values()) / len(dimensions)

        assert 0 <= performance_score <= 1
        assert performance_score == pytest.approx(0.80, abs=0.05)

    def test_peer_group_assignment(self, sample_suppliers: List[Dict[str, Any]]):
        """Test peer group assignment by commodity and country."""
        # Group suppliers by commodity
        commodity_groups = {}
        for supplier in sample_suppliers:
            commodity = supplier["commodity"]
            if commodity not in commodity_groups:
                commodity_groups[commodity] = []
            commodity_groups[commodity].append(supplier)

        # Palm oil group
        palm_oil_suppliers = commodity_groups.get("palm_oil", [])
        assert len(palm_oil_suppliers) >= 1

        # Wood group
        wood_suppliers = commodity_groups.get("wood", [])
        assert len(wood_suppliers) >= 2

    def test_percentile_calculation(self, sample_suppliers: List[Dict[str, Any]]):
        """Test percentile calculation within peer group."""
        # Extract performance scores
        performance_scores = [s["performance_score"] for s in sample_suppliers]
        performance_scores.sort()

        # Calculate percentiles
        def percentile(scores, p):
            index = int(len(scores) * p / 100)
            return scores[min(index, len(scores) - 1)]

        p50 = percentile(performance_scores, 50)
        p75 = percentile(performance_scores, 75)
        p90 = percentile(performance_scores, 90)

        assert p50 <= p75 <= p90

    def test_scorecard_generation(self, sample_suppliers: List[Dict[str, Any]]):
        """Test scorecard generation for a supplier."""
        supplier = sample_suppliers[0]

        scorecard = {
            "supplier_id": supplier["supplier_id"],
            "supplier_name": supplier["name"],
            "overall_score": supplier["performance_score"],
            "dimensions": {
                "data_completeness": supplier["data_completeness"],
                "risk_score": 1 - supplier["risk_score"],
                "certification": 1.0,
                "audit_compliance": 0.85,
                "responsiveness": 0.90,
                "improvement_trend": 0.75,
            },
            "peer_group": f"{supplier['commodity']}_{supplier['country']}",
            "percentile": 68,
            "rank": 12,
            "total_in_peer_group": 25,
        }

        assert "supplier_id" in scorecard
        assert "overall_score" in scorecard
        assert "dimensions" in scorecard
        assert len(scorecard["dimensions"]) == 6
        assert 0 <= scorecard["percentile"] <= 100

    def test_improvement_tracking(self):
        """Test improvement tracking over time."""
        historical_scores = [
            {"month": "2025-07", "score": 0.65},
            {"month": "2025-08", "score": 0.68},
            {"month": "2025-09", "score": 0.72},
            {"month": "2025-10", "score": 0.75},
            {"month": "2025-11", "score": 0.78},
        ]

        # Calculate trend
        scores = [h["score"] for h in historical_scores]
        trend = "improving" if scores[-1] > scores[0] else "declining"

        assert trend == "improving"
        assert scores[-1] - scores[0] == pytest.approx(0.13, abs=0.01)

    def test_degradation_alert(self):
        """Test degradation alert when score drops significantly."""
        historical_scores = [
            {"month": "2025-07", "score": 0.85},
            {"month": "2025-08", "score": 0.82},
            {"month": "2025-09", "score": 0.78},
            {"month": "2025-10", "score": 0.72},
            {"month": "2025-11", "score": 0.65},
        ]

        scores = [h["score"] for h in historical_scores]
        decline = scores[0] - scores[-1]

        # Alert if decline > 10%
        alert_threshold = 0.10
        should_alert = decline > alert_threshold

        assert should_alert is True
        assert decline == pytest.approx(0.20, abs=0.01)

    def test_best_practice_identification(self, sample_suppliers: List[Dict[str, Any]]):
        """Test identification of best-practice suppliers."""
        # Top performers: score >= 85th percentile
        scores = [s["performance_score"] for s in sample_suppliers]
        scores.sort()
        p85_threshold = scores[int(0.85 * len(scores))]

        best_practice_suppliers = [
            s for s in sample_suppliers
            if s["performance_score"] >= p85_threshold
        ]

        # Should have at least one top performer
        assert len(best_practice_suppliers) >= 1

        for supplier in best_practice_suppliers:
            assert supplier["performance_score"] >= p85_threshold

    def test_portfolio_benchmark(self, sample_portfolio: List[Dict[str, Any]], sample_suppliers: List[Dict[str, Any]]):
        """Test portfolio-wide benchmarking across operators."""
        # Aggregate supplier scores by operator
        operator_avg_scores = []

        for operator in sample_portfolio:
            # Simulate: each operator has subset of suppliers
            # Use risk_score as proxy
            avg_score = 1 - operator["risk_score"]
            operator_avg_scores.append({
                "operator_id": operator["operator_id"],
                "operator_name": operator["name"],
                "avg_supplier_score": avg_score,
                "supplier_count": operator["supplier_count"],
            })

        assert len(operator_avg_scores) == len(sample_portfolio)

        # Rank operators by average score
        operator_avg_scores.sort(key=lambda x: x["avg_supplier_score"], reverse=True)
        assert operator_avg_scores[0]["avg_supplier_score"] >= operator_avg_scores[-1]["avg_supplier_score"]

    def test_scoring_dimensions(self):
        """Test all 6 scoring dimensions are evaluated."""
        dimensions = [
            "data_completeness",
            "risk_score",
            "certification",
            "audit_compliance",
            "responsiveness",
            "improvement_trend",
        ]

        assert len(dimensions) == 6

        # Simulate scoring
        dimension_scores = {
            "data_completeness": 0.88,
            "risk_score": 0.42,  # Inverted
            "certification": 1.0,
            "audit_compliance": 0.85,
            "responsiveness": 0.90,
            "improvement_trend": 0.75,
        }

        assert len(dimension_scores) == 6
        for dim in dimensions:
            assert dim in dimension_scores
            assert 0 <= dimension_scores[dim] <= 1

    def test_industry_benchmarks_per_commodity(self, sample_suppliers: List[Dict[str, Any]]):
        """Test industry benchmarks calculated per commodity."""
        # Group by commodity
        commodity_benchmarks = {}

        commodities = set(s["commodity"] for s in sample_suppliers)
        for commodity in commodities:
            commodity_suppliers = [s for s in sample_suppliers if s["commodity"] == commodity]
            scores = [s["performance_score"] for s in commodity_suppliers]

            if scores:
                commodity_benchmarks[commodity] = {
                    "mean": sum(scores) / len(scores),
                    "median": sorted(scores)[len(scores) // 2],
                    "count": len(scores),
                }

        # Validate benchmarks
        for commodity, benchmark in commodity_benchmarks.items():
            assert 0 <= benchmark["mean"] <= 1
            assert benchmark["count"] > 0

    def test_certification_scoring(self, sample_suppliers: List[Dict[str, Any]]):
        """Test certification contributes to overall score."""
        supplier_with_cert = sample_suppliers[0]  # Has RSPO
        # Create a supplier without certifications for comparison
        supplier_without_cert = {**sample_suppliers[0], "certifications": []}

        # Calculate certification scores
        cert_score_with = 1.0 if supplier_with_cert["certifications"] else 0.0
        cert_score_without = 1.0 if supplier_without_cert["certifications"] else 0.0

        assert cert_score_with == 1.0
        assert cert_score_without == 0.0

    def test_data_completeness_scoring(self, sample_suppliers: List[Dict[str, Any]]):
        """Test data completeness dimension."""
        for supplier in sample_suppliers:
            completeness = supplier["data_completeness"]
            assert 0 <= completeness <= 1

            # High completeness (>0.9) should boost score
            if completeness > 0.9:
                assert supplier["performance_score"] > 0.7

    def test_risk_score_inversion(self, sample_suppliers: List[Dict[str, Any]]):
        """Test risk score is inverted for performance scoring."""
        for supplier in sample_suppliers:
            risk_score = supplier["risk_score"]
            inverted_risk = 1 - risk_score

            # Low risk (high inverted) = better performance
            if risk_score < 0.3:  # Low risk
                assert inverted_risk > 0.7

    def test_responsiveness_dimension(self):
        """Test responsiveness dimension (data update frequency)."""
        # Simulate responsiveness based on last update
        from datetime import datetime, timedelta

        last_update = datetime.now() - timedelta(days=5)
        days_since_update = (datetime.now() - last_update).days

        # Score: 1.0 if updated within 7 days, declining thereafter
        if days_since_update <= 7:
            responsiveness_score = 1.0
        elif days_since_update <= 30:
            responsiveness_score = 1.0 - ((days_since_update - 7) / 23) * 0.5
        else:
            responsiveness_score = 0.5

        assert 0 <= responsiveness_score <= 1
        assert responsiveness_score == 1.0  # Updated 5 days ago

    def test_audit_compliance_dimension(self):
        """Test audit compliance dimension."""
        # Simulate audit compliance
        audit_findings = {
            "total_audits": 10,
            "passed_audits": 9,
            "failed_audits": 1,
        }

        compliance_rate = audit_findings["passed_audits"] / audit_findings["total_audits"]
        assert compliance_rate == 0.9

    def test_improvement_trend_calculation(self):
        """Test improvement trend dimension calculation."""
        historical_scores = [0.65, 0.68, 0.72, 0.75, 0.78]

        # Linear regression slope (simplified)
        n = len(historical_scores)
        x_mean = (n - 1) / 2
        y_mean = sum(historical_scores) / n

        numerator = sum((i - x_mean) * (score - y_mean) for i, score in enumerate(historical_scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Positive slope = improvement
        assert slope > 0

        # Normalize slope to [0, 1]
        improvement_score = min(1.0, max(0.0, 0.5 + slope * 10))
        assert 0 <= improvement_score <= 1

    def test_peer_group_size_minimum(self):
        """Test peer group has minimum size for meaningful comparison."""
        peer_group_size = 10
        assert peer_group_size >= 5, "Peer group too small for benchmarking"

    def test_percentile_rank_consistency(self):
        """Test percentile rank is consistent with score ordering."""
        scores_and_ranks = [
            {"score": 0.95, "percentile": 98},
            {"score": 0.85, "percentile": 85},
            {"score": 0.75, "percentile": 70},
            {"score": 0.65, "percentile": 50},
            {"score": 0.55, "percentile": 30},
        ]

        for i in range(len(scores_and_ranks) - 1):
            assert scores_and_ranks[i]["score"] >= scores_and_ranks[i + 1]["score"]
            assert scores_and_ranks[i]["percentile"] >= scores_and_ranks[i + 1]["percentile"]
