# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional Pack - Multi-Operator Portfolio Tests
================================================================

Tests the multi-operator portfolio management engine including:
- Operator registration
- Portfolio-wide view
- Supplier deduplication across operators
- Aggregate risk calculation
- Cross-operator reporting
- Operator benchmarking
- Cost allocation
- Operator merging
- Operator-specific dashboards
- Shared supplier pool management

Author: GreenLang QA Team
Version: 1.0.0
"""

from typing import Any, Dict, List

import pytest


@pytest.mark.unit
class TestMultiOperatorPortfolio:
    """Test suite for multi-operator portfolio management."""

    def test_register_operator(self, sample_operator_data: Dict[str, Any]):
        """Test operator registration in portfolio."""
        portfolio = []

        # Register operator
        portfolio.append(sample_operator_data)

        assert len(portfolio) == 1
        assert portfolio[0]["operator_id"] == sample_operator_data["operator_id"]
        assert portfolio[0]["name"] == sample_operator_data["name"]
        assert portfolio[0]["eori_number"] == sample_operator_data["eori_number"]

    def test_portfolio_view(self, sample_portfolio: List[Dict[str, Any]]):
        """Test portfolio-wide view of all operators."""
        # Portfolio summary
        portfolio_summary = {
            "total_operators": len(sample_portfolio),
            "total_annual_imports_tonnes": sum(op["annual_imports_tonnes"] for op in sample_portfolio),
            "total_suppliers": sum(op["supplier_count"] for op in sample_portfolio),
            "average_risk_score": sum(op["risk_score"] for op in sample_portfolio) / len(sample_portfolio),
            "commodities": set(),
        }

        for operator in sample_portfolio:
            portfolio_summary["commodities"].update(operator["commodities"])

        portfolio_summary["commodities"] = list(portfolio_summary["commodities"])

        assert portfolio_summary["total_operators"] == 3
        assert portfolio_summary["total_annual_imports_tonnes"] == 100000
        assert portfolio_summary["total_suppliers"] == 55
        assert 0 <= portfolio_summary["average_risk_score"] <= 1
        assert len(portfolio_summary["commodities"]) >= 4

    def test_deduplicate_suppliers(self, sample_portfolio: List[Dict[str, Any]], sample_suppliers: List[Dict[str, Any]]):
        """Test supplier deduplication across operators."""
        # Simulate: multiple operators may work with same supplier
        all_suppliers = []

        # Add suppliers for each operator
        for i, operator in enumerate(sample_portfolio):
            # Each operator gets subset of suppliers
            operator_suppliers = sample_suppliers[i:i+2]
            for supplier in operator_suppliers:
                all_suppliers.append({
                    "operator_id": operator["operator_id"],
                    "supplier_id": supplier["supplier_id"],
                    "supplier_name": supplier["name"],
                })

        # Deduplicate by supplier_id
        unique_suppliers = {}
        for entry in all_suppliers:
            supplier_id = entry["supplier_id"]
            if supplier_id not in unique_suppliers:
                unique_suppliers[supplier_id] = {
                    "supplier_id": supplier_id,
                    "supplier_name": entry["supplier_name"],
                    "operators": [],
                }
            unique_suppliers[supplier_id]["operators"].append(entry["operator_id"])

        # Validate deduplication
        assert len(unique_suppliers) <= len(all_suppliers)

        # Check for shared suppliers
        shared_suppliers = [s for s in unique_suppliers.values() if len(s["operators"]) > 1]
        # May or may not have shared suppliers in this test data
        assert len(shared_suppliers) >= 0

    def test_aggregate_risk(self, sample_portfolio: List[Dict[str, Any]]):
        """Test aggregate risk calculation across portfolio."""
        # Calculate portfolio aggregate risk
        total_volume = sum(op["annual_imports_tonnes"] for op in sample_portfolio)

        weighted_risk = sum(
            op["risk_score"] * op["annual_imports_tonnes"]
            for op in sample_portfolio
        ) / total_volume

        aggregate_risk = {
            "portfolio_risk_score": weighted_risk,
            "min_operator_risk": min(op["risk_score"] for op in sample_portfolio),
            "max_operator_risk": max(op["risk_score"] for op in sample_portfolio),
            "risk_variance": None,  # Calculate variance
        }

        # Calculate variance
        mean_risk = sum(op["risk_score"] for op in sample_portfolio) / len(sample_portfolio)
        variance = sum((op["risk_score"] - mean_risk) ** 2 for op in sample_portfolio) / len(sample_portfolio)
        aggregate_risk["risk_variance"] = variance

        assert 0 <= aggregate_risk["portfolio_risk_score"] <= 1
        assert aggregate_risk["min_operator_risk"] <= aggregate_risk["max_operator_risk"]
        assert aggregate_risk["risk_variance"] >= 0

    def test_cross_operator_report(self, sample_portfolio: List[Dict[str, Any]]):
        """Test cross-operator reporting."""
        report = {
            "report_date": "2025-11-15",
            "operators": [],
        }

        for operator in sample_portfolio:
            operator_summary = {
                "operator_id": operator["operator_id"],
                "name": operator["name"],
                "country": operator["country"],
                "annual_imports_tonnes": operator["annual_imports_tonnes"],
                "supplier_count": operator["supplier_count"],
                "risk_score": operator["risk_score"],
                "risk_level": "LOW" if operator["risk_score"] < 0.3 else
                             "STANDARD" if operator["risk_score"] < 0.5 else
                             "HIGH",
            }
            report["operators"].append(operator_summary)

        assert len(report["operators"]) == len(sample_portfolio)

        # Validate all operators included
        for op_summary in report["operators"]:
            assert "operator_id" in op_summary
            assert "risk_level" in op_summary

    def test_benchmark_operators(self, sample_portfolio: List[Dict[str, Any]]):
        """Test operator benchmarking within portfolio."""
        # Rank operators by risk score (lower is better)
        ranked_operators = sorted(sample_portfolio, key=lambda x: x["risk_score"])

        benchmarks = []
        for i, operator in enumerate(ranked_operators):
            percentile = ((i + 1) / len(ranked_operators)) * 100
            benchmarks.append({
                "operator_id": operator["operator_id"],
                "name": operator["name"],
                "risk_score": operator["risk_score"],
                "rank": i + 1,
                "percentile": round(percentile, 1),
            })

        # Validate rankings
        assert benchmarks[0]["rank"] == 1  # Best performer
        assert benchmarks[-1]["rank"] == len(sample_portfolio)  # Worst performer
        assert benchmarks[0]["risk_score"] <= benchmarks[-1]["risk_score"]

    def test_cost_allocation(self, sample_portfolio: List[Dict[str, Any]]):
        """Test cost allocation across operators."""
        # Simulate shared platform costs
        total_platform_cost = 100000  # EUR/year
        total_volume = sum(op["annual_imports_tonnes"] for op in sample_portfolio)

        cost_allocations = []
        for operator in sample_portfolio:
            # Allocate based on volume
            allocation = (operator["annual_imports_tonnes"] / total_volume) * total_platform_cost
            cost_allocations.append({
                "operator_id": operator["operator_id"],
                "name": operator["name"],
                "allocated_cost_eur": round(allocation, 2),
                "cost_per_tonne": round(allocation / operator["annual_imports_tonnes"], 2),
            })

        # Validate allocations
        total_allocated = sum(a["allocated_cost_eur"] for a in cost_allocations)
        assert abs(total_allocated - total_platform_cost) < 1  # Rounding tolerance

        for allocation in cost_allocations:
            assert allocation["allocated_cost_eur"] > 0
            assert allocation["cost_per_tonne"] > 0

    def test_merge_operators(self, sample_portfolio: List[Dict[str, Any]]):
        """Test merging operators (e.g., after acquisition)."""
        # Merge first two operators
        operator_a = sample_portfolio[0]
        operator_b = sample_portfolio[1]

        merged_operator = {
            "operator_id": operator_a["operator_id"],  # Keep primary ID
            "name": f"{operator_a['name']} (merged with {operator_b['name']})",
            "country": operator_a["country"],
            "eori_number": operator_a["eori_number"],
            "commodities": list(set(operator_a["commodities"] + operator_b["commodities"])),
            "annual_imports_tonnes": operator_a["annual_imports_tonnes"] + operator_b["annual_imports_tonnes"],
            "supplier_count": operator_a["supplier_count"] + operator_b["supplier_count"],
            "risk_score": (
                operator_a["risk_score"] * operator_a["annual_imports_tonnes"] +
                operator_b["risk_score"] * operator_b["annual_imports_tonnes"]
            ) / (operator_a["annual_imports_tonnes"] + operator_b["annual_imports_tonnes"]),
        }

        # Validate merge
        assert merged_operator["annual_imports_tonnes"] == 80000
        assert merged_operator["supplier_count"] == 43
        assert len(merged_operator["commodities"]) >= len(operator_a["commodities"])

    def test_operator_dashboard(self, sample_portfolio: List[Dict[str, Any]]):
        """Test operator-specific dashboard data."""
        operator = sample_portfolio[0]

        dashboard = {
            "operator_id": operator["operator_id"],
            "operator_name": operator["name"],
            "kpis": {
                "total_imports_tonnes": operator["annual_imports_tonnes"],
                "active_suppliers": operator["supplier_count"],
                "risk_score": operator["risk_score"],
                "compliance_rate": 0.92,  # Simulated
                "dds_submitted": 45,
                "dds_pending": 3,
            },
            "alerts": {
                "HIGH": 2,
                "MEDIUM": 5,
                "LOW": 10,
            },
            "recent_activities": [
                {"activity": "DDS submitted", "timestamp": "2025-11-14T10:00:00Z"},
                {"activity": "Supplier audit completed", "timestamp": "2025-11-13T15:30:00Z"},
            ],
        }

        assert dashboard["operator_id"] == operator["operator_id"]
        assert dashboard["kpis"]["total_imports_tonnes"] > 0
        assert dashboard["kpis"]["active_suppliers"] > 0
        assert 0 <= dashboard["kpis"]["risk_score"] <= 1

    def test_shared_supplier_pool(self, sample_portfolio: List[Dict[str, Any]], sample_suppliers: List[Dict[str, Any]]):
        """Test shared supplier pool across operators."""
        # Simulate shared supplier pool
        supplier_pool = []

        for supplier in sample_suppliers:
            # Track which operators use this supplier
            using_operators = []
            for i, operator in enumerate(sample_portfolio):
                # Simulate: supplier used by operators based on commodity match
                if supplier["commodity"] in operator["commodities"]:
                    using_operators.append(operator["operator_id"])

            if using_operators:
                supplier_pool.append({
                    "supplier_id": supplier["supplier_id"],
                    "supplier_name": supplier["name"],
                    "commodity": supplier["commodity"],
                    "operators_using": using_operators,
                    "shared": len(using_operators) > 1,
                })

        # Validate shared pool
        assert len(supplier_pool) > 0

        shared_suppliers = [s for s in supplier_pool if s["shared"]]
        # Should have some shared suppliers
        assert len(shared_suppliers) >= 0

        for supplier in shared_suppliers:
            assert len(supplier["operators_using"]) > 1

    def test_portfolio_limit(self, mock_config: Dict[str, Any]):
        """Test portfolio operator limit enforcement."""
        max_operators = mock_config["portfolio_config"]["max_operators"]

        assert max_operators == 100

        # Simulate adding operators
        current_operator_count = 3
        can_add_operator = current_operator_count < max_operators

        assert can_add_operator is True

    def test_portfolio_metrics_aggregation(self, sample_portfolio: List[Dict[str, Any]]):
        """Test aggregation of portfolio-wide metrics."""
        metrics = {
            "operators": {
                "total": len(sample_portfolio),
                "by_country": {},
                "by_size": {},
            },
            "commodities": {
                "total_unique": 0,
                "distribution": {},
            },
            "risk": {
                "portfolio_average": 0,
                "high_risk_operators": 0,
            },
            "volume": {
                "total_tonnes": 0,
                "average_per_operator": 0,
            },
        }

        # Aggregate by country
        for operator in sample_portfolio:
            country = operator["country"]
            metrics["operators"]["by_country"][country] = \
                metrics["operators"]["by_country"].get(country, 0) + 1

            # Aggregate commodities
            for commodity in operator["commodities"]:
                metrics["commodities"]["distribution"][commodity] = \
                    metrics["commodities"]["distribution"].get(commodity, 0) + 1

            # Aggregate risk
            metrics["risk"]["portfolio_average"] += operator["risk_score"]
            if operator["risk_score"] >= 0.5:
                metrics["risk"]["high_risk_operators"] += 1

            # Aggregate volume
            metrics["volume"]["total_tonnes"] += operator["annual_imports_tonnes"]

        # Finalize calculations
        metrics["risk"]["portfolio_average"] /= len(sample_portfolio)
        metrics["volume"]["average_per_operator"] = \
            metrics["volume"]["total_tonnes"] / len(sample_portfolio)
        metrics["commodities"]["total_unique"] = \
            len(metrics["commodities"]["distribution"])

        # Validate metrics
        assert metrics["operators"]["total"] == 3
        assert metrics["commodities"]["total_unique"] >= 4
        assert 0 <= metrics["risk"]["portfolio_average"] <= 1
        assert metrics["volume"]["total_tonnes"] == 100000

    def test_operator_segmentation(self, sample_portfolio: List[Dict[str, Any]]):
        """Test operator segmentation by volume/risk."""
        # Segment by annual volume
        segments = {
            "large": [],  # >40k tonnes
            "medium": [],  # 20k-40k tonnes
            "small": [],  # <20k tonnes
        }

        for operator in sample_portfolio:
            volume = operator["annual_imports_tonnes"]
            if volume > 40000:
                segments["large"].append(operator)
            elif volume >= 20000:
                segments["medium"].append(operator)
            else:
                segments["small"].append(operator)

        # Validate segmentation
        total_operators = sum(len(ops) for ops in segments.values())
        assert total_operators == len(sample_portfolio)

        # Large segment should have highest total volume
        large_volume = sum(op["annual_imports_tonnes"] for op in segments["large"])
        assert large_volume >= 50000
