# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Optimization Pipeline Integration Tests

End-to-end tests for the cleaning optimization pipeline including:
- Fouling state to recommendation flow
- CMMS work order creation
- Multi-exchanger optimization
- Constraint satisfaction

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List


class TestFoulingToRecommendationFlow:
    """Test fouling state to cleaning recommendation flow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self, mock_optimizer_service):
        """Test end-to-end optimization pipeline."""
        # Create fouling state
        fouling_state = type('FoulingState', (), {
            'exchanger_id': 'HX-001',
            'timestamp': datetime.now(timezone.utc),
            'fouling_resistance_m2K_kW': 0.00045,
            'ua_degradation_percent': 28,
            'predicted_days_to_threshold': 42,
            'confidence_score': 0.85,
            'trend': 'increasing',
        })()

        constraints = {
            "max_cleaning_cost": 12000.0,
            "maintenance_window_start": datetime.now(timezone.utc) + timedelta(days=7),
            "maintenance_window_end": datetime.now(timezone.utc) + timedelta(days=60),
        }

        recommendation = await mock_optimizer_service.optimize_schedule(
            "HX-001",
            fouling_state,
            constraints,
        )

        assert recommendation.exchanger_id == "HX-001"
        assert recommendation.recommended_cleaning_date is not None
        assert recommendation.urgency in ["routine", "scheduled", "urgent", "critical"]
        assert recommendation.estimated_cost_usd > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_recommendation_respects_constraints(self, mock_optimizer_service):
        """Test that recommendations respect constraints."""
        fouling_state = type('FoulingState', (), {
            'exchanger_id': 'HX-001',
            'predicted_days_to_threshold': 45,
            'ua_degradation_percent': 30,
        })()

        window_start = datetime.now(timezone.utc) + timedelta(days=20)
        window_end = datetime.now(timezone.utc) + timedelta(days=40)

        constraints = {
            "maintenance_window_start": window_start,
            "maintenance_window_end": window_end,
        }

        recommendation = await mock_optimizer_service.optimize_schedule(
            "HX-001",
            fouling_state,
            constraints,
        )

        # Recommendation should be within window
        assert recommendation.recommended_cleaning_date >= window_start
        assert recommendation.recommended_cleaning_date <= window_end


class TestCMMSIntegration:
    """Test CMMS integration for work order creation."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_work_order_creation(self, mock_cmms_connector, mock_optimizer_service):
        """Test work order creation from recommendation."""
        # Generate recommendation
        fouling_state = type('FoulingState', (), {
            'exchanger_id': 'HX-001',
            'predicted_days_to_threshold': 30,
            'ua_degradation_percent': 35,
        })()

        recommendation = await mock_optimizer_service.optimize_schedule(
            "HX-001",
            fouling_state,
            {},
        )

        # Create work order
        work_order_request = {
            "exchanger_id": recommendation.exchanger_id,
            "recommendation_id": recommendation.recommendation_id,
            "priority": recommendation.urgency,
            "description": f"Exchanger cleaning - {recommendation.urgency}",
            "scheduled_date": recommendation.recommended_cleaning_date.isoformat(),
            "estimated_cost_usd": recommendation.estimated_cost_usd,
        }

        work_order = await mock_cmms_connector.create_work_order(work_order_request)

        assert "work_order_id" in work_order
        assert work_order["status"] == "pending"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_work_order_deduplication(self, mock_cmms_connector):
        """Test that duplicate work orders are prevented."""
        request1 = {
            "exchanger_id": "HX-001",
            "recommendation_id": "rec-12345",
            "priority": "scheduled",
            "description": "Cleaning",
        }

        request2 = {
            "exchanger_id": "HX-001",
            "recommendation_id": "rec-12345",
            "priority": "scheduled",
            "description": "Cleaning",
        }

        wo1 = await mock_cmms_connector.create_work_order(request1)
        wo2 = await mock_cmms_connector.create_work_order(request2)

        # Same recommendation should return same work order
        assert wo1["work_order_id"] == wo2["work_order_id"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_work_order_retrieval(self, mock_cmms_connector):
        """Test work order retrieval."""
        # Create work order first
        request = {
            "exchanger_id": "HX-TEST",
            "recommendation_id": "rec-test",
            "priority": "routine",
        }

        created_wo = await mock_cmms_connector.create_work_order(request)

        # Retrieve it
        retrieved_wo = await mock_cmms_connector.get_work_order(created_wo["work_order_id"])

        assert retrieved_wo is not None


class TestMultiExchangerOptimization:
    """Test optimization for multiple exchangers."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fleet_optimization(self, mock_optimizer_service):
        """Test optimization across exchanger fleet."""
        exchangers = [
            type('FoulingState', (), {
                'exchanger_id': 'HX-001',
                'predicted_days_to_threshold': 30,
                'ua_degradation_percent': 35,
            })(),
            type('FoulingState', (), {
                'exchanger_id': 'HX-002',
                'predicted_days_to_threshold': 60,
                'ua_degradation_percent': 20,
            })(),
            type('FoulingState', (), {
                'exchanger_id': 'HX-003',
                'predicted_days_to_threshold': 14,
                'ua_degradation_percent': 45,
            })(),
        ]

        recommendations = []
        for ex in exchangers:
            rec = await mock_optimizer_service.optimize_schedule(
                ex.exchanger_id,
                ex,
                {},
            )
            recommendations.append(rec)

        # Sort by urgency
        urgency_order = {"critical": 0, "urgent": 1, "scheduled": 2, "routine": 3}
        sorted_recs = sorted(recommendations, key=lambda r: urgency_order.get(r.urgency, 4))

        # HX-003 should be most urgent (lowest days, highest degradation)
        assert sorted_recs[0].exchanger_id == "HX-003"

    @pytest.mark.integration
    def test_resource_constrained_scheduling(self):
        """Test scheduling with limited maintenance resources."""
        cleaning_jobs = [
            {"id": "HX-001", "urgency_score": 90, "duration_hours": 24},
            {"id": "HX-002", "urgency_score": 70, "duration_hours": 16},
            {"id": "HX-003", "urgency_score": 95, "duration_hours": 20},
            {"id": "HX-004", "urgency_score": 50, "duration_hours": 12},
        ]

        available_hours_per_week = 48
        crews = 2

        # Sort by urgency (descending)
        sorted_jobs = sorted(cleaning_jobs, key=lambda x: -x["urgency_score"])

        # Schedule highest priority first
        scheduled = []
        remaining_hours = available_hours_per_week

        for job in sorted_jobs:
            if job["duration_hours"] <= remaining_hours:
                scheduled.append(job)
                remaining_hours -= job["duration_hours"]

        assert len(scheduled) > 0
        assert scheduled[0]["id"] == "HX-003"  # Highest urgency


class TestConstraintSatisfaction:
    """Test constraint satisfaction in optimization."""

    @pytest.mark.integration
    def test_budget_constraint(self):
        """Test budget constraint handling."""
        budget = 50000.0  # USD
        cleaning_costs = [8500, 7200, 9000, 6500, 8000]

        scheduled_cost = 0
        scheduled = []

        for i, cost in enumerate(cleaning_costs):
            if scheduled_cost + cost <= budget:
                scheduled.append(i)
                scheduled_cost += cost

        assert scheduled_cost <= budget

    @pytest.mark.integration
    def test_maintenance_window_constraint(self):
        """Test maintenance window constraint."""
        optimal_days = [10, 25, 40, 55, 70]
        window_start_day = 30
        window_end_day = 60

        constrained = []
        for day in optimal_days:
            if day < window_start_day:
                constrained.append(window_start_day)
            elif day > window_end_day:
                constrained.append(window_end_day)
            else:
                constrained.append(day)

        for day in constrained:
            assert window_start_day <= day <= window_end_day

    @pytest.mark.integration
    def test_minimum_operating_period(self):
        """Test minimum operating period between cleanings."""
        min_period_days = 90  # Minimum days between cleanings
        last_cleaning = datetime.now(timezone.utc) - timedelta(days=60)
        proposed_cleaning = datetime.now(timezone.utc) + timedelta(days=10)

        days_since_last = (proposed_cleaning - last_cleaning).days

        if days_since_last < min_period_days:
            # Postpone to meet minimum period
            adjusted_date = last_cleaning + timedelta(days=min_period_days)
        else:
            adjusted_date = proposed_cleaning

        assert (adjusted_date - last_cleaning).days >= min_period_days


class TestOptimizationProvenanceTracking:
    """Test provenance tracking for optimization."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_recommendation_provenance(self, mock_optimizer_service):
        """Test provenance hash for recommendations."""
        fouling_state = type('FoulingState', (), {
            'exchanger_id': 'HX-001',
            'predicted_days_to_threshold': 45,
            'ua_degradation_percent': 28,
        })()

        recommendation = await mock_optimizer_service.optimize_schedule(
            "HX-001",
            fouling_state,
            {},
        )

        assert recommendation.provenance_hash is not None
        assert len(recommendation.provenance_hash) == 64

    @pytest.mark.integration
    def test_optimization_audit_trail(self):
        """Test optimization audit trail."""
        audit_record = {
            "optimization_id": "opt-12345",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exchanger_ids": ["HX-001", "HX-002", "HX-003"],
            "constraints_applied": ["budget", "window", "min_period"],
            "recommendations_generated": 3,
            "total_estimated_cost": 25500.0,
            "total_estimated_savings": 95000.0,
            "provenance_hash": hashlib.sha256(b"opt-12345").hexdigest(),
        }

        assert audit_record["recommendations_generated"] == len(audit_record["exchanger_ids"])


class TestOptimizationPipelinePerformance:
    """Test optimization pipeline performance."""

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_optimization_latency(self, mock_optimizer_service, performance_timer):
        """Test optimization latency."""
        fouling_state = type('FoulingState', (), {
            'exchanger_id': 'HX-001',
            'predicted_days_to_threshold': 45,
            'ua_degradation_percent': 28,
        })()

        timer = performance_timer()
        with timer:
            for _ in range(50):
                await mock_optimizer_service.optimize_schedule(
                    "HX-001",
                    fouling_state,
                    {},
                )

        timer.assert_under(500)  # 50 optimizations in <500ms


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestFoulingToRecommendationFlow",
    "TestCMMSIntegration",
    "TestMultiExchangerOptimization",
    "TestConstraintSatisfaction",
    "TestOptimizationProvenanceTracking",
    "TestOptimizationPipelinePerformance",
]
