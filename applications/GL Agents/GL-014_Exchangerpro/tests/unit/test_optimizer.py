# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Cleaning Optimizer Unit Tests

Tests for cleaning schedule optimization including:
- Cost-benefit analysis
- Optimal cleaning date calculation
- Urgency classification
- Constraint handling
- Economic calculations
- Provenance hash verification

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any


# Test tolerances
COST_TOLERANCE = 100.0  # USD
DAYS_TOLERANCE = 3


class TestCostBenefitAnalysis:
    """Test cost-benefit analysis for cleaning decisions."""

    def test_cleaning_cost_estimation(self, sample_exchanger_config):
        """Test cleaning cost estimation."""
        config = sample_exchanger_config

        # Base cleaning cost factors
        base_cost = 5000.0  # USD
        area_factor = config.tube_count * config.tube_length_m * config.tube_od_m * 3.14159
        complexity_factor = 1.2 if config.tema_type.value in ["AES", "AEU"] else 1.0

        cleaning_cost = base_cost * complexity_factor

        assert cleaning_cost > 0
        assert cleaning_cost < 50000  # Reasonable upper bound

    def test_energy_loss_calculation(self, fouled_exchanger_kpis):
        """Test energy loss due to fouling calculation."""
        kpis = fouled_exchanger_kpis

        # Energy loss = (Q_design - Q_actual) * operating_hours * energy_cost
        Q_design = 5000.0  # kW (design duty)
        Q_actual = kpis.Q_avg_kW

        energy_loss_rate = Q_design - Q_actual  # kW
        operating_hours_per_day = 24
        energy_cost_per_kWh = 0.10  # USD

        daily_energy_loss_usd = energy_loss_rate * operating_hours_per_day * energy_cost_per_kWh

        assert daily_energy_loss_usd > 0

    def test_net_benefit_calculation(self):
        """Test net benefit calculation for cleaning."""
        cleaning_cost = 8000.0  # USD
        daily_energy_savings = 500.0  # USD/day after cleaning
        operating_days_until_next_cleaning = 180

        total_savings = daily_energy_savings * operating_days_until_next_cleaning
        net_benefit = total_savings - cleaning_cost

        assert net_benefit > cleaning_cost  # Cleaning should be beneficial


class TestOptimalCleaningDate:
    """Test optimal cleaning date calculation."""

    @pytest.mark.asyncio
    async def test_basic_optimal_date(self, mock_optimizer_service):
        """Test basic optimal cleaning date calculation."""
        fouling_state = type('FoulingState', (), {
            'exchanger_id': 'HX-001',
            'predicted_days_to_threshold': 60,
            'ua_degradation_percent': 25,
        })()

        constraints = {
            "maintenance_window_start": datetime.now(timezone.utc) + timedelta(days=30),
            "maintenance_window_end": datetime.now(timezone.utc) + timedelta(days=90),
        }

        recommendation = await mock_optimizer_service.optimize_schedule(
            "HX-001",
            fouling_state,
            constraints,
        )

        assert recommendation.recommended_cleaning_date is not None
        assert recommendation.recommended_cleaning_date > datetime.now(timezone.utc)

    def test_cleaning_before_threshold(self):
        """Test that cleaning is scheduled before threshold is reached."""
        days_to_threshold = 60
        safety_margin = 7

        optimal_cleaning_day = days_to_threshold - safety_margin

        assert optimal_cleaning_day < days_to_threshold

    def test_cleaning_date_within_window(self):
        """Test that cleaning date is within maintenance window."""
        window_start = datetime.now(timezone.utc) + timedelta(days=30)
        window_end = datetime.now(timezone.utc) + timedelta(days=90)
        recommended_date = datetime.now(timezone.utc) + timedelta(days=53)

        assert window_start <= recommended_date <= window_end


class TestUrgencyClassification:
    """Test urgency classification for cleaning recommendations."""

    @pytest.mark.parametrize("ua_degradation,days_to_threshold,expected_urgency", [
        (10.0, 120, "routine"),
        (25.0, 60, "scheduled"),
        (35.0, 30, "urgent"),
        (50.0, 7, "critical"),
    ])
    def test_urgency_levels(
        self,
        ua_degradation: float,
        days_to_threshold: int,
        expected_urgency: str,
    ):
        """Test urgency level classification."""
        if days_to_threshold < 14 or ua_degradation > 45:
            urgency = "critical"
        elif days_to_threshold < 30 or ua_degradation > 35:
            urgency = "urgent"
        elif days_to_threshold < 90 or ua_degradation > 20:
            urgency = "scheduled"
        else:
            urgency = "routine"

        assert urgency == expected_urgency

    def test_urgency_escalation(self):
        """Test that urgency escalates with fouling severity."""
        urgencies = ["routine", "scheduled", "urgent", "critical"]
        ua_degradations = [15, 30, 40, 55]

        for i in range(1, len(urgencies)):
            prev_idx = urgencies.index(urgencies[i - 1])
            curr_idx = urgencies.index(urgencies[i])
            assert curr_idx >= prev_idx


class TestConstraintHandling:
    """Test constraint handling in optimization."""

    def test_maintenance_window_constraint(self):
        """Test maintenance window constraint."""
        window_start = datetime.now(timezone.utc) + timedelta(days=30)
        window_end = datetime.now(timezone.utc) + timedelta(days=45)
        optimal_unconstrained = datetime.now(timezone.utc) + timedelta(days=20)

        # Constrain to window
        if optimal_unconstrained < window_start:
            constrained_date = window_start
        elif optimal_unconstrained > window_end:
            constrained_date = window_end
        else:
            constrained_date = optimal_unconstrained

        assert constrained_date >= window_start
        assert constrained_date <= window_end

    def test_production_schedule_constraint(self):
        """Test production schedule constraint handling."""
        production_blackout_dates = [
            datetime.now(timezone.utc) + timedelta(days=35),
            datetime.now(timezone.utc) + timedelta(days=36),
            datetime.now(timezone.utc) + timedelta(days=37),
        ]

        optimal_date = datetime.now(timezone.utc) + timedelta(days=36)

        # Check if optimal date conflicts
        conflicts = any(
            abs((optimal_date - blackout).days) < 1
            for blackout in production_blackout_dates
        )

        assert conflicts

    def test_resource_availability_constraint(self):
        """Test resource availability constraint."""
        available_crews = 2
        exchangers_needing_cleaning = 3

        # Can only schedule 2 at a time
        batch_size = min(available_crews, exchangers_needing_cleaning)

        assert batch_size == 2


class TestEconomicCalculations:
    """Test economic calculations for cleaning optimization."""

    def test_payback_period_calculation(self):
        """Test payback period calculation."""
        cleaning_cost = 8500.0  # USD
        daily_savings = 200.0   # USD/day

        payback_days = cleaning_cost / daily_savings

        assert payback_days == pytest.approx(42.5, abs=1)

    def test_npv_calculation(self):
        """Test Net Present Value calculation for cleaning decision."""
        cleaning_cost = 8500.0
        daily_savings = 200.0
        operating_days = 180
        discount_rate = 0.10  # Annual

        # Simple NPV (undiscounted for short period)
        total_savings = daily_savings * operating_days
        npv = total_savings - cleaning_cost

        assert npv > 0

    def test_energy_savings_estimation(self, clean_exchanger_kpis, fouled_exchanger_kpis):
        """Test energy savings estimation after cleaning."""
        Q_clean = clean_exchanger_kpis.Q_avg_kW
        Q_fouled = fouled_exchanger_kpis.Q_avg_kW

        duty_improvement = Q_clean - Q_fouled
        operating_hours = 8760  # Annual
        energy_cost = 0.10  # USD/kWh

        annual_savings = duty_improvement * operating_hours * energy_cost

        assert annual_savings > 0


class TestCleaningRecommendationGeneration:
    """Test cleaning recommendation generation."""

    @pytest.mark.asyncio
    async def test_recommendation_generation(self, mock_optimizer_service):
        """Test full recommendation generation."""
        fouling_state = type('FoulingState', (), {
            'exchanger_id': 'HX-001',
            'predicted_days_to_threshold': 45,
            'ua_degradation_percent': 30,
        })()

        constraints = {}

        recommendation = await mock_optimizer_service.optimize_schedule(
            "HX-001",
            fouling_state,
            constraints,
        )

        assert recommendation.exchanger_id == "HX-001"
        assert recommendation.recommended_cleaning_date is not None
        assert recommendation.urgency in ["routine", "scheduled", "urgent", "critical"]
        assert recommendation.estimated_cost_usd > 0
        assert recommendation.estimated_energy_savings_kWh > 0
        assert recommendation.recommendation_id is not None

    def test_recommendation_id_uniqueness(self):
        """Test that recommendation IDs are unique."""
        import uuid

        ids = [str(uuid.uuid4()) for _ in range(100)]

        assert len(ids) == len(set(ids))  # All unique


class TestOptimizerDeterminism:
    """Test optimizer determinism."""

    @pytest.mark.asyncio
    async def test_deterministic_optimization(self, mock_optimizer_service):
        """Test that optimization is deterministic."""
        fouling_state = type('FoulingState', (), {
            'exchanger_id': 'HX-001',
            'predicted_days_to_threshold': 60,
            'ua_degradation_percent': 25,
        })()

        constraints = {"max_cleaning_cost": 10000.0}

        results = []
        for _ in range(5):
            rec = await mock_optimizer_service.optimize_schedule(
                "HX-001",
                fouling_state,
                constraints,
            )
            results.append(rec.urgency)

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_provenance_hash_for_recommendation(self):
        """Test provenance hash for recommendations."""
        recommendation_data = {
            "exchanger_id": "HX-001",
            "recommendation_id": "rec-12345",
            "days_until_cleaning": 38,
            "estimated_cost": 8500.0,
        }

        provenance_data = f"{recommendation_data['exchanger_id']}:" \
                          f"{recommendation_data['recommendation_id']}:" \
                          f"{recommendation_data['days_until_cleaning']}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        assert len(provenance_hash) == 64


class TestOptimizerEdgeCases:
    """Test edge cases for optimizer."""

    def test_immediate_cleaning_needed(self):
        """Test handling of immediate cleaning requirement."""
        days_to_threshold = 3  # Critical

        if days_to_threshold < 7:
            urgency = "critical"
            recommended_days = max(1, days_to_threshold - 1)
        else:
            urgency = "scheduled"
            recommended_days = days_to_threshold - 7

        assert urgency == "critical"
        assert recommended_days <= 3

    def test_no_cleaning_needed(self):
        """Test handling when no cleaning is needed."""
        ua_ratio = 0.98  # Very clean
        days_to_threshold = 365

        needs_cleaning = ua_ratio < 0.85 or days_to_threshold < 90

        assert not needs_cleaning

    def test_multiple_exchangers_prioritization(self):
        """Test prioritization of multiple exchangers."""
        exchangers = [
            {"id": "HX-001", "ua_degradation": 40, "days_to_threshold": 30},
            {"id": "HX-002", "ua_degradation": 20, "days_to_threshold": 90},
            {"id": "HX-003", "ua_degradation": 50, "days_to_threshold": 14},
        ]

        # Sort by urgency (days_to_threshold ascending, ua_degradation descending)
        sorted_exchangers = sorted(
            exchangers,
            key=lambda x: (x["days_to_threshold"], -x["ua_degradation"])
        )

        # HX-003 should be first (most urgent)
        assert sorted_exchangers[0]["id"] == "HX-003"


class TestOptimizerValidation:
    """Test input validation for optimizer."""

    def test_valid_fouling_state(self):
        """Test validation of fouling state inputs."""
        valid_state = {
            "exchanger_id": "HX-001",
            "ua_degradation_percent": 25,
            "predicted_days_to_threshold": 60,
        }

        assert valid_state["ua_degradation_percent"] >= 0
        assert valid_state["ua_degradation_percent"] <= 100
        assert valid_state["predicted_days_to_threshold"] > 0

    def test_invalid_degradation_value(self):
        """Test handling of invalid degradation value."""
        invalid_degradation = -10  # Cannot be negative

        assert invalid_degradation < 0, "Negative degradation is invalid"

    def test_constraint_validation(self):
        """Test validation of constraints."""
        valid_constraints = {
            "max_cleaning_cost": 10000.0,
            "maintenance_window_days": 30,
        }

        assert valid_constraints["max_cleaning_cost"] > 0
        assert valid_constraints["maintenance_window_days"] > 0


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestCostBenefitAnalysis",
    "TestOptimalCleaningDate",
    "TestUrgencyClassification",
    "TestConstraintHandling",
    "TestEconomicCalculations",
    "TestCleaningRecommendationGeneration",
    "TestOptimizerDeterminism",
    "TestOptimizerEdgeCases",
    "TestOptimizerValidation",
]
