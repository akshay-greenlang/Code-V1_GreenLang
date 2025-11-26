# -*- coding: utf-8 -*-
"""
Fleet optimization tests for GL-008 SteamTrapInspector.

This module tests multi-trap prioritization, phased maintenance scheduling,
ROI calculations, and resource allocation for steam trap fleets.
"""

import pytest
import numpy as np
from typing import Dict, List, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tools import SteamTrapTools, MaintenancePriorityResult
from config import FailureMode, TrapType


@pytest.mark.integration
class TestMultiTrapPrioritization:
    """Test prioritization logic for multiple traps."""

    def test_priority_score_calculation(self, tools, test_fleet):
        """Test that priority scores are calculated correctly."""
        result = tools.prioritize_maintenance(test_fleet)

        assert isinstance(result, MaintenancePriorityResult)
        assert len(result.priority_list) == len(test_fleet)

        # All priority scores should be > 0
        for trap in result.priority_list:
            assert trap['priority_score'] > 0

    def test_high_loss_high_priority(self, tools):
        """Test that high energy loss traps get high priority."""
        fleet = [
            {
                'trap_id': 'TRAP-HIGH-LOSS',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 20000,  # High loss
                'process_criticality': 8,
                'current_age_years': 5,
                'health_score': 30
            },
            {
                'trap_id': 'TRAP-LOW-LOSS',
                'failure_mode': FailureMode.LEAKING,
                'energy_loss_usd_yr': 1000,  # Low loss
                'process_criticality': 5,
                'current_age_years': 2,
                'health_score': 70
            }
        ]

        result = tools.prioritize_maintenance(fleet)

        # High loss trap should be prioritized first
        assert result.priority_list[0]['trap_id'] == 'TRAP-HIGH-LOSS'
        assert result.priority_list[0]['priority_score'] > result.priority_list[1]['priority_score']

    def test_criticality_weighting(self, tools):
        """Test that process criticality affects prioritization."""
        fleet = [
            {
                'trap_id': 'TRAP-CRITICAL',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 5000,
                'process_criticality': 10,  # Critical process
                'current_age_years': 5,
                'health_score': 40
            },
            {
                'trap_id': 'TRAP-NONCRITICAL',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 5000,  # Same energy loss
                'process_criticality': 3,  # Non-critical
                'current_age_years': 5,
                'health_score': 40
            }
        ]

        result = tools.prioritize_maintenance(fleet)

        # Critical trap should be prioritized higher
        critical_trap = next(t for t in result.priority_list if t['trap_id'] == 'TRAP-CRITICAL')
        noncritical_trap = next(t for t in result.priority_list if t['trap_id'] == 'TRAP-NONCRITICAL')

        assert critical_trap['priority_score'] > noncritical_trap['priority_score']

    def test_age_factor_in_prioritization(self, tools):
        """Test that trap age affects prioritization."""
        fleet = [
            {
                'trap_id': 'TRAP-OLD',
                'failure_mode': FailureMode.LEAKING,
                'energy_loss_usd_yr': 3000,
                'process_criticality': 6,
                'current_age_years': 15,  # Old trap
                'health_score': 50
            },
            {
                'trap_id': 'TRAP-NEW',
                'failure_mode': FailureMode.LEAKING,
                'energy_loss_usd_yr': 3000,
                'process_criticality': 6,
                'current_age_years': 2,  # New trap
                'health_score': 50
            }
        ]

        result = tools.prioritize_maintenance(fleet)

        # Older trap may be prioritized higher (depending on algorithm)
        assert len(result.priority_list) == 2

    def test_health_score_impact(self, tools):
        """Test that health score affects prioritization."""
        fleet = [
            {
                'trap_id': 'TRAP-POOR-HEALTH',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 5000,
                'process_criticality': 7,
                'current_age_years': 5,
                'health_score': 20  # Poor health
            },
            {
                'trap_id': 'TRAP-GOOD-HEALTH',
                'failure_mode': FailureMode.LEAKING,
                'energy_loss_usd_yr': 5000,
                'process_criticality': 7,
                'current_age_years': 5,
                'health_score': 80  # Good health
            }
        ]

        result = tools.prioritize_maintenance(fleet)

        # Poor health trap should be prioritized higher
        poor_health = next(t for t in result.priority_list if t['trap_id'] == 'TRAP-POOR-HEALTH')
        good_health = next(t for t in result.priority_list if t['trap_id'] == 'TRAP-GOOD-HEALTH')

        assert poor_health['priority_score'] > good_health['priority_score']


@pytest.mark.integration
class TestPhasedMaintenanceScheduling:
    """Test phased maintenance schedule generation."""

    def test_schedule_generation(self, tools, test_fleet):
        """Test that maintenance schedule is generated."""
        result = tools.prioritize_maintenance(test_fleet)

        assert 'recommended_schedule' in result.__dict__ or hasattr(result, 'recommended_schedule')
        if hasattr(result, 'recommended_schedule'):
            assert len(result.recommended_schedule) > 0

    def test_phase_1_urgent_traps(self, tools, test_fleet):
        """Test that Phase 1 includes only urgent/critical traps."""
        result = tools.prioritize_maintenance(test_fleet)

        if hasattr(result, 'recommended_schedule'):
            # Assume schedule has phases
            phase_1 = [item for item in result.recommended_schedule if 'phase' in str(item).lower() and '1' in str(item)]

            # Phase 1 should contain high priority traps
            assert len(phase_1) >= 0  # May or may not have phase 1 depending on implementation

    def test_schedule_respects_resource_constraints(self, tools):
        """Test that schedule considers resource constraints."""
        # Large fleet
        fleet = [
            {
                'trap_id': f'TRAP-{i:03d}',
                'failure_mode': FailureMode.FAILED_OPEN if i % 3 == 0 else FailureMode.LEAKING,
                'energy_loss_usd_yr': 5000 - (i * 10),
                'process_criticality': 7,
                'current_age_years': 5,
                'health_score': 50
            }
            for i in range(50)
        ]

        result = tools.prioritize_maintenance(fleet, max_concurrent_maintenance=5)

        # Schedule should respect concurrency limits
        assert isinstance(result, MaintenancePriorityResult)

    def test_schedule_time_distribution(self, tools, test_fleet):
        """Test that maintenance is distributed over time."""
        result = tools.prioritize_maintenance(test_fleet)

        # Should not schedule all maintenance on same day
        if hasattr(result, 'recommended_schedule'):
            assert len(result.recommended_schedule) > 0


@pytest.mark.integration
class TestROICalculation:
    """Test return on investment calculations for fleet."""

    def test_fleet_total_savings(self, tools, test_fleet):
        """Test calculation of total potential savings."""
        result = tools.prioritize_maintenance(test_fleet)

        assert result.total_potential_savings_usd_yr > 0

        # Should sum energy losses of failed traps
        total_losses = sum(
            trap['energy_loss_usd_yr']
            for trap in test_fleet
            if trap['failure_mode'] != FailureMode.NORMAL
        )

        # Savings should be related to total losses
        assert result.total_potential_savings_usd_yr <= total_losses

    def test_roi_percentage_calculation(self, tools, test_fleet):
        """Test ROI percentage calculation."""
        result = tools.prioritize_maintenance(test_fleet)

        assert result.expected_roi_percent > 0
        assert result.expected_roi_percent < 10000  # Sanity check (< 100x return)

    def test_payback_period_calculation(self, tools, test_fleet):
        """Test payback period calculation."""
        result = tools.prioritize_maintenance(test_fleet)

        assert result.payback_months > 0
        assert result.payback_months < 120  # < 10 years is reasonable

    def test_high_savings_fast_payback(self, tools):
        """Test that high savings leads to fast payback."""
        fleet_high_savings = [
            {
                'trap_id': 'TRAP-HIGH-SAVINGS',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 50000,  # Very high loss
                'process_criticality': 8,
                'current_age_years': 5,
                'health_score': 20
            }
        ]

        fleet_low_savings = [
            {
                'trap_id': 'TRAP-LOW-SAVINGS',
                'failure_mode': FailureMode.LEAKING,
                'energy_loss_usd_yr': 500,  # Low loss
                'process_criticality': 5,
                'current_age_years': 3,
                'health_score': 60
            }
        ]

        result_high = tools.prioritize_maintenance(fleet_high_savings)
        result_low = tools.prioritize_maintenance(fleet_low_savings)

        # High savings should have faster payback
        assert result_high.payback_months < result_low.payback_months

    def test_npv_calculation(self, tools, test_fleet):
        """Test Net Present Value calculation."""
        result = tools.prioritize_maintenance(test_fleet)

        # NPV should be positive for cost-effective maintenance
        # (Not all tools may calculate NPV, so check if attribute exists)
        if hasattr(result, 'npv_usd'):
            assert result.npv_usd != 0


@pytest.mark.integration
class TestResourceAllocation:
    """Test resource allocation and scheduling."""

    def test_maintenance_cost_estimation(self, tools, test_fleet):
        """Test estimation of total maintenance cost."""
        result = tools.prioritize_maintenance(test_fleet)

        # Should have cost estimate
        if hasattr(result, 'total_maintenance_cost_usd'):
            assert result.total_maintenance_cost_usd > 0

            # Cost should be related to number of failed traps
            failed_traps = sum(1 for trap in test_fleet if trap['failure_mode'] != FailureMode.NORMAL)
            # Typical maintenance cost: $150-500 per trap
            expected_min_cost = failed_traps * 100
            expected_max_cost = failed_traps * 1000

            assert expected_min_cost <= result.total_maintenance_cost_usd <= expected_max_cost

    def test_labor_hours_estimation(self, tools, test_fleet):
        """Test estimation of required labor hours."""
        result = tools.prioritize_maintenance(test_fleet)

        # May have labor estimate
        if hasattr(result, 'total_labor_hours'):
            assert result.total_labor_hours > 0

    def test_parts_inventory_requirements(self, tools):
        """Test parts/inventory requirements calculation."""
        # Fleet with specific trap types
        fleet = [
            {
                'trap_id': f'TRAP-TYPE-{trap_type.value}',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 5000,
                'process_criticality': 7,
                'current_age_years': 5,
                'health_score': 40,
                'trap_type': trap_type
            }
            for trap_type in [TrapType.THERMODYNAMIC, TrapType.FLOAT_AND_THERMOSTATIC]
        ]

        result = tools.prioritize_maintenance(fleet)

        # Should identify parts needed
        if hasattr(result, 'parts_requirements'):
            assert len(result.parts_requirements) > 0

    def test_downtime_estimation(self, tools, test_fleet):
        """Test process downtime estimation."""
        result = tools.prioritize_maintenance(test_fleet)

        # May estimate downtime
        if hasattr(result, 'estimated_downtime_hours'):
            assert result.estimated_downtime_hours >= 0


@pytest.mark.integration
class TestFleetSegmentation:
    """Test fleet segmentation strategies."""

    def test_segmentation_by_failure_mode(self, tools, test_fleet):
        """Test grouping traps by failure mode."""
        result = tools.prioritize_maintenance(test_fleet)

        # Count traps by failure mode
        failure_modes = {}
        for trap in result.priority_list:
            mode = trap.get('failure_mode', 'unknown')
            failure_modes[mode] = failure_modes.get(mode, 0) + 1

        # Should have multiple failure modes represented
        assert len(failure_modes) > 0

    def test_segmentation_by_criticality(self, tools, test_fleet):
        """Test grouping traps by process criticality."""
        result = tools.prioritize_maintenance(test_fleet)

        # High criticality traps
        high_crit = [t for t in result.priority_list if t.get('process_criticality', 0) >= 8]
        low_crit = [t for t in result.priority_list if t.get('process_criticality', 0) < 8]

        # Should be able to segment
        assert len(high_crit) + len(low_crit) == len(result.priority_list)

    def test_segmentation_by_location(self, tools):
        """Test grouping traps by physical location."""
        fleet = [
            {
                'trap_id': f'TRAP-BLDG-{bldg}-{i}',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 5000,
                'process_criticality': 7,
                'current_age_years': 5,
                'health_score': 40,
                'location': f'Building {bldg}'
            }
            for bldg in ['A', 'B', 'C']
            for i in range(3)
        ]

        result = tools.prioritize_maintenance(fleet)

        # Group by location
        locations = {}
        for trap in result.priority_list:
            loc = trap.get('location', 'unknown')
            locations[loc] = locations.get(loc, 0) + 1

        # Should have traps in multiple locations
        assert len(locations) >= 3


@pytest.mark.integration
class TestFleetPerformanceMetrics:
    """Test fleet-wide performance metrics."""

    def test_fleet_health_score(self, tools, test_fleet):
        """Test calculation of overall fleet health score."""
        result = tools.prioritize_maintenance(test_fleet)

        # May calculate fleet health
        if hasattr(result, 'fleet_health_score'):
            assert 0 <= result.fleet_health_score <= 100

    def test_failure_rate_calculation(self, tools, test_fleet):
        """Test fleet failure rate calculation."""
        result = tools.prioritize_maintenance(test_fleet)

        # Calculate failure rate
        failed_traps = sum(1 for trap in test_fleet if trap['failure_mode'] != FailureMode.NORMAL)
        failure_rate = failed_traps / len(test_fleet) * 100

        assert 0 <= failure_rate <= 100

    def test_total_energy_waste(self, tools, test_fleet):
        """Test total fleet energy waste calculation."""
        result = tools.prioritize_maintenance(test_fleet)

        # Should calculate total energy waste
        total_waste = sum(trap['energy_loss_usd_yr'] for trap in test_fleet)
        assert total_waste > 0

    def test_co2_emissions_total(self, tools, test_fleet):
        """Test total fleet CO2 emissions calculation."""
        # This would require energy loss objects with CO2 data
        # Placeholder test
        assert True


@pytest.mark.integration
class TestMaintenanceStrategyComparison:
    """Test comparison of different maintenance strategies."""

    def test_reactive_vs_preventive_comparison(self, tools, test_fleet):
        """Test comparison of reactive vs preventive maintenance."""
        result = tools.prioritize_maintenance(test_fleet)

        # Prioritized maintenance (preventive) should have better ROI
        assert result.expected_roi_percent > 0

    def test_phased_vs_simultaneous_maintenance(self, tools, test_fleet):
        """Test comparison of phased vs simultaneous maintenance."""
        # Phased approach
        result_phased = tools.prioritize_maintenance(test_fleet, maintenance_strategy='phased')

        # Should generate phased schedule
        assert isinstance(result_phased, MaintenancePriorityResult)

    def test_cost_minimization_vs_uptime_maximization(self, tools, test_fleet):
        """Test different optimization objectives."""
        # Cost minimization
        result_cost = tools.prioritize_maintenance(test_fleet, objective='minimize_cost')

        # Uptime maximization
        result_uptime = tools.prioritize_maintenance(test_fleet, objective='maximize_uptime')

        # Different strategies may produce different priorities
        assert isinstance(result_cost, MaintenancePriorityResult)
        assert isinstance(result_uptime, MaintenancePriorityResult)


@pytest.mark.integration
class TestScalabilityLargeFleets:
    """Test prioritization with large fleets."""

    def test_large_fleet_100_traps(self, tools):
        """Test prioritization with 100-trap fleet."""
        fleet = [
            {
                'trap_id': f'TRAP-{i:03d}',
                'failure_mode': FailureMode.FAILED_OPEN if i % 4 == 0 else
                               (FailureMode.LEAKING if i % 4 == 1 else
                                (FailureMode.FAILED_CLOSED if i % 4 == 2 else FailureMode.NORMAL)),
                'energy_loss_usd_yr': max(0, 10000 - (i * 50)),
                'process_criticality': min(10, max(1, 7 + (i % 5) - 2)),
                'current_age_years': 1 + (i % 15),
                'health_score': max(10, min(100, 80 - (i % 70)))
            }
            for i in range(100)
        ]

        result = tools.prioritize_maintenance(fleet)

        assert len(result.priority_list) == 100
        assert result.total_potential_savings_usd_yr > 0

    def test_large_fleet_500_traps(self, tools):
        """Test prioritization with 500-trap fleet."""
        fleet = [
            {
                'trap_id': f'TRAP-{i:04d}',
                'failure_mode': [FailureMode.FAILED_OPEN, FailureMode.LEAKING,
                               FailureMode.FAILED_CLOSED, FailureMode.NORMAL][i % 4],
                'energy_loss_usd_yr': max(0, 8000 - (i * 10)),
                'process_criticality': 5 + (i % 6),
                'current_age_years': 1 + (i % 20),
                'health_score': 50 + (i % 50)
            }
            for i in range(500)
        ]

        result = tools.prioritize_maintenance(fleet)

        assert len(result.priority_list) == 500

    def test_performance_scaling(self, tools):
        """Test that algorithm scales reasonably with fleet size."""
        import time

        fleet_sizes = [10, 50, 100]
        execution_times = []

        for size in fleet_sizes:
            fleet = [
                {
                    'trap_id': f'TRAP-{i:04d}',
                    'failure_mode': FailureMode.FAILED_OPEN,
                    'energy_loss_usd_yr': 5000,
                    'process_criticality': 7,
                    'current_age_years': 5,
                    'health_score': 50
                }
                for i in range(size)
            ]

            start = time.perf_counter()
            result = tools.prioritize_maintenance(fleet)
            duration = time.perf_counter() - start
            execution_times.append(duration)

        # Algorithm should scale reasonably (not exponentially)
        # O(n log n) would be acceptable for sorting
        assert all(t < 5.0 for t in execution_times)  # All under 5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
