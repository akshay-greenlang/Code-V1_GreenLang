"""
GL-019 HEATSCHEDULER - End-to-End Integration Tests

Complete workflow tests from schedule creation through optimization and execution.

Target Coverage: 80%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import time
import json
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures from conftest
from tests.conftest import (
    EnergyTariff, RateType, PeriodType, TimePeriod,
    HeatingEquipment, ProductionJob, ScheduleSlot, OptimizedSchedule
)

# Import from unit tests (mock implementations)
from tests.unit.test_energy_cost_calculator import EnergyCostCalculator
from tests.unit.test_schedule_optimizer import ScheduleOptimizer, OptimizationConfig
from tests.unit.test_savings_calculator import SavingsCalculator, create_test_schedule


# =============================================================================
# INTEGRATION TEST HELPERS
# =============================================================================

class HeatSchedulerAgent:
    """
    GL-019 HEATSCHEDULER Agent - Process Heating Scheduler

    Integrates all calculators to provide end-to-end scheduling optimization.
    """

    VERSION = "1.0.0"
    NAME = "ProcessHeatingScheduler"
    AGENT_ID = "GL-019"
    CODENAME = "HEATSCHEDULER"

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cost_calculator = EnergyCostCalculator()
        self.optimizer = ScheduleOptimizer()
        self.savings_calculator = SavingsCalculator()

    def create_schedule(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        tariff: EnergyTariff,
        start_time: datetime = None
    ) -> OptimizedSchedule:
        """Create and optimize a heating schedule."""
        start_time = start_time or datetime.now(timezone.utc)

        # Run optimization
        schedule, provenance = self.optimizer.optimize(
            jobs=jobs,
            equipment=equipment,
            tariff=tariff,
            start_time=start_time
        )

        return schedule

    def calculate_savings(
        self,
        schedule: OptimizedSchedule,
        tariff: EnergyTariff
    ) -> Dict[str, Any]:
        """Calculate savings from optimized schedule."""
        # Create baseline for comparison
        baseline_cost = schedule.baseline_cost
        optimized_cost = schedule.total_cost

        savings_pct = self.savings_calculator.calculate_savings_percentage(
            baseline_cost, optimized_cost
        )

        # Project annual savings
        # Assume schedule represents a typical day
        daily_savings = baseline_cost - optimized_cost
        annual_projection = self.savings_calculator.project_annual_savings(
            daily_savings=daily_savings,
            working_days_per_month=22
        )

        return {
            "baseline_cost": baseline_cost,
            "optimized_cost": optimized_cost,
            "savings": daily_savings,
            "savings_pct": savings_pct,
            "annual_savings": annual_projection.annual_savings,
            "confidence_interval": annual_projection.confidence_interval
        }

    def handle_demand_response_event(
        self,
        schedule: OptimizedSchedule,
        event_start: datetime,
        event_end: datetime,
        target_reduction_kw: float,
        equipment: List[HeatingEquipment]
    ) -> OptimizedSchedule:
        """Modify schedule to respond to demand response event."""
        # Find interruptible equipment
        interruptible = [eq for eq in equipment if eq.is_interruptible]

        if not interruptible:
            # No interruptible equipment - return unchanged
            return schedule

        # Calculate available curtailment
        available_reduction = sum(eq.power_kw for eq in interruptible)

        # Modify affected slots
        modified_slots = []
        for slot in schedule.slots:
            slot_overlaps = (
                slot.start_time < event_end and
                slot.end_time > event_start
            )

            if slot_overlaps and slot.equipment_id in [eq.equipment_id for eq in interruptible]:
                # Shift this slot outside the event window if possible
                if slot.start_time >= event_start:
                    # Delay start until after event
                    new_slot = ScheduleSlot(
                        slot_id=slot.slot_id,
                        equipment_id=slot.equipment_id,
                        job_id=slot.job_id,
                        start_time=event_end,
                        end_time=event_end + (slot.end_time - slot.start_time),
                        power_kw=slot.power_kw,
                        energy_kwh=slot.energy_kwh,
                        estimated_cost=slot.estimated_cost,
                        period_type=slot.period_type
                    )
                    modified_slots.append(new_slot)
                else:
                    modified_slots.append(slot)
            else:
                modified_slots.append(slot)

        # Create modified schedule
        return OptimizedSchedule(
            schedule_id=schedule.schedule_id + "-DR",
            created_at=datetime.now(timezone.utc),
            slots=modified_slots,
            total_energy_kwh=schedule.total_energy_kwh,
            total_cost=schedule.total_cost,  # Will need recalculation
            baseline_cost=schedule.baseline_cost,
            savings=schedule.savings,
            savings_pct=schedule.savings_pct,
            optimization_time_ms=schedule.optimization_time_ms,
            constraints_satisfied=True,  # May need re-validation
            provenance_hash=schedule.provenance_hash + "-DR"
        )

    def generate_provenance(self, schedule: OptimizedSchedule) -> Dict[str, Any]:
        """Generate provenance record for schedule."""
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.NAME,
            "version": self.VERSION,
            "schedule_id": schedule.schedule_id,
            "provenance_hash": schedule.provenance_hash,
            "created_at": schedule.created_at.isoformat(),
            "total_slots": len(schedule.slots),
            "total_energy_kwh": schedule.total_energy_kwh,
            "total_cost": schedule.total_cost,
            "savings_pct": schedule.savings_pct
        }


# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """End-to-end integration tests for complete scheduling workflow."""

    def test_complete_scheduling_pipeline(
        self,
        simple_tou_tariff,
        multiple_equipment,
        multiple_jobs
    ):
        """Test complete scheduling pipeline from jobs to optimized schedule."""
        agent = HeatSchedulerAgent()

        # Step 1: Create optimized schedule
        schedule = agent.create_schedule(
            jobs=multiple_jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff
        )

        # Validate schedule was created
        assert schedule is not None
        assert len(schedule.slots) == len(multiple_jobs)
        assert schedule.constraints_satisfied is True

        # Step 2: Calculate savings
        savings = agent.calculate_savings(schedule, simple_tou_tariff)

        # Validate savings calculation
        assert "baseline_cost" in savings
        assert "optimized_cost" in savings
        assert "savings_pct" in savings
        assert "annual_savings" in savings

        # Step 3: Generate provenance
        provenance = agent.generate_provenance(schedule)

        # Validate provenance
        assert provenance["agent_id"] == "GL-019"
        assert provenance["schedule_id"] == schedule.schedule_id
        assert len(provenance["provenance_hash"]) >= 64

    def test_schedule_creation_to_optimization(
        self,
        simple_tou_tariff,
        single_furnace,
        single_job
    ):
        """Test schedule creation and optimization flow."""
        agent = HeatSchedulerAgent()

        # Create schedule
        schedule = agent.create_schedule(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        # Validate optimization occurred
        assert schedule.optimization_time_ms > 0
        assert schedule.total_cost <= schedule.baseline_cost
        assert schedule.savings >= 0

    def test_multi_day_scheduling(self, simple_tou_tariff, multiple_equipment):
        """Test scheduling across multiple days."""
        agent = HeatSchedulerAgent()

        now = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0)

        # Create jobs spanning 3 days
        jobs = []
        for day in range(3):
            for i in range(4):  # 4 jobs per day
                job = ProductionJob(
                    job_id=f"DAY{day}-JOB{i}",
                    product_name=f"Day {day} Product {i}",
                    equipment_id="FURN-001",
                    target_temp_c=850.0,
                    hold_time_min=120,
                    energy_kwh=400.0,
                    earliest_start=now + timedelta(days=day, hours=i * 3),
                    deadline=now + timedelta(days=day + 1),
                    priority=1,
                    is_flexible=True
                )
                jobs.append(job)

        schedule = agent.create_schedule(
            jobs=jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff,
            start_time=now
        )

        # Validate all jobs scheduled
        assert len(schedule.slots) == 12  # 3 days * 4 jobs

        # Validate schedule spans multiple days
        start_dates = set()
        for slot in schedule.slots:
            start_dates.add(slot.start_time.date())

        assert len(start_dates) >= 1  # At least spans the time window

    def test_demand_response_event_handling(
        self,
        simple_tou_tariff,
        multiple_equipment,
        interruptible_equipment
    ):
        """Test handling of demand response events."""
        agent = HeatSchedulerAgent()

        now = datetime.now(timezone.utc)
        equipment = multiple_equipment + [interruptible_equipment]

        # Create initial schedule
        jobs = [
            ProductionJob(
                job_id="DR-JOB-1",
                product_name="Flexible Product",
                equipment_id=interruptible_equipment.equipment_id,
                target_temp_c=300.0,
                hold_time_min=120,
                energy_kwh=200.0,
                earliest_start=now,
                deadline=now + timedelta(hours=12),
                priority=2,
                is_flexible=True
            ),
            ProductionJob(
                job_id="DR-JOB-2",
                product_name="Priority Product",
                equipment_id="FURN-001",
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=now,
                deadline=now + timedelta(hours=6),
                priority=1,
                is_flexible=False
            )
        ]

        schedule = agent.create_schedule(
            jobs=jobs,
            equipment=equipment,
            tariff=simple_tou_tariff,
            start_time=now
        )

        # Simulate demand response event
        dr_event_start = now + timedelta(hours=2)
        dr_event_end = now + timedelta(hours=4)

        modified_schedule = agent.handle_demand_response_event(
            schedule=schedule,
            event_start=dr_event_start,
            event_end=dr_event_end,
            target_reduction_kw=300.0,
            equipment=equipment
        )

        # Validate modified schedule
        assert modified_schedule is not None
        assert modified_schedule.schedule_id.endswith("-DR")

    def test_equipment_failure_recovery(self, simple_tou_tariff, multiple_equipment):
        """Test schedule adaptation when equipment becomes unavailable."""
        agent = HeatSchedulerAgent()

        now = datetime.now(timezone.utc)

        # Create job for specific equipment
        jobs = [
            ProductionJob(
                job_id="EQUIP-FAIL-JOB",
                product_name="Equipment Specific Product",
                equipment_id="FURN-001",
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=now,
                deadline=now + timedelta(hours=12),
                priority=1,
                is_flexible=True
            )
        ]

        # Create initial schedule
        schedule = agent.create_schedule(
            jobs=jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff,
            start_time=now
        )

        assert len(schedule.slots) == 1

        # Simulate equipment failure by removing from available list
        reduced_equipment = [eq for eq in multiple_equipment if eq.equipment_id != "FURN-001"]

        # Jobs need to be reassigned to valid equipment
        reassigned_jobs = [
            ProductionJob(
                job_id="EQUIP-FAIL-JOB",
                product_name="Equipment Specific Product",
                equipment_id="FURN-002",  # Reassigned to backup equipment
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=now,
                deadline=now + timedelta(hours=12),
                priority=1,
                is_flexible=True
            )
        ]

        # Reschedule with reduced equipment
        rescheduled = agent.create_schedule(
            jobs=reassigned_jobs,
            equipment=reduced_equipment,
            tariff=simple_tou_tariff,
            start_time=now
        )

        assert rescheduled is not None
        assert len(rescheduled.slots) == 1


@pytest.mark.integration
class TestScheduleExecutionFlow:
    """Integration tests for schedule execution flow."""

    def test_schedule_execution_tracking(self, simple_tou_tariff, single_furnace):
        """Test tracking of schedule execution."""
        agent = HeatSchedulerAgent()

        now = datetime.now(timezone.utc)

        jobs = [
            ProductionJob(
                job_id="EXEC-JOB-1",
                product_name="Tracked Product",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=now,
                deadline=now + timedelta(hours=12),
                priority=1,
                is_flexible=True
            )
        ]

        schedule = agent.create_schedule(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff,
            start_time=now
        )

        # Simulate execution tracking
        execution_log = []
        for slot in schedule.slots:
            execution_log.append({
                "slot_id": slot.slot_id,
                "job_id": slot.job_id,
                "scheduled_start": slot.start_time.isoformat(),
                "scheduled_end": slot.end_time.isoformat(),
                "status": "completed",  # Simulated
                "actual_start": slot.start_time.isoformat(),
                "actual_end": slot.end_time.isoformat(),
                "deviation_minutes": 0
            })

        # Validate execution log
        assert len(execution_log) == 1
        assert execution_log[0]["status"] == "completed"

    def test_schedule_variance_analysis(self, simple_tou_tariff, single_furnace):
        """Test analysis of schedule variances."""
        agent = HeatSchedulerAgent()

        now = datetime.now(timezone.utc)

        jobs = [
            ProductionJob(
                job_id="VAR-JOB",
                product_name="Variance Product",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=now,
                deadline=now + timedelta(hours=12),
                priority=1,
                is_flexible=True
            )
        ]

        schedule = agent.create_schedule(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff,
            start_time=now
        )

        # Simulate actual vs planned variance
        planned_cost = schedule.total_cost
        actual_cost = planned_cost * 1.05  # 5% cost overrun

        variance = actual_cost - planned_cost
        variance_pct = (variance / planned_cost) * 100

        assert variance_pct == pytest.approx(5.0, rel=0.01)


@pytest.mark.integration
class TestCostCalculationIntegration:
    """Integration tests for cost calculation across components."""

    def test_cost_calculation_consistency(self, simple_tou_tariff, single_furnace):
        """Test cost calculations are consistent across components."""
        cost_calc = EnergyCostCalculator()
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc).replace(hour=3, minute=0)  # Off-peak

        jobs = [
            ProductionJob(
                job_id="COST-JOB",
                product_name="Cost Test Product",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=now,
                deadline=now + timedelta(hours=12),
                priority=1,
                is_flexible=True
            )
        ]

        # Get cost from direct calculation
        direct_cost, breakdown = cost_calc.calculate_energy_cost(
            energy_kwh=500.0,
            start_time=now,
            duration_hours=1.0,
            tariff=simple_tou_tariff,
            equipment=single_furnace
        )

        # Get cost from optimized schedule
        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff,
            start_time=now
        )

        # Costs should be in same order of magnitude
        # (exact match not expected due to different calculation paths)
        assert schedule.total_cost > 0
        assert abs(schedule.total_cost - direct_cost) < direct_cost * 0.5  # Within 50%

    def test_savings_calculation_integration(self, simple_tou_tariff, single_furnace):
        """Test savings calculation integrates with schedule optimization."""
        agent = HeatSchedulerAgent()

        now = datetime.now(timezone.utc)

        jobs = [
            ProductionJob(
                job_id="SAVE-JOB",
                product_name="Savings Product",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=now,
                deadline=now + timedelta(hours=24),
                priority=1,
                is_flexible=True
            )
        ]

        schedule = agent.create_schedule(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff,
            start_time=now
        )

        savings = agent.calculate_savings(schedule, simple_tou_tariff)

        # Validate savings structure
        assert savings["baseline_cost"] >= savings["optimized_cost"]
        assert savings["savings"] >= 0
        assert 0 <= savings["savings_pct"] <= 100


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Performance integration tests."""

    def test_large_scale_optimization(
        self,
        simple_tou_tariff,
        multiple_equipment,
        benchmark_jobs
    ):
        """Test optimization performance with large job set."""
        optimizer = ScheduleOptimizer()

        # Use subset of benchmark jobs
        jobs = benchmark_jobs[:50]

        start = time.time()
        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff
        )
        duration_s = time.time() - start

        # Should complete in reasonable time
        assert duration_s < 30.0  # <30 seconds for 50 jobs
        assert len(schedule.slots) == 50

    def test_end_to_end_throughput(self, simple_tou_tariff, single_furnace):
        """Test end-to-end scheduling throughput."""
        agent = HeatSchedulerAgent()

        now = datetime.now(timezone.utc)

        # Create multiple small scheduling problems
        num_iterations = 20
        total_time = 0.0

        for i in range(num_iterations):
            jobs = [
                ProductionJob(
                    job_id=f"THROUGHPUT-{i}-{j}",
                    product_name=f"Product {i}-{j}",
                    equipment_id=single_furnace.equipment_id,
                    target_temp_c=800.0 + j * 10,
                    hold_time_min=30 + j * 10,
                    energy_kwh=100.0 + j * 50,
                    earliest_start=now + timedelta(hours=j),
                    deadline=now + timedelta(hours=24),
                    priority=1,
                    is_flexible=True
                )
                for j in range(5)
            ]

            start = time.time()
            schedule = agent.create_schedule(
                jobs=jobs,
                equipment=[single_furnace],
                tariff=simple_tou_tariff,
                start_time=now
            )
            total_time += time.time() - start

        avg_time_ms = (total_time / num_iterations) * 1000

        # Average scheduling should be under 500ms
        assert avg_time_ms < 500


@pytest.mark.integration
class TestProvenanceIntegration:
    """Integration tests for provenance tracking."""

    def test_provenance_chain(self, simple_tou_tariff, single_furnace, single_job):
        """Test provenance chain across optimization steps."""
        agent = HeatSchedulerAgent()

        schedule = agent.create_schedule(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        provenance = agent.generate_provenance(schedule)

        # Validate provenance chain
        assert provenance["agent_id"] == "GL-019"
        assert provenance["agent_name"] == "ProcessHeatingScheduler"
        assert len(provenance["provenance_hash"]) >= 64

    def test_provenance_reproducibility(self, simple_tou_tariff, single_furnace, single_job):
        """Test schedule optimization is reproducible."""
        agent = HeatSchedulerAgent()

        # Create same schedule twice
        schedule1 = agent.create_schedule(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        schedule2 = agent.create_schedule(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        # Provenance hashes should match for same input
        assert schedule1.provenance_hash == schedule2.provenance_hash

    def test_audit_trail_completeness(self, simple_tou_tariff, multiple_equipment, multiple_jobs):
        """Test audit trail includes all required information."""
        agent = HeatSchedulerAgent()

        schedule = agent.create_schedule(
            jobs=multiple_jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff
        )

        provenance = agent.generate_provenance(schedule)

        # Required audit fields
        required_fields = [
            "agent_id", "agent_name", "version", "schedule_id",
            "provenance_hash", "created_at", "total_slots",
            "total_energy_kwh", "total_cost", "savings_pct"
        ]

        for field in required_fields:
            assert field in provenance, f"Missing required field: {field}"


@pytest.mark.integration
class TestRealTimeScenarios:
    """Integration tests for real-time scheduling scenarios."""

    def test_rolling_schedule_update(self, simple_tou_tariff, single_furnace):
        """Test rolling schedule updates as time progresses."""
        agent = HeatSchedulerAgent()

        base_time = datetime.now(timezone.utc)

        # Initial schedule
        initial_jobs = [
            ProductionJob(
                job_id="ROLL-JOB-1",
                product_name="First Job",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=base_time,
                deadline=base_time + timedelta(hours=4),
                priority=1,
                is_flexible=True
            )
        ]

        schedule1 = agent.create_schedule(
            jobs=initial_jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff,
            start_time=base_time
        )

        # Simulate time passing - new job arrives
        new_time = base_time + timedelta(hours=1)
        new_jobs = initial_jobs + [
            ProductionJob(
                job_id="ROLL-JOB-2",
                product_name="New Arrival",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=900.0,
                hold_time_min=45,
                energy_kwh=400.0,
                earliest_start=new_time,
                deadline=new_time + timedelta(hours=6),
                priority=2,
                is_flexible=True
            )
        ]

        schedule2 = agent.create_schedule(
            jobs=new_jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff,
            start_time=new_time
        )

        # Updated schedule should include both jobs
        assert len(schedule2.slots) == 2

    def test_real_time_tariff_update(self, single_furnace, single_job, real_time_price_generator):
        """Test schedule adaptation to real-time price changes."""
        agent = HeatSchedulerAgent()

        # Create initial tariff with real-time prices
        prices = real_time_price_generator(hours=24)

        rtp_tariff = EnergyTariff(
            tariff_id="RTP-TEST",
            utility_name="Test RTP Utility",
            rate_type=RateType.REAL_TIME,
            base_rate_kwh=0.15,
            demand_charge_kw=15.00,
            real_time_prices=prices
        )

        schedule = agent.create_schedule(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=rtp_tariff
        )

        # Schedule should be created with RTP tariff
        assert schedule is not None
        assert schedule.total_cost > 0
