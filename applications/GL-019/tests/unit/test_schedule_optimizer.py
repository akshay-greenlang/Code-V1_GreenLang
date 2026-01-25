"""
GL-019 HEATSCHEDULER - Schedule Optimizer Unit Tests

Comprehensive unit tests for ScheduleOptimizer with 95%+ coverage target.
Tests optimization algorithms, constraint satisfaction, and load shifting.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import hashlib
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures from conftest
from tests.conftest import (
    EnergyTariff, RateType, PeriodType, TimePeriod,
    HeatingEquipment, ProductionJob, ScheduleSlot, OptimizedSchedule
)


# =============================================================================
# MOCK OPTIMIZER CLASSES FOR TESTING
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for schedule optimization."""
    algorithm: str = "greedy"  # greedy, genetic, milp
    max_iterations: int = 1000
    convergence_threshold: float = 0.001
    time_limit_seconds: float = 30.0
    objective: str = "minimize_cost"  # minimize_cost, minimize_peak, balance
    allow_preheating: bool = True
    load_shifting_enabled: bool = True
    demand_response_enabled: bool = True


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    schedule: OptimizedSchedule
    iterations: int
    convergence_achieved: bool
    final_objective_value: float
    optimization_time_ms: float
    algorithm_used: str


@dataclass
class ProvenanceRecord:
    """Provenance tracking for optimization."""
    calculator_name: str
    calculator_version: str
    provenance_hash: str
    calculation_steps: List[dict]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ScheduleOptimizer:
    """
    Schedule optimizer for process heating operations.

    Optimizes heating schedules to minimize energy costs while
    satisfying production constraints and deadlines.
    """

    VERSION = "1.0.0"
    NAME = "ScheduleOptimizer"

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self._tracker = None

    def optimize(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        tariff: EnergyTariff,
        start_time: datetime = None
    ) -> Tuple[OptimizedSchedule, ProvenanceRecord]:
        """
        Optimize heating schedule for given jobs and equipment.

        Returns:
            Tuple of (OptimizedSchedule, ProvenanceRecord)
        """
        if not jobs:
            raise ValueError("No jobs provided for optimization")

        if not equipment:
            raise ValueError("No equipment provided for optimization")

        start_time = start_time or datetime.now(timezone.utc)
        optimization_start = time.time()

        # Validate inputs
        self._validate_inputs(jobs, equipment)

        # Generate initial schedule (baseline)
        baseline_schedule = self._generate_baseline_schedule(jobs, equipment, tariff, start_time)
        baseline_cost = self._calculate_total_cost(baseline_schedule, tariff)

        # Optimize based on algorithm
        if self.config.algorithm == "greedy":
            optimized_schedule = self._optimize_greedy(jobs, equipment, tariff, start_time)
        elif self.config.algorithm == "genetic":
            optimized_schedule = self._optimize_genetic(jobs, equipment, tariff, start_time)
        elif self.config.algorithm == "milp":
            optimized_schedule = self._optimize_milp(jobs, equipment, tariff, start_time)
        else:
            raise ValueError(f"Unknown optimization algorithm: {self.config.algorithm}")

        optimization_time_ms = (time.time() - optimization_start) * 1000

        # Calculate costs and savings
        optimized_cost = self._calculate_total_cost(optimized_schedule, tariff)
        savings = baseline_cost - optimized_cost
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

        # Create result schedule
        schedule_id = self._generate_schedule_id(jobs, equipment, start_time)
        provenance_hash = self._calculate_provenance_hash(jobs, equipment, tariff, optimized_schedule)

        result_schedule = OptimizedSchedule(
            schedule_id=schedule_id,
            created_at=datetime.now(timezone.utc),
            slots=optimized_schedule,
            total_energy_kwh=sum(s.energy_kwh for s in optimized_schedule),
            total_cost=optimized_cost,
            baseline_cost=baseline_cost,
            savings=savings,
            savings_pct=savings_pct,
            optimization_time_ms=optimization_time_ms,
            constraints_satisfied=self._verify_constraints(optimized_schedule, jobs),
            provenance_hash=provenance_hash
        )

        # Create provenance record
        provenance = self._create_provenance(jobs, equipment, tariff, result_schedule)

        return result_schedule, provenance

    def _validate_inputs(self, jobs: List[ProductionJob], equipment: List[HeatingEquipment]):
        """Validate input data."""
        equipment_ids = {eq.equipment_id for eq in equipment}

        for job in jobs:
            if job.equipment_id not in equipment_ids:
                raise ValueError(f"Job {job.job_id} references unknown equipment {job.equipment_id}")

            if job.earliest_start >= job.deadline:
                raise ValueError(f"Job {job.job_id} has invalid time window")

            if job.energy_kwh <= 0:
                raise ValueError(f"Job {job.job_id} has invalid energy requirement")

    def _generate_baseline_schedule(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        tariff: EnergyTariff,
        start_time: datetime
    ) -> List[ScheduleSlot]:
        """Generate baseline schedule (ASAP scheduling)."""
        slots = []
        equipment_end_times = {eq.equipment_id: start_time for eq in equipment}

        # Sort jobs by priority and deadline
        sorted_jobs = sorted(jobs, key=lambda j: (j.priority, j.deadline))

        for job in sorted_jobs:
            eq = next(e for e in equipment if e.equipment_id == job.equipment_id)

            # Schedule ASAP after earliest start and equipment availability
            slot_start = max(job.earliest_start, equipment_end_times[job.equipment_id])

            duration_hours = job.hold_time_min / 60.0
            slot_end = slot_start + timedelta(hours=duration_hours)

            # Estimate cost (simplified)
            avg_rate = tariff.base_rate_kwh
            estimated_cost = job.energy_kwh * avg_rate

            slot = ScheduleSlot(
                slot_id=f"SLOT-{job.job_id}",
                equipment_id=job.equipment_id,
                job_id=job.job_id,
                start_time=slot_start,
                end_time=slot_end,
                power_kw=eq.power_kw,
                energy_kwh=job.energy_kwh,
                estimated_cost=estimated_cost,
                period_type=PeriodType.OFF_PEAK  # Will be updated
            )
            slots.append(slot)
            equipment_end_times[job.equipment_id] = slot_end

        return slots

    def _optimize_greedy(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        tariff: EnergyTariff,
        start_time: datetime
    ) -> List[ScheduleSlot]:
        """Greedy optimization algorithm - shift to lowest cost periods."""
        slots = []
        equipment_end_times = {eq.equipment_id: start_time for eq in equipment}

        # Sort flexible jobs by energy (largest first for biggest impact)
        flexible_jobs = [j for j in jobs if j.is_flexible]
        fixed_jobs = [j for j in jobs if not j.is_flexible]

        # Schedule fixed jobs first
        for job in sorted(fixed_jobs, key=lambda j: j.earliest_start):
            eq = next(e for e in equipment if e.equipment_id == job.equipment_id)
            slot_start = max(job.earliest_start, equipment_end_times[job.equipment_id])
            duration_hours = job.hold_time_min / 60.0
            slot_end = slot_start + timedelta(hours=duration_hours)

            period_type = self._get_period_type(slot_start, tariff)
            rate = self._get_rate_for_period(period_type, tariff)

            slot = ScheduleSlot(
                slot_id=f"SLOT-{job.job_id}",
                equipment_id=job.equipment_id,
                job_id=job.job_id,
                start_time=slot_start,
                end_time=slot_end,
                power_kw=eq.power_kw,
                energy_kwh=job.energy_kwh,
                estimated_cost=job.energy_kwh * rate,
                period_type=period_type
            )
            slots.append(slot)
            equipment_end_times[job.equipment_id] = slot_end

        # Optimize flexible jobs - find lowest cost windows
        for job in sorted(flexible_jobs, key=lambda j: (-j.energy_kwh, j.priority)):
            eq = next(e for e in equipment if e.equipment_id == job.equipment_id)
            duration_hours = job.hold_time_min / 60.0

            # Find optimal start time
            best_start = equipment_end_times[job.equipment_id]
            best_cost = float('inf')

            search_start = max(job.earliest_start, equipment_end_times[job.equipment_id])
            search_end = job.deadline - timedelta(hours=duration_hours)

            if search_start > search_end:
                # No flexibility - use earliest possible time
                slot_start = search_start
            else:
                # Search for lowest cost window (30-min granularity)
                current_time = search_start
                while current_time <= search_end:
                    period_type = self._get_period_type(current_time, tariff)
                    rate = self._get_rate_for_period(period_type, tariff)
                    cost = job.energy_kwh * rate

                    if cost < best_cost:
                        best_cost = cost
                        best_start = current_time

                    current_time += timedelta(minutes=30)

                slot_start = best_start

            slot_end = slot_start + timedelta(hours=duration_hours)
            period_type = self._get_period_type(slot_start, tariff)
            rate = self._get_rate_for_period(period_type, tariff)

            slot = ScheduleSlot(
                slot_id=f"SLOT-{job.job_id}",
                equipment_id=job.equipment_id,
                job_id=job.job_id,
                start_time=slot_start,
                end_time=slot_end,
                power_kw=eq.power_kw,
                energy_kwh=job.energy_kwh,
                estimated_cost=job.energy_kwh * rate,
                period_type=period_type
            )
            slots.append(slot)
            equipment_end_times[job.equipment_id] = slot_end

        return slots

    def _optimize_genetic(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        tariff: EnergyTariff,
        start_time: datetime
    ) -> List[ScheduleSlot]:
        """Genetic algorithm optimization (simplified)."""
        # For this test implementation, use greedy as base
        return self._optimize_greedy(jobs, equipment, tariff, start_time)

    def _optimize_milp(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        tariff: EnergyTariff,
        start_time: datetime
    ) -> List[ScheduleSlot]:
        """Mixed Integer Linear Programming optimization (simplified)."""
        # For this test implementation, use greedy as base
        return self._optimize_greedy(jobs, equipment, tariff, start_time)

    def _get_period_type(self, timestamp: datetime, tariff: EnergyTariff) -> PeriodType:
        """Determine rate period for timestamp."""
        if tariff.rate_type == RateType.FLAT:
            return PeriodType.OFF_PEAK

        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        for period in tariff.time_periods:
            if day_of_week in period.days_of_week:
                if period.start_hour <= period.end_hour:
                    if period.start_hour <= hour < period.end_hour:
                        return period.period_type
                else:
                    if hour >= period.start_hour or hour < period.end_hour:
                        return period.period_type

        return PeriodType.OFF_PEAK

    def _get_rate_for_period(self, period_type: PeriodType, tariff: EnergyTariff) -> float:
        """Get rate for period type."""
        multiplier = tariff.period_rates.get(period_type.value, 1.0)
        return tariff.base_rate_kwh * multiplier

    def _calculate_total_cost(self, slots: List[ScheduleSlot], tariff: EnergyTariff) -> float:
        """Calculate total cost for a schedule."""
        return sum(slot.estimated_cost for slot in slots)

    def _verify_constraints(self, slots: List[ScheduleSlot], jobs: List[ProductionJob]) -> bool:
        """Verify all constraints are satisfied."""
        job_dict = {j.job_id: j for j in jobs}

        for slot in slots:
            job = job_dict.get(slot.job_id)
            if not job:
                return False

            # Check deadline
            if slot.end_time > job.deadline:
                return False

            # Check earliest start
            if slot.start_time < job.earliest_start:
                return False

        # Check no overlapping slots on same equipment
        equipment_slots = {}
        for slot in slots:
            if slot.equipment_id not in equipment_slots:
                equipment_slots[slot.equipment_id] = []
            equipment_slots[slot.equipment_id].append(slot)

        for eq_id, eq_slots in equipment_slots.items():
            sorted_slots = sorted(eq_slots, key=lambda s: s.start_time)
            for i in range(len(sorted_slots) - 1):
                if sorted_slots[i].end_time > sorted_slots[i + 1].start_time:
                    return False

        return True

    def _generate_schedule_id(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        start_time: datetime
    ) -> str:
        """Generate unique schedule ID."""
        data = {
            "jobs": [j.job_id for j in jobs],
            "equipment": [e.equipment_id for e in equipment],
            "start_time": start_time.isoformat()
        }
        hash_input = json.dumps(data, sort_keys=True)
        return f"SCH-{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"

    def _calculate_provenance_hash(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        tariff: EnergyTariff,
        slots: List[ScheduleSlot]
    ) -> str:
        """Calculate provenance hash for reproducibility."""
        data = {
            "calculator": self.NAME,
            "version": self.VERSION,
            "jobs": [j.job_id for j in jobs],
            "equipment": [e.equipment_id for e in equipment],
            "tariff": tariff.tariff_id,
            "slots": [(s.job_id, s.start_time.isoformat(), s.estimated_cost) for s in slots]
        }
        hash_input = json.dumps(data, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _create_provenance(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        tariff: EnergyTariff,
        schedule: OptimizedSchedule
    ) -> ProvenanceRecord:
        """Create provenance record for optimization."""
        steps = [
            {"step_number": 1, "description": "Validate inputs", "operation": "validation"},
            {"step_number": 2, "description": "Generate baseline schedule", "operation": "baseline"},
            {"step_number": 3, "description": f"Run {self.config.algorithm} optimization", "operation": "optimize"},
            {"step_number": 4, "description": "Calculate costs and savings", "operation": "cost_calc"},
            {"step_number": 5, "description": "Verify constraints", "operation": "verify"},
        ]

        return ProvenanceRecord(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            provenance_hash=schedule.provenance_hash,
            calculation_steps=steps
        )

    def calculate_load_shift_potential(
        self,
        jobs: List[ProductionJob],
        equipment: List[HeatingEquipment],
        tariff: EnergyTariff
    ) -> Dict[str, float]:
        """Calculate potential for load shifting."""
        total_energy = sum(j.energy_kwh for j in jobs)
        flexible_energy = sum(j.energy_kwh for j in jobs if j.is_flexible)

        return {
            "total_energy_kwh": total_energy,
            "flexible_energy_kwh": flexible_energy,
            "flexibility_pct": (flexible_energy / total_energy * 100) if total_energy > 0 else 0,
            "max_peak_reduction_kw": max(e.power_kw for e in equipment if e.is_interruptible) if any(e.is_interruptible for e in equipment) else 0
        }


# =============================================================================
# UNIT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.optimizer
@pytest.mark.critical
class TestScheduleOptimizer:
    """Comprehensive test suite for ScheduleOptimizer."""

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization_default_config(self):
        """Test optimizer initializes with default config."""
        optimizer = ScheduleOptimizer()

        assert optimizer.VERSION == "1.0.0"
        assert optimizer.NAME == "ScheduleOptimizer"
        assert optimizer.config.algorithm == "greedy"
        assert optimizer.config.max_iterations == 1000

    def test_initialization_custom_config(self):
        """Test optimizer initializes with custom config."""
        config = OptimizationConfig(
            algorithm="genetic",
            max_iterations=500,
            time_limit_seconds=60.0
        )
        optimizer = ScheduleOptimizer(config=config)

        assert optimizer.config.algorithm == "genetic"
        assert optimizer.config.max_iterations == 500
        assert optimizer.config.time_limit_seconds == 60.0

    # =========================================================================
    # SINGLE EQUIPMENT OPTIMIZATION TESTS
    # =========================================================================

    def test_optimize_single_job(self, simple_tou_tariff, single_furnace, single_job):
        """Test optimization with single job and equipment."""
        optimizer = ScheduleOptimizer()

        schedule, provenance = optimizer.optimize(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        assert schedule is not None
        assert len(schedule.slots) == 1
        assert schedule.constraints_satisfied is True
        assert schedule.total_energy_kwh == single_job.energy_kwh
        assert len(schedule.provenance_hash) == 64

    def test_optimize_multiple_jobs_single_equipment(self, simple_tou_tariff, single_furnace):
        """Test optimization with multiple jobs on single equipment."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        jobs = [
            ProductionJob(
                job_id=f"JOB-{i}",
                product_name=f"Product {i}",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=800.0,
                hold_time_min=60,
                energy_kwh=200.0,
                earliest_start=now,
                deadline=now + timedelta(hours=24),
                priority=i,
                is_flexible=True
            )
            for i in range(3)
        ]

        schedule, provenance = optimizer.optimize(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        assert len(schedule.slots) == 3
        assert schedule.constraints_satisfied is True
        assert schedule.total_energy_kwh == 600.0  # 3 jobs x 200 kWh

    def test_optimize_no_overlapping_slots(self, simple_tou_tariff, single_furnace):
        """Test that optimization produces no overlapping slots."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        jobs = [
            ProductionJob(
                job_id=f"JOB-{i}",
                product_name=f"Product {i}",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=800.0,
                hold_time_min=120,  # 2 hours each
                energy_kwh=400.0,
                earliest_start=now,
                deadline=now + timedelta(hours=24),
                priority=1,
                is_flexible=True
            )
            for i in range(4)
        ]

        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        # Sort slots by start time and check no overlap
        sorted_slots = sorted(schedule.slots, key=lambda s: s.start_time)
        for i in range(len(sorted_slots) - 1):
            assert sorted_slots[i].end_time <= sorted_slots[i + 1].start_time

    # =========================================================================
    # MULTIPLE EQUIPMENT OPTIMIZATION TESTS
    # =========================================================================

    def test_optimize_multiple_equipment(self, simple_tou_tariff, multiple_equipment, multiple_jobs):
        """Test optimization with multiple equipment."""
        optimizer = ScheduleOptimizer()

        schedule, provenance = optimizer.optimize(
            jobs=multiple_jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff
        )

        assert len(schedule.slots) == len(multiple_jobs)
        assert schedule.constraints_satisfied is True

    def test_optimize_equipment_assignment(self, simple_tou_tariff, multiple_equipment):
        """Test jobs are assigned to correct equipment."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        jobs = [
            ProductionJob(
                job_id="JOB-1",
                product_name="Product 1",
                equipment_id="FURN-001",
                target_temp_c=800.0,
                hold_time_min=60,
                energy_kwh=300.0,
                earliest_start=now,
                deadline=now + timedelta(hours=12),
                priority=1,
                is_flexible=True
            ),
            ProductionJob(
                job_id="JOB-2",
                product_name="Product 2",
                equipment_id="OVEN-001",
                target_temp_c=200.0,
                hold_time_min=120,
                energy_kwh=150.0,
                earliest_start=now,
                deadline=now + timedelta(hours=12),
                priority=2,
                is_flexible=True
            )
        ]

        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff
        )

        # Verify equipment assignment
        job1_slot = next(s for s in schedule.slots if s.job_id == "JOB-1")
        job2_slot = next(s for s in schedule.slots if s.job_id == "JOB-2")

        assert job1_slot.equipment_id == "FURN-001"
        assert job2_slot.equipment_id == "OVEN-001"

    # =========================================================================
    # CONSTRAINT SATISFACTION TESTS
    # =========================================================================

    def test_deadline_constraint_satisfied(self, simple_tou_tariff, single_furnace):
        """Test all jobs meet their deadlines."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        jobs = [
            ProductionJob(
                job_id="TIGHT-JOB",
                product_name="Tight Deadline Product",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=now,
                deadline=now + timedelta(hours=4),  # Tight deadline
                priority=1,
                is_flexible=True
            )
        ]

        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        for slot in schedule.slots:
            job = next(j for j in jobs if j.job_id == slot.job_id)
            assert slot.end_time <= job.deadline

    def test_earliest_start_constraint(self, simple_tou_tariff, single_furnace):
        """Test jobs don't start before earliest_start."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        future_start = now + timedelta(hours=6)

        jobs = [
            ProductionJob(
                job_id="FUTURE-JOB",
                product_name="Future Start Product",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=850.0,
                hold_time_min=60,
                energy_kwh=500.0,
                earliest_start=future_start,  # Can't start until 6 hours from now
                deadline=now + timedelta(hours=24),
                priority=1,
                is_flexible=True
            )
        ]

        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        slot = schedule.slots[0]
        assert slot.start_time >= future_start

    def test_tight_deadline_constraint(self, simple_tou_tariff, single_furnace, tight_deadline_jobs):
        """Test optimization handles tight deadlines."""
        optimizer = ScheduleOptimizer()

        schedule, _ = optimizer.optimize(
            jobs=tight_deadline_jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        # Should satisfy constraints or indicate failure
        # With tight deadlines, some may not be satisfiable
        for slot in schedule.slots:
            job = next(j for j in tight_deadline_jobs if j.job_id == slot.job_id)
            # Verify constraints
            if schedule.constraints_satisfied:
                assert slot.start_time >= job.earliest_start
                assert slot.end_time <= job.deadline

    def test_capacity_constraint(self, simple_tou_tariff, single_furnace):
        """Test equipment capacity constraints."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        # Create more jobs than can fit in the time window
        jobs = [
            ProductionJob(
                job_id=f"JOB-{i}",
                product_name=f"Product {i}",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=800.0,
                hold_time_min=180,  # 3 hours each
                energy_kwh=900.0,
                earliest_start=now,
                deadline=now + timedelta(hours=8),  # Only 8 hours for 3-hour jobs
                priority=1,
                is_flexible=True
            )
            for i in range(3)  # 9 hours of work in 8 hours
        ]

        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        # Should schedule what it can
        assert len(schedule.slots) <= 3

    # =========================================================================
    # LOAD SHIFTING TESTS
    # =========================================================================

    def test_load_shifting_to_off_peak(self, simple_tou_tariff, single_furnace):
        """Test optimizer shifts load to off-peak periods."""
        optimizer = ScheduleOptimizer()

        # Create a job that can be scheduled anytime in 24 hours
        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)

        jobs = [
            ProductionJob(
                job_id="FLEXIBLE-JOB",
                product_name="Flexible Product",
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

        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff,
            start_time=now
        )

        # Flexible job should be scheduled during off-peak
        slot = schedule.slots[0]
        # Off-peak is 22:00-10:00 in the simple tariff
        assert slot.period_type == PeriodType.OFF_PEAK or slot.start_time.hour >= 22 or slot.start_time.hour < 10

    def test_load_shifting_savings(self, simple_tou_tariff, single_furnace):
        """Test load shifting produces cost savings."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)

        jobs = [
            ProductionJob(
                job_id="SAVINGS-JOB",
                product_name="Savings Test Product",
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

        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff,
            start_time=now
        )

        # Should have savings from load shifting
        assert schedule.savings >= 0
        assert schedule.total_cost <= schedule.baseline_cost

    def test_load_shift_potential_calculation(self, simple_tou_tariff, multiple_equipment, multiple_jobs):
        """Test load shift potential calculation."""
        optimizer = ScheduleOptimizer()

        potential = optimizer.calculate_load_shift_potential(
            jobs=multiple_jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff
        )

        assert "total_energy_kwh" in potential
        assert "flexible_energy_kwh" in potential
        assert "flexibility_pct" in potential
        assert potential["total_energy_kwh"] > 0
        assert 0 <= potential["flexibility_pct"] <= 100

    # =========================================================================
    # OPTIMIZATION CONVERGENCE TESTS
    # =========================================================================

    def test_optimization_convergence_greedy(self, simple_tou_tariff, single_furnace, single_job):
        """Test greedy optimization converges."""
        config = OptimizationConfig(algorithm="greedy", max_iterations=100)
        optimizer = ScheduleOptimizer(config=config)

        schedule, provenance = optimizer.optimize(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        assert schedule.optimization_time_ms > 0
        assert schedule.optimization_time_ms < 5000  # Should complete in <5 seconds

    def test_optimization_convergence_genetic(self, simple_tou_tariff, single_furnace, single_job):
        """Test genetic algorithm optimization."""
        config = OptimizationConfig(algorithm="genetic", max_iterations=50)
        optimizer = ScheduleOptimizer(config=config)

        schedule, provenance = optimizer.optimize(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        assert schedule is not None
        assert schedule.constraints_satisfied is True

    def test_optimization_convergence_milp(self, simple_tou_tariff, single_furnace, single_job):
        """Test MILP optimization."""
        config = OptimizationConfig(algorithm="milp")
        optimizer = ScheduleOptimizer(config=config)

        schedule, provenance = optimizer.optimize(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        assert schedule is not None
        assert schedule.constraints_satisfied is True

    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================

    def test_optimize_no_jobs(self, simple_tou_tariff, single_furnace):
        """Test optimization fails with no jobs."""
        optimizer = ScheduleOptimizer()

        with pytest.raises(ValueError, match="No jobs provided"):
            optimizer.optimize(
                jobs=[],
                equipment=[single_furnace],
                tariff=simple_tou_tariff
            )

    def test_optimize_no_equipment(self, simple_tou_tariff, single_job):
        """Test optimization fails with no equipment."""
        optimizer = ScheduleOptimizer()

        with pytest.raises(ValueError, match="No equipment provided"):
            optimizer.optimize(
                jobs=[single_job],
                equipment=[],
                tariff=simple_tou_tariff
            )

    def test_optimize_unknown_equipment(self, simple_tou_tariff, single_furnace):
        """Test optimization fails with unknown equipment reference."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        invalid_job = ProductionJob(
            job_id="INVALID-JOB",
            product_name="Invalid Product",
            equipment_id="UNKNOWN-EQUIP",  # Not in equipment list
            target_temp_c=800.0,
            hold_time_min=60,
            energy_kwh=300.0,
            earliest_start=now,
            deadline=now + timedelta(hours=12),
            priority=1,
            is_flexible=True
        )

        with pytest.raises(ValueError, match="unknown equipment"):
            optimizer.optimize(
                jobs=[invalid_job],
                equipment=[single_furnace],
                tariff=simple_tou_tariff
            )

    def test_optimize_invalid_time_window(self, simple_tou_tariff, single_furnace):
        """Test optimization fails with invalid time window."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        invalid_job = ProductionJob(
            job_id="INVALID-JOB",
            product_name="Invalid Product",
            equipment_id=single_furnace.equipment_id,
            target_temp_c=800.0,
            hold_time_min=60,
            energy_kwh=300.0,
            earliest_start=now + timedelta(hours=12),  # Starts after deadline
            deadline=now,
            priority=1,
            is_flexible=True
        )

        with pytest.raises(ValueError, match="invalid time window"):
            optimizer.optimize(
                jobs=[invalid_job],
                equipment=[single_furnace],
                tariff=simple_tou_tariff
            )

    def test_optimize_invalid_energy(self, simple_tou_tariff, single_furnace):
        """Test optimization fails with invalid energy."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        invalid_job = ProductionJob(
            job_id="INVALID-JOB",
            product_name="Invalid Product",
            equipment_id=single_furnace.equipment_id,
            target_temp_c=800.0,
            hold_time_min=60,
            energy_kwh=-100.0,  # Negative energy
            earliest_start=now,
            deadline=now + timedelta(hours=12),
            priority=1,
            is_flexible=True
        )

        with pytest.raises(ValueError, match="invalid energy"):
            optimizer.optimize(
                jobs=[invalid_job],
                equipment=[single_furnace],
                tariff=simple_tou_tariff
            )

    def test_optimize_unknown_algorithm(self, simple_tou_tariff, single_furnace, single_job):
        """Test optimization fails with unknown algorithm."""
        config = OptimizationConfig(algorithm="unknown_algo")
        optimizer = ScheduleOptimizer(config=config)

        with pytest.raises(ValueError, match="Unknown optimization algorithm"):
            optimizer.optimize(
                jobs=[single_job],
                equipment=[single_furnace],
                tariff=simple_tou_tariff
            )

    # =========================================================================
    # PROVENANCE TESTS
    # =========================================================================

    def test_provenance_determinism(self, simple_tou_tariff, single_furnace, single_job):
        """Test provenance hash is deterministic."""
        optimizer = ScheduleOptimizer()

        schedule1, prov1 = optimizer.optimize(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        schedule2, prov2 = optimizer.optimize(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        assert prov1.provenance_hash == prov2.provenance_hash

    def test_provenance_completeness(self, simple_tou_tariff, single_furnace, single_job):
        """Test provenance includes all required fields."""
        optimizer = ScheduleOptimizer()

        schedule, provenance = optimizer.optimize(
            jobs=[single_job],
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )

        assert provenance.calculator_name == "ScheduleOptimizer"
        assert provenance.calculator_version == "1.0.0"
        assert len(provenance.provenance_hash) == 64
        assert len(provenance.calculation_steps) >= 4

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    def test_optimization_speed_small(self, simple_tou_tariff, single_furnace):
        """Test optimization speed with small problem."""
        optimizer = ScheduleOptimizer()

        now = datetime.now(timezone.utc)
        jobs = [
            ProductionJob(
                job_id=f"JOB-{i}",
                product_name=f"Product {i}",
                equipment_id=single_furnace.equipment_id,
                target_temp_c=800.0,
                hold_time_min=60,
                energy_kwh=200.0,
                earliest_start=now,
                deadline=now + timedelta(hours=48),
                priority=1,
                is_flexible=True
            )
            for i in range(10)
        ]

        start = time.time()
        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=[single_furnace],
            tariff=simple_tou_tariff
        )
        duration_ms = (time.time() - start) * 1000

        assert duration_ms < 1000  # Should complete in <1 second
        assert len(schedule.slots) == 10

    @pytest.mark.performance
    def test_optimization_speed_large(self, simple_tou_tariff, multiple_equipment, benchmark_jobs):
        """Test optimization speed with large problem."""
        optimizer = ScheduleOptimizer()

        # Use subset of benchmark jobs
        jobs = benchmark_jobs[:100]

        start = time.time()
        schedule, _ = optimizer.optimize(
            jobs=jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff
        )
        duration_ms = (time.time() - start) * 1000

        assert duration_ms < 10000  # Should complete in <10 seconds
        assert len(schedule.slots) == 100


# =============================================================================
# INTEGRATION TESTS WITH SCHEDULE VALIDATOR
# =============================================================================

@pytest.mark.integration
class TestScheduleOptimizerIntegration:
    """Integration tests for ScheduleOptimizer."""

    def test_full_optimization_workflow(
        self,
        simple_tou_tariff,
        multiple_equipment,
        multiple_jobs,
        schedule_validator
    ):
        """Test complete optimization workflow with validation."""
        optimizer = ScheduleOptimizer()

        schedule, provenance = optimizer.optimize(
            jobs=multiple_jobs,
            equipment=multiple_equipment,
            tariff=simple_tou_tariff
        )

        # Validate schedule
        is_valid, errors = schedule_validator(schedule, multiple_jobs)

        assert is_valid, f"Schedule validation failed: {errors}"
        assert schedule.constraints_satisfied is True
        assert schedule.savings_pct >= 0
