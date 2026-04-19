"""
GL-019 HEATSCHEDULER - Test Configuration and Fixtures

Provides shared fixtures, test data generators, and test utilities
for comprehensive testing of ProcessHeatingScheduler calculators and agent.

Author: GL-TestEngineer
Version: 1.0.0
"""

import sys
import os
import json
import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, time, timedelta, timezone
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import logging
import random

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# DATA CLASSES FOR TEST INPUTS
# =============================================================================

class RateType(str, Enum):
    """Energy rate types."""
    TIME_OF_USE = "time_of_use"
    DEMAND = "demand"
    REAL_TIME = "real_time"
    FLAT = "flat"


class PeriodType(str, Enum):
    """Time-of-Use period types."""
    ON_PEAK = "on_peak"
    MID_PEAK = "mid_peak"
    OFF_PEAK = "off_peak"
    SUPER_OFF_PEAK = "super_off_peak"
    CRITICAL_PEAK = "critical_peak"


@dataclass
class TimePeriod:
    """Time period definition for ToU rates."""
    start_hour: int  # 0-23
    end_hour: int  # 0-23
    period_type: PeriodType
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri


@dataclass
class EnergyTariff:
    """Energy tariff structure."""
    tariff_id: str
    utility_name: str
    rate_type: RateType
    base_rate_kwh: float  # $/kWh base rate
    demand_charge_kw: float = 0.0  # $/kW demand charge
    time_periods: List[TimePeriod] = field(default_factory=list)
    period_rates: Dict[str, float] = field(default_factory=dict)  # Period type -> rate multiplier
    real_time_prices: Dict[str, float] = field(default_factory=dict)  # ISO timestamp -> price
    effective_date: date = field(default_factory=date.today)
    expiration_date: Optional[date] = None


@dataclass
class HeatingEquipment:
    """Heating equipment configuration."""
    equipment_id: str
    name: str
    equipment_type: str  # furnace, oven, boiler, heat_treat
    power_kw: float
    min_power_kw: float = 0.0
    max_temp_c: float = 1200.0
    min_temp_c: float = 20.0
    ramp_rate_c_per_min: float = 10.0
    thermal_mass_kwh_per_c: float = 1.0
    efficiency: float = 0.85
    is_interruptible: bool = False
    max_interruption_min: int = 0
    priority: int = 1  # 1 = highest priority


@dataclass
class ProductionJob:
    """Production job requiring heating."""
    job_id: str
    product_name: str
    equipment_id: str
    target_temp_c: float
    hold_time_min: int
    energy_kwh: float
    earliest_start: datetime
    deadline: datetime
    priority: int = 1
    is_flexible: bool = True
    min_continuous_run_min: int = 0


@dataclass
class ScheduleSlot:
    """Scheduled time slot for equipment operation."""
    slot_id: str
    equipment_id: str
    job_id: str
    start_time: datetime
    end_time: datetime
    power_kw: float
    energy_kwh: float
    estimated_cost: float
    period_type: PeriodType


@dataclass
class OptimizedSchedule:
    """Complete optimized schedule."""
    schedule_id: str
    created_at: datetime
    slots: List[ScheduleSlot]
    total_energy_kwh: float
    total_cost: float
    baseline_cost: float
    savings: float
    savings_pct: float
    optimization_time_ms: float
    constraints_satisfied: bool
    provenance_hash: str


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# =============================================================================
# ENERGY TARIFF FIXTURES
# =============================================================================

@pytest.fixture
def simple_tou_tariff():
    """Simple Time-of-Use tariff with peak/off-peak."""
    return EnergyTariff(
        tariff_id="TOU-001",
        utility_name="Test Utility Co",
        rate_type=RateType.TIME_OF_USE,
        base_rate_kwh=0.10,
        demand_charge_kw=15.00,
        time_periods=[
            TimePeriod(start_hour=14, end_hour=19, period_type=PeriodType.ON_PEAK),
            TimePeriod(start_hour=10, end_hour=14, period_type=PeriodType.MID_PEAK),
            TimePeriod(start_hour=19, end_hour=22, period_type=PeriodType.MID_PEAK),
            TimePeriod(start_hour=22, end_hour=10, period_type=PeriodType.OFF_PEAK),
        ],
        period_rates={
            PeriodType.ON_PEAK.value: 2.5,  # 2.5x base rate = $0.25/kWh
            PeriodType.MID_PEAK.value: 1.5,  # 1.5x base rate = $0.15/kWh
            PeriodType.OFF_PEAK.value: 0.6,  # 0.6x base rate = $0.06/kWh
        }
    )


@pytest.fixture
def complex_tou_tariff():
    """Complex Time-of-Use tariff with seasonal and weekend variations."""
    return EnergyTariff(
        tariff_id="TOU-002",
        utility_name="Complex Utility Inc",
        rate_type=RateType.TIME_OF_USE,
        base_rate_kwh=0.12,
        demand_charge_kw=18.50,
        time_periods=[
            # Weekday periods
            TimePeriod(start_hour=12, end_hour=18, period_type=PeriodType.ON_PEAK, days_of_week=[0, 1, 2, 3, 4]),
            TimePeriod(start_hour=8, end_hour=12, period_type=PeriodType.MID_PEAK, days_of_week=[0, 1, 2, 3, 4]),
            TimePeriod(start_hour=18, end_hour=21, period_type=PeriodType.MID_PEAK, days_of_week=[0, 1, 2, 3, 4]),
            TimePeriod(start_hour=21, end_hour=8, period_type=PeriodType.OFF_PEAK, days_of_week=[0, 1, 2, 3, 4]),
            # Weekend - all off-peak
            TimePeriod(start_hour=0, end_hour=24, period_type=PeriodType.OFF_PEAK, days_of_week=[5, 6]),
        ],
        period_rates={
            PeriodType.ON_PEAK.value: 3.0,  # $0.36/kWh
            PeriodType.MID_PEAK.value: 1.8,  # $0.216/kWh
            PeriodType.OFF_PEAK.value: 0.5,  # $0.06/kWh
            PeriodType.SUPER_OFF_PEAK.value: 0.3,  # $0.036/kWh
            PeriodType.CRITICAL_PEAK.value: 5.0,  # $0.60/kWh
        }
    )


@pytest.fixture
def demand_only_tariff():
    """Demand-only tariff without time-of-use."""
    return EnergyTariff(
        tariff_id="DEM-001",
        utility_name="Demand Power Co",
        rate_type=RateType.DEMAND,
        base_rate_kwh=0.08,
        demand_charge_kw=25.00,
        time_periods=[],
        period_rates={}
    )


@pytest.fixture
def real_time_tariff():
    """Real-time pricing tariff with hourly prices."""
    base_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Generate 24 hours of real-time prices with typical daily pattern
    hourly_prices = {}
    base_prices = [
        0.04, 0.035, 0.03, 0.03, 0.035, 0.05,  # 00:00-05:00 (low)
        0.08, 0.12, 0.15, 0.18, 0.20, 0.22,    # 06:00-11:00 (rising)
        0.25, 0.28, 0.32, 0.35, 0.38, 0.35,    # 12:00-17:00 (peak)
        0.28, 0.22, 0.15, 0.10, 0.06, 0.045    # 18:00-23:00 (falling)
    ]

    for hour, price in enumerate(base_prices):
        timestamp = (base_date + timedelta(hours=hour)).isoformat()
        hourly_prices[timestamp] = price

    return EnergyTariff(
        tariff_id="RTP-001",
        utility_name="Real-Time Energy Market",
        rate_type=RateType.REAL_TIME,
        base_rate_kwh=0.15,  # Fallback rate
        demand_charge_kw=20.00,
        real_time_prices=hourly_prices
    )


@pytest.fixture
def flat_rate_tariff():
    """Simple flat rate tariff."""
    return EnergyTariff(
        tariff_id="FLAT-001",
        utility_name="Simple Power Co",
        rate_type=RateType.FLAT,
        base_rate_kwh=0.12,
        demand_charge_kw=10.00
    )


# =============================================================================
# EQUIPMENT FIXTURES
# =============================================================================

@pytest.fixture
def single_furnace():
    """Single industrial furnace."""
    return HeatingEquipment(
        equipment_id="FURN-001",
        name="Heat Treat Furnace #1",
        equipment_type="furnace",
        power_kw=500.0,
        min_power_kw=100.0,
        max_temp_c=1200.0,
        min_temp_c=20.0,
        ramp_rate_c_per_min=15.0,
        thermal_mass_kwh_per_c=2.5,
        efficiency=0.88,
        is_interruptible=False,
        priority=1
    )


@pytest.fixture
def multiple_equipment():
    """Multiple heating equipment for load balancing tests."""
    return [
        HeatingEquipment(
            equipment_id="FURN-001",
            name="Heat Treat Furnace #1",
            equipment_type="furnace",
            power_kw=500.0,
            min_power_kw=100.0,
            max_temp_c=1200.0,
            ramp_rate_c_per_min=15.0,
            thermal_mass_kwh_per_c=2.5,
            efficiency=0.88,
            is_interruptible=False,
            priority=1
        ),
        HeatingEquipment(
            equipment_id="FURN-002",
            name="Heat Treat Furnace #2",
            equipment_type="furnace",
            power_kw=750.0,
            min_power_kw=150.0,
            max_temp_c=1000.0,
            ramp_rate_c_per_min=12.0,
            thermal_mass_kwh_per_c=3.0,
            efficiency=0.85,
            is_interruptible=True,
            max_interruption_min=15,
            priority=2
        ),
        HeatingEquipment(
            equipment_id="OVEN-001",
            name="Curing Oven #1",
            equipment_type="oven",
            power_kw=200.0,
            min_power_kw=50.0,
            max_temp_c=350.0,
            ramp_rate_c_per_min=5.0,
            thermal_mass_kwh_per_c=1.0,
            efficiency=0.92,
            is_interruptible=True,
            max_interruption_min=30,
            priority=3
        ),
        HeatingEquipment(
            equipment_id="BOIL-001",
            name="Steam Boiler #1",
            equipment_type="boiler",
            power_kw=1000.0,
            min_power_kw=300.0,
            max_temp_c=180.0,
            ramp_rate_c_per_min=2.0,
            thermal_mass_kwh_per_c=10.0,
            efficiency=0.82,
            is_interruptible=False,
            priority=1
        ),
    ]


@pytest.fixture
def interruptible_equipment():
    """Equipment that can be interrupted for demand response."""
    return HeatingEquipment(
        equipment_id="OVEN-002",
        name="Flexible Curing Oven",
        equipment_type="oven",
        power_kw=300.0,
        min_power_kw=75.0,
        max_temp_c=400.0,
        ramp_rate_c_per_min=8.0,
        thermal_mass_kwh_per_c=1.5,
        efficiency=0.90,
        is_interruptible=True,
        max_interruption_min=60,
        priority=4
    )


# =============================================================================
# PRODUCTION SCHEDULE FIXTURES
# =============================================================================

@pytest.fixture
def single_job():
    """Single production job."""
    now = datetime.now(timezone.utc)
    return ProductionJob(
        job_id="JOB-001",
        product_name="Steel Part A",
        equipment_id="FURN-001",
        target_temp_c=850.0,
        hold_time_min=120,
        energy_kwh=1000.0,
        earliest_start=now,
        deadline=now + timedelta(hours=24),
        priority=1,
        is_flexible=True
    )


@pytest.fixture
def multiple_jobs():
    """Multiple production jobs for scheduling."""
    now = datetime.now(timezone.utc)
    return [
        ProductionJob(
            job_id="JOB-001",
            product_name="Steel Part A",
            equipment_id="FURN-001",
            target_temp_c=850.0,
            hold_time_min=120,
            energy_kwh=1000.0,
            earliest_start=now,
            deadline=now + timedelta(hours=24),
            priority=1,
            is_flexible=True
        ),
        ProductionJob(
            job_id="JOB-002",
            product_name="Aluminum Part B",
            equipment_id="FURN-002",
            target_temp_c=550.0,
            hold_time_min=90,
            energy_kwh=750.0,
            earliest_start=now + timedelta(hours=2),
            deadline=now + timedelta(hours=18),
            priority=2,
            is_flexible=True
        ),
        ProductionJob(
            job_id="JOB-003",
            product_name="Composite Part C",
            equipment_id="OVEN-001",
            target_temp_c=180.0,
            hold_time_min=240,
            energy_kwh=400.0,
            earliest_start=now,
            deadline=now + timedelta(hours=36),
            priority=3,
            is_flexible=True
        ),
        ProductionJob(
            job_id="JOB-004",
            product_name="Urgent Steel Part D",
            equipment_id="FURN-001",
            target_temp_c=900.0,
            hold_time_min=60,
            energy_kwh=500.0,
            earliest_start=now,
            deadline=now + timedelta(hours=8),
            priority=1,
            is_flexible=False,  # Must run as scheduled
            min_continuous_run_min=60
        ),
    ]


@pytest.fixture
def tight_deadline_jobs():
    """Jobs with tight deadlines for constraint testing."""
    now = datetime.now(timezone.utc)
    return [
        ProductionJob(
            job_id="TIGHT-001",
            product_name="Rush Order A",
            equipment_id="FURN-001",
            target_temp_c=800.0,
            hold_time_min=60,
            energy_kwh=500.0,
            earliest_start=now,
            deadline=now + timedelta(hours=3),  # Very tight deadline
            priority=1,
            is_flexible=False
        ),
        ProductionJob(
            job_id="TIGHT-002",
            product_name="Rush Order B",
            equipment_id="FURN-001",
            target_temp_c=750.0,
            hold_time_min=45,
            energy_kwh=400.0,
            earliest_start=now + timedelta(hours=1),
            deadline=now + timedelta(hours=4),
            priority=1,
            is_flexible=False
        ),
    ]


# =============================================================================
# EDGE CASE FIXTURES
# =============================================================================

@pytest.fixture
def midnight_crossing_schedule():
    """Schedule that crosses midnight for DST testing."""
    now = datetime.now(timezone.utc).replace(hour=22, minute=0, second=0, microsecond=0)
    return ProductionJob(
        job_id="MIDNIGHT-001",
        product_name="Overnight Heat Treat",
        equipment_id="FURN-001",
        target_temp_c=900.0,
        hold_time_min=360,  # 6 hours crossing midnight
        energy_kwh=3000.0,
        earliest_start=now,
        deadline=now + timedelta(hours=10),
        priority=1,
        is_flexible=True
    )


@pytest.fixture
def dst_transition_dates():
    """Dates around DST transitions for edge case testing."""
    return {
        "spring_forward_2025": datetime(2025, 3, 9, 1, 30, tzinfo=timezone.utc),
        "fall_back_2025": datetime(2025, 11, 2, 1, 30, tzinfo=timezone.utc),
    }


@pytest.fixture
def demand_response_event():
    """Demand response event for load curtailment testing."""
    now = datetime.now(timezone.utc)
    return {
        "event_id": "DR-001",
        "event_type": "critical_peak",
        "start_time": now + timedelta(hours=2),
        "end_time": now + timedelta(hours=4),
        "target_reduction_kw": 500.0,
        "incentive_per_kwh": 0.50,
        "penalty_per_kwh": 1.00,
        "notification_time": now,
        "is_mandatory": False
    }


# =============================================================================
# PARAMETERIZED TEST DATA
# =============================================================================

@pytest.fixture
def tou_rate_test_cases():
    """Test cases for ToU rate calculations with known values."""
    return [
        # (hour, day_of_week, expected_period, expected_rate_multiplier)
        (14, 0, PeriodType.ON_PEAK, 2.5),  # Monday 2pm - peak
        (15, 2, PeriodType.ON_PEAK, 2.5),  # Wednesday 3pm - peak
        (10, 1, PeriodType.MID_PEAK, 1.5),  # Tuesday 10am - mid-peak
        (20, 3, PeriodType.MID_PEAK, 1.5),  # Thursday 8pm - mid-peak
        (22, 4, PeriodType.OFF_PEAK, 0.6),  # Friday 10pm - off-peak
        (3, 0, PeriodType.OFF_PEAK, 0.6),   # Monday 3am - off-peak
        (12, 5, PeriodType.OFF_PEAK, 0.6),  # Saturday noon - weekend off-peak
        (18, 6, PeriodType.OFF_PEAK, 0.6),  # Sunday 6pm - weekend off-peak
    ]


@pytest.fixture
def demand_charge_test_cases():
    """Test cases for demand charge calculations."""
    return [
        # (peak_demand_kw, demand_rate, expected_charge)
        (100.0, 15.00, 1500.00),
        (500.0, 15.00, 7500.00),
        (1000.0, 20.00, 20000.00),
        (250.0, 18.50, 4625.00),
        (0.0, 15.00, 0.00),  # Edge case: no demand
    ]


@pytest.fixture
def energy_cost_test_cases():
    """Test cases for energy cost calculations with known values."""
    return [
        # (energy_kwh, rate_per_kwh, expected_cost)
        (100.0, 0.10, 10.00),
        (500.0, 0.25, 125.00),
        (1000.0, 0.15, 150.00),
        (2500.0, 0.08, 200.00),
        (0.0, 0.10, 0.00),  # Edge case: no energy
    ]


@pytest.fixture
def optimization_test_cases():
    """Test cases for schedule optimization."""
    return [
        {
            "name": "single_job_flexible",
            "jobs": 1,
            "equipment": 1,
            "expected_savings_min_pct": 10.0,
            "expected_constraints_satisfied": True
        },
        {
            "name": "multiple_jobs_single_equipment",
            "jobs": 3,
            "equipment": 1,
            "expected_savings_min_pct": 15.0,
            "expected_constraints_satisfied": True
        },
        {
            "name": "multiple_jobs_multiple_equipment",
            "jobs": 4,
            "equipment": 4,
            "expected_savings_min_pct": 20.0,
            "expected_constraints_satisfied": True
        },
    ]


@pytest.fixture
def savings_test_cases():
    """Test cases for savings calculations."""
    return [
        # (baseline_cost, optimized_cost, expected_savings, expected_pct)
        (1000.00, 850.00, 150.00, 15.0),
        (5000.00, 4000.00, 1000.00, 20.0),
        (10000.00, 7500.00, 2500.00, 25.0),
        (2500.00, 2500.00, 0.00, 0.0),  # No savings
        (1000.00, 1100.00, -100.00, -10.0),  # Negative savings (edge case)
    ]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

@pytest.fixture
def provenance_validator():
    """Helper function to validate provenance records."""
    def validate(provenance):
        """Validate provenance record structure and hashes."""
        assert provenance is not None
        assert hasattr(provenance, 'calculator_name')
        assert hasattr(provenance, 'calculator_version')
        assert hasattr(provenance, 'provenance_hash')
        assert len(provenance.provenance_hash) == 64  # SHA-256
        assert len(provenance.calculation_steps) > 0

        # Validate each step
        for step in provenance.calculation_steps:
            assert step.step_number > 0
            assert step.description != ""
            assert step.operation != ""

        return True

    return validate


@pytest.fixture
def tolerance_checker():
    """Helper function for floating point comparisons."""
    def check(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
        """Check if actual is within tolerance of expected."""
        import math
        return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)

    return check


@pytest.fixture
def schedule_validator():
    """Helper function to validate optimized schedules."""
    def validate(schedule: OptimizedSchedule, jobs: List[ProductionJob]):
        """Validate schedule meets all constraints."""
        errors = []

        # Check all jobs are scheduled
        scheduled_job_ids = {slot.job_id for slot in schedule.slots}
        required_job_ids = {job.job_id for job in jobs}
        missing_jobs = required_job_ids - scheduled_job_ids
        if missing_jobs:
            errors.append(f"Missing jobs: {missing_jobs}")

        # Check deadline constraints
        for slot in schedule.slots:
            matching_jobs = [j for j in jobs if j.job_id == slot.job_id]
            if matching_jobs:
                job = matching_jobs[0]
                if slot.end_time > job.deadline:
                    errors.append(f"Job {job.job_id} misses deadline by {slot.end_time - job.deadline}")
                if slot.start_time < job.earliest_start:
                    errors.append(f"Job {job.job_id} starts before earliest_start")

        # Check no overlapping slots for same equipment
        equipment_slots = {}
        for slot in schedule.slots:
            if slot.equipment_id not in equipment_slots:
                equipment_slots[slot.equipment_id] = []
            equipment_slots[slot.equipment_id].append(slot)

        for eq_id, slots in equipment_slots.items():
            sorted_slots = sorted(slots, key=lambda s: s.start_time)
            for i in range(len(sorted_slots) - 1):
                if sorted_slots[i].end_time > sorted_slots[i + 1].start_time:
                    errors.append(f"Overlapping slots on {eq_id}")

        return len(errors) == 0, errors

    return validate


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

@pytest.fixture
def production_schedule_generator():
    """Generate mock production schedules for testing."""
    def generate(num_jobs: int = 10, equipment_ids: List[str] = None, seed: int = 42):
        """Generate random production jobs."""
        random.seed(seed)
        if equipment_ids is None:
            equipment_ids = ["FURN-001", "FURN-002", "OVEN-001"]

        now = datetime.now(timezone.utc)
        jobs = []

        for i in range(num_jobs):
            duration_min = random.randint(30, 480)  # 30 min to 8 hours
            energy = random.uniform(100, 2000)
            earliest = now + timedelta(hours=random.randint(0, 24))
            deadline = earliest + timedelta(hours=random.randint(4, 48))

            job = ProductionJob(
                job_id=f"GEN-{i:04d}",
                product_name=f"Product {i}",
                equipment_id=random.choice(equipment_ids),
                target_temp_c=random.uniform(100, 1000),
                hold_time_min=duration_min,
                energy_kwh=energy,
                earliest_start=earliest,
                deadline=deadline,
                priority=random.randint(1, 5),
                is_flexible=random.choice([True, True, True, False])  # 75% flexible
            )
            jobs.append(job)

        return jobs

    return generate


@pytest.fixture
def real_time_price_generator():
    """Generate mock real-time prices for testing."""
    def generate(hours: int = 24, volatility: float = 0.3, seed: int = 42):
        """Generate hourly real-time prices with realistic patterns."""
        random.seed(seed)
        base_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        # Base pattern (typical daily curve)
        base_pattern = [
            0.04, 0.035, 0.03, 0.03, 0.035, 0.05,
            0.08, 0.12, 0.15, 0.18, 0.20, 0.22,
            0.25, 0.28, 0.32, 0.35, 0.38, 0.35,
            0.28, 0.22, 0.15, 0.10, 0.06, 0.045
        ]

        prices = {}
        for hour in range(hours):
            pattern_hour = hour % 24
            base_price = base_pattern[pattern_hour]
            # Add random volatility
            noise = random.gauss(0, volatility * base_price)
            price = max(0.01, base_price + noise)  # Ensure positive price
            timestamp = (base_date + timedelta(hours=hour)).isoformat()
            prices[timestamp] = round(price, 4)

        return prices

    return generate


# =============================================================================
# TEST DATA FILES
# =============================================================================

@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_tariff_data(test_data_dir):
    """Load sample tariff test data."""
    tariff_file = test_data_dir / "sample_tariffs.json"

    if tariff_file.exists():
        return json.loads(tariff_file.read_text())

    # Return default data if file doesn't exist
    return {
        "tariffs": [],
        "test_scenarios": []
    }


@pytest.fixture
def sample_schedule_data(test_data_dir):
    """Load sample production schedule test data."""
    schedule_file = test_data_dir / "sample_production_schedule.json"

    if schedule_file.exists():
        return json.loads(schedule_file.read_text())

    # Return default data if file doesn't exist
    return {
        "jobs": [],
        "equipment": [],
        "constraints": {}
    }


# =============================================================================
# BENCHMARK FIXTURES
# =============================================================================

@pytest.fixture
def benchmark_jobs():
    """Large dataset for performance benchmarking."""
    now = datetime.now(timezone.utc)
    return [
        ProductionJob(
            job_id=f"BENCH-{i:05d}",
            product_name=f"Benchmark Product {i}",
            equipment_id=f"EQUIP-{i % 10:03d}",
            target_temp_c=200.0 + (i % 500),
            hold_time_min=30 + (i % 120),
            energy_kwh=100.0 + (i % 1000),
            earliest_start=now + timedelta(hours=i % 48),
            deadline=now + timedelta(hours=(i % 48) + 24),
            priority=(i % 5) + 1,
            is_flexible=i % 4 != 0
        )
        for i in range(1000)
    ]


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "critical: mark test as critical (must pass)"
    )
    config.addinivalue_line(
        "markers", "calculator: mark test as calculator test (95%+ coverage)"
    )
    config.addinivalue_line(
        "markers", "optimizer: mark test as optimizer test"
    )
    config.addinivalue_line(
        "markers", "scheduling: mark test as scheduling test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add calculator marker to calculator tests
        if "calculator" in item.nodeid:
            item.add_marker(pytest.mark.calculator)

        # Add optimizer marker to optimizer tests
        if "optimizer" in item.nodeid:
            item.add_marker(pytest.mark.optimizer)

        # Add critical marker to validation tests
        if "validation" in item.nodeid or "provenance" in item.nodeid:
            item.add_marker(pytest.mark.critical)
