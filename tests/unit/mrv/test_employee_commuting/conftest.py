# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-020: Employee Commuting Agent.

Provides comprehensive test fixtures for:
- Personal vehicle inputs (SOV, carpool, motorcycle, EV)
- Public transit inputs (bus, metro, rail, ferry)
- Active transport inputs (cycling, walking, e-bike, e-scooter)
- Telework inputs (full remote, hybrid patterns)
- Survey inputs (responses, modal split, extrapolation)
- Compliance inputs (7 frameworks)
- Configuration objects (15 frozen dataclass configs)
- Mock engines (database, calculators, compliance, pipeline)
- Emission factors, working days, and batch inputs

Usage:
    def test_something(sample_sov_input, mock_database_engine):
        result = calculate(sample_sov_input, mock_database_engine)
        assert result.total_co2e > 0

Author: GL-TestEngineer
Date: February 2026
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock
import pytest


# ============================================================================
# PERSONAL VEHICLE INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_sov_input() -> Dict[str, Any]:
    """Medium petrol car SOV, 15km one-way, full-time, US."""
    return {
        "mode": "sov",
        "vehicle_type": "car_medium_petrol",
        "one_way_distance_km": Decimal("15.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
    }


@pytest.fixture
def sample_sov_diesel_input() -> Dict[str, Any]:
    """Large diesel car SOV, 25km one-way, US."""
    return {
        "mode": "sov",
        "vehicle_type": "car_large_diesel",
        "one_way_distance_km": Decimal("25.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
    }


@pytest.fixture
def sample_ev_input() -> Dict[str, Any]:
    """Battery electric vehicle, 20km one-way, US."""
    return {
        "mode": "sov",
        "vehicle_type": "bev",
        "one_way_distance_km": Decimal("20.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
    }


@pytest.fixture
def sample_phev_input() -> Dict[str, Any]:
    """Plug-in hybrid, 18km one-way, GB."""
    return {
        "mode": "sov",
        "vehicle_type": "plugin_hybrid",
        "one_way_distance_km": Decimal("18.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "GB",
    }


@pytest.fixture
def sample_motorcycle_input() -> Dict[str, Any]:
    """Motorcycle, 12km one-way, global."""
    return {
        "mode": "motorcycle",
        "vehicle_type": "motorcycle",
        "one_way_distance_km": Decimal("12.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "GLOBAL",
    }


# ============================================================================
# CARPOOL INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_carpool_input() -> Dict[str, Any]:
    """Average car carpool, 3 occupants, 20km one-way."""
    return {
        "vehicle_type": "car_average",
        "one_way_distance_km": Decimal("20.0"),
        "occupants": 3,
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
    }


# ============================================================================
# PUBLIC TRANSIT INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_bus_input() -> Dict[str, Any]:
    """Local bus, 10km one-way."""
    return {
        "transit_type": "bus_local",
        "one_way_distance_km": Decimal("10.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
    }


@pytest.fixture
def sample_metro_input() -> Dict[str, Any]:
    """Metro / subway, 8.5km one-way."""
    return {
        "transit_type": "metro",
        "one_way_distance_km": Decimal("8.5"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
    }


@pytest.fixture
def sample_commuter_rail_input() -> Dict[str, Any]:
    """Commuter rail, 35km one-way."""
    return {
        "transit_type": "commuter_rail",
        "one_way_distance_km": Decimal("35.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "GB",
    }


@pytest.fixture
def sample_ferry_input() -> Dict[str, Any]:
    """Ferry, 15km one-way."""
    return {
        "transit_type": "ferry",
        "one_way_distance_km": Decimal("15.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "GLOBAL",
    }


# ============================================================================
# ACTIVE TRANSPORT INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_cycling_input() -> Dict[str, Any]:
    """Bicycle, 6km one-way."""
    return {
        "mode": "cycling",
        "one_way_distance_km": Decimal("6.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
    }


@pytest.fixture
def sample_walking_input() -> Dict[str, Any]:
    """Walking, 2km one-way."""
    return {
        "mode": "walking",
        "one_way_distance_km": Decimal("2.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
    }


@pytest.fixture
def sample_ebike_input() -> Dict[str, Any]:
    """E-bike, 10km one-way, US grid."""
    return {
        "mode": "e_bike",
        "one_way_distance_km": Decimal("10.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
    }


@pytest.fixture
def sample_escooter_input() -> Dict[str, Any]:
    """E-scooter, 5km one-way."""
    return {
        "mode": "e_scooter",
        "one_way_distance_km": Decimal("5.0"),
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
    }


# ============================================================================
# TELEWORK INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_telework_full_remote() -> Dict[str, Any]:
    """Full remote worker, US, full seasonal adjustment."""
    return {
        "frequency": "full_remote",
        "region": "US",
        "seasonal_adjustment": "full_seasonal",
        "work_schedule": "full_time",
    }


@pytest.fixture
def sample_telework_hybrid_3() -> Dict[str, Any]:
    """Hybrid 3-day remote, GB."""
    return {
        "frequency": "hybrid_3",
        "region": "GB",
        "seasonal_adjustment": "none",
        "work_schedule": "full_time",
    }


# ============================================================================
# FUEL-BASED INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_fuel_input() -> Dict[str, Any]:
    """Petrol fuel-based, 12.5 litres/week, 48 weeks."""
    return {
        "fuel_type": "petrol",
        "litres_per_week": Decimal("12.5"),
        "commute_weeks_per_year": 48,
    }


# ============================================================================
# SPEND-BASED INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_spend_input() -> Dict[str, Any]:
    """Ground passenger transport EEIO, $50,000 USD, 2024."""
    return {
        "naics_code": "485000",
        "amount": Decimal("50000.00"),
        "currency": "USD",
        "reporting_year": 2024,
    }


# ============================================================================
# SURVEY INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_survey_response() -> Dict[str, Any]:
    """Single employee survey response."""
    return {
        "employee_id": "EMP-001",
        "mode": "sov",
        "vehicle_type": "car_medium_petrol",
        "one_way_distance_km": Decimal("18.5"),
        "commute_days_per_week": 5,
        "telework_frequency": "office_full",
        "department": "Engineering",
        "site": "HQ-NYC",
    }


@pytest.fixture
def sample_survey_input() -> Dict[str, Any]:
    """Full employee survey with 3 responses."""
    return {
        "survey_method": "random_sample",
        "total_employees": 500,
        "reporting_period": "2025",
        "region": "US",
    }


# ============================================================================
# COMPLIANCE INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_compliance_input() -> Dict[str, Any]:
    """Compliance check for GHG Protocol, CSRD, CDP."""
    return {
        "frameworks": ["ghg_protocol", "csrd_esrs", "cdp"],
        "total_co2e": Decimal("125000.00"),
        "method_used": "employee_specific",
        "reporting_period": "2025",
    }


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def default_config():
    """Default EmployeeCommutingConfig with all 15 sections."""
    from greenlang.employee_commuting.config import EmployeeCommutingConfig
    return EmployeeCommutingConfig()


@pytest.fixture
def default_general_config():
    """Default GeneralConfig."""
    from greenlang.employee_commuting.config import GeneralConfig
    return GeneralConfig()


@pytest.fixture
def default_database_config():
    """Default DatabaseConfig."""
    from greenlang.employee_commuting.config import DatabaseConfig
    return DatabaseConfig()


@pytest.fixture
def default_telework_config():
    """Default TeleworkConfig."""
    from greenlang.employee_commuting.config import TeleworkConfig
    return TeleworkConfig()


@pytest.fixture
def default_commute_mode_config():
    """Default CommuteModeConfig."""
    from greenlang.employee_commuting.config import CommuteModeConfig
    return CommuteModeConfig()


@pytest.fixture
def default_survey_config():
    """Default SurveyConfig."""
    from greenlang.employee_commuting.config import SurveyConfig
    return SurveyConfig()
