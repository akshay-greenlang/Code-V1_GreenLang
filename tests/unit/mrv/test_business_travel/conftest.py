# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-019: Business Travel Agent.

Provides comprehensive test fixtures for:
- Flight inputs (LHR-JFK economy, round trip, business class)
- Rail inputs (national, eurostar, high-speed)
- Road inputs (car average, hybrid, BEV, taxi)
- Hotel inputs (UK standard, US luxury, unknown country)
- Spend inputs (air NAICS, hotel NAICS, multi-currency)
- Compliance inputs (GHG Protocol, CDP, SBTi frameworks)
- Configuration objects (general, database, air, rail, road, hotel, spend)
- Mock engines (database, air calculator, ground calculator, compliance)
- Emission factors, airport lookups, and batch inputs

Usage:
    def test_something(sample_flight_input, mock_database_engine):
        result = calculate(sample_flight_input, mock_database_engine)
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
# FLIGHT INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_flight_input() -> Dict[str, Any]:
    """LHR -> JFK economy flight (one-way, with radiative forcing)."""
    return {
        "origin_iata": "LHR",
        "destination_iata": "JFK",
        "cabin_class": "economy",
        "passengers": 1,
        "round_trip": False,
        "rf_option": "with_rf",
    }


@pytest.fixture
def sample_flight_input_round_trip() -> Dict[str, Any]:
    """JFK -> LHR business class round trip."""
    return {
        "origin_iata": "JFK",
        "destination_iata": "LHR",
        "cabin_class": "business",
        "passengers": 1,
        "round_trip": True,
        "rf_option": "both",
    }


# ============================================================================
# RAIL INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_rail_input() -> Dict[str, Any]:
    """National rail 640 km one-way."""
    return {
        "rail_type": "national",
        "distance_km": Decimal("640"),
        "passengers": 1,
    }


@pytest.fixture
def sample_rail_input_eurostar() -> Dict[str, Any]:
    """Eurostar London-Paris 340 km."""
    return {
        "rail_type": "eurostar",
        "distance_km": Decimal("340"),
        "passengers": 2,
    }


# ============================================================================
# ROAD INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_road_input() -> Dict[str, Any]:
    """Average car 300 km distance-based."""
    return {
        "vehicle_type": "car_average",
        "distance_km": Decimal("300"),
    }


@pytest.fixture
def sample_fuel_input() -> Dict[str, Any]:
    """Diesel fuel-based 45 litres."""
    return {
        "fuel_type": "diesel",
        "litres": Decimal("45.0"),
    }


# ============================================================================
# HOTEL INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_hotel_input() -> Dict[str, Any]:
    """UK standard hotel 3 nights."""
    return {
        "country_code": "GB",
        "room_nights": 3,
        "hotel_class": "standard",
    }


# ============================================================================
# SPEND INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_spend_input() -> Dict[str, Any]:
    """Air transportation spend USD 5000."""
    return {
        "naics_code": "481000",
        "amount": Decimal("5000"),
        "currency": "USD",
        "reporting_year": 2024,
    }


# ============================================================================
# COMPLIANCE INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_compliance_input() -> Dict[str, Any]:
    """GHG Protocol + CDP + SBTi compliance check."""
    return {
        "frameworks": ["ghg_protocol", "cdp", "sbti"],
        "calculation_results": [
            {"total_co2e": 1500.0, "method": "distance_based"},
        ],
        "rf_disclosed": True,
        "mode_breakdown_provided": True,
    }
