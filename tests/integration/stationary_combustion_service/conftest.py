# -*- coding: utf-8 -*-
"""
Integration test fixtures for Stationary Combustion Agent - AGENT-MRV-001

Provides shared fixtures for integration testing:
- Fully configured StationaryCombustionService
- 12-month natural gas input records
- Multi-fuel input sets (natural gas, diesel, coal, biomass)
- Facility-tagged inputs for aggregation testing
- Known emission reference values for validation

Author: GreenLang Test Engineering
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pytest

from greenlang.stationary_combustion.config import (
    StationaryCombustionConfig,
    reset_config,
)
from greenlang.stationary_combustion.setup import StationaryCombustionService


# =====================================================================
# Service fixture
# =====================================================================


@pytest.fixture
def config():
    """Create integration test StationaryCombustionConfig."""
    return StationaryCombustionConfig(
        enable_biogenic_tracking=True,
        monte_carlo_iterations=500,
        enable_metrics=False,
        default_gwp_source="AR6",
        default_tier=1,
    )


@pytest.fixture
def service(config):
    """Fully configured StationaryCombustionService for integration tests."""
    svc = StationaryCombustionService(config=config)
    svc.startup()
    yield svc
    svc.shutdown()
    reset_config()


# =====================================================================
# Natural gas input fixtures (12 monthly records)
# =====================================================================


@pytest.fixture
def natural_gas_inputs() -> List[Dict[str, Any]]:
    """12 monthly natural gas consumption records for a single facility.

    Simulates a natural gas boiler consuming ~1000 m3 per month
    with seasonal variation.
    """
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    monthly_consumption = [
        1200.0, 1100.0, 1000.0, 800.0, 600.0, 400.0,
        300.0, 350.0, 500.0, 800.0, 1000.0, 1300.0,
    ]

    inputs = []
    for month_idx, qty in enumerate(monthly_consumption):
        period_start = base.replace(month=month_idx + 1)
        if month_idx + 1 < 12:
            period_end = base.replace(month=month_idx + 2)
        else:
            period_end = datetime(2026, 1, 1, tzinfo=timezone.utc)

        inputs.append({
            "fuel_type": "NATURAL_GAS",
            "quantity": qty,
            "unit": "CUBIC_METERS",
            "facility_id": "FAC-INT-001",
            "equipment_id": "EQ-BOILER-001",
            "heating_value_basis": "HHV",
            "ef_source": "EPA",
        })

    return inputs


# =====================================================================
# Multi-fuel input fixtures
# =====================================================================


@pytest.fixture
def multi_fuel_inputs() -> List[Dict[str, Any]]:
    """Mix of natural gas, diesel, coal, and biomass inputs.

    Represents a mixed-fuel facility with different source types.
    """
    return [
        {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "unit": "CUBIC_METERS",
            "facility_id": "FAC-MULTI-001",
            "heating_value_basis": "HHV",
        },
        {
            "fuel_type": "DIESEL",
            "quantity": 500.0,
            "unit": "LITERS",
            "facility_id": "FAC-MULTI-001",
            "heating_value_basis": "HHV",
        },
        {
            "fuel_type": "COAL_BITUMINOUS",
            "quantity": 1.0,
            "unit": "TONNES",
            "facility_id": "FAC-MULTI-001",
            "heating_value_basis": "HHV",
        },
        {
            "fuel_type": "WOOD",
            "quantity": 2.0,
            "unit": "TONNES",
            "facility_id": "FAC-MULTI-001",
            "heating_value_basis": "HHV",
        },
        {
            "fuel_type": "LPG",
            "quantity": 200.0,
            "unit": "LITERS",
            "facility_id": "FAC-MULTI-001",
            "heating_value_basis": "HHV",
        },
    ]


# =====================================================================
# Facility input fixtures (multiple facilities)
# =====================================================================


@pytest.fixture
def facility_inputs() -> List[Dict[str, Any]]:
    """Inputs tagged with facility_id for aggregation testing.

    Three facilities with different fuel mixes:
    - FAC-A: Natural gas boilers
    - FAC-B: Diesel generators
    - FAC-C: Coal and biomass
    """
    return [
        # Facility A - Natural gas
        {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "unit": "CUBIC_METERS",
            "facility_id": "FAC-A",
            "equipment_id": "EQ-A-BOILER-1",
        },
        {
            "fuel_type": "NATURAL_GAS",
            "quantity": 800.0,
            "unit": "CUBIC_METERS",
            "facility_id": "FAC-A",
            "equipment_id": "EQ-A-BOILER-2",
        },
        # Facility B - Diesel
        {
            "fuel_type": "DIESEL",
            "quantity": 500.0,
            "unit": "LITERS",
            "facility_id": "FAC-B",
            "equipment_id": "EQ-B-GEN-1",
        },
        {
            "fuel_type": "DIESEL",
            "quantity": 300.0,
            "unit": "LITERS",
            "facility_id": "FAC-B",
            "equipment_id": "EQ-B-GEN-2",
        },
        # Facility C - Coal and wood
        {
            "fuel_type": "COAL_BITUMINOUS",
            "quantity": 2.0,
            "unit": "TONNES",
            "facility_id": "FAC-C",
            "equipment_id": "EQ-C-FURNACE-1",
        },
        {
            "fuel_type": "WOOD",
            "quantity": 5.0,
            "unit": "TONNES",
            "facility_id": "FAC-C",
            "equipment_id": "EQ-C-BOILER-1",
        },
    ]


# =====================================================================
# Known reference values for emission validation
# =====================================================================


@pytest.fixture
def known_emission_ranges() -> Dict[str, Dict[str, Any]]:
    """Approximate emission ranges for common fuels (tCO2e per unit).

    These ranges are intentionally wide to accommodate Tier 1 defaults
    from different sources (EPA, IPCC, DEFRA). Tests validate that
    calculated values fall within these reasonable bounds rather than
    exact point values.
    """
    return {
        "NATURAL_GAS_1000_m3": {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "unit": "CUBIC_METERS",
            "min_co2e_tonnes": 1.5,
            "max_co2e_tonnes": 3.0,
            "note": "~2.0-2.2 tCO2e per 1000 m3 natural gas (varies by source)",
        },
        "DIESEL_1000_liters": {
            "fuel_type": "DIESEL",
            "quantity": 1000.0,
            "unit": "LITERS",
            "min_co2e_tonnes": 2.0,
            "max_co2e_tonnes": 3.5,
            "note": "~2.7 tCO2e per 1000 liters diesel (varies by source)",
        },
        "COAL_BITUMINOUS_1_tonne": {
            "fuel_type": "COAL_BITUMINOUS",
            "quantity": 1.0,
            "unit": "TONNES",
            "min_co2e_tonnes": 2.0,
            "max_co2e_tonnes": 3.5,
            "note": "~2.5 tCO2e per tonne bituminous coal (varies by source)",
        },
        "WOOD_1_tonne": {
            "fuel_type": "WOOD",
            "quantity": 1.0,
            "unit": "TONNES",
            "min_biogenic_co2e_tonnes": 0.0,
            "max_biogenic_co2e_tonnes": 2.5,
            "fossil_co2e_tonnes": 0.0,
            "note": "Wood is biogenic; fossil CO2 should be 0",
        },
    }
