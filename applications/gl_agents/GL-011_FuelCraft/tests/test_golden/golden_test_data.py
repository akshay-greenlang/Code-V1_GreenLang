# -*- coding: utf-8 -*-
"""
Golden Test Data for GL-011 FuelCraft

Provides pre-computed reference values for determinism testing.
All values are hand-calculated and verified for regulatory compliance.

These values serve as the ground truth for:
- Blend calculation verification
- Carbon emission calculations
- Heating value calculations
- Unit conversions
- Cost model calculations

Author: GL-TestEngineer
Date: 2025-01-01
Version: 1.0.0
"""

from decimal import Decimal
from datetime import date
from typing import Dict, List, Any


# =============================================================================
# Golden Values for Blend Calculations
# =============================================================================

GOLDEN_BLEND_CALCULATIONS: List[Dict[str, Any]] = [
    {
        "test_id": "BLEND-001",
        "description": "50/50 diesel/HFO blend",
        "inputs": {
            "components": [
                {
                    "fuel_type": "diesel",
                    "mass_kg": Decimal("1000"),
                    "lhv_mj_kg": Decimal("43.00"),
                    "sulfur_wt_pct": Decimal("0.05"),
                    "carbon_intensity_kg_co2e_mj": Decimal("0.0741"),
                },
                {
                    "fuel_type": "heavy_fuel_oil",
                    "mass_kg": Decimal("1000"),
                    "lhv_mj_kg": Decimal("40.00"),
                    "sulfur_wt_pct": Decimal("3.50"),
                    "carbon_intensity_kg_co2e_mj": Decimal("0.0771"),
                },
            ],
            "fractions": [Decimal("0.5"), Decimal("0.5")],
        },
        "expected_outputs": {
            "total_mass_kg": Decimal("2000.000000"),
            "blend_lhv_mj_kg": Decimal("41.500000"),  # (43*0.5) + (40*0.5)
            "blend_sulfur_wt_pct": Decimal("1.775000"),  # (0.05*0.5) + (3.5*0.5)
            "total_energy_mj": Decimal("83000.000000"),  # 1000*43 + 1000*40
        },
        "tolerance": Decimal("1e-6"),
    },
    {
        "test_id": "BLEND-002",
        "description": "70/30 natural gas/diesel blend",
        "inputs": {
            "components": [
                {
                    "fuel_type": "natural_gas",
                    "mass_kg": Decimal("700"),
                    "lhv_mj_kg": Decimal("50.00"),
                    "sulfur_wt_pct": Decimal("0.00"),
                    "carbon_intensity_kg_co2e_mj": Decimal("0.0561"),
                },
                {
                    "fuel_type": "diesel",
                    "mass_kg": Decimal("300"),
                    "lhv_mj_kg": Decimal("43.00"),
                    "sulfur_wt_pct": Decimal("0.05"),
                    "carbon_intensity_kg_co2e_mj": Decimal("0.0741"),
                },
            ],
            "fractions": [Decimal("0.7"), Decimal("0.3")],
        },
        "expected_outputs": {
            "total_mass_kg": Decimal("1000.000000"),
            "blend_lhv_mj_kg": Decimal("47.900000"),  # (50*0.7) + (43*0.3)
            "blend_sulfur_wt_pct": Decimal("0.015000"),  # (0*0.7) + (0.05*0.3)
            "total_energy_mj": Decimal("47900.000000"),
        },
        "tolerance": Decimal("1e-6"),
    },
    {
        "test_id": "BLEND-003",
        "description": "Pure natural gas (single component)",
        "inputs": {
            "components": [
                {
                    "fuel_type": "natural_gas",
                    "mass_kg": Decimal("5000"),
                    "lhv_mj_kg": Decimal("50.00"),
                    "sulfur_wt_pct": Decimal("0.00"),
                    "carbon_intensity_kg_co2e_mj": Decimal("0.0561"),
                },
            ],
            "fractions": [Decimal("1.0")],
        },
        "expected_outputs": {
            "total_mass_kg": Decimal("5000.000000"),
            "blend_lhv_mj_kg": Decimal("50.000000"),
            "blend_sulfur_wt_pct": Decimal("0.000000"),
            "total_energy_mj": Decimal("250000.000000"),
        },
        "tolerance": Decimal("1e-6"),
    },
]


# =============================================================================
# Golden Values for Carbon Calculations
# =============================================================================

GOLDEN_CARBON_CALCULATIONS: List[Dict[str, Any]] = [
    {
        "test_id": "CARBON-001",
        "description": "Diesel TTW emissions",
        "inputs": {
            "fuel_type": "diesel",
            "energy_mj": Decimal("1000000"),  # 1 TJ
            "boundary": "TTW",
        },
        "expected_outputs": {
            "ttw_emissions_kg_co2e": Decimal("74100.000000"),  # 1000000 * 0.0741
            "ttw_intensity_kg_co2e_mj": Decimal("0.074100"),
        },
        "tolerance": Decimal("1e-4"),
    },
    {
        "test_id": "CARBON-002",
        "description": "Natural gas WTW emissions",
        "inputs": {
            "fuel_type": "natural_gas",
            "energy_mj": Decimal("500000"),  # 0.5 TJ
            "boundary": "WTW",
        },
        "expected_outputs": {
            "ttw_emissions_kg_co2e": Decimal("28050.000000"),  # 500000 * 0.0561
            "wtt_emissions_kg_co2e": Decimal("9150.000000"),   # 500000 * 0.0183
            "wtw_emissions_kg_co2e": Decimal("37200.000000"),  # 500000 * 0.0744
            "wtw_intensity_kg_co2e_mj": Decimal("0.074400"),
        },
        "tolerance": Decimal("1e-4"),
    },
    {
        "test_id": "CARBON-003",
        "description": "Hydrogen zero TTW emissions",
        "inputs": {
            "fuel_type": "hydrogen",
            "energy_mj": Decimal("100000"),
            "boundary": "TTW",
        },
        "expected_outputs": {
            "ttw_emissions_kg_co2e": Decimal("0.000000"),
            "ttw_intensity_kg_co2e_mj": Decimal("0.000000"),
        },
        "tolerance": Decimal("1e-9"),
    },
]


# =============================================================================
# Golden Values for Heating Value Calculations
# =============================================================================

GOLDEN_HEATING_VALUE_CALCULATIONS: List[Dict[str, Any]] = [
    {
        "test_id": "HV-001",
        "description": "Diesel 1000 kg energy content",
        "inputs": {
            "fuel_type": "diesel",
            "quantity": Decimal("1000"),
            "quantity_unit": "kg",
        },
        "expected_outputs": {
            "lhv_mj": Decimal("43000.000000"),  # 1000 * 43
            "hhv_mj": Decimal("45800.000000"),  # 1000 * 45.8
            "mass_kg": Decimal("1000.000000"),
        },
        "tolerance": Decimal("1e-6"),
    },
    {
        "test_id": "HV-002",
        "description": "Natural gas 1 m3 energy content",
        "inputs": {
            "fuel_type": "natural_gas",
            "quantity": Decimal("1000"),  # 1000 m3
            "quantity_unit": "m3",
        },
        "expected_outputs": {
            "mass_kg": Decimal("717.000000"),  # 1000 * 0.717 kg/m3
            "lhv_mj": Decimal("35850.000000"),  # 717 * 50
        },
        "tolerance": Decimal("1e-3"),
    },
    {
        "test_id": "HV-003",
        "description": "Heavy fuel oil volume conversion",
        "inputs": {
            "fuel_type": "heavy_fuel_oil",
            "quantity": Decimal("1"),  # 1 barrel
            "quantity_unit": "bbl",
        },
        "expected_outputs": {
            # 1 bbl = 0.158987 m3, density = 990 kg/m3
            # mass = 0.158987 * 990 = 157.4 kg
            # LHV = 157.4 * 40 = 6296 MJ
            "mass_kg": Decimal("157.397"),
            "lhv_mj": Decimal("6295.880"),
        },
        "tolerance": Decimal("1e-2"),
    },
]


# =============================================================================
# Golden Values for Unit Conversions
# =============================================================================

GOLDEN_UNIT_CONVERSIONS: List[Dict[str, Any]] = [
    {
        "test_id": "CONV-001",
        "description": "MMBtu to MJ conversion",
        "inputs": {
            "value": Decimal("100"),
            "from_unit": "MMBtu",
            "to_unit": "MJ",
        },
        "expected_outputs": {
            "output_value": Decimal("105505.585262"),  # 100 * 1055.05585262
        },
        "tolerance": Decimal("1e-3"),
    },
    {
        "test_id": "CONV-002",
        "description": "kg to lb conversion",
        "inputs": {
            "value": Decimal("1000"),
            "from_unit": "kg",
            "to_unit": "lb",
        },
        "expected_outputs": {
            "output_value": Decimal("2204.622622"),  # 1000 / 0.45359237
        },
        "tolerance": Decimal("1e-3"),
    },
    {
        "test_id": "CONV-003",
        "description": "Barrel to liters conversion",
        "inputs": {
            "value": Decimal("1"),
            "from_unit": "bbl",
            "to_unit": "L",
        },
        "expected_outputs": {
            "output_value": Decimal("158.987295"),  # 1 bbl = 158.987294928 L
        },
        "tolerance": Decimal("1e-3"),
    },
    {
        "test_id": "CONV-004",
        "description": "kWh to MJ conversion (exact)",
        "inputs": {
            "value": Decimal("1000"),
            "from_unit": "kWh",
            "to_unit": "MJ",
        },
        "expected_outputs": {
            "output_value": Decimal("3600.000000"),  # 1000 * 3.6 (exact)
        },
        "tolerance": Decimal("1e-9"),
    },
]


# =============================================================================
# Golden Values for Cost Calculations
# =============================================================================

GOLDEN_COST_CALCULATIONS: List[Dict[str, Any]] = [
    {
        "test_id": "COST-001",
        "description": "Natural gas purchase cost",
        "inputs": {
            "fuel_type": "natural_gas",
            "quantity_mj": Decimal("1000000"),
            "price_per_mj": Decimal("0.0035"),
        },
        "expected_outputs": {
            "purchase_cost": Decimal("3500.000000"),  # 1000000 * 0.0035
        },
        "tolerance": Decimal("1e-6"),
    },
    {
        "test_id": "COST-002",
        "description": "Carbon cost calculation",
        "inputs": {
            "fuel_type": "diesel",
            "quantity_mj": Decimal("1000000"),
            "carbon_intensity": Decimal("0.0741"),
            "carbon_price_per_kg": Decimal("0.050"),
        },
        "expected_outputs": {
            "emissions_kg": Decimal("74100.000000"),  # 1000000 * 0.0741
            "carbon_cost": Decimal("3705.000000"),   # 74100 * 0.050
        },
        "tolerance": Decimal("1e-4"),
    },
    {
        "test_id": "COST-003",
        "description": "Total cost with all components",
        "inputs": {
            "fuel_type": "diesel",
            "quantity_mj": Decimal("1000000"),
            "price_per_mj": Decimal("0.0045"),
            "logistics_cost_per_mj": Decimal("0.0001"),
            "carbon_intensity": Decimal("0.0741"),
            "carbon_price_per_kg": Decimal("0.050"),
        },
        "expected_outputs": {
            "purchase_cost": Decimal("4500.000000"),  # 1000000 * 0.0045
            "logistics_cost": Decimal("100.000000"),  # 1000000 * 0.0001
            "carbon_cost": Decimal("3705.000000"),    # 1000000 * 0.0741 * 0.050
            "total_cost": Decimal("8305.000000"),     # 4500 + 100 + 3705
        },
        "tolerance": Decimal("1e-4"),
    },
]


# =============================================================================
# Golden Values for Provenance Hashes
# =============================================================================

GOLDEN_PROVENANCE_HASHES: List[Dict[str, Any]] = [
    {
        "test_id": "PROV-001",
        "description": "Simple data hash",
        "inputs": {
            "data": {"key": "value", "number": 123},
        },
        "expected_outputs": {
            # SHA-256 of '{"key": "value", "number": 123}' sorted
            "hash_prefix": "7d",  # First 2 chars of expected hash
            "hash_length": 64,
        },
    },
    {
        "test_id": "PROV-002",
        "description": "Order-independent hash",
        "inputs": {
            "data1": {"a": 1, "b": 2, "c": 3},
            "data2": {"c": 3, "a": 1, "b": 2},
        },
        "expected_outputs": {
            "hashes_equal": True,
        },
    },
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_golden_blend_test(test_id: str) -> Dict[str, Any]:
    """Get a specific golden blend test by ID."""
    for test in GOLDEN_BLEND_CALCULATIONS:
        if test["test_id"] == test_id:
            return test
    raise ValueError(f"Golden test {test_id} not found")


def get_golden_carbon_test(test_id: str) -> Dict[str, Any]:
    """Get a specific golden carbon test by ID."""
    for test in GOLDEN_CARBON_CALCULATIONS:
        if test["test_id"] == test_id:
            return test
    raise ValueError(f"Golden test {test_id} not found")


def get_golden_heating_value_test(test_id: str) -> Dict[str, Any]:
    """Get a specific golden heating value test by ID."""
    for test in GOLDEN_HEATING_VALUE_CALCULATIONS:
        if test["test_id"] == test_id:
            return test
    raise ValueError(f"Golden test {test_id} not found")


def get_golden_conversion_test(test_id: str) -> Dict[str, Any]:
    """Get a specific golden conversion test by ID."""
    for test in GOLDEN_UNIT_CONVERSIONS:
        if test["test_id"] == test_id:
            return test
    raise ValueError(f"Golden test {test_id} not found")


def get_golden_cost_test(test_id: str) -> Dict[str, Any]:
    """Get a specific golden cost test by ID."""
    for test in GOLDEN_COST_CALCULATIONS:
        if test["test_id"] == test_id:
            return test
    raise ValueError(f"Golden test {test_id} not found")
