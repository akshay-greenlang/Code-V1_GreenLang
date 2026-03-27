# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-024: Use of Sold Products Agent.

Provides comprehensive test fixtures for:
- Product inputs by use-phase category (vehicles, appliances, HVAC, IT, industrial, fuels)
- Direct emission inputs (fuel combustion, refrigerant leakage, chemical release)
- Indirect emission inputs (electricity consumption, heating fuel, steam/cooling)
- Fuels and feedstocks inputs (fuel sales, feedstock oxidation)
- Lifetime modeling inputs (default lifetimes, degradation, Weibull, fleet)
- Compliance inputs (GHG Protocol, CDP, SBTi, CSRD, ISO 14064, SB 253, GRI)
- Calculation results, compliance results, provenance entries
- Mock database engine with EF lookups, product profiles, refrigerant GWPs
- Configuration objects (18 frozen dataclass configs)
- 7 engine singletons with reset autouse fixture

Usage:
    def test_something(sample_vehicle_gasoline, mock_database_engine):
        result = calculate(sample_vehicle_gasoline, mock_database_engine)
        assert result.total_co2e > 0

Author: GL-TestEngineer
Date: February 2026
"""

from datetime import datetime, date, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import threading
import pytest


# ============================================================================
# GRACEFUL IMPORTS
# ============================================================================

try:
    from greenlang.agents.mrv.use_of_sold_products.product_use_database import (
        ProductUseDatabaseEngine,
    )
    PRODUCT_DB_AVAILABLE = True
except ImportError:
    PRODUCT_DB_AVAILABLE = False
    ProductUseDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.use_of_sold_products.direct_emissions_calculator import (
        DirectEmissionsCalculatorEngine,
    )
    DIRECT_CALC_AVAILABLE = True
except ImportError:
    DIRECT_CALC_AVAILABLE = False
    DirectEmissionsCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.use_of_sold_products.indirect_emissions_calculator import (
        IndirectEmissionsCalculatorEngine,
    )
    INDIRECT_CALC_AVAILABLE = True
except ImportError:
    INDIRECT_CALC_AVAILABLE = False
    IndirectEmissionsCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.use_of_sold_products.fuels_feedstocks_calculator import (
        FuelsAndFeedstocksCalculatorEngine,
    )
    FUELS_CALC_AVAILABLE = True
except ImportError:
    FUELS_CALC_AVAILABLE = False
    FuelsAndFeedstocksCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.use_of_sold_products.lifetime_modeling import (
        LifetimeModelingEngine,
    )
    LIFETIME_AVAILABLE = True
except ImportError:
    LIFETIME_AVAILABLE = False
    LifetimeModelingEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.use_of_sold_products.compliance_checker import (
        ComplianceCheckerEngine,
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.use_of_sold_products.use_of_sold_products_pipeline import (
        UseOfSoldProductsPipelineEngine,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    UseOfSoldProductsPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.use_of_sold_products.config import get_config, reset_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from greenlang.agents.mrv.use_of_sold_products.provenance import (
        ProvenanceChainBuilder,
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False

try:
    from greenlang.agents.mrv.use_of_sold_products.setup import (
        UseOfSoldProductsService,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False


# Engine availability flags
_ENGINE_CLASSES = [
    ("ProductUseDatabaseEngine", ProductUseDatabaseEngine, PRODUCT_DB_AVAILABLE),
    ("DirectEmissionsCalculatorEngine", DirectEmissionsCalculatorEngine, DIRECT_CALC_AVAILABLE),
    ("IndirectEmissionsCalculatorEngine", IndirectEmissionsCalculatorEngine, INDIRECT_CALC_AVAILABLE),
    ("FuelsAndFeedstocksCalculatorEngine", FuelsAndFeedstocksCalculatorEngine, FUELS_CALC_AVAILABLE),
    ("LifetimeModelingEngine", LifetimeModelingEngine, LIFETIME_AVAILABLE),
    ("ComplianceCheckerEngine", ComplianceCheckerEngine, COMPLIANCE_AVAILABLE),
    ("UseOfSoldProductsPipelineEngine", UseOfSoldProductsPipelineEngine, PIPELINE_AVAILABLE),
]

_SKIP_DB = pytest.mark.skipif(
    not PRODUCT_DB_AVAILABLE,
    reason="ProductUseDatabaseEngine not available",
)
_SKIP_DIRECT = pytest.mark.skipif(
    not DIRECT_CALC_AVAILABLE,
    reason="DirectEmissionsCalculatorEngine not available",
)
_SKIP_INDIRECT = pytest.mark.skipif(
    not INDIRECT_CALC_AVAILABLE,
    reason="IndirectEmissionsCalculatorEngine not available",
)
_SKIP_FUELS = pytest.mark.skipif(
    not FUELS_CALC_AVAILABLE,
    reason="FuelsAndFeedstocksCalculatorEngine not available",
)
_SKIP_LIFETIME = pytest.mark.skipif(
    not LIFETIME_AVAILABLE,
    reason="LifetimeModelingEngine not available",
)
_SKIP_COMPLIANCE = pytest.mark.skipif(
    not COMPLIANCE_AVAILABLE,
    reason="ComplianceCheckerEngine not available",
)
_SKIP_PIPELINE = pytest.mark.skipif(
    not PIPELINE_AVAILABLE,
    reason="UseOfSoldProductsPipelineEngine not available",
)
_SKIP_CONFIG = pytest.mark.skipif(
    not CONFIG_AVAILABLE,
    reason="Config module not available",
)
_SKIP_PROVENANCE = pytest.mark.skipif(
    not PROVENANCE_AVAILABLE,
    reason="Provenance module not available",
)
_SKIP_SETUP = pytest.mark.skipif(
    not SETUP_AVAILABLE,
    reason="Setup module not available",
)


# ============================================================================
# AUTOUSE SINGLETON RESET FIXTURE
# ============================================================================

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all 7 engine singletons before and after each test for isolation."""
    _do_reset()
    yield
    _do_reset()


def _do_reset():
    """Attempt to call reset_instance or reset_singleton on each engine class."""
    for name, cls, available in _ENGINE_CLASSES:
        if not available or cls is None:
            continue
        for method_name in ("reset_instance", "reset_singleton", "_reset", "reset"):
            reset_fn = getattr(cls, method_name, None)
            if reset_fn is not None and callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    pass
                break
    # Also reset config singleton
    if CONFIG_AVAILABLE:
        try:
            reset_config()
        except Exception:
            pass


# ============================================================================
# PRODUCT INPUT FIXTURES (by use-phase category)
# ============================================================================

@pytest.fixture
def sample_vehicle_gasoline() -> Dict[str, Any]:
    """Gasoline passenger vehicle -- direct fuel combustion during use.

    1000 units, 15yr lifetime, 1200 litres/yr, EF=2.315 kg CO2e/litre.
    Expected: 1000 x 15 x 1200 x 2.315 = 41,670,000 kgCO2e.
    """
    return {
        "product_id": "PRD-VEH-001",
        "product_name": "Sedan 2.0L Gasoline",
        "category": "vehicles",
        "subcategory": "passenger_car",
        "units_sold": 1000,
        "fuel_type": "gasoline",
        "fuel_consumption_litres_per_year": Decimal("1200.0"),
        "fuel_ef_kg_co2e_per_litre": Decimal("2.315"),
        "expected_lifetime_years": 15,
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "direct",
    }


@pytest.fixture
def sample_appliance_fridge() -> Dict[str, Any]:
    """Refrigerator -- indirect electricity consumption during use.

    10,000 units, 15yr lifetime, 400 kWh/yr, grid EF=0.417 kgCO2e/kWh.
    Expected: 10000 x 15 x 400 x 0.417 = 25,020,000 kgCO2e.
    """
    return {
        "product_id": "PRD-APP-001",
        "product_name": "Energy Star Refrigerator 500L",
        "category": "appliances",
        "subcategory": "refrigerator",
        "units_sold": 10000,
        "energy_consumption_kwh_per_year": Decimal("400.0"),
        "grid_ef_kg_co2e_per_kwh": Decimal("0.417"),
        "expected_lifetime_years": 15,
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "indirect",
    }


@pytest.fixture
def sample_hvac_ac() -> Dict[str, Any]:
    """Air conditioner -- direct refrigerant leakage + indirect electricity.

    500 units, 12yr lifetime, 3kg R-410A charge, 5% annual leak rate.
    R-410A GWP (AR5) = 2088.
    Direct: 500 x 3 x 0.05 x 2088 x 12 = 1,879,200 kgCO2e.
    Indirect: 500 x 12 x 1500kWh x 0.417 = 3,753,000 kgCO2e.
    """
    return {
        "product_id": "PRD-HVAC-001",
        "product_name": "Split AC 3.5kW R-410A",
        "category": "hvac",
        "subcategory": "air_conditioner",
        "units_sold": 500,
        "refrigerant_type": "R-410A",
        "refrigerant_charge_kg": Decimal("3.0"),
        "annual_leak_rate": Decimal("0.05"),
        "refrigerant_gwp": Decimal("2088"),
        "energy_consumption_kwh_per_year": Decimal("1500.0"),
        "grid_ef_kg_co2e_per_kwh": Decimal("0.417"),
        "expected_lifetime_years": 12,
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "both",
    }


@pytest.fixture
def sample_it_laptop() -> Dict[str, Any]:
    """Laptop -- indirect electricity consumption during use.

    50,000 units, 5yr lifetime, 50 kWh/yr, grid EF=0.417 kgCO2e/kWh.
    Expected: 50000 x 5 x 50 x 0.417 = 5,212,500 kgCO2e.
    """
    return {
        "product_id": "PRD-IT-001",
        "product_name": "Business Laptop 14-inch",
        "category": "it_equipment",
        "subcategory": "laptop",
        "units_sold": 50000,
        "energy_consumption_kwh_per_year": Decimal("50.0"),
        "grid_ef_kg_co2e_per_kwh": Decimal("0.417"),
        "expected_lifetime_years": 5,
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "indirect",
    }


@pytest.fixture
def sample_industrial_generator() -> Dict[str, Any]:
    """Diesel generator -- direct fuel combustion + indirect electricity backup.

    200 units, 20yr lifetime, 5000 litres/yr diesel, EF=2.68 kgCO2e/litre.
    Direct: 200 x 20 x 5000 x 2.68 = 53,600,000 kgCO2e.
    """
    return {
        "product_id": "PRD-IND-001",
        "product_name": "Diesel Generator 500kVA",
        "category": "industrial_equipment",
        "subcategory": "generator",
        "units_sold": 200,
        "fuel_type": "diesel",
        "fuel_consumption_litres_per_year": Decimal("5000.0"),
        "fuel_ef_kg_co2e_per_litre": Decimal("2.68"),
        "expected_lifetime_years": 20,
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "direct",
    }


@pytest.fixture
def sample_fuel_sale() -> Dict[str, Any]:
    """Gasoline fuel sale -- combustion by end user.

    1,000,000 litres gasoline, EF=2.315 kgCO2e/litre.
    Expected: 1000000 x 2.315 = 2,315,000 kgCO2e.
    """
    return {
        "product_id": "PRD-FUEL-001",
        "product_name": "Unleaded Gasoline 95",
        "category": "fuels_feedstocks",
        "subcategory": "gasoline",
        "quantity_sold_litres": Decimal("1000000.0"),
        "fuel_type": "gasoline",
        "fuel_ef_kg_co2e_per_litre": Decimal("2.315"),
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "direct",
    }


@pytest.fixture
def sample_consumer_aerosol() -> Dict[str, Any]:
    """Aerosol spray -- direct chemical release of propellant.

    100,000 units, propellant HFC-134a, 0.15 kg/unit, GWP=1430.
    Expected: 100000 x 0.15 x 1430 = 21,450,000 kgCO2e.
    """
    return {
        "product_id": "PRD-CON-001",
        "product_name": "Aerosol Spray 300ml",
        "category": "consumer_products",
        "subcategory": "aerosol",
        "units_sold": 100000,
        "chemical_type": "HFC-134a",
        "chemical_mass_kg_per_unit": Decimal("0.15"),
        "chemical_gwp": Decimal("1430"),
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "direct",
    }


@pytest.fixture
def sample_lighting_led() -> Dict[str, Any]:
    """LED bulb -- indirect electricity consumption.

    500,000 units, 25yr lifetime (25,000 hours / 1000 hr/yr),
    10W = 10 kWh/yr, grid EF=0.417 kgCO2e/kWh.
    Expected: 500000 x 25 x 10 x 0.417 = 52,125,000 kgCO2e.
    """
    return {
        "product_id": "PRD-LGT-001",
        "product_name": "LED Bulb 10W A19",
        "category": "lighting",
        "subcategory": "led_bulb",
        "units_sold": 500000,
        "energy_consumption_kwh_per_year": Decimal("10.0"),
        "grid_ef_kg_co2e_per_kwh": Decimal("0.417"),
        "expected_lifetime_years": 25,
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "indirect",
    }


@pytest.fixture
def sample_building_furnace() -> Dict[str, Any]:
    """Gas furnace -- indirect heating fuel consumption.

    5,000 units, 20yr lifetime, 2000 m3 natural gas/yr, EF=1.93 kgCO2e/m3.
    Expected: 5000 x 20 x 2000 x 1.93 = 386,000,000 kgCO2e.
    """
    return {
        "product_id": "PRD-BLD-001",
        "product_name": "Gas Furnace 80k BTU",
        "category": "building_products",
        "subcategory": "furnace",
        "units_sold": 5000,
        "fuel_type": "natural_gas",
        "fuel_consumption_m3_per_year": Decimal("2000.0"),
        "fuel_ef_kg_co2e_per_m3": Decimal("1.93"),
        "expected_lifetime_years": 20,
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "indirect",
    }


@pytest.fixture
def sample_medical_mri() -> Dict[str, Any]:
    """MRI machine -- indirect electricity consumption.

    100 units, 12yr lifetime, 50,000 kWh/yr, grid EF=0.417 kgCO2e/kWh.
    Expected: 100 x 12 x 50000 x 0.417 = 25,020,000 kgCO2e.
    """
    return {
        "product_id": "PRD-MED-001",
        "product_name": "MRI Scanner 3T",
        "category": "medical_devices",
        "subcategory": "imaging",
        "units_sold": 100,
        "energy_consumption_kwh_per_year": Decimal("50000.0"),
        "grid_ef_kg_co2e_per_kwh": Decimal("0.417"),
        "expected_lifetime_years": 12,
        "reporting_year": 2024,
        "region": "US",
        "emission_type": "indirect",
    }


# ============================================================================
# CALCULATION RESULT FIXTURES
# ============================================================================

@pytest.fixture
def sample_calculation_result() -> Dict[str, Any]:
    """A fully populated calculation result for testing downstream engines."""
    return {
        "calculation_id": "calc-usp-001",
        "product_id": "PRD-VEH-001",
        "product_name": "Sedan 2.0L Gasoline",
        "category": "vehicles",
        "emission_type": "direct",
        "method": "direct_fuel_combustion",
        "units_sold": 1000,
        "lifetime_years": 15,
        "total_co2e_kg": Decimal("41670000.0"),
        "co2e_per_unit": Decimal("41670.0"),
        "co2e_per_unit_per_year": Decimal("2778.0"),
        "dqi_score": Decimal("80.0"),
        "uncertainty_lower": Decimal("37503000.0"),
        "uncertainty_upper": Decimal("45837000.0"),
        "provenance_hash": "a" * 64,
        "region": "US",
        "reporting_year": 2024,
        "calculated_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_compliance_result() -> Dict[str, Any]:
    """A fully compliant result for framework checks."""
    return {
        "total_co2e_kg": Decimal("41670000.0"),
        "method": "direct_fuel_combustion",
        "calculation_method": "direct_fuel_combustion",
        "ef_sources": ["defra", "epa"],
        "ef_source": "defra",
        "product_category": "vehicles",
        "emission_type": "direct",
        "boundary": "use_phase_only",
        "exclusions": "None - all use-phase emissions included",
        "dqi_score": Decimal("80.0"),
        "data_quality_score": Decimal("80.0"),
        "uncertainty_analysis": {"method": "propagation", "ci_95": [37503000, 45837000]},
        "uncertainty": {"method": "propagation"},
        "base_year": 2019,
        "methodology": "GHG Protocol Scope 3 Category 11, direct fuel combustion",
        "targets": "Reduce use-phase emissions 30% by 2030",
        "reduction_targets": "30% by 2030",
        "actions": "Fleet efficiency improvement, electrification",
        "reduction_actions": "Fleet efficiency programs",
        "verification_status": "limited_assurance",
        "verified": True,
        "assurance": "limited",
        "assurance_opinion": "limited_assurance",
        "reporting_period": "2024",
        "period": "2024",
        "gases_included": ["CO2", "CH4", "N2O"],
        "emission_gases": ["CO2", "CH4", "N2O"],
        "standards_used": ["GHG Protocol", "ISO 14064"],
        "standards": ["GHG Protocol"],
        "total_scope3_co2e": Decimal("120000000.0"),
        "product_count": 15,
        "by_category": {
            "vehicles": Decimal("41670000.0"),
            "appliances": Decimal("25020000.0"),
        },
        "by_emission_type": {
            "direct": Decimal("41670000.0"),
            "indirect": Decimal("25020000.0"),
        },
        "by_method": {
            "direct_fuel_combustion": Decimal("41670000.0"),
            "indirect_electricity": Decimal("25020000.0"),
        },
        "completeness_score": Decimal("85.0"),
        "method_coverage": Decimal("90.0"),
        "lifetime_assumptions_documented": True,
        "use_profiles_documented": True,
        "intermediate_only": False,
        "end_product_included": True,
        "allocation_method": "units_sold",
        "progress_tracking": {"2023": 44000000, "2024": 41670000},
        "year_over_year_change": Decimal("-5.3"),
        "target_coverage": "80%",
        "sbti_coverage": "80%",
    }


# ============================================================================
# EMISSION FACTOR DATABASE FIXTURES
# ============================================================================

@pytest.fixture
def sample_fuel_ef_db() -> Dict[str, Dict[str, Any]]:
    """Fuel emission factor database for fuel combustion lookups.

    Keys: fuel_type -> {ef_kg_per_litre, ef_kg_per_m3, source}.
    """
    return {
        "gasoline": {"ef_kg_per_litre": Decimal("2.315"), "source": "DEFRA"},
        "diesel": {"ef_kg_per_litre": Decimal("2.680"), "source": "DEFRA"},
        "natural_gas": {"ef_kg_per_m3": Decimal("1.930"), "source": "DEFRA"},
        "lpg": {"ef_kg_per_litre": Decimal("1.553"), "source": "DEFRA"},
        "ethanol_e85": {"ef_kg_per_litre": Decimal("1.610"), "source": "EPA"},
        "biodiesel_b20": {"ef_kg_per_litre": Decimal("2.144"), "source": "EPA"},
        "cng": {"ef_kg_per_m3": Decimal("2.020"), "source": "IPCC"},
        "lng": {"ef_kg_per_litre": Decimal("1.180"), "source": "IPCC"},
        "kerosene": {"ef_kg_per_litre": Decimal("2.540"), "source": "DEFRA"},
        "heating_oil": {"ef_kg_per_litre": Decimal("2.960"), "source": "DEFRA"},
        "propane": {"ef_kg_per_litre": Decimal("1.510"), "source": "DEFRA"},
        "coal_anthracite": {"ef_kg_per_kg": Decimal("2.860"), "source": "IPCC"},
        "coal_bituminous": {"ef_kg_per_kg": Decimal("2.420"), "source": "IPCC"},
        "wood_pellets": {"ef_kg_per_kg": Decimal("0.039"), "source": "DEFRA"},
        "hydrogen": {"ef_kg_per_kg": Decimal("0.000"), "source": "IPCC"},
    }


@pytest.fixture
def sample_refrigerant_gwp_db() -> Dict[str, Dict[str, Decimal]]:
    """Refrigerant GWP database (AR5 and AR6 values).

    Keys: refrigerant_name -> {gwp_ar5, gwp_ar6}.
    """
    return {
        "R-134a": {"gwp_ar5": Decimal("1430"), "gwp_ar6": Decimal("1530")},
        "R-410A": {"gwp_ar5": Decimal("2088"), "gwp_ar6": Decimal("2256")},
        "R-32": {"gwp_ar5": Decimal("675"), "gwp_ar6": Decimal("771")},
        "R-404A": {"gwp_ar5": Decimal("3922"), "gwp_ar6": Decimal("4728")},
        "R-407C": {"gwp_ar5": Decimal("1774"), "gwp_ar6": Decimal("1908")},
        "R-290": {"gwp_ar5": Decimal("3"), "gwp_ar6": Decimal("0.02")},
        "R-600a": {"gwp_ar5": Decimal("3"), "gwp_ar6": Decimal("0.02")},
        "R-744": {"gwp_ar5": Decimal("1"), "gwp_ar6": Decimal("1")},
        "R-1234yf": {"gwp_ar5": Decimal("4"), "gwp_ar6": Decimal("0.50")},
        "R-1234ze": {"gwp_ar5": Decimal("7"), "gwp_ar6": Decimal("1.40")},
    }


@pytest.fixture
def sample_grid_ef_db() -> Dict[str, Dict[str, Any]]:
    """Grid emission factor database by region (kgCO2e/kWh)."""
    return {
        "US": {"ef_kg_per_kwh": Decimal("0.417"), "source": "eGRID"},
        "US_CAMX": {"ef_kg_per_kwh": Decimal("0.275"), "source": "eGRID"},
        "US_RFCW": {"ef_kg_per_kwh": Decimal("0.520"), "source": "eGRID"},
        "US_SRMW": {"ef_kg_per_kwh": Decimal("0.680"), "source": "eGRID"},
        "DE": {"ef_kg_per_kwh": Decimal("0.350"), "source": "IEA"},
        "CN": {"ef_kg_per_kwh": Decimal("0.580"), "source": "IEA"},
        "GB": {"ef_kg_per_kwh": Decimal("0.230"), "source": "IEA"},
        "JP": {"ef_kg_per_kwh": Decimal("0.470"), "source": "IEA"},
        "IN": {"ef_kg_per_kwh": Decimal("0.710"), "source": "IEA"},
        "BR": {"ef_kg_per_kwh": Decimal("0.080"), "source": "IEA"},
        "FR": {"ef_kg_per_kwh": Decimal("0.060"), "source": "IEA"},
        "AU": {"ef_kg_per_kwh": Decimal("0.630"), "source": "IEA"},
        "CA": {"ef_kg_per_kwh": Decimal("0.130"), "source": "IEA"},
        "KR": {"ef_kg_per_kwh": Decimal("0.460"), "source": "IEA"},
        "ZA": {"ef_kg_per_kwh": Decimal("0.920"), "source": "IEA"},
        "GLOBAL": {"ef_kg_per_kwh": Decimal("0.440"), "source": "IEA"},
    }


@pytest.fixture
def sample_product_profiles_db() -> Dict[str, Dict[str, Any]]:
    """Product use profile database with default lifetimes and usage patterns."""
    return {
        "passenger_car": {
            "default_lifetime_years": 15,
            "fuel_consumption_litres_per_year": Decimal("1200.0"),
            "degradation_rate": Decimal("0.005"),
        },
        "refrigerator": {
            "default_lifetime_years": 15,
            "energy_consumption_kwh_per_year": Decimal("400.0"),
            "degradation_rate": Decimal("0.01"),
        },
        "air_conditioner": {
            "default_lifetime_years": 12,
            "energy_consumption_kwh_per_year": Decimal("1500.0"),
            "refrigerant_charge_kg": Decimal("3.0"),
            "annual_leak_rate": Decimal("0.05"),
            "degradation_rate": Decimal("0.008"),
        },
        "laptop": {
            "default_lifetime_years": 5,
            "energy_consumption_kwh_per_year": Decimal("50.0"),
            "degradation_rate": Decimal("0.02"),
        },
        "generator": {
            "default_lifetime_years": 20,
            "fuel_consumption_litres_per_year": Decimal("5000.0"),
            "degradation_rate": Decimal("0.003"),
        },
        "led_bulb": {
            "default_lifetime_years": 25,
            "energy_consumption_kwh_per_year": Decimal("10.0"),
            "degradation_rate": Decimal("0.002"),
        },
        "furnace": {
            "default_lifetime_years": 20,
            "fuel_consumption_m3_per_year": Decimal("2000.0"),
            "degradation_rate": Decimal("0.005"),
        },
        "washing_machine": {
            "default_lifetime_years": 12,
            "energy_consumption_kwh_per_year": Decimal("250.0"),
            "degradation_rate": Decimal("0.01"),
        },
        "heat_pump": {
            "default_lifetime_years": 15,
            "energy_consumption_kwh_per_year": Decimal("3000.0"),
            "refrigerant_charge_kg": Decimal("5.0"),
            "annual_leak_rate": Decimal("0.03"),
            "degradation_rate": Decimal("0.005"),
        },
        "server": {
            "default_lifetime_years": 7,
            "energy_consumption_kwh_per_year": Decimal("4380.0"),
            "degradation_rate": Decimal("0.0"),
        },
        "electric_oven": {
            "default_lifetime_years": 15,
            "energy_consumption_kwh_per_year": Decimal("300.0"),
            "degradation_rate": Decimal("0.005"),
        },
        "commercial_chiller": {
            "default_lifetime_years": 20,
            "energy_consumption_kwh_per_year": Decimal("25000.0"),
            "refrigerant_charge_kg": Decimal("50.0"),
            "annual_leak_rate": Decimal("0.10"),
            "degradation_rate": Decimal("0.003"),
        },
    }


@pytest.fixture
def sample_steam_cooling_ef_db() -> Dict[str, Dict[str, Any]]:
    """Steam and cooling emission factors for district energy."""
    return {
        "steam_boiler_gas": {"ef_kg_per_kwh": Decimal("0.200"), "source": "DEFRA"},
        "steam_boiler_oil": {"ef_kg_per_kwh": Decimal("0.270"), "source": "DEFRA"},
        "steam_chp": {"ef_kg_per_kwh": Decimal("0.150"), "source": "DEFRA"},
        "cooling_electric_chiller": {"ef_kg_per_kwh": Decimal("0.140"), "source": "IEA"},
        "cooling_absorption": {"ef_kg_per_kwh": Decimal("0.090"), "source": "IEA"},
        "district_heating": {"ef_kg_per_kwh": Decimal("0.180"), "source": "IEA"},
        "district_cooling": {"ef_kg_per_kwh": Decimal("0.120"), "source": "IEA"},
    }


@pytest.fixture
def sample_chemical_ef_db() -> Dict[str, Dict[str, Any]]:
    """Chemical product emission factors for direct chemical release."""
    return {
        "HFC-134a": {"gwp": Decimal("1430"), "source": "IPCC_AR5"},
        "HFC-152a": {"gwp": Decimal("124"), "source": "IPCC_AR5"},
        "propane_propellant": {"gwp": Decimal("3"), "source": "IPCC_AR5"},
        "butane_propellant": {"gwp": Decimal("4"), "source": "IPCC_AR5"},
        "dimethyl_ether": {"gwp": Decimal("1"), "source": "IPCC_AR5"},
        "sf6": {"gwp": Decimal("22800"), "source": "IPCC_AR5"},
        "nf3": {"gwp": Decimal("17200"), "source": "IPCC_AR5"},
    }


@pytest.fixture
def sample_feedstock_db() -> Dict[str, Dict[str, Any]]:
    """Feedstock properties for feedstock oxidation calculations."""
    return {
        "naphtha": {
            "carbon_content": Decimal("0.836"),
            "oxidation_factor": Decimal("1.00"),
            "ef_kg_co2_per_kg": Decimal("3.065"),
        },
        "ethane": {
            "carbon_content": Decimal("0.799"),
            "oxidation_factor": Decimal("1.00"),
            "ef_kg_co2_per_kg": Decimal("2.930"),
        },
        "natural_gas_liquid": {
            "carbon_content": Decimal("0.830"),
            "oxidation_factor": Decimal("1.00"),
            "ef_kg_co2_per_kg": Decimal("3.043"),
        },
        "coal_feedstock": {
            "carbon_content": Decimal("0.710"),
            "oxidation_factor": Decimal("0.98"),
            "ef_kg_co2_per_kg": Decimal("2.552"),
        },
    }


# ============================================================================
# MOCK DATABASE ENGINE FIXTURE
# ============================================================================

@pytest.fixture
def mock_database_engine(
    sample_fuel_ef_db,
    sample_refrigerant_gwp_db,
    sample_grid_ef_db,
    sample_product_profiles_db,
    sample_steam_cooling_ef_db,
    sample_chemical_ef_db,
    sample_feedstock_db,
) -> MagicMock:
    """Create a fully mocked ProductUseDatabaseEngine with EF lookups."""
    engine = MagicMock(spec=["get_fuel_ef", "get_refrigerant_gwp",
                             "get_grid_ef", "get_product_profile",
                             "get_steam_cooling_ef", "get_chemical_ef",
                             "get_feedstock_properties",
                             "get_default_lifetime", "get_degradation_rate",
                             "get_database_summary", "get_lookup_count",
                             "validate_category", "validate_fuel_type",
                             "get_available_categories", "get_available_fuel_types",
                             "get_available_regions", "get_available_refrigerants",
                             "lookup_product_category"])

    def _get_fuel_ef(fuel_type):
        if fuel_type in sample_fuel_ef_db:
            return sample_fuel_ef_db[fuel_type]
        raise ValueError(f"Fuel EF not found for {fuel_type}")

    def _get_refrigerant_gwp(refrigerant, gwp_version="AR5"):
        if refrigerant in sample_refrigerant_gwp_db:
            key = f"gwp_{gwp_version.lower()}" if isinstance(gwp_version, str) else "gwp_ar5"
            return sample_refrigerant_gwp_db[refrigerant].get(key, sample_refrigerant_gwp_db[refrigerant]["gwp_ar5"])
        raise ValueError(f"Refrigerant GWP not found for {refrigerant}")

    def _get_grid_ef(region):
        if region in sample_grid_ef_db:
            return sample_grid_ef_db[region]
        if "GLOBAL" in sample_grid_ef_db:
            return sample_grid_ef_db["GLOBAL"]
        raise ValueError(f"Grid EF not found for region {region}")

    def _get_product_profile(subcategory):
        if subcategory in sample_product_profiles_db:
            return sample_product_profiles_db[subcategory]
        return None

    def _get_steam_cooling_ef(system_type):
        if system_type in sample_steam_cooling_ef_db:
            return sample_steam_cooling_ef_db[system_type]
        raise ValueError(f"Steam/cooling EF not found for {system_type}")

    def _get_chemical_ef(chemical_type):
        if chemical_type in sample_chemical_ef_db:
            return sample_chemical_ef_db[chemical_type]
        raise ValueError(f"Chemical EF not found for {chemical_type}")

    def _get_feedstock_properties(feedstock_type):
        if feedstock_type in sample_feedstock_db:
            return sample_feedstock_db[feedstock_type]
        raise ValueError(f"Feedstock not found for {feedstock_type}")

    def _get_default_lifetime(subcategory):
        if subcategory in sample_product_profiles_db:
            return sample_product_profiles_db[subcategory]["default_lifetime_years"]
        return 10  # fallback default

    def _get_degradation_rate(subcategory):
        if subcategory in sample_product_profiles_db:
            return sample_product_profiles_db[subcategory].get("degradation_rate", Decimal("0.0"))
        return Decimal("0.0")

    engine.get_fuel_ef.side_effect = _get_fuel_ef
    engine.get_refrigerant_gwp.side_effect = _get_refrigerant_gwp
    engine.get_grid_ef.side_effect = _get_grid_ef
    engine.get_product_profile.side_effect = _get_product_profile
    engine.get_steam_cooling_ef.side_effect = _get_steam_cooling_ef
    engine.get_chemical_ef.side_effect = _get_chemical_ef
    engine.get_feedstock_properties.side_effect = _get_feedstock_properties
    engine.get_default_lifetime.side_effect = _get_default_lifetime
    engine.get_degradation_rate.side_effect = _get_degradation_rate
    engine.get_lookup_count.return_value = 0
    engine.validate_category.return_value = True
    engine.validate_fuel_type.return_value = True
    engine.get_available_categories.return_value = [
        "vehicles", "appliances", "hvac", "lighting", "it_equipment",
        "industrial_equipment", "fuels_feedstocks", "building_products",
        "consumer_products", "medical_devices",
    ]
    engine.get_available_fuel_types.return_value = [
        "gasoline", "diesel", "natural_gas", "lpg", "ethanol_e85",
        "biodiesel_b20", "cng", "lng", "kerosene", "heating_oil",
        "propane", "coal_anthracite", "coal_bituminous", "wood_pellets",
        "hydrogen",
    ]
    engine.get_available_regions.return_value = [
        "US", "US_CAMX", "US_RFCW", "US_SRMW", "DE", "CN", "GB",
        "JP", "IN", "BR", "FR", "AU", "CA", "KR", "ZA", "GLOBAL",
    ]
    engine.get_available_refrigerants.return_value = [
        "R-134a", "R-410A", "R-32", "R-404A", "R-407C",
        "R-290", "R-600a", "R-744", "R-1234yf", "R-1234ze",
    ]
    engine.lookup_product_category.return_value = "vehicles"

    engine.get_database_summary.return_value = {
        "categories": 10,
        "fuel_types": 15,
        "regions": 16,
        "refrigerants": 10,
        "product_profiles": 12,
        "steam_cooling_types": 7,
        "chemical_types": 7,
        "feedstock_types": 4,
    }

    return engine


# ============================================================================
# MOCK CONFIG FIXTURE
# ============================================================================

@pytest.fixture
def mock_config() -> MagicMock:
    """Create a fully mocked configuration object."""
    cfg = MagicMock()
    cfg.general.enabled = True
    cfg.general.debug = False
    cfg.general.log_level = "INFO"
    cfg.general.max_batch_size = 1000
    cfg.general.agent_id = "GL-MRV-S3-011"
    cfg.general.agent_component = "AGENT-MRV-024"
    cfg.general.version = "1.0.0"
    cfg.general.api_prefix = "/api/v1/use-of-sold-products"
    cfg.general.default_gwp = "AR5"
    cfg.database.host = "localhost"
    cfg.database.port = 5432
    cfg.database.database = "greenlang"
    cfg.database.table_prefix = "gl_usp_"
    cfg.database.schema = "use_of_sold_products_service"
    cfg.direct.enable_fuel_combustion = True
    cfg.direct.enable_refrigerant_leakage = True
    cfg.direct.enable_chemical_release = True
    cfg.indirect.enable_electricity = True
    cfg.indirect.enable_heating_fuel = True
    cfg.indirect.enable_steam_cooling = True
    cfg.indirect.default_grid_region = "GLOBAL"
    cfg.fuels.enable_fuel_sales = True
    cfg.fuels.enable_feedstock_oxidation = True
    cfg.lifetime.default_degradation_model = "linear"
    cfg.lifetime.enable_weibull = True
    cfg.lifetime.enable_fleet_survival = True
    cfg.lifetime.discount_rate = Decimal("0.0")
    cfg.compliance.get_frameworks.return_value = [
        "GHG_PROTOCOL_SCOPE3", "ISO_14064", "CSRD_ESRS_E1",
        "CDP", "SBTI", "SB_253", "GRI",
    ]
    cfg.compliance.strict_mode = False
    cfg.compliance.materiality_threshold = Decimal("0.01")
    cfg.provenance.enable_provenance = True
    cfg.provenance.hash_algorithm = "sha256"
    cfg.cache.enable_cache = True
    cfg.cache.ttl_seconds = 3600
    cfg.metrics.enable_metrics = True
    cfg.metrics.prefix = "gl_usp_"
    cfg.uncertainty.default_method = "propagation"
    cfg.uncertainty.confidence_level = Decimal("0.95")
    cfg.api.enable_api = True
    cfg.api.prefix = "/api/v1/use-of-sold-products"
    return cfg


# ============================================================================
# BATCH INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_batch_inputs(
    sample_vehicle_gasoline,
    sample_appliance_fridge,
    sample_hvac_ac,
    sample_it_laptop,
    sample_fuel_sale,
) -> List[Dict[str, Any]]:
    """Sample batch of 5 diverse product inputs."""
    return [
        sample_vehicle_gasoline,
        sample_appliance_fridge,
        sample_hvac_ac,
        sample_it_laptop,
        sample_fuel_sale,
    ]


@pytest.fixture
def sample_portfolio_inputs(
    sample_vehicle_gasoline,
    sample_appliance_fridge,
    sample_hvac_ac,
    sample_it_laptop,
    sample_industrial_generator,
    sample_fuel_sale,
    sample_consumer_aerosol,
    sample_lighting_led,
    sample_building_furnace,
    sample_medical_mri,
) -> List[Dict[str, Any]]:
    """Full portfolio of 10 product types."""
    return [
        sample_vehicle_gasoline,
        sample_appliance_fridge,
        sample_hvac_ac,
        sample_it_laptop,
        sample_industrial_generator,
        sample_fuel_sale,
        sample_consumer_aerosol,
        sample_lighting_led,
        sample_building_furnace,
        sample_medical_mri,
    ]


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def mock_direct_engine() -> Mock:
    """Mock direct emissions calculator engine."""
    mock = Mock()
    mock.calculate_fuel_combustion = Mock(return_value={
        "total_co2e_kg": Decimal("41670000.0"),
        "co2e_per_unit": Decimal("41670.0"),
        "fuel_type": "gasoline",
        "lifetime_years": 15,
    })
    mock.calculate_refrigerant_leakage = Mock(return_value={
        "total_co2e_kg": Decimal("1879200.0"),
        "co2e_per_unit": Decimal("3758.4"),
        "refrigerant_type": "R-410A",
        "lifetime_years": 12,
    })
    mock.calculate_chemical_release = Mock(return_value={
        "total_co2e_kg": Decimal("21450000.0"),
        "co2e_per_unit": Decimal("214.5"),
        "chemical_type": "HFC-134a",
    })
    return mock


@pytest.fixture
def mock_indirect_engine() -> Mock:
    """Mock indirect emissions calculator engine."""
    mock = Mock()
    mock.calculate_electricity = Mock(return_value={
        "total_co2e_kg": Decimal("25020000.0"),
        "co2e_per_unit": Decimal("2502.0"),
        "grid_ef_region": "US",
        "lifetime_years": 15,
    })
    mock.calculate_heating_fuel = Mock(return_value={
        "total_co2e_kg": Decimal("386000000.0"),
        "co2e_per_unit": Decimal("77200.0"),
        "fuel_type": "natural_gas",
        "lifetime_years": 20,
    })
    mock.calculate_steam_cooling = Mock(return_value={
        "total_co2e_kg": Decimal("5000000.0"),
        "co2e_per_unit": Decimal("10000.0"),
    })
    return mock


@pytest.fixture
def mock_fuels_engine() -> Mock:
    """Mock fuels and feedstocks calculator engine."""
    mock = Mock()
    mock.calculate_fuel_sales = Mock(return_value={
        "total_co2e_kg": Decimal("2315000.0"),
        "fuel_type": "gasoline",
        "quantity_litres": Decimal("1000000.0"),
    })
    mock.calculate_feedstock_oxidation = Mock(return_value={
        "total_co2e_kg": Decimal("3065000.0"),
        "feedstock_type": "naphtha",
        "quantity_kg": Decimal("1000000.0"),
    })
    return mock


@pytest.fixture
def mock_lifetime_engine() -> Mock:
    """Mock lifetime modeling engine."""
    mock = Mock()
    mock.get_default_lifetime = Mock(return_value=15)
    mock.calculate_degradation = Mock(return_value=Decimal("0.93"))
    mock.calculate_weibull_survival = Mock(return_value=Decimal("0.88"))
    mock.calculate_fleet_emissions = Mock(return_value=Decimal("38000000.0"))
    mock.calculate_discounted_emissions = Mock(return_value=Decimal("35000000.0"))
    return mock


@pytest.fixture
def mock_compliance_engine() -> Mock:
    """Mock compliance checker engine."""
    mock = Mock()
    mock.check_compliance = Mock(return_value={
        "compliant": True,
        "framework": "GHG_PROTOCOL_SCOPE3",
        "issues": [],
        "warnings": [],
        "timestamp": datetime.now(timezone.utc),
    })
    return mock


@pytest.fixture
def mock_pipeline_engine() -> Mock:
    """Mock use of sold products pipeline orchestrator."""
    mock = Mock()
    mock.execute = Mock(return_value={
        "calculation_id": "CALC-USP-001",
        "total_emissions_co2e_kg": Decimal("70000000.0"),
        "direct_emissions_co2e_kg": Decimal("43549200.0"),
        "indirect_emissions_co2e_kg": Decimal("25020000.0"),
        "fuels_feedstocks_co2e_kg": Decimal("2315000.0"),
        "product_count": 5,
        "compliant": True,
        "data_quality_score": Decimal("80.0"),
    })
    return mock


@pytest.fixture
def mock_service() -> Mock:
    """Mock use of sold products service facade."""
    mock = Mock()
    mock.calculate_use_phase_emissions = AsyncMock(return_value={
        "calculation_id": "CALC-USP-001",
        "emissions_tco2e": Decimal("70000.0"),
        "status": "success",
    })
    return mock


# ============================================================================
# METRICS AND PROVENANCE FIXTURES
# ============================================================================

@pytest.fixture
def mock_metrics() -> Mock:
    """Mock Prometheus metrics recorder."""
    mock = Mock()
    mock.record_calculation = Mock()
    mock.record_emissions = Mock()
    mock.record_product_count = Mock()
    mock.record_error = Mock()
    return mock


@pytest.fixture
def mock_provenance() -> Mock:
    """Mock provenance tracker."""
    mock = Mock()
    mock.generate_hash = Mock(return_value="a" * 64)
    mock.record_calculation = Mock()
    return mock


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_full_compliance_result(**overrides) -> Dict[str, Any]:
    """Build a fully compliant result dict with all required fields.

    Pass keyword overrides to modify or nullify specific fields.
    """
    base = {
        "total_co2e_kg": Decimal("41670000.0"),
        "method": "direct_fuel_combustion",
        "calculation_method": "direct_fuel_combustion",
        "ef_sources": ["defra", "epa"],
        "ef_source": "defra",
        "product_category": "vehicles",
        "emission_type": "direct",
        "boundary": "use_phase_only",
        "exclusions": "None - all use-phase emissions included",
        "dqi_score": Decimal("80.0"),
        "data_quality_score": Decimal("80.0"),
        "uncertainty_analysis": {"method": "propagation", "ci_95": [37503000, 45837000]},
        "uncertainty": {"method": "propagation"},
        "base_year": 2019,
        "methodology": "GHG Protocol Scope 3 Category 11",
        "targets": "Reduce use-phase emissions 30% by 2030",
        "reduction_targets": "30% by 2030",
        "actions": "Fleet efficiency improvement",
        "reduction_actions": "Efficiency programs",
        "verification_status": "limited_assurance",
        "verified": True,
        "assurance": "limited",
        "assurance_opinion": "limited_assurance",
        "reporting_period": "2024",
        "period": "2024",
        "gases_included": ["CO2", "CH4", "N2O"],
        "emission_gases": ["CO2", "CH4", "N2O"],
        "standards_used": ["GHG Protocol", "ISO 14064"],
        "standards": ["GHG Protocol"],
        "total_scope3_co2e": Decimal("120000000.0"),
        "product_count": 15,
        "by_category": {"vehicles": Decimal("41670000.0")},
        "by_emission_type": {"direct": Decimal("41670000.0")},
        "by_method": {"direct_fuel_combustion": Decimal("41670000.0")},
        "completeness_score": Decimal("85.0"),
        "method_coverage": Decimal("90.0"),
        "lifetime_assumptions_documented": True,
        "use_profiles_documented": True,
        "intermediate_only": False,
        "end_product_included": True,
        "allocation_method": "units_sold",
        "progress_tracking": {"2023": 44000000, "2024": 41670000},
        "year_over_year_change": Decimal("-5.3"),
        "target_coverage": "80%",
        "sbti_coverage": "80%",
    }
    base.update(overrides)
    return base
