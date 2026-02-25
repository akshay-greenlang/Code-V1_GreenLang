"""Shared fixtures for AGENT-MRV-012 Cooling Purchase Agent tests."""

import pytest
import sys
from decimal import Decimal
from typing import Dict, Any


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all singletons before each test."""
    modules = [
        "greenlang.cooling_purchase.config",
        "greenlang.cooling_purchase.metrics",
        "greenlang.cooling_purchase.provenance",
        "greenlang.cooling_purchase.cooling_database",
        "greenlang.cooling_purchase.electric_chiller_calculator",
        "greenlang.cooling_purchase.absorption_cooling_calculator",
        "greenlang.cooling_purchase.district_cooling_calculator",
        "greenlang.cooling_purchase.uncertainty_quantifier",
        "greenlang.cooling_purchase.compliance_checker",
        "greenlang.cooling_purchase.cooling_purchase_pipeline",
        "greenlang.cooling_purchase.setup",
    ]
    for mod_name in modules:
        try:
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, "reset"):
                mod.reset()
        except Exception:
            pass
    yield


@pytest.fixture
def sample_electric_request() -> Dict[str, Any]:
    """Return dict with valid ElectricChillerRequest params."""
    return {
        "calculation_id": "TEST-EC-001",
        "organization_id": "ORG-001",
        "facility_id": "FAC-001",
        "reporting_period_start": "2024-01-01",
        "reporting_period_end": "2024-12-31",
        "technology": "ELECTRIC_CHILLER",
        "cooling_load_mwh": Decimal("1000.0"),
        "compressor_type": "CENTRIFUGAL",
        "condenser_type": "WATER_COOLED",
        "refrigerant": "R134A",
        "rated_cop": Decimal("5.5"),
        "electricity_source": "GRID",
        "electricity_ef_kg_co2e_kwh": Decimal("0.45"),
        "ambient_temp_c": Decimal("30.0"),
        "include_refrigerant_leakage": True,
        "annual_leakage_rate_pct": Decimal("2.0"),
        "refrigerant_charge_kg": Decimal("100.0"),
    }


@pytest.fixture
def sample_absorption_request() -> Dict[str, Any]:
    """Return dict with valid AbsorptionCoolingRequest params."""
    return {
        "calculation_id": "TEST-AC-001",
        "organization_id": "ORG-001",
        "facility_id": "FAC-001",
        "reporting_period_start": "2024-01-01",
        "reporting_period_end": "2024-12-31",
        "technology": "ABSORPTION_CHILLER",
        "cooling_load_mwh": Decimal("500.0"),
        "absorption_type": "SINGLE_EFFECT",
        "heat_source": "NATURAL_GAS",
        "rated_thermal_cop": Decimal("0.7"),
        "heat_input_mwh": Decimal("714.3"),
        "heat_source_ef_kg_co2e_kwh": Decimal("0.18"),
        "include_parasitic_electricity": True,
        "parasitic_electricity_mwh": Decimal("50.0"),
        "electricity_ef_kg_co2e_kwh": Decimal("0.45"),
    }


@pytest.fixture
def sample_district_request() -> Dict[str, Any]:
    """Return dict with valid DistrictCoolingRequest params."""
    return {
        "calculation_id": "TEST-DC-001",
        "organization_id": "ORG-001",
        "facility_id": "FAC-001",
        "reporting_period_start": "2024-01-01",
        "reporting_period_end": "2024-12-31",
        "technology": "DISTRICT_COOLING",
        "cooling_purchased_mwh": Decimal("2000.0"),
        "provider_name": "City District Cooling",
        "provider_emission_factor_kg_co2e_kwh": Decimal("0.12"),
        "distribution_loss_pct": Decimal("5.0"),
        "include_transmission_losses": True,
    }


@pytest.fixture
def sample_calculation_result() -> Dict[str, Any]:
    """Return a mock CalculationResult dict."""
    return {
        "calculation_id": "TEST-001",
        "technology": "ELECTRIC_CHILLER",
        "cooling_output_mwh": Decimal("1000.0"),
        "total_emissions_kg_co2e": Decimal("450000.0"),
        "co2_emissions_kg": Decimal("450000.0"),
        "ch4_emissions_kg": Decimal("0.0"),
        "n2o_emissions_kg": Decimal("0.0"),
        "refrigerant_emissions_kg_co2e": Decimal("0.0"),
        "energy_input_mwh": Decimal("181.82"),
        "cop_actual": Decimal("5.5"),
        "emission_factor_kg_co2e_kwh": Decimal("0.45"),
        "calculation_method": "AHRI_PART_LOAD",
        "tier": "TIER_2",
        "uncertainty_pct": Decimal("15.0"),
        "confidence_level": "MEDIUM",
        "data_quality_score": Decimal("75.0"),
        "metadata": {},
    }


@pytest.fixture
def sample_uncertainty_params() -> Dict[str, Any]:
    """Return dict with valid UncertaintyQuantificationRequest params."""
    return {
        "calculation_id": "TEST-UNC-001",
        "technology": "ELECTRIC_CHILLER",
        "cooling_load_mwh": Decimal("1000.0"),
        "cop_uncertainty_pct": Decimal("5.0"),
        "ef_uncertainty_pct": Decimal("10.0"),
        "leakage_uncertainty_pct": Decimal("20.0"),
        "data_quality_tier": "TIER_2",
        "measurement_accuracy_pct": Decimal("2.0"),
    }


@pytest.fixture
def sample_compliance_params() -> Dict[str, Any]:
    """Return dict with valid ComplianceCheckRequest params."""
    return {
        "calculation_id": "TEST-COMP-001",
        "organization_id": "ORG-001",
        "frameworks": ["GHG_PROTOCOL", "ISO_14064"],
        "technology": "ELECTRIC_CHILLER",
        "cooling_load_mwh": Decimal("1000.0"),
        "total_emissions_kg_co2e": Decimal("450000.0"),
        "calculation_method": "AHRI_PART_LOAD",
        "tier": "TIER_2",
        "has_third_party_verification": False,
    }
