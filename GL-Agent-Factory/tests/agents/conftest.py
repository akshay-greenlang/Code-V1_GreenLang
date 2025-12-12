"""
Shared Fixtures and Configuration for Agent Unit Tests

This module provides reusable fixtures for testing GreenLang agents,
including sample data generators, mock objects, and assertion helpers.

Usage:
    pytest tests/agents/ -v --cov=backend/agents --cov-report=html
"""

import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance benchmarks")
    config.addinivalue_line("markers", "compliance: Regulatory compliance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


# =============================================================================
# Common Assertion Helpers
# =============================================================================


def assert_valid_provenance_hash(hash_value: str) -> None:
    """Assert that a provenance hash is valid SHA-256 format."""
    assert hash_value is not None, "Provenance hash should not be None"
    assert len(hash_value) == 64, f"SHA-256 hash should be 64 chars, got {len(hash_value)}"
    assert all(c in "0123456789abcdef" for c in hash_value.lower()), (
        "Invalid hex characters in hash"
    )


def assert_provenance_deterministic(hash1: str, hash2: str) -> None:
    """Assert that two provenance hashes from same input are identical."""
    assert hash1 == hash2, (
        f"Provenance hashes should be deterministic. Got:\n"
        f"  Hash 1: {hash1}\n"
        f"  Hash 2: {hash2}"
    )


def assert_emissions_in_range(
    emissions: float,
    min_val: float,
    max_val: float,
    unit: str = "kgCO2e"
) -> None:
    """Assert emissions value is within expected range."""
    assert min_val <= emissions <= max_val, (
        f"Emissions {emissions} {unit} outside expected range [{min_val}, {max_val}]"
    )


def assert_calculation_accuracy(
    actual: float,
    expected: float,
    tolerance_pct: float = 0.001
) -> None:
    """Assert calculation matches expected value within tolerance."""
    if expected == 0:
        assert actual == 0, f"Expected 0, got {actual}"
    else:
        diff_pct = abs(actual - expected) / abs(expected) * 100
        assert diff_pct <= tolerance_pct, (
            f"Calculation {actual} differs from expected {expected} "
            f"by {diff_pct:.4f}% (tolerance: {tolerance_pct}%)"
        )


def assert_processing_time_within_target(
    processing_time_ms: float,
    target_ms: float = 100.0
) -> None:
    """Assert processing time meets performance target."""
    assert processing_time_ms <= target_ms, (
        f"Processing time {processing_time_ms:.2f}ms exceeds target {target_ms:.2f}ms"
    )


# =============================================================================
# Carbon Emissions Agent (GL-001) Fixtures
# =============================================================================


@pytest.fixture
def carbon_agent():
    """Create CarbonEmissionsAgent instance for testing."""
    from agents.gl_001_carbon_emissions.agent import CarbonEmissionsAgent
    return CarbonEmissionsAgent()


@pytest.fixture
def carbon_valid_input():
    """Create valid CarbonEmissionsInput for natural gas."""
    from agents.gl_001_carbon_emissions.agent import (
        CarbonEmissionsInput,
        FuelType,
        Scope,
    )
    return CarbonEmissionsInput(
        fuel_type=FuelType.NATURAL_GAS,
        quantity=1000.0,
        unit="m3",
        region="US",
        scope=Scope.SCOPE_1,
        calculation_method="location",
    )


@pytest.fixture
def carbon_diesel_input():
    """Create valid CarbonEmissionsInput for diesel."""
    from agents.gl_001_carbon_emissions.agent import (
        CarbonEmissionsInput,
        FuelType,
        Scope,
    )
    return CarbonEmissionsInput(
        fuel_type=FuelType.DIESEL,
        quantity=500.0,
        unit="L",
        region="US",
        scope=Scope.SCOPE_1,
    )


@pytest.fixture
def carbon_electricity_input():
    """Create valid CarbonEmissionsInput for electricity (Scope 2)."""
    from agents.gl_001_carbon_emissions.agent import (
        CarbonEmissionsInput,
        FuelType,
        Scope,
    )
    return CarbonEmissionsInput(
        fuel_type=FuelType.ELECTRICITY_GRID,
        quantity=10000.0,
        unit="kWh",
        region="US",
        scope=Scope.SCOPE_2,
        calculation_method="location",
    )


# =============================================================================
# CBAM Compliance Agent (GL-002) Fixtures
# =============================================================================


@pytest.fixture
def cbam_agent():
    """Create CBAMComplianceAgent instance for testing."""
    from agents.gl_002_cbam_compliance.agent import CBAMComplianceAgent
    return CBAMComplianceAgent()


@pytest.fixture
def cbam_steel_input():
    """Create valid CBAM input for steel import from China."""
    from agents.gl_002_cbam_compliance.agent import CBAMInput
    return CBAMInput(
        cn_code="72081000",
        quantity_tonnes=1000.0,
        country_of_origin="CN",
        installation_id="CN-STEEL-001",
        reporting_period="Q1 2026",
    )


@pytest.fixture
def cbam_aluminium_input():
    """Create valid CBAM input for aluminium import."""
    from agents.gl_002_cbam_compliance.agent import CBAMInput
    return CBAMInput(
        cn_code="76011000",
        quantity_tonnes=500.0,
        country_of_origin="CN",
        reporting_period="Q1 2026",
    )


@pytest.fixture
def cbam_cement_input():
    """Create valid CBAM input for cement import."""
    from agents.gl_002_cbam_compliance.agent import CBAMInput
    return CBAMInput(
        cn_code="25231000",
        quantity_tonnes=2000.0,
        country_of_origin="IN",
        reporting_period="Q2 2026",
    )


# =============================================================================
# SBTi Validation Agent (GL-010) Fixtures
# =============================================================================


@pytest.fixture
def sbti_agent():
    """Create SBTiValidationAgent instance for testing."""
    from agents.gl_010_sbti_validation.agent import SBTiValidationAgent
    return SBTiValidationAgent()


@pytest.fixture
def sbti_near_term_input():
    """Create valid SBTi input with near-term 1.5C aligned target."""
    from agents.gl_010_sbti_validation.agent import (
        SBTiInput,
        ScopeEmissions,
        TargetDefinition,
        TargetType,
        PathwayType,
        ScopeType,
    )
    return SBTiInput(
        company_id="COMPANY-001",
        company_name="Test Corporation",
        base_year=2019,
        base_year_emissions=ScopeEmissions(
            scope1=10000.0,
            scope2=5000.0,
            scope3=30000.0,
        ),
        targets=[
            TargetDefinition(
                target_id="NT-001",
                target_year=2030,
                target_type=TargetType.NEAR_TERM,
                reduction_pct=46.2,
                scopes_covered=[ScopeType.SCOPE_1, ScopeType.SCOPE_2],
                pathway_type=PathwayType.ACA,
            )
        ],
    )


@pytest.fixture
def sbti_insufficient_target_input():
    """Create SBTi input with insufficient reduction target."""
    from agents.gl_010_sbti_validation.agent import (
        SBTiInput,
        ScopeEmissions,
        TargetDefinition,
        TargetType,
    )
    return SBTiInput(
        company_id="COMPANY-002",
        base_year=2020,
        base_year_emissions=ScopeEmissions(
            scope1=5000.0,
            scope2=3000.0,
            scope3=20000.0,
        ),
        targets=[
            TargetDefinition(
                target_year=2030,
                target_type=TargetType.NEAR_TERM,
                reduction_pct=20.0,  # Insufficient for 1.5C
            )
        ],
    )


# =============================================================================
# Economizer Performance Agent (GL-020) Fixtures
# =============================================================================


@pytest.fixture
def economizer_agent():
    """Create EconomizerPerformanceAgent instance for testing."""
    from agents.gl_020_economizer_performance.agent import EconomizerPerformanceAgent
    return EconomizerPerformanceAgent()


@pytest.fixture
def economizer_valid_input():
    """Create valid economizer input with typical boiler conditions."""
    from agents.gl_020_economizer_performance.agent import (
        EconomizerInput,
        FlueGasComposition,
        WaterSideConditions,
        HeatExchangerGeometry,
        FlowArrangement,
    )
    return EconomizerInput(
        flue_gas=FlueGasComposition(
            temperature_in_celsius=350.0,
            temperature_out_celsius=150.0,
            mass_flow_kg_s=50.0,
            H2O_percent=8.0,
            SO3_ppmv=15.0,
            total_pressure_kPa=101.325,
        ),
        water_side=WaterSideConditions(
            inlet_temperature_celsius=105.0,
            outlet_temperature_celsius=180.0,
            mass_flow_kg_s=20.0,
            drum_pressure_MPa=4.0,
        ),
        heat_exchanger=HeatExchangerGeometry(
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
            tube_outer_diameter_mm=51.0,
            tube_wall_thickness_mm=4.0,
        ),
    )


@pytest.fixture
def economizer_steaming_risk_input():
    """Create input with steaming risk conditions."""
    from agents.gl_020_economizer_performance.agent import (
        EconomizerInput,
        FlueGasComposition,
        WaterSideConditions,
    )
    return EconomizerInput(
        flue_gas=FlueGasComposition(
            temperature_in_celsius=400.0,
            temperature_out_celsius=180.0,
            mass_flow_kg_s=60.0,
            H2O_percent=10.0,
            SO3_ppmv=20.0,
        ),
        water_side=WaterSideConditions(
            inlet_temperature_celsius=140.0,
            outlet_temperature_celsius=245.0,  # Very close to saturation at 4 MPa
            mass_flow_kg_s=15.0,
            drum_pressure_MPa=4.0,
        ),
    )


@pytest.fixture
def economizer_corrosion_risk_input():
    """Create input with cold-end corrosion risk conditions."""
    from agents.gl_020_economizer_performance.agent import (
        EconomizerInput,
        FlueGasComposition,
        WaterSideConditions,
    )
    return EconomizerInput(
        flue_gas=FlueGasComposition(
            temperature_in_celsius=300.0,
            temperature_out_celsius=110.0,  # Low outlet temp
            mass_flow_kg_s=40.0,
            H2O_percent=12.0,  # High moisture
            SO3_ppmv=50.0,  # High SO3
        ),
        water_side=WaterSideConditions(
            inlet_temperature_celsius=80.0,  # Low water inlet
            outlet_temperature_celsius=150.0,
            mass_flow_kg_s=25.0,
            drum_pressure_MPa=3.0,
        ),
    )


# =============================================================================
# Burner Maintenance Agent (GL-021) Fixtures
# =============================================================================


@pytest.fixture
def burner_agent():
    """Create BurnerMaintenancePredictorAgent instance for testing."""
    from agents.gl_021_burner_maintenance.agent import (
        BurnerMaintenancePredictorAgent,
        AgentConfig,
    )
    return BurnerMaintenancePredictorAgent(AgentConfig())


@pytest.fixture
def burner_valid_input():
    """Create valid burner maintenance input for healthy burner."""
    from agents.gl_021_burner_maintenance.agent import (
        BurnerInput,
        WeibullParameters,
        FlameQualityMetrics,
        FuelType,
    )
    return BurnerInput(
        burner_id="BRN-001",
        burner_model="ACME-5000",
        fuel_type=FuelType.NATURAL_GAS,
        operating_hours=15000.0,
        design_life_hours=50000.0,
        cycles_count=2000,
        operating_hours_per_day=20.0,
        weibull_params=WeibullParameters(beta=2.5, eta=40000.0),
        flame_metrics=FlameQualityMetrics(
            flame_temperature_c=1200.0,
            stability_index=0.95,
            o2_percent=3.5,
            co_ppm=50.0,
            nox_ppm=80.0,
        ),
        installation_date=date(2020, 1, 15),
        health_history=[95.0, 92.0, 89.0, 86.0, 83.0],
        health_history_interval_hours=3000.0,
    )


@pytest.fixture
def burner_degraded_input():
    """Create input for degraded burner needing maintenance."""
    from agents.gl_021_burner_maintenance.agent import (
        BurnerInput,
        WeibullParameters,
        FlameQualityMetrics,
        FuelType,
    )
    return BurnerInput(
        burner_id="BRN-002",
        burner_model="ACME-5000",
        fuel_type=FuelType.NATURAL_GAS,
        operating_hours=35000.0,
        design_life_hours=50000.0,
        cycles_count=8000,
        operating_hours_per_day=22.0,
        weibull_params=WeibullParameters(beta=2.5, eta=40000.0),
        flame_metrics=FlameQualityMetrics(
            flame_temperature_c=1100.0,  # Lower than optimal
            stability_index=0.75,  # Reduced stability
            o2_percent=5.5,  # High excess air
            co_ppm=200.0,  # Elevated CO
            nox_ppm=120.0,
        ),
        installation_date=date(2018, 6, 1),
        health_history=[80.0, 70.0, 60.0, 50.0, 45.0],
        health_history_interval_hours=3000.0,
    )


@pytest.fixture
def burner_critical_input():
    """Create input for critical burner requiring replacement."""
    from agents.gl_021_burner_maintenance.agent import (
        BurnerInput,
        WeibullParameters,
        FlameQualityMetrics,
        FuelType,
    )
    return BurnerInput(
        burner_id="BRN-003",
        burner_model="ACME-3000",
        fuel_type=FuelType.FUEL_OIL_2,
        operating_hours=45000.0,
        design_life_hours=50000.0,
        cycles_count=15000,
        operating_hours_per_day=24.0,
        weibull_params=WeibullParameters(beta=3.0, eta=35000.0),
        flame_metrics=FlameQualityMetrics(
            flame_temperature_c=950.0,
            stability_index=0.55,  # Very unstable
            o2_percent=7.0,  # Very high excess air
            co_ppm=500.0,  # High CO
            nox_ppm=150.0,
        ),
        installation_date=date(2015, 3, 10),
        health_history=[60.0, 45.0, 35.0, 28.0, 22.0],
        health_history_interval_hours=3000.0,
        repair_cost_ratio=0.6,
    )


# =============================================================================
# Weibull Calculator Fixtures (for GL-021)
# =============================================================================


@pytest.fixture
def weibull_params_typical():
    """Typical Weibull parameters for industrial burner."""
    return {"beta": 2.5, "eta": 40000.0}


@pytest.fixture
def weibull_params_infant_mortality():
    """Weibull parameters indicating infant mortality failures."""
    return {"beta": 0.7, "eta": 50000.0}


@pytest.fixture
def weibull_params_random():
    """Weibull parameters indicating random failures (exponential)."""
    return {"beta": 1.0, "eta": 30000.0}


@pytest.fixture
def weibull_params_wearout():
    """Weibull parameters indicating wear-out failures."""
    return {"beta": 3.5, "eta": 35000.0}


# =============================================================================
# Test Data Generators
# =============================================================================


class TestDataGenerator:
    """Generate test data for various agent scenarios."""

    @staticmethod
    def generate_emission_scenarios(num_scenarios: int = 10) -> List[Dict[str, Any]]:
        """Generate multiple emission calculation scenarios."""
        import random

        fuel_types = ["natural_gas", "diesel", "gasoline", "electricity_grid"]
        regions = ["US", "EU", "DE", "FR", "CN"]

        scenarios = []
        for i in range(num_scenarios):
            scenarios.append({
                "fuel_type": random.choice(fuel_types),
                "quantity": random.uniform(100, 10000),
                "region": random.choice(regions),
                "expected_ef_range": (0.05, 10.0),  # kgCO2e per unit
            })
        return scenarios

    @staticmethod
    def generate_cbam_shipments(num_shipments: int = 10) -> List[Dict[str, Any]]:
        """Generate CBAM shipment test data."""
        import random

        products = [
            {"cn_prefix": "7208", "category": "iron_steel"},
            {"cn_prefix": "7601", "category": "aluminium"},
            {"cn_prefix": "2523", "category": "cement"},
            {"cn_prefix": "3102", "category": "fertilizers"},
        ]
        countries = ["CN", "IN", "RU", "TR", "UA"]

        shipments = []
        for i in range(num_shipments):
            product = random.choice(products)
            shipments.append({
                "cn_code": f"{product['cn_prefix']}1000",
                "quantity_tonnes": random.uniform(100, 5000),
                "country_of_origin": random.choice(countries),
                "expected_category": product["category"],
            })
        return shipments

    @staticmethod
    def generate_burner_health_history(
        initial_health: float = 100.0,
        degradation_rate: float = 2.0,
        num_readings: int = 10
    ) -> List[float]:
        """Generate realistic burner health history."""
        import random

        history = []
        health = initial_health
        for _ in range(num_readings):
            # Add some noise to degradation
            noise = random.uniform(-0.5, 0.5)
            health = max(0, health - degradation_rate + noise)
            history.append(round(health, 1))
        return history


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


# =============================================================================
# Mock Services
# =============================================================================


@pytest.fixture
def mock_emission_factor_db():
    """Mock emission factor database lookup."""
    factors = {
        ("natural_gas", "US"): {"value": 1.93, "unit": "kgCO2e/m3", "source": "EPA"},
        ("natural_gas", "EU"): {"value": 2.02, "unit": "kgCO2e/m3", "source": "DEFRA"},
        ("diesel", "US"): {"value": 2.68, "unit": "kgCO2e/L", "source": "EPA"},
        ("diesel", "EU"): {"value": 2.62, "unit": "kgCO2e/L", "source": "DEFRA"},
        ("electricity_grid", "US"): {"value": 0.417, "unit": "kgCO2e/kWh", "source": "EPA eGRID"},
        ("electricity_grid", "EU"): {"value": 0.276, "unit": "kgCO2e/kWh", "source": "IEA"},
    }
    return factors


@pytest.fixture
def mock_carbon_price():
    """Mock EU ETS carbon price."""
    return 85.0  # EUR/tCO2


# =============================================================================
# Performance Testing Helpers
# =============================================================================


@pytest.fixture
def performance_timer():
    """Timer for performance measurements."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        @property
        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return None

    return Timer()


# =============================================================================
# Known Values for Validation
# =============================================================================


# Known emission factor values for validation testing
KNOWN_EMISSION_FACTORS = {
    "natural_gas_US_m3": 1.93,  # kgCO2e/m3
    "diesel_US_L": 2.68,  # kgCO2e/L
    "electricity_US_kWh": 0.417,  # kgCO2e/kWh
}

# Known Weibull calculation results for validation
KNOWN_WEIBULL_VALUES = {
    # R(t=10000, beta=2.5, eta=40000) = 0.9851...
    "reliability_10k_2.5_40k": 0.9851,
    # MTTF(beta=2.5, eta=40000) = 35437.6...
    "mttf_2.5_40k": 35437.6,
}

# Known Verhoff-Banchero acid dew point results
KNOWN_ACID_DEW_POINTS = {
    # T_dew at P_H2O=0.08 atm, P_SO3=15e-6 atm
    "h2o_0.08_so3_15ppm": 127.5,  # deg C (approximate)
}


@pytest.fixture
def known_values():
    """Provide known calculation values for validation."""
    return {
        "emission_factors": KNOWN_EMISSION_FACTORS,
        "weibull": KNOWN_WEIBULL_VALUES,
        "acid_dew_point": KNOWN_ACID_DEW_POINTS,
    }
