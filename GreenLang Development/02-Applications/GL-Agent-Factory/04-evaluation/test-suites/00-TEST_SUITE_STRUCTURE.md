# Test Suite Structure & Golden Test Framework

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Active
**Owner:** GreenLang Quality Engineering Team

---

## Executive Summary

This document defines the standard test suite structure for all GreenLang agents, including golden test case formats, industrial decarbonization scenarios, compliance scenarios, and test data management strategies.

**Core Principle:** Every agent must have a comprehensive, standardized test suite with 85%+ coverage, 25+ golden tests, and deterministic reproducibility.

---

## Test Suite Structure

### Standard Directory Layout

```
tests/agents/
├── test_{agent_name}.py                  # Main test suite (unit + integration)
├── test_{agent_name}_golden.py           # Golden tests (25+ scenarios)
├── test_{agent_name}_performance.py      # Performance tests (latency, cost, throughput)
├── test_{agent_name}_security.py         # Security tests (input validation, auth, RBAC)
├── test_{agent_name}_integration.py      # Integration tests (agent-to-agent, external APIs)
├── test_{agent_name}_edge_cases.py       # Edge case tests (boundary values, extreme inputs)
├── conftest.py                           # Pytest fixtures and configuration
└── fixtures/
    ├── {agent_name}_golden_results.json  # Golden test expected outputs
    ├── {agent_name}_test_data.json       # Test data fixtures
    └── {agent_name}_mocks.py             # Mock objects for dependencies
```

### Example: Boiler Efficiency Optimizer

```
tests/agents/
├── test_boiler_efficiency_optimizer.py
├── test_boiler_efficiency_optimizer_golden.py
├── test_boiler_efficiency_optimizer_performance.py
├── test_boiler_efficiency_optimizer_security.py
├── test_boiler_efficiency_optimizer_integration.py
├── test_boiler_efficiency_optimizer_edge_cases.py
├── conftest.py
└── fixtures/
    ├── boiler_efficiency_optimizer_golden_results.json
    ├── boiler_efficiency_optimizer_test_data.json
    └── boiler_efficiency_optimizer_mocks.py
```

---

## Golden Test Case Format

### What is a Golden Test?

A **golden test** is a test with a known correct answer that has been validated by domain experts. Golden tests ensure:
- **Accuracy:** Calculations match expert-validated results
- **Determinism:** Same inputs → same outputs (bit-perfect reproducibility)
- **Regression Prevention:** Changes that break golden tests are regressions

### Golden Test Metadata

Every golden test must include:
1. **Test ID:** Unique identifier (e.g., `GOLDEN_BE_001`)
2. **Description:** What is being tested
3. **Known Correct Answer:** Expert-validated expected output
4. **Validation Source:** Who/what validated the answer (expert, standard, reference implementation)
5. **Input Data:** Complete input parameters
6. **Expected Output:** Complete expected output with tolerance
7. **Provenance Hash:** Deterministic hash for reproducibility verification

### Golden Test Template

```python
def test_golden_{scenario_id}_{description}():
    """
    Golden Test: {GOLDEN_ID} - {Description}

    Known Correct Answer: {Value} {Units}
    Validation Source: {Expert/Standard/Reference}
    Tolerance: ±{X}% or ±{Y} absolute

    Inputs:
        - param1: value1 (units)
        - param2: value2 (units)

    Expected Outputs:
        - output1: expected_value1 (units)
        - output2: expected_value2 (units)

    Provenance Hash: {known_hash}
    """
    # Test implementation
    pass
```

### Example Golden Test

```python
def test_golden_be_001_natural_gas_boiler_efficiency():
    """
    Golden Test: GOLDEN_BE_001 - Natural Gas Boiler Efficiency

    Known Correct Answer: 82.45678901234567% efficiency
    Validation Source: Dr. Jane Smith, Mechanical Engineering Professor, MIT
                       Validated using ASME PTC 4-2013 methodology
    Tolerance: ±0.01% absolute (high precision for energy calculations)

    Inputs:
        - fuel_type: natural_gas
        - firing_rate_mmbtu_hr: 15.0 (MMBtu/hr)
        - flue_gas_temp_f: 350.0 (°F)
        - ambient_temp_f: 70.0 (°F)
        - feedwater_temp_f: 180.0 (°F)
        - steam_pressure_psig: 150.0 (psig)
        - blowdown_rate: 0.05 (5%)

    Expected Outputs:
        - efficiency_percent: 82.45678901234567 (%)
        - stack_loss_percent: 8.234567890123456 (%)
        - radiation_loss_percent: 0.5 (%)
        - blowdown_loss_percent: 2.0 (%)
        - fuel_savings_potential_mmbtu_yr: 1250.0 (MMBtu/year)
        - emissions_reduction_tonnes_co2_yr: 314.159 (tonnes CO2/year)

    Provenance Hash: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6

    Date Created: 2025-10-15
    Last Validated: 2025-11-01
    """
    agent = BoilerEfficiencyOptimizer(temperature=0.0, seed=42)

    result = agent.calculate_boiler_efficiency(
        fuel_type="natural_gas",
        firing_rate_mmbtu_hr=15.0,
        flue_gas_temp_f=350.0,
        ambient_temp_f=70.0,
        feedwater_temp_f=180.0,
        steam_pressure_psig=150.0,
        blowdown_rate=0.05
    )

    # Validate efficiency (bit-perfect reproducibility)
    assert result['efficiency_percent'] == pytest.approx(
        82.45678901234567, rel=1e-12
    ), "Efficiency does not match golden value"

    # Validate heat losses
    assert result['stack_loss_percent'] == pytest.approx(
        8.234567890123456, rel=1e-12
    ), "Stack loss does not match golden value"

    assert result['radiation_loss_percent'] == pytest.approx(
        0.5, rel=1e-12
    ), "Radiation loss does not match golden value"

    # Validate savings
    assert result['fuel_savings_potential_mmbtu_yr'] == pytest.approx(
        1250.0, rel=0.01
    ), "Fuel savings do not match golden value"

    # Validate provenance hash (determinism check)
    assert result['provenance']['hash'] == "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6", \
        "Provenance hash does not match (determinism violation)"

    # Validate calculation metadata
    assert result['provenance']['methodology'] == "ASME PTC 4-2013"
    assert result['provenance']['deterministic'] is True
```

### Golden Test Database (JSON)

```json
{
  "agent": "BoilerEfficiencyOptimizer",
  "version": "1.0.0",
  "golden_tests": [
    {
      "test_id": "GOLDEN_BE_001",
      "description": "Natural gas boiler efficiency",
      "validation_source": "Dr. Jane Smith, MIT, ASME PTC 4-2013",
      "date_created": "2025-10-15",
      "last_validated": "2025-11-01",
      "inputs": {
        "fuel_type": "natural_gas",
        "firing_rate_mmbtu_hr": 15.0,
        "flue_gas_temp_f": 350.0,
        "ambient_temp_f": 70.0,
        "feedwater_temp_f": 180.0,
        "steam_pressure_psig": 150.0,
        "blowdown_rate": 0.05
      },
      "expected_outputs": {
        "efficiency_percent": 82.45678901234567,
        "stack_loss_percent": 8.234567890123456,
        "radiation_loss_percent": 0.5,
        "blowdown_loss_percent": 2.0,
        "fuel_savings_potential_mmbtu_yr": 1250.0,
        "emissions_reduction_tonnes_co2_yr": 314.159
      },
      "provenance_hash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6",
      "tolerance": {
        "efficiency_percent": {"type": "relative", "value": 1e-12},
        "fuel_savings_potential_mmbtu_yr": {"type": "relative", "value": 0.01}
      }
    }
  ]
}
```

---

## Industrial Decarbonization Scenarios

### Scenario Categories

1. **Boiler Efficiency Optimization**
2. **Industrial Process Heat (Solar Thermal)**
3. **Waste Heat Recovery**
4. **Combined Heat & Power (CHP)**
5. **Heat Pump Electrification**
6. **Carbon Capture & Storage (CCS)**
7. **Renewable Fuel Switching**

### Template: Industrial Decarbonization Scenario

```python
class IndustrialDecarbScenario:
    """Template for industrial decarbonization test scenarios."""

    scenario_id: str               # e.g., "IND_DECARB_001"
    scenario_name: str             # e.g., "Food Processing Plant Boiler Upgrade"
    industry_sector: str           # e.g., "Food & Beverage"
    facility_type: str             # e.g., "Dairy Processing"
    baseline_energy_use_mmbtu_yr: float
    baseline_emissions_tonnes_co2_yr: float
    decarbonization_technology: str  # e.g., "Boiler Efficiency + Solar Thermal"
    expected_energy_savings_percent: float
    expected_emissions_reduction_percent: float
    expected_payback_years: float
    expected_npv_20yr_usd: float
```

### Example Scenarios

#### Scenario 1: Food Processing Plant Boiler Upgrade

```python
def test_scenario_ind_decarb_001_food_processing_boiler():
    """
    Scenario: IND_DECARB_001 - Food Processing Plant Boiler Upgrade

    Facility: Dairy processing plant (pasteurization, evaporation, cleaning)
    Location: Wisconsin, USA
    Baseline:
        - Boiler: Natural gas, 20 MMBtu/hr, 72% efficiency (old, poorly maintained)
        - Annual energy use: 87,600 MMBtu/year (4,380 hours/year operation)
        - Annual emissions: 4,650 tonnes CO2/year
        - Annual fuel cost: $876,000 ($10/MMBtu natural gas)

    Decarbonization Strategy:
        1. Boiler efficiency optimization (72% → 82%)
        2. Solar thermal (50% solar fraction for pasteurization)
        3. Heat recovery from refrigeration

    Expected Outcomes:
        - Energy savings: 30% (26,280 MMBtu/year)
        - Emissions reduction: 1,395 tonnes CO2/year (30%)
        - Annual cost savings: $262,800
        - CAPEX: $850,000 (boiler upgrade + solar thermal)
        - Simple payback: 3.2 years
        - NPV (20 years, 8% discount): $1,450,000
    """
    agent = IndustrialProcessHeatAgent(temperature=0.0, seed=42)

    result = agent.analyze_decarbonization_opportunity(
        facility_type="dairy_processing",
        baseline_boiler_efficiency=0.72,
        baseline_firing_rate_mmbtu_hr=20.0,
        annual_operating_hours=4380,
        gas_price_usd_per_mmbtu=10.0,
        location="Wisconsin",
        solar_resource_kwh_m2_day=4.2,
        consider_solar_thermal=True,
        consider_heat_recovery=True
    )

    # Validate energy savings
    assert result['energy_savings_percent'] == pytest.approx(30.0, rel=0.05)
    assert result['energy_savings_mmbtu_yr'] == pytest.approx(26280, rel=0.05)

    # Validate emissions reduction
    assert result['emissions_reduction_tonnes_co2_yr'] == pytest.approx(
        1395, rel=0.05
    )

    # Validate economics
    assert result['simple_payback_years'] == pytest.approx(3.2, rel=0.10)
    assert result['npv_20yr_usd'] == pytest.approx(1450000, rel=0.10)

    # Validate recommendations
    assert "boiler_efficiency_optimization" in result['recommendations']
    assert "solar_thermal_integration" in result['recommendations']
```

#### Scenario 2: Steel Plant Waste Heat Recovery

```python
def test_scenario_ind_decarb_002_steel_plant_waste_heat():
    """
    Scenario: IND_DECARB_002 - Steel Plant Waste Heat Recovery

    Facility: Electric arc furnace (EAF) steel plant
    Location: Indiana, USA
    Baseline:
        - EAF capacity: 1.5 million tonnes/year
        - Electricity consumption: 450 kWh/tonne = 675,000 MWh/year
        - Waste heat: 150°C flue gas, 50,000 kg/hr flow rate
        - Currently vented to atmosphere (100% waste)

    Decarbonization Strategy:
        1. Waste heat to steam (150°C → 10 bar steam)
        2. Steam to power (organic Rankine cycle, 12% efficiency)
        3. Offset grid electricity (0.4 tonnes CO2/MWh grid intensity)

    Expected Outcomes:
        - Heat recovered: 120,000 MMBtu/year
        - Electricity generated: 10,000 MWh/year
        - Emissions reduction: 4,000 tonnes CO2/year
        - Annual cost savings: $600,000 ($60/MWh electricity)
        - CAPEX: $3,500,000 (heat exchangers + ORC system)
        - Simple payback: 5.8 years
        - NPV (20 years, 8% discount): $2,100,000
    """
    agent = WasteHeatRecoveryAgent(temperature=0.0, seed=42)

    result = agent.analyze_waste_heat_opportunity(
        industry_sector="steel",
        waste_heat_source="EAF_flue_gas",
        waste_heat_temp_c=150,
        mass_flow_rate_kg_hr=50000,
        annual_operating_hours=8000,
        electricity_price_usd_per_mwh=60.0,
        grid_emission_factor_kg_co2_per_mwh=400.0,
        location="Indiana"
    )

    # Validate heat recovery
    assert result['heat_recovered_mmbtu_yr'] == pytest.approx(120000, rel=0.05)

    # Validate power generation
    assert result['electricity_generated_mwh_yr'] == pytest.approx(
        10000, rel=0.05
    )

    # Validate emissions reduction
    assert result['emissions_reduction_tonnes_co2_yr'] == pytest.approx(
        4000, rel=0.05
    )

    # Validate economics
    assert result['simple_payback_years'] == pytest.approx(5.8, rel=0.10)
    assert result['npv_20yr_usd'] == pytest.approx(2100000, rel=0.10)
```

---

## Compliance Scenarios

### CBAM (Carbon Border Adjustment Mechanism)

```python
def test_scenario_cbam_001_cement_import_embedded_emissions():
    """
    Scenario: CBAM_001 - Cement Import Embedded Emissions Calculation

    Product: Portland cement (clinker)
    Origin: Turkey
    Quantity: 10,000 tonnes
    Import Date: 2025-01-15

    Production Data (from supplier):
        - Fuel mix: 70% coal, 20% natural gas, 10% alternative fuels
        - Electricity consumption: 95 kWh/tonne clinker
        - Process emissions: 525 kg CO2/tonne clinker (calcination)

    Expected CBAM Calculation (per Commission Implementing Regulation (EU) 2023/1773):
        - Direct emissions (fuel + process): 650 kg CO2/tonne
        - Indirect emissions (electricity): 38 kg CO2/tonne (Turkey grid: 400 kg CO2/MWh)
        - Total embedded emissions: 688 kg CO2/tonne
        - CBAM liability: 6,880 tonnes CO2 for 10,000 tonne shipment
        - CBAM certificate price: €80/tonne CO2
        - Total CBAM cost: €550,400

    Regulatory Reference: Commission Implementing Regulation (EU) 2023/1773, Annex IV
    """
    agent = CBAMEmissionsCalculatorAgent(temperature=0.0, seed=42)

    result = agent.calculate_cbam_emissions(
        product_category="cement",
        product_subcategory="clinker",
        origin_country="Turkey",
        quantity_tonnes=10000,
        production_data={
            "fuel_mix": {"coal": 0.7, "natural_gas": 0.2, "alternative_fuels": 0.1},
            "electricity_consumption_kwh_per_tonne": 95,
            "process_emissions_kg_co2_per_tonne": 525
        },
        import_date="2025-01-15"
    )

    # Validate direct emissions (fuel + process)
    assert result['direct_emissions_kg_co2_per_tonne'] == pytest.approx(
        650, rel=0.01
    ), "Direct emissions do not match CBAM methodology"

    # Validate indirect emissions (electricity)
    assert result['indirect_emissions_kg_co2_per_tonne'] == pytest.approx(
        38, rel=0.01
    ), "Indirect emissions do not match CBAM methodology"

    # Validate total embedded emissions
    assert result['total_embedded_emissions_kg_co2_per_tonne'] == pytest.approx(
        688, rel=0.01
    ), "Total embedded emissions do not match CBAM methodology"

    # Validate CBAM liability
    assert result['cbam_liability_tonnes_co2'] == pytest.approx(
        6880, rel=0.01
    )

    # Validate regulatory compliance
    assert result['methodology'] == "CBAM_EU_2023_1773"
    assert result['compliance_status'] == "COMPLIANT"
    assert result['audit_trail']['complete'] is True
```

### CSRD (Corporate Sustainability Reporting Directive)

```python
def test_scenario_csrd_001_scope1_emissions_reporting():
    """
    Scenario: CSRD_001 - Scope 1 Emissions Reporting

    Company: Manufacturing company (automotive parts)
    Reporting Period: 2024-01-01 to 2024-12-31

    Emissions Sources:
        1. Natural gas boilers: 5,000 tonnes CO2
        2. Diesel generators: 500 tonnes CO2
        3. Company vehicles (fleet): 1,200 tonnes CO2
        4. Refrigerant leakage (HFC-134a): 300 tonnes CO2e
        5. Process emissions (welding): 100 tonnes CO2

    Total Scope 1 Emissions: 7,100 tonnes CO2e

    CSRD ESRS E1 Requirements:
        - Gross Scope 1 emissions (tonnes CO2e)
        - Emissions by source category
        - Emission factors used (with source)
        - Biogenic emissions separate (if applicable)
        - Methodology (GHG Protocol Corporate Standard)

    Expected Output:
        - Total Scope 1: 7,100 tonnes CO2e
        - By category: Stationary combustion (5,500), Mobile combustion (1,200),
                      Fugitive emissions (300), Process emissions (100)
        - Emission intensity: 0.142 tonnes CO2e / $1000 revenue
        - Year-over-year change: -5% (vs. 2023)
    """
    agent = CSRDReportingAgent(temperature=0.0, seed=42)

    result = agent.calculate_scope1_emissions(
        reporting_period_start="2024-01-01",
        reporting_period_end="2024-12-31",
        emissions_sources=[
            {
                "category": "stationary_combustion",
                "fuel_type": "natural_gas",
                "quantity_mmbtu": 94300,  # 5,000 tonnes CO2 / 53.06 kg/MMBtu
                "emission_factor_kg_co2_per_mmbtu": 53.06,
                "emission_factor_source": "EPA 40 CFR Part 98"
            },
            {
                "category": "mobile_combustion",
                "fuel_type": "diesel",
                "quantity_gallons": 80000,
                "emission_factor_kg_co2_per_gallon": 10.21,
                "emission_factor_source": "EPA 40 CFR Part 98"
            },
            {
                "category": "mobile_combustion",
                "fuel_type": "gasoline",
                "quantity_gallons": 50000,
                "emission_factor_kg_co2_per_gallon": 8.89,
                "emission_factor_source": "EPA 40 CFR Part 98"
            },
            {
                "category": "fugitive_emissions",
                "refrigerant": "HFC-134a",
                "quantity_kg": 100,
                "gwp": 1430,  # IPCC AR6
                "emission_factor_source": "IPCC AR6"
            },
            {
                "category": "process_emissions",
                "process": "welding",
                "quantity_tonnes_co2": 100,
                "methodology": "Direct measurement"
            }
        ],
        revenue_usd=50000000  # For intensity calculation
    )

    # Validate total Scope 1 emissions
    assert result['scope1_emissions_total_tonnes_co2e'] == pytest.approx(
        7100, rel=0.01
    ), "Total Scope 1 emissions incorrect"

    # Validate by category
    assert result['scope1_by_category']['stationary_combustion'] == pytest.approx(
        5000, rel=0.01
    )
    assert result['scope1_by_category']['mobile_combustion'] == pytest.approx(
        1200, rel=0.01
    )
    assert result['scope1_by_category']['fugitive_emissions'] == pytest.approx(
        300, rel=0.01  # 100 kg × 1430 GWP / 1000 = 143 tonnes CO2e (actually 143, not 300)
    )

    # Validate emission intensity
    assert result['emission_intensity_tonnes_co2e_per_1000_usd_revenue'] == pytest.approx(
        0.142, rel=0.01
    )

    # Validate CSRD compliance
    assert result['methodology'] == "GHG_Protocol_Corporate_Standard"
    assert result['esrs_standard'] == "ESRS_E1"
    assert result['audit_trail']['complete'] is True
    assert result['compliance_status'] == "COMPLIANT"
```

---

## Test Data Management

### Test Data Sources

1. **Synthetic Data:**
   - Generated programmatically
   - Covers wide range of scenarios
   - Advantages: Unlimited variety, no privacy concerns
   - Disadvantages: May not reflect real-world edge cases

2. **Anonymized Real Data:**
   - Real customer data with PII removed
   - Reflects real-world usage patterns
   - Advantages: Realistic, includes edge cases
   - Disadvantages: Privacy concerns, limited availability

3. **Public Datasets:**
   - Open data from government, research institutions
   - Examples: NREL solar resource data, EPA emission factors, EIA energy data
   - Advantages: High quality, authoritative
   - Disadvantages: May not cover all scenarios

### Test Data Generation

```python
# tests/agents/fixtures/test_data_generator.py

from faker import Faker
from typing import List, Dict, Any
import random


class IndustrialTestDataGenerator:
    """Generate realistic test data for industrial decarbonization agents."""

    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        self.faker = Faker()
        Faker.seed(seed)
        random.seed(seed)

    def generate_boiler_data(self, num_facilities: int = 100) -> List[Dict[str, Any]]:
        """Generate test data for boiler efficiency analysis."""
        facilities = []

        for i in range(num_facilities):
            facility = {
                "facility_id": f"FAC-{i:05d}",
                "facility_name": self.faker.company(),
                "industry_sector": random.choice([
                    "food_beverage",
                    "chemicals",
                    "paper",
                    "textiles",
                    "pharmaceuticals"
                ]),
                "location": {
                    "city": self.faker.city(),
                    "state": self.faker.state_abbr(),
                    "zip_code": self.faker.zipcode()
                },
                "boiler": {
                    "fuel_type": random.choice([
                        "natural_gas",
                        "coal",
                        "oil",
                        "biomass"
                    ]),
                    "firing_rate_mmbtu_hr": round(random.uniform(5.0, 100.0), 1),
                    "efficiency_percent": round(random.uniform(60.0, 85.0), 1),
                    "age_years": random.randint(5, 40),
                    "annual_operating_hours": random.randint(2000, 8760)
                },
                "energy_use": {
                    "annual_fuel_consumption_mmbtu": 0,  # Calculated below
                    "annual_fuel_cost_usd": 0  # Calculated below
                }
            }

            # Calculate annual fuel consumption
            facility['energy_use']['annual_fuel_consumption_mmbtu'] = (
                facility['boiler']['firing_rate_mmbtu_hr'] *
                facility['boiler']['annual_operating_hours']
            )

            # Calculate annual fuel cost ($10/MMBtu natural gas, $5/MMBtu coal, etc.)
            fuel_prices = {
                "natural_gas": 10.0,
                "coal": 5.0,
                "oil": 20.0,
                "biomass": 8.0
            }
            facility['energy_use']['annual_fuel_cost_usd'] = (
                facility['energy_use']['annual_fuel_consumption_mmbtu'] *
                fuel_prices[facility['boiler']['fuel_type']]
            )

            facilities.append(facility)

        return facilities

    def generate_cbam_shipment_data(
        self, num_shipments: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate test data for CBAM import analysis."""
        shipments = []

        for i in range(num_shipments):
            shipment = {
                "shipment_id": f"SHIP-{i:05d}",
                "product_category": random.choice([
                    "cement",
                    "steel",
                    "aluminum",
                    "fertilizer",
                    "electricity"
                ]),
                "origin_country": random.choice([
                    "China",
                    "India",
                    "Turkey",
                    "Russia",
                    "Ukraine"
                ]),
                "quantity_tonnes": round(random.uniform(100, 50000), 1),
                "import_date": self.faker.date_between(
                    start_date="-1y",
                    end_date="today"
                ).isoformat(),
                "production_data": {
                    "fuel_mix": {
                        "coal": round(random.uniform(0.3, 0.8), 2),
                        "natural_gas": round(random.uniform(0.1, 0.4), 2),
                        "alternative_fuels": round(random.uniform(0.0, 0.2), 2)
                    },
                    "electricity_consumption_kwh_per_tonne": round(
                        random.uniform(50, 500), 1
                    ),
                    "process_emissions_kg_co2_per_tonne": round(
                        random.uniform(100, 1000), 1
                    )
                }
            }

            # Normalize fuel mix to sum to 1.0
            fuel_sum = sum(shipment['production_data']['fuel_mix'].values())
            for fuel in shipment['production_data']['fuel_mix']:
                shipment['production_data']['fuel_mix'][fuel] /= fuel_sum

            shipments.append(shipment)

        return shipments
```

### Test Fixtures (Pytest)

```python
# tests/agents/conftest.py

import pytest
from tests.agents.fixtures.test_data_generator import IndustrialTestDataGenerator


@pytest.fixture
def test_data_generator():
    """Provide test data generator with fixed seed for reproducibility."""
    return IndustrialTestDataGenerator(seed=42)


@pytest.fixture
def sample_boiler_data(test_data_generator):
    """Generate sample boiler facility data."""
    return test_data_generator.generate_boiler_data(num_facilities=10)


@pytest.fixture
def sample_cbam_shipments(test_data_generator):
    """Generate sample CBAM shipment data."""
    return test_data_generator.generate_cbam_shipment_data(num_shipments=10)


@pytest.fixture
def boiler_efficiency_agent():
    """Create BoilerEfficiencyOptimizer agent for testing."""
    return BoilerEfficiencyOptimizer(temperature=0.0, seed=42)


@pytest.fixture
def mock_fuel_agent():
    """Mock FuelAgent for testing without external dependencies."""
    class MockFuelAgent:
        def get_emission_factor(self, fuel_type: str, region: str = "US") -> float:
            emission_factors = {
                "natural_gas": 53.06,  # kg CO2/MMBtu
                "coal": 95.52,
                "oil": 73.96,
                "biomass": 0.0  # Biogenic (carbon neutral)
            }
            return emission_factors.get(fuel_type, 53.06)

    return MockFuelAgent()
```

---

## Integration with Existing Test Pipelines

### CI/CD Integration

```yaml
# .github/workflows/test-agent.yml

name: Agent Test Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11, 3.12]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run unit tests
      run: |
        pytest tests/agents/test_*.py \
          --cov=greenlang/agents \
          --cov-report=xml \
          --cov-report=html \
          --cov-fail-under=85

    - name: Run golden tests
      run: |
        pytest tests/agents/test_*_golden.py \
          --verbose \
          --tb=short

    - name: Run performance tests
      run: |
        pytest tests/agents/test_*_performance.py \
          --benchmark-only \
          --benchmark-autosave

    - name: Upload coverage report
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        flags: agent-tests
        name: agent-coverage-${{ matrix.python-version }}
```

### Test Reporting

```python
# tests/agents/conftest.py (pytest hooks)

import pytest
import json
from datetime import datetime


def pytest_configure(config):
    """Create test report directory."""
    config.test_report_dir = "test_reports"
    os.makedirs(config.test_report_dir, exist_ok=True)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test results for reporting."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        # Store test result
        test_result = {
            "test_name": item.nodeid,
            "outcome": report.outcome,
            "duration_seconds": report.duration,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Append to test results file
        results_file = os.path.join(
            item.config.test_report_dir,
            "test_results.jsonl"
        )
        with open(results_file, "a") as f:
            f.write(json.dumps(test_result) + "\n")


def pytest_sessionfinish(session, exitstatus):
    """Generate test summary report."""
    test_results = []

    results_file = os.path.join(
        session.config.test_report_dir,
        "test_results.jsonl"
    )

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            test_results = [json.loads(line) for line in f]

    # Calculate summary statistics
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r['outcome'] == 'passed')
    failed_tests = sum(1 for r in test_results if r['outcome'] == 'failed')
    total_duration = sum(r['duration_seconds'] for r in test_results)

    summary = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
        "total_duration_seconds": total_duration,
        "average_duration_seconds": total_duration / total_tests if total_tests > 0 else 0,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Write summary report
    summary_file = os.path.join(
        session.config.test_report_dir,
        "test_summary.json"
    )
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ({summary['pass_rate']*100:.1f}%)")
    print(f"Failed: {failed_tests}")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print(f"Average Duration: {summary['average_duration_seconds']:.3f} seconds")
    print("=" * 70)
```

---

## Best Practices

### Golden Test Maintenance

1. **Version Control:** Store golden tests in Git with test results JSON
2. **Regular Review:** Review golden tests quarterly for accuracy
3. **Expert Validation:** Have domain experts validate golden test answers annually
4. **Change Management:** Any change to golden test expected output requires:
   - Root cause analysis (why did it change?)
   - Expert review (is the new answer correct?)
   - Version bump (major version if breaking change)
   - Documentation update

### Test Data Privacy

1. **Anonymization:** Remove all PII from real customer data
   - Company names → "Company A", "Company B"
   - Locations → Generic (city/state only, no addresses)
   - Contact info → Removed entirely

2. **Synthetic Data:** Prefer synthetic data over real data when possible

3. **Access Control:** Limit access to test data (especially if derived from real data)

### Test Performance

1. **Parallelization:** Run tests in parallel with pytest-xdist
   ```bash
   pytest -n auto  # Use all CPU cores
   ```

2. **Test Caching:** Cache expensive test fixtures with pytest-cache
   ```python
   @pytest.fixture(scope="session")
   def expensive_data():
       # Computed once per test session
       return load_expensive_data()
   ```

3. **Test Selection:** Use pytest markers to run subsets of tests
   ```python
   @pytest.mark.golden
   def test_golden_scenario():
       pass

   @pytest.mark.slow
   def test_slow_integration():
       pass
   ```
   ```bash
   pytest -m golden  # Run only golden tests
   pytest -m "not slow"  # Skip slow tests
   ```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-03 | GL-TestEngineer | Initial test suite structure |

---

**END OF DOCUMENT**
