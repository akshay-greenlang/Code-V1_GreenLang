"""
Test Data Generators for GreenLang Agent Tests

Provides utilities for generating realistic test data for:
- Carbon emissions scenarios
- CBAM shipments
- SBTi target submissions
- Economizer operating conditions
- Burner health histories

Usage:
    from tests.agents.generators import TestDataGenerator

    gen = TestDataGenerator(seed=42)
    emissions_data = gen.generate_emission_scenarios(100)
"""

import random
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from faker import Faker


class TestDataGenerator:
    """
    Generate realistic test data for GreenLang agent testing.

    All generators use a seed for reproducible test data.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize generator with seed for reproducibility.

        Args:
            seed: Random seed for reproducible data generation
        """
        self.faker = Faker()
        Faker.seed(seed)
        random.seed(seed)

    # =========================================================================
    # Carbon Emissions Data (GL-001)
    # =========================================================================

    def generate_emission_scenarios(
        self,
        num_scenarios: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate carbon emission calculation scenarios.

        Args:
            num_scenarios: Number of scenarios to generate

        Returns:
            List of emission input dictionaries
        """
        fuel_configs = [
            {"fuel_type": "natural_gas", "unit": "m3", "ef_range": (1.8, 2.1)},
            {"fuel_type": "diesel", "unit": "L", "ef_range": (2.5, 2.8)},
            {"fuel_type": "gasoline", "unit": "L", "ef_range": (2.1, 2.4)},
            {"fuel_type": "electricity_grid", "unit": "kWh", "ef_range": (0.05, 0.6)},
        ]
        regions = ["US", "EU", "DE", "FR", "UK", "CN", "JP", "AU"]

        scenarios = []
        for _ in range(num_scenarios):
            fuel_config = random.choice(fuel_configs)
            scenarios.append({
                "fuel_type": fuel_config["fuel_type"],
                "quantity": round(random.uniform(10, 100000), 2),
                "unit": fuel_config["unit"],
                "region": random.choice(regions),
                "scope": random.choice([1, 2]),
                "expected_ef_range": fuel_config["ef_range"],
            })

        return scenarios

    def generate_emission_edge_cases(self) -> List[Dict[str, Any]]:
        """
        Generate edge case scenarios for emission calculations.

        Returns:
            List of edge case input dictionaries
        """
        return [
            # Zero quantity
            {
                "fuel_type": "natural_gas",
                "quantity": 0.0,
                "unit": "m3",
                "region": "US",
                "expected_emissions": 0.0,
            },
            # Very small quantity
            {
                "fuel_type": "diesel",
                "quantity": 0.001,
                "unit": "L",
                "region": "EU",
            },
            # Very large quantity
            {
                "fuel_type": "electricity_grid",
                "quantity": 1_000_000_000,
                "unit": "kWh",
                "region": "US",
            },
            # Unknown region (should use fallback)
            {
                "fuel_type": "natural_gas",
                "quantity": 1000,
                "unit": "m3",
                "region": "XX",
            },
        ]

    # =========================================================================
    # CBAM Shipment Data (GL-002)
    # =========================================================================

    def generate_cbam_shipments(
        self,
        num_shipments: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate CBAM shipment test data.

        Args:
            num_shipments: Number of shipments to generate

        Returns:
            List of CBAM input dictionaries
        """
        products = [
            {"cn_prefix": "7208", "category": "iron_steel", "ef_range": (1.8, 2.5)},
            {"cn_prefix": "7210", "category": "iron_steel", "ef_range": (1.8, 2.5)},
            {"cn_prefix": "7601", "category": "aluminium", "ef_range": (6.0, 12.0)},
            {"cn_prefix": "7604", "category": "aluminium", "ef_range": (6.0, 12.0)},
            {"cn_prefix": "2523", "category": "cement", "ef_range": (0.7, 0.95)},
            {"cn_prefix": "3102", "category": "fertilizers", "ef_range": (2.0, 3.0)},
        ]
        countries = ["CN", "IN", "RU", "TR", "UA", "BR", "KR", "VN"]
        periods = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"]

        shipments = []
        for i in range(num_shipments):
            product = random.choice(products)
            shipments.append({
                "shipment_id": f"CBAM-{i:05d}",
                "cn_code": f"{product['cn_prefix']}{random.randint(1000, 9999)}",
                "quantity_tonnes": round(random.uniform(10, 10000), 2),
                "country_of_origin": random.choice(countries),
                "reporting_period": random.choice(periods),
                "expected_category": product["category"],
                "expected_ef_range": product["ef_range"],
            })

        return shipments

    # =========================================================================
    # SBTi Target Data (GL-010)
    # =========================================================================

    def generate_sbti_submissions(
        self,
        num_companies: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate SBTi target submission test data.

        Args:
            num_companies: Number of company submissions to generate

        Returns:
            List of SBTi input dictionaries
        """
        sectors = [
            "power_generation", "steel", "cement", "aluminum",
            "transport_road", "chemicals", "general"
        ]

        submissions = []
        for i in range(num_companies):
            base_year = random.randint(2018, 2022)
            scope1 = random.uniform(1000, 100000)
            scope2 = random.uniform(500, 50000)
            scope3 = random.uniform(scope1 + scope2, (scope1 + scope2) * 5)

            # Generate near-term target
            target_year = base_year + random.randint(7, 12)
            years_to_target = target_year - base_year

            # Calculate 1.5C-aligned reduction (4.2% annual)
            aligned_reduction = (1 - (1 - 0.042) ** years_to_target) * 100

            submissions.append({
                "company_id": f"COMPANY-{i:04d}",
                "company_name": self.faker.company(),
                "base_year": base_year,
                "scope1_emissions": round(scope1, 2),
                "scope2_emissions": round(scope2, 2),
                "scope3_emissions": round(scope3, 2),
                "target_year": target_year,
                "target_type": "near_term",
                "reduction_pct": round(random.uniform(30, 60), 1),
                "aligned_1_5c_reduction": round(aligned_reduction, 1),
                "sector": random.choice(sectors),
            })

        return submissions

    # =========================================================================
    # Economizer Performance Data (GL-020)
    # =========================================================================

    def generate_economizer_conditions(
        self,
        num_conditions: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate economizer operating condition test data.

        Args:
            num_conditions: Number of operating conditions to generate

        Returns:
            List of economizer input dictionaries
        """
        conditions = []
        for i in range(num_conditions):
            gas_in = random.uniform(250, 500)
            gas_out = random.uniform(100, gas_in - 50)
            water_in = random.uniform(80, 150)
            water_out = random.uniform(water_in + 30, 280)
            drum_pressure = random.uniform(2.0, 15.0)

            conditions.append({
                "condition_id": f"ECON-{i:04d}",
                "flue_gas": {
                    "temperature_in_celsius": round(gas_in, 1),
                    "temperature_out_celsius": round(gas_out, 1),
                    "mass_flow_kg_s": round(random.uniform(20, 100), 1),
                    "H2O_percent": round(random.uniform(5, 15), 1),
                    "SO3_ppmv": round(random.uniform(1, 50), 1),
                },
                "water_side": {
                    "inlet_temperature_celsius": round(water_in, 1),
                    "outlet_temperature_celsius": round(water_out, 1),
                    "mass_flow_kg_s": round(random.uniform(10, 60), 1),
                    "drum_pressure_MPa": round(drum_pressure, 2),
                },
            })

        return conditions

    def generate_corrosion_risk_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generate scenarios for cold-end corrosion risk testing.

        Returns:
            List of corrosion risk scenarios
        """
        return [
            # Low risk - high water temp, low SO3
            {
                "name": "Low Risk",
                "water_inlet_c": 120,
                "H2O_percent": 6,
                "SO3_ppmv": 5,
                "expected_risk": "LOW",
            },
            # Moderate risk
            {
                "name": "Moderate Risk",
                "water_inlet_c": 100,
                "H2O_percent": 10,
                "SO3_ppmv": 20,
                "expected_risk": "MODERATE",
            },
            # High risk - low water temp, high SO3
            {
                "name": "High Risk",
                "water_inlet_c": 80,
                "H2O_percent": 15,
                "SO3_ppmv": 50,
                "expected_risk": "HIGH",
            },
        ]

    # =========================================================================
    # Burner Maintenance Data (GL-021)
    # =========================================================================

    def generate_burner_fleet(
        self,
        num_burners: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate burner fleet test data with varying health states.

        Args:
            num_burners: Number of burners to generate

        Returns:
            List of burner input dictionaries
        """
        models = ["ACME-5000", "FURNEX-3000", "PYRO-2500", "HEATMAX-4000"]
        fuel_types = ["natural_gas", "fuel_oil_2", "propane", "hydrogen"]

        burners = []
        for i in range(num_burners):
            # Generate varying ages and conditions
            age_years = random.uniform(0, 15)
            hours_per_year = random.uniform(4000, 8000)
            operating_hours = age_years * hours_per_year

            # Correlate health with age
            base_health = 100 - (age_years * 5) + random.uniform(-10, 10)
            base_health = max(10, min(100, base_health))

            # Generate health history
            history_length = min(10, int(age_years * 2))
            health_history = self._generate_degrading_health(
                current_health=base_health,
                num_readings=max(0, history_length),
            )

            # Weibull parameters based on age characteristics
            beta = random.uniform(2.0, 3.5)  # Wear-out failures
            eta = random.uniform(30000, 60000)

            installation_date = date.today() - timedelta(days=int(age_years * 365))

            burners.append({
                "burner_id": f"BRN-{i:04d}",
                "burner_model": random.choice(models),
                "fuel_type": random.choice(fuel_types),
                "operating_hours": round(operating_hours, 0),
                "design_life_hours": 50000,
                "cycles_count": int(operating_hours / 8),  # ~8 hours per cycle
                "weibull_beta": round(beta, 2),
                "weibull_eta": round(eta, 0),
                "flame_temperature_c": round(1200 - (age_years * 15) + random.uniform(-50, 50), 0),
                "stability_index": round(max(0.5, 0.98 - (age_years * 0.03) + random.uniform(-0.05, 0.05)), 2),
                "o2_percent": round(3.0 + (age_years * 0.2) + random.uniform(-0.5, 0.5), 1),
                "co_ppm": round(20 + (age_years * 20) + random.uniform(-20, 50), 0),
                "nox_ppm": round(70 + (age_years * 5) + random.uniform(-10, 20), 0),
                "installation_date": installation_date.isoformat(),
                "health_history": health_history,
                "estimated_health": round(base_health, 1),
            })

        return burners

    def _generate_degrading_health(
        self,
        current_health: float,
        num_readings: int
    ) -> List[float]:
        """Generate a realistic degrading health history."""
        if num_readings == 0:
            return []

        # Work backwards from current health
        history = [current_health]
        health = current_health

        for _ in range(num_readings - 1):
            # Add ~2-5 points per reading (going backwards = higher in past)
            degradation = random.uniform(2, 5)
            noise = random.uniform(-1, 1)
            health = min(100, health + degradation + noise)
            history.insert(0, round(health, 1))

        return history

    def generate_weibull_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate Weibull calculation test cases with known results.

        Returns:
            List of Weibull test cases with expected values
        """
        import math

        test_cases = []

        # Standard cases
        params = [
            (2.5, 40000, 10000),
            (1.0, 30000, 15000),  # Exponential
            (3.0, 50000, 20000),
            (0.7, 60000, 5000),   # Infant mortality
        ]

        for beta, eta, t in params:
            # Calculate expected values
            reliability = math.exp(-((t / eta) ** beta))

            if t == 0:
                if beta < 1:
                    failure_rate = float('inf')
                elif beta == 1:
                    failure_rate = 1 / eta
                else:
                    failure_rate = 0
            else:
                failure_rate = (beta / eta) * ((t / eta) ** (beta - 1))

            mttf = eta * math.gamma(1 + 1 / beta)

            test_cases.append({
                "beta": beta,
                "eta": eta,
                "t": t,
                "expected_reliability": round(reliability, 6),
                "expected_failure_rate": failure_rate,
                "expected_mttf": round(mttf, 2),
            })

        return test_cases

    # =========================================================================
    # General Utilities
    # =========================================================================

    def generate_timestamp_range(
        self,
        start_date: date,
        end_date: date,
        num_timestamps: int
    ) -> List[datetime]:
        """Generate evenly spaced timestamps within a date range."""
        delta = (end_date - start_date) / num_timestamps
        return [
            datetime.combine(start_date + delta * i, datetime.min.time())
            for i in range(num_timestamps)
        ]

    def generate_random_hash(self) -> str:
        """Generate a random SHA-256-like hash for testing."""
        import hashlib
        random_str = str(random.random()).encode()
        return hashlib.sha256(random_str).hexdigest()


# =============================================================================
# Pre-built Test Data Sets
# =============================================================================


# Known emission calculation values for validation
KNOWN_EMISSION_VALUES = [
    {
        "fuel_type": "natural_gas",
        "quantity": 1000,
        "unit": "m3",
        "region": "US",
        "ef": 1.93,
        "expected_emissions": 1930.0,
    },
    {
        "fuel_type": "diesel",
        "quantity": 500,
        "unit": "L",
        "region": "US",
        "ef": 2.68,
        "expected_emissions": 1340.0,
    },
    {
        "fuel_type": "electricity_grid",
        "quantity": 10000,
        "unit": "kWh",
        "region": "US",
        "ef": 0.417,
        "expected_emissions": 4170.0,
    },
]


# Known CBAM calculation values
KNOWN_CBAM_VALUES = [
    {
        "cn_code": "72081000",
        "quantity_tonnes": 1000,
        "country": "CN",
        "direct_ef": 2.10,
        "indirect_ef": 0.45,
        "expected_direct": 2100.0,
        "expected_indirect": 450.0,
        "expected_total": 2550.0,
    },
    {
        "cn_code": "76011000",
        "quantity_tonnes": 100,
        "country": "CN",
        "direct_ef": 1.65,
        "indirect_ef": 10.20,
        "expected_total": 1185.0,
    },
]


# Known Weibull calculation values
KNOWN_WEIBULL_VALUES = [
    {
        "beta": 2.5,
        "eta": 40000,
        "t": 10000,
        "expected_R": 0.9692,
        "tolerance": 0.001,
    },
    {
        "beta": 1.0,
        "eta": 30000,
        "t": 15000,
        "expected_R": 0.6065,  # exp(-0.5)
        "tolerance": 0.001,
    },
]


if __name__ == "__main__":
    # Demo usage
    gen = TestDataGenerator(seed=42)

    print("Generated 5 emission scenarios:")
    for scenario in gen.generate_emission_scenarios(5):
        print(f"  {scenario}")

    print("\nGenerated 3 CBAM shipments:")
    for shipment in gen.generate_cbam_shipments(3):
        print(f"  {shipment}")

    print("\nGenerated 2 burners:")
    for burner in gen.generate_burner_fleet(2):
        print(f"  {burner['burner_id']}: {burner['operating_hours']:.0f}h, "
              f"health={burner['estimated_health']:.1f}")
