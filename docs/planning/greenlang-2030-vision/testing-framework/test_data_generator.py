# -*- coding: utf-8 -*-
"""
Test data generators for GreenLang testing.

Provides realistic test data generation for various testing scenarios.
"""

import random
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from faker import Faker
import pandas as pd
import numpy as np
from greenlang.determinism import deterministic_random
from greenlang.determinism import deterministic_uuid, DeterministicClock


class TestDataGenerator:
    """Generate test data for GreenLang applications."""

    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        self.faker = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Pre-defined data sets
        self.product_categories = ["cement", "steel", "aluminum", "fertilizer", "iron", "electricity"]
        self.fuel_types = ["diesel", "natural_gas", "coal", "electricity", "gasoline", "biomass"]
        self.countries = ["US", "CN", "DE", "FR", "IN", "BR", "UK", "JP", "CA", "AU"]
        self.regions = ["north_america", "europe", "asia", "south_america", "africa", "oceania"]
        self.facility_types = ["manufacturing", "warehouse", "office", "datacenter", "retail"]

    # ========================================================================
    # CBAM Test Data
    # ========================================================================

    def generate_cbam_shipment(
        self,
        product_category: Optional[str] = None,
        with_errors: bool = False
    ) -> Dict[str, Any]:
        """Generate single CBAM shipment record."""
        if product_category is None:
            product_category = deterministic_random().choice(self.product_categories)

        shipment = {
            "shipment_id": self.faker.deterministic_uuid(__name__, str(DeterministicClock.now())),
            "product_category": product_category,
            "weight_tonnes": round(random.uniform(0.1, 100.0), 2),
            "origin_country": deterministic_random().choice(["CN", "IN", "RU", "TR", "UA"]),
            "destination_port": deterministic_random().choice(["Rotterdam", "Hamburg", "Antwerp"]),
            "import_date": self.faker.date_between(start_date="-6m", end_date="today").isoformat(),
            "supplier_name": self.faker.company(),
            "supplier_id": f"SUP-{deterministic_random().randint(1000, 9999)}",
            "hs_code": self._get_hs_code(product_category),
            "declared_emissions": round(random.uniform(0.5, 5.0), 3),
            "verification_status": deterministic_random().choice(["verified", "pending", "rejected"]),
            "documents": self._generate_document_list()
        }

        if with_errors:
            # Inject data quality issues
            if deterministic_random().random() < 0.2:
                shipment["declared_emissions"] = None  # Missing emissions
            if deterministic_random().random() < 0.1:
                shipment["weight_tonnes"] = -10  # Invalid weight

        return shipment

    def generate_cbam_batch(
        self,
        count: int = 100,
        error_rate: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Generate batch of CBAM shipments."""
        shipments = []
        for i in range(count):
            with_errors = deterministic_random().random() < error_rate
            shipments.append(self.generate_cbam_shipment(with_errors=with_errors))
        return shipments

    def _get_hs_code(self, product_category: str) -> str:
        """Get HS code for product category."""
        hs_codes = {
            "cement": "2523.10",
            "steel": "7208.10",
            "aluminum": "7601.10",
            "fertilizer": "3105.10",
            "iron": "7201.10",
            "electricity": "2716.00"
        }
        return hs_codes.get(product_category, "9999.99")

    def _generate_document_list(self) -> List[str]:
        """Generate list of document filenames."""
        doc_types = ["invoice", "certificate", "bill_of_lading", "customs", "verification"]
        num_docs = deterministic_random().randint(2, 5)
        docs = []
        for _ in range(num_docs):
            doc_type = deterministic_random().choice(doc_types)
            doc_id = deterministic_random().randint(1000, 9999)
            docs.append(f"{doc_type}_{doc_id}.pdf")
        return docs

    # ========================================================================
    # Emission Data Generators
    # ========================================================================

    def generate_fuel_consumption(
        self,
        facility_id: Optional[str] = None,
        period: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate fuel consumption data."""
        if facility_id is None:
            facility_id = f"FAC-{deterministic_random().randint(100, 999)}"

        if period is None:
            year = 2025
            quarter = deterministic_random().randint(1, 4)
            period = f"{year}-Q{quarter}"

        consumption = {
            "facility_id": facility_id,
            "facility_type": deterministic_random().choice(self.facility_types),
            "reporting_period": period,
            "location": deterministic_random().choice(self.countries),
            "fuel_data": []
        }

        # Generate 2-5 fuel types consumed
        num_fuels = deterministic_random().randint(2, 5)
        for _ in range(num_fuels):
            fuel_type = deterministic_random().choice(self.fuel_types)
            fuel_record = {
                "fuel_type": fuel_type,
                "quantity": round(random.uniform(100, 10000), 2),
                "unit": self._get_fuel_unit(fuel_type),
                "combustion_type": deterministic_random().choice(["stationary", "mobile"]),
                "emission_factor": round(random.uniform(1.5, 3.5), 3)
            }
            consumption["fuel_data"].append(fuel_record)

        consumption["total_emissions"] = sum(
            f["quantity"] * f["emission_factor"] for f in consumption["fuel_data"]
        )

        return consumption

    def _get_fuel_unit(self, fuel_type: str) -> str:
        """Get appropriate unit for fuel type."""
        units = {
            "diesel": "liters",
            "gasoline": "liters",
            "natural_gas": "m3",
            "coal": "kg",
            "electricity": "kWh",
            "biomass": "kg"
        }
        return units.get(fuel_type, "kg")

    def generate_scope3_emissions(
        self,
        company_id: Optional[str] = None,
        year: int = 2025
    ) -> Dict[str, Any]:
        """Generate Scope 3 emissions data (15 categories)."""
        if company_id is None:
            company_id = f"COMP-{deterministic_random().randint(1000, 9999)}"

        categories = {
            "cat1_purchased_goods": round(random.uniform(1000, 50000), 2),
            "cat2_capital_goods": round(random.uniform(500, 10000), 2),
            "cat3_fuel_energy": round(random.uniform(200, 5000), 2),
            "cat4_upstream_transportation": round(random.uniform(500, 8000), 2),
            "cat5_waste": round(random.uniform(50, 1000), 2),
            "cat6_business_travel": round(random.uniform(100, 2000), 2),
            "cat7_employee_commuting": round(random.uniform(50, 1500), 2),
            "cat8_upstream_leased": round(random.uniform(0, 3000), 2),
            "cat9_downstream_transportation": round(random.uniform(500, 8000), 2),
            "cat10_processing_sold_products": round(random.uniform(0, 10000), 2),
            "cat11_use_of_sold_products": round(random.uniform(1000, 100000), 2),
            "cat12_end_of_life": round(random.uniform(100, 5000), 2),
            "cat13_downstream_leased": round(random.uniform(0, 3000), 2),
            "cat14_franchises": round(random.uniform(0, 2000), 2),
            "cat15_investments": round(random.uniform(0, 10000), 2)
        }

        return {
            "company_id": company_id,
            "company_name": self.faker.company(),
            "reporting_year": year,
            "categories": categories,
            "total_scope3": sum(categories.values()),
            "verification_status": deterministic_random().choice(["verified", "pending", "in_review"]),
            "data_quality_score": round(random.uniform(0.6, 1.0), 2),
            "submission_date": self.faker.date_between(
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31"
            ).isoformat()
        }

    # ========================================================================
    # Supply Chain Data
    # ========================================================================

    def generate_supply_chain_data(
        self,
        num_suppliers: int = 10,
        num_products: int = 50
    ) -> Dict[str, Any]:
        """Generate supply chain network data."""
        suppliers = []
        for i in range(num_suppliers):
            supplier = {
                "supplier_id": f"SUP-{i+1000}",
                "name": self.faker.company(),
                "country": deterministic_random().choice(self.countries),
                "tier": deterministic_random().randint(1, 3),
                "risk_score": round(random.uniform(0, 100), 1),
                "sustainability_rating": deterministic_random().choice(["A", "B", "C", "D", "E"]),
                "products": []
            }

            # Add 3-10 products per supplier
            num_supplier_products = deterministic_random().randint(3, min(10, num_products))
            for j in range(num_supplier_products):
                product = {
                    "product_id": f"PROD-{deterministic_random().randint(1000, 9999)}",
                    "name": self.faker.word().capitalize() + " " + deterministic_random().choice(["Widget", "Component", "Material"]),
                    "category": deterministic_random().choice(["raw_material", "component", "finished_good"]),
                    "carbon_intensity": round(random.uniform(0.1, 10.0), 2),
                    "unit_price": round(random.uniform(10, 1000), 2),
                    "lead_time_days": deterministic_random().randint(7, 90)
                }
                supplier["products"].append(product)

            suppliers.append(supplier)

        return {
            "network_id": self.faker.deterministic_uuid(__name__, str(DeterministicClock.now())),
            "company": self.faker.company(),
            "suppliers": suppliers,
            "total_suppliers": num_suppliers,
            "total_products": sum(len(s["products"]) for s in suppliers),
            "assessment_date": datetime.now(timezone.utc).isoformat()
        }

    # ========================================================================
    # Performance Test Data
    # ========================================================================

    def generate_load_test_data(
        self,
        num_records: int = 10000,
        record_type: str = "emission"
    ) -> pd.DataFrame:
        """Generate data for load testing."""
        data = []

        for i in range(num_records):
            if record_type == "emission":
                record = {
                    "id": i,
                    "timestamp": self.faker.date_time_between(start_date="-1y", end_date="now"),
                    "facility_id": f"FAC-{deterministic_random().randint(100, 999)}",
                    "fuel_type": deterministic_random().choice(self.fuel_types),
                    "quantity": round(random.uniform(10, 1000), 2),
                    "emissions": round(random.uniform(10, 5000), 2),
                    "data_quality": round(random.uniform(0.5, 1.0), 2)
                }
            elif record_type == "shipment":
                record = self.generate_cbam_shipment()
                record["id"] = i
            else:
                record = {"id": i, "value": deterministic_random().random()}

            data.append(record)

        return pd.DataFrame(data)

    def generate_stress_test_payload(
        self,
        size_mb: int = 10
    ) -> Dict[str, Any]:
        """Generate large payload for stress testing."""
        # Calculate approximate number of records for target size
        record_size_bytes = 1000  # Approximate size per record
        num_records = (size_mb * 1024 * 1024) // record_size_bytes

        payload = {
            "test_id": self.faker.deterministic_uuid(__name__, str(DeterministicClock.now())),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "records": []
        }

        for _ in range(num_records):
            record = {
                "id": self.faker.deterministic_uuid(__name__, str(DeterministicClock.now())),
                "data": self.faker.text(max_nb_chars=800),
                "values": [deterministic_random().random() for _ in range(10)]
            }
            payload["records"].append(record)

        return payload

    # ========================================================================
    # Edge Cases and Error Scenarios
    # ========================================================================

    def generate_edge_cases(self) -> List[Dict[str, Any]]:
        """Generate edge case test data."""
        edge_cases = [
            # Boundary values
            {"case": "zero_emissions", "value": 0.0},
            {"case": "negative_value", "value": -100.0},
            {"case": "very_large_number", "value": 1e15},
            {"case": "very_small_number", "value": 1e-15},

            # Special characters
            {"case": "special_chars", "value": "Test@#$%^&*()_+"},
            {"case": "unicode", "value": "测试数据 テストデータ"},
            {"case": "sql_injection", "value": "'; DROP TABLE users; --"},
            {"case": "xss_attempt", "value": "<script>alert('XSS')</script>"},

            # Null/Empty values
            {"case": "null_value", "value": None},
            {"case": "empty_string", "value": ""},
            {"case": "empty_array", "value": []},
            {"case": "empty_object", "value": {}},

            # Date edge cases
            {"case": "future_date", "value": "2099-12-31"},
            {"case": "past_date", "value": "1900-01-01"},
            {"case": "invalid_date", "value": "2025-13-32"},

            # Numeric precision
            {"case": "max_decimal_places", "value": Decimal("0.123456789012345678901234567890")},
            {"case": "scientific_notation", "value": "1.23e-10"}
        ]

        return edge_cases

    # ========================================================================
    # Mock External API Responses
    # ========================================================================

    def generate_mock_api_response(
        self,
        api_type: str = "emission_factors",
        status: str = "success",
        delay_ms: int = 0
    ) -> Dict[str, Any]:
        """Generate mock API responses for testing."""
        if api_type == "emission_factors":
            if status == "success":
                return {
                    "status": 200,
                    "data": {
                        "factors": [
                            {
                                "fuel_type": fuel,
                                "region": region,
                                "factor": round(random.uniform(1.5, 3.5), 3),
                                "unit": "kg CO2e/unit",
                                "source": "IPCC 2024",
                                "uncertainty": round(random.uniform(0.05, 0.15), 2)
                            }
                            for fuel in self.fuel_types[:3]
                            for region in self.regions[:2]
                        ]
                    },
                    "response_time_ms": delay_ms
                }
            else:
                return {
                    "status": 500,
                    "error": "Internal server error",
                    "message": "Database connection failed"
                }

        elif api_type == "weather":
            return {
                "status": 200,
                "data": {
                    "temperature": round(random.uniform(-10, 40), 1),
                    "humidity": deterministic_random().randint(20, 90),
                    "pressure": round(random.uniform(990, 1030), 1),
                    "wind_speed": round(random.uniform(0, 30), 1)
                }
            }

        elif api_type == "erp":
            return {
                "status": 200,
                "data": {
                    "orders": [
                        {
                            "order_id": self.faker.deterministic_uuid(__name__, str(DeterministicClock.now())),
                            "total": round(random.uniform(100, 10000), 2),
                            "status": deterministic_random().choice(["pending", "processing", "completed"])
                        }
                        for _ in range(deterministic_random().randint(5, 15))
                    ]
                }
            }

        return {"status": 404, "error": "Unknown API type"}

    # ========================================================================
    # Compliance Test Data
    # ========================================================================

    def generate_compliance_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for regulatory compliance."""
        test_cases = []

        # GHG Protocol compliance
        test_cases.append({
            "regulation": "GHG Protocol",
            "test_type": "calculation_methodology",
            "input": {
                "activity_data": 1000.0,
                "emission_factor": 2.68,
                "expected_result": 2680.0,
                "tolerance": 0.0001
            }
        })

        # EU CBAM compliance
        test_cases.append({
            "regulation": "EU CBAM",
            "test_type": "reporting_completeness",
            "required_fields": [
                "cn_code", "quantity", "emissions", "country_of_origin",
                "installation_id", "production_method"
            ]
        })

        # ISO 14064 compliance
        test_cases.append({
            "regulation": "ISO 14064",
            "test_type": "verification_requirements",
            "requirements": [
                "documented_methodology",
                "uncertainty_assessment",
                "third_party_verification",
                "management_review"
            ]
        })

        return test_cases


# ========================================================================
# Test Data Factory
# ========================================================================

class TestDataFactory:
    """Factory for creating test data with specific patterns."""

    def __init__(self, generator: Optional[TestDataGenerator] = None):
        """Initialize factory with generator."""
        self.generator = generator or TestDataGenerator()

    def create_deterministic_sequence(
        self,
        pattern: str = "linear",
        length: int = 100
    ) -> List[float]:
        """Create deterministic data sequence for testing."""
        if pattern == "linear":
            return [i * 0.1 for i in range(length)]
        elif pattern == "exponential":
            return [2 ** (i * 0.1) for i in range(length)]
        elif pattern == "sine":
            return [np.sin(i * 0.1) for i in range(length)]
        elif pattern == "random_walk":
            values = [0.0]
            for _ in range(length - 1):
                values.append(values[-1] + random.gauss(0, 1))
            return values
        else:
            return [deterministic_random().random() for _ in range(length)]

    def create_test_scenario(
        self,
        scenario_type: str = "normal"
    ) -> Dict[str, Any]:
        """Create complete test scenario."""
        scenarios = {
            "normal": {
                "name": "Normal Operations",
                "load": "medium",
                "error_rate": 0.01,
                "data_quality": 0.95,
                "network_latency_ms": 50
            },
            "peak": {
                "name": "Peak Load",
                "load": "high",
                "error_rate": 0.05,
                "data_quality": 0.90,
                "network_latency_ms": 150
            },
            "degraded": {
                "name": "Degraded Service",
                "load": "medium",
                "error_rate": 0.20,
                "data_quality": 0.70,
                "network_latency_ms": 500
            },
            "failure": {
                "name": "Service Failure",
                "load": "low",
                "error_rate": 0.50,
                "data_quality": 0.50,
                "network_latency_ms": 5000
            }
        }

        return scenarios.get(scenario_type, scenarios["normal"])