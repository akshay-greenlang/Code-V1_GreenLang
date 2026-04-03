"""
Test Data Generators for GreenLang Testing.

Creates synthetic test data for comprehensive testing scenarios.
All generators use seeded random for reproducibility.  No external
dependencies (no ``faker``).

Usage::

    generator = IndustrialTestDataGenerator(seed=42)
    fuel_data = generator.generate_fuel_data(num_records=100)

    gen = EmissionDataGenerator(seed=42)
    scope1 = gen.generate_scope1_data(num_records=20)
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import random
import string
import hashlib
import json


class BaseTestDataGenerator:
    """Base class for test data generators with seeded random."""

    def __init__(self, seed: int = 42):
        """
        Initialize generator with seed for reproducibility.

        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        self.random = random.Random(seed)
        self._record_count = 0

    def _generate_id(self, prefix: str = "TEST") -> str:
        """Generate unique test ID."""
        self._record_count += 1
        return f"{prefix}-{self._record_count:06d}"

    def _random_float(self, min_val: float, max_val: float, decimals: int = 4) -> float:
        """Generate random float within range."""
        return round(self.random.uniform(min_val, max_val), decimals)

    def _random_choice(self, options: List[Any]) -> Any:
        """Make random choice from options."""
        return self.random.choice(options)

    def _random_date(self, start_year: int = 2023, end_year: int = 2025) -> str:
        """Generate random date string."""
        year = self.random.randint(start_year, end_year)
        month = self.random.randint(1, 12)
        day = self.random.randint(1, 28)
        return f"{year}-{month:02d}-{day:02d}"


class IndustrialTestDataGenerator(BaseTestDataGenerator):
    """Generate industrial facility and fuel consumption test data."""

    FUEL_TYPES = [
        "natural_gas", "diesel", "gasoline", "lpg", "propane",
        "fuel_oil", "kerosene", "coal", "biomass", "electricity"
    ]

    FUEL_UNITS = {
        "natural_gas": ["MJ", "GJ", "m3", "kWh", "MMBTU"],
        "diesel": ["L", "gal"],
        "gasoline": ["L", "gal"],
        "lpg": ["kg", "L"],
        "propane": ["kg", "L", "gal"],
        "fuel_oil": ["L", "gal"],
        "kerosene": ["L", "gal"],
        "coal": ["kg", "tonnes"],
        "biomass": ["kg", "tonnes"],
        "electricity": ["kWh", "MWh"],
    }

    REGIONS = ["US", "EU", "GB", "GLOBAL"]

    FACILITY_TYPES = [
        "Manufacturing", "Warehouse", "Office", "Retail",
        "Hospital", "School", "Hotel", "Data Center"
    ]

    def generate_fuel_data(self, num_records: int = 100) -> List[Dict[str, Any]]:
        """
        Generate fuel consumption test data.

        Args:
            num_records: Number of records to generate

        Returns:
            List of fuel consumption dictionaries
        """
        records = []

        for _ in range(num_records):
            fuel_type = self._random_choice(self.FUEL_TYPES)
            unit = self._random_choice(self.FUEL_UNITS.get(fuel_type, ["MJ"]))

            # Generate plausible quantities based on fuel type and unit
            quantity_ranges = {
                "MJ": (100, 100000),
                "GJ": (0.1, 1000),
                "kWh": (100, 500000),
                "MWh": (0.1, 500),
                "MMBTU": (1, 1000),
                "L": (10, 100000),
                "gal": (5, 25000),
                "kg": (10, 50000),
                "tonnes": (0.1, 100),
                "m3": (10, 50000),
            }

            min_qty, max_qty = quantity_ranges.get(unit, (1, 10000))
            quantity = self._random_float(min_qty, max_qty)

            record = {
                "record_id": self._generate_id("FUEL"),
                "fuel_type": fuel_type,
                "fuel_quantity": quantity,
                "fuel_unit": unit,
                "region": self._random_choice(self.REGIONS),
                "year": self.random.randint(2020, 2025),
                "facility_id": self._generate_id("FAC"),
                "facility_type": self._random_choice(self.FACILITY_TYPES),
                "reporting_date": self._random_date(),
                "data_quality_score": self._random_float(0.7, 1.0, 2),
            }
            records.append(record)

        return records

    def generate_facility_data(self, num_facilities: int = 50) -> List[Dict[str, Any]]:
        """
        Generate industrial facility test data.

        Args:
            num_facilities: Number of facilities to generate

        Returns:
            List of facility dictionaries
        """
        facilities = []

        for _ in range(num_facilities):
            facility = {
                "facility_id": self._generate_id("FAC"),
                "facility_name": f"Test Facility {self._record_count}",
                "facility_type": self._random_choice(self.FACILITY_TYPES),
                "floor_area_sqm": self._random_float(500, 100000, 0),
                "address": {
                    "country": self._random_choice(["US", "GB", "DE", "FR", "IT"]),
                    "region": self._random_choice(["Northeast", "Southeast", "Midwest", "West"]),
                    "postal_code": f"{self.random.randint(10000, 99999)}",
                },
                "operational_hours": self._random_float(2000, 8760, 0),
                "year_built": self.random.randint(1970, 2024),
                "climate_zone": self._random_choice(["1A", "2A", "3A", "4A", "5A", "6A", "7"]),
            }
            facilities.append(facility)

        return facilities


class CBAMShipmentGenerator(BaseTestDataGenerator):
    """Generate CBAM import shipment test data."""

    PRODUCT_TYPES = [
        "steel_hot_rolled_coil",
        "steel_cold_rolled_coil",
        "steel_rebar",
        "cement_portland",
        "cement_clinker",
        "aluminum_unwrought",
        "aluminum_bars",
        "fertilizer_ammonia",
        "fertilizer_urea",
        "electricity",
        "hydrogen",
    ]

    CN_CODES = {
        "steel_hot_rolled_coil": ["7208.10", "7208.25", "7208.26", "7208.27"],
        "steel_cold_rolled_coil": ["7209.15", "7209.16", "7209.17", "7209.18"],
        "steel_rebar": ["7214.20", "7214.99"],
        "cement_portland": ["2523.21", "2523.29"],
        "cement_clinker": ["2523.10"],
        "aluminum_unwrought": ["7601.10", "7601.20"],
        "aluminum_bars": ["7604.10", "7604.21", "7604.29"],
        "fertilizer_ammonia": ["2814.10", "2814.20"],
        "fertilizer_urea": ["3102.10"],
        "electricity": ["2716.00"],
        "hydrogen": ["2804.10"],
    }

    ORIGIN_COUNTRIES = [
        "CN", "IN", "RU", "TR", "UA", "EG", "SA", "BR", "ZA", "KR"
    ]

    def generate_shipments(self, num_shipments: int = 100) -> List[Dict[str, Any]]:
        """
        Generate CBAM shipment test data.

        Args:
            num_shipments: Number of shipments to generate

        Returns:
            List of shipment dictionaries
        """
        shipments = []

        for _ in range(num_shipments):
            product_type = self._random_choice(self.PRODUCT_TYPES)
            cn_code = self._random_choice(self.CN_CODES.get(product_type, ["9999.00"]))

            # Generate realistic tonnage based on product type
            tonnage_ranges = {
                "steel_hot_rolled_coil": (100, 50000),
                "steel_cold_rolled_coil": (50, 25000),
                "steel_rebar": (100, 30000),
                "cement_portland": (500, 100000),
                "cement_clinker": (1000, 200000),
                "aluminum_unwrought": (10, 5000),
                "aluminum_bars": (5, 2000),
                "fertilizer_ammonia": (100, 50000),
                "fertilizer_urea": (200, 80000),
                "electricity": (1, 10000),  # MWh
                "hydrogen": (0.1, 1000),
            }

            min_tonnage, max_tonnage = tonnage_ranges.get(product_type, (10, 10000))
            quantity = self._random_float(min_tonnage, max_tonnage, 2)

            # Calculate emissions based on typical intensities
            typical_intensities = {
                "steel_hot_rolled_coil": 1.85,
                "steel_cold_rolled_coil": 2.10,
                "steel_rebar": 1.55,
                "cement_portland": 0.67,
                "cement_clinker": 0.85,
                "aluminum_unwrought": 8.60,
                "aluminum_bars": 9.20,
                "fertilizer_ammonia": 2.40,
                "fertilizer_urea": 1.80,
                "electricity": 0.429,
                "hydrogen": 9.00,
            }

            intensity = typical_intensities.get(product_type, 1.0)
            variation = self._random_float(0.8, 1.2)
            actual_emissions = quantity * intensity * variation

            shipment = {
                "shipment_id": self._generate_id("CBAM"),
                "product_type": product_type,
                "cn_code": cn_code,
                "origin_country": self._random_choice(self.ORIGIN_COUNTRIES),
                "quantity_tonnes": quantity,
                "total_emissions_tco2e": round(actual_emissions, 4),
                "direct_emissions_tco2e": round(actual_emissions * 0.85, 4),
                "indirect_emissions_tco2e": round(actual_emissions * 0.15, 4),
                "import_date": self._random_date(2024, 2025),
                "supplier_name": f"Supplier {self.random.randint(1, 100)}",
                "supplier_installation_id": self._generate_id("INST"),
                "verification_status": self._random_choice(["verified", "unverified", "pending"]),
                "reporting_period": f"Q{self.random.randint(1, 4)} {self.random.randint(2024, 2025)}",
            }
            shipments.append(shipment)

        return shipments


class BuildingEnergyGenerator(BaseTestDataGenerator):
    """Generate building energy performance test data."""

    BUILDING_TYPES = [
        "office", "retail", "hotel", "hospital",
        "school", "university", "warehouse",
        "multifamily_residential", "data_center"
    ]

    CLIMATE_ZONES = ["1A", "2A", "2B", "3A", "3B", "3C", "4A", "4B", "4C", "5A", "5B", "6A", "6B", "7"]

    # Typical EUI ranges by building type (kWh/sqm/year)
    TYPICAL_EUI = {
        "office": (100, 300),
        "retail": (150, 400),
        "hotel": (200, 500),
        "hospital": (300, 700),
        "school": (80, 200),
        "university": (150, 350),
        "warehouse": (50, 150),
        "multifamily_residential": (80, 180),
        "data_center": (500, 2000),
    }

    def generate_buildings(self, num_buildings: int = 100) -> List[Dict[str, Any]]:
        """
        Generate building energy test data.

        Args:
            num_buildings: Number of buildings to generate

        Returns:
            List of building dictionaries
        """
        buildings = []

        for _ in range(num_buildings):
            building_type = self._random_choice(self.BUILDING_TYPES)
            climate_zone = self._random_choice(self.CLIMATE_ZONES)

            # Generate floor area
            floor_area_ranges = {
                "office": (1000, 100000),
                "retail": (500, 50000),
                "hotel": (5000, 150000),
                "hospital": (10000, 500000),
                "school": (2000, 30000),
                "university": (5000, 200000),
                "warehouse": (2000, 500000),
                "multifamily_residential": (1000, 50000),
                "data_center": (500, 50000),
            }

            min_area, max_area = floor_area_ranges.get(building_type, (1000, 50000))
            floor_area = self._random_float(min_area, max_area, 0)

            # Generate EUI based on building type
            min_eui, max_eui = self.TYPICAL_EUI.get(building_type, (100, 300))
            eui = self._random_float(min_eui, max_eui, 2)

            # Calculate total energy consumption
            energy_consumption = floor_area * eui

            building = {
                "building_id": self._generate_id("BLDG"),
                "building_name": f"Test Building {self._record_count}",
                "building_type": building_type,
                "floor_area_sqm": floor_area,
                "climate_zone": climate_zone,
                "year_built": self.random.randint(1950, 2024),
                "energy_consumption_kwh": round(energy_consumption, 2),
                "eui_kwh_per_sqm": eui,
                "reporting_year": self.random.randint(2022, 2025),
                "jurisdiction": self._random_choice(["NYC", "CA", "WA", "MA", "DC", "CO"]),
                "energy_star_score": self.random.randint(1, 100) if self.random.random() > 0.2 else None,
                "ghg_emissions_kgco2e_per_sqm": round(eui * 0.4, 2),  # Simplified
            }
            buildings.append(building)

        return buildings


class EUDRCommodityGenerator(BaseTestDataGenerator):
    """Generate EUDR commodity and supply chain test data."""

    COMMODITIES = ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soy", "wood"]

    CN_CODES = {
        "cattle": ["0102.21", "0102.29", "0201.10", "0201.20", "0201.30", "0202.10", "0202.20", "0202.30"],
        "cocoa": ["1801.00", "1802.00", "1803.10", "1803.20", "1804.00", "1805.00", "1806.10", "1806.20"],
        "coffee": ["0901.11", "0901.12", "0901.21", "0901.22", "0901.90"],
        "palm_oil": ["1511.10", "1511.90", "1513.11", "1513.19", "1513.21", "1513.29"],
        "rubber": ["4001.10", "4001.21", "4001.22", "4001.29", "4001.30"],
        "soy": ["1201.10", "1201.90", "1208.10", "2304.00"],
        "wood": ["4401.11", "4401.12", "4401.21", "4401.22", "4403.11", "4403.12", "4403.21"],
    }

    RISK_LEVELS = ["low", "standard", "high"]

    COUNTRIES = {
        "low": ["DE", "FR", "NL", "BE", "AT", "SE", "FI", "US", "CA", "AU"],
        "standard": ["BR", "AR", "MX", "TH", "VN", "PH", "MY", "ID", "CO", "PE"],
        "high": ["NG", "CM", "CI", "GH", "EC", "PY", "BO", "MM", "LA", "KH"],
    }

    def generate_commodities(self, num_records: int = 100) -> List[Dict[str, Any]]:
        """
        Generate EUDR commodity test data.

        Args:
            num_records: Number of records to generate

        Returns:
            List of commodity dictionaries
        """
        records = []

        for _ in range(num_records):
            commodity = self._random_choice(self.COMMODITIES)
            cn_code = self._random_choice(self.CN_CODES.get(commodity, ["9999.00"]))
            risk_level = self._random_choice(self.RISK_LEVELS)
            country = self._random_choice(self.COUNTRIES.get(risk_level, ["XX"]))

            # Generate plausible quantities
            quantity_ranges = {
                "cattle": (1, 1000),  # head
                "cocoa": (100, 50000),  # kg
                "coffee": (100, 30000),  # kg
                "palm_oil": (1000, 100000),  # kg
                "rubber": (100, 50000),  # kg
                "soy": (1000, 500000),  # kg
                "wood": (1, 10000),  # m3
            }

            min_qty, max_qty = quantity_ranges.get(commodity, (100, 10000))
            quantity = self._random_float(min_qty, max_qty, 2)

            # Generate geolocation
            lat = self._random_float(-60, 60, 6)
            lon = self._random_float(-180, 180, 6)

            record = {
                "shipment_id": self._generate_id("EUDR"),
                "commodity_type": commodity,
                "cn_code": cn_code,
                "quantity_kg": quantity if commodity != "cattle" else None,
                "quantity_head": quantity if commodity == "cattle" else None,
                "quantity_m3": quantity if commodity == "wood" else None,
                "origin_country": country,
                "risk_level": risk_level,
                "production_date": self._random_date(2021, 2025),
                "import_date": self._random_date(2024, 2025),
                "geolocation": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                    "precision_meters": self._random_float(1, 50, 1),
                },
                "deforestation_free": self.random.random() > 0.1,
                "cutoff_date_compliant": self.random.random() > 0.05,
                "legal_compliance": self.random.random() > 0.1,
                "traceability_score": self._random_float(0.5, 1.0, 2),
                "operator_eori": f"XX{self.random.randint(100000000, 999999999)}",
            }
            records.append(record)

        return records


class EdgeCaseGenerator(BaseTestDataGenerator):
    """Generate edge case and boundary condition test data."""

    def generate_boundary_values(self) -> List[Dict[str, Any]]:
        """Generate boundary value test cases."""
        return [
            # Zero values
            {"name": "zero_quantity", "value": 0.0, "expected_behavior": "handle_gracefully"},
            {"name": "zero_area", "value": 0.0, "expected_behavior": "raise_error"},
            {"name": "zero_emissions", "value": 0.0, "expected_behavior": "valid_result"},

            # Negative values
            {"name": "negative_quantity", "value": -100.0, "expected_behavior": "raise_error"},
            {"name": "negative_area", "value": -1000.0, "expected_behavior": "raise_error"},
            {"name": "negative_emissions", "value": -50.0, "expected_behavior": "raise_error"},

            # Very small values
            {"name": "tiny_quantity", "value": 0.000001, "expected_behavior": "handle_gracefully"},
            {"name": "tiny_area", "value": 0.001, "expected_behavior": "handle_gracefully"},

            # Very large values
            {"name": "huge_quantity", "value": 1e12, "expected_behavior": "handle_gracefully"},
            {"name": "huge_area", "value": 1e9, "expected_behavior": "handle_gracefully"},

            # Precision edge cases
            {"name": "max_precision", "value": 0.123456789012345, "expected_behavior": "truncate_precision"},
            {"name": "float_rounding", "value": 0.1 + 0.2, "expected_behavior": "handle_float_error"},

            # Special floats
            {"name": "infinity", "value": float("inf"), "expected_behavior": "raise_error"},
            {"name": "negative_infinity", "value": float("-inf"), "expected_behavior": "raise_error"},
        ]

    def generate_invalid_inputs(self) -> List[Dict[str, Any]]:
        """Generate invalid input test cases."""
        return [
            # Type errors
            {"name": "string_as_number", "value": "not_a_number", "field": "quantity"},
            {"name": "list_as_string", "value": ["a", "b"], "field": "fuel_type"},
            {"name": "dict_as_number", "value": {"key": "value"}, "field": "emissions"},

            # Missing required fields
            {"name": "missing_fuel_type", "missing": "fuel_type"},
            {"name": "missing_quantity", "missing": "quantity"},
            {"name": "missing_region", "missing": "region"},

            # Invalid enumerations
            {"name": "invalid_fuel_type", "value": "unicorn_tears", "field": "fuel_type"},
            {"name": "invalid_region", "value": "MARS", "field": "region"},
            {"name": "invalid_unit", "value": "parsecs", "field": "unit"},

            # Format errors
            {"name": "invalid_date_format", "value": "2025/01/01", "field": "date"},
            {"name": "invalid_coordinates", "value": [999, 999], "field": "coordinates"},
            {"name": "invalid_cn_code", "value": "ABCD.XX", "field": "cn_code"},
        ]


# =============================================================================
# Emission Data Generator
# =============================================================================


class EmissionDataGenerator(BaseTestDataGenerator):
    """
    Generate emission calculation test data with Decimal-precision factors.

    Produces Scope 1, Scope 2, and Scope 3 activity records together with
    deterministic emission factors suitable for verifying calculation engines.
    """

    # GHG Protocol scopes
    SCOPE1_SOURCES = ["stationary_combustion", "mobile_combustion", "process", "fugitive", "refrigerant"]
    SCOPE2_SOURCES = ["electricity", "steam", "heating", "cooling"]
    SCOPE3_CATEGORIES = [
        "purchased_goods", "capital_goods", "fuel_energy_activities",
        "upstream_transport", "waste", "business_travel", "employee_commuting",
        "upstream_leased", "downstream_transport", "processing_sold",
        "use_of_sold", "eol_sold", "downstream_leased",
        "franchises", "investments",
    ]

    # Deterministic emission factors (Decimal for zero-hallucination maths)
    _EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
        "natural_gas": {"kgCO2e_per_MJ": Decimal("0.0561"), "kgCO2e_per_therm": Decimal("5.30")},
        "diesel": {"kgCO2e_per_litre": Decimal("2.68"), "kgCO2e_per_gallon": Decimal("10.21")},
        "gasoline": {"kgCO2e_per_litre": Decimal("2.31"), "kgCO2e_per_gallon": Decimal("8.78")},
        "electricity_us": {"kgCO2e_per_kWh": Decimal("0.385")},
        "electricity_uk": {"kgCO2e_per_kWh": Decimal("0.207")},
        "electricity_eu": {"kgCO2e_per_kWh": Decimal("0.256")},
        "coal": {"kgCO2e_per_kg": Decimal("2.42")},
        "lpg": {"kgCO2e_per_litre": Decimal("1.56")},
    }

    def generate_emission_factors(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Return the full set of deterministic emission factors.

        Returns:
            Nested dict: fuel -> unit -> Decimal factor.
        """
        # Return a deep copy so tests cannot mutate the class constant
        return {
            fuel: dict(factors)
            for fuel, factors in self._EMISSION_FACTORS.items()
        }

    def generate_scope1_data(self, num_records: int = 50) -> List[Dict[str, Any]]:
        """
        Generate Scope 1 (direct emissions) activity records.

        Args:
            num_records: Number of records.

        Returns:
            List of Scope 1 activity dicts.
        """
        records: List[Dict[str, Any]] = []
        fuels_for_source = {
            "stationary_combustion": ["natural_gas", "diesel", "coal", "lpg"],
            "mobile_combustion": ["diesel", "gasoline"],
            "process": ["natural_gas"],
            "fugitive": ["natural_gas"],
            "refrigerant": ["R-410A", "HFC-134a"],
        }

        for _ in range(num_records):
            source = self._random_choice(self.SCOPE1_SOURCES)
            fuel = self._random_choice(fuels_for_source.get(source, ["natural_gas"]))
            quantity = self._random_float(10.0, 50000.0, 2)

            records.append({
                "record_id": self._generate_id("S1"),
                "scope": "scope_1",
                "source_type": source,
                "fuel_type": fuel,
                "quantity": quantity,
                "unit": self._random_choice(["MJ", "L", "kg", "kWh"]),
                "facility_id": f"FAC-{self.random.randint(1, 20):03d}",
                "reporting_year": self.random.randint(2022, 2025),
                "reporting_date": self._random_date(2022, 2025),
                "data_quality_score": self._random_float(0.7, 1.0, 2),
            })
        return records

    def generate_scope2_data(self, num_records: int = 30) -> List[Dict[str, Any]]:
        """
        Generate Scope 2 (indirect energy) activity records.

        Args:
            num_records: Number of records.

        Returns:
            List of Scope 2 activity dicts.
        """
        records: List[Dict[str, Any]] = []
        for _ in range(num_records):
            source = self._random_choice(self.SCOPE2_SOURCES)
            quantity = self._random_float(100.0, 500000.0, 2)
            region = self._random_choice(["US", "GB", "EU", "DE", "FR", "IN"])

            records.append({
                "record_id": self._generate_id("S2"),
                "scope": "scope_2",
                "source_type": source,
                "quantity_kwh": quantity,
                "region": region,
                "method": self._random_choice(["location_based", "market_based"]),
                "facility_id": f"FAC-{self.random.randint(1, 20):03d}",
                "reporting_year": self.random.randint(2022, 2025),
                "data_quality_score": self._random_float(0.8, 1.0, 2),
            })
        return records

    def generate_scope3_data(self, num_records: int = 100) -> List[Dict[str, Any]]:
        """
        Generate Scope 3 (value chain) activity records.

        Args:
            num_records: Number of records.

        Returns:
            List of Scope 3 activity dicts.
        """
        records: List[Dict[str, Any]] = []
        for _ in range(num_records):
            category = self._random_choice(self.SCOPE3_CATEGORIES)
            cat_index = self.SCOPE3_CATEGORIES.index(category) + 1
            spend = self._random_float(1000.0, 5000000.0, 2)
            ef = self._random_float(0.01, 2.0, 6)

            records.append({
                "record_id": self._generate_id("S3"),
                "scope": "scope_3",
                "category": category,
                "category_number": cat_index,
                "activity_data": spend,
                "activity_unit": self._random_choice(["USD", "EUR", "GBP", "kg", "tonne-km"]),
                "emission_factor": ef,
                "supplier_name": f"Supplier-{self.random.randint(1, 200):04d}",
                "reporting_year": self.random.randint(2022, 2025),
                "data_quality_score": self._random_float(0.5, 1.0, 2),
            })
        return records


# =============================================================================
# Supply Chain Data Generator
# =============================================================================


class SupplyChainDataGenerator(BaseTestDataGenerator):
    """
    Generate supply chain test data (suppliers, shipments, invoices).

    Useful for EUDR traceability, CBAM upstream mapping, and Scope 3
    upstream transport tests.
    """

    SUPPLIER_TIERS = ["tier_1", "tier_2", "tier_3", "raw_material"]
    TRANSPORT_MODES = ["road", "rail", "sea", "air", "multimodal"]
    INCOTERMS = ["EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP", "FAS", "FOB", "CFR", "CIF"]
    CURRENCIES = ["USD", "EUR", "GBP", "INR", "CNY", "BRL"]

    COUNTRIES = [
        "US", "GB", "DE", "FR", "IT", "ES", "CN", "IN", "BR",
        "JP", "KR", "AU", "CA", "MX", "ZA", "TR", "ID", "TH",
    ]

    def generate_suppliers(self, num_suppliers: int = 50) -> List[Dict[str, Any]]:
        """
        Generate supplier master records.

        Args:
            num_suppliers: Number of suppliers.

        Returns:
            List of supplier dicts.
        """
        suppliers: List[Dict[str, Any]] = []
        for _ in range(num_suppliers):
            country = self._random_choice(self.COUNTRIES)
            suppliers.append({
                "supplier_id": self._generate_id("SUP"),
                "supplier_name": f"Supplier {self._record_count}",
                "tier": self._random_choice(self.SUPPLIER_TIERS),
                "country": country,
                "region": self._random_choice(["APAC", "EMEA", "AMERICAS"]),
                "industry": self._random_choice([
                    "chemicals", "metals", "textiles", "electronics",
                    "agriculture", "energy", "logistics", "packaging",
                ]),
                "annual_spend_usd": self._random_float(10000, 50000000, 2),
                "risk_score": self._random_float(0.0, 1.0, 2),
                "sustainability_rating": self._random_choice(["A", "B", "C", "D", "unrated"]),
                "onboarding_date": self._random_date(2018, 2024),
                "active": self.random.random() > 0.1,
            })
        return suppliers

    def generate_shipments(self, num_shipments: int = 100) -> List[Dict[str, Any]]:
        """
        Generate logistics shipment records.

        Args:
            num_shipments: Number of shipments.

        Returns:
            List of shipment dicts.
        """
        shipments: List[Dict[str, Any]] = []
        for _ in range(num_shipments):
            origin = self._random_choice(self.COUNTRIES)
            dest = self._random_choice([c for c in self.COUNTRIES if c != origin])
            weight = self._random_float(100, 50000, 2)
            distance = self._random_float(50, 20000, 0)
            mode = self._random_choice(self.TRANSPORT_MODES)

            shipments.append({
                "shipment_id": self._generate_id("SHP"),
                "supplier_id": f"SUP-{self.random.randint(1, 50):06d}",
                "origin_country": origin,
                "destination_country": dest,
                "transport_mode": mode,
                "weight_kg": weight,
                "distance_km": distance,
                "tonne_km": round(weight * distance / 1000, 2),
                "departure_date": self._random_date(2023, 2025),
                "incoterm": self._random_choice(self.INCOTERMS),
                "customs_cleared": self.random.random() > 0.05,
            })
        return shipments

    def generate_invoices(self, num_invoices: int = 200) -> List[Dict[str, Any]]:
        """
        Generate purchase invoice records.

        Args:
            num_invoices: Number of invoices.

        Returns:
            List of invoice dicts.
        """
        invoices: List[Dict[str, Any]] = []
        for _ in range(num_invoices):
            currency = self._random_choice(self.CURRENCIES)
            amount = self._random_float(500, 2000000, 2)

            invoices.append({
                "invoice_id": self._generate_id("INV"),
                "supplier_id": f"SUP-{self.random.randint(1, 50):06d}",
                "invoice_date": self._random_date(2023, 2025),
                "due_date": self._random_date(2023, 2025),
                "amount": amount,
                "currency": currency,
                "line_items": self.random.randint(1, 15),
                "category": self._random_choice([
                    "raw_materials", "components", "services",
                    "energy", "logistics", "equipment",
                ]),
                "payment_status": self._random_choice(["paid", "pending", "overdue"]),
                "cost_center": f"CC-{self.random.randint(1, 30):03d}",
            })
        return invoices


# =============================================================================
# Facility Data Generator
# =============================================================================


class FacilityDataGenerator(BaseTestDataGenerator):
    """
    Generate facility and building test data.

    Produces records suitable for site-level GHG inventories,
    energy audits, and multi-site consolidation tests.
    """

    FACILITY_TYPES = [
        "manufacturing_plant", "warehouse", "office", "retail_store",
        "distribution_center", "data_center", "laboratory", "hospital",
    ]

    OWNERSHIP_TYPES = ["owned", "leased", "joint_venture", "franchise"]

    def generate_facilities(self, num_facilities: int = 20) -> List[Dict[str, Any]]:
        """
        Generate facility master records.

        Args:
            num_facilities: Number of facilities.

        Returns:
            List of facility dicts with operational metadata.
        """
        facilities: List[Dict[str, Any]] = []
        for _ in range(num_facilities):
            ftype = self._random_choice(self.FACILITY_TYPES)
            area = self._random_float(500, 200000, 0)
            country = self._random_choice(["US", "GB", "DE", "IN", "CN", "BR", "AU", "JP"])

            facilities.append({
                "facility_id": self._generate_id("FAC"),
                "facility_name": f"Test Facility {self._record_count}",
                "facility_type": ftype,
                "ownership_type": self._random_choice(self.OWNERSHIP_TYPES),
                "country": country,
                "state_province": f"State-{self.random.randint(1, 50)}",
                "floor_area_sqm": area,
                "operational_hours_per_year": self._random_float(2000, 8760, 0),
                "headcount": self.random.randint(10, 5000),
                "year_commissioned": self.random.randint(1960, 2024),
                "scope1_sources": self._random_choice([
                    ["stationary_combustion"],
                    ["stationary_combustion", "mobile_combustion"],
                    ["stationary_combustion", "process", "fugitive"],
                ]),
                "has_onsite_renewables": self.random.random() > 0.7,
                "equity_share_pct": self._random_float(10, 100, 1),
            })
        return facilities

    def generate_buildings(self, num_buildings: int = 30) -> List[Dict[str, Any]]:
        """
        Generate building energy profile records.

        Args:
            num_buildings: Number of buildings.

        Returns:
            List of building dicts with energy/envelope attributes.
        """
        buildings: List[Dict[str, Any]] = []
        envelope_types = ["masonry", "curtain_wall", "precast_concrete", "steel_frame", "wood_frame"]

        for _ in range(num_buildings):
            btype = self._random_choice([
                "office", "retail", "hotel", "hospital",
                "school", "warehouse", "data_center",
            ])
            area = self._random_float(500, 100000, 0)
            floors = self.random.randint(1, 40)

            buildings.append({
                "building_id": self._generate_id("BLDG"),
                "building_name": f"Building {self._record_count}",
                "building_type": btype,
                "floor_area_sqm": area,
                "num_floors": floors,
                "envelope_type": self._random_choice(envelope_types),
                "year_built": self.random.randint(1950, 2024),
                "hvac_type": self._random_choice(["central_chiller", "split", "vrf", "district"]),
                "lighting_type": self._random_choice(["led", "fluorescent", "mixed"]),
                "eui_kwh_per_sqm": self._random_float(50, 600, 2),
                "climate_zone": self._random_choice(["1A", "2A", "3A", "4A", "5A", "6A", "7"]),
                "energy_star_score": self.random.randint(1, 100) if self.random.random() > 0.3 else None,
            })
        return buildings


# =============================================================================
# Compliance Data Generator
# =============================================================================


class ComplianceDataGenerator(BaseTestDataGenerator):
    """
    Generate compliance and regulatory test data.

    Produces audit trails, compliance reports, and regulatory filing
    records for CSRD, CBAM, EUDR, ISO 14064, and GHG Protocol tests.
    """

    FRAMEWORKS = ["CSRD", "CBAM", "EUDR", "GHG_PROTOCOL", "ISO_14064", "CDP", "TCFD", "SBTi"]
    COMPLIANCE_STATUSES = ["compliant", "non_compliant", "partially_compliant", "pending_review"]
    AUDIT_ACTIONS = [
        "data_submitted", "calculation_executed", "report_generated",
        "review_requested", "review_completed", "approved", "rejected",
        "declaration_filed", "verification_initiated", "verification_completed",
    ]
    ACTOR_ROLES = ["data_entry", "analyst", "reviewer", "approver", "auditor", "system"]

    def generate_audit_trail(self, num_entries: int = 100) -> List[Dict[str, Any]]:
        """
        Generate audit trail entries.

        Args:
            num_entries: Number of audit entries.

        Returns:
            List of audit entry dicts with provenance hashes.
        """
        entries: List[Dict[str, Any]] = []
        base_time = datetime(2024, 1, 1, 8, 0, 0)

        for i in range(num_entries):
            action = self._random_choice(self.AUDIT_ACTIONS)
            actor_role = self._random_choice(self.ACTOR_ROLES)
            actor_id = f"user-{actor_role}-{self.random.randint(1, 20):03d}"

            # Build deterministic provenance hash
            payload = f"{action}:{actor_id}:{i}"
            provenance_hash = hashlib.sha256(payload.encode()).hexdigest()

            entry_time = base_time + timedelta(
                days=self.random.randint(0, 365),
                hours=self.random.randint(0, 23),
                minutes=self.random.randint(0, 59),
            )

            entries.append({
                "entry_id": self._generate_id("AUD"),
                "timestamp": entry_time.isoformat(),
                "action": action,
                "actor": actor_id,
                "actor_role": actor_role,
                "resource_type": self._random_choice(["emission_record", "report", "declaration", "dataset"]),
                "resource_id": f"RES-{self.random.randint(1, 500):06d}",
                "framework": self._random_choice(self.FRAMEWORKS),
                "provenance_hash": provenance_hash,
                "ip_address": "127.0.0.1",
                "details": f"Test audit action: {action}",
            })
        return entries

    def generate_compliance_report(self, num_reports: int = 20) -> List[Dict[str, Any]]:
        """
        Generate compliance report metadata records.

        Args:
            num_reports: Number of reports.

        Returns:
            List of compliance report dicts.
        """
        reports: List[Dict[str, Any]] = []
        for _ in range(num_reports):
            framework = self._random_choice(self.FRAMEWORKS)
            status = self._random_choice(self.COMPLIANCE_STATUSES)
            year = self.random.randint(2022, 2025)
            total_findings = self.random.randint(0, 50)
            critical_findings = self.random.randint(0, min(total_findings, 5))

            payload = f"{framework}:{year}:{self._record_count}"
            provenance_hash = hashlib.sha256(payload.encode()).hexdigest()

            reports.append({
                "report_id": self._generate_id("RPT"),
                "framework": framework,
                "reporting_year": year,
                "reporting_period": f"FY{year}",
                "status": status,
                "total_findings": total_findings,
                "critical_findings": critical_findings,
                "completeness_pct": self._random_float(60, 100, 1),
                "generated_at": self._random_date(year, year),
                "approved_by": f"user-approver-{self.random.randint(1, 5):03d}" if status == "compliant" else None,
                "provenance_hash": provenance_hash,
                "tenant_id": "tenant-test-001",
            })
        return reports

    def generate_regulatory_filings(self, num_filings: int = 10) -> List[Dict[str, Any]]:
        """
        Generate regulatory filing records (CBAM declarations, EUDR statements, etc.).

        Args:
            num_filings: Number of filings.

        Returns:
            List of filing dicts.
        """
        filings: List[Dict[str, Any]] = []
        filing_types = {
            "CBAM": "quarterly_declaration",
            "EUDR": "due_diligence_statement",
            "CSRD": "annual_sustainability_report",
            "CDP": "annual_questionnaire",
        }

        for _ in range(num_filings):
            framework = self._random_choice(list(filing_types.keys()))
            filing_type = filing_types[framework]

            filings.append({
                "filing_id": self._generate_id("FIL"),
                "framework": framework,
                "filing_type": filing_type,
                "period": f"Q{self.random.randint(1, 4)} {self.random.randint(2024, 2025)}",
                "submitted_at": self._random_date(2024, 2025),
                "status": self._random_choice(["draft", "submitted", "accepted", "rejected"]),
                "reference_number": f"{framework}-{self.random.randint(100000, 999999)}",
                "authority": self._random_choice(["EU_Commission", "National_CA", "CDP_Global"]),
                "tenant_id": "tenant-test-001",
            })
        return filings


# =============================================================================
# Convenience Functions
# =============================================================================


def create_test_generators(seed: int = 42) -> Dict[str, BaseTestDataGenerator]:
    """
    Create all test data generators with the same seed.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dictionary of generator instances keyed by short name.
    """
    return {
        "industrial": IndustrialTestDataGenerator(seed),
        "cbam": CBAMShipmentGenerator(seed),
        "building": BuildingEnergyGenerator(seed),
        "eudr": EUDRCommodityGenerator(seed),
        "edge_cases": EdgeCaseGenerator(seed),
        "emissions": EmissionDataGenerator(seed),
        "supply_chain": SupplyChainDataGenerator(seed),
        "facility": FacilityDataGenerator(seed),
        "compliance": ComplianceDataGenerator(seed),
    }
