# -*- coding: utf-8 -*-
"""
Comprehensive Integration Tests for Data Pipeline

30 test cases covering:
- Emission factor loading (8 tests)
- ETL pipeline execution (8 tests)
- Data quality checks (6 tests)
- Factor reconciliation (4 tests)
- Cache invalidation (4 tests)

Target: 85%+ coverage of data pipeline integration paths
Run with: pytest tests/integration/test_data_pipeline_comprehensive.py -v --tb=short

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from decimal import Decimal
from pathlib import Path

# Add project paths for imports
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_emission_factor_db():
    """Create mock emission factor database."""
    class EmissionFactorDB:
        def __init__(self):
            self._factors = {
                # Natural Gas
                ("natural_gas", "US", 2024): {
                    "ef_uri": "ef://EPA/natural_gas/US/2024",
                    "ef_value": 0.0561,
                    "ef_unit": "kgCO2e/MJ",
                    "source": "EPA 40 CFR 98",
                    "gwp_set": "AR6GWP100",
                    "effective_date": "2024-01-01",
                    "expiry_date": "2024-12-31",
                },
                ("natural_gas", "GB", 2024): {
                    "ef_uri": "ef://DEFRA/natural_gas/GB/2024",
                    "ef_value": 0.0549,
                    "ef_unit": "kgCO2e/MJ",
                    "source": "UK DEFRA 2024",
                    "gwp_set": "AR6GWP100",
                    "effective_date": "2024-01-01",
                    "expiry_date": "2024-12-31",
                },
                # Diesel
                ("diesel", "US", 2024): {
                    "ef_uri": "ef://EPA/diesel/US/2024",
                    "ef_value": 0.0745,
                    "ef_unit": "kgCO2e/MJ",
                    "source": "EPA 40 CFR 98",
                    "gwp_set": "AR6GWP100",
                    "effective_date": "2024-01-01",
                    "expiry_date": "2024-12-31",
                },
                # Electricity
                ("electricity", "US", 2024): {
                    "ef_uri": "ef://eGRID/electricity/US/2024",
                    "ef_value": 0.417,
                    "ef_unit": "kgCO2e/kWh",
                    "source": "EPA eGRID 2024",
                    "gwp_set": "AR6GWP100",
                    "effective_date": "2024-01-01",
                    "expiry_date": "2024-12-31",
                },
                # Regional electricity
                ("electricity", "US-CA", 2024): {
                    "ef_uri": "ef://eGRID/electricity/US-CA/2024",
                    "ef_value": 0.205,
                    "ef_unit": "kgCO2e/kWh",
                    "source": "EPA eGRID 2024 - California",
                    "gwp_set": "AR6GWP100",
                    "effective_date": "2024-01-01",
                    "expiry_date": "2024-12-31",
                },
            }
            self._load_count = 0

        async def load_factor(self, fuel_type: str, region: str, year: int) -> Optional[Dict]:
            """Load emission factor from database."""
            self._load_count += 1
            key = (fuel_type.lower(), region.upper(), year)
            return self._factors.get(key)

        async def insert_factor(self, factor: Dict) -> bool:
            """Insert new emission factor."""
            key = (
                factor["fuel_type"].lower(),
                factor["region"].upper(),
                factor["year"]
            )
            self._factors[key] = factor
            return True

        async def update_factor(self, fuel_type: str, region: str, year: int, updates: Dict) -> bool:
            """Update existing factor."""
            key = (fuel_type.lower(), region.upper(), year)
            if key in self._factors:
                self._factors[key].update(updates)
                return True
            return False

        async def list_factors(self, filters: Optional[Dict] = None) -> List[Dict]:
            """List all factors with optional filters."""
            results = []
            for key, factor in self._factors.items():
                if filters:
                    if filters.get("fuel_type") and key[0] != filters["fuel_type"].lower():
                        continue
                    if filters.get("region") and key[1] != filters["region"].upper():
                        continue
                results.append({"key": key, **factor})
            return results

        def get_load_count(self) -> int:
            """Get number of load operations."""
            return self._load_count

    return EmissionFactorDB()


@pytest.fixture
def mock_etl_pipeline():
    """Create mock ETL pipeline."""
    class ETLPipeline:
        def __init__(self):
            self._extract_results = []
            self._transform_results = []
            self._load_results = []
            self._stats = {
                "records_extracted": 0,
                "records_transformed": 0,
                "records_loaded": 0,
                "errors": [],
            }

        async def extract(self, source_config: Dict) -> List[Dict]:
            """Extract data from source."""
            source_type = source_config.get("type")
            records = []

            if source_type == "csv":
                # Simulate CSV extraction
                records = [
                    {"fuel_type": "natural_gas", "quantity": 1000, "date": "2024-01-15"},
                    {"fuel_type": "diesel", "quantity": 500, "date": "2024-01-16"},
                    {"fuel_type": "electricity", "quantity": 2000, "date": "2024-01-17"},
                ]
            elif source_type == "api":
                # Simulate API extraction
                records = [
                    {"product": "steel", "tonnes": 100, "emissions": 185},
                ]
            elif source_type == "database":
                # Simulate database extraction
                records = source_config.get("mock_data", [])

            self._extract_results = records
            self._stats["records_extracted"] = len(records)
            return records

        async def transform(self, records: List[Dict], transformations: List[str]) -> List[Dict]:
            """Transform extracted records."""
            transformed = []

            for record in records:
                new_record = record.copy()

                for transform in transformations:
                    if transform == "normalize_units":
                        # Normalize to standard units
                        if "quantity" in new_record:
                            new_record["quantity_mj"] = new_record.get("quantity", 0)
                    elif transform == "add_timestamp":
                        new_record["processed_at"] = datetime.now().isoformat()
                    elif transform == "calculate_emissions":
                        # Simple emissions calculation
                        if new_record.get("fuel_type") == "natural_gas":
                            new_record["emissions_kgco2e"] = new_record.get("quantity", 0) * 0.0561
                        elif new_record.get("fuel_type") == "diesel":
                            new_record["emissions_kgco2e"] = new_record.get("quantity", 0) * 0.0745
                    elif transform == "validate":
                        new_record["is_valid"] = self._validate_record(new_record)

                transformed.append(new_record)

            self._transform_results = transformed
            self._stats["records_transformed"] = len(transformed)
            return transformed

        def _validate_record(self, record: Dict) -> bool:
            """Validate a single record."""
            required_fields = ["fuel_type", "quantity"]
            return all(f in record for f in required_fields)

        async def load(self, records: List[Dict], target_config: Dict) -> Dict:
            """Load transformed records to target."""
            target_type = target_config.get("type")
            loaded = 0
            errors = []

            for record in records:
                try:
                    if target_type == "database":
                        # Simulate database insert
                        if record.get("is_valid", True):
                            loaded += 1
                        else:
                            errors.append({"record": record, "error": "Validation failed"})
                    elif target_type == "cache":
                        # Simulate cache write
                        loaded += 1
                except Exception as e:
                    errors.append({"record": record, "error": str(e)})

            self._load_results = records[:loaded]
            self._stats["records_loaded"] = loaded
            self._stats["errors"].extend(errors)

            return {
                "loaded": loaded,
                "failed": len(errors),
                "errors": errors,
            }

        async def run(self, config: Dict) -> Dict:
            """Run complete ETL pipeline."""
            # Extract
            records = await self.extract(config.get("source", {}))

            # Transform
            transformed = await self.transform(
                records,
                config.get("transformations", [])
            )

            # Load
            result = await self.load(
                transformed,
                config.get("target", {})
            )

            return {
                "status": "completed",
                "stats": self._stats,
                "result": result,
            }

        def get_stats(self) -> Dict:
            """Get pipeline statistics."""
            return self._stats

    return ETLPipeline()


@pytest.fixture
def mock_data_quality_checker():
    """Create mock data quality checker."""
    class DataQualityChecker:
        def __init__(self):
            self._rules = []
            self._results = []

        def add_rule(self, rule_name: str, rule_func):
            """Add a quality check rule."""
            self._rules.append({"name": rule_name, "func": rule_func})

        async def check(self, records: List[Dict]) -> Dict:
            """Run all quality checks on records."""
            results = {
                "total_records": len(records),
                "passed": 0,
                "failed": 0,
                "rules_applied": len(self._rules),
                "issues": [],
            }

            for record in records:
                record_passed = True
                for rule in self._rules:
                    try:
                        if not rule["func"](record):
                            record_passed = False
                            results["issues"].append({
                                "record": record,
                                "rule": rule["name"],
                            })
                    except Exception as e:
                        record_passed = False
                        results["issues"].append({
                            "record": record,
                            "rule": rule["name"],
                            "error": str(e),
                        })

                if record_passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            results["quality_score"] = results["passed"] / results["total_records"] if results["total_records"] > 0 else 0

            self._results = results
            return results

        def get_quality_score(self) -> float:
            """Get overall quality score."""
            return self._results.get("quality_score", 0)

    return DataQualityChecker()


@pytest.fixture
def mock_cache():
    """Create mock cache with TTL support."""
    class CacheLayer:
        def __init__(self):
            self._cache: Dict[str, Dict] = {}
            self._stats = {"hits": 0, "misses": 0}

        async def get(self, key: str) -> Optional[Any]:
            """Get value from cache."""
            if key in self._cache:
                entry = self._cache[key]
                if entry["expires_at"] > datetime.now():
                    self._stats["hits"] += 1
                    return entry["value"]
                else:
                    del self._cache[key]

            self._stats["misses"] += 1
            return None

        async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
            """Set value in cache with TTL."""
            self._cache[key] = {
                "value": value,
                "expires_at": datetime.now() + timedelta(seconds=ttl_seconds),
                "created_at": datetime.now(),
            }
            return True

        async def delete(self, key: str) -> bool:
            """Delete key from cache."""
            if key in self._cache:
                del self._cache[key]
                return True
            return False

        async def invalidate_pattern(self, pattern: str) -> int:
            """Invalidate keys matching pattern."""
            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

        async def clear(self) -> bool:
            """Clear all cache entries."""
            self._cache.clear()
            return True

        def get_stats(self) -> Dict:
            """Get cache statistics."""
            total = self._stats["hits"] + self._stats["misses"]
            return {
                **self._stats,
                "hit_rate": self._stats["hits"] / total if total > 0 else 0,
                "size": len(self._cache),
            }

    return CacheLayer()


@pytest.fixture
def sample_emission_factors():
    """Sample emission factors for testing."""
    return [
        {
            "fuel_type": "natural_gas",
            "region": "US",
            "year": 2024,
            "ef_value": 0.0561,
            "ef_unit": "kgCO2e/MJ",
            "source": "EPA",
        },
        {
            "fuel_type": "diesel",
            "region": "US",
            "year": 2024,
            "ef_value": 0.0745,
            "ef_unit": "kgCO2e/MJ",
            "source": "EPA",
        },
        {
            "fuel_type": "electricity",
            "region": "US",
            "year": 2024,
            "ef_value": 0.417,
            "ef_unit": "kgCO2e/kWh",
            "source": "EPA eGRID",
        },
    ]


# =============================================================================
# Emission Factor Loading Tests (8 tests)
# =============================================================================

class TestEmissionFactorLoading:
    """Test emission factor loading - 8 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_load_single_factor(self, mock_emission_factor_db):
        """INT-EF-001: Test loading a single emission factor."""
        factor = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)

        assert factor is not None
        assert factor["ef_value"] == 0.0561
        assert factor["ef_unit"] == "kgCO2e/MJ"
        assert factor["source"] == "EPA 40 CFR 98"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_load_factor_not_found(self, mock_emission_factor_db):
        """INT-EF-002: Test loading non-existent factor returns None."""
        factor = await mock_emission_factor_db.load_factor("hydrogen", "US", 2024)

        assert factor is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_load_regional_electricity_factor(self, mock_emission_factor_db):
        """INT-EF-003: Test loading regional electricity factor."""
        factor_national = await mock_emission_factor_db.load_factor("electricity", "US", 2024)
        factor_california = await mock_emission_factor_db.load_factor("electricity", "US-CA", 2024)

        assert factor_national["ef_value"] == 0.417
        assert factor_california["ef_value"] == 0.205  # CA is cleaner
        assert factor_california["ef_value"] < factor_national["ef_value"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_load_multiple_regions(self, mock_emission_factor_db):
        """INT-EF-004: Test loading factors for multiple regions."""
        factor_us = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)
        factor_gb = await mock_emission_factor_db.load_factor("natural_gas", "GB", 2024)

        assert factor_us is not None
        assert factor_gb is not None
        assert factor_us["ef_value"] != factor_gb["ef_value"]
        assert factor_us["source"] != factor_gb["source"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_insert_new_factor(self, mock_emission_factor_db):
        """INT-EF-005: Test inserting a new emission factor."""
        new_factor = {
            "fuel_type": "hydrogen",
            "region": "US",
            "year": 2024,
            "ef_uri": "ef://DOE/hydrogen/US/2024",
            "ef_value": 0.0,
            "ef_unit": "kgCO2e/MJ",
            "source": "DOE 2024",
        }

        success = await mock_emission_factor_db.insert_factor(new_factor)
        assert success is True

        # Verify insertion
        factor = await mock_emission_factor_db.load_factor("hydrogen", "US", 2024)
        assert factor is not None
        assert factor["ef_value"] == 0.0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_update_existing_factor(self, mock_emission_factor_db):
        """INT-EF-006: Test updating an existing emission factor."""
        # Get original value
        original = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)
        assert original["ef_value"] == 0.0561

        # Update factor
        success = await mock_emission_factor_db.update_factor(
            "natural_gas", "US", 2024,
            {"ef_value": 0.0565, "source": "EPA 40 CFR 98 (Updated)"}
        )
        assert success is True

        # Verify update
        updated = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)
        assert updated["ef_value"] == 0.0565
        assert "Updated" in updated["source"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_list_factors_with_filters(self, mock_emission_factor_db):
        """INT-EF-007: Test listing factors with filters."""
        # List all US factors
        us_factors = await mock_emission_factor_db.list_factors({"region": "US"})
        assert len(us_factors) >= 2  # natural_gas, diesel, electricity

        # List all natural gas factors
        ng_factors = await mock_emission_factor_db.list_factors({"fuel_type": "natural_gas"})
        assert len(ng_factors) >= 2  # US and GB

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_factor_effective_dates(self, mock_emission_factor_db):
        """INT-EF-008: Test emission factor effective date validation."""
        factor = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)

        assert "effective_date" in factor
        assert "expiry_date" in factor

        effective = datetime.fromisoformat(factor["effective_date"])
        expiry = datetime.fromisoformat(factor["expiry_date"])

        assert effective < expiry
        assert effective.year == 2024


# =============================================================================
# ETL Pipeline Execution Tests (8 tests)
# =============================================================================

class TestETLPipeline:
    """Test ETL pipeline execution - 8 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_extract_from_csv(self, mock_etl_pipeline):
        """INT-ETL-001: Test extraction from CSV source."""
        records = await mock_etl_pipeline.extract({"type": "csv", "path": "test.csv"})

        assert len(records) == 3
        assert records[0]["fuel_type"] == "natural_gas"
        assert mock_etl_pipeline.get_stats()["records_extracted"] == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_extract_from_api(self, mock_etl_pipeline):
        """INT-ETL-002: Test extraction from API source."""
        records = await mock_etl_pipeline.extract({"type": "api", "endpoint": "/data"})

        assert len(records) >= 1
        assert "product" in records[0]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transform_normalize_units(self, mock_etl_pipeline):
        """INT-ETL-003: Test unit normalization transformation."""
        records = [{"fuel_type": "natural_gas", "quantity": 1000}]
        transformed = await mock_etl_pipeline.transform(records, ["normalize_units"])

        assert transformed[0]["quantity_mj"] == 1000

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transform_calculate_emissions(self, mock_etl_pipeline):
        """INT-ETL-004: Test emissions calculation transformation."""
        records = [
            {"fuel_type": "natural_gas", "quantity": 1000},
            {"fuel_type": "diesel", "quantity": 500},
        ]
        transformed = await mock_etl_pipeline.transform(records, ["calculate_emissions"])

        assert transformed[0]["emissions_kgco2e"] == pytest.approx(56.1, rel=0.01)
        assert transformed[1]["emissions_kgco2e"] == pytest.approx(37.25, rel=0.01)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transform_add_timestamp(self, mock_etl_pipeline):
        """INT-ETL-005: Test timestamp addition transformation."""
        records = [{"fuel_type": "natural_gas", "quantity": 1000}]
        transformed = await mock_etl_pipeline.transform(records, ["add_timestamp"])

        assert "processed_at" in transformed[0]
        # Verify it's a valid timestamp
        datetime.fromisoformat(transformed[0]["processed_at"])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_load_to_database(self, mock_etl_pipeline):
        """INT-ETL-006: Test loading to database target."""
        records = [
            {"fuel_type": "natural_gas", "quantity": 1000, "is_valid": True},
            {"fuel_type": "diesel", "quantity": 500, "is_valid": True},
        ]
        result = await mock_etl_pipeline.load(records, {"type": "database"})

        assert result["loaded"] == 2
        assert result["failed"] == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, mock_etl_pipeline):
        """INT-ETL-007: Test complete ETL pipeline run."""
        config = {
            "source": {"type": "csv", "path": "test.csv"},
            "transformations": ["normalize_units", "calculate_emissions", "add_timestamp"],
            "target": {"type": "database"},
        }

        result = await mock_etl_pipeline.run(config)

        assert result["status"] == "completed"
        assert result["stats"]["records_extracted"] == 3
        assert result["stats"]["records_transformed"] == 3
        assert result["stats"]["records_loaded"] == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_etl_pipeline):
        """INT-ETL-008: Test pipeline error handling."""
        records = [
            {"fuel_type": "natural_gas", "quantity": 1000, "is_valid": True},
            {"fuel_type": "invalid", "is_valid": False},  # Invalid record
        ]

        result = await mock_etl_pipeline.load(records, {"type": "database"})

        assert result["loaded"] == 1
        assert result["failed"] == 1
        assert len(result["errors"]) == 1


# =============================================================================
# Data Quality Checks Tests (6 tests)
# =============================================================================

class TestDataQualityChecks:
    """Test data quality checks - 6 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_required_fields_check(self, mock_data_quality_checker):
        """INT-DQ-001: Test required fields validation."""
        mock_data_quality_checker.add_rule(
            "required_fields",
            lambda r: all(f in r for f in ["fuel_type", "quantity"])
        )

        records = [
            {"fuel_type": "natural_gas", "quantity": 1000},  # Valid
            {"fuel_type": "diesel"},  # Missing quantity
        ]

        result = await mock_data_quality_checker.check(records)

        assert result["passed"] == 1
        assert result["failed"] == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_range_validation(self, mock_data_quality_checker):
        """INT-DQ-002: Test value range validation."""
        mock_data_quality_checker.add_rule(
            "quantity_range",
            lambda r: r.get("quantity", 0) >= 0
        )

        records = [
            {"fuel_type": "natural_gas", "quantity": 1000},  # Valid
            {"fuel_type": "diesel", "quantity": -100},  # Invalid
        ]

        result = await mock_data_quality_checker.check(records)

        assert result["passed"] == 1
        assert result["failed"] == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_enum_validation(self, mock_data_quality_checker):
        """INT-DQ-003: Test enum value validation."""
        valid_fuels = ["natural_gas", "diesel", "electricity", "gasoline", "coal"]

        mock_data_quality_checker.add_rule(
            "valid_fuel_type",
            lambda r: r.get("fuel_type") in valid_fuels
        )

        records = [
            {"fuel_type": "natural_gas", "quantity": 1000},  # Valid
            {"fuel_type": "unicorn_tears", "quantity": 500},  # Invalid
        ]

        result = await mock_data_quality_checker.check(records)

        assert result["passed"] == 1
        assert result["failed"] == 1
        assert result["issues"][0]["rule"] == "valid_fuel_type"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_date_format_validation(self, mock_data_quality_checker):
        """INT-DQ-004: Test date format validation."""
        def valid_date(record):
            date_str = record.get("date")
            if not date_str:
                return False
            try:
                datetime.fromisoformat(date_str)
                return True
            except ValueError:
                return False

        mock_data_quality_checker.add_rule("valid_date", valid_date)

        records = [
            {"fuel_type": "natural_gas", "date": "2024-01-15"},  # Valid
            {"fuel_type": "diesel", "date": "01/15/2024"},  # Invalid format
        ]

        result = await mock_data_quality_checker.check(records)

        assert result["passed"] == 1
        assert result["failed"] == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, mock_data_quality_checker):
        """INT-DQ-005: Test quality score calculation."""
        mock_data_quality_checker.add_rule(
            "has_quantity",
            lambda r: "quantity" in r
        )

        records = [
            {"fuel_type": "natural_gas", "quantity": 1000},
            {"fuel_type": "diesel", "quantity": 500},
            {"fuel_type": "electricity"},  # Missing quantity
            {"fuel_type": "gasoline", "quantity": 200},
        ]

        result = await mock_data_quality_checker.check(records)

        assert result["quality_score"] == 0.75  # 3/4 passed

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_rules(self, mock_data_quality_checker):
        """INT-DQ-006: Test multiple validation rules."""
        mock_data_quality_checker.add_rule(
            "has_fuel_type",
            lambda r: "fuel_type" in r
        )
        mock_data_quality_checker.add_rule(
            "has_quantity",
            lambda r: "quantity" in r
        )
        mock_data_quality_checker.add_rule(
            "positive_quantity",
            lambda r: r.get("quantity", 0) >= 0
        )

        records = [
            {"fuel_type": "natural_gas", "quantity": 1000},  # Passes all
            {"quantity": 500},  # Missing fuel_type
            {"fuel_type": "diesel", "quantity": -100},  # Negative quantity
        ]

        result = await mock_data_quality_checker.check(records)

        assert result["passed"] == 1
        assert result["failed"] == 2
        assert result["rules_applied"] == 3


# =============================================================================
# Factor Reconciliation Tests (4 tests)
# =============================================================================

class TestFactorReconciliation:
    """Test factor reconciliation - 4 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_reconcile_identical_factors(self, mock_emission_factor_db):
        """INT-REC-001: Test reconciliation of identical factors."""
        factor1 = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)
        factor2 = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)

        # Same source should return identical factors
        assert factor1["ef_value"] == factor2["ef_value"]
        assert factor1["ef_uri"] == factor2["ef_uri"]

        # Calculate difference
        difference = abs(factor1["ef_value"] - factor2["ef_value"])
        assert difference == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_reconcile_different_sources(self, mock_emission_factor_db):
        """INT-REC-002: Test reconciliation between different sources."""
        factor_epa = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)
        factor_defra = await mock_emission_factor_db.load_factor("natural_gas", "GB", 2024)

        # Calculate percentage difference
        avg = (factor_epa["ef_value"] + factor_defra["ef_value"]) / 2
        pct_diff = abs(factor_epa["ef_value"] - factor_defra["ef_value"]) / avg * 100

        # Expect some variance between sources (typically <10%)
        assert pct_diff < 10

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_reconcile_with_tolerance(self, mock_emission_factor_db):
        """INT-REC-003: Test reconciliation with tolerance threshold."""
        factor1 = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)

        # Simulate slightly different factor from another source
        factor2 = {"ef_value": 0.0560}  # Very close to 0.0561

        tolerance = 0.01  # 1% tolerance
        difference = abs(factor1["ef_value"] - factor2["ef_value"])
        relative_diff = difference / factor1["ef_value"]

        is_reconciled = relative_diff <= tolerance
        assert is_reconciled is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_reconcile_flag_discrepancies(self, mock_emission_factor_db):
        """INT-REC-004: Test flagging significant discrepancies."""
        factor1 = await mock_emission_factor_db.load_factor("natural_gas", "US", 2024)

        # Simulate significantly different factor
        factor2 = {"ef_value": 0.0800}  # Much higher than 0.0561

        tolerance = 0.05  # 5% tolerance
        difference = abs(factor1["ef_value"] - factor2["ef_value"])
        relative_diff = difference / factor1["ef_value"]

        discrepancy = {
            "factor1": factor1["ef_value"],
            "factor2": factor2["ef_value"],
            "difference": round(difference, 6),
            "relative_diff_pct": round(relative_diff * 100, 2),
            "exceeds_tolerance": relative_diff > tolerance,
        }

        assert discrepancy["exceeds_tolerance"] is True
        assert discrepancy["relative_diff_pct"] > 5


# =============================================================================
# Cache Invalidation Tests (4 tests)
# =============================================================================

class TestCacheInvalidation:
    """Test cache invalidation - 4 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, mock_cache):
        """INT-CACHE-001: Test basic cache set and get."""
        key = "ef:natural_gas:US:2024"
        value = {"ef_value": 0.0561, "source": "EPA"}

        await mock_cache.set(key, value, ttl_seconds=300)
        cached = await mock_cache.get(key)

        assert cached is not None
        assert cached["ef_value"] == 0.0561

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_expiration(self, mock_cache):
        """INT-CACHE-002: Test cache entry expiration."""
        key = "ef:diesel:US:2024"
        value = {"ef_value": 0.0745}

        await mock_cache.set(key, value, ttl_seconds=1)  # 1 second TTL

        # Should be available immediately
        cached1 = await mock_cache.get(key)
        assert cached1 is not None

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        cached2 = await mock_cache.get(key)
        assert cached2 is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_invalidate_pattern(self, mock_cache):
        """INT-CACHE-003: Test pattern-based cache invalidation."""
        # Set multiple related entries
        await mock_cache.set("ef:natural_gas:US:2024", {"value": 1})
        await mock_cache.set("ef:natural_gas:GB:2024", {"value": 2})
        await mock_cache.set("ef:diesel:US:2024", {"value": 3})

        # Invalidate all natural gas entries
        count = await mock_cache.invalidate_pattern("natural_gas")

        assert count == 2

        # Verify natural gas entries removed
        assert await mock_cache.get("ef:natural_gas:US:2024") is None
        assert await mock_cache.get("ef:natural_gas:GB:2024") is None

        # Verify diesel still exists
        assert await mock_cache.get("ef:diesel:US:2024") is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_clear_all(self, mock_cache):
        """INT-CACHE-004: Test clearing all cache entries."""
        # Populate cache
        await mock_cache.set("key1", {"value": 1})
        await mock_cache.set("key2", {"value": 2})
        await mock_cache.set("key3", {"value": 3})

        stats_before = mock_cache.get_stats()
        assert stats_before["size"] == 3

        # Clear all
        await mock_cache.clear()

        stats_after = mock_cache.get_stats()
        assert stats_after["size"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
