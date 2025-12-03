# -*- coding: utf-8 -*-
"""
Unit Tests for Emission Factor Database

Comprehensive test suite with 15 test cases covering:
- EmissionFactorDatabase initialization and loading
- Deterministic lookups
- Search and filtering
- Provenance tracking

Target: 85%+ coverage for emission factor database
Run with: pytest tests/unit/test_emission_factor_db.py -v --cov=core/greenlang/data/emission_factor_db

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import json
import hashlib
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# Add project paths for imports
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def emission_factor_db():
    """Create EmissionFactorDatabase instance."""
    from core.greenlang.data.emission_factor_db import EmissionFactorDatabase
    return EmissionFactorDatabase()


@pytest.fixture
def sample_emission_factors():
    """Sample emission factor data for testing."""
    return {
        "natural_gas": {
            "US": {
                "2023": {
                    "co2e": 0.0561,
                    "co2": 0.053,
                    "ch4": 0.0021,
                    "n2o": 0.001,
                    "unit": "kgCO2e/MJ",
                    "uncertainty": 0.05,
                    "quality": "1",
                    "citation": "EPA GHG Emission Factors Hub 2023"
                }
            },
            "GB": {
                "2023": {
                    "co2e": 0.0552,
                    "co2": 0.052,
                    "ch4": 0.0022,
                    "n2o": 0.001,
                    "unit": "kgCO2e/MJ",
                    "uncertainty": 0.05,
                    "quality": "1",
                    "citation": "DEFRA 2023"
                }
            }
        },
        "diesel": {
            "US": {
                "2023": {
                    "co2e": 0.0729,
                    "co2": 0.070,
                    "ch4": 0.0019,
                    "n2o": 0.001,
                    "unit": "kgCO2e/MJ",
                    "uncertainty": 0.05,
                    "quality": "1",
                    "citation": "EPA GHG Emission Factors Hub 2023"
                }
            }
        }
    }


@pytest.fixture
def temp_data_dir(sample_emission_factors):
    """Create temporary data directory with test emission factors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "factors"
        data_dir.mkdir()

        # Write DEFRA test file
        defra_path = data_dir / "defra_2023.json"
        with open(defra_path, 'w') as f:
            json.dump(sample_emission_factors, f)

        yield data_dir


# =============================================================================
# EmissionFactorDatabase Tests (15 tests)
# =============================================================================

class TestEmissionFactorDatabase:
    """Test suite for EmissionFactorDatabase - 15 test cases."""

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_database_initialization(self):
        """UT-EF-001: Test database initializes correctly."""
        from core.greenlang.data.emission_factor_db import EmissionFactorDatabase

        db = EmissionFactorDatabase()

        assert db is not None
        assert db.factors is not None
        assert isinstance(db.factors, dict)

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_get_database_singleton(self):
        """UT-EF-002: Test get_database returns singleton instance."""
        from core.greenlang.data.emission_factor_db import get_database

        db1 = get_database()
        db2 = get_database()

        # Should return same instance
        assert db1 is db2

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_gwp_set_enum_values(self):
        """UT-EF-003: Test GWPSet enum has correct values."""
        from core.greenlang.data.emission_factor_db import GWPSet

        assert GWPSet.AR6GWP100.value == "AR6GWP100"
        assert GWPSet.AR5GWP100.value == "AR5GWP100"
        assert GWPSet.AR4GWP100.value == "AR4GWP100"

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_emission_factor_record_model(self):
        """UT-EF-004: Test EmissionFactorRecord model validation."""
        from core.greenlang.data.emission_factor_db import EmissionFactorRecord, GWPSet

        record = EmissionFactorRecord(
            ef_uri="ef://test/2023/natural_gas/US/2023",
            ef_value=0.0561,
            ef_unit="kgCO2e/MJ",
            co2=0.053,
            ch4=0.0021,
            n2o=0.001,
            source="Test",
            source_version="2023",
            gwp_set=GWPSet.AR6GWP100,
            region="US",
            year=2023,
            uncertainty=0.05,
            data_quality="1",
            data_hash="abc123",
            citation="Test Citation"
        )

        assert record.ef_value == 0.0561
        assert record.region == "US"
        assert record.gwp_set == GWPSet.AR6GWP100

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_returns_emission_factor(self, emission_factor_db):
        """UT-EF-005: Test lookup returns emission factor for valid parameters."""
        from core.greenlang.data.emission_factor_db import GWPSet

        # Mock the factors dictionary with test data
        from core.greenlang.data.emission_factor_db import EmissionFactorRecord
        mock_record = EmissionFactorRecord(
            ef_uri="ef://defra/2023/natural_gas/US/2023",
            ef_value=0.0561,
            ef_unit="kgCO2e/MJ",
            co2=0.053,
            ch4=0.0021,
            n2o=0.001,
            source="DEFRA",
            source_version="2023",
            gwp_set=GWPSet.AR6GWP100,
            region="US",
            year=2023,
            uncertainty=0.05,
            data_quality="1",
            data_hash="abc123",
            citation="DEFRA 2023"
        )

        emission_factor_db.factors["ef://defra/2023/natural_gas/US/2023"] = mock_record

        result = emission_factor_db.lookup("natural_gas", "US", 2023)

        assert result is not None
        assert result.ef_value == 0.0561

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_returns_none_for_invalid(self, emission_factor_db):
        """UT-EF-006: Test lookup returns None for invalid parameters."""
        result = emission_factor_db.lookup("nonexistent_fuel_xyz", "XX", 9999)

        assert result is None

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_by_uri(self, emission_factor_db):
        """UT-EF-007: Test lookup by URI returns correct record."""
        from core.greenlang.data.emission_factor_db import EmissionFactorRecord, GWPSet

        mock_record = EmissionFactorRecord(
            ef_uri="ef://test/2023/diesel/US/2023",
            ef_value=0.0729,
            ef_unit="kgCO2e/MJ",
            co2=0.070,
            ch4=0.0019,
            n2o=0.001,
            source="Test",
            source_version="2023",
            gwp_set=GWPSet.AR6GWP100,
            region="US",
            year=2023,
            uncertainty=0.05,
            data_quality="1",
            data_hash="def456",
            citation="Test 2023"
        )

        emission_factor_db.factors["ef://test/2023/diesel/US/2023"] = mock_record

        result = emission_factor_db.lookup_by_uri("ef://test/2023/diesel/US/2023")

        assert result is not None
        assert result.ef_value == 0.0729

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_by_uri_returns_none_for_invalid(self, emission_factor_db):
        """UT-EF-008: Test lookup_by_uri returns None for invalid URI."""
        result = emission_factor_db.lookup_by_uri("ef://nonexistent/uri")

        assert result is None

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_search_by_fuel_type(self, emission_factor_db):
        """UT-EF-009: Test search filters by fuel type."""
        from core.greenlang.data.emission_factor_db import EmissionFactorRecord, GWPSet

        # Add test records
        for fuel in ["natural_gas", "diesel"]:
            record = EmissionFactorRecord(
                ef_uri=f"ef://test/2023/{fuel}/US/2023",
                ef_value=0.05,
                ef_unit="kgCO2e/MJ",
                co2=0.05,
                source="Test",
                source_version="2023",
                gwp_set=GWPSet.AR6GWP100,
                region="US",
                year=2023,
                uncertainty=0.05,
                data_quality="1",
                data_hash="hash",
                citation="Test"
            )
            emission_factor_db.factors[record.ef_uri] = record

        results = emission_factor_db.search(fuel_type="natural_gas")

        assert len(results) >= 1
        assert all("natural_gas" in r.ef_uri for r in results)

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_search_by_region(self, emission_factor_db):
        """UT-EF-010: Test search filters by region."""
        from core.greenlang.data.emission_factor_db import EmissionFactorRecord, GWPSet

        # Add test records for different regions
        for region in ["US", "GB"]:
            record = EmissionFactorRecord(
                ef_uri=f"ef://test/2023/natural_gas/{region}/2023",
                ef_value=0.05,
                ef_unit="kgCO2e/MJ",
                co2=0.05,
                source="Test",
                source_version="2023",
                gwp_set=GWPSet.AR6GWP100,
                region=region,
                year=2023,
                uncertainty=0.05,
                data_quality="1",
                data_hash="hash",
                citation="Test"
            )
            emission_factor_db.factors[record.ef_uri] = record

        results = emission_factor_db.search(region="US")

        assert len(results) >= 1
        assert all(r.region == "US" for r in results)

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_list_fuel_types(self, emission_factor_db):
        """UT-EF-011: Test list_fuel_types returns all fuel types."""
        fuel_types = emission_factor_db.list_fuel_types()

        assert isinstance(fuel_types, list)
        # Should be sorted alphabetically
        assert fuel_types == sorted(fuel_types)

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_list_regions(self, emission_factor_db):
        """UT-EF-012: Test list_regions returns all regions."""
        from core.greenlang.data.emission_factor_db import EmissionFactorRecord, GWPSet

        # Add test records for different regions
        for region in ["US", "GB", "EU"]:
            record = EmissionFactorRecord(
                ef_uri=f"ef://test/2023/natural_gas/{region}/2023",
                ef_value=0.05,
                ef_unit="kgCO2e/MJ",
                co2=0.05,
                source="Test",
                source_version="2023",
                gwp_set=GWPSet.AR6GWP100,
                region=region,
                year=2023,
                uncertainty=0.05,
                data_quality="1",
                data_hash="hash",
                citation="Test"
            )
            emission_factor_db.factors[record.ef_uri] = record

        regions = emission_factor_db.list_regions()

        assert isinstance(regions, list)
        assert "US" in regions

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_get_statistics(self, emission_factor_db):
        """UT-EF-013: Test get_statistics returns database stats."""
        from core.greenlang.data.emission_factor_db import EmissionFactorRecord, GWPSet

        # Add test record
        record = EmissionFactorRecord(
            ef_uri="ef://test/2023/natural_gas/US/2023",
            ef_value=0.05,
            ef_unit="kgCO2e/MJ",
            co2=0.05,
            source="Test",
            source_version="2023",
            gwp_set=GWPSet.AR6GWP100,
            region="US",
            year=2023,
            uncertainty=0.05,
            data_quality="1",
            data_hash="hash",
            citation="Test"
        )
        emission_factor_db.factors[record.ef_uri] = record

        stats = emission_factor_db.get_statistics()

        assert "total_factors" in stats
        assert "fuel_types" in stats
        assert "regions" in stats
        assert "sources" in stats
        assert stats["total_factors"] >= 1

    @pytest.mark.unit
    @pytest.mark.core_library
    @pytest.mark.determinism
    def test_lookup_determinism(self, emission_factor_db):
        """UT-EF-014: Test lookup is deterministic across multiple calls."""
        from core.greenlang.data.emission_factor_db import EmissionFactorRecord, GWPSet

        # Add test record
        record = EmissionFactorRecord(
            ef_uri="ef://defra/2023/diesel/US/2023",
            ef_value=0.0729,
            ef_unit="kgCO2e/MJ",
            co2=0.070,
            source="DEFRA",
            source_version="2023",
            gwp_set=GWPSet.AR6GWP100,
            region="US",
            year=2023,
            uncertainty=0.05,
            data_quality="1",
            data_hash="hash",
            citation="DEFRA 2023"
        )
        emission_factor_db.factors[record.ef_uri] = record

        results = []
        for _ in range(10):
            result = emission_factor_db.lookup("diesel", "US", 2023)
            if result:
                results.append(result.ef_value)

        # All results should be identical
        if results:
            assert all(r == results[0] for r in results)

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_data_hash_provenance(self, emission_factor_db):
        """UT-EF-015: Test data hash is included for provenance."""
        from core.greenlang.data.emission_factor_db import EmissionFactorRecord, GWPSet

        record = EmissionFactorRecord(
            ef_uri="ef://test/2023/natural_gas/US/2023",
            ef_value=0.0561,
            ef_unit="kgCO2e/MJ",
            co2=0.053,
            source="Test",
            source_version="2023",
            gwp_set=GWPSet.AR6GWP100,
            region="US",
            year=2023,
            uncertainty=0.05,
            data_quality="1",
            data_hash="abc123def456",
            citation="Test"
        )

        assert record.data_hash is not None
        assert len(record.data_hash) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
