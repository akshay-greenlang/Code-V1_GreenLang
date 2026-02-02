# -*- coding: utf-8 -*-
"""
Unit Tests for BPS Thresholds Database

Comprehensive test suite with 10 test cases covering:
- BPSThresholdDatabase initialization and loading
- Building type lookups
- Climate zone variations
- NYC Local Law 97 compliance

Target: 85%+ coverage for BPS thresholds database
Run with: pytest tests/unit/test_bps_thresholds.py -v --cov=core/greenlang/data/bps_thresholds

Author: GL-TestEngineer
Version: 1.0.0

BPS (Building Performance Standards) thresholds are energy and emissions limits
for buildings, primarily from NYC Local Law 97 and ASHRAE standards.
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def bps_database():
    """Create BPSThresholdDatabase instance."""
    from core.greenlang.data.bps_thresholds import BPSThresholdDatabase
    return BPSThresholdDatabase()


@pytest.fixture
def expected_thresholds():
    """Expected BPS threshold values from NYC LL97 and ENERGY STAR."""
    return {
        ("office", "4A"): 80.0,
        ("office", "5A"): 85.0,
        ("residential", "4A"): 100.0,
        ("retail", "4A"): 120.0,
        ("warehouse", "default"): 50.0,
        ("hospital", "default"): 250.0,
    }


# =============================================================================
# BPSThresholdDatabase Tests (10 tests)
# =============================================================================

class TestBPSThresholdDatabase:
    """Test suite for BPSThresholdDatabase - 10 test cases."""

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_database_initialization(self, bps_database):
        """UT-BPS-DB-001: Test database initializes correctly."""
        assert bps_database is not None
        assert bps_database.thresholds is not None
        assert isinstance(bps_database.thresholds, dict)
        assert len(bps_database.thresholds) > 0

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_get_bps_database_singleton(self):
        """UT-BPS-DB-002: Test get_bps_database returns singleton instance."""
        from core.greenlang.data.bps_thresholds import get_bps_database

        db1 = get_bps_database()
        db2 = get_bps_database()

        # Should return same instance
        assert db1 is db2

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_office_threshold(self, bps_database):
        """UT-BPS-DB-003: Test lookup returns correct office threshold."""
        result = bps_database.lookup("office", "4A")

        assert result is not None
        assert result.threshold_kwh_per_sqm == 80.0
        assert result.building_type == "office"
        assert result.climate_zone == "4A"

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_residential_threshold(self, bps_database):
        """UT-BPS-DB-004: Test lookup returns correct residential threshold."""
        result = bps_database.lookup("residential", "4A")

        assert result is not None
        assert result.threshold_kwh_per_sqm == 100.0
        assert result.building_type == "residential"

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_without_climate_zone_uses_default(self, bps_database):
        """UT-BPS-DB-005: Test lookup without climate zone uses default."""
        result = bps_database.lookup("warehouse")

        assert result is not None
        assert result.threshold_kwh_per_sqm == 50.0
        assert result.climate_zone == "default"

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_returns_none_for_invalid(self, bps_database):
        """UT-BPS-DB-006: Test lookup returns None for invalid building type."""
        result = bps_database.lookup("invalid_building_xyz")

        assert result is None

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_list_building_types_returns_all_types(self, bps_database):
        """UT-BPS-DB-007: Test list_building_types returns all building types."""
        building_types = bps_database.list_building_types()

        assert isinstance(building_types, list)
        assert len(building_types) >= 5  # At least 5 building types

        # Check key building types are present
        assert "office" in building_types
        assert "residential" in building_types

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_threshold_includes_ghg_limit(self, bps_database):
        """UT-BPS-DB-008: Test threshold includes GHG emissions limit."""
        result = bps_database.lookup("office", "4A")

        assert result is not None
        assert result.ghg_threshold_kgco2e_per_sqm is not None
        assert result.ghg_threshold_kgco2e_per_sqm == 5.4

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_get_statistics(self, bps_database):
        """UT-BPS-DB-009: Test get_statistics returns database stats."""
        stats = bps_database.get_statistics()

        assert "total_thresholds" in stats
        assert "building_types" in stats
        assert "avg_eui" in stats
        assert "max_eui" in stats
        assert "min_eui" in stats

        assert stats["total_thresholds"] >= 10
        assert stats["avg_eui"] > 0

    @pytest.mark.unit
    @pytest.mark.core_library
    @pytest.mark.compliance
    @pytest.mark.parametrize("building_type,climate_zone,expected_eui", [
        ("office", "4A", 80.0),
        ("office", "5A", 85.0),
        ("residential", "4A", 100.0),
        ("retail", "4A", 120.0),
        ("hotel", "4A", 140.0),
        ("hospital", "default", 250.0),
    ])
    def test_threshold_values_match_standards(
        self,
        bps_database,
        building_type,
        climate_zone,
        expected_eui
    ):
        """UT-BPS-DB-010: Test threshold values match NYC LL97/ENERGY STAR."""
        result = bps_database.lookup(building_type, climate_zone)

        assert result is not None, f"Threshold for {building_type}/{climate_zone} not found"
        assert result.threshold_kwh_per_sqm == expected_eui, \
            f"Threshold for {building_type}/{climate_zone} should be {expected_eui}"


# =============================================================================
# BPS Data Quality Tests
# =============================================================================

class TestBPSDataQuality:
    """Test suite for BPS data quality and provenance."""

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_threshold_has_source_citation(self, bps_database):
        """Test threshold includes source citation."""
        result = bps_database.lookup("office", "4A")

        assert result is not None
        assert result.source is not None
        assert len(result.source) > 0

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_threshold_has_jurisdiction(self, bps_database):
        """Test threshold includes jurisdiction."""
        result = bps_database.lookup("office", "4A")

        assert result is not None
        assert result.jurisdiction is not None
        # Should be NYC for Local Law 97 thresholds
        assert "NYC" in result.jurisdiction or "US" in result.jurisdiction

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_threshold_has_effective_date(self, bps_database):
        """Test threshold includes effective date."""
        result = bps_database.lookup("office", "4A")

        assert result is not None
        assert result.effective_date is not None
        assert "2024" in result.effective_date  # NYC LL97 first compliance period


# =============================================================================
# Climate Zone Variation Tests
# =============================================================================

class TestClimateZoneVariation:
    """Test suite for climate zone threshold variations."""

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_colder_climate_has_higher_threshold(self, bps_database):
        """Test colder climate zones have higher energy thresholds."""
        zone_4a = bps_database.lookup("office", "4A")  # Mixed humid (NYC)
        zone_5a = bps_database.lookup("office", "5A")  # Cool humid (Chicago)

        assert zone_4a is not None
        assert zone_5a is not None

        # Colder climate should allow higher energy use
        assert zone_5a.threshold_kwh_per_sqm >= zone_4a.threshold_kwh_per_sqm


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
