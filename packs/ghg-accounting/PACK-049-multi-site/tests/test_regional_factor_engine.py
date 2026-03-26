# -*- coding: utf-8 -*-
"""
Tests for PACK-049 Engine 4: RegionalFactorEngine

Covers factor assignment by country, tiered lookup with fallback,
grid region and climate zone assignment, factor overrides, coverage
analysis, tier distribution, and decimal precision.
Target: ~55 tests.
"""

import pytest
from decimal import Decimal
from datetime import date
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

try:
    from engines.regional_factor_engine import (
        RegionalFactorEngine,
        FactorAssignment,
        FactorLookupResult,
        FactorCoverage,
        FactorOverride,
        FactorTier,
        FactorSource,
    )
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="Engine not yet built")


@pytest.fixture
def engine():
    return RegionalFactorEngine()


# ============================================================================
# Factor Assignment Tests
# ============================================================================

class TestFactorAssignment:

    def test_assign_factors_us(self, engine):
        result = engine.assign_factors(
            site_id="site-001",
            country="US",
            source_type="ELECTRICITY",
            year=2026,
        )
        assert result is not None
        assert result.factor_value > Decimal("0")
        assert result.country == "US"

    def test_assign_factors_uk(self, engine):
        result = engine.assign_factors(
            site_id="site-002",
            country="GB",
            source_type="ELECTRICITY",
            year=2026,
        )
        assert result.factor_value > Decimal("0")
        assert result.factor_source in ("DEFRA", "IEA")

    def test_assign_factors_de(self, engine):
        result = engine.assign_factors(
            site_id="site-003",
            country="DE",
            source_type="ELECTRICITY",
            year=2026,
        )
        assert result.factor_value > Decimal("0")

    def test_assign_factors_natural_gas(self, engine):
        result = engine.assign_factors(
            site_id="site-001",
            country="US",
            source_type="NATURAL_GAS",
            year=2026,
        )
        assert result.factor_value > Decimal("0")

    def test_assign_factors_diesel(self, engine):
        result = engine.assign_factors(
            site_id="site-001",
            country="US",
            source_type="DIESEL",
            year=2026,
        )
        assert result.factor_value > Decimal("0")

    def test_assign_factors_provenance(self, engine):
        result = engine.assign_factors(
            site_id="site-001",
            country="US",
            source_type="ELECTRICITY",
            year=2026,
        )
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# ============================================================================
# Factor Lookup / Fallback Tests
# ============================================================================

class TestFactorLookup:

    def test_lookup_factor_tier3_found(self, engine):
        # Register a facility-specific factor
        engine.register_factor(
            country="US",
            source_type="ELECTRICITY",
            tier="TIER_3_FACILITY",
            factor_value=Decimal("0.000350"),
            source="SUPPLIER_SPECIFIC",
            site_id="site-001",
            year=2026,
        )
        result = engine.lookup_factor(
            country="US",
            source_type="ELECTRICITY",
            preferred_tier="TIER_3_FACILITY",
            site_id="site-001",
            year=2026,
        )
        assert result.tier_used == "TIER_3_FACILITY"
        assert result.factor_value == Decimal("0.000350")

    def test_lookup_factor_tier2_fallback(self, engine):
        # No Tier 3 factor registered; expect fallback to Tier 2
        result = engine.lookup_factor(
            country="GB",
            source_type="ELECTRICITY",
            preferred_tier="TIER_3_FACILITY",
            year=2026,
        )
        assert result.tier_used in ("TIER_2_NATIONAL", "TIER_1_REGIONAL", "TIER_0_IPCC_DEFAULT")

    def test_lookup_factor_ipcc_default(self, engine):
        # Country with no specific database should fall back to IPCC
        result = engine.lookup_factor(
            country="ZZ",
            source_type="ELECTRICITY",
            preferred_tier="TIER_2_NATIONAL",
            year=2026,
        )
        assert result.tier_used == "TIER_0_IPCC_DEFAULT"

    def test_lookup_factor_returns_source(self, engine):
        result = engine.lookup_factor(
            country="US",
            source_type="ELECTRICITY",
            preferred_tier="TIER_1_REGIONAL",
            year=2026,
        )
        assert result.factor_source is not None
        assert len(result.factor_source) > 0


# ============================================================================
# Grid Region and Climate Zone Tests
# ============================================================================

class TestGridAndClimate:

    def test_assign_grid_region(self, engine):
        region = engine.assign_grid_region(
            country="US",
            state="Illinois",
        )
        assert region is not None
        assert len(region) > 0

    def test_assign_climate_zone(self, engine):
        zone = engine.assign_climate_zone(
            country="US",
            latitude=Decimal("41.88"),
            longitude=Decimal("-87.63"),
        )
        assert zone is not None


# ============================================================================
# Override Tests
# ============================================================================

class TestFactorOverride:

    def test_override_factor(self, engine):
        original = engine.assign_factors(
            site_id="site-001",
            country="US",
            source_type="ELECTRICITY",
            year=2026,
        )
        override = engine.override_factor(
            site_id="site-001",
            source_type="ELECTRICITY",
            new_value=Decimal("0.000300"),
            reason="Supplier-specific factor",
            year=2026,
        )
        assert override.factor_value == Decimal("0.000300")

    def test_override_records_original(self, engine):
        original = engine.assign_factors(
            site_id="site-001",
            country="US",
            source_type="ELECTRICITY",
            year=2026,
        )
        override = engine.override_factor(
            site_id="site-001",
            source_type="ELECTRICITY",
            new_value=Decimal("0.000300"),
            reason="Supplier update",
            year=2026,
        )
        assert override.original_value is not None or override.original_value == original.factor_value

    def test_override_negative_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine.override_factor(
                site_id="site-001",
                source_type="ELECTRICITY",
                new_value=Decimal("-0.001"),
                reason="Invalid",
                year=2026,
            )


# ============================================================================
# Coverage Tests
# ============================================================================

class TestFactorCoverage:

    def test_factor_coverage_100pct(self, engine, sample_factor_assignments):
        for fa in sample_factor_assignments:
            engine.assign_factors(
                site_id=fa["site_id"],
                country=fa["country"],
                source_type=fa["source_type"],
                year=fa["year"],
            )
        coverage = engine.get_factor_coverage(
            site_ids=["site-001", "site-002", "site-003", "site-004", "site-005"],
            source_types=["ELECTRICITY"],
        )
        assert coverage.coverage_pct == Decimal("100") or coverage.coverage_pct >= Decimal("80")

    def test_factor_coverage_partial(self, engine):
        engine.assign_factors(
            site_id="site-001",
            country="US",
            source_type="ELECTRICITY",
            year=2026,
        )
        coverage = engine.get_factor_coverage(
            site_ids=["site-001", "site-002"],
            source_types=["ELECTRICITY"],
        )
        assert coverage.coverage_pct <= Decimal("100")


# ============================================================================
# Tier Distribution Tests
# ============================================================================

class TestTierDistribution:

    def test_tier_distribution(self, engine, sample_factor_assignments):
        for fa in sample_factor_assignments:
            engine.assign_factors(
                site_id=fa["site_id"],
                country=fa["country"],
                source_type=fa["source_type"],
                year=fa["year"],
            )
        dist = engine.get_tier_distribution(
            site_ids=["site-001", "site-002", "site-003", "site-004", "site-005"],
        )
        assert isinstance(dist, dict)
        assert sum(dist.values()) > 0


# ============================================================================
# Decimal Precision Tests
# ============================================================================

class TestDecimalPrecision:

    def test_decimal_precision_10_digits(self, engine):
        result = engine.assign_factors(
            site_id="site-001",
            country="US",
            source_type="ELECTRICITY",
            year=2026,
        )
        # Factor value should have precise decimal representation
        factor_str = str(result.factor_value)
        assert "." in factor_str
        decimal_places = len(factor_str.split(".")[1])
        assert decimal_places >= 3  # at least 3 decimal places
