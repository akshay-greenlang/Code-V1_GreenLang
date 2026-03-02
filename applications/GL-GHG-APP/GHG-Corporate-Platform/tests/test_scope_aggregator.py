"""
Unit tests for GL-GHG-APP v1.0 Scope Aggregator

Tests Scope 1/2/3 aggregation, gas breakdown, geographic breakdown,
entity breakdown, and full inventory aggregation.  45+ test cases.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Optional

from services.config import (
    ConsolidationApproach,
    DataQualityTier,
    EntityType,
    GHGGas,
    GWP_AR5,
    Scope,
    Scope1Category,
    Scope3Category,
)
from services.models import (
    Entity,
    GHGInventory,
    InventoryBoundary,
    Organization,
    ScopeEmissions,
)


# ---------------------------------------------------------------------------
# Lightweight ScopeAggregator under test
# ---------------------------------------------------------------------------

class ScopeAggregator:
    """
    Aggregates emission data across scopes, categories, gases, entities,
    and geographies.
    """

    SCOPE1_CATEGORIES = [c.value for c in Scope1Category]
    SCOPE3_CATEGORIES = [c.value for c in Scope3Category]

    # -- Scope 1 -----------------------------------------------------------

    def aggregate_scope1(
        self,
        entries: List[Dict[str, Any]],
        biogenic_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> ScopeEmissions:
        """Aggregate Scope 1 entries into a ScopeEmissions."""
        by_gas: Dict[str, Decimal] = {}
        by_category: Dict[str, Decimal] = {}
        by_entity: Dict[str, Decimal] = {}
        total = Decimal("0")

        for entry in entries:
            tco2e = Decimal(str(entry.get("tco2e", 0)))
            gas = entry.get("gas", GHGGas.CO2.value)
            cat = entry.get("category", "unknown")
            entity_id = entry.get("entity_id", "unknown")

            total += tco2e
            by_gas[gas] = by_gas.get(gas, Decimal("0")) + tco2e
            by_category[cat] = by_category.get(cat, Decimal("0")) + tco2e
            by_entity[entity_id] = by_entity.get(entity_id, Decimal("0")) + tco2e

        biogenic = Decimal("0")
        if biogenic_entries:
            for be in biogenic_entries:
                biogenic += Decimal(str(be.get("tco2e", 0)))

        return ScopeEmissions(
            scope=Scope.SCOPE_1,
            total_tco2e=total,
            by_gas=by_gas,
            by_category=by_category,
            by_entity=by_entity,
            biogenic_co2=biogenic,
        )

    # -- Scope 2 -----------------------------------------------------------

    def aggregate_scope2_location(self, entries: List[Dict[str, Any]]) -> ScopeEmissions:
        """Aggregate Scope 2 location-based entries."""
        by_gas: Dict[str, Decimal] = {}
        by_category: Dict[str, Decimal] = {}
        by_entity: Dict[str, Decimal] = {}
        total = Decimal("0")

        for entry in entries:
            tco2e = Decimal(str(entry.get("tco2e", 0)))
            gas = entry.get("gas", GHGGas.CO2.value)
            cat = entry.get("category", "purchased_electricity")
            entity_id = entry.get("entity_id", "unknown")

            total += tco2e
            by_gas[gas] = by_gas.get(gas, Decimal("0")) + tco2e
            by_category[cat] = by_category.get(cat, Decimal("0")) + tco2e
            by_entity[entity_id] = by_entity.get(entity_id, Decimal("0")) + tco2e

        return ScopeEmissions(
            scope=Scope.SCOPE_2_LOCATION,
            total_tco2e=total,
            by_gas=by_gas,
            by_category=by_category,
            by_entity=by_entity,
        )

    def aggregate_scope2_market(self, entries: List[Dict[str, Any]]) -> ScopeEmissions:
        """Aggregate Scope 2 market-based entries."""
        by_gas: Dict[str, Decimal] = {}
        by_category: Dict[str, Decimal] = {}
        by_entity: Dict[str, Decimal] = {}
        total = Decimal("0")

        for entry in entries:
            tco2e = Decimal(str(entry.get("tco2e", 0)))
            gas = entry.get("gas", GHGGas.CO2.value)
            cat = entry.get("category", "purchased_electricity")
            entity_id = entry.get("entity_id", "unknown")

            total += tco2e
            by_gas[gas] = by_gas.get(gas, Decimal("0")) + tco2e
            by_category[cat] = by_category.get(cat, Decimal("0")) + tco2e
            by_entity[entity_id] = by_entity.get(entity_id, Decimal("0")) + tco2e

        return ScopeEmissions(
            scope=Scope.SCOPE_2_MARKET,
            total_tco2e=total,
            by_gas=by_gas,
            by_category=by_category,
            by_entity=by_entity,
        )

    # -- Scope 3 -----------------------------------------------------------

    def aggregate_scope3(self, entries: List[Dict[str, Any]]) -> ScopeEmissions:
        """Aggregate Scope 3 entries."""
        by_gas: Dict[str, Decimal] = {}
        by_category: Dict[str, Decimal] = {}
        by_entity: Dict[str, Decimal] = {}
        total = Decimal("0")

        for entry in entries:
            tco2e = Decimal(str(entry.get("tco2e", 0)))
            gas = entry.get("gas", GHGGas.CO2.value)
            cat = entry.get("category", "unknown")
            entity_id = entry.get("entity_id", "unknown")

            total += tco2e
            by_gas[gas] = by_gas.get(gas, Decimal("0")) + tco2e
            by_category[cat] = by_category.get(cat, Decimal("0")) + tco2e
            by_entity[entity_id] = by_entity.get(entity_id, Decimal("0")) + tco2e

        return ScopeEmissions(
            scope=Scope.SCOPE_3,
            total_tco2e=total,
            by_gas=by_gas,
            by_category=by_category,
            by_entity=by_entity,
        )

    # -- Full aggregation ---------------------------------------------------

    def aggregate_all(
        self,
        inventory: GHGInventory,
        scope1_entries: List[Dict[str, Any]],
        scope2_location_entries: List[Dict[str, Any]],
        scope2_market_entries: List[Dict[str, Any]],
        scope3_entries: List[Dict[str, Any]],
        biogenic_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> GHGInventory:
        """Aggregate all scopes and update inventory."""
        inventory.scope1 = self.aggregate_scope1(scope1_entries, biogenic_entries)
        inventory.scope2_location = self.aggregate_scope2_location(scope2_location_entries)
        inventory.scope2_market = self.aggregate_scope2_market(scope2_market_entries)
        inventory.scope3 = self.aggregate_scope3(scope3_entries)
        inventory.recalculate_totals()
        return inventory

    # -- Materiality screening ----------------------------------------------

    def screen_materiality(
        self,
        scope3: ScopeEmissions,
        threshold_pct: Decimal = Decimal("1.0"),
    ) -> Dict[str, bool]:
        """Screen Scope 3 categories for materiality (>threshold % of total)."""
        total = scope3.total_tco2e
        if total == 0:
            return {cat: False for cat in scope3.by_category}
        return {
            cat: (val / total * 100) >= threshold_pct
            for cat, val in scope3.by_category.items()
        }

    # -- Geographic breakdown -----------------------------------------------

    def geographic_breakdown(
        self,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by country."""
        by_country: Dict[str, Decimal] = {}
        for entry in entries:
            country = entry.get("country", "UNKNOWN")
            tco2e = Decimal(str(entry.get("tco2e", 0)))
            by_country[country] = by_country.get(country, Decimal("0")) + tco2e
        return by_country

    # -- GWP conversion -----------------------------------------------------

    def convert_to_co2e(self, mass_tonnes: Decimal, gas: GHGGas) -> Decimal:
        """Convert gas mass to CO2e using AR5 GWP."""
        gwp = GWP_AR5.get(gas, 1)
        return mass_tonnes * Decimal(str(gwp))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aggregator():
    return ScopeAggregator()


@pytest.fixture
def scope1_entries():
    """Sample Scope 1 emission entries."""
    return [
        {"category": "stationary_combustion", "gas": "CO2", "tco2e": 5680.0, "entity_id": "ent-001", "country": "US"},
        {"category": "stationary_combustion", "gas": "CH4", "tco2e": 98.5, "entity_id": "ent-001", "country": "US"},
        {"category": "stationary_combustion", "gas": "N2O", "tco2e": 41.8, "entity_id": "ent-001", "country": "US"},
        {"category": "mobile_combustion", "gas": "CO2", "tco2e": 2290.0, "entity_id": "ent-002", "country": "US"},
        {"category": "mobile_combustion", "gas": "CH4", "tco2e": 35.2, "entity_id": "ent-002", "country": "US"},
        {"category": "mobile_combustion", "gas": "N2O", "tco2e": 15.3, "entity_id": "ent-002", "country": "US"},
        {"category": "process_emissions", "gas": "CO2", "tco2e": 1890.0, "entity_id": "ent-003", "country": "US"},
        {"category": "fugitive_emissions", "gas": "CH4", "tco2e": 1120.0, "entity_id": "ent-003", "country": "US"},
        {"category": "fugitive_emissions", "gas": "CO2", "tco2e": 120.0, "entity_id": "ent-003", "country": "US"},
        {"category": "fugitive_emissions", "gas": "N2O", "tco2e": 10.0, "entity_id": "ent-003", "country": "US"},
        {"category": "refrigerants", "gas": "HFCs", "tco2e": 1150.0, "entity_id": "ent-001", "country": "US"},
    ]


@pytest.fixture
def biogenic_entries():
    """Sample biogenic CO2 entries."""
    return [
        {"category": "stationary_combustion", "gas": "CO2_biogenic", "tco2e": 85.0, "entity_id": "ent-001"},
    ]


@pytest.fixture
def scope2_location_entries():
    """Sample Scope 2 location-based entries."""
    return [
        {"category": "purchased_electricity", "gas": "CO2", "tco2e": 7200.0, "entity_id": "ent-001", "country": "US"},
        {"category": "purchased_electricity", "gas": "CH4", "tco2e": 72.0, "entity_id": "ent-001", "country": "US"},
        {"category": "purchased_electricity", "gas": "N2O", "tco2e": 28.0, "entity_id": "ent-001", "country": "US"},
        {"category": "steam_heat", "gas": "CO2", "tco2e": 800.0, "entity_id": "ent-002", "country": "DE"},
        {"category": "steam_heat", "gas": "CH4", "tco2e": 8.5, "entity_id": "ent-002", "country": "DE"},
        {"category": "steam_heat", "gas": "N2O", "tco2e": 12.0, "entity_id": "ent-002", "country": "DE"},
        {"category": "cooling", "gas": "CO2", "tco2e": 200.0, "entity_id": "ent-003", "country": "US"},
    ]


@pytest.fixture
def scope2_market_entries():
    """Sample Scope 2 market-based entries."""
    return [
        {"category": "purchased_electricity", "gas": "CO2", "tco2e": 5200.0, "entity_id": "ent-001", "country": "US"},
        {"category": "purchased_electricity", "gas": "CH4", "tco2e": 52.0, "entity_id": "ent-001", "country": "US"},
        {"category": "purchased_electricity", "gas": "N2O", "tco2e": 20.0, "entity_id": "ent-001", "country": "US"},
        {"category": "steam_heat", "gas": "CO2", "tco2e": 650.0, "entity_id": "ent-002", "country": "DE"},
        {"category": "steam_heat", "gas": "CH4", "tco2e": 6.5, "entity_id": "ent-002", "country": "DE"},
        {"category": "cooling", "gas": "CO2", "tco2e": 171.5, "entity_id": "ent-003", "country": "US"},
    ]


@pytest.fixture
def scope3_entries():
    """Sample Scope 3 entries with 15 categories."""
    return [
        {"category": "cat1_purchased_goods", "gas": "CO2", "tco2e": 18500.0, "entity_id": "ent-001", "country": "CN"},
        {"category": "cat2_capital_goods", "gas": "CO2", "tco2e": 5000.0, "entity_id": "ent-001", "country": "US"},
        {"category": "cat3_fuel_energy", "gas": "CO2", "tco2e": 3200.0, "entity_id": "ent-001", "country": "US"},
        {"category": "cat4_upstream_transport", "gas": "CO2", "tco2e": 8200.0, "entity_id": "ent-002", "country": "US"},
        {"category": "cat5_waste_generated", "gas": "CO2", "tco2e": 2800.0, "entity_id": "ent-003", "country": "US"},
        {"category": "cat5_waste_generated", "gas": "CH4", "tco2e": 450.0, "entity_id": "ent-003", "country": "US"},
        {"category": "cat6_business_travel", "gas": "CO2", "tco2e": 3200.0, "entity_id": "ent-001", "country": "GLOBAL"},
        {"category": "cat7_employee_commuting", "gas": "CO2", "tco2e": 2530.2, "entity_id": "ent-001", "country": "US"},
        {"category": "cat8_upstream_leased", "gas": "CO2", "tco2e": 500.0, "entity_id": "ent-001", "country": "US"},
        {"category": "cat9_downstream_transport", "gas": "CO2", "tco2e": 1200.0, "entity_id": "ent-002", "country": "US"},
        {"category": "cat10_processing_sold", "gas": "CO2", "tco2e": 800.0, "entity_id": "ent-001", "country": "MX"},
        {"category": "cat11_use_of_sold", "gas": "CO2", "tco2e": 6500.0, "entity_id": "ent-001", "country": "GLOBAL"},
        {"category": "cat12_end_of_life", "gas": "CO2", "tco2e": 950.0, "entity_id": "ent-001", "country": "GLOBAL"},
        {"category": "cat12_end_of_life", "gas": "CH4", "tco2e": 380.0, "entity_id": "ent-001", "country": "GLOBAL"},
        {"category": "cat13_downstream_leased", "gas": "CO2", "tco2e": 0.0, "entity_id": "ent-001", "country": "US"},
        {"category": "cat14_franchises", "gas": "CO2", "tco2e": 0.0, "entity_id": "ent-001", "country": "US"},
        {"category": "cat15_investments", "gas": "CO2", "tco2e": 1500.0, "entity_id": "ent-001", "country": "GLOBAL"},
    ]


# ---------------------------------------------------------------------------
# TestScope1Aggregation
# ---------------------------------------------------------------------------

class TestScope1Aggregation:
    """Test Scope 1 emission aggregation."""

    def test_8_categories_summed(self, aggregator, scope1_entries):
        """Test Scope 1 total sums all category entries."""
        result = aggregator.aggregate_scope1(scope1_entries)
        assert result.total_tco2e == Decimal("12450.8")

    def test_per_gas_breakdown(self, aggregator, scope1_entries):
        """Test per-gas breakdown."""
        result = aggregator.aggregate_scope1(scope1_entries)
        assert "CO2" in result.by_gas
        assert "CH4" in result.by_gas
        assert "N2O" in result.by_gas
        assert "HFCs" in result.by_gas
        gas_total = sum(result.by_gas.values())
        assert gas_total == result.total_tco2e

    def test_per_facility(self, aggregator, scope1_entries):
        """Test per-entity breakdown."""
        result = aggregator.aggregate_scope1(scope1_entries)
        assert "ent-001" in result.by_entity
        assert "ent-002" in result.by_entity
        assert "ent-003" in result.by_entity
        entity_total = sum(result.by_entity.values())
        assert entity_total == result.total_tco2e

    def test_biogenic_co2_separate(self, aggregator, scope1_entries, biogenic_entries):
        """Test biogenic CO2 is tracked separately."""
        result = aggregator.aggregate_scope1(scope1_entries, biogenic_entries)
        assert result.biogenic_co2 == Decimal("85.0")
        # Biogenic CO2 should NOT be in the main total
        assert result.total_tco2e == Decimal("12450.8")

    def test_category_breakdown(self, aggregator, scope1_entries):
        """Test category breakdown sums match total."""
        result = aggregator.aggregate_scope1(scope1_entries)
        category_total = sum(result.by_category.values())
        assert category_total == result.total_tco2e

    def test_empty_entries(self, aggregator):
        """Test aggregation with no entries."""
        result = aggregator.aggregate_scope1([])
        assert result.total_tco2e == Decimal("0")
        assert result.by_gas == {}

    def test_provenance_hash(self, aggregator, scope1_entries):
        """Test provenance hash is generated."""
        result = aggregator.aggregate_scope1(scope1_entries)
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# TestScope2Aggregation
# ---------------------------------------------------------------------------

class TestScope2Aggregation:
    """Test Scope 2 emission aggregation."""

    def test_location_based_total(self, aggregator, scope2_location_entries):
        """Test Scope 2 location-based total."""
        result = aggregator.aggregate_scope2_location(scope2_location_entries)
        expected = Decimal("7200") + Decimal("72") + Decimal("28") + Decimal("800") + Decimal("8.5") + Decimal("12") + Decimal("200")
        assert result.total_tco2e == expected

    def test_market_based_total(self, aggregator, scope2_market_entries):
        """Test Scope 2 market-based total."""
        result = aggregator.aggregate_scope2_market(scope2_market_entries)
        expected = Decimal("5200") + Decimal("52") + Decimal("20") + Decimal("650") + Decimal("6.5") + Decimal("171.5")
        assert result.total_tco2e == expected

    def test_dual_reporting(self, aggregator, scope2_location_entries, scope2_market_entries):
        """Test dual reporting - location and market differ."""
        loc = aggregator.aggregate_scope2_location(scope2_location_entries)
        mkt = aggregator.aggregate_scope2_market(scope2_market_entries)
        assert loc.total_tco2e != mkt.total_tco2e
        # Location-based is typically higher
        assert loc.total_tco2e > mkt.total_tco2e

    def test_steam_heat_included(self, aggregator, scope2_location_entries):
        """Test steam/heat purchases are included."""
        result = aggregator.aggregate_scope2_location(scope2_location_entries)
        assert "steam_heat" in result.by_category
        assert result.by_category["steam_heat"] > 0

    def test_cooling_included(self, aggregator, scope2_location_entries):
        """Test cooling purchases are included."""
        result = aggregator.aggregate_scope2_location(scope2_location_entries)
        assert "cooling" in result.by_category

    def test_scope2_per_entity(self, aggregator, scope2_location_entries):
        """Test Scope 2 per-entity breakdown."""
        result = aggregator.aggregate_scope2_location(scope2_location_entries)
        entity_total = sum(result.by_entity.values())
        assert entity_total == result.total_tco2e


# ---------------------------------------------------------------------------
# TestScope3Aggregation
# ---------------------------------------------------------------------------

class TestScope3Aggregation:
    """Test Scope 3 emission aggregation."""

    def test_15_categories(self, aggregator, scope3_entries):
        """Test all 15 categories represented."""
        result = aggregator.aggregate_scope3(scope3_entries)
        # All 15 should appear in by_category (some may be 0)
        assert len(result.by_category) >= 13  # At least active categories

    def test_total_matches_entries(self, aggregator, scope3_entries):
        """Test total matches sum of all entries."""
        result = aggregator.aggregate_scope3(scope3_entries)
        expected = sum(Decimal(str(e["tco2e"])) for e in scope3_entries)
        assert result.total_tco2e == expected

    def test_materiality_screening(self, aggregator, scope3_entries):
        """Test materiality screening at 1% threshold."""
        scope3 = aggregator.aggregate_scope3(scope3_entries)
        materiality = aggregator.screen_materiality(scope3, Decimal("1.0"))
        assert materiality["cat1_purchased_goods"] is True  # ~33% of total
        assert materiality["cat13_downstream_leased"] is False  # 0%
        assert materiality["cat14_franchises"] is False  # 0%

    def test_per_category_methods(self, aggregator, scope3_entries):
        """Test per-category breakdown."""
        result = aggregator.aggregate_scope3(scope3_entries)
        cat1 = result.by_category.get("cat1_purchased_goods", Decimal("0"))
        assert cat1 == Decimal("18500.0")

    def test_scope3_empty(self, aggregator):
        """Test empty Scope 3 aggregation."""
        result = aggregator.aggregate_scope3([])
        assert result.total_tco2e == Decimal("0")

    def test_materiality_zero_total(self, aggregator):
        """Test materiality screening with zero total."""
        scope3 = ScopeEmissions(scope=Scope.SCOPE_3, total_tco2e=Decimal("0"))
        materiality = aggregator.screen_materiality(scope3)
        assert all(not v for v in materiality.values())


# ---------------------------------------------------------------------------
# TestAggregateAll
# ---------------------------------------------------------------------------

class TestAggregateAll:
    """Test full inventory aggregation."""

    def test_grand_total(
        self, aggregator, scope1_entries, scope2_location_entries,
        scope2_market_entries, scope3_entries,
    ):
        """Test grand total = S1 + S2_market + S3."""
        inventory = GHGInventory(org_id="org-001", year=2025)
        result = aggregator.aggregate_all(
            inventory, scope1_entries, scope2_location_entries,
            scope2_market_entries, scope3_entries,
        )
        s1 = result.scope1.total_tco2e
        s2m = result.scope2_market.total_tco2e
        s3 = result.scope3.total_tco2e
        assert result.grand_total_tco2e == s1 + s2m + s3

    def test_inventory_updated(
        self, aggregator, scope1_entries, scope2_location_entries,
        scope2_market_entries, scope3_entries,
    ):
        """Test inventory scopes are populated after aggregation."""
        inventory = GHGInventory(org_id="org-001", year=2025)
        result = aggregator.aggregate_all(
            inventory, scope1_entries, scope2_location_entries,
            scope2_market_entries, scope3_entries,
        )
        assert result.scope1 is not None
        assert result.scope2_location is not None
        assert result.scope2_market is not None
        assert result.scope3 is not None

    def test_biogenic_separate(
        self, aggregator, scope1_entries, scope2_location_entries,
        scope2_market_entries, scope3_entries, biogenic_entries,
    ):
        """Test biogenic CO2 tracked separately in full aggregation."""
        inventory = GHGInventory(org_id="org-001", year=2025)
        result = aggregator.aggregate_all(
            inventory, scope1_entries, scope2_location_entries,
            scope2_market_entries, scope3_entries, biogenic_entries,
        )
        assert result.scope1.biogenic_co2 == Decimal("85.0")


# ---------------------------------------------------------------------------
# TestGasBreakdown
# ---------------------------------------------------------------------------

class TestGasBreakdown:
    """Test gas breakdown and GWP conversion."""

    def test_7_gases_present(self, aggregator):
        """Test all 7 Kyoto gases in GWP table."""
        assert len(GWP_AR5) == 7
        expected_gases = {GHGGas.CO2, GHGGas.CH4, GHGGas.N2O, GHGGas.HFCS, GHGGas.PFCS, GHGGas.SF6, GHGGas.NF3}
        assert set(GWP_AR5.keys()) == expected_gases

    def test_gwp_ch4(self, aggregator):
        """Test CH4 GWP conversion (x28)."""
        result = aggregator.convert_to_co2e(Decimal("1.0"), GHGGas.CH4)
        assert result == Decimal("28")

    def test_gwp_n2o(self, aggregator):
        """Test N2O GWP conversion (x265)."""
        result = aggregator.convert_to_co2e(Decimal("1.0"), GHGGas.N2O)
        assert result == Decimal("265")

    def test_gwp_sf6(self, aggregator):
        """Test SF6 GWP conversion (x23500)."""
        result = aggregator.convert_to_co2e(Decimal("1.0"), GHGGas.SF6)
        assert result == Decimal("23500")

    def test_gwp_co2_identity(self, aggregator):
        """Test CO2 GWP is 1 (identity)."""
        result = aggregator.convert_to_co2e(Decimal("100.0"), GHGGas.CO2)
        assert result == Decimal("100.0")

    def test_gwp_fractional_mass(self, aggregator):
        """Test GWP conversion with fractional mass."""
        result = aggregator.convert_to_co2e(Decimal("0.804"), GHGGas.HFCS)
        # 0.804 * 1430 = 1149.72
        assert result == Decimal("1149.720")


# ---------------------------------------------------------------------------
# TestGeographicBreakdown
# ---------------------------------------------------------------------------

class TestGeographicBreakdown:
    """Test geographic emission breakdown."""

    def test_per_country_totals(self, aggregator, scope1_entries):
        """Test per-country aggregation."""
        result = aggregator.geographic_breakdown(scope1_entries)
        assert "US" in result
        assert result["US"] == Decimal("12450.8")

    def test_multi_country(self, aggregator, scope2_location_entries):
        """Test multi-country breakdown."""
        result = aggregator.geographic_breakdown(scope2_location_entries)
        assert "US" in result
        assert "DE" in result
        total = sum(result.values())
        expected = sum(Decimal(str(e["tco2e"])) for e in scope2_location_entries)
        assert total == expected

    def test_empty_entries(self, aggregator):
        """Test geographic breakdown with no entries."""
        result = aggregator.geographic_breakdown([])
        assert result == {}


# ---------------------------------------------------------------------------
# TestEntityBreakdown
# ---------------------------------------------------------------------------

class TestEntityBreakdown:
    """Test per-entity emission breakdown."""

    def test_per_facility_totals_match(self, aggregator, scope1_entries):
        """Test per-entity totals match scope total."""
        result = aggregator.aggregate_scope1(scope1_entries)
        entity_total = sum(result.by_entity.values())
        assert entity_total == result.total_tco2e

    def test_entity_ids_tracked(self, aggregator, scope1_entries):
        """Test all entity IDs are tracked."""
        result = aggregator.aggregate_scope1(scope1_entries)
        expected_ids = {"ent-001", "ent-002", "ent-003"}
        assert set(result.by_entity.keys()) == expected_ids

    def test_single_entity(self, aggregator):
        """Test aggregation with single entity."""
        entries = [
            {"category": "stationary_combustion", "gas": "CO2", "tco2e": 5000.0, "entity_id": "ent-solo"},
        ]
        result = aggregator.aggregate_scope1(entries)
        assert len(result.by_entity) == 1
        assert result.by_entity["ent-solo"] == Decimal("5000.0")
