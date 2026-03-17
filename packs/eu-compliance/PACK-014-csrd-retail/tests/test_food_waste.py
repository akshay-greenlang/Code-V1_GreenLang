# -*- coding: utf-8 -*-
"""
Unit tests for FoodWasteEngine (PACK-014, Engine 5)
====================================================

Tests all methods of FoodWasteEngine with 85%+ coverage.
Validates business logic, error handling, and edge cases.

Test count: ~43 tests
"""

import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Dynamic import via importlib
# ---------------------------------------------------------------------------

_ENGINE_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "engines",
    "food_waste_engine.py",
)
_ENGINE_PATH = os.path.normpath(_ENGINE_PATH)

_spec = importlib.util.spec_from_file_location("food_waste_engine", _ENGINE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FoodWasteEngine = _mod.FoodWasteEngine
FoodWasteRecord = _mod.FoodWasteRecord
FoodWasteBaseline = _mod.FoodWasteBaseline
FoodWasteResult = _mod.FoodWasteResult
FoodWasteCategory = _mod.FoodWasteCategory
WasteDestination = _mod.WasteDestination
WasteHierarchyLevel = _mod.WasteHierarchyLevel
MeasurementMethod = _mod.MeasurementMethod
FOOD_WASTE_EMISSION_FACTORS = _mod.FOOD_WASTE_EMISSION_FACTORS
WASTE_HIERARCHY_WEIGHTS = _mod.WASTE_HIERARCHY_WEIGHTS
REDISTRIBUTION_CREDIT = _mod.REDISTRIBUTION_CREDIT
EU_FOOD_WASTE_REDUCTION_TARGET = _mod.EU_FOOD_WASTE_REDUCTION_TARGET
AVG_COST_PER_KG = _mod.AVG_COST_PER_KG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a FoodWasteEngine instance."""
    return FoodWasteEngine()


@pytest.fixture
def sample_records():
    """Create a list of sample food waste records."""
    return [
        FoodWasteRecord(
            store_id="STORE-001",
            category=FoodWasteCategory.BAKERY,
            quantity_kg=150.0,
            destination=WasteDestination.REDISTRIBUTION,
            reporting_period="2025-Q1",
        ),
        FoodWasteRecord(
            store_id="STORE-001",
            category=FoodWasteCategory.MEAT_POULTRY,
            quantity_kg=80.0,
            destination=WasteDestination.COMPOSTING,
            reporting_period="2025-Q1",
        ),
        FoodWasteRecord(
            store_id="STORE-002",
            category=FoodWasteCategory.PRODUCE,
            quantity_kg=200.0,
            destination=WasteDestination.LANDFILL,
            reporting_period="2025-Q1",
        ),
        FoodWasteRecord(
            store_id="STORE-002",
            category=FoodWasteCategory.DAIRY,
            quantity_kg=50.0,
            destination=WasteDestination.ANIMAL_FEED,
            reporting_period="2025-Q1",
        ),
    ]


@pytest.fixture
def baseline():
    """Create a sample baseline."""
    return FoodWasteBaseline(
        baseline_year=2021,
        total_waste_kg=50000.0,
    )


# ===========================================================================
# TestInitialization
# ===========================================================================


class TestInitialization:
    """Test engine initialisation."""

    def test_default_instantiation(self):
        """Engine can be created with no arguments."""
        engine = FoodWasteEngine()
        assert engine is not None

    def test_engine_version(self):
        """Engine exposes a version string."""
        engine = FoodWasteEngine()
        assert engine.engine_version == "1.0.0"

    def test_config_dict(self):
        """Engine can accept attributes set after init."""
        engine = FoodWasteEngine()
        engine.engine_version = "2.0.0"
        assert engine.engine_version == "2.0.0"

    def test_none_records_raises(self, engine):
        """Passing empty records list raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            engine.calculate([])


# ===========================================================================
# TestFoodWasteCategories
# ===========================================================================


class TestFoodWasteCategories:
    """Test FoodWasteCategory enum."""

    def test_all_12_defined(self):
        """There must be exactly 12 food waste categories."""
        assert len(FoodWasteCategory) == 12

    def test_enum_values(self):
        """All expected category values present."""
        values = {c.value for c in FoodWasteCategory}
        expected = {
            "bakery", "produce", "dairy", "meat_poultry", "seafood",
            "prepared_food", "packaged_goods", "beverages", "frozen",
            "deli", "confectionery", "other",
        }
        assert values == expected

    def test_bakery_value(self):
        """BAKERY enum has correct string value."""
        assert FoodWasteCategory.BAKERY.value == "bakery"

    def test_meat_poultry_value(self):
        """MEAT_POULTRY enum has correct string value."""
        assert FoodWasteCategory.MEAT_POULTRY.value == "meat_poultry"


# ===========================================================================
# TestWasteDestinations
# ===========================================================================


class TestWasteDestinations:
    """Test WasteDestination enum."""

    def test_all_8_defined(self):
        """There must be exactly 8 waste destinations."""
        assert len(WasteDestination) == 8

    def test_redistribution_value(self):
        """REDISTRIBUTION has correct value."""
        assert WasteDestination.REDISTRIBUTION.value == "redistribution"

    def test_composting_value(self):
        """COMPOSTING has correct value."""
        assert WasteDestination.COMPOSTING.value == "composting"

    def test_landfill_value(self):
        """LANDFILL has correct value."""
        assert WasteDestination.LANDFILL.value == "landfill"


# ===========================================================================
# TestWasteCalculation
# ===========================================================================


class TestWasteCalculation:
    """Test core waste calculation methods."""

    def test_total_waste_kg(self, engine, sample_records):
        """Total waste is the sum of all record quantities."""
        result = engine.calculate(sample_records)
        assert result.total_waste_kg == pytest.approx(480.0, rel=1e-2)

    def test_waste_by_category(self, engine, sample_records):
        """Waste is correctly aggregated by category."""
        result = engine.calculate(sample_records)
        cat_map = {c.category: c.quantity_kg for c in result.waste_by_category}
        assert cat_map["produce"] == pytest.approx(200.0, rel=1e-2)

    def test_waste_by_destination(self, engine, sample_records):
        """Waste is correctly aggregated by destination."""
        result = engine.calculate(sample_records)
        dest_map = {d.destination: d.quantity_kg for d in result.waste_by_destination}
        assert "redistribution" in dest_map
        assert dest_map["redistribution"] == pytest.approx(150.0, rel=1e-2)

    def test_waste_intensity(self, engine):
        """calculate_waste_intensity returns correct per-store metric."""
        result = engine.calculate_waste_intensity(
            total_waste_kg=10000.0,
            store_count=10,
        )
        assert result["waste_per_store_kg"] == pytest.approx(1000.0, rel=1e-2)

    def test_financial_value(self, engine, sample_records):
        """Financial value uses AVG_COST_PER_KG when no explicit value."""
        result = engine.calculate(sample_records)
        # bakery: 150 * 3.50 = 525
        assert result.financial_value_by_category["bakery"] == pytest.approx(525.0, rel=1e-2)

    def test_single_record_emissions(self, engine):
        """calculate_single_record_emissions gives correct result."""
        rec = FoodWasteRecord(
            store_id="S1",
            category=FoodWasteCategory.BAKERY,
            quantity_kg=100.0,
            destination=WasteDestination.LANDFILL,
            reporting_period="2025-Q1",
        )
        res = engine.calculate_single_record_emissions(rec)
        # bakery EF = 0.89 => 100 * 0.89 = 89
        assert res["emissions_kg_co2e"] == pytest.approx(89.0, rel=1e-2)
        assert res["hierarchy_weight"] == pytest.approx(0.0, rel=1e-6)


# ===========================================================================
# TestWasteHierarchy
# ===========================================================================


class TestWasteHierarchy:
    """Test waste hierarchy scoring."""

    def test_hierarchy_score_all_redistribution(self, engine):
        """All waste to redistribution => score near 0.9."""
        records = [
            FoodWasteRecord(
                store_id="S1", category=FoodWasteCategory.BAKERY,
                quantity_kg=100.0, destination=WasteDestination.REDISTRIBUTION,
                reporting_period="2025-Q1",
            ),
        ]
        result = engine.calculate(records)
        assert result.waste_hierarchy_score == pytest.approx(0.9, rel=1e-2)

    def test_hierarchy_score_all_landfill(self, engine):
        """All waste to landfill => score = 0.0."""
        records = [
            FoodWasteRecord(
                store_id="S1", category=FoodWasteCategory.BAKERY,
                quantity_kg=100.0, destination=WasteDestination.LANDFILL,
                reporting_period="2025-Q1",
            ),
        ]
        result = engine.calculate(records)
        assert result.waste_hierarchy_score == pytest.approx(0.0, abs=1e-6)

    def test_mixed_hierarchy(self, engine, sample_records):
        """Mixed destinations produce intermediate score."""
        result = engine.calculate(sample_records)
        assert 0.0 < result.waste_hierarchy_score < 1.0

    def test_hierarchy_levels_defined(self):
        """WasteHierarchyLevel enum has exactly 6 members."""
        assert len(WasteHierarchyLevel) == 6

    def test_waste_hierarchy_weights(self):
        """WASTE_HIERARCHY_WEIGHTS has expected values."""
        assert WASTE_HIERARCHY_WEIGHTS["prevention"] == 1.0
        assert WASTE_HIERARCHY_WEIGHTS["disposal"] == 0.0
        assert WASTE_HIERARCHY_WEIGHTS["redistribution"] == 0.9


# ===========================================================================
# TestReductionTarget
# ===========================================================================


class TestReductionTarget:
    """Test food waste reduction tracking against EU 2030 target."""

    def test_baseline_establishment(self, engine, sample_records, baseline):
        """Reduction tracking is populated when baseline is provided."""
        result = engine.calculate(sample_records, baseline=baseline)
        assert result.reduction_tracking is not None

    def test_reduction_percentage(self, engine, sample_records, baseline):
        """Reduction percentage is calculated correctly."""
        result = engine.calculate(sample_records, baseline=baseline)
        # baseline 50000, current 480 => reduction_pct ~ 99.04
        assert result.reduction_tracking.reduction_pct > 90.0

    def test_on_track_for_2030(self, engine, baseline):
        """Should be on track when reduction exceeds expected."""
        records = [
            FoodWasteRecord(
                store_id="S1", category=FoodWasteCategory.BAKERY,
                quantity_kg=30000.0, destination=WasteDestination.COMPOSTING,
                reporting_period="2025-Q1",
            ),
        ]
        result = engine.calculate(records, baseline=baseline, reporting_year=2025)
        # reduction = (50000 - 30000) / 50000 * 100 = 40%
        # expected at 2025 (4 years from 2021, 9 total to 2030): 30 * 4/9 ~ 13.3
        assert result.on_track_for_2030_target is True

    def test_off_track(self, engine, baseline):
        """Off track when reduction is below expected linear path."""
        records = [
            FoodWasteRecord(
                store_id="S1", category=FoodWasteCategory.BAKERY,
                quantity_kg=49000.0, destination=WasteDestination.LANDFILL,
                reporting_period="2025-Q1",
            ),
        ]
        result = engine.calculate(records, baseline=baseline, reporting_year=2025)
        # reduction = (50000-49000)/50000 * 100 = 2%
        assert result.on_track_for_2030_target is False

    def test_eu_target_30pct(self):
        """EU target constant is 30%."""
        assert EU_FOOD_WASTE_REDUCTION_TARGET == 30.0


# ===========================================================================
# TestRedistribution
# ===========================================================================


class TestRedistribution:
    """Test redistribution programme metrics."""

    def test_redistribution_rate(self, engine, sample_records):
        """Redistribution rate = redistributed / total."""
        result = engine.calculate(sample_records)
        # 150 / 480 * 100 = 31.25
        assert result.redistribution_rate_pct == pytest.approx(31.25, rel=1e-2)

    def test_redistribution_credit(self):
        """REDISTRIBUTION_CREDIT constant is 0.85."""
        assert REDISTRIBUTION_CREDIT == pytest.approx(0.85, rel=1e-6)

    def test_donation_tracking(self, engine, sample_records):
        """Redistribution kg is tracked correctly."""
        result = engine.calculate(sample_records)
        assert result.redistribution_kg == pytest.approx(150.0, rel=1e-2)

    def test_zero_redistribution(self, engine):
        """Zero redistribution when no records go to redistribution."""
        records = [
            FoodWasteRecord(
                store_id="S1", category=FoodWasteCategory.BAKERY,
                quantity_kg=100.0, destination=WasteDestination.LANDFILL,
                reporting_period="2025-Q1",
            ),
        ]
        result = engine.calculate(records)
        assert result.redistribution_kg == pytest.approx(0.0, abs=1e-6)
        assert result.redistribution_rate_pct == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# TestEmissions
# ===========================================================================


class TestEmissions:
    """Test GHG emission calculations from food waste."""

    def test_emission_factors_defined(self):
        """All 12 categories have emission factors."""
        assert len(FOOD_WASTE_EMISSION_FACTORS) == 12

    def test_total_emissions_tco2e(self, engine, sample_records):
        """Total emissions in tCO2e is positive for non-zero waste."""
        result = engine.calculate(sample_records)
        assert result.emissions_from_waste_tco2e > 0.0

    def test_emissions_by_category(self, engine, sample_records):
        """Emissions are broken down by category."""
        result = engine.calculate(sample_records)
        # bakery: 150 * 0.89 = 133.5
        assert result.emissions_by_category["bakery"] == pytest.approx(133.5, rel=1e-2)
        # meat_poultry: 80 * 13.31 = 1064.8
        assert result.emissions_by_category["meat_poultry"] == pytest.approx(1064.8, rel=1e-2)


# ===========================================================================
# TestProvenance
# ===========================================================================


class TestProvenance:
    """Test SHA-256 provenance hashing."""

    def test_hash_length(self, engine, sample_records):
        """Provenance hash is 64 hex characters (SHA-256)."""
        result = engine.calculate(sample_records)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self, engine):
        """Provenance hash is a valid hex string derived from result data."""
        records = [
            FoodWasteRecord(
                store_id="S1", category=FoodWasteCategory.BAKERY,
                quantity_kg=100.0, destination=WasteDestination.LANDFILL,
                reporting_period="2025-Q1",
            ),
        ]
        result = engine.calculate(records)
        # Hash is valid hex (only hex characters)
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)
        # Recomputing: engine computes hash when provenance_hash is still ""
        from hashlib import sha256
        import json
        saved_hash = result.provenance_hash
        result.provenance_hash = ""
        serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True, default=str)
        expected = sha256(serialized.encode("utf-8")).hexdigest()
        result.provenance_hash = saved_hash
        assert saved_hash == expected

    def test_different_input(self, engine):
        """Different inputs produce different hashes."""
        r1 = engine.calculate([
            FoodWasteRecord(
                store_id="S1", category=FoodWasteCategory.BAKERY,
                quantity_kg=100.0, destination=WasteDestination.LANDFILL,
                reporting_period="2025-Q1",
            ),
        ])
        r2 = engine.calculate([
            FoodWasteRecord(
                store_id="S1", category=FoodWasteCategory.BAKERY,
                quantity_kg=200.0, destination=WasteDestination.LANDFILL,
                reporting_period="2025-Q1",
            ),
        ])
        assert r1.provenance_hash != r2.provenance_hash


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_records_raises(self, engine):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate([])

    def test_zero_waste(self, engine):
        """A record with zero quantity is valid."""
        records = [
            FoodWasteRecord(
                store_id="S1", category=FoodWasteCategory.BAKERY,
                quantity_kg=0.0, destination=WasteDestination.LANDFILL,
                reporting_period="2025-Q1",
            ),
        ]
        result = engine.calculate(records)
        assert result.total_waste_kg == pytest.approx(0.0, abs=1e-6)

    def test_large_dataset(self, engine):
        """Engine handles a large number of records without error."""
        records = [
            FoodWasteRecord(
                store_id=f"S-{i}",
                category=FoodWasteCategory.BAKERY,
                quantity_kg=10.0,
                destination=WasteDestination.COMPOSTING,
                reporting_period="2025-Q1",
            )
            for i in range(500)
        ]
        result = engine.calculate(records)
        assert result.total_waste_kg == pytest.approx(5000.0, rel=1e-2)
        assert result.store_count == 500

    def test_single_store(self, engine):
        """Engine works correctly with a single store."""
        records = [
            FoodWasteRecord(
                store_id="ONLY", category=FoodWasteCategory.PRODUCE,
                quantity_kg=500.0, destination=WasteDestination.COMPOSTING,
                reporting_period="2025-Q1",
            ),
        ]
        result = engine.calculate(records)
        assert result.store_count == 1
        assert result.waste_per_store_kg == pytest.approx(500.0, rel=1e-2)

    def test_result_fields(self, engine, sample_records):
        """Result object contains all expected fields."""
        result = engine.calculate(sample_records)
        assert hasattr(result, "total_waste_kg")
        assert hasattr(result, "waste_hierarchy_score")
        assert hasattr(result, "emissions_from_waste_tco2e")
        assert hasattr(result, "financial_value_wasted_eur")
        assert hasattr(result, "redistribution_rate_pct")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms > 0.0
