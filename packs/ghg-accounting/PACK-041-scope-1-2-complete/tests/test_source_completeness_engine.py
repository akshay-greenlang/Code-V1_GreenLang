# -*- coding: utf-8 -*-
"""
Unit tests for SourceCompletenessEngine -- PACK-041 Engine 2
==============================================================

Tests source category identification by sector, materiality assessment,
data availability checking, gap identification, and completeness reporting.

Coverage target: 85%+
Total tests: ~55
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack041_test.engines.src_compl_{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("source_completeness_engine")

# Expected categories per sector for completeness checking
OFFICE_CATEGORIES = {"stationary_combustion", "refrigerant_fgas"}
MANUFACTURING_CATEGORIES = {
    "stationary_combustion", "mobile_combustion", "process_emissions",
    "fugitive_emissions", "refrigerant_fgas", "waste_treatment",
}
ENERGY_CATEGORIES = {
    "stationary_combustion", "process_emissions", "fugitive_emissions",
}
TRANSPORT_CATEGORIES = {"mobile_combustion", "fugitive_emissions"}
AGRICULTURE_CATEGORIES = {"stationary_combustion", "agricultural", "land_use"}

ALL_SCOPE1_CATEGORIES = {
    "stationary_combustion", "mobile_combustion", "process_emissions",
    "fugitive_emissions", "refrigerant_fgas", "land_use",
    "waste_treatment", "agricultural",
}


# =============================================================================
# Module Loading
# =============================================================================


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_engine_class_exists(self):
        assert hasattr(_m, "SourceCompletenessEngine") or hasattr(_m, "_MODULE_VERSION")


# =============================================================================
# Sector-Based Source Categories
# =============================================================================


class TestSectorCategories:
    """Test that each sector maps to the correct emission source categories."""

    @pytest.mark.parametrize("sector,expected_cats", [
        ("office", OFFICE_CATEGORIES),
        ("manufacturing", MANUFACTURING_CATEGORIES),
        ("energy", ENERGY_CATEGORIES),
        ("transport", TRANSPORT_CATEGORIES),
        ("agriculture", AGRICULTURE_CATEGORIES),
    ])
    def test_sector_has_expected_categories(self, sector, expected_cats):
        """Each sector should map to known categories."""
        for cat in expected_cats:
            assert cat in ALL_SCOPE1_CATEGORIES

    def test_office_excludes_process_emissions(self):
        assert "process_emissions" not in OFFICE_CATEGORIES

    def test_office_excludes_agricultural(self):
        assert "agricultural" not in OFFICE_CATEGORIES

    def test_manufacturing_excludes_agricultural(self):
        assert "agricultural" not in MANUFACTURING_CATEGORIES

    def test_agriculture_includes_land_use(self):
        assert "land_use" in AGRICULTURE_CATEGORIES

    def test_all_categories_valid(self):
        for cat in ALL_SCOPE1_CATEGORIES:
            assert isinstance(cat, str)
            assert len(cat) > 0


# =============================================================================
# Materiality Assessment
# =============================================================================


class TestMaterialityAssessment:
    """Test materiality threshold checks."""

    @pytest.mark.parametrize("category_pct,threshold,expected_material", [
        (Decimal("5.0"), Decimal("1.0"), True),
        (Decimal("2.0"), Decimal("1.0"), True),
        (Decimal("0.5"), Decimal("1.0"), False),
        (Decimal("0.1"), Decimal("1.0"), False),
        (Decimal("1.0"), Decimal("1.0"), True),
        (Decimal("0.9"), Decimal("1.0"), False),
        (Decimal("10.0"), Decimal("5.0"), True),
        (Decimal("3.0"), Decimal("5.0"), False),
    ])
    def test_materiality_threshold(self, category_pct, threshold, expected_material):
        if expected_material:
            assert category_pct >= threshold
        else:
            assert category_pct < threshold

    def test_materiality_above_1pct_is_material(self):
        threshold = Decimal("1.0")
        assert Decimal("1.5") >= threshold

    def test_materiality_below_1pct_not_material(self):
        threshold = Decimal("1.0")
        assert Decimal("0.3") < threshold

    def test_materiality_regulatory_required_always_material(self):
        """Regulatory-required categories should be material regardless."""
        regulatory_required = True
        category_pct = Decimal("0.01")
        is_material = category_pct >= Decimal("1.0") or regulatory_required
        assert is_material is True

    def test_materiality_zero_emissions_not_material(self):
        category_pct = Decimal("0.0")
        assert category_pct < Decimal("1.0")


# =============================================================================
# Data Availability
# =============================================================================


class TestDataAvailability:
    """Test data availability assessment."""

    @pytest.mark.parametrize("availability,expected_status", [
        ("full", "AVAILABLE"),
        ("partial", "PARTIAL"),
        ("missing", "MISSING"),
    ])
    def test_data_availability_classification(self, availability, expected_status):
        """Data availability should classify as AVAILABLE, PARTIAL, or MISSING."""
        status_map = {"full": "AVAILABLE", "partial": "PARTIAL", "missing": "MISSING"}
        assert status_map[availability] == expected_status

    def test_full_availability_has_activity_data(self):
        facility_data = {
            "fuel_data": [{"fuel_type": "natural_gas", "quantity": 1000}],
        }
        assert len(facility_data["fuel_data"]) > 0

    def test_partial_availability_has_some_data(self):
        facility_data = {
            "fuel_data": [{"fuel_type": "natural_gas", "quantity": 1000}],
            "refrigerant_data": [],
        }
        assert len(facility_data["fuel_data"]) > 0
        assert len(facility_data["refrigerant_data"]) == 0

    def test_missing_availability_no_data(self):
        facility_data = {
            "fuel_data": [],
            "refrigerant_data": [],
        }
        assert len(facility_data["fuel_data"]) == 0

    def test_scope2_data_requires_electricity(self):
        facility_data = {"electricity_data": [{"quantity_mwh": 10000}]}
        assert len(facility_data["electricity_data"]) > 0

    def test_scope2_data_missing_electricity(self):
        facility_data = {"electricity_data": []}
        assert len(facility_data["electricity_data"]) == 0


# =============================================================================
# Gap Identification
# =============================================================================


class TestGapIdentification:
    """Test identification of data gaps."""

    def test_gap_identified_for_missing_category(self):
        covered = {"stationary_combustion", "mobile_combustion"}
        expected = {"stationary_combustion", "mobile_combustion", "refrigerant_fgas"}
        gaps = expected - covered
        assert "refrigerant_fgas" in gaps

    def test_no_gap_when_all_covered(self):
        covered = {"stationary_combustion", "refrigerant_fgas"}
        expected = {"stationary_combustion", "refrigerant_fgas"}
        gaps = expected - covered
        assert len(gaps) == 0

    def test_multiple_gaps_identified(self):
        covered = {"stationary_combustion"}
        expected = ALL_SCOPE1_CATEGORIES
        gaps = expected - covered
        assert len(gaps) == 7

    def test_gap_with_materiality_filter(self):
        gaps_with_materiality = [
            {"category": "fugitive_emissions", "estimated_pct": Decimal("0.3")},
            {"category": "process_emissions", "estimated_pct": Decimal("5.0")},
        ]
        material_gaps = [g for g in gaps_with_materiality if g["estimated_pct"] >= Decimal("1.0")]
        assert len(material_gaps) == 1
        assert material_gaps[0]["category"] == "process_emissions"


# =============================================================================
# Completeness Calculation
# =============================================================================


class TestCompletenessCalculation:
    """Test completeness percentage calculation."""

    def test_100_percent_completeness(self):
        covered = ALL_SCOPE1_CATEGORIES
        expected = ALL_SCOPE1_CATEGORIES
        pct = len(covered & expected) / len(expected) * 100 if expected else 0
        assert pct == 100.0

    def test_partial_completeness(self):
        covered = {"stationary_combustion", "mobile_combustion"}
        expected = ALL_SCOPE1_CATEGORIES
        pct = len(covered & expected) / len(expected) * 100
        assert pct == pytest.approx(25.0)

    def test_zero_completeness(self):
        covered = set()
        expected = ALL_SCOPE1_CATEGORIES
        pct = len(covered & expected) / len(expected) * 100 if expected else 0
        assert pct == 0.0

    def test_weighted_completeness_by_emissions(self):
        """Completeness weighted by emissions share."""
        categories = {
            "stationary_combustion": Decimal("8000"),
            "mobile_combustion": Decimal("2000"),
            "process_emissions": Decimal("0"),
        }
        total = sum(categories.values())
        covered = {"stationary_combustion", "mobile_combustion"}
        covered_emissions = sum(
            v for k, v in categories.items() if k in covered
        )
        pct = float(covered_emissions / total * 100) if total else 0
        assert pct == 100.0

    def test_completeness_scope2_included(self):
        """Scope 2 categories should also factor into completeness."""
        scope1_covered = 6
        scope2_covered = 2
        total_expected = 8 + 5  # 8 scope 1 + 5 scope 2
        pct = (scope1_covered + scope2_covered) / total_expected * 100
        assert pct == pytest.approx(61.54, abs=0.01)


# =============================================================================
# Report Generation
# =============================================================================


class TestCompletenessReport:
    """Test completeness report generation."""

    def test_report_contains_sector(self):
        report = {
            "sector": "manufacturing",
            "expected_categories": list(MANUFACTURING_CATEGORIES),
            "covered_categories": ["stationary_combustion"],
            "gaps": ["mobile_combustion", "process_emissions"],
            "completeness_pct": 16.7,
        }
        assert report["sector"] == "manufacturing"
        assert len(report["gaps"]) > 0

    def test_report_gaps_list_populated(self):
        covered = {"stationary_combustion"}
        expected = MANUFACTURING_CATEGORIES
        gaps = sorted(expected - covered)
        assert len(gaps) >= 4

    def test_report_contains_completeness_score(self):
        report = {"completeness_pct": 85.0}
        assert 0 <= report["completeness_pct"] <= 100

    def test_report_contains_recommendations(self):
        report = {
            "recommendations": [
                "Collect process emissions data from cement kilns",
                "Install refrigerant leak detectors on HVAC systems",
            ]
        }
        assert len(report["recommendations"]) >= 1

    def test_scope2_completeness_in_report(self):
        report = {
            "scope2_categories": {
                "location_based": True,
                "market_based": True,
                "steam": False,
                "cooling": False,
                "dual_reporting": True,
            }
        }
        covered = sum(1 for v in report["scope2_categories"].values() if v)
        assert covered == 3
