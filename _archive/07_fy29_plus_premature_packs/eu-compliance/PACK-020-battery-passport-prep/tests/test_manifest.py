# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Manifest Tests
=========================================================

Tests pack.yaml: YAML structure, metadata, regulation references,
compliance references, battery categories, component declarations,
application dates, and recycled content targets.

Author: GreenLang Platform Team (GL-TestEngineer)
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PACK_ROOT / "pack.yaml"

# ---------------------------------------------------------------------------
# Load YAML
# ---------------------------------------------------------------------------

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]


def _load_manifest() -> Dict[str, Any]:
    if yaml is None:
        pytest.skip("PyYAML not installed")
    if not MANIFEST_PATH.exists():
        pytest.skip(f"pack.yaml not found at {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def manifest() -> Dict[str, Any]:
    return _load_manifest()


# =========================================================================
# Top-Level Structure
# =========================================================================


class TestManifestStructure:
    """Validate top-level YAML structure."""

    def test_manifest_file_exists(self):
        assert MANIFEST_PATH.exists(), "pack.yaml must exist"

    def test_top_level_keys(self, manifest):
        assert "metadata" in manifest
        assert "components" in manifest

    def test_metadata_is_dict(self, manifest):
        assert isinstance(manifest["metadata"], dict)

    def test_components_is_dict(self, manifest):
        assert isinstance(manifest["components"], dict)


# =========================================================================
# Metadata Section
# =========================================================================


class TestManifestMetadata:
    """Validate metadata fields."""

    def test_name(self, manifest):
        meta = manifest["metadata"]
        assert meta["name"] == "PACK-020-battery-passport-prep"

    def test_version(self, manifest):
        meta = manifest["metadata"]
        assert meta["version"] == "1.0.0"

    def test_display_name(self, manifest):
        meta = manifest["metadata"]
        assert meta["display_name"] == "Battery Passport Prep Pack"

    def test_description_exists(self, manifest):
        meta = manifest["metadata"]
        assert isinstance(meta["description"], str)
        assert len(meta["description"]) > 100

    def test_category(self, manifest):
        meta = manifest["metadata"]
        assert meta["category"] == "eu-compliance"

    def test_tier(self, manifest):
        meta = manifest["metadata"]
        assert meta["tier"] == "standalone"

    def test_author(self, manifest):
        meta = manifest["metadata"]
        assert meta["author"] == "GreenLang Platform Team"

    def test_min_platform_version(self, manifest):
        meta = manifest["metadata"]
        assert meta["min_platform_version"] == "2.0.0"

    def test_release_date(self, manifest):
        meta = manifest["metadata"]
        assert "release_date" in meta

    def test_tags_minimum_count(self, manifest):
        meta = manifest["metadata"]
        tags = meta.get("tags", [])
        assert len(tags) >= 30, f"Expected >= 30 tags, got {len(tags)}"

    def test_essential_tags_present(self, manifest):
        meta = manifest["metadata"]
        tags = meta.get("tags", [])
        required_tags = [
            "battery-passport",
            "battery-regulation",
            "eu-2023-1542",
            "digital-product-passport",
            "carbon-footprint",
            "recycled-content",
            "ev-battery",
            "provenance",
        ]
        for tag in required_tags:
            assert tag in tags, f"Tag '{tag}' missing from manifest tags"


# =========================================================================
# Regulation Section
# =========================================================================


class TestManifestRegulation:
    """Validate regulation references."""

    def test_primary_regulation_exists(self, manifest):
        meta = manifest["metadata"]
        reg = meta.get("regulation", {})
        assert "primary" in reg

    def test_primary_regulation_name(self, manifest):
        meta = manifest["metadata"]
        primary = meta["regulation"]["primary"]
        assert primary["name"] == "EU Battery Regulation"

    def test_primary_regulation_reference(self, manifest):
        meta = manifest["metadata"]
        primary = meta["regulation"]["primary"]
        assert primary["reference"] == "Regulation (EU) 2023/1542"

    def test_primary_regulation_dates(self, manifest):
        meta = manifest["metadata"]
        primary = meta["regulation"]["primary"]
        assert "adopted_date" in primary
        assert "published_date" in primary
        assert "entry_into_force" in primary

    def test_secondary_regulations_exist(self, manifest):
        meta = manifest["metadata"]
        secondary = meta["regulation"].get("secondary", [])
        assert isinstance(secondary, list)
        assert len(secondary) >= 8

    def test_secondary_regulation_names(self, manifest):
        meta = manifest["metadata"]
        secondary = meta["regulation"]["secondary"]
        names = [s["name"] for s in secondary]
        assert any("OECD" in n for n in names)
        assert any("Taxonomy" in n for n in names)
        assert any("CSRD" in n or "Corporate Sustainability Reporting" in n for n in names)

    def test_each_secondary_has_reference(self, manifest):
        meta = manifest["metadata"]
        secondary = meta["regulation"]["secondary"]
        for s in secondary:
            assert "reference" in s, f"Secondary '{s.get('name')}' missing 'reference'"


# =========================================================================
# Compliance References
# =========================================================================


class TestManifestComplianceReferences:
    """Validate compliance reference entries."""

    def test_compliance_references_exist(self, manifest):
        meta = manifest["metadata"]
        refs = meta.get("compliance_references", [])
        assert isinstance(refs, list)
        assert len(refs) >= 9

    def test_required_reference_ids(self, manifest):
        meta = manifest["metadata"]
        refs = meta.get("compliance_references", [])
        ids = [r["id"] for r in refs]
        required_ids = [
            "BATT_ART7",
            "BATT_ART8",
            "BATT_ART77_78",
            "BATT_ART10",
            "BATT_ART48",
            "BATT_ART13_14",
            "BATT_ART56_71",
            "BATT_ART17_22",
        ]
        for rid in required_ids:
            assert rid in ids, f"Compliance reference '{rid}' missing"

    def test_each_reference_has_fields(self, manifest):
        meta = manifest["metadata"]
        refs = meta.get("compliance_references", [])
        for ref in refs:
            assert "id" in ref
            assert "name" in ref
            assert "description" in ref
            assert isinstance(ref["description"], str)
            assert len(ref["description"]) > 50

    def test_art7_carbon_footprint(self, manifest):
        meta = manifest["metadata"]
        refs = meta.get("compliance_references", [])
        art7 = next((r for r in refs if r["id"] == "BATT_ART7"), None)
        assert art7 is not None
        assert "Carbon Footprint" in art7["name"]
        assert "CO2e" in art7["description"] or "carbon footprint" in art7["description"].lower()

    def test_art77_battery_passport(self, manifest):
        meta = manifest["metadata"]
        refs = meta.get("compliance_references", [])
        art77 = next((r for r in refs if r["id"] == "BATT_ART77_78"), None)
        assert art77 is not None
        assert "Passport" in art77["name"]
        assert "Annex XIII" in art77["description"]

    def test_oecd_minerals_reference(self, manifest):
        meta = manifest["metadata"]
        refs = meta.get("compliance_references", [])
        oecd = next((r for r in refs if r["id"] == "OECD_MINERALS"), None)
        assert oecd is not None
        assert "OECD" in oecd["name"]

    def test_gba_reference(self, manifest):
        meta = manifest["metadata"]
        refs = meta.get("compliance_references", [])
        gba = next((r for r in refs if r["id"] == "GBA"), None)
        assert gba is not None
        assert "GBA" in gba["name"] or "Global Battery Alliance" in gba["name"]


# =========================================================================
# Battery Categories
# =========================================================================


class TestManifestBatteryCategories:
    """Validate battery category definitions."""

    def test_battery_categories_exist(self, manifest):
        meta = manifest["metadata"]
        cats = meta.get("battery_categories", [])
        assert isinstance(cats, list)
        assert len(cats) == 5

    def test_all_categories_present(self, manifest):
        meta = manifest["metadata"]
        cats = meta.get("battery_categories", [])
        cat_ids = [c["id"] for c in cats]
        for expected in ["EV", "INDUSTRIAL", "LMT", "PORTABLE", "SLI"]:
            assert expected in cat_ids, f"Battery category '{expected}' missing"

    def test_ev_passport_required(self, manifest):
        meta = manifest["metadata"]
        cats = meta.get("battery_categories", [])
        ev = next((c for c in cats if c["id"] == "EV"), None)
        assert ev is not None
        assert ev["passport_required"] is True
        assert ev["carbon_footprint_required"] is True

    def test_portable_passport_not_required(self, manifest):
        meta = manifest["metadata"]
        cats = meta.get("battery_categories", [])
        port = next((c for c in cats if c["id"] == "PORTABLE"), None)
        assert port is not None
        assert port["passport_required"] is False
        assert port["carbon_footprint_required"] is False

    def test_lmt_passport_required(self, manifest):
        meta = manifest["metadata"]
        cats = meta.get("battery_categories", [])
        lmt = next((c for c in cats if c["id"] == "LMT"), None)
        assert lmt is not None
        assert lmt["passport_required"] is True
        assert lmt["carbon_footprint_required"] is False


# =========================================================================
# Application Dates
# =========================================================================


class TestManifestApplicationDates:
    """Validate application date timeline."""

    def test_application_dates_exist(self, manifest):
        meta = manifest["metadata"]
        dates = meta.get("application_dates", [])
        assert isinstance(dates, list)
        assert len(dates) >= 10

    def test_each_date_has_fields(self, manifest):
        meta = manifest["metadata"]
        dates = meta.get("application_dates", [])
        for d in dates:
            assert "requirement" in d
            assert "date" in d
            assert "article" in d
            assert "categories" in d

    def test_passport_date_2027(self, manifest):
        meta = manifest["metadata"]
        dates = meta.get("application_dates", [])
        passport_date = next(
            (d for d in dates if "passport" in d["requirement"].lower() and "mandatory" in d["requirement"].lower()),
            None,
        )
        assert passport_date is not None
        assert "2027" in str(passport_date["date"])


# =========================================================================
# Recycled Content Targets
# =========================================================================


class TestManifestRecycledContentTargets:
    """Validate recycled content target structure."""

    def test_targets_exist(self, manifest):
        meta = manifest["metadata"]
        targets = meta.get("recycled_content_targets", {})
        assert isinstance(targets, dict)
        for mineral in ["cobalt", "lithium", "nickel", "lead"]:
            assert mineral in targets, f"Recycled content target for '{mineral}' missing"

    def test_cobalt_targets(self, manifest):
        meta = manifest["metadata"]
        cobalt = meta["recycled_content_targets"]["cobalt"]
        assert cobalt["phase_2_target_pct"] == 16
        assert cobalt["phase_3_target_pct"] == 26

    def test_lithium_targets(self, manifest):
        meta = manifest["metadata"]
        lithium = meta["recycled_content_targets"]["lithium"]
        assert lithium["phase_2_target_pct"] == 6
        assert lithium["phase_3_target_pct"] == 12

    def test_nickel_targets(self, manifest):
        meta = manifest["metadata"]
        nickel = meta["recycled_content_targets"]["nickel"]
        assert nickel["phase_2_target_pct"] == 6
        assert nickel["phase_3_target_pct"] == 15

    def test_lead_targets(self, manifest):
        meta = manifest["metadata"]
        lead = meta["recycled_content_targets"]["lead"]
        assert lead["phase_2_target_pct"] == 85
        assert lead["phase_3_target_pct"] == 85


# =========================================================================
# Component Declarations
# =========================================================================


class TestManifestComponents:
    """Validate component declarations."""

    def test_total_engines(self, manifest):
        meta = manifest["metadata"]
        assert meta.get("total_engines") == 8

    def test_total_workflows(self, manifest):
        meta = manifest["metadata"]
        assert meta.get("total_workflows") == 8

    def test_total_templates(self, manifest):
        meta = manifest["metadata"]
        assert meta.get("total_templates") == 8

    def test_total_integrations(self, manifest):
        meta = manifest["metadata"]
        assert meta.get("total_integrations") == 10

    def test_total_presets(self, manifest):
        meta = manifest["metadata"]
        assert meta.get("total_presets") == 6

    def test_engines_declared(self, manifest):
        engines = manifest["components"].get("engines", [])
        assert isinstance(engines, list)
        assert len(engines) == 8

    def test_engine_ids(self, manifest):
        engines = manifest["components"]["engines"]
        ids = [e["id"] for e in engines]
        expected = [
            "carbon_footprint",
            "recycled_content",
            "battery_passport",
            "performance_durability",
            "supply_chain_dd",
            "labelling_compliance",
            "end_of_life",
            "conformity_assessment",
        ]
        for eid in expected:
            assert eid in ids, f"Engine '{eid}' not declared in manifest"

    def test_each_engine_has_required_fields(self, manifest):
        engines = manifest["components"]["engines"]
        for eng in engines:
            assert "id" in eng
            assert "name" in eng
            assert "file" in eng
            assert "class" in eng
            assert "description" in eng
            assert len(eng["description"]) > 50

    def test_engine_files_exist(self, manifest):
        engines = manifest["components"]["engines"]
        for eng in engines:
            filepath = PACK_ROOT / eng["file"]
            assert filepath.exists(), f"Engine file {eng['file']} does not exist"

    def test_articles_covered(self, manifest):
        meta = manifest["metadata"]
        articles = meta.get("articles_covered", [])
        assert isinstance(articles, list)
        assert 7 in articles
        assert 8 in articles
        assert 77 in articles
        assert 78 in articles
        assert 48 in articles

    def test_carbon_footprint_performance_classes(self, manifest):
        meta = manifest["metadata"]
        classes = meta.get("carbon_footprint_performance_classes", {})
        assert "class_a" in classes
        assert "class_e" in classes
        assert classes["class_a"]["label"] == "A"

    def test_battery_chemistries_declared(self, manifest):
        meta = manifest["metadata"]
        chemistries = meta.get("battery_chemistries", [])
        assert isinstance(chemistries, list)
        assert len(chemistries) >= 10
        assert "NMC" in chemistries
        assert "LFP" in chemistries
        assert "NMC811" in chemistries
