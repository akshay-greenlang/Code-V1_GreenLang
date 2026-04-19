# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Pack Manifest Validation Tests
=============================================================

Validates the integrity and structure of pack.yaml for the EUDR Starter Pack.
Ensures all EUDR-specific metadata, commodity coverage, engine definitions,
workflow definitions, template listings, integration specifications, agent
references, presets, and sectors are correctly declared.

Test count: 15
Author: GreenLang QA Team
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from conftest import (
    PACK_YAML_PATH,
    EUDR_COMMODITIES,
    ANNEX_I_CN_CODES,
    ALL_CN_CODES,
)


# ---------------------------------------------------------------------------
# Helper: load YAML safely
# ---------------------------------------------------------------------------

def _load_pack_yaml() -> Dict[str, Any]:
    """Load pack.yaml, returning empty dict if not found."""
    if PACK_YAML_PATH.exists():
        import yaml
        return yaml.safe_load(PACK_YAML_PATH.read_text(encoding="utf-8")) or {}
    return {}


PACK_DATA = _load_pack_yaml()
SKIP_IF_NO_MANIFEST = pytest.mark.skipif(
    not PACK_YAML_PATH.exists(),
    reason="pack.yaml not yet created",
)


# =========================================================================
# Manifest Tests
# =========================================================================


class TestManifest:
    """Validates PACK-006 pack.yaml structure and EUDR-specific content."""

    # 1
    @SKIP_IF_NO_MANIFEST
    def test_manifest_exists(self, pack_yaml_path):
        """pack.yaml file exists at the expected location."""
        assert pack_yaml_path.exists(), (
            f"pack.yaml not found at {pack_yaml_path}"
        )

    # 2
    @SKIP_IF_NO_MANIFEST
    def test_manifest_valid_yaml(self, pack_yaml):
        """pack.yaml parses as valid YAML into a non-empty dict."""
        assert isinstance(pack_yaml, dict), (
            "pack.yaml did not parse into a dictionary"
        )
        assert len(pack_yaml) > 0, "pack.yaml is empty"

    # 3
    @SKIP_IF_NO_MANIFEST
    def test_manifest_has_metadata(self, pack_yaml):
        """pack.yaml contains a metadata section with required fields."""
        assert "metadata" in pack_yaml, "Missing 'metadata' section"
        meta = pack_yaml["metadata"]
        required_fields = [
            "name", "version", "category", "display_name", "description",
        ]
        for field in required_fields:
            assert field in meta, f"Missing metadata field: {field}"

    # 4
    @SKIP_IF_NO_MANIFEST
    def test_manifest_tier_is_starter(self, pack_yaml):
        """pack.yaml metadata indicates this is a 'starter' tier pack."""
        meta = pack_yaml.get("metadata", {})
        # Check name or tier field
        name = meta.get("name", "")
        assert "eudr" in name.lower(), f"Pack name should contain 'eudr': {name}"
        assert "starter" in name.lower() or meta.get("tier", "").lower() == "starter", (
            "Pack should be tier=starter or name should include 'starter'"
        )

    # 5
    @SKIP_IF_NO_MANIFEST
    def test_manifest_regulation_eudr(self, pack_yaml):
        """pack.yaml references the EU Deforestation Regulation."""
        meta = pack_yaml.get("metadata", {})
        # Check compliance_references or regulation field
        refs = meta.get("compliance_references", [])
        if refs:
            ref_ids = {r.get("id", "") for r in refs}
            assert "EUDR" in ref_ids or any("deforestation" in str(r).lower() for r in refs), (
                "Missing EUDR compliance reference"
            )
        else:
            regulation = meta.get("regulation", "")
            assert "deforestation" in regulation.lower() or "eudr" in regulation.lower() or "1115" in regulation, (
                "pack.yaml should reference the EU Deforestation Regulation"
            )

    # 6
    @SKIP_IF_NO_MANIFEST
    def test_manifest_seven_commodities(self, pack_yaml):
        """pack.yaml lists all 7 EUDR-regulated commodities."""
        # Look in metadata, config, or commodities section
        commodities_found = set()
        yaml_str = str(pack_yaml).lower()
        for c in EUDR_COMMODITIES:
            if c.lower().replace("_", " ") in yaml_str or c.lower() in yaml_str:
                commodities_found.add(c)
        assert len(commodities_found) >= 7, (
            f"Expected 7 EUDR commodities, found {len(commodities_found)}: {commodities_found}. "
            f"Missing: {set(EUDR_COMMODITIES) - commodities_found}"
        )

    # 7
    @SKIP_IF_NO_MANIFEST
    def test_manifest_engine_count(self, pack_yaml):
        """pack.yaml declares at least 7 engines."""
        engines = pack_yaml.get("engines", pack_yaml.get("components", {}).get("engines", []))
        if isinstance(engines, dict):
            engine_count = len(engines)
        elif isinstance(engines, list):
            engine_count = len(engines)
        else:
            engine_count = 0
        assert engine_count >= 7, (
            f"Expected at least 7 engines, found {engine_count}"
        )

    # 8
    @SKIP_IF_NO_MANIFEST
    def test_manifest_workflow_count(self, pack_yaml):
        """pack.yaml declares at least 6 workflows."""
        # Look in top-level 'workflows' or nested under 'components.workflows'
        workflows = pack_yaml.get("workflows", [])
        if not workflows:
            workflows = pack_yaml.get("components", {}).get("workflows", [])
        if isinstance(workflows, dict):
            wf_count = len(workflows)
        elif isinstance(workflows, list):
            wf_count = len(workflows)
        else:
            wf_count = 0
        assert wf_count >= 6, (
            f"Expected at least 6 workflows, found {wf_count}"
        )

    # 9
    @SKIP_IF_NO_MANIFEST
    def test_manifest_template_count(self, pack_yaml):
        """pack.yaml declares at least 7 templates."""
        # Look in top-level 'templates' or nested under 'components.templates'
        # Also check 'report_templates' as an alternative key
        templates = pack_yaml.get("templates", [])
        if not templates:
            templates = pack_yaml.get("components", {}).get("templates", [])
        if not templates:
            templates = pack_yaml.get("report_templates", [])
        if isinstance(templates, list):
            tmpl_count = len(templates)
        elif isinstance(templates, dict):
            tmpl_count = len(templates)
        else:
            tmpl_count = 0
        assert tmpl_count >= 7, (
            f"Expected at least 7 templates, found {tmpl_count}"
        )

    # 10
    @SKIP_IF_NO_MANIFEST
    def test_manifest_integration_count(self, pack_yaml):
        """pack.yaml declares at least 8 integrations."""
        # Look in top-level 'integrations' or nested under 'components.integrations'
        integrations = pack_yaml.get("integrations", [])
        if not integrations:
            integrations = pack_yaml.get("components", {}).get("integrations", [])
        if isinstance(integrations, list):
            int_count = len(integrations)
        elif isinstance(integrations, dict):
            int_count = len(integrations)
        else:
            int_count = 0
        assert int_count >= 8, (
            f"Expected at least 8 integrations, found {int_count}"
        )

    # 11
    @SKIP_IF_NO_MANIFEST
    def test_manifest_eudr_agent_count(self, pack_yaml):
        """pack.yaml references at least 18 EUDR agents."""
        yaml_str = str(pack_yaml)
        # Count unique AGENT-EUDR-NNN references
        agent_refs = set(re.findall(r"AGENT-EUDR-\d{3}", yaml_str))
        assert len(agent_refs) >= 18, (
            f"Expected at least 18 AGENT-EUDR references, found {len(agent_refs)}: {sorted(agent_refs)}"
        )

    # 12
    @SKIP_IF_NO_MANIFEST
    def test_manifest_cn_code_coverage(self, pack_yaml):
        """pack.yaml references CN codes covering EUDR Annex I commodities."""
        # Collect CN codes from the commodities section
        cn_codes_found = set()
        commodities = pack_yaml.get("commodities", {})
        if isinstance(commodities, dict):
            for commodity_key, commodity_data in commodities.items():
                if isinstance(commodity_data, dict):
                    codes = commodity_data.get("cn_codes", [])
                    if isinstance(codes, list):
                        cn_codes_found.update(str(c) for c in codes)
        # Also search for CN code patterns in the full YAML string as fallback
        if len(cn_codes_found) < 10:
            yaml_str = str(pack_yaml)
            # Match 4-digit CN heading codes (e.g., '0102', '1511')
            cn_short = re.findall(r"'(\d{4})'", yaml_str)
            cn_codes_found.update(cn_short)
            # Match full 8-digit codes (e.g., '1511 10 90')
            cn_full = re.findall(r"\d{4}\s?\d{2}\s?\d{2}", yaml_str)
            cn_codes_found.update(cn_full)
        assert len(cn_codes_found) >= 10, (
            f"Expected at least 10 CN code references, found {len(cn_codes_found)}"
        )

    # 13
    @SKIP_IF_NO_MANIFEST
    def test_manifest_presets_listed(self, pack_yaml):
        """pack.yaml declares at least 4 presets."""
        presets = pack_yaml.get("presets", {})
        if isinstance(presets, dict):
            size_presets = presets.get("size_presets", presets.get("presets", []))
            if isinstance(size_presets, list):
                preset_count = len(size_presets)
            else:
                preset_count = len(presets)
        elif isinstance(presets, list):
            preset_count = len(presets)
        else:
            preset_count = 0
        assert preset_count >= 4, (
            f"Expected at least 4 presets, found {preset_count}"
        )

    # 14
    @SKIP_IF_NO_MANIFEST
    def test_manifest_sectors_listed(self, pack_yaml):
        """pack.yaml declares at least 5 sectors."""
        presets = pack_yaml.get("presets", {})
        if isinstance(presets, dict):
            sector_presets = presets.get("sector_presets", presets.get("sectors", []))
            if isinstance(sector_presets, list):
                sector_count = len(sector_presets)
            else:
                sector_count = 0
        else:
            sectors = pack_yaml.get("sectors", [])
            sector_count = len(sectors) if isinstance(sectors, list) else 0
        assert sector_count >= 5, (
            f"Expected at least 5 sectors, found {sector_count}"
        )

    # 15
    @SKIP_IF_NO_MANIFEST
    def test_manifest_version_format(self, pack_yaml):
        """pack.yaml version follows semantic versioning (MAJOR.MINOR.PATCH)."""
        meta = pack_yaml.get("metadata", {})
        version = meta.get("version", "")
        semver_pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(semver_pattern, str(version)), (
            f"Version '{version}' does not follow semver (MAJOR.MINOR.PATCH)"
        )
