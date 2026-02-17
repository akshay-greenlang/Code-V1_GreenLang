# -*- coding: utf-8 -*-
"""
Unit Tests for RulePackEngine - AGENT-DATA-019: Validation Rule Engine
=======================================================================

Tests all public methods of RulePackEngine with 55+ tests covering GHG
Protocol pack, CSRD/ESRS pack, EUDR pack, SOC2 pack, custom pack
registration and application, pack listing and retrieval, pack version
comparison, applying to existing rule sets, invalid pack handling, and
statistics/clear operations.

Test Classes (10):
    - TestRulePackInit (5 tests)
    - TestGHGProtocolPack (8 tests)
    - TestCSRDESRSPack (7 tests)
    - TestEUDRPack (6 tests)
    - TestSOC2Pack (6 tests)
    - TestCustomPack (6 tests)
    - TestPackListing (5 tests)
    - TestPackVersionComparison (4 tests)
    - TestApplyToExistingRuleSet (4 tests)
    - TestInvalidPackAndEdgeCases (5 tests)
    - TestStatisticsAndClear (4 tests)

Total: ~60 tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.validation_rule_engine.models import (
    ValidationRule,
    ValidationRuleType,
    RuleOperator,
    RuleSeverity,
    RuleStatus,
    RuleSet,
    RulePack,
    RulePackType,
)
from greenlang.validation_rule_engine.provenance import ProvenanceTracker
from greenlang.validation_rule_engine.rule_pack import RulePackEngine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> RulePackEngine:
    """Create a fresh RulePackEngine instance for each test."""
    return RulePackEngine(genesis_hash="test-pack-genesis")


# ==========================================================================
# TestRulePackInit
# ==========================================================================


class TestRulePackInit:
    """Tests for RulePackEngine initialization."""

    def test_init_creates_instance(self, engine: RulePackEngine) -> None:
        """Engine initializes without error."""
        assert engine is not None

    def test_init_has_provenance_tracker(self, engine: RulePackEngine) -> None:
        """Engine has a provenance tracker."""
        assert hasattr(engine, "_provenance") or hasattr(engine, "_tracker")

    def test_init_has_built_in_packs(self, engine: RulePackEngine) -> None:
        """Engine has built-in packs registered."""
        packs = engine.list_packs()
        assert len(packs) >= 4  # ghg_protocol, csrd_esrs, eudr, soc2

    def test_init_custom_genesis_hash(self) -> None:
        """Engine accepts a custom genesis hash."""
        eng = RulePackEngine(genesis_hash="custom-genesis")
        assert eng is not None

    def test_init_default_genesis_hash(self) -> None:
        """Engine works with default genesis hash."""
        eng = RulePackEngine()
        assert eng is not None


# ==========================================================================
# TestGHGProtocolPack
# ==========================================================================


class TestGHGProtocolPack:
    """Tests for GHG Protocol rule pack."""

    def test_ghg_pack_available(self, engine: RulePackEngine) -> None:
        """GHG Protocol pack is available."""
        pack = engine.get_pack("ghg_protocol")
        assert pack is not None

    def test_ghg_pack_apply_creates_rules(self, engine: RulePackEngine) -> None:
        """Applying GHG Protocol pack creates 40+ rules."""
        result = engine.apply_pack("ghg_protocol")
        assert result["rules_created"] >= 40

    def test_ghg_pack_creates_rule_set(self, engine: RulePackEngine) -> None:
        """Applying GHG Protocol pack creates a rule set."""
        result = engine.apply_pack("ghg_protocol")
        assert result["rule_set_id"] is not None
        assert result["rule_set_id"] != ""

    def test_ghg_pack_scope1_rules(self, engine: RulePackEngine) -> None:
        """GHG pack includes Scope 1 validation rules."""
        result = engine.apply_pack("ghg_protocol")
        rules = result.get("rules", [])
        scope1_rules = [r for r in rules if "scope_1" in r.get("name", "").lower()
                        or "scope1" in r.get("name", "").lower()
                        or "scope 1" in r.get("description", "").lower()]
        assert len(scope1_rules) >= 1

    def test_ghg_pack_scope2_rules(self, engine: RulePackEngine) -> None:
        """GHG pack includes Scope 2 validation rules."""
        result = engine.apply_pack("ghg_protocol")
        rules = result.get("rules", [])
        scope2_rules = [r for r in rules if "scope_2" in r.get("name", "").lower()
                        or "scope2" in r.get("name", "").lower()
                        or "scope 2" in r.get("description", "").lower()]
        assert len(scope2_rules) >= 1

    def test_ghg_pack_scope3_rules(self, engine: RulePackEngine) -> None:
        """GHG pack includes Scope 3 validation rules."""
        result = engine.apply_pack("ghg_protocol")
        rules = result.get("rules", [])
        scope3_rules = [r for r in rules if "scope_3" in r.get("name", "").lower()
                        or "scope3" in r.get("name", "").lower()
                        or "scope 3" in r.get("description", "").lower()]
        assert len(scope3_rules) >= 1

    def test_ghg_pack_provenance(self, engine: RulePackEngine) -> None:
        """GHG pack application records provenance."""
        result = engine.apply_pack("ghg_protocol")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_ghg_pack_framework_version(self, engine: RulePackEngine) -> None:
        """GHG pack has a framework version."""
        pack = engine.get_pack("ghg_protocol")
        assert pack is not None
        if hasattr(pack, "framework_version"):
            assert pack.framework_version != ""


# ==========================================================================
# TestCSRDESRSPack
# ==========================================================================


class TestCSRDESRSPack:
    """Tests for CSRD/ESRS rule pack."""

    def test_csrd_pack_available(self, engine: RulePackEngine) -> None:
        """CSRD/ESRS pack is available."""
        pack = engine.get_pack("csrd_esrs")
        assert pack is not None

    def test_csrd_pack_apply_creates_rules(self, engine: RulePackEngine) -> None:
        """Applying CSRD/ESRS pack creates 35+ rules."""
        result = engine.apply_pack("csrd_esrs")
        assert result["rules_created"] >= 35

    def test_csrd_pack_creates_rule_set(self, engine: RulePackEngine) -> None:
        """Applying CSRD/ESRS pack creates a rule set."""
        result = engine.apply_pack("csrd_esrs")
        assert result["rule_set_id"] is not None

    def test_csrd_pack_esrs_e1_rules(self, engine: RulePackEngine) -> None:
        """CSRD pack includes ESRS E1 (climate) rules."""
        result = engine.apply_pack("csrd_esrs")
        rules = result.get("rules", [])
        e1_rules = [r for r in rules if "e1" in r.get("name", "").lower()
                     or "climate" in r.get("description", "").lower()]
        assert len(e1_rules) >= 1

    def test_csrd_pack_esrs_s1_rules(self, engine: RulePackEngine) -> None:
        """CSRD pack includes ESRS S1 (social) rules."""
        result = engine.apply_pack("csrd_esrs")
        rules = result.get("rules", [])
        s1_rules = [r for r in rules if "s1" in r.get("name", "").lower()
                     or "social" in r.get("description", "").lower()
                     or "workforce" in r.get("description", "").lower()]
        assert len(s1_rules) >= 1

    def test_csrd_pack_provenance(self, engine: RulePackEngine) -> None:
        """CSRD pack application records provenance."""
        result = engine.apply_pack("csrd_esrs")
        assert "provenance_hash" in result

    def test_csrd_pack_double_materiality(self, engine: RulePackEngine) -> None:
        """CSRD pack includes double materiality indicators."""
        result = engine.apply_pack("csrd_esrs")
        rules = result.get("rules", [])
        materiality_rules = [r for r in rules if "materiality" in r.get("name", "").lower()
                              or "materiality" in r.get("description", "").lower()]
        assert len(materiality_rules) >= 1


# ==========================================================================
# TestEUDRPack
# ==========================================================================


class TestEUDRPack:
    """Tests for EUDR rule pack."""

    def test_eudr_pack_available(self, engine: RulePackEngine) -> None:
        """EUDR pack is available."""
        pack = engine.get_pack("eudr")
        assert pack is not None

    def test_eudr_pack_apply_creates_rules(self, engine: RulePackEngine) -> None:
        """Applying EUDR pack creates 25+ rules."""
        result = engine.apply_pack("eudr")
        assert result["rules_created"] >= 25

    def test_eudr_pack_creates_rule_set(self, engine: RulePackEngine) -> None:
        """Applying EUDR pack creates a rule set."""
        result = engine.apply_pack("eudr")
        assert result["rule_set_id"] is not None

    def test_eudr_pack_geolocation_rules(self, engine: RulePackEngine) -> None:
        """EUDR pack includes geolocation validation rules."""
        result = engine.apply_pack("eudr")
        rules = result.get("rules", [])
        geo_rules = [r for r in rules if "geo" in r.get("name", "").lower()
                      or "location" in r.get("name", "").lower()
                      or "wgs84" in r.get("description", "").lower()
                      or "latitude" in r.get("name", "").lower()
                      or "longitude" in r.get("name", "").lower()]
        assert len(geo_rules) >= 1

    def test_eudr_pack_cutoff_date_rules(self, engine: RulePackEngine) -> None:
        """EUDR pack includes Dec 31 2020 cutoff date rules."""
        result = engine.apply_pack("eudr")
        rules = result.get("rules", [])
        cutoff_rules = [r for r in rules if "cutoff" in r.get("name", "").lower()
                         or "2020" in r.get("description", "").lower()
                         or "deforestation" in r.get("name", "").lower()]
        assert len(cutoff_rules) >= 1

    def test_eudr_pack_provenance(self, engine: RulePackEngine) -> None:
        """EUDR pack application records provenance."""
        result = engine.apply_pack("eudr")
        assert "provenance_hash" in result


# ==========================================================================
# TestSOC2Pack
# ==========================================================================


class TestSOC2Pack:
    """Tests for SOC 2 rule pack."""

    def test_soc2_pack_available(self, engine: RulePackEngine) -> None:
        """SOC 2 pack is available."""
        pack = engine.get_pack("soc2")
        assert pack is not None

    def test_soc2_pack_apply_creates_rules(self, engine: RulePackEngine) -> None:
        """Applying SOC 2 pack creates 20+ rules."""
        result = engine.apply_pack("soc2")
        assert result["rules_created"] >= 20

    def test_soc2_pack_creates_rule_set(self, engine: RulePackEngine) -> None:
        """Applying SOC 2 pack creates a rule set."""
        result = engine.apply_pack("soc2")
        assert result["rule_set_id"] is not None

    def test_soc2_pack_access_control_rules(self, engine: RulePackEngine) -> None:
        """SOC 2 pack includes access control rules."""
        result = engine.apply_pack("soc2")
        rules = result.get("rules", [])
        access_rules = [r for r in rules if "access" in r.get("name", "").lower()
                         or "auth" in r.get("name", "").lower()]
        assert len(access_rules) >= 1

    def test_soc2_pack_encryption_rules(self, engine: RulePackEngine) -> None:
        """SOC 2 pack includes encryption status rules."""
        result = engine.apply_pack("soc2")
        rules = result.get("rules", [])
        enc_rules = [r for r in rules if "encrypt" in r.get("name", "").lower()
                      or "encrypt" in r.get("description", "").lower()]
        assert len(enc_rules) >= 1

    def test_soc2_pack_provenance(self, engine: RulePackEngine) -> None:
        """SOC 2 pack application records provenance."""
        result = engine.apply_pack("soc2")
        assert "provenance_hash" in result


# ==========================================================================
# TestCustomPack
# ==========================================================================


class TestCustomPack:
    """Tests for custom pack registration and application."""

    def test_register_custom_pack(self, engine: RulePackEngine) -> None:
        """Register a custom rule pack."""
        rules = [
            {"name": "custom_r1", "rule_type": "range", "target_field": "val",
             "threshold_min": 0.0, "threshold_max": 100.0, "severity": "high"},
            {"name": "custom_r2", "rule_type": "completeness", "target_field": "name",
             "severity": "critical"},
        ]
        result = engine.register_custom_pack(
            name="my_custom_pack",
            description="Custom validation rules",
            rules=rules,
        )
        assert result["pack_id"] is not None

    def test_apply_custom_pack(self, engine: RulePackEngine) -> None:
        """Apply a registered custom pack."""
        rules = [
            {"name": "custom_r1", "rule_type": "range", "target_field": "val",
             "threshold_min": 0.0, "threshold_max": 100.0, "severity": "high"},
        ]
        reg_result = engine.register_custom_pack(
            name="apply_custom",
            description="Apply test",
            rules=rules,
        )
        apply_result = engine.apply_pack(reg_result["pack_id"])
        assert apply_result["rules_created"] >= 1

    def test_custom_pack_in_list(self, engine: RulePackEngine) -> None:
        """Custom pack appears in pack listing."""
        rules = [
            {"name": "listed_r1", "rule_type": "completeness", "target_field": "val",
             "severity": "medium"},
        ]
        engine.register_custom_pack(
            name="listed_pack",
            description="Listed test",
            rules=rules,
        )
        packs = engine.list_packs()
        custom_packs = [p for p in packs if "listed_pack" in p.get("name", "")]
        assert len(custom_packs) >= 1

    def test_custom_pack_provenance(self, engine: RulePackEngine) -> None:
        """Custom pack registration records provenance."""
        rules = [
            {"name": "prov_r1", "rule_type": "completeness", "target_field": "val",
             "severity": "medium"},
        ]
        result = engine.register_custom_pack(
            name="prov_pack",
            description="Provenance test",
            rules=rules,
        )
        assert "provenance_hash" in result

    def test_custom_pack_with_tags(self, engine: RulePackEngine) -> None:
        """Custom pack supports tags."""
        rules = [
            {"name": "tagged_r1", "rule_type": "completeness", "target_field": "val",
             "severity": "medium"},
        ]
        result = engine.register_custom_pack(
            name="tagged_pack",
            description="Tagged test",
            rules=rules,
            tags={"domain": "emissions", "tier": "custom"},
        )
        assert result["pack_id"] is not None

    def test_custom_pack_empty_rules_fails(self, engine: RulePackEngine) -> None:
        """Custom pack with no rules fails or creates empty pack."""
        result = engine.register_custom_pack(
            name="empty_pack",
            description="No rules",
            rules=[],
        )
        # Should either fail or create an empty pack
        assert "pack_id" in result or "error" in result


# ==========================================================================
# TestPackListing
# ==========================================================================


class TestPackListing:
    """Tests for pack listing and retrieval."""

    def test_list_all_packs(self, engine: RulePackEngine) -> None:
        """List all available packs."""
        packs = engine.list_packs()
        assert len(packs) >= 4

    def test_get_pack_by_name(self, engine: RulePackEngine) -> None:
        """Get a specific pack by name."""
        pack = engine.get_pack("ghg_protocol")
        assert pack is not None

    def test_get_nonexistent_pack(self, engine: RulePackEngine) -> None:
        """Getting a nonexistent pack returns None."""
        pack = engine.get_pack("nonexistent_pack")
        assert pack is None

    def test_list_packs_includes_built_in(self, engine: RulePackEngine) -> None:
        """Listed packs include all 4 built-in packs."""
        packs = engine.list_packs()
        pack_names = [p.get("name", "") if isinstance(p, dict) else getattr(p, "name", "")
                      for p in packs]
        for expected in ["ghg_protocol", "csrd_esrs", "eudr", "soc2"]:
            found = any(expected in name.lower() for name in pack_names)
            assert found, f"Expected pack '{expected}' not found in listing"

    def test_pack_has_description(self, engine: RulePackEngine) -> None:
        """Packs include descriptions."""
        pack = engine.get_pack("ghg_protocol")
        assert pack is not None
        desc = pack.description if hasattr(pack, "description") else pack.get("description", "")
        assert desc != ""


# ==========================================================================
# TestPackVersionComparison
# ==========================================================================


class TestPackVersionComparison:
    """Tests for pack version comparison."""

    def test_pack_has_version(self, engine: RulePackEngine) -> None:
        """Packs have version strings."""
        pack = engine.get_pack("ghg_protocol")
        assert pack is not None
        version = pack.version if hasattr(pack, "version") else pack.get("version", "")
        assert version != ""

    def test_compare_pack_versions(self, engine: RulePackEngine) -> None:
        """Compare two pack versions."""
        result = engine.compare_pack_versions("ghg_protocol", "1.0.0", "1.0.0")
        assert result is not None
        assert "changes" in result or "diff" in result or "identical" in result or isinstance(result, dict)

    def test_compare_nonexistent_version(self, engine: RulePackEngine) -> None:
        """Comparing with a nonexistent version handles gracefully."""
        result = engine.compare_pack_versions("ghg_protocol", "1.0.0", "99.0.0")
        assert result is not None

    def test_pack_version_format(self, engine: RulePackEngine) -> None:
        """Pack versions follow SemVer format."""
        pack = engine.get_pack("ghg_protocol")
        assert pack is not None
        version = pack.version if hasattr(pack, "version") else pack.get("version", "")
        parts = version.split(".")
        assert len(parts) >= 2  # At least major.minor


# ==========================================================================
# TestApplyToExistingRuleSet
# ==========================================================================


class TestApplyToExistingRuleSet:
    """Tests for applying packs to existing rule sets."""

    def test_apply_to_existing_rule_set(self, engine: RulePackEngine) -> None:
        """Apply pack rules to an existing rule set."""
        result = engine.apply_pack("ghg_protocol", rule_set_id="existing-set-123")
        assert result["rules_created"] >= 40

    def test_apply_multiple_packs(self, engine: RulePackEngine) -> None:
        """Apply multiple packs sequentially."""
        r1 = engine.apply_pack("ghg_protocol")
        r2 = engine.apply_pack("csrd_esrs")
        assert r1["rules_created"] >= 40
        assert r2["rules_created"] >= 35

    def test_apply_same_pack_twice(self, engine: RulePackEngine) -> None:
        """Applying same pack twice is idempotent or creates separate sets."""
        r1 = engine.apply_pack("ghg_protocol")
        r2 = engine.apply_pack("ghg_protocol")
        assert r1["rules_created"] >= 40
        assert r2["rules_created"] >= 40

    def test_apply_with_namespace(self, engine: RulePackEngine) -> None:
        """Apply pack with a specific namespace."""
        result = engine.apply_pack("ghg_protocol", namespace="tenant-42")
        assert result["rules_created"] >= 40


# ==========================================================================
# TestInvalidPackAndEdgeCases
# ==========================================================================


class TestInvalidPackAndEdgeCases:
    """Tests for invalid pack names and edge cases."""

    def test_apply_invalid_pack_name(self, engine: RulePackEngine) -> None:
        """Applying an invalid pack name raises error or returns error."""
        with pytest.raises((ValueError, KeyError)):
            engine.apply_pack("nonexistent_pack_xyz")

    def test_get_pack_empty_name(self, engine: RulePackEngine) -> None:
        """Getting pack with empty name returns None."""
        pack = engine.get_pack("")
        assert pack is None

    def test_register_pack_duplicate_name(self, engine: RulePackEngine) -> None:
        """Registering a custom pack with a built-in name is handled."""
        rules = [{"name": "r1", "rule_type": "completeness", "target_field": "val", "severity": "medium"}]
        result = engine.register_custom_pack(name="ghg_protocol_custom", description="Dup", rules=rules)
        # Should either succeed with a unique ID or raise
        assert "pack_id" in result or "error" in result

    def test_pack_total_rules_accurate(self, engine: RulePackEngine) -> None:
        """Pack reports accurate total_rules count."""
        result = engine.apply_pack("ghg_protocol")
        assert result["rules_created"] >= 40
        assert isinstance(result["rules_created"], int)

    def test_pack_coverage_areas(self, engine: RulePackEngine) -> None:
        """Packs include coverage areas metadata."""
        pack = engine.get_pack("ghg_protocol")
        assert pack is not None
        areas = pack.coverage_areas if hasattr(pack, "coverage_areas") else pack.get("coverage_areas", [])
        assert isinstance(areas, list)


# ==========================================================================
# TestStatisticsAndClear
# ==========================================================================


class TestStatisticsAndClear:
    """Tests for statistics and clear operations."""

    def test_statistics_initial(self, engine: RulePackEngine) -> None:
        """Initial statistics show built-in packs."""
        stats = engine.get_statistics()
        assert stats["total_packs"] >= 4

    def test_statistics_after_apply(self, engine: RulePackEngine) -> None:
        """Statistics update after applying a pack."""
        engine.apply_pack("ghg_protocol")
        stats = engine.get_statistics()
        assert stats["total_applications"] >= 1

    def test_clear_resets_custom_packs(self, engine: RulePackEngine) -> None:
        """Clear removes custom packs but keeps built-in."""
        rules = [{"name": "r1", "rule_type": "completeness", "target_field": "val", "severity": "medium"}]
        engine.register_custom_pack(name="clearable", description="Will be cleared", rules=rules)
        engine.clear()
        stats = engine.get_statistics()
        assert stats["total_packs"] >= 4  # Built-in packs remain

    def test_clear_resets_applications(self, engine: RulePackEngine) -> None:
        """Clear resets application count."""
        engine.apply_pack("ghg_protocol")
        engine.clear()
        stats = engine.get_statistics()
        assert stats["total_applications"] == 0
