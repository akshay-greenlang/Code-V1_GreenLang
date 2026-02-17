# -*- coding: utf-8 -*-
"""
Unit Tests for RuleComposerEngine - AGENT-DATA-019
=====================================================

Comprehensive tests for the Rule Composer Engine covering:
- Compound rules: AND with 2+ rules, OR with 2+ rules, NOT with 1 rule
- Validation: NOT with >1 rule, AND/OR with <2, invalid operator,
  nonexistent rule_ids
- Flatten: nested compound rules
- Rule sets: create, get, update, delete, add/remove rules
- SemVer: version bump on rule set changes
- Templates: create, instantiate, with overrides
- Inheritance: child rule sets, override rules, get_inheritance_chain
- Dependencies: add dependency, evaluation order (topological sort),
  cycle detection
- Comparison: diff two rule sets
- Statistics, clear
- Max compound depth enforcement

Target: 111+ test functions, 85%+ coverage of rule_composer.py

Test classes:
    - TestRuleComposerEngineInit         (5 tests)
    - TestCompoundRulesAND               (8 tests)
    - TestCompoundRulesOR                (8 tests)
    - TestCompoundRulesNOT               (7 tests)
    - TestCompoundRuleValidation         (8 tests)
    - TestFlattenCompoundRules           (6 tests)
    - TestRuleSets                       (16 tests)
    - TestRuleSetVersioning              (8 tests)
    - TestTemplates                      (10 tests)
    - TestInheritance                    (10 tests)
    - TestDependencies                   (10 tests)
    - TestComparison                     (6 tests)
    - TestStatisticsAndClear             (5 tests)
    - TestMaxCompoundDepth               (4 tests)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List

import pytest

from greenlang.validation_rule_engine.rule_composer import (
    MAX_COMPOUND_DEPTH,
    VALID_COMPOUND_OPERATORS,
    RuleComposerEngine,
)
from greenlang.validation_rule_engine.rule_registry import (
    RuleRegistryEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> RuleRegistryEngine:
    """Create a fresh RuleRegistryEngine with sample rules pre-registered."""
    engine = RuleRegistryEngine()
    return engine


@pytest.fixture
def composer(registry) -> RuleComposerEngine:
    """Create a fresh RuleComposerEngine linked to the registry."""
    return RuleComposerEngine(registry=registry)


@pytest.fixture
def base_rules(registry) -> List[Dict[str, Any]]:
    """Register 5 base atomic rules for compound rule composition."""
    rules = []
    rules.append(registry.register_rule(
        name="not_null_company",
        rule_type="COMPLETENESS",
        operator="IS_NULL",
        column="company_name",
        parameters={"allow_null": False},
        severity="CRITICAL",
        tags=["completeness"],
    ))
    rules.append(registry.register_rule(
        name="co2_range",
        rule_type="RANGE",
        operator="BETWEEN",
        column="co2_tonnes",
        parameters={"min_value": 0, "max_value": 1000000},
        severity="HIGH",
        tags=["emissions"],
    ))
    rules.append(registry.register_rule(
        name="email_format",
        rule_type="FORMAT",
        operator="MATCHES",
        column="email",
        parameters={"pattern": r"^[\w.]+@[\w]+\.[\w]+$"},
        severity="MEDIUM",
        tags=["format"],
    ))
    rules.append(registry.register_rule(
        name="country_ref",
        rule_type="REFERENTIAL",
        operator="IN_SET",
        column="country_code",
        parameters={"reference_values": ["US", "DE", "FR", "GB", "JP"]},
        severity="LOW",
        tags=["reference"],
    ))
    rules.append(registry.register_rule(
        name="total_cross_field",
        rule_type="CROSS_FIELD",
        operator="EQUALS",
        column="total_emissions",
        parameters={"related_column": "scope1_plus_scope2"},
        severity="HIGH",
        tags=["cross-field"],
    ))
    return rules


@pytest.fixture
def rule_ids(base_rules) -> List[str]:
    """Extract rule_ids from base_rules fixture."""
    return [r["rule_id"] for r in base_rules]


# ===========================================================================
# TestRuleComposerEngineInit
# ===========================================================================


class TestRuleComposerEngineInit:
    """Test engine initialisation and default state."""

    def test_init_creates_empty_compound_store(self, composer):
        """Engine starts with no compound rules."""
        stats = composer.get_statistics()
        assert stats["compound_rules_count"] == 0

    def test_init_creates_empty_rule_set_store(self, composer):
        """Engine starts with no rule sets."""
        stats = composer.get_statistics()
        assert stats["rule_sets_count"] == 0

    def test_init_creates_empty_template_store(self, composer):
        """Engine starts with no templates."""
        stats = composer.get_statistics()
        assert stats["templates_count"] == 0

    def test_init_has_provenance_tracker(self, composer):
        """Engine has a provenance tracker instance."""
        assert composer._provenance is not None

    def test_init_has_thread_lock(self, composer):
        """Engine has a threading RLock for thread safety."""
        assert isinstance(composer._lock, type(threading.RLock()))


# ===========================================================================
# TestCompoundRulesAND
# ===========================================================================


class TestCompoundRulesAND:
    """Test AND compound rule composition."""

    def test_and_with_two_rules(self, composer, rule_ids):
        """AND compound with two rules creates valid compound."""
        compound = composer.create_compound_rule(
            name="and_2",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        assert compound["operator"] == "AND"
        assert len(compound["rule_ids"]) == 2
        assert compound["compound_id"] is not None

    def test_and_with_three_rules(self, composer, rule_ids):
        """AND compound with three rules creates valid compound."""
        compound = composer.create_compound_rule(
            name="and_3",
            operator="AND",
            rule_ids=rule_ids[:3],
        )
        assert compound["operator"] == "AND"
        assert len(compound["rule_ids"]) == 3

    def test_and_with_all_five_rules(self, composer, rule_ids):
        """AND compound with all five rules creates valid compound."""
        compound = composer.create_compound_rule(
            name="and_5",
            operator="AND",
            rule_ids=rule_ids,
        )
        assert len(compound["rule_ids"]) == 5

    def test_and_generates_prefixed_id(self, composer, rule_ids):
        """AND compound has a CMP-prefixed compound_id."""
        compound = composer.create_compound_rule(
            name="and_uuid",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        assert compound["compound_id"].startswith("CMP-")
        assert len(compound["compound_id"]) > 4

    def test_and_has_provenance_hash(self, composer, rule_ids):
        """AND compound has a non-empty provenance_hash."""
        compound = composer.create_compound_rule(
            name="and_prov",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        assert compound["provenance_hash"] != ""
        assert len(compound["provenance_hash"]) == 64

    def test_and_with_description(self, composer, rule_ids):
        """AND compound stores description correctly."""
        compound = composer.create_compound_rule(
            name="and_desc",
            operator="AND",
            rule_ids=rule_ids[:2],
            description="Both rules must pass.",
        )
        assert compound["description"] == "Both rules must pass."

    def test_and_get_compound_by_id(self, composer, rule_ids):
        """Retrieve AND compound by ID."""
        compound = composer.create_compound_rule(
            name="and_get",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        result = composer.get_compound_rule(compound["compound_id"])
        assert result is not None
        assert result["operator"] == "AND"

    def test_and_with_nested_compound(self, composer, rule_ids):
        """AND compound can include another compound rule ID."""
        inner = composer.create_compound_rule(
            name="inner_and",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        outer = composer.create_compound_rule(
            name="outer_and",
            operator="AND",
            rule_ids=[inner["compound_id"], rule_ids[2]],
        )
        assert len(outer["rule_ids"]) == 2


# ===========================================================================
# TestCompoundRulesOR
# ===========================================================================


class TestCompoundRulesOR:
    """Test OR compound rule composition."""

    def test_or_with_two_rules(self, composer, rule_ids):
        """OR compound with two rules creates valid compound."""
        compound = composer.create_compound_rule(
            name="or_2",
            operator="OR",
            rule_ids=rule_ids[:2],
        )
        assert compound["operator"] == "OR"
        assert len(compound["rule_ids"]) == 2

    def test_or_with_three_rules(self, composer, rule_ids):
        """OR compound with three rules creates valid compound."""
        compound = composer.create_compound_rule(
            name="or_3",
            operator="OR",
            rule_ids=rule_ids[:3],
        )
        assert len(compound["rule_ids"]) == 3

    def test_or_with_all_rules(self, composer, rule_ids):
        """OR compound with all rules creates valid compound."""
        compound = composer.create_compound_rule(
            name="or_all",
            operator="OR",
            rule_ids=rule_ids,
        )
        assert len(compound["rule_ids"]) == 5

    def test_or_generates_prefixed_id(self, composer, rule_ids):
        """OR compound has a CMP-prefixed compound_id."""
        compound = composer.create_compound_rule(
            name="or_uuid",
            operator="OR",
            rule_ids=rule_ids[:2],
        )
        assert compound["compound_id"].startswith("CMP-")
        assert len(compound["compound_id"]) > 4

    def test_or_has_provenance_hash(self, composer, rule_ids):
        """OR compound has a non-empty provenance hash."""
        compound = composer.create_compound_rule(
            name="or_prov",
            operator="OR",
            rule_ids=rule_ids[:2],
        )
        assert len(compound["provenance_hash"]) == 64

    def test_or_with_description(self, composer, rule_ids):
        """OR compound stores description correctly."""
        compound = composer.create_compound_rule(
            name="or_desc",
            operator="OR",
            rule_ids=rule_ids[:2],
            description="At least one rule must pass.",
        )
        assert compound["description"] == "At least one rule must pass."

    def test_or_get_compound_by_id(self, composer, rule_ids):
        """Retrieve OR compound by ID."""
        compound = composer.create_compound_rule(
            name="or_get",
            operator="OR",
            rule_ids=rule_ids[:2],
        )
        result = composer.get_compound_rule(compound["compound_id"])
        assert result is not None
        assert result["operator"] == "OR"

    def test_or_with_nested_compound(self, composer, rule_ids):
        """OR compound can include another compound rule ID."""
        inner = composer.create_compound_rule(
            name="inner_or",
            operator="OR",
            rule_ids=rule_ids[:2],
        )
        outer = composer.create_compound_rule(
            name="outer_or",
            operator="OR",
            rule_ids=[inner["compound_id"], rule_ids[3]],
        )
        assert len(outer["rule_ids"]) == 2


# ===========================================================================
# TestCompoundRulesNOT
# ===========================================================================


class TestCompoundRulesNOT:
    """Test NOT compound rule composition."""

    def test_not_with_one_rule(self, composer, rule_ids):
        """NOT compound with exactly one rule creates valid compound."""
        compound = composer.create_compound_rule(
            name="not_1",
            operator="NOT",
            rule_ids=[rule_ids[0]],
        )
        assert compound["operator"] == "NOT"
        assert len(compound["rule_ids"]) == 1

    def test_not_generates_prefixed_id(self, composer, rule_ids):
        """NOT compound has a CMP-prefixed compound_id."""
        compound = composer.create_compound_rule(
            name="not_uuid",
            operator="NOT",
            rule_ids=[rule_ids[0]],
        )
        assert compound["compound_id"].startswith("CMP-")
        assert len(compound["compound_id"]) > 4

    def test_not_has_provenance_hash(self, composer, rule_ids):
        """NOT compound has a non-empty provenance hash."""
        compound = composer.create_compound_rule(
            name="not_prov",
            operator="NOT",
            rule_ids=[rule_ids[0]],
        )
        assert len(compound["provenance_hash"]) == 64

    def test_not_with_description(self, composer, rule_ids):
        """NOT compound stores description correctly."""
        compound = composer.create_compound_rule(
            name="not_desc",
            operator="NOT",
            rule_ids=[rule_ids[0]],
            description="Inverts the rule result.",
        )
        assert compound["description"] == "Inverts the rule result."

    def test_not_get_compound_by_id(self, composer, rule_ids):
        """Retrieve NOT compound by ID."""
        compound = composer.create_compound_rule(
            name="not_get",
            operator="NOT",
            rule_ids=[rule_ids[0]],
        )
        result = composer.get_compound_rule(compound["compound_id"])
        assert result is not None
        assert result["operator"] == "NOT"

    def test_not_with_nested_compound(self, composer, rule_ids):
        """NOT can negate another compound rule."""
        inner = composer.create_compound_rule(
            name="inner_for_not",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        outer = composer.create_compound_rule(
            name="not_nested",
            operator="NOT",
            rule_ids=[inner["compound_id"]],
        )
        assert outer["operator"] == "NOT"
        assert len(outer["rule_ids"]) == 1

    def test_not_preserves_rule_id(self, composer, rule_ids):
        """NOT compound stores the exact rule ID."""
        compound = composer.create_compound_rule(
            name="not_exact",
            operator="NOT",
            rule_ids=[rule_ids[2]],
        )
        assert compound["rule_ids"][0] == rule_ids[2]


# ===========================================================================
# TestCompoundRuleValidation
# ===========================================================================


class TestCompoundRuleValidation:
    """Test compound rule input validation."""

    def test_not_with_multiple_rules_rejected(self, composer, rule_ids):
        """NOT with more than 1 rule raises ValueError."""
        with pytest.raises(ValueError, match="NOT.*exactly 1"):
            composer.create_compound_rule(
                name="not_multi",
                operator="NOT",
                rule_ids=rule_ids[:2],
            )

    def test_and_with_less_than_two_rules_rejected(self, composer, rule_ids):
        """AND with fewer than 2 rules raises ValueError."""
        with pytest.raises(ValueError, match="AND.*2 or more"):
            composer.create_compound_rule(
                name="and_one",
                operator="AND",
                rule_ids=[rule_ids[0]],
            )

    def test_or_with_less_than_two_rules_rejected(self, composer, rule_ids):
        """OR with fewer than 2 rules raises ValueError."""
        with pytest.raises(ValueError, match="OR.*2 or more"):
            composer.create_compound_rule(
                name="or_one",
                operator="OR",
                rule_ids=[rule_ids[0]],
            )

    def test_invalid_operator_rejected(self, composer, rule_ids):
        """Invalid compound operator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid.*operator"):
            composer.create_compound_rule(
                name="bad_op",
                operator="XOR",
                rule_ids=rule_ids[:2],
            )

    def test_empty_rule_ids_rejected(self, composer):
        """Empty rule_ids list raises ValueError."""
        with pytest.raises(ValueError, match="rule_ids"):
            composer.create_compound_rule(
                name="empty_ids",
                operator="AND",
                rule_ids=[],
            )

    def test_nonexistent_rule_id_rejected(self, composer):
        """Nonexistent rule_id raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            composer.create_compound_rule(
                name="bad_id",
                operator="NOT",
                rule_ids=["nonexistent-rule-id"],
            )

    def test_get_nonexistent_compound_returns_none(self, composer):
        """Retrieving a non-existent compound rule returns None."""
        result = composer.get_compound_rule("nonexistent-id")
        assert result is None

    def test_and_with_whitespace_name_stores_stripped(self, composer, rule_ids):
        """Creating a compound rule with whitespace name stores stripped value."""
        compound = composer.create_compound_rule(
            name="  spaced_name  ",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        assert compound["name"] == "spaced_name"


# ===========================================================================
# TestFlattenCompoundRules
# ===========================================================================


class TestFlattenCompoundRules:
    """Test compound rule flattening for debugging and inspection."""

    def test_flatten_simple_and(self, composer, rule_ids):
        """Flatten a simple AND compound returns leaf atomic rule IDs."""
        compound = composer.create_compound_rule(
            name="flat_and",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        flat = composer.flatten_compound_rule(compound["compound_id"])
        # Returns a list of atomic rule IDs
        assert isinstance(flat, list)
        assert len(flat) == 2
        assert rule_ids[0] in flat
        assert rule_ids[1] in flat

    def test_flatten_simple_not(self, composer, rule_ids):
        """Flatten a simple NOT compound returns one atomic rule ID."""
        compound = composer.create_compound_rule(
            name="flat_not",
            operator="NOT",
            rule_ids=[rule_ids[0]],
        )
        flat = composer.flatten_compound_rule(compound["compound_id"])
        assert isinstance(flat, list)
        assert len(flat) == 1
        assert flat[0] == rule_ids[0]

    def test_flatten_nested_compound(self, composer, rule_ids):
        """Flatten nested compound recursively expands to atomic IDs."""
        inner = composer.create_compound_rule(
            name="nested_inner",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        outer = composer.create_compound_rule(
            name="nested_outer",
            operator="OR",
            rule_ids=[inner["compound_id"], rule_ids[2]],
        )
        flat = composer.flatten_compound_rule(outer["compound_id"])
        assert isinstance(flat, list)
        # Should contain the 2 inner atomic IDs + the 1 outer atomic ID = 3
        assert len(flat) == 3
        assert rule_ids[0] in flat
        assert rule_ids[1] in flat
        assert rule_ids[2] in flat

    def test_flatten_deeply_nested(self, composer, rule_ids):
        """Flatten handles 3 levels of nesting."""
        level1 = composer.create_compound_rule(
            name="deep_l1",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        level2 = composer.create_compound_rule(
            name="deep_l2",
            operator="OR",
            rule_ids=[level1["compound_id"], rule_ids[2]],
        )
        level3 = composer.create_compound_rule(
            name="deep_l3",
            operator="NOT",
            rule_ids=[level2["compound_id"]],
        )
        flat = composer.flatten_compound_rule(level3["compound_id"])
        assert isinstance(flat, list)
        assert len(flat) == 3

    def test_flatten_nonexistent_raises(self, composer):
        """Flattening non-existent compound raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            composer.flatten_compound_rule("nonexistent-id")

    def test_flatten_deduplicates_shared_children(self, composer, rule_ids):
        """Flatten deduplicates rule IDs shared across branches."""
        # Two compounds sharing the same child rule
        branch_a = composer.create_compound_rule(
            name="branch_a",
            operator="AND",
            rule_ids=[rule_ids[0], rule_ids[1]],
        )
        branch_b = composer.create_compound_rule(
            name="branch_b",
            operator="AND",
            rule_ids=[rule_ids[0], rule_ids[2]],
        )
        root = composer.create_compound_rule(
            name="root_dedup",
            operator="OR",
            rule_ids=[branch_a["compound_id"], branch_b["compound_id"]],
        )
        flat = composer.flatten_compound_rule(root["compound_id"])
        # rule_ids[0] appears in both branches but should be deduplicated
        assert flat.count(rule_ids[0]) == 1
        assert len(flat) == 3  # rule_ids[0], [1], [2]


# ===========================================================================
# TestRuleSets
# ===========================================================================


class TestRuleSets:
    """Test rule set CRUD operations."""

    def test_create_rule_set(self, composer, rule_ids):
        """Create a new rule set."""
        rs = composer.create_rule_set(
            name="ghg-completeness",
            description="GHG Protocol completeness checks",
            rule_ids=rule_ids[:2],
            tags=["ghg", "completeness"],
        )
        assert rs["name"] == "ghg-completeness"
        assert rs["description"] == "GHG Protocol completeness checks"
        assert len(rs["rule_ids"]) == 2
        assert rs["version"] == "1.0.0"

    def test_create_rule_set_generates_prefixed_id(self, composer, rule_ids):
        """Created rule set has a RS-prefixed set_id."""
        rs = composer.create_rule_set(
            name="uuid-set",
            description="test",
            rule_ids=rule_ids[:2],
        )
        assert rs["set_id"].startswith("RS-")
        assert len(rs["set_id"]) > 3

    def test_create_rule_set_has_provenance(self, composer, rule_ids):
        """Created rule set has a non-empty provenance_hash."""
        rs = composer.create_rule_set(
            name="prov-set",
            description="test",
            rule_ids=rule_ids[:2],
        )
        assert len(rs["provenance_hash"]) == 64

    def test_create_rule_set_empty_name_rejected(self, composer, rule_ids):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="[Nn]ame.*empty|[Ee]mpty.*name|must not be empty"):
            composer.create_rule_set(
                name="", description="test", rule_ids=rule_ids[:2],
            )

    def test_create_rule_set_nonexistent_rule_rejected(self, composer):
        """Rule set with nonexistent rule_id raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            composer.create_rule_set(
                name="bad-rules",
                description="test",
                rule_ids=["nonexistent-id"],
            )

    def test_get_rule_set(self, composer, rule_ids):
        """Retrieve a rule set by ID."""
        rs = composer.create_rule_set(
            name="get-set", description="test", rule_ids=rule_ids[:2],
        )
        result = composer.get_rule_set(rs["set_id"])
        assert result is not None
        assert result["name"] == "get-set"

    def test_get_nonexistent_rule_set(self, composer):
        """Retrieving non-existent rule set returns None."""
        result = composer.get_rule_set("nonexistent-id")
        assert result is None

    def test_update_rule_set_description(self, composer, rule_ids):
        """Update rule set description."""
        rs = composer.create_rule_set(
            name="upd-set", description="original", rule_ids=rule_ids[:2],
        )
        result = composer.update_rule_set(
            rs["set_id"], description="Updated description.",
        )
        assert result["description"] == "Updated description."

    def test_update_rule_set_tags(self, composer, rule_ids):
        """Update rule set tags."""
        rs = composer.create_rule_set(
            name="tag-set", description="test", rule_ids=rule_ids[:2],
        )
        result = composer.update_rule_set(
            rs["set_id"], tags=["new-tag"],
        )
        assert "new-tag" in result["tags"]

    def test_delete_rule_set(self, composer, rule_ids):
        """Delete a rule set returns True."""
        rs = composer.create_rule_set(
            name="del-set", description="test", rule_ids=rule_ids[:2],
        )
        result = composer.delete_rule_set(rs["set_id"])
        assert result is True

    def test_delete_nonexistent_rule_set_returns_false(self, composer):
        """Deleting non-existent rule set returns False."""
        result = composer.delete_rule_set("nonexistent-id")
        assert result is False

    def test_add_rules_to_set(self, composer, rule_ids):
        """Add a rule to an existing rule set."""
        rs = composer.create_rule_set(
            name="add-set", description="test", rule_ids=rule_ids[:2],
        )
        result = composer.add_rules_to_set(rs["set_id"], [rule_ids[2]])
        assert rule_ids[2] in result["rule_ids"]
        assert len(result["rule_ids"]) == 3

    def test_add_duplicate_rule_to_set_is_noop(self, composer, rule_ids):
        """Adding a rule already in the set is a no-op."""
        rs = composer.create_rule_set(
            name="dup-add-set", description="test", rule_ids=rule_ids[:2],
        )
        result = composer.add_rules_to_set(rs["set_id"], [rule_ids[0]])
        assert result["rule_ids"].count(rule_ids[0]) == 1

    def test_remove_rules_from_set(self, composer, rule_ids):
        """Remove a rule from an existing rule set."""
        rs = composer.create_rule_set(
            name="rem-set", description="test", rule_ids=rule_ids[:3],
        )
        result = composer.remove_rules_from_set(rs["set_id"], [rule_ids[1]])
        assert rule_ids[1] not in result["rule_ids"]
        assert len(result["rule_ids"]) == 2

    def test_remove_nonmember_from_set_is_noop(self, composer, rule_ids):
        """Removing a rule not in the set is a no-op."""
        rs = composer.create_rule_set(
            name="noop-rem-set", description="test", rule_ids=rule_ids[:2],
        )
        result = composer.remove_rules_from_set(rs["set_id"], [rule_ids[4]])
        assert len(result["rule_ids"]) == 2

    def test_rule_set_preserves_all_fields(self, composer, rule_ids):
        """Retrieved rule set contains all expected fields."""
        rs = composer.create_rule_set(
            name="fields-set",
            rule_ids=rule_ids[:2],
            description="Test fields",
            tags=["test"],
        )
        result = composer.get_rule_set(rs["set_id"])
        expected_fields = {
            "set_id", "name", "description", "rule_ids", "tags",
            "version", "status", "created_at", "updated_at",
            "provenance_hash",
        }
        assert expected_fields.issubset(set(result.keys()))


# ===========================================================================
# TestRuleSetVersioning
# ===========================================================================


class TestRuleSetVersioning:
    """Test SemVer version bumping on rule set changes."""

    def test_initial_version_is_1_0_0(self, composer, rule_ids):
        """New rule set starts at version 1.0.0."""
        rs = composer.create_rule_set(
            name="ver-init", description="test", rule_ids=rule_ids[:2],
        )
        assert rs["version"] == "1.0.0"

    def test_add_rule_minor_bump(self, composer, rule_ids):
        """Adding a rule triggers a minor version bump."""
        rs = composer.create_rule_set(
            name="ver-add", description="test", rule_ids=rule_ids[:2],
        )
        result = composer.add_rules_to_set(rs["set_id"], [rule_ids[2]])
        assert result["version"] == "1.1.0"

    def test_remove_rule_minor_bump(self, composer, rule_ids):
        """Removing a rule triggers a minor version bump."""
        rs = composer.create_rule_set(
            name="ver-rem", description="test", rule_ids=rule_ids[:3],
        )
        result = composer.remove_rules_from_set(rs["set_id"], [rule_ids[0]])
        assert result["version"] == "1.1.0"

    def test_update_description_patch_bump(self, composer, rule_ids):
        """Updating description triggers a patch version bump."""
        rs = composer.create_rule_set(
            name="ver-desc", description="original", rule_ids=rule_ids[:2],
        )
        result = composer.update_rule_set(
            rs["set_id"], description="New desc.",
        )
        assert result["version"] == "1.0.1"

    def test_update_tags_patch_bump(self, composer, rule_ids):
        """Updating tags triggers a patch version bump."""
        rs = composer.create_rule_set(
            name="ver-tags", description="test", rule_ids=rule_ids[:2],
        )
        result = composer.update_rule_set(
            rs["set_id"], tags=["new-tag"],
        )
        assert result["version"] == "1.0.1"

    def test_multiple_version_bumps(self, composer, rule_ids):
        """Multiple operations produce correct SemVer sequence."""
        rs = composer.create_rule_set(
            name="ver-multi", description="test", rule_ids=rule_ids[:2],
        )
        # Patch bump: 1.0.0 -> 1.0.1
        composer.update_rule_set(rs["set_id"], description="v1")
        # Minor bump: 1.0.1 -> 1.1.0
        composer.add_rules_to_set(rs["set_id"], [rule_ids[2]])
        # Minor bump: 1.1.0 -> 1.2.0
        result = composer.remove_rules_from_set(rs["set_id"], [rule_ids[0]])
        assert result["version"] == "1.2.0"

    def test_version_history_tracked(self, composer, rule_ids):
        """Rule set version history is accessible."""
        rs = composer.create_rule_set(
            name="ver-hist", description="test", rule_ids=rule_ids[:2],
        )
        composer.update_rule_set(rs["set_id"], description="update 1")
        composer.add_rules_to_set(rs["set_id"], [rule_ids[2]])
        versions = composer.get_rule_set_versions(rs["set_id"])
        assert len(versions) == 3
        version_numbers = [v["version"] for v in versions]
        assert "1.0.0" in version_numbers

    def test_version_history_nonexistent_returns_empty(self, composer):
        """Requesting versions for non-existent set returns empty list."""
        versions = composer.get_rule_set_versions("nonexistent-id")
        assert versions == []


# ===========================================================================
# TestTemplates
# ===========================================================================


class TestTemplates:
    """Test rule set templates: creation, instantiation, overrides."""

    def test_create_template(self, composer):
        """Create a rule set template."""
        tmpl = composer.create_template(
            name="ghg-base-template",
            rule_definitions=[
                {"rule_type": "COMPLETENESS", "parameters": {"column": "co2"}},
                {"rule_type": "RANGE", "parameters": {"min": 0, "max": 100}},
            ],
            description="Base GHG Protocol rule set template.",
        )
        assert tmpl["name"] == "ghg-base-template"
        assert tmpl["template_id"] is not None
        assert tmpl["template_id"].startswith("TPL-")

    def test_create_template_generates_prefixed_id(self, composer):
        """Template has a TPL-prefixed template_id."""
        tmpl = composer.create_template(
            name="tmpl-uuid",
            rule_definitions=[
                {"rule_type": "FORMAT", "parameters": {}},
            ],
        )
        assert tmpl["template_id"].startswith("TPL-")
        assert len(tmpl["template_id"]) > 4

    def test_create_template_empty_name_rejected(self, composer):
        """Empty template name raises ValueError."""
        with pytest.raises(ValueError, match="[Nn]ame.*empty|must not be empty"):
            composer.create_template(
                name="",
                rule_definitions=[{"rule_type": "COMPLETENESS", "parameters": {}}],
            )

    def test_create_template_empty_definitions_rejected(self, composer):
        """Empty rule_definitions raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            composer.create_template(
                name="empty-defs",
                rule_definitions=[],
            )

    def test_instantiate_template(self, composer):
        """Instantiate a template into a new rule set."""
        tmpl = composer.create_template(
            name="inst-tmpl",
            rule_definitions=[
                {"rule_type": "COMPLETENESS", "parameters": {"col": "a"}},
                {"rule_type": "RANGE", "parameters": {"min": 0}},
                {"rule_type": "FORMAT", "parameters": {"pattern": ".*"}},
            ],
        )
        rs = composer.instantiate_template(tmpl["template_id"])
        assert rs["set_id"] is not None
        assert rs["version"] == "1.0.0"
        assert rs["template_id"] == tmpl["template_id"]
        assert rs["rule_count"] == 3

    def test_instantiate_template_with_name_override(self, composer):
        """Instantiate template with custom name via overrides."""
        tmpl = composer.create_template(
            name="name-tmpl",
            rule_definitions=[
                {"rule_type": "COMPLETENESS", "parameters": {}},
            ],
        )
        rs = composer.instantiate_template(
            tmpl["template_id"],
            overrides={"name": "custom-instance-name"},
        )
        assert rs["name"] == "custom-instance-name"

    def test_instantiate_template_with_description_override(self, composer):
        """Instantiate template with custom description via overrides."""
        tmpl = composer.create_template(
            name="desc-tmpl",
            rule_definitions=[
                {"rule_type": "FORMAT", "parameters": {}},
            ],
        )
        rs = composer.instantiate_template(
            tmpl["template_id"],
            overrides={"description": "Custom description for instance."},
        )
        assert rs["description"] == "Custom description for instance."

    def test_instantiate_template_with_rule_overrides(self, composer):
        """Instantiate template with rule-level parameter overrides."""
        tmpl = composer.create_template(
            name="override-tmpl",
            rule_definitions=[
                {"rule_type": "RANGE", "parameters": {"min": 0, "max": 100}},
                {"rule_type": "FORMAT", "parameters": {"pattern": ".*"}},
            ],
        )
        rs = composer.instantiate_template(
            tmpl["template_id"],
            overrides={
                "rule_overrides": {"0": {"parameters": {"min": 10, "max": 200}}},
            },
        )
        # The first rule definition should have been updated
        assert rs["rule_definitions"][0]["parameters"]["min"] == 10
        assert rs["rule_definitions"][0]["parameters"]["max"] == 200

    def test_instantiate_nonexistent_template_raises(self, composer):
        """Instantiating non-existent template raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            composer.instantiate_template("nonexistent-id")

    def test_instantiate_increments_count(self, composer):
        """Instantiating a template increments its instantiation_count."""
        tmpl = composer.create_template(
            name="count-tmpl",
            rule_definitions=[{"rule_type": "COMPLETENESS", "parameters": {}}],
        )
        assert tmpl["instantiation_count"] == 0
        composer.instantiate_template(tmpl["template_id"])
        composer.instantiate_template(
            tmpl["template_id"],
            overrides={"name": "second-inst"},
        )
        # Check the template in the internal store (via statistics)
        stats = composer.get_statistics()
        assert stats["templates_count"] == 1
        assert stats["rule_sets_from_templates"] == 2


# ===========================================================================
# TestInheritance
# ===========================================================================


class TestInheritance:
    """Test rule set inheritance: child extends parent, overrides rules."""

    def test_create_child_rule_set(self, composer, rule_ids):
        """Create a child rule set that extends a parent."""
        parent = composer.create_rule_set(
            name="parent-set",
            description="parent",
            rule_ids=rule_ids[:3],
            tags=["base"],
        )
        child = composer.create_child_rule_set(
            parent_set_id=parent["set_id"],
            name="child-set",
        )
        assert child["parent_set_id"] == parent["set_id"]
        assert child["name"] == "child-set"

    def test_child_inherits_parent_rules(self, composer, rule_ids):
        """Child rule set inherits all parent rules."""
        parent = composer.create_rule_set(
            name="inherit-parent",
            description="parent",
            rule_ids=rule_ids[:3],
        )
        child = composer.create_child_rule_set(
            parent_set_id=parent["set_id"],
            name="inherit-child",
        )
        # Child should have parent's rules
        assert len(child["rule_ids"]) >= 3

    def test_child_can_add_rules(self, composer, rule_ids):
        """Child rule set can add additional rules."""
        parent = composer.create_rule_set(
            name="add-parent",
            description="parent",
            rule_ids=rule_ids[:2],
        )
        child = composer.create_child_rule_set(
            parent_set_id=parent["set_id"],
            name="add-child",
            additional_rules=[rule_ids[3], rule_ids[4]],
        )
        assert rule_ids[3] in child["rule_ids"]
        assert rule_ids[4] in child["rule_ids"]

    def test_child_can_override_rules(self, composer, rule_ids):
        """Child can override (replace) specific parent rules."""
        parent = composer.create_rule_set(
            name="override-parent",
            description="parent",
            rule_ids=rule_ids[:3],
        )
        # Override rule_ids[0] with rule_ids[4]
        child = composer.create_child_rule_set(
            parent_set_id=parent["set_id"],
            name="override-child",
            override_rules={rule_ids[0]: rule_ids[4]},
        )
        assert rule_ids[0] not in child["rule_ids"]
        assert rule_ids[4] in child["rule_ids"]

    def test_child_nonexistent_parent_raises(self, composer):
        """Creating child with non-existent parent raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            composer.create_child_rule_set(
                parent_set_id="nonexistent-id",
                name="orphan-child",
            )

    def test_get_inheritance_chain(self, composer, rule_ids):
        """Get inheritance chain returns child -> parent lineage."""
        parent = composer.create_rule_set(
            name="chain-parent",
            description="parent",
            rule_ids=rule_ids[:2],
        )
        child = composer.create_child_rule_set(
            parent_set_id=parent["set_id"],
            name="chain-child",
        )
        chain = composer.get_inheritance_chain(child["set_id"])
        assert len(chain) == 2
        # Chain goes from child up to parent
        assert chain[0]["set_id"] == child["set_id"]
        assert chain[1]["set_id"] == parent["set_id"]

    def test_get_inheritance_chain_root(self, composer, rule_ids):
        """Root rule set (no parent) has chain length 1."""
        root = composer.create_rule_set(
            name="chain-root",
            description="test",
            rule_ids=rule_ids[:2],
        )
        chain = composer.get_inheritance_chain(root["set_id"])
        assert len(chain) == 1

    def test_multi_level_inheritance(self, composer, rule_ids):
        """Three-level inheritance chain is supported."""
        grandparent = composer.create_rule_set(
            name="gp-set",
            description="gp",
            rule_ids=rule_ids[:1],
        )
        parent = composer.create_child_rule_set(
            parent_set_id=grandparent["set_id"],
            name="p-set",
            additional_rules=[rule_ids[1]],
        )
        child = composer.create_child_rule_set(
            parent_set_id=parent["set_id"],
            name="c-set",
            additional_rules=[rule_ids[2]],
        )
        chain = composer.get_inheritance_chain(child["set_id"])
        assert len(chain) == 3

    def test_child_has_provenance(self, composer, rule_ids):
        """Child rule set has its own provenance_hash."""
        parent = composer.create_rule_set(
            name="prov-parent",
            description="parent",
            rule_ids=rule_ids[:2],
        )
        child = composer.create_child_rule_set(
            parent_set_id=parent["set_id"],
            name="prov-child",
        )
        assert len(child["provenance_hash"]) == 64

    def test_inheritance_chain_nonexistent_returns_empty(self, composer):
        """Getting chain for non-existent set returns empty list."""
        chain = composer.get_inheritance_chain("nonexistent-id")
        assert chain == []


# ===========================================================================
# TestDependencies
# ===========================================================================


class TestDependencies:
    """Test rule dependencies, evaluation order, and cycle detection."""

    def test_add_dependency(self, composer, rule_ids):
        """Add a dependency between two rules."""
        result = composer.add_rule_dependency(
            rule_ids[1], depends_on_rule_id=rule_ids[0],
        )
        assert result["rule_id"] == rule_ids[1]
        assert result["depends_on"] == rule_ids[0]

    def test_evaluation_order_simple(self, composer, rule_ids):
        """Evaluation order respects dependency (topological sort)."""
        composer.add_rule_dependency(
            rule_ids[1], depends_on_rule_id=rule_ids[0],
        )
        order = composer.get_evaluation_order([rule_ids[0], rule_ids[1]])
        # rule_ids[0] must come before rule_ids[1]
        idx_a = order.index(rule_ids[0])
        idx_b = order.index(rule_ids[1])
        assert idx_a < idx_b

    def test_evaluation_order_three_rules(self, composer, rule_ids):
        """Topological sort with 3 dependent rules."""
        composer.add_rule_dependency(
            rule_ids[1], depends_on_rule_id=rule_ids[0],
        )
        composer.add_rule_dependency(
            rule_ids[2], depends_on_rule_id=rule_ids[1],
        )
        order = composer.get_evaluation_order(
            [rule_ids[2], rule_ids[0], rule_ids[1]]
        )
        idx_a = order.index(rule_ids[0])
        idx_b = order.index(rule_ids[1])
        idx_c = order.index(rule_ids[2])
        assert idx_a < idx_b < idx_c

    def test_cycle_detection_direct(self, composer, rule_ids):
        """Direct cycle (A depends on B, B depends on A) is detected."""
        composer.add_rule_dependency(
            rule_ids[1], depends_on_rule_id=rule_ids[0],
        )
        with pytest.raises(ValueError, match="[Cc]ycle"):
            composer.add_rule_dependency(
                rule_ids[0], depends_on_rule_id=rule_ids[1],
            )

    def test_cycle_detection_transitive(self, composer, rule_ids):
        """Transitive cycle (A->B->C->A) is detected."""
        composer.add_rule_dependency(
            rule_ids[1], depends_on_rule_id=rule_ids[0],
        )
        composer.add_rule_dependency(
            rule_ids[2], depends_on_rule_id=rule_ids[1],
        )
        with pytest.raises(ValueError, match="[Cc]ycle"):
            composer.add_rule_dependency(
                rule_ids[0], depends_on_rule_id=rule_ids[2],
            )

    def test_self_dependency_rejected(self, composer, rule_ids):
        """Self-dependency is rejected."""
        with pytest.raises(ValueError, match="cannot depend on itself"):
            composer.add_rule_dependency(
                rule_ids[0], depends_on_rule_id=rule_ids[0],
            )

    def test_add_dependency_nonexistent_rule_raises(self, composer, rule_ids):
        """Adding dependency on non-existent rule raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            composer.add_rule_dependency(
                rule_ids[0], depends_on_rule_id="nonexistent-id",
            )

    def test_evaluation_order_independent_rules(self, composer, rule_ids):
        """Independent rules can be in any order but all present."""
        order = composer.get_evaluation_order([rule_ids[0], rule_ids[1]])
        assert len(order) == 2
        assert set(order) == {rule_ids[0], rule_ids[1]}

    def test_duplicate_dependency_is_noop(self, composer, rule_ids):
        """Adding the same dependency twice is a no-op."""
        composer.add_rule_dependency(
            rule_ids[1], depends_on_rule_id=rule_ids[0],
        )
        result = composer.add_rule_dependency(
            rule_ids[1], depends_on_rule_id=rule_ids[0],
        )
        # Should return a result with already_exists status
        assert result["status"] == "already_exists"

    def test_evaluation_order_diamond_dependency(self, composer, rule_ids):
        """Diamond dependency (A->B, A->C, B->D, C->D) is resolved correctly."""
        # B depends on A, C depends on A, D depends on B and C
        composer.add_rule_dependency(
            rule_ids[1], depends_on_rule_id=rule_ids[0],
        )
        composer.add_rule_dependency(
            rule_ids[2], depends_on_rule_id=rule_ids[0],
        )
        composer.add_rule_dependency(
            rule_ids[3], depends_on_rule_id=rule_ids[1],
        )
        composer.add_rule_dependency(
            rule_ids[3], depends_on_rule_id=rule_ids[2],
        )
        order = composer.get_evaluation_order([
            rule_ids[3], rule_ids[2],
            rule_ids[1], rule_ids[0],
        ])
        idx_a = order.index(rule_ids[0])
        idx_b = order.index(rule_ids[1])
        idx_c = order.index(rule_ids[2])
        idx_d = order.index(rule_ids[3])
        assert idx_a < idx_b
        assert idx_a < idx_c
        assert idx_b < idx_d
        assert idx_c < idx_d


# ===========================================================================
# TestComparison
# ===========================================================================


class TestComparison:
    """Test rule set comparison (diff between two rule sets)."""

    def test_compare_detects_added_rules(self, composer, rule_ids):
        """Compare detects rules in set B that are not in set A."""
        rs_a = composer.create_rule_set(
            name="cmp-a", description="a", rule_ids=rule_ids[:2],
        )
        rs_b = composer.create_rule_set(
            name="cmp-b", description="b", rule_ids=rule_ids[:3],
        )
        diff = composer.compare_rule_sets(rs_a["set_id"], rs_b["set_id"])
        assert len(diff["added_rules"]) == 1
        assert rule_ids[2] in diff["added_rules"]

    def test_compare_detects_removed_rules(self, composer, rule_ids):
        """Compare detects rules in set A that are not in set B."""
        rs_a = composer.create_rule_set(
            name="cmp-rem-a", description="a", rule_ids=rule_ids[:3],
        )
        rs_b = composer.create_rule_set(
            name="cmp-rem-b", description="b", rule_ids=rule_ids[1:3],
        )
        diff = composer.compare_rule_sets(rs_a["set_id"], rs_b["set_id"])
        assert len(diff["removed_rules"]) == 1
        assert rule_ids[0] in diff["removed_rules"]

    def test_compare_identical_sets(self, composer, rule_ids):
        """Compare identical rule sets shows no rule changes."""
        rs_a = composer.create_rule_set(
            name="cmp-same-a", description="a", rule_ids=rule_ids[:2],
        )
        rs_b = composer.create_rule_set(
            name="cmp-same-b", description="a", rule_ids=rule_ids[:2],
        )
        diff = composer.compare_rule_sets(rs_a["set_id"], rs_b["set_id"])
        assert len(diff["added_rules"]) == 0
        assert len(diff["removed_rules"]) == 0

    def test_compare_nonexistent_set_raises(self, composer, rule_ids):
        """Compare with non-existent set raises ValueError."""
        rs_a = composer.create_rule_set(
            name="cmp-exist", description="a", rule_ids=rule_ids[:2],
        )
        with pytest.raises(ValueError, match="does not exist"):
            composer.compare_rule_sets(rs_a["set_id"], "nonexistent-id")

    def test_compare_both_nonexistent_raises(self, composer):
        """Compare with both non-existent sets raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            composer.compare_rule_sets("nonexistent-a", "nonexistent-b")

    def test_compare_includes_version_metadata(self, composer, rule_ids):
        """Compare result includes version information."""
        rs_a = composer.create_rule_set(
            name="cmp-meta-a", description="a", rule_ids=rule_ids[:2],
        )
        rs_b = composer.create_rule_set(
            name="cmp-meta-b", description="b", rule_ids=rule_ids[:3],
        )
        diff = composer.compare_rule_sets(rs_a["set_id"], rs_b["set_id"])
        assert "a_version" in diff
        assert "b_version" in diff
        assert diff["a_version"] == "1.0.0"
        assert diff["b_version"] == "1.0.0"


# ===========================================================================
# TestStatisticsAndClear
# ===========================================================================


class TestStatisticsAndClear:
    """Test statistics aggregation and clear/reset."""

    def test_statistics_compound_count(self, composer, rule_ids):
        """Statistics report compound rule count."""
        composer.create_compound_rule(
            name="stat-c1", operator="AND", rule_ids=rule_ids[:2],
        )
        composer.create_compound_rule(
            name="stat-c2", operator="OR", rule_ids=rule_ids[2:4],
        )
        stats = composer.get_statistics()
        assert stats["compound_rules_count"] == 2

    def test_statistics_rule_set_count(self, composer, rule_ids):
        """Statistics report rule set count."""
        composer.create_rule_set(
            name="stat-rs1", description="a", rule_ids=rule_ids[:2],
        )
        composer.create_rule_set(
            name="stat-rs2", description="b", rule_ids=rule_ids[2:4],
        )
        stats = composer.get_statistics()
        assert stats["rule_sets_count"] == 2

    def test_statistics_template_count(self, composer):
        """Statistics report template count."""
        composer.create_template(
            name="stat-tmpl",
            rule_definitions=[{"rule_type": "COMPLETENESS", "parameters": {}}],
        )
        stats = composer.get_statistics()
        assert stats["templates_count"] == 1

    def test_clear_resets_all(self, composer, rule_ids):
        """Clear removes all compounds, rule sets, and templates."""
        composer.create_compound_rule(
            name="clear-c", operator="AND", rule_ids=rule_ids[:2],
        )
        composer.create_rule_set(
            name="clear-rs", description="test", rule_ids=rule_ids[:2],
        )
        composer.create_template(
            name="clear-tmpl",
            rule_definitions=[{"rule_type": "COMPLETENESS", "parameters": {}}],
        )
        composer.clear()
        stats = composer.get_statistics()
        assert stats["compound_rules_count"] == 0
        assert stats["rule_sets_count"] == 0
        assert stats["templates_count"] == 0

    def test_clear_allows_reuse_of_names(self, composer, rule_ids):
        """After clear, previously used compound names can be recreated."""
        composer.create_compound_rule(
            name="reuse-name", operator="AND", rule_ids=rule_ids[:2],
        )
        composer.clear()
        compound = composer.create_compound_rule(
            name="reuse-name", operator="OR", rule_ids=rule_ids[:2],
        )
        assert compound["operator"] == "OR"


# ===========================================================================
# TestMaxCompoundDepth
# ===========================================================================


class TestMaxCompoundDepth:
    """Test maximum compound rule nesting depth enforcement."""

    def test_depth_within_limit(self, composer, rule_ids):
        """Compound nesting within MAX_COMPOUND_DEPTH is allowed."""
        # Build a chain of 3 nested compounds (should be well within limit)
        level1 = composer.create_compound_rule(
            name="depth_l1",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        level2 = composer.create_compound_rule(
            name="depth_l2",
            operator="OR",
            rule_ids=[level1["compound_id"], rule_ids[2]],
        )
        level3 = composer.create_compound_rule(
            name="depth_l3",
            operator="NOT",
            rule_ids=[level2["compound_id"]],
        )
        assert level3["compound_id"] is not None

    def test_depth_exceeds_limit_raises(self, composer, rule_ids):
        """Nesting exceeding MAX_COMPOUND_DEPTH raises ValueError."""
        # Build a chain deeper than MAX_COMPOUND_DEPTH
        # First create a base compound with 2 rules
        base = composer.create_compound_rule(
            name="exceed_base",
            operator="AND",
            rule_ids=rule_ids[:2],
        )
        current_id = base["compound_id"]

        # Build nesting chain up to MAX_COMPOUND_DEPTH
        for i in range(MAX_COMPOUND_DEPTH):
            try:
                new_compound = composer.create_compound_rule(
                    name=f"exceed_level_{i}",
                    operator="NOT",
                    rule_ids=[current_id],
                )
                current_id = new_compound["compound_id"]
            except ValueError as e:
                # Depth exceeded -- this is expected
                assert "depth" in str(e).lower() or "nesting" in str(e).lower()
                return

        # If we got here without error, try one more level
        with pytest.raises(ValueError, match="[Dd]epth|[Nn]esting"):
            composer.create_compound_rule(
                name="exceed_final",
                operator="NOT",
                rule_ids=[current_id],
            )

    def test_max_compound_depth_constant(self, composer):
        """MAX_COMPOUND_DEPTH constant is a positive integer."""
        assert MAX_COMPOUND_DEPTH > 0
        assert isinstance(MAX_COMPOUND_DEPTH, int)

    def test_valid_compound_operators(self):
        """VALID_COMPOUND_OPERATORS contains AND, OR, NOT."""
        assert "AND" in VALID_COMPOUND_OPERATORS
        assert "OR" in VALID_COMPOUND_OPERATORS
        assert "NOT" in VALID_COMPOUND_OPERATORS
        assert len(VALID_COMPOUND_OPERATORS) == 3
