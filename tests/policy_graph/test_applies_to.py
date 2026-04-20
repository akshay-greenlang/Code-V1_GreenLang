# -*- coding: utf-8 -*-
"""Tests for Phase 2.4 Policy Graph `applies_to()` API."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from greenlang.policy_graph import (
    ApplicabilityResult,
    DEFAULT_RULES,
    PolicyGraph,
    RegulationApplicability,
)


# --------------------------------------------------------------------------
# Return types
# --------------------------------------------------------------------------


class TestReturnShape:
    def test_result_shape(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={},
            activity={"category": "purchased_electricity"},
            jurisdiction="GLOBAL",
            date="2026-06-01",
        )
        assert isinstance(res, ApplicabilityResult)
        # GHG Protocol always applies.
        names = {r.name for r in res.applicable_regulations}
        assert "GHG-Protocol" in names

    def test_evaluation_context_echoed(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={"hq_country": "IN"},
            activity={"category": "x"},
            jurisdiction="EU",
            date="2026-06-01",
        )
        assert res.evaluation_context["jurisdiction"] == "EU"
        assert res.evaluation_context["date"] == "2026-06-01"


# --------------------------------------------------------------------------
# CBAM
# --------------------------------------------------------------------------


class TestCBAM:
    def test_cbam_steel_to_eu_triggers(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={"hq_country": "IN", "operates_in": ["EU"]},
            activity={"category": "cbam_covered_goods", "goods": "steel"},
            jurisdiction="EU",
            date="2026-06-15",
        )
        names = {r.name for r in res.applicable_regulations}
        assert "CBAM" in names
        cbam = next(r for r in res.applicable_regulations if r.name == "CBAM")
        assert cbam.deadline == "2026-07-31"  # Q2 end-of-following-month deadline
        assert "Certified" in cbam.required_factor_classes

    def test_cbam_does_not_apply_before_2026(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={"hq_country": "IN", "operates_in": ["EU"]},
            activity={"category": "cbam_covered_goods", "goods": "steel"},
            jurisdiction="EU",
            date="2025-06-15",
        )
        names = {r.name for r in res.applicable_regulations}
        assert "CBAM" not in names

    def test_cbam_skips_non_eu_jurisdiction(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={"hq_country": "IN"},
            activity={"category": "cbam_covered_goods", "goods": "steel"},
            jurisdiction="US",
            date="2026-06-15",
        )
        names = {r.name for r in res.applicable_regulations}
        assert "CBAM" not in names


# --------------------------------------------------------------------------
# CSRD
# --------------------------------------------------------------------------


class TestCSRD:
    def test_csrd_large_eu_company(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={"hq_country": "DE", "employees": 1000, "turnover_m_eur": 200},
            activity={"category": "scope1_fuel_combustion"},
            jurisdiction="EU",
            date="2026-06-01",
        )
        names = {r.name for r in res.applicable_regulations}
        assert "CSRD" in names

    def test_csrd_skips_small_company(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={"hq_country": "DE", "employees": 10, "turnover_m_eur": 2},
            activity={"category": "scope1_fuel_combustion"},
            jurisdiction="EU",
            date="2026-06-01",
        )
        names = {r.name for r in res.applicable_regulations}
        assert "CSRD" not in names

    def test_csrd_indian_subsidiary_of_eu_parent(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={"hq_country": "IN", "operates_in": ["EU"], "employees": 500},
            activity={"category": "scope2_electricity"},
            jurisdiction="EU",
            date="2026-06-01",
        )
        names = {r.name for r in res.applicable_regulations}
        assert "CSRD" in names


# --------------------------------------------------------------------------
# SB-253
# --------------------------------------------------------------------------


class TestSB253:
    def test_sb253_applies_to_large_ca_operator(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={
                "hq_country": "US",
                "operates_in": ["US-CA"],
                "revenue_usd": 2_000_000_000,
            },
            activity={"category": "scope1_fuel_combustion"},
            jurisdiction="US-CA",
            date="2026-06-01",
        )
        names = {r.name for r in res.applicable_regulations}
        assert "SB-253" in names
        sb = next(r for r in res.applicable_regulations if r.name == "SB-253")
        assert sb.deadline == "2026-08-10"

    def test_sb253_skips_small_operator(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={
                "hq_country": "US",
                "operates_in": ["US-CA"],
                "revenue_usd": 500_000_000,
            },
            activity={"category": "scope1_fuel_combustion"},
            jurisdiction="US-CA",
            date="2026-06-01",
        )
        names = {r.name for r in res.applicable_regulations}
        assert "SB-253" not in names

    def test_sb253_scope3_deadline_after_first_period(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={
                "hq_country": "US",
                "operates_in": ["US-CA"],
                "revenue_usd": 2_000_000_000,
            },
            activity={"category": "scope3_purchased_goods"},
            jurisdiction="US-CA",
            date="2026-12-15",  # After 2026-08-10
        )
        sb = next(
            r for r in res.applicable_regulations if r.name == "SB-253"
        )
        assert sb.deadline == "2027-08-10"


# --------------------------------------------------------------------------
# GHG Protocol (always applicable) + TCFD
# --------------------------------------------------------------------------


class TestAlwaysApplicable:
    def test_ghg_protocol_always(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={}, activity={}, jurisdiction="GLOBAL", date="2026-01-01"
        )
        names = {r.name for r in res.applicable_regulations}
        assert "GHG-Protocol" in names

    def test_tcfd_uk_entity(self):
        pg = PolicyGraph()
        res = pg.applies_to(
            entity={"hq_country": "GB"},
            activity={},
            jurisdiction="UK",
            date="2026-06-01",
        )
        names = {r.name for r in res.applicable_regulations}
        assert "TCFD" in names


# --------------------------------------------------------------------------
# Date coercion
# --------------------------------------------------------------------------


class TestDateCoercion:
    def test_date_object_accepted(self):
        pg = PolicyGraph()
        pg.applies_to(
            entity={}, activity={}, jurisdiction="GLOBAL", date=date(2026, 6, 1)
        )

    def test_datetime_object_accepted(self):
        pg = PolicyGraph()
        pg.applies_to(
            entity={}, activity={}, jurisdiction="GLOBAL",
            date=datetime(2026, 6, 1, 12, 34),
        )


# --------------------------------------------------------------------------
# YAML rule loading
# --------------------------------------------------------------------------


class TestYAMLRuleLoading:
    def test_register_rule_file(self, tmp_path: Path):
        yaml_text = """
rules:
  - name: MY-REG
    full_name: My Custom Regulation
    jurisdiction: [EU]
    deadline: "2026-12-31"
    required_factor_classes: [Certified]
    rationale: Custom rule matched on entity.type and activity.category.
    when:
      entity:
        type: corporation
      activity:
        category: special_activity
"""
        p = tmp_path / "rules.yaml"
        p.write_text(yaml_text, encoding="utf-8")

        pg = PolicyGraph()
        count = pg.register_rule_file(p)
        assert count == 1

        # Matches
        res = pg.applies_to(
            entity={"type": "corporation"},
            activity={"category": "special_activity"},
            jurisdiction="EU",
            date="2026-06-01",
        )
        assert any(r.name == "MY-REG" for r in res.applicable_regulations)

        # Does not match on different jurisdiction
        res2 = pg.applies_to(
            entity={"type": "corporation"},
            activity={"category": "special_activity"},
            jurisdiction="US",
            date="2026-06-01",
        )
        assert not any(r.name == "MY-REG" for r in res2.applicable_regulations)


# --------------------------------------------------------------------------
# Rule introspection
# --------------------------------------------------------------------------


class TestIntrospection:
    def test_list_rules(self):
        pg = PolicyGraph()
        rules = pg.list_rules()
        assert any("cbam" in r.lower() for r in rules)
        assert any("csrd" in r.lower() for r in rules)
        assert any("sb253" in r.lower() for r in rules)

    def test_default_rules_constant(self):
        assert len(DEFAULT_RULES) >= 5
