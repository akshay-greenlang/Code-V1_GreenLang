# -*- coding: utf-8 -*-
"""PolicyGraph smoke tests (PLATFORM 1, task #25)."""

from __future__ import annotations


def test_policy_engine_importable():
    from greenlang.policy_graph import PolicyEngine  # noqa: F401


def test_rule_set_importable():
    from greenlang.policy_graph.rules import RuleSet  # noqa: F401


def test_compliance_registry_importable():
    from greenlang.policy_graph.compliance import ComplianceRegistry  # noqa: F401
