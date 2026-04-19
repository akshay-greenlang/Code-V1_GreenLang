# -*- coding: utf-8 -*-
"""
GreenLang Policy Graph - v3 Intelligence Layer
===============================================

The Policy Graph is GreenLang's L3 "Intelligence" layer providing a unified API
for policy enforcement, compliance checking, and regulatory rule evaluation.

This module is a thin product facade that wraps the existing governance/policy
infrastructure under clean v3 product names. It does not duplicate logic; all
functionality delegates to :mod:`greenlang.governance.policy` and
:mod:`greenlang.governance.compliance`.

Quick Start::

    from greenlang.policy_graph import PolicyEngine, ComplianceRegistry, RuleSet

    # Policy enforcement (OPA-backed)
    engine = PolicyEngine()
    result = engine.check_install(manifest, path="/packs/my-pack")
    assert result["allowed"] is True

    # Compliance engine discovery
    registry = ComplianceRegistry()
    for eng in registry.list_engines():
        print(eng["name"], eng["jurisdiction"])

    # Custom rule evaluation
    rules = RuleSet("my-rules", jurisdiction="EU")
    rules.add_rule("r1", "Emissions below threshold", lambda ctx: ctx["co2"] < 100)
    results = rules.evaluate({"co2": 42})

Submodules:
    enforcer: :class:`PolicyEngine` -- OPA-backed policy enforcement
    compliance: :class:`ComplianceRegistry` plus re-exports from governance.compliance
    rules: :class:`RuleSet` -- lightweight rule collection and evaluation
"""

from __future__ import annotations

from greenlang.policy_graph.enforcer import PolicyEngine
from greenlang.policy_graph.compliance import (
    ComplianceRegistry,
    IEDComplianceManager,
    ComplianceStatus,
    BATAEL,
    ComplianceAssessment,
)
from greenlang.policy_graph.rules import RuleSet

__version__ = "0.1.0"

__all__ = [
    # Core facade classes
    "PolicyEngine",
    "ComplianceRegistry",
    "RuleSet",
    # Re-exported compliance models
    "IEDComplianceManager",
    "ComplianceStatus",
    "BATAEL",
    "ComplianceAssessment",
]
