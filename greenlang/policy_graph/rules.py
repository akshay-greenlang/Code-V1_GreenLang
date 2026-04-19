# -*- coding: utf-8 -*-
"""
RuleSet - v3 Policy Graph lightweight rule evaluation
=====================================================

Provides a simple, in-process rule engine for cases where a full OPA
evaluation is unnecessary. Rules are pure Python callables registered
against a :class:`RuleSet` and evaluated deterministically against a
context dict.

Usage::

    from greenlang.policy_graph import RuleSet

    rules = RuleSet("emissions-gate", jurisdiction="EU")
    rules.add_rule(
        rule_id="max-co2",
        description="CO2 must be below 500 tCO2e",
        check_fn=lambda ctx: ctx.get("co2_tonnes", 0) < 500,
        severity="error",
    )
    rules.add_rule(
        rule_id="data-quality",
        description="Data quality score >= 0.8",
        check_fn=lambda ctx: ctx.get("dq_score", 0) >= 0.8,
        severity="warning",
    )

    results = rules.evaluate({"co2_tonnes": 120, "dq_score": 0.92})
    for r in results:
        print(r["rule_id"], "PASS" if r["passed"] else "FAIL")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

_VALID_SEVERITIES = frozenset({"error", "warning", "info"})


class RuleSet:
    """
    A named collection of policy rules for the v3 Policy Graph.

    Each rule is a pure-Python callable that receives a context dict and
    returns ``True`` (pass) or ``False`` (fail).  Rules are evaluated
    deterministically -- no LLM involvement.

    Args:
        name: Human-readable name for this rule set.
        jurisdiction: Jurisdiction scope (e.g. ``"EU"``, ``"US"``,
            ``"global"``).

    Example::

        rs = RuleSet("my-checks")
        rs.add_rule("r1", "Value positive", lambda ctx: ctx["val"] > 0)
        results = rs.evaluate({"val": 42})
        assert results[0]["passed"] is True
    """

    def __init__(self, name: str, jurisdiction: str = "global") -> None:
        self.name = name
        self.jurisdiction = jurisdiction
        self._rules: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_rule(
        self,
        rule_id: str,
        description: str,
        check_fn: Callable[[Dict[str, Any]], bool],
        severity: str = "error",
    ) -> None:
        """
        Register a rule in this set.

        Args:
            rule_id: Unique identifier for the rule within this set.
            description: Human-readable description of what the rule checks.
            check_fn: A callable ``(context: dict) -> bool``.  Must return
                ``True`` when the rule passes.
            severity: One of ``"error"``, ``"warning"``, or ``"info"``.

        Raises:
            ValueError: If *severity* is not one of the accepted values,
                or if *rule_id* is already registered.
        """
        if severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity {severity!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_SEVERITIES))}"
            )

        if any(r["rule_id"] == rule_id for r in self._rules):
            raise ValueError(
                f"Rule {rule_id!r} is already registered in "
                f"RuleSet {self.name!r}"
            )

        self._rules.append(
            {
                "rule_id": rule_id,
                "description": description,
                "check_fn": check_fn,
                "severity": severity,
            }
        )
        logger.info(
            "Rule %s added to RuleSet %s (severity=%s)",
            rule_id,
            self.name,
            severity,
        )

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate all rules against *context*.

        Each rule's ``check_fn`` is called with *context*.  If the callable
        raises an exception, the rule is marked as failed and the exception
        message is included in the ``message`` field.

        Args:
            context: Arbitrary dict supplied as input to every rule.

        Returns:
            A list of result dicts, one per rule::

                [
                    {
                        "rule_id": str,
                        "passed": bool,
                        "message": str,
                        "severity": str,
                    },
                    ...
                ]
        """
        results: List[Dict[str, Any]] = []

        for rule in self._rules:
            rule_id = rule["rule_id"]
            try:
                passed = bool(rule["check_fn"](context))
                message = (
                    rule["description"]
                    if passed
                    else f"FAILED: {rule['description']}"
                )
            except Exception as exc:
                passed = False
                message = f"ERROR evaluating {rule_id}: {exc}"
                logger.error(
                    "Rule %s raised an exception: %s",
                    rule_id,
                    exc,
                    exc_info=True,
                )

            results.append(
                {
                    "rule_id": rule_id,
                    "passed": passed,
                    "message": message,
                    "severity": rule["severity"],
                }
            )

        passed_count = sum(1 for r in results if r["passed"])
        logger.info(
            "RuleSet %s evaluated: %d/%d passed",
            self.name,
            passed_count,
            len(results),
        )

        return results

    def summary(self) -> Dict[str, Any]:
        """
        Return a metadata summary of this rule set.

        Returns:
            A dict with keys ``name``, ``jurisdiction``, ``rule_count``,
            and ``rules`` (list of rule descriptors without callables).
        """
        return {
            "name": self.name,
            "jurisdiction": self.jurisdiction,
            "rule_count": len(self._rules),
            "rules": [
                {
                    "rule_id": r["rule_id"],
                    "description": r["description"],
                    "severity": r["severity"],
                }
                for r in self._rules
            ],
        }
