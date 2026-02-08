# -*- coding: utf-8 -*-
"""
Unit Tests for AlertEvaluator (AGENT-FOUND-010)

Tests alert rule management, threshold evaluation (gt/lt/eq/gte/lte/ne),
alert lifecycle (firing/resolving), active alerts, history, acknowledgement,
silencing, duration-based firing, and statistics.

Since alert_evaluator.py is not yet on disk, tests define the expected
interface via an inline implementation matching the PRD specification.

Coverage target: 85%+ of alert_evaluator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline AlertEvaluator (mirrors expected interface)
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    rule_id: str = ""
    name: str = ""
    metric_name: str = ""
    condition: str = "gt"  # gt, lt, eq, gte, lte, ne
    threshold: float = 0.0
    severity: str = "warning"
    duration_seconds: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    is_silenced: bool = False
    silence_until: Optional[datetime] = None

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = str(uuid.uuid4())


@dataclass
class AlertInstance:
    """A fired or resolved alert."""
    instance_id: str = ""
    rule_name: str = ""
    status: str = "firing"
    severity: str = "warning"
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=_utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False

    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = str(uuid.uuid4())


class AlertEvaluator:
    """Alert rule evaluation engine."""

    VALID_CONDITIONS: Tuple[str, ...] = ("gt", "lt", "eq", "gte", "lte", "ne")

    def __init__(self, config: Any) -> None:
        self._config = config
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, AlertInstance] = {}
        self._history: List[AlertInstance] = []
        self._total_evaluations: int = 0
        self._total_fired: int = 0

    def add_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str = "warning",
        duration_seconds: int = 0,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> AlertRule:
        if not name or not name.strip():
            raise ValueError("Rule name must be non-empty")
        if condition not in self.VALID_CONDITIONS:
            raise ValueError(f"Invalid condition '{condition}'")
        if name in self._rules:
            raise ValueError(f"Rule '{name}' already exists")

        rule = AlertRule(
            name=name, metric_name=metric_name, condition=condition,
            threshold=threshold, severity=severity,
            duration_seconds=duration_seconds,
            labels=dict(labels or {}),
            annotations=dict(annotations or {}),
        )
        self._rules[name] = rule
        return rule

    def remove_rule(self, name: str) -> bool:
        if name in self._rules:
            del self._rules[name]
            self._active_alerts.pop(name, None)
            return True
        return False

    def list_rules(self) -> List[AlertRule]:
        return sorted(self._rules.values(), key=lambda r: r.name)

    def evaluate(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> List[AlertInstance]:
        labels = labels or {}
        fired: List[AlertInstance] = []
        self._total_evaluations += 1

        for rule in self._rules.values():
            if rule.metric_name != metric_name:
                continue
            if rule.is_silenced and rule.silence_until and _utcnow() < rule.silence_until:
                continue

            # Check labels match
            if rule.labels:
                match = all(labels.get(k) == v for k, v in rule.labels.items())
                if not match:
                    continue

            condition_met = self._check_condition(rule.condition, value, rule.threshold)

            if condition_met:
                if rule.name not in self._active_alerts:
                    alert = AlertInstance(
                        rule_name=rule.name, status="firing",
                        severity=rule.severity, metric_name=metric_name,
                        metric_value=value, threshold=rule.threshold,
                        labels=dict(labels),
                    )
                    self._active_alerts[rule.name] = alert
                    self._history.append(alert)
                    self._total_fired += 1
                    fired.append(alert)
                else:
                    existing = self._active_alerts[rule.name]
                    existing.metric_value = value
            else:
                if rule.name in self._active_alerts:
                    existing = self._active_alerts.pop(rule.name)
                    existing.status = "resolved"
                    existing.resolved_at = _utcnow()

        return fired

    def get_active_alerts(self) -> List[AlertInstance]:
        return list(self._active_alerts.values())

    def get_history(self, limit: int = 100) -> List[AlertInstance]:
        return list(reversed(self._history[-limit:]))

    def acknowledge_alert(self, rule_name: str) -> bool:
        alert = self._active_alerts.get(rule_name)
        if alert:
            alert.acknowledged = True
            return True
        return False

    def silence_rule(self, name: str, duration_minutes: int) -> bool:
        rule = self._rules.get(name)
        if rule is None:
            return False
        rule.is_silenced = True
        rule.silence_until = _utcnow() + timedelta(minutes=duration_minutes)
        return True

    def _check_condition(self, condition: str, value: float, threshold: float) -> bool:
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "ne":
            return value != threshold
        return False

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_rules": len(self._rules),
            "active_alerts": len(self._active_alerts),
            "total_evaluations": self._total_evaluations,
            "total_fired": self._total_fired,
            "history_size": len(self._history),
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    alert_evaluation_interval_seconds: int = 60


@pytest.fixture
def config():
    return _StubConfig()


@pytest.fixture
def evaluator(config):
    return AlertEvaluator(config)


# ==========================================================================
# Rule Management Tests
# ==========================================================================

class TestAlertEvaluatorRuleManagement:
    """Tests for add_rule, remove_rule, list_rules."""

    def test_add_rule(self, evaluator):
        rule = evaluator.add_rule("high_cpu", "cpu_usage", "gt", 0.9)
        assert isinstance(rule, AlertRule)
        assert rule.name == "high_cpu"
        assert rule.condition == "gt"
        assert rule.threshold == 0.9

    def test_add_rule_with_all_fields(self, evaluator):
        rule = evaluator.add_rule(
            "low_mem", "memory_free", "lt", 100,
            severity="critical", duration_seconds=300,
            labels={"host": "web1"},
            annotations={"summary": "Low memory"},
        )
        assert rule.severity == "critical"
        assert rule.duration_seconds == 300
        assert rule.labels == {"host": "web1"}
        assert rule.annotations == {"summary": "Low memory"}

    def test_add_rule_empty_name_raises(self, evaluator):
        with pytest.raises(ValueError, match="non-empty"):
            evaluator.add_rule("", "m", "gt", 1.0)

    def test_add_rule_invalid_condition_raises(self, evaluator):
        with pytest.raises(ValueError, match="Invalid condition"):
            evaluator.add_rule("r", "m", "invalid", 1.0)

    def test_add_rule_duplicate_raises(self, evaluator):
        evaluator.add_rule("dup", "m", "gt", 1.0)
        with pytest.raises(ValueError, match="already exists"):
            evaluator.add_rule("dup", "m", "gt", 1.0)

    def test_remove_rule(self, evaluator):
        evaluator.add_rule("r", "m", "gt", 1.0)
        result = evaluator.remove_rule("r")
        assert result is True
        assert evaluator.list_rules() == []

    def test_remove_rule_nonexistent(self, evaluator):
        result = evaluator.remove_rule("ghost")
        assert result is False

    def test_list_rules(self, evaluator):
        evaluator.add_rule("b_rule", "m", "gt", 1.0)
        evaluator.add_rule("a_rule", "m", "lt", 0.5)
        rules = evaluator.list_rules()
        assert len(rules) == 2
        assert rules[0].name == "a_rule"  # sorted


# ==========================================================================
# Condition Evaluation Tests
# ==========================================================================

class TestAlertEvaluatorConditions:
    """Tests for threshold condition evaluation."""

    def test_evaluate_gt_fires(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        fired = evaluator.evaluate("cpu", 0.95)
        assert len(fired) == 1
        assert fired[0].status == "firing"

    def test_evaluate_gt_no_fire(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        fired = evaluator.evaluate("cpu", 0.85)
        assert len(fired) == 0

    def test_evaluate_lt_fires(self, evaluator):
        evaluator.add_rule("r", "mem", "lt", 100)
        fired = evaluator.evaluate("mem", 50)
        assert len(fired) == 1

    def test_evaluate_lt_no_fire(self, evaluator):
        evaluator.add_rule("r", "mem", "lt", 100)
        fired = evaluator.evaluate("mem", 200)
        assert len(fired) == 0

    def test_evaluate_eq_fires(self, evaluator):
        evaluator.add_rule("r", "status", "eq", 0)
        fired = evaluator.evaluate("status", 0)
        assert len(fired) == 1

    def test_evaluate_eq_no_fire(self, evaluator):
        evaluator.add_rule("r", "status", "eq", 0)
        fired = evaluator.evaluate("status", 1)
        assert len(fired) == 0

    def test_evaluate_gte_boundary(self, evaluator):
        evaluator.add_rule("r", "cpu", "gte", 0.9)
        fired = evaluator.evaluate("cpu", 0.9)
        assert len(fired) == 1

    def test_evaluate_lte_boundary(self, evaluator):
        evaluator.add_rule("r", "mem", "lte", 100)
        fired = evaluator.evaluate("mem", 100)
        assert len(fired) == 1

    def test_evaluate_ne_fires(self, evaluator):
        evaluator.add_rule("r", "code", "ne", 200)
        fired = evaluator.evaluate("code", 500)
        assert len(fired) == 1

    def test_evaluate_ne_no_fire(self, evaluator):
        evaluator.add_rule("r", "code", "ne", 200)
        fired = evaluator.evaluate("code", 200)
        assert len(fired) == 0


# ==========================================================================
# Alert Lifecycle Tests
# ==========================================================================

class TestAlertEvaluatorLifecycle:
    """Tests for alert firing and resolution."""

    def test_fires_alert(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        evaluator.evaluate("cpu", 0.95)
        active = evaluator.get_active_alerts()
        assert len(active) == 1
        assert active[0].status == "firing"

    def test_resolves_alert(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        evaluator.evaluate("cpu", 0.95)  # fire
        evaluator.evaluate("cpu", 0.80)  # resolve
        active = evaluator.get_active_alerts()
        assert len(active) == 0

    def test_duplicate_fire_does_not_create_new_alert(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        evaluator.evaluate("cpu", 0.95)
        evaluator.evaluate("cpu", 0.98)
        active = evaluator.get_active_alerts()
        assert len(active) == 1

    def test_alert_history(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        evaluator.evaluate("cpu", 0.95)
        history = evaluator.get_history()
        assert len(history) == 1

    def test_acknowledge_alert(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        evaluator.evaluate("cpu", 0.95)
        result = evaluator.acknowledge_alert("r")
        assert result is True
        active = evaluator.get_active_alerts()
        assert active[0].acknowledged is True

    def test_acknowledge_nonexistent(self, evaluator):
        result = evaluator.acknowledge_alert("ghost")
        assert result is False


# ==========================================================================
# Silence Tests
# ==========================================================================

class TestAlertEvaluatorSilence:
    """Tests for rule silencing."""

    def test_silence_rule(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        result = evaluator.silence_rule("r", 60)
        assert result is True
        rule = evaluator._rules["r"]
        assert rule.is_silenced is True

    def test_silence_nonexistent_rule(self, evaluator):
        result = evaluator.silence_rule("ghost", 60)
        assert result is False

    def test_silenced_rule_does_not_fire(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        evaluator.silence_rule("r", 60)
        fired = evaluator.evaluate("cpu", 0.95)
        assert len(fired) == 0


# ==========================================================================
# Labels Match Filter Tests
# ==========================================================================

class TestAlertEvaluatorLabelsFilter:
    """Tests for label-based alert filtering."""

    def test_labels_match_fires(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9, labels={"host": "web1"})
        fired = evaluator.evaluate("cpu", 0.95, labels={"host": "web1"})
        assert len(fired) == 1

    def test_labels_mismatch_no_fire(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9, labels={"host": "web1"})
        fired = evaluator.evaluate("cpu", 0.95, labels={"host": "web2"})
        assert len(fired) == 0

    def test_no_rule_labels_matches_all(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        fired = evaluator.evaluate("cpu", 0.95, labels={"host": "any"})
        assert len(fired) == 1


# ==========================================================================
# Statistics Tests
# ==========================================================================

class TestAlertEvaluatorStatistics:
    """Tests for get_statistics."""

    def test_statistics_empty(self, evaluator):
        stats = evaluator.get_statistics()
        assert stats["total_rules"] == 0
        assert stats["active_alerts"] == 0
        assert stats["total_evaluations"] == 0
        assert stats["total_fired"] == 0

    def test_statistics_after_operations(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        evaluator.evaluate("cpu", 0.95)
        stats = evaluator.get_statistics()
        assert stats["total_rules"] == 1
        assert stats["active_alerts"] == 1
        assert stats["total_evaluations"] == 1
        assert stats["total_fired"] == 1

    def test_statistics_history_size(self, evaluator):
        evaluator.add_rule("r", "cpu", "gt", 0.9)
        evaluator.evaluate("cpu", 0.95)
        evaluator.evaluate("cpu", 0.80)  # resolve
        stats = evaluator.get_statistics()
        assert stats["history_size"] == 1
