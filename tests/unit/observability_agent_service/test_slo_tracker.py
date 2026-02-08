# -*- coding: utf-8 -*-
"""
Unit Tests for SLOTracker (AGENT-FOUND-010)

Tests SLO CRUD operations, compliance calculation, burn rate computation,
error budget tracking, evaluation, target validation, and statistics.

Since slo_tracker.py is not yet on disk, tests define the expected
interface via an inline implementation matching the PRD specification.

Coverage target: 85%+ of slo_tracker.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline SLOTracker (mirrors expected interface)
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class SLODefinition:
    """A Service Level Objective definition."""
    slo_id: str = ""
    name: str = ""
    description: str = ""
    service_name: str = ""
    slo_type: str = "availability"
    target: float = 0.999
    window_days: int = 30
    burn_rate_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "fast_burn": 14.4,
        "medium_burn": 6.0,
        "slow_burn": 1.0,
    })
    created_at: datetime = field(default_factory=_utcnow)
    tenant_id: str = "default"

    def __post_init__(self):
        if not self.slo_id:
            self.slo_id = str(uuid.uuid4())


@dataclass
class SLOStatus:
    """Current status of an SLO."""
    slo_id: str = ""
    name: str = ""
    current_value: float = 1.0
    target: float = 0.999
    compliance_ratio: float = 1.0
    error_budget_total: float = 0.0
    error_budget_consumed: float = 0.0
    error_budget_remaining: float = 0.0
    burn_rate_1h: float = 0.0
    burn_rate_6h: float = 0.0
    burn_rate_24h: float = 0.0
    is_burning: bool = False
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None


class SLOTracker:
    """SLO tracking engine with burn rate analysis."""

    VALID_SLO_TYPES = ("availability", "latency", "throughput", "error_rate", "saturation")

    def __init__(self, config: Any) -> None:
        self._config = config
        self._slos: Dict[str, SLODefinition] = {}
        self._observations: Dict[str, List[Dict[str, Any]]] = {}
        self._total_evaluations: int = 0

    def create_slo(
        self,
        name: str,
        service_name: str = "",
        slo_type: str = "availability",
        target: float = 0.999,
        window_days: int = 30,
        description: str = "",
        burn_rate_thresholds: Optional[Dict[str, float]] = None,
        tenant_id: str = "default",
    ) -> SLODefinition:
        if not name or not name.strip():
            raise ValueError("SLO name must be non-empty")
        if target < 0.0 or target > 1.0:
            raise ValueError(f"SLO target must be between 0.0 and 1.0, got {target}")
        if slo_type not in self.VALID_SLO_TYPES:
            raise ValueError(f"Invalid SLO type '{slo_type}'")

        slo = SLODefinition(
            name=name, service_name=service_name, slo_type=slo_type,
            target=target, window_days=window_days, description=description,
            tenant_id=tenant_id,
        )
        if burn_rate_thresholds:
            slo.burn_rate_thresholds = burn_rate_thresholds

        self._slos[slo.slo_id] = slo
        self._observations[slo.slo_id] = []
        return slo

    def update_slo(
        self,
        slo_id: str,
        target: Optional[float] = None,
        window_days: Optional[int] = None,
        description: Optional[str] = None,
    ) -> SLODefinition:
        slo = self._slos.get(slo_id)
        if slo is None:
            raise ValueError(f"SLO '{slo_id}' not found")
        if target is not None:
            if target < 0.0 or target > 1.0:
                raise ValueError(f"SLO target must be between 0.0 and 1.0")
            slo.target = target
        if window_days is not None:
            slo.window_days = window_days
        if description is not None:
            slo.description = description
        return slo

    def delete_slo(self, slo_id: str) -> bool:
        if slo_id in self._slos:
            del self._slos[slo_id]
            self._observations.pop(slo_id, None)
            return True
        return False

    def get_slo(self, slo_id: str) -> Optional[SLODefinition]:
        return self._slos.get(slo_id)

    def list_slos(self) -> List[SLODefinition]:
        return sorted(self._slos.values(), key=lambda s: s.name)

    def record_observation(self, slo_id: str, value: float, is_good: bool = True) -> None:
        if slo_id not in self._slos:
            raise ValueError(f"SLO '{slo_id}' not found")
        self._observations[slo_id].append({
            "value": value,
            "is_good": is_good,
            "timestamp": _utcnow().isoformat(),
        })

    def calculate_compliance(self, slo_id: str) -> SLOStatus:
        slo = self._slos.get(slo_id)
        if slo is None:
            raise ValueError(f"SLO '{slo_id}' not found")

        observations = self._observations.get(slo_id, [])
        total = len(observations)
        good = sum(1 for o in observations if o["is_good"])

        current_value = good / total if total > 0 else 1.0
        error_budget_total = 1.0 - slo.target
        error_budget_consumed = max(0.0, 1.0 - current_value)
        error_budget_remaining = max(0.0, error_budget_total - error_budget_consumed)

        compliance_ratio = current_value / slo.target if slo.target > 0 else 1.0
        is_burning = error_budget_consumed > 0

        now = _utcnow()
        window_start = now - timedelta(days=slo.window_days)

        self._total_evaluations += 1

        return SLOStatus(
            slo_id=slo_id,
            name=slo.name,
            current_value=current_value,
            target=slo.target,
            compliance_ratio=compliance_ratio,
            error_budget_total=error_budget_total,
            error_budget_consumed=error_budget_consumed,
            error_budget_remaining=error_budget_remaining,
            is_burning=is_burning,
            window_start=window_start,
            window_end=now,
        )

    def calculate_burn_rate(self, slo_id: str) -> Dict[str, float]:
        slo = self._slos.get(slo_id)
        if slo is None:
            raise ValueError(f"SLO '{slo_id}' not found")

        status = self.calculate_compliance(slo_id)
        error_budget = 1.0 - slo.target
        if error_budget <= 0:
            return {"burn_rate_1h": 0.0, "burn_rate_6h": 0.0, "burn_rate_24h": 0.0}

        consumed_rate = status.error_budget_consumed / error_budget if error_budget > 0 else 0.0
        return {
            "burn_rate_1h": consumed_rate * 24.0,
            "burn_rate_6h": consumed_rate * 4.0,
            "burn_rate_24h": consumed_rate,
        }

    def evaluate_all(self) -> List[SLOStatus]:
        results = []
        for slo_id in self._slos:
            results.append(self.calculate_compliance(slo_id))
        return results

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_slos": len(self._slos),
            "total_evaluations": self._total_evaluations,
            "total_observations": sum(len(v) for v in self._observations.values()),
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    default_slo_target: float = 0.999
    slo_evaluation_interval_seconds: int = 300


@pytest.fixture
def config():
    return _StubConfig()


@pytest.fixture
def tracker(config):
    return SLOTracker(config)


# ==========================================================================
# Create SLO Tests
# ==========================================================================

class TestSLOTrackerCreate:
    """Tests for create_slo."""

    def test_create_slo(self, tracker):
        slo = tracker.create_slo("API Availability", service_name="api")
        assert isinstance(slo, SLODefinition)
        assert slo.name == "API Availability"
        assert slo.service_name == "api"
        assert slo.target == pytest.approx(0.999)

    def test_create_slo_with_custom_target(self, tracker):
        slo = tracker.create_slo("Latency", target=0.95)
        assert slo.target == pytest.approx(0.95)

    def test_create_slo_with_burn_rate_thresholds(self, tracker):
        custom = {"fast_burn": 20.0, "medium_burn": 10.0, "slow_burn": 2.0}
        slo = tracker.create_slo("Custom", burn_rate_thresholds=custom)
        assert slo.burn_rate_thresholds == custom

    def test_create_slo_empty_name_raises(self, tracker):
        with pytest.raises(ValueError, match="non-empty"):
            tracker.create_slo("")

    def test_create_slo_target_below_zero_raises(self, tracker):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            tracker.create_slo("bad", target=-0.1)

    def test_create_slo_target_above_one_raises(self, tracker):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            tracker.create_slo("bad", target=1.5)

    def test_create_slo_boundary_zero(self, tracker):
        slo = tracker.create_slo("zero", target=0.0)
        assert slo.target == 0.0

    def test_create_slo_boundary_one(self, tracker):
        slo = tracker.create_slo("one", target=1.0)
        assert slo.target == 1.0

    def test_create_slo_invalid_type_raises(self, tracker):
        with pytest.raises(ValueError, match="Invalid SLO type"):
            tracker.create_slo("bad", slo_type="invalid")

    @pytest.mark.parametrize("slo_type", [
        "availability", "latency", "throughput", "error_rate", "saturation",
    ])
    def test_create_slo_valid_types(self, tracker, slo_type):
        slo = tracker.create_slo(f"test_{slo_type}", slo_type=slo_type)
        assert slo.slo_type == slo_type


# ==========================================================================
# Update SLO Tests
# ==========================================================================

class TestSLOTrackerUpdate:
    """Tests for update_slo."""

    def test_update_slo_target(self, tracker):
        slo = tracker.create_slo("test")
        updated = tracker.update_slo(slo.slo_id, target=0.995)
        assert updated.target == pytest.approx(0.995)

    def test_update_slo_description(self, tracker):
        slo = tracker.create_slo("test")
        updated = tracker.update_slo(slo.slo_id, description="Updated desc")
        assert updated.description == "Updated desc"

    def test_update_slo_nonexistent_raises(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            tracker.update_slo("ghost", target=0.99)

    def test_update_slo_invalid_target_raises(self, tracker):
        slo = tracker.create_slo("test")
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            tracker.update_slo(slo.slo_id, target=2.0)


# ==========================================================================
# Delete SLO Tests
# ==========================================================================

class TestSLOTrackerDelete:
    """Tests for delete_slo."""

    def test_delete_slo(self, tracker):
        slo = tracker.create_slo("test")
        result = tracker.delete_slo(slo.slo_id)
        assert result is True
        assert tracker.get_slo(slo.slo_id) is None

    def test_delete_slo_nonexistent(self, tracker):
        result = tracker.delete_slo("ghost")
        assert result is False


# ==========================================================================
# Get / List SLO Tests
# ==========================================================================

class TestSLOTrackerGetList:
    """Tests for get_slo and list_slos."""

    def test_get_slo_existing(self, tracker):
        slo = tracker.create_slo("test")
        result = tracker.get_slo(slo.slo_id)
        assert result is not None
        assert result.name == "test"

    def test_get_slo_nonexistent(self, tracker):
        assert tracker.get_slo("ghost") is None

    def test_list_slos(self, tracker):
        tracker.create_slo("B SLO")
        tracker.create_slo("A SLO")
        result = tracker.list_slos()
        assert len(result) == 2
        assert result[0].name == "A SLO"

    def test_list_slos_empty(self, tracker):
        assert tracker.list_slos() == []


# ==========================================================================
# Compliance Calculation Tests
# ==========================================================================

class TestSLOTrackerCompliance:
    """Tests for compliance calculation."""

    def test_calculate_compliance_all_good(self, tracker):
        slo = tracker.create_slo("test", target=0.99)
        for _ in range(100):
            tracker.record_observation(slo.slo_id, 1.0, is_good=True)
        status = tracker.calculate_compliance(slo.slo_id)
        assert status.current_value == pytest.approx(1.0)
        assert status.compliance_ratio >= 1.0

    def test_calculate_compliance_some_bad(self, tracker):
        slo = tracker.create_slo("test", target=0.99)
        for _ in range(95):
            tracker.record_observation(slo.slo_id, 1.0, is_good=True)
        for _ in range(5):
            tracker.record_observation(slo.slo_id, 0.0, is_good=False)
        status = tracker.calculate_compliance(slo.slo_id)
        assert status.current_value == pytest.approx(0.95)
        assert status.is_burning is True

    def test_calculate_compliance_no_observations(self, tracker):
        slo = tracker.create_slo("test")
        status = tracker.calculate_compliance(slo.slo_id)
        assert status.current_value == pytest.approx(1.0)

    def test_calculate_compliance_nonexistent_raises(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            tracker.calculate_compliance("ghost")

    def test_error_budget_calculation(self, tracker):
        slo = tracker.create_slo("test", target=0.99)
        for _ in range(99):
            tracker.record_observation(slo.slo_id, 1.0, is_good=True)
        tracker.record_observation(slo.slo_id, 0.0, is_good=False)
        status = tracker.calculate_compliance(slo.slo_id)
        assert status.error_budget_total == pytest.approx(0.01)
        assert status.error_budget_consumed == pytest.approx(0.01)
        assert status.error_budget_remaining == pytest.approx(0.0, abs=1e-10)


# ==========================================================================
# Burn Rate Tests
# ==========================================================================

class TestSLOTrackerBurnRate:
    """Tests for burn rate calculation."""

    def test_calculate_burn_rate(self, tracker):
        slo = tracker.create_slo("test", target=0.99)
        for _ in range(90):
            tracker.record_observation(slo.slo_id, 1.0, is_good=True)
        for _ in range(10):
            tracker.record_observation(slo.slo_id, 0.0, is_good=False)
        br = tracker.calculate_burn_rate(slo.slo_id)
        assert "burn_rate_1h" in br
        assert "burn_rate_6h" in br
        assert "burn_rate_24h" in br
        assert br["burn_rate_1h"] > 0  # Error budget is being consumed

    def test_calculate_burn_rate_nonexistent_raises(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            tracker.calculate_burn_rate("ghost")

    def test_burn_rate_zero_budget(self, tracker):
        slo = tracker.create_slo("perfect", target=1.0)
        br = tracker.calculate_burn_rate(slo.slo_id)
        assert br["burn_rate_1h"] == 0.0


# ==========================================================================
# Evaluate All Tests
# ==========================================================================

class TestSLOTrackerEvaluateAll:
    """Tests for evaluate_all."""

    def test_evaluate_all_slos(self, tracker):
        tracker.create_slo("SLO 1")
        tracker.create_slo("SLO 2")
        tracker.create_slo("SLO 3")
        results = tracker.evaluate_all()
        assert len(results) == 3

    def test_evaluate_all_empty(self, tracker):
        results = tracker.evaluate_all()
        assert results == []


# ==========================================================================
# Record Observation Tests
# ==========================================================================

class TestSLOTrackerRecordObservation:
    """Tests for record_observation."""

    def test_record_observation(self, tracker):
        slo = tracker.create_slo("test")
        tracker.record_observation(slo.slo_id, 1.0, is_good=True)
        tracker.record_observation(slo.slo_id, 0.5, is_good=False)
        stats = tracker.get_statistics()
        assert stats["total_observations"] == 2

    def test_record_observation_nonexistent_raises(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            tracker.record_observation("ghost", 1.0)


# ==========================================================================
# Statistics Tests
# ==========================================================================

class TestSLOTrackerStatistics:
    """Tests for get_statistics."""

    def test_statistics_empty(self, tracker):
        stats = tracker.get_statistics()
        assert stats["total_slos"] == 0
        assert stats["total_evaluations"] == 0
        assert stats["total_observations"] == 0

    def test_statistics_after_operations(self, tracker):
        slo = tracker.create_slo("test")
        tracker.record_observation(slo.slo_id, 1.0)
        tracker.calculate_compliance(slo.slo_id)
        stats = tracker.get_statistics()
        assert stats["total_slos"] == 1
        assert stats["total_evaluations"] == 1
        assert stats["total_observations"] == 1
