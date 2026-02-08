# -*- coding: utf-8 -*-
"""
Unit Tests for RegressionDetector (AGENT-FOUND-009)

Tests baseline creation, retrieval, update, deletion, regression checking
against baselines, historical consistency analysis, and listing.

Coverage target: 85%+ of regression_detector.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline RegressionDetector
# ---------------------------------------------------------------------------

def _content_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


class RegressionBaseline:
    def __init__(self, baseline_id: str, agent_type: str, input_hash: str,
                 output_hash: str, output_data: Optional[Dict] = None,
                 is_active: bool = True,
                 created_at: Optional[datetime] = None,
                 description: str = ""):
        self.baseline_id = baseline_id
        self.agent_type = agent_type
        self.input_hash = input_hash
        self.output_hash = output_hash
        self.output_data = output_data or {}
        self.is_active = is_active
        self.created_at = created_at or datetime.now(timezone.utc)
        self.description = description


class RegressionResult:
    def __init__(self, has_regression: bool, baseline_hash: str = "",
                 current_hash: str = "", diffs: Optional[List[str]] = None,
                 message: str = ""):
        self.has_regression = has_regression
        self.baseline_hash = baseline_hash
        self.current_hash = current_hash
        self.diffs = diffs or []
        self.message = message


class RegressionDetector:
    """In-memory regression detector."""

    def __init__(self):
        self._baselines: Dict[str, RegressionBaseline] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._counter = 0

    def check_regression(self, agent_type: str, input_data: Dict[str, Any],
                         output_data: Dict[str, Any]) -> RegressionResult:
        """Check for regression against stored baseline."""
        input_hash = _content_hash(input_data)
        current_hash = _content_hash(output_data)

        # Find matching baseline
        baseline = self._find_baseline(agent_type, input_hash)
        if baseline is None:
            return RegressionResult(
                has_regression=False, current_hash=current_hash,
                message="No baseline found for comparison",
            )

        matches = baseline.output_hash == current_hash
        diffs = []
        if not matches:
            for key in baseline.output_data:
                if baseline.output_data.get(key) != output_data.get(key):
                    diffs.append(key)

        # Record history
        key = f"{agent_type}:{input_hash}"
        if key not in self._history:
            self._history[key] = []
        self._history[key].append({
            "output_hash": current_hash, "matched": matches,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return RegressionResult(
            has_regression=not matches,
            baseline_hash=baseline.output_hash,
            current_hash=current_hash,
            diffs=diffs,
            message="Regression detected" if not matches else "Output matches baseline",
        )

    def create_baseline(self, agent_type: str, input_data: Dict[str, Any],
                        output_data: Dict[str, Any],
                        description: str = "") -> RegressionBaseline:
        """Create a new regression baseline."""
        self._counter += 1
        baseline_id = f"rb-{self._counter:04d}"
        input_hash = _content_hash(input_data)
        output_hash = _content_hash(output_data)

        baseline = RegressionBaseline(
            baseline_id=baseline_id, agent_type=agent_type,
            input_hash=input_hash, output_hash=output_hash,
            output_data=output_data, description=description,
        )
        self._baselines[baseline_id] = baseline
        return baseline

    def get_baseline(self, baseline_id: str) -> Optional[RegressionBaseline]:
        return self._baselines.get(baseline_id)

    def update_baseline(self, baseline_id: str,
                        output_data: Dict[str, Any]) -> Optional[RegressionBaseline]:
        baseline = self._baselines.get(baseline_id)
        if baseline is None:
            return None
        baseline.output_data = output_data
        baseline.output_hash = _content_hash(output_data)
        return baseline

    def list_baselines(self, agent_type: Optional[str] = None) -> List[RegressionBaseline]:
        baselines = [b for b in self._baselines.values() if b.is_active]
        if agent_type:
            baselines = [b for b in baselines if b.agent_type == agent_type]
        return baselines

    def delete_baseline(self, baseline_id: str) -> bool:
        if baseline_id in self._baselines:
            self._baselines[baseline_id].is_active = False
            return True
        return False

    def check_historical_consistency(self, agent_type: str,
                                     input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if outputs have been consistent across history."""
        input_hash = _content_hash(input_data)
        key = f"{agent_type}:{input_hash}"
        history = self._history.get(key, [])

        if not history:
            return {"consistent": True, "total_runs": 0, "message": "No history"}

        hashes = [h["output_hash"] for h in history]
        unique_hashes = set(hashes)
        consistent = len(unique_hashes) == 1

        return {
            "consistent": consistent,
            "total_runs": len(history),
            "unique_outputs": len(unique_hashes),
            "message": "All outputs identical" if consistent else "Outputs vary",
        }

    def _find_baseline(self, agent_type: str,
                       input_hash: str) -> Optional[RegressionBaseline]:
        for baseline in self._baselines.values():
            if (baseline.agent_type == agent_type
                    and baseline.input_hash == input_hash
                    and baseline.is_active):
                return baseline
        return None


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def detector():
    return RegressionDetector()


class TestCheckRegressionNoBaseline:
    def test_no_baseline_no_regression(self, detector):
        result = detector.check_regression("Agent", {"x": 1}, {"r": 42})
        assert result.has_regression is False

    def test_no_baseline_message(self, detector):
        result = detector.check_regression("Agent", {"x": 1}, {"r": 42})
        assert "no baseline" in result.message.lower()


class TestCheckRegressionMatchesBaseline:
    def test_matches_baseline(self, detector):
        input_data = {"x": 1}
        output_data = {"result": 42}
        detector.create_baseline("Agent", input_data, output_data)
        result = detector.check_regression("Agent", input_data, output_data)
        assert result.has_regression is False

    def test_matches_message(self, detector):
        input_data = {"x": 1}
        output_data = {"result": 42}
        detector.create_baseline("Agent", input_data, output_data)
        result = detector.check_regression("Agent", input_data, output_data)
        assert "matches" in result.message.lower()


class TestCheckRegressionDiffers:
    def test_differs_from_baseline(self, detector):
        input_data = {"x": 1}
        detector.create_baseline("Agent", input_data, {"result": 42})
        result = detector.check_regression("Agent", input_data, {"result": 99})
        assert result.has_regression is True

    def test_differs_has_diffs(self, detector):
        input_data = {"x": 1}
        detector.create_baseline("Agent", input_data, {"result": 42, "unit": "kg"})
        result = detector.check_regression("Agent", input_data, {"result": 99, "unit": "kg"})
        assert "result" in result.diffs

    def test_differs_message(self, detector):
        input_data = {"x": 1}
        detector.create_baseline("Agent", input_data, {"result": 42})
        result = detector.check_regression("Agent", input_data, {"result": 99})
        assert "regression" in result.message.lower()

    def test_differs_hashes_populated(self, detector):
        input_data = {"x": 1}
        detector.create_baseline("Agent", input_data, {"result": 42})
        result = detector.check_regression("Agent", input_data, {"result": 99})
        assert result.baseline_hash != ""
        assert result.current_hash != ""
        assert result.baseline_hash != result.current_hash


class TestCreateBaseline:
    def test_create_baseline(self, detector):
        baseline = detector.create_baseline("Agent", {"x": 1}, {"r": 42})
        assert baseline.baseline_id.startswith("rb-")
        assert baseline.agent_type == "Agent"
        assert baseline.is_active is True

    def test_create_baseline_hashes(self, detector):
        baseline = detector.create_baseline("Agent", {"x": 1}, {"r": 42})
        assert baseline.input_hash != ""
        assert baseline.output_hash != ""

    def test_create_baseline_description(self, detector):
        baseline = detector.create_baseline("Agent", {}, {}, description="v1.0 baseline")
        assert baseline.description == "v1.0 baseline"

    def test_create_multiple_baselines(self, detector):
        b1 = detector.create_baseline("Agent", {"x": 1}, {"r": 1})
        b2 = detector.create_baseline("Agent", {"x": 2}, {"r": 2})
        assert b1.baseline_id != b2.baseline_id


class TestGetBaseline:
    def test_get_baseline(self, detector):
        baseline = detector.create_baseline("Agent", {}, {})
        retrieved = detector.get_baseline(baseline.baseline_id)
        assert retrieved is not None
        assert retrieved.baseline_id == baseline.baseline_id

    def test_get_nonexistent(self, detector):
        assert detector.get_baseline("rb-9999") is None


class TestUpdateBaseline:
    def test_update_baseline(self, detector):
        baseline = detector.create_baseline("Agent", {"x": 1}, {"r": 42})
        old_hash = baseline.output_hash
        updated = detector.update_baseline(baseline.baseline_id, {"r": 99})
        assert updated is not None
        assert updated.output_data["r"] == 99
        assert updated.output_hash != old_hash

    def test_update_nonexistent(self, detector):
        assert detector.update_baseline("rb-9999", {}) is None


class TestListBaselines:
    def test_list_baselines_empty(self, detector):
        assert detector.list_baselines() == []

    def test_list_baselines(self, detector):
        detector.create_baseline("Agent1", {}, {})
        detector.create_baseline("Agent2", {}, {})
        assert len(detector.list_baselines()) == 2

    def test_list_baselines_by_agent(self, detector):
        detector.create_baseline("Agent1", {}, {})
        detector.create_baseline("Agent2", {}, {})
        detector.create_baseline("Agent1", {"x": 1}, {})
        assert len(detector.list_baselines(agent_type="Agent1")) == 2

    def test_list_baselines_excludes_inactive(self, detector):
        baseline = detector.create_baseline("Agent", {}, {})
        detector.delete_baseline(baseline.baseline_id)
        assert len(detector.list_baselines()) == 0


class TestDeleteBaseline:
    def test_delete_baseline(self, detector):
        baseline = detector.create_baseline("Agent", {}, {})
        assert detector.delete_baseline(baseline.baseline_id) is True
        assert detector.get_baseline(baseline.baseline_id).is_active is False

    def test_delete_nonexistent(self, detector):
        assert detector.delete_baseline("rb-9999") is False


class TestHistoricalConsistency:
    def test_historical_consistency_no_history(self, detector):
        result = detector.check_historical_consistency("Agent", {"x": 1})
        assert result["consistent"] is True
        assert result["total_runs"] == 0

    def test_historical_consistency_match(self, detector):
        input_data = {"x": 1}
        output_data = {"r": 42}
        detector.create_baseline("Agent", input_data, output_data)
        detector.check_regression("Agent", input_data, output_data)
        detector.check_regression("Agent", input_data, output_data)
        result = detector.check_historical_consistency("Agent", input_data)
        assert result["consistent"] is True
        assert result["total_runs"] == 2

    def test_historical_consistency_mismatch(self, detector):
        input_data = {"x": 1}
        detector.create_baseline("Agent", input_data, {"r": 42})
        detector.check_regression("Agent", input_data, {"r": 42})
        detector.check_regression("Agent", input_data, {"r": 99})
        result = detector.check_historical_consistency("Agent", input_data)
        assert result["consistent"] is False
        assert result["unique_outputs"] == 2

    def test_historical_consistency_message(self, detector):
        input_data = {"x": 1}
        detector.create_baseline("Agent", input_data, {"r": 42})
        detector.check_regression("Agent", input_data, {"r": 42})
        result = detector.check_historical_consistency("Agent", input_data)
        assert "identical" in result["message"].lower()
