# -*- coding: utf-8 -*-
"""
Integration Tests for Drift Monitoring (AGENT-FOUND-008)

Tests drift lifecycle (create-detect-update), severity escalation,
multiple baselines, and field-level tracking.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline drift detector for integration testing
# ---------------------------------------------------------------------------

class DriftSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class DriftDetector:
    def __init__(self, tolerance=1e-9, soft=0.01, hard=0.05):
        self.tolerance = tolerance
        self.soft = soft
        self.hard = hard
        self._baselines: Dict[str, Dict] = {}

    def _hash(self, data: Dict) -> str:
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def create_baseline(self, bl_id: str, name: str, data: Dict) -> Dict:
        bl = {"baseline_id": bl_id, "name": name, "data": data,
              "data_hash": self._hash(data), "is_active": True}
        self._baselines[bl_id] = bl
        return bl

    def update_baseline(self, bl_id: str, data: Dict) -> Optional[Dict]:
        if bl_id not in self._baselines:
            return None
        self._baselines[bl_id]["data"] = data
        self._baselines[bl_id]["data_hash"] = self._hash(data)
        return self._baselines[bl_id]

    def deactivate_baseline(self, bl_id: str) -> bool:
        if bl_id in self._baselines:
            self._baselines[bl_id]["is_active"] = False
            return True
        return False

    def detect_drift(self, baseline: Dict, current: Dict) -> Dict:
        drifted = []
        details = {}
        max_pct = 0.0
        for k in set(list(baseline.keys()) + list(current.keys())):
            bv = baseline.get(k)
            cv = current.get(k)
            if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
                if abs(bv - cv) > self.tolerance:
                    pct = abs(cv - bv) / abs(bv) if bv != 0 else 0
                    drifted.append(k)
                    max_pct = max(max_pct, pct)
                    details[k] = {"baseline": bv, "current": cv, "pct": pct}
            elif bv != cv:
                drifted.append(k)
                details[k] = {"baseline": str(bv), "current": str(cv)}

        severity = self._classify(max_pct, len(drifted) == 0 and self._hash(baseline) == self._hash(current))
        return {
            "severity": severity.value,
            "max_drift_pct": max_pct * 100,
            "drifted_fields": drifted,
            "details": details,
            "is_acceptable": severity in (DriftSeverity.NONE, DriftSeverity.MINOR),
        }

    def _classify(self, max_pct: float, exact_match: bool) -> DriftSeverity:
        if max_pct == 0 and exact_match:
            return DriftSeverity.NONE
        if max_pct <= self.soft:
            return DriftSeverity.MINOR
        if max_pct <= self.hard:
            return DriftSeverity.MODERATE
        return DriftSeverity.CRITICAL

    def compare_to_baseline(self, bl_id: str, current: Dict) -> Optional[Dict]:
        bl = self._baselines.get(bl_id)
        if bl is None:
            return None
        return self.detect_drift(bl["data"], current)

    def list_active(self) -> List[Dict]:
        return [b for b in self._baselines.values() if b["is_active"]]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestDriftLifecycle:
    """Test create -> detect -> update lifecycle."""

    def test_create_detect_no_drift(self):
        d = DriftDetector()
        d.create_baseline("bl-001", "emissions", {"total": 100.0})
        result = d.compare_to_baseline("bl-001", {"total": 100.0})
        assert result["severity"] == "none"

    def test_create_detect_drift_update(self):
        d = DriftDetector()
        d.create_baseline("bl-001", "emissions", {"total": 100.0})
        # Detect drift
        result = d.compare_to_baseline("bl-001", {"total": 110.0})
        assert result["severity"] != "none"
        # Update baseline to new value
        d.update_baseline("bl-001", {"total": 110.0})
        # No more drift
        result2 = d.compare_to_baseline("bl-001", {"total": 110.0})
        assert result2["severity"] == "none"

    def test_create_deactivate_create_new(self):
        d = DriftDetector()
        d.create_baseline("bl-001", "old", {"v": 1})
        d.deactivate_baseline("bl-001")
        d.create_baseline("bl-002", "new", {"v": 2})
        active = d.list_active()
        assert len(active) == 1
        assert active[0]["baseline_id"] == "bl-002"

    def test_full_lifecycle_multiple_updates(self):
        d = DriftDetector()
        d.create_baseline("bl-001", "test", {"v": 100.0})
        for new_val in [101.0, 102.0, 103.0]:
            d.compare_to_baseline("bl-001", {"v": new_val})
            d.update_baseline("bl-001", {"v": new_val})
        result = d.compare_to_baseline("bl-001", {"v": 103.0})
        assert result["severity"] == "none"


class TestDriftSeverityEscalation:
    """Test drift severity escalation."""

    def test_no_drift_to_minor(self):
        d = DriftDetector()
        baseline = {"v": 1000.0}
        current = {"v": 1005.0}  # 0.5%
        result = d.detect_drift(baseline, current)
        assert result["severity"] == "minor"

    def test_minor_to_moderate(self):
        d = DriftDetector()
        baseline = {"v": 1000.0}
        current = {"v": 1030.0}  # 3%
        result = d.detect_drift(baseline, current)
        assert result["severity"] == "moderate"

    def test_moderate_to_critical(self):
        d = DriftDetector()
        baseline = {"v": 1000.0}
        current = {"v": 1100.0}  # 10%
        result = d.detect_drift(baseline, current)
        assert result["severity"] == "critical"

    def test_escalation_sequence(self):
        d = DriftDetector()
        baseline = {"v": 100.0}
        severities = []
        for pct in [0.0, 0.5, 3.0, 10.0]:
            current = {"v": 100.0 + pct}
            result = d.detect_drift(baseline, current)
            severities.append(result["severity"])
        assert severities[0] == "none"
        assert severities[1] == "minor"
        assert severities[2] == "moderate"
        assert severities[3] == "critical"

    def test_acceptable_threshold(self):
        d = DriftDetector()
        baseline = {"v": 100.0}
        # Minor is acceptable
        r1 = d.detect_drift(baseline, {"v": 100.5})
        assert r1["is_acceptable"] is True
        # Moderate is not acceptable
        r2 = d.detect_drift(baseline, {"v": 103.0})
        assert r2["is_acceptable"] is False


class TestMultipleBaselines:
    """Test drift detection with multiple baselines."""

    def test_independent_baselines(self):
        d = DriftDetector()
        d.create_baseline("bl-emissions", "emissions", {"total": 100.0})
        d.create_baseline("bl-energy", "energy", {"kwh": 5000.0})
        r1 = d.compare_to_baseline("bl-emissions", {"total": 100.0})
        r2 = d.compare_to_baseline("bl-energy", {"kwh": 5500.0})
        assert r1["severity"] == "none"
        assert r2["severity"] != "none"

    def test_baseline_isolation(self):
        d = DriftDetector()
        d.create_baseline("bl-1", "a", {"v": 100.0})
        d.create_baseline("bl-2", "b", {"v": 200.0})
        d.update_baseline("bl-1", {"v": 150.0})
        # bl-2 should be unaffected
        r = d.compare_to_baseline("bl-2", {"v": 200.0})
        assert r["severity"] == "none"

    def test_list_active_after_deactivation(self):
        d = DriftDetector()
        d.create_baseline("bl-1", "a", {"v": 1})
        d.create_baseline("bl-2", "b", {"v": 2})
        d.create_baseline("bl-3", "c", {"v": 3})
        d.deactivate_baseline("bl-2")
        active = d.list_active()
        assert len(active) == 2
        ids = [b["baseline_id"] for b in active]
        assert "bl-2" not in ids


class TestFieldLevelTracking:
    """Test field-level drift tracking."""

    def test_single_field_drift(self):
        d = DriftDetector()
        baseline = {"a": 100.0, "b": 200.0, "c": 300.0}
        current = {"a": 100.0, "b": 210.0, "c": 300.0}
        result = d.detect_drift(baseline, current)
        assert "b" in result["drifted_fields"]
        assert "a" not in result["drifted_fields"]
        assert "c" not in result["drifted_fields"]

    def test_multiple_field_drift(self):
        d = DriftDetector()
        baseline = {"a": 100.0, "b": 200.0}
        current = {"a": 110.0, "b": 220.0}
        result = d.detect_drift(baseline, current)
        assert "a" in result["drifted_fields"]
        assert "b" in result["drifted_fields"]

    def test_drift_details_contain_values(self):
        d = DriftDetector()
        baseline = {"v": 100.0}
        current = {"v": 110.0}
        result = d.detect_drift(baseline, current)
        assert "v" in result["details"]
        assert result["details"]["v"]["baseline"] == 100.0
        assert result["details"]["v"]["current"] == 110.0

    def test_string_field_change(self):
        d = DriftDetector()
        baseline = {"unit": "kg", "v": 100.0}
        current = {"unit": "tonnes", "v": 100.0}
        result = d.detect_drift(baseline, current)
        assert "unit" in result["drifted_fields"]

    def test_missing_field_drift(self):
        d = DriftDetector()
        baseline = {"a": 1, "b": 2}
        current = {"a": 1}
        result = d.detect_drift(baseline, current)
        assert "b" in result["drifted_fields"]
