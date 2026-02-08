# -*- coding: utf-8 -*-
"""
Integration Tests for Full Verification Flow (AGENT-FOUND-008)

Tests end-to-end verification workflows combining hashing, drift detection,
environment capture, seed management, version pinning, and report generation.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Shared inline components for integration testing
# ---------------------------------------------------------------------------

def _content_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class DriftSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class ArtifactHasher:
    def __init__(self):
        self._cache: Dict[str, str] = {}

    def compute_hash(self, data: Any) -> str:
        normalized = json.dumps(data, sort_keys=True, ensure_ascii=True) if isinstance(data, dict) else str(data)
        if normalized in self._cache:
            return self._cache[normalized]
        h = hashlib.sha256(normalized.encode()).hexdigest()
        self._cache[normalized] = h
        return h

    def verify_hash(self, data: Any, expected: str) -> bool:
        return self.compute_hash(data) == expected


class DriftDetector:
    def __init__(self, tolerance=1e-9, soft=0.01, hard=0.05):
        self.tolerance = tolerance
        self.soft = soft
        self.hard = hard
        self._baselines: Dict[str, Dict] = {}

    def create_baseline(self, bl_id: str, data: Dict) -> Dict:
        bl = {"baseline_id": bl_id, "data": data, "data_hash": _content_hash(data)}
        self._baselines[bl_id] = bl
        return bl

    def detect_drift(self, baseline: Dict, current: Dict) -> Dict:
        drifted = []
        max_pct = 0.0
        for k in set(list(baseline.keys()) + list(current.keys())):
            bv = baseline.get(k)
            cv = current.get(k)
            if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
                if abs(bv - cv) > self.tolerance:
                    pct = abs(cv - bv) / abs(bv) if bv != 0 else 0
                    drifted.append(k)
                    max_pct = max(max_pct, pct)
            elif bv != cv:
                drifted.append(k)
        if max_pct == 0 and not drifted:
            severity = DriftSeverity.NONE
        elif max_pct <= self.soft:
            severity = DriftSeverity.MINOR
        elif max_pct <= self.hard:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.CRITICAL
        return {
            "severity": severity.value,
            "drift_percentage": max_pct * 100,
            "drifted_fields": drifted,
            "is_acceptable": severity in (DriftSeverity.NONE, DriftSeverity.MINOR),
        }

    def compare_to_baseline(self, bl_id: str, current: Dict) -> Optional[Dict]:
        bl = self._baselines.get(bl_id)
        if bl is None:
            return None
        return self.detect_drift(bl["data"], current)


class SeedManager:
    def __init__(self):
        self._current_seed: Optional[int] = None

    def apply_seed(self, seed: int):
        self._current_seed = seed
        random.seed(seed)

    def verify_seed(self, expected: int) -> bool:
        return self._current_seed == expected


class EnvironmentCapture:
    def capture(self) -> Dict:
        return {
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
            "machine": platform.machine(),
        }

    def compare(self, e1: Dict, e2: Dict) -> bool:
        return e1.get("python_version") == e2.get("python_version") and \
               e1.get("platform") == e2.get("platform")


class VersionPinner:
    def __init__(self):
        self._manifests: Dict[str, Dict] = {}

    def pin(self, components: Dict[str, str]) -> Dict:
        mid = f"manifest-{len(self._manifests) + 1}"
        m = {"manifest_id": mid, "components": components,
             "hash": _content_hash(components)[:16]}
        self._manifests[mid] = m
        return m

    def verify(self, manifest_id: str, current: Dict[str, str]) -> bool:
        m = self._manifests.get(manifest_id)
        if m is None:
            return False
        return m["components"] == current


# ===========================================================================
# Integration Test Classes
# ===========================================================================


class TestEndToEndVerificationPass:
    """Test full verification flow with passing checks."""

    def test_hash_compute_then_verify(self):
        hasher = ArtifactHasher()
        data = {"emissions": 100.5, "fuel": "diesel"}
        h = hasher.compute_hash(data)
        assert hasher.verify_hash(data, h) is True

    def test_full_pipeline_input_output(self):
        hasher = ArtifactHasher()
        inp = {"fuel_type": "diesel", "quantity": 1000}
        out = {"total_emissions": 2680.0}
        ih = hasher.compute_hash(inp)
        oh = hasher.compute_hash(out)
        assert hasher.verify_hash(inp, ih) is True
        assert hasher.verify_hash(out, oh) is True

    def test_full_pipeline_with_environment(self):
        hasher = ArtifactHasher()
        ec = EnvironmentCapture()
        data = {"x": 1}
        h = hasher.compute_hash(data)
        env = ec.capture()
        assert hasher.verify_hash(data, h) is True
        assert "python_version" in env


class TestEndToEndVerificationFail:
    """Test full verification flow with failing checks."""

    def test_hash_mismatch_detected(self):
        hasher = ArtifactHasher()
        data = {"emissions": 100.5}
        assert hasher.verify_hash(data, "wrong_hash") is False

    def test_modified_data_hash_mismatch(self):
        hasher = ArtifactHasher()
        original = {"emissions": 100.5}
        h = hasher.compute_hash(original)
        modified = {"emissions": 101.0}
        assert hasher.verify_hash(modified, h) is False


class TestHashThenVerifyRoundtrip:
    """Test hash-then-verify roundtrip for various data shapes."""

    @pytest.mark.parametrize("data", [
        {"a": 1},
        {"nested": {"b": 2.5}},
        {"list": [1, 2, 3]},
        {"empty": {}},
        {"mixed": {"int": 1, "float": 2.5, "str": "hello", "none": None}},
    ])
    def test_roundtrip(self, data):
        hasher = ArtifactHasher()
        h = hasher.compute_hash(data)
        assert hasher.verify_hash(data, h) is True


class TestBaselineThenDetectDrift:
    """Test baseline creation followed by drift detection."""

    def test_create_baseline_then_no_drift(self):
        detector = DriftDetector()
        data = {"emissions": 100.0, "unit": "kg"}
        detector.create_baseline("bl-001", data)
        result = detector.compare_to_baseline("bl-001", data)
        assert result["severity"] == "none"
        assert result["is_acceptable"] is True

    def test_create_baseline_then_detect_drift(self):
        detector = DriftDetector()
        detector.create_baseline("bl-001", {"emissions": 100.0})
        result = detector.compare_to_baseline("bl-001", {"emissions": 110.0})
        assert result["severity"] != "none"
        assert "emissions" in result["drifted_fields"]

    def test_baseline_not_found(self):
        detector = DriftDetector()
        result = detector.compare_to_baseline("nonexistent", {"a": 1})
        assert result is None


class TestEnvironmentCaptureAndCompare:
    """Test environment capture then comparison."""

    def test_capture_and_compare_same(self):
        ec = EnvironmentCapture()
        e1 = ec.capture()
        e2 = ec.capture()
        assert ec.compare(e1, e2) is True

    def test_capture_and_compare_different(self):
        ec = EnvironmentCapture()
        e1 = ec.capture()
        e2 = {"python_version": "2.7.0", "platform": "Unknown"}
        assert ec.compare(e1, e2) is False


class TestSeedApplyThenVerify:
    """Test seed application and verification."""

    def test_seed_apply_and_verify(self):
        mgr = SeedManager()
        mgr.apply_seed(42)
        assert mgr.verify_seed(42) is True

    def test_seed_apply_deterministic_random(self):
        mgr = SeedManager()
        mgr.apply_seed(42)
        v1 = random.random()
        mgr.apply_seed(42)
        v2 = random.random()
        assert v1 == v2

    def test_seed_mismatch_detected(self):
        mgr = SeedManager()
        mgr.apply_seed(42)
        assert mgr.verify_seed(99) is False


class TestVersionPinThenVerify:
    """Test version pinning and verification."""

    def test_pin_and_verify_match(self):
        pinner = VersionPinner()
        m = pinner.pin({"a1": "1.0.0", "a2": "2.0.0"})
        assert pinner.verify(m["manifest_id"], {"a1": "1.0.0", "a2": "2.0.0"}) is True

    def test_pin_and_verify_mismatch(self):
        pinner = VersionPinner()
        m = pinner.pin({"a1": "1.0.0"})
        assert pinner.verify(m["manifest_id"], {"a1": "2.0.0"}) is False


class TestFullReplayFlow:
    """Test full replay flow: capture, replay, verify."""

    def test_capture_and_replay(self):
        hasher = ArtifactHasher()
        sm = SeedManager()
        ec = EnvironmentCapture()

        # Original execution
        sm.apply_seed(42)
        inp = {"emissions": random.uniform(99, 101)}
        ih = hasher.compute_hash(inp)
        env = ec.capture()

        # Replay
        sm.apply_seed(42)
        replay_inp = {"emissions": random.uniform(99, 101)}
        rih = hasher.compute_hash(replay_inp)
        replay_env = ec.capture()

        assert ih == rih
        assert ec.compare(env, replay_env) is True


class TestReportGenerationAfterVerification:
    """Test report generation after running verifications."""

    def test_report_after_pass(self):
        hasher = ArtifactHasher()
        data = {"x": 1}
        h = hasher.compute_hash(data)
        verified = hasher.verify_hash(data, h)
        report = {
            "status": "pass" if verified else "fail",
            "is_reproducible": verified,
            "input_hash": h,
        }
        assert report["status"] == "pass"
        assert report["is_reproducible"] is True


class TestStatisticsAfterMultipleVerifications:
    """Test statistics accumulation."""

    def test_statistics_accuracy(self):
        hasher = ArtifactHasher()
        results = []
        for i in range(10):
            data = {"value": i}
            h = hasher.compute_hash(data)
            results.append(hasher.verify_hash(data, h))
        assert sum(results) == 10

    def test_statistics_mixed_results(self):
        hasher = ArtifactHasher()
        passed = 0
        failed = 0
        for i in range(5):
            data = {"v": i}
            h = hasher.compute_hash(data)
            if hasher.verify_hash(data, h):
                passed += 1
            else:
                failed += 1
        # Force one failure
        if not hasher.verify_hash({"v": 999}, "wrong"):
            failed += 1
        assert passed == 5
        assert failed == 1
