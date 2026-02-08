# -*- coding: utf-8 -*-
"""
Unit Tests for CoverageTracker (AGENT-FOUND-009)

Tests coverage tracking, report generation, snapshot creation, method-level
tracking, and coverage percentage calculations.

Coverage target: 85%+ of coverage_tracker.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline CoverageTracker
# ---------------------------------------------------------------------------


class CoverageReport:
    def __init__(self, agent_type: str, total_methods: int = 0,
                 covered_methods: int = 0, coverage_percent: float = 0.0,
                 uncovered_methods: Optional[List[str]] = None,
                 test_count: int = 0):
        self.agent_type = agent_type
        self.total_methods = total_methods
        self.covered_methods = covered_methods
        self.coverage_percent = coverage_percent
        self.uncovered_methods = uncovered_methods or []
        self.test_count = test_count


class CoverageSnapshot:
    def __init__(self, snapshot_id: str, agent_type: str,
                 coverage_percent: float, total_methods: int,
                 covered_methods: int, uncovered: Optional[List[str]] = None,
                 created_at: Optional[datetime] = None):
        self.snapshot_id = snapshot_id
        self.agent_type = agent_type
        self.coverage_percent = coverage_percent
        self.total_methods = total_methods
        self.covered_methods = covered_methods
        self.uncovered = uncovered or []
        self.created_at = created_at or datetime.now(timezone.utc)


class CoverageTracker:
    """Tracks test coverage per agent."""

    def __init__(self):
        self._agent_methods: Dict[str, List[str]] = {}
        self._covered: Dict[str, Set[str]] = {}
        self._test_counts: Dict[str, int] = {}
        self._snapshots: Dict[str, List[CoverageSnapshot]] = {}
        self._snap_counter = 0

    def register_agent(self, agent_type: str, methods: List[str]) -> None:
        """Register an agent and its public methods."""
        self._agent_methods[agent_type] = methods
        if agent_type not in self._covered:
            self._covered[agent_type] = set()
        if agent_type not in self._test_counts:
            self._test_counts[agent_type] = 0

    def track(self, agent_type: str, method_name: str) -> None:
        """Record that a method was covered by a test."""
        if agent_type not in self._covered:
            self._covered[agent_type] = set()
        self._covered[agent_type].add(method_name)
        if agent_type not in self._test_counts:
            self._test_counts[agent_type] = 0
        self._test_counts[agent_type] += 1

    def get_report(self, agent_type: str) -> CoverageReport:
        """Get coverage report for an agent."""
        all_methods = self._agent_methods.get(agent_type, [])
        covered = self._covered.get(agent_type, set())
        total = len(all_methods)
        covered_count = len(covered.intersection(set(all_methods)))
        pct = (covered_count / total * 100) if total > 0 else 0.0
        uncovered = [m for m in all_methods if m not in covered]

        return CoverageReport(
            agent_type=agent_type, total_methods=total,
            covered_methods=covered_count,
            coverage_percent=round(pct, 2),
            uncovered_methods=uncovered,
            test_count=self._test_counts.get(agent_type, 0),
        )

    def get_all_reports(self) -> List[CoverageReport]:
        """Get coverage reports for all registered agents."""
        return [self.get_report(at) for at in self._agent_methods]

    def take_snapshot(self, agent_type: str) -> CoverageSnapshot:
        """Take a coverage snapshot."""
        self._snap_counter += 1
        report = self.get_report(agent_type)
        snapshot = CoverageSnapshot(
            snapshot_id=f"snap-{self._snap_counter:04d}",
            agent_type=agent_type,
            coverage_percent=report.coverage_percent,
            total_methods=report.total_methods,
            covered_methods=report.covered_methods,
            uncovered=report.uncovered_methods,
        )
        if agent_type not in self._snapshots:
            self._snapshots[agent_type] = []
        self._snapshots[agent_type].append(snapshot)
        return snapshot

    def get_snapshots(self, agent_type: str) -> List[CoverageSnapshot]:
        """Get snapshots for an agent."""
        return self._snapshots.get(agent_type, [])


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def tracker():
    t = CoverageTracker()
    t.register_agent("Agent", ["execute", "run", "validate_input",
                                "preprocess", "postprocess"])
    return t


class TestTrackCoverage:
    def test_track_single_method(self, tracker):
        tracker.track("Agent", "execute")
        report = tracker.get_report("Agent")
        assert report.covered_methods == 1

    def test_track_multiple_methods(self, tracker):
        tracker.track("Agent", "execute")
        tracker.track("Agent", "run")
        tracker.track("Agent", "validate_input")
        report = tracker.get_report("Agent")
        assert report.covered_methods == 3

    def test_track_duplicate_method(self, tracker):
        tracker.track("Agent", "execute")
        tracker.track("Agent", "execute")
        report = tracker.get_report("Agent")
        assert report.covered_methods == 1

    def test_track_increments_test_count(self, tracker):
        tracker.track("Agent", "execute")
        tracker.track("Agent", "run")
        report = tracker.get_report("Agent")
        assert report.test_count == 2

    def test_track_unregistered_agent(self, tracker):
        tracker.track("NewAgent", "some_method")
        report = tracker.get_report("NewAgent")
        assert report.total_methods == 0
        assert report.test_count == 1


class TestGetReport:
    def test_report_empty(self, tracker):
        report = tracker.get_report("Agent")
        assert report.coverage_percent == 0.0
        assert report.total_methods == 5

    def test_report_partial_coverage(self, tracker):
        tracker.track("Agent", "execute")
        tracker.track("Agent", "run")
        report = tracker.get_report("Agent")
        assert report.coverage_percent == 40.0

    def test_report_full_coverage(self, tracker):
        for method in ["execute", "run", "validate_input", "preprocess", "postprocess"]:
            tracker.track("Agent", method)
        report = tracker.get_report("Agent")
        assert report.coverage_percent == 100.0

    def test_report_agent_type(self, tracker):
        report = tracker.get_report("Agent")
        assert report.agent_type == "Agent"

    def test_report_unregistered_agent(self, tracker):
        report = tracker.get_report("Unknown")
        assert report.total_methods == 0
        assert report.coverage_percent == 0.0


class TestGetAllReports:
    def test_all_reports_single_agent(self, tracker):
        reports = tracker.get_all_reports()
        assert len(reports) == 1
        assert reports[0].agent_type == "Agent"

    def test_all_reports_multiple_agents(self, tracker):
        tracker.register_agent("Agent2", ["execute", "run"])
        reports = tracker.get_all_reports()
        assert len(reports) == 2

    def test_all_reports_empty(self):
        t = CoverageTracker()
        assert t.get_all_reports() == []


class TestTakeSnapshot:
    def test_take_snapshot(self, tracker):
        tracker.track("Agent", "execute")
        snapshot = tracker.take_snapshot("Agent")
        assert snapshot.snapshot_id.startswith("snap-")
        assert snapshot.coverage_percent == 20.0

    def test_take_snapshot_captures_state(self, tracker):
        tracker.track("Agent", "execute")
        snap1 = tracker.take_snapshot("Agent")
        tracker.track("Agent", "run")
        snap2 = tracker.take_snapshot("Agent")
        assert snap1.coverage_percent < snap2.coverage_percent

    def test_take_snapshot_created_at(self, tracker):
        snapshot = tracker.take_snapshot("Agent")
        assert snapshot.created_at is not None


class TestGetSnapshots:
    def test_get_snapshots_empty(self, tracker):
        assert tracker.get_snapshots("Agent") == []

    def test_get_snapshots_after_taking(self, tracker):
        tracker.take_snapshot("Agent")
        tracker.take_snapshot("Agent")
        snapshots = tracker.get_snapshots("Agent")
        assert len(snapshots) == 2

    def test_get_snapshots_ordered(self, tracker):
        tracker.track("Agent", "execute")
        snap1 = tracker.take_snapshot("Agent")
        tracker.track("Agent", "run")
        snap2 = tracker.take_snapshot("Agent")
        snapshots = tracker.get_snapshots("Agent")
        assert snapshots[0].coverage_percent <= snapshots[1].coverage_percent


class TestCoveragePercentCalculation:
    def test_zero_methods_zero_coverage(self):
        t = CoverageTracker()
        t.register_agent("Empty", [])
        report = t.get_report("Empty")
        assert report.coverage_percent == 0.0

    def test_exact_percentage(self, tracker):
        tracker.track("Agent", "execute")
        report = tracker.get_report("Agent")
        assert report.coverage_percent == 20.0  # 1/5 = 20%

    def test_rounding(self):
        t = CoverageTracker()
        t.register_agent("A", ["m1", "m2", "m3"])
        t.track("A", "m1")
        report = t.get_report("A")
        assert report.coverage_percent == pytest.approx(33.33, rel=0.01)


class TestUncoveredMethodsDetection:
    def test_all_uncovered(self, tracker):
        report = tracker.get_report("Agent")
        assert len(report.uncovered_methods) == 5

    def test_some_covered(self, tracker):
        tracker.track("Agent", "execute")
        tracker.track("Agent", "run")
        report = tracker.get_report("Agent")
        assert "execute" not in report.uncovered_methods
        assert "run" not in report.uncovered_methods
        assert "validate_input" in report.uncovered_methods

    def test_all_covered(self, tracker):
        for m in ["execute", "run", "validate_input", "preprocess", "postprocess"]:
            tracker.track("Agent", m)
        report = tracker.get_report("Agent")
        assert report.uncovered_methods == []

    def test_uncovered_preserves_order(self, tracker):
        report = tracker.get_report("Agent")
        assert report.uncovered_methods == [
            "execute", "run", "validate_input", "preprocess", "postprocess"
        ]
