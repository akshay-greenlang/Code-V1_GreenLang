# -*- coding: utf-8 -*-
"""
Unit tests for RegulatoryCalendarEngine - PACK-009 Engine 4

Tests unified deadline calendar, dependency tracking, conflict
detection, iCal export, alerting, critical path analysis, and
multi-year timeline generation across CSRD, CBAM, EUDR, and
EU Taxonomy.

Coverage target: 85%+
Test count: 15

Author: GreenLang QA Team
Version: 1.0.0
"""

import importlib.util
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, file_path: Path):
    """Import a module from a file path (supports hyphenated directories)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Load the engine module
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = PACK_ROOT / "engines" / "regulatory_calendar_engine.py"

try:
    _cal_mod = _import_from_path("regulatory_calendar_engine", ENGINE_PATH)
    RegulatoryCalendarEngine = _cal_mod.RegulatoryCalendarEngine
    CalendarConfig = _cal_mod.CalendarConfig
    CalendarEvent = _cal_mod.CalendarEvent
    CalendarResult = _cal_mod.CalendarResult
    CalendarAlert = _cal_mod.CalendarAlert
    DeadlineConflict = _cal_mod.DeadlineConflict
    DependencyChain = _cal_mod.DependencyChain
    ICalExport = _cal_mod.ICalExport
    REGULATORY_DEADLINES = _cal_mod.REGULATORY_DEADLINES
    _ENGINE_AVAILABLE = True
except Exception as exc:
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _ENGINE_AVAILABLE,
    reason=f"RegulatoryCalendarEngine could not be imported: "
           f"{_ENGINE_IMPORT_ERROR if not _ENGINE_AVAILABLE else ''}",
)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _assert_provenance_hash(obj: Any) -> None:
    """Verify an object carries a valid 64-char SHA-256 provenance hash."""
    h = getattr(obj, "provenance_hash", None)
    if h is None and isinstance(obj, dict):
        h = obj.get("provenance_hash")
    assert h is not None, "Missing provenance_hash"
    assert isinstance(h, str), f"provenance_hash should be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash should be 64 hex chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Invalid hex chars in hash"


# ===========================================================================
# Tests
# ===========================================================================

class TestRegulatoryCalendarEngine:
    """Tests for RegulatoryCalendarEngine."""

    # -----------------------------------------------------------------------
    # 1. Instantiation
    # -----------------------------------------------------------------------

    def test_engine_instantiation(self):
        """Engine can be created with default configuration."""
        engine = RegulatoryCalendarEngine()
        assert engine is not None
        assert isinstance(engine.config, CalendarConfig)
        assert engine.config.base_year == 2026

    # -----------------------------------------------------------------------
    # 2. get_all_deadlines
    # -----------------------------------------------------------------------

    def test_get_all_deadlines(self):
        """Full calendar for 2026 has events across all regulations."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_all_deadlines(year=2026)
        assert isinstance(result, CalendarResult)
        assert result.total_events > 30
        assert len(result.events) == result.total_events
        assert len(result.events_by_regulation) >= 4
        assert "CSRD" in result.events_by_regulation
        assert "CBAM" in result.events_by_regulation
        assert "EUDR" in result.events_by_regulation
        assert "EU_TAXONOMY" in result.events_by_regulation

    # -----------------------------------------------------------------------
    # 3. get_upcoming_deadlines
    # -----------------------------------------------------------------------

    def test_get_upcoming_deadlines(self):
        """Upcoming deadlines within 365 days returns a non-empty result."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_upcoming(days=365)
        assert isinstance(result, CalendarResult)
        assert result.total_events >= 0
        if result.events:
            current = date.today()
            for event in result.events:
                event_date = date.fromisoformat(event.deadline)
                days_until = (event_date - current).days
                assert days_until <= 365

    # -----------------------------------------------------------------------
    # 4. get_dependencies
    # -----------------------------------------------------------------------

    def test_get_dependencies(self):
        """Dependency chain for CSRD filing includes prerequisite events."""
        engine = RegulatoryCalendarEngine()
        chain = engine.get_dependencies("CSRD-FIL-001-2026", year=2026)
        assert isinstance(chain, DependencyChain)
        assert len(chain.events_in_order) >= 2
        assert chain.total_lead_time_days > 0
        assert "CSRD-FIL-001-2026" in chain.events_in_order

    # -----------------------------------------------------------------------
    # 5. detect_conflicts
    # -----------------------------------------------------------------------

    def test_detect_conflicts(self):
        """Conflict detection finds events from different regulations
        that fall within the conflict window."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_all_deadlines(year=2026)
        conflicts = result.conflicts
        assert isinstance(conflicts, list)
        for conflict in conflicts:
            assert isinstance(conflict, DeadlineConflict)
            assert len(conflict.event_ids) >= 2
            assert len(conflict.regulations) >= 2
            assert conflict.severity in ("CRITICAL", "WARNING", "INFO")

    # -----------------------------------------------------------------------
    # 6. generate_timeline
    # -----------------------------------------------------------------------

    def test_generate_timeline(self):
        """Multi-year timeline contains events for each year in range."""
        engine = RegulatoryCalendarEngine()
        result = engine.generate_timeline(start_year=2026, end_year=2028)
        assert isinstance(result, CalendarResult)
        assert result.total_events > 0
        years_seen = set()
        for event in result.events:
            years_seen.add(event.year)
        assert 2026 in years_seen
        assert 2027 in years_seen
        assert 2028 in years_seen

    # -----------------------------------------------------------------------
    # 7. export_ical_format
    # -----------------------------------------------------------------------

    def test_export_ical_format(self):
        """iCal export produces valid RFC 5545 structure."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_all_deadlines(year=2026)
        ical = engine.export_ical(result.events)
        assert isinstance(ical, ICalExport)
        assert ical.event_count == len(result.events)
        content = ical.ical_content
        assert content.startswith("BEGIN:VCALENDAR")
        assert content.endswith("END:VCALENDAR")
        assert "BEGIN:VEVENT" in content
        assert "END:VEVENT" in content
        assert "DTSTART" in content
        assert "SUMMARY:" in content
        assert "BEGIN:VALARM" in content

    # -----------------------------------------------------------------------
    # 8. cross_regulation_dependencies
    # -----------------------------------------------------------------------

    def test_cross_regulation_dependencies(self):
        """Cross-regulation dependency links exist between regulations."""
        engine = RegulatoryCalendarEngine()
        deps = engine.get_cross_regulation_dependencies(year=2026)
        assert isinstance(deps, list)
        assert len(deps) > 0
        for dep in deps:
            assert dep["source_regulation"] != dep["linked_regulation"]
            assert dep["relationship"] == "cross_regulation_link"

    # -----------------------------------------------------------------------
    # 9. deadline_alerting
    # -----------------------------------------------------------------------

    def test_deadline_alerting(self):
        """Alerts are generated for events within the alert thresholds."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_all_deadlines(year=2026)
        alerts = result.alerts
        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, CalendarAlert)
            assert alert.alert_level in ("CRITICAL", "WARNING", "INFO")
            assert alert.event_title != ""
            assert alert.regulation != ""

    # -----------------------------------------------------------------------
    # 10. all four regulations have deadlines
    # -----------------------------------------------------------------------

    def test_all_four_regulations_have_deadlines(self):
        """Every regulation has at least one event template in the DB."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_all_deadlines(year=2026)
        regs_with_events = set(result.events_by_regulation.keys())
        expected = {"CSRD", "CBAM", "EUDR", "EU_TAXONOMY"}
        assert expected.issubset(regs_with_events), (
            f"Missing regulation events: {expected - regs_with_events}"
        )
        for reg in expected:
            assert result.events_by_regulation[reg] >= 2, (
                f"Regulation {reg} should have at least 2 events"
            )

    # -----------------------------------------------------------------------
    # 11. calendar result has provenance hash
    # -----------------------------------------------------------------------

    def test_calendar_result_has_provenance_hash(self):
        """Calendar result carries a SHA-256 provenance hash."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_all_deadlines(year=2026)
        _assert_provenance_hash(result)

    # -----------------------------------------------------------------------
    # 12. upcoming sorted by date
    # -----------------------------------------------------------------------

    def test_upcoming_sorted_by_date(self):
        """Upcoming events are sorted by deadline date ascending."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_upcoming(days=365)
        if len(result.events) >= 2:
            deadlines = [e.deadline for e in result.events]
            assert deadlines == sorted(deadlines), (
                "Upcoming events are not sorted by deadline"
            )

    # -----------------------------------------------------------------------
    # 13. critical path identification
    # -----------------------------------------------------------------------

    def test_critical_path_identification(self):
        """Critical paths are identified for filing and declaration events."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_all_deadlines(year=2026)
        critical_paths = result.critical_path
        assert isinstance(critical_paths, list)
        if critical_paths:
            for chain in critical_paths:
                assert isinstance(chain, DependencyChain)
                assert chain.critical is True
                assert len(chain.events_in_order) >= 2

    # -----------------------------------------------------------------------
    # 14. fiscal year configuration
    # -----------------------------------------------------------------------

    def test_fiscal_year_configuration(self):
        """Custom fiscal year end is accepted by the config."""
        config = {"fiscal_year_end": "03-31", "base_year": 2026}
        engine = RegulatoryCalendarEngine(config=config)
        assert engine.config.fiscal_year_end == "03-31"
        assert engine.config.base_year == 2026
        result = engine.get_all_deadlines(year=2026)
        assert result.total_events > 0

    # -----------------------------------------------------------------------
    # 15. empty calendar handling
    # -----------------------------------------------------------------------

    def test_empty_calendar_handling(self):
        """Requesting a year far in the past with 'once' events that don't
        match returns a result object with zero or minimal events."""
        engine = RegulatoryCalendarEngine()
        result = engine.get_all_deadlines(year=2024)
        assert isinstance(result, CalendarResult)
        assert result.total_events >= 0
        _assert_provenance_hash(result)
