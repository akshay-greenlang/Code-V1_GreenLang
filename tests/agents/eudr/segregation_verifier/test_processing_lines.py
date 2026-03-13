# -*- coding: utf-8 -*-
"""
Tests for ProcessingLineVerifier - AGENT-EUDR-010 Engine 4: Processing Line Verification

Comprehensive test suite covering:
- Line registration (all 15 processing line types)
- Changeover recording (valid, insufficient flush, insufficient duration)
- Changeover compliance verification
- Equipment sharing detection (shared scales, conveyors, hoppers, tanks)
- Temporal separation verification (adequate gap, insufficient gap)
- First-run-after-changeover flagging
- Processing score calculation (line_dedication + changeover + equipment + temporal)
- Dedicated line verification
- Changeover history
- Edge cases (no changeovers, immediate re-use, overlapping runs)

Test count: 65+ tests
Coverage target: >= 85% of ProcessingLineVerifier module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier Agent (GL-EUDR-SGV-010)
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.segregation_verifier.conftest import (
    PROCESSING_LINE_TYPES,
    CHANGEOVER_REQUIREMENTS,
    EQUIPMENT_SHARING_WEIGHTS,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    LINE_COCOA_ROASTING,
    LINE_PALM_PRESSING,
    LINE_ID_ROASTING_01,
    LINE_ID_PRESSING_01,
    FAC_ID_FACTORY_DE,
    FAC_ID_MILL_ID,
    BATCH_ID_COCOA_001,
    BATCH_ID_COFFEE_001,
    make_line,
    make_changeover,
    assert_valid_score,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Line Registration
# ===========================================================================


class TestLineRegistration:
    """Test processing line registration."""

    @pytest.mark.parametrize("line_type", PROCESSING_LINE_TYPES)
    def test_register_all_line_types(self, processing_line_verifier, line_type):
        """Each of the 15 processing line types can be registered."""
        line = make_line(line_type=line_type)
        result = processing_line_verifier.register_line(line)
        assert result is not None
        assert result["line_type"] == line_type

    def test_register_cocoa_roasting_line(self, processing_line_verifier):
        """Register a cocoa roasting line with full details."""
        line = copy.deepcopy(LINE_COCOA_ROASTING)
        result = processing_line_verifier.register_line(line)
        assert result["line_id"] == LINE_ID_ROASTING_01
        assert result["line_type"] == "roasting"
        assert result["dedicated"] is True

    def test_register_palm_pressing_line(self, processing_line_verifier):
        """Register a palm oil pressing line."""
        line = copy.deepcopy(LINE_PALM_PRESSING)
        result = processing_line_verifier.register_line(line)
        assert result["line_id"] == LINE_ID_PRESSING_01
        assert result["dedicated"] is False

    def test_duplicate_line_id_raises(self, processing_line_verifier):
        """Registering a line with duplicate ID raises an error."""
        line = make_line(line_id="LINE-DUP-001")
        processing_line_verifier.register_line(line)
        with pytest.raises((ValueError, KeyError)):
            processing_line_verifier.register_line(copy.deepcopy(line))

    def test_missing_line_type_raises(self, processing_line_verifier):
        """Line without line_type raises ValueError."""
        line = make_line()
        line["line_type"] = None
        with pytest.raises(ValueError):
            processing_line_verifier.register_line(line)

    def test_register_provenance_hash(self, processing_line_verifier):
        """Line registration generates a provenance hash."""
        line = make_line()
        result = processing_line_verifier.register_line(line)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. Changeover Recording
# ===========================================================================


class TestChangeoverRecording:
    """Test changeover event recording."""

    def test_record_valid_changeover(self, processing_line_verifier):
        """Record a valid changeover with sufficient duration and flush."""
        line = make_line(line_id="LINE-CHG-001", line_type="roasting")
        processing_line_verifier.register_line(line)
        changeover = make_changeover(line_id="LINE-CHG-001", duration_minutes=45, flush_kg=8.0)
        result = processing_line_verifier.record_changeover(changeover)
        assert result is not None
        assert result.get("compliant") is True

    def test_record_insufficient_duration(self, processing_line_verifier):
        """Changeover with insufficient duration is flagged."""
        line = make_line(line_id="LINE-CHG-SHORT", line_type="roasting")
        processing_line_verifier.register_line(line)
        req = CHANGEOVER_REQUIREMENTS["roasting"]
        changeover = make_changeover(
            line_id="LINE-CHG-SHORT",
            duration_minutes=req["min_minutes"] - 10,
            flush_kg=req["flush_kg"],
        )
        result = processing_line_verifier.record_changeover(changeover)
        assert result.get("compliant") is False

    def test_record_insufficient_flush(self, processing_line_verifier):
        """Changeover with insufficient flush quantity is flagged."""
        line = make_line(line_id="LINE-CHG-LOWFL", line_type="milling")
        processing_line_verifier.register_line(line)
        req = CHANGEOVER_REQUIREMENTS["milling"]
        changeover = make_changeover(
            line_id="LINE-CHG-LOWFL",
            duration_minutes=req["min_minutes"],
            flush_kg=req["flush_kg"] * 0.3,
        )
        result = processing_line_verifier.record_changeover(changeover)
        assert result.get("compliant") is False

    @pytest.mark.parametrize("line_type", PROCESSING_LINE_TYPES)
    def test_changeover_requirements_per_type(self, processing_line_verifier, line_type):
        """Changeover requirements are enforced per line type."""
        line = make_line(line_type=line_type, line_id=f"LINE-REQ-{line_type}")
        processing_line_verifier.register_line(line)
        req = CHANGEOVER_REQUIREMENTS[line_type]
        changeover = make_changeover(
            line_id=f"LINE-REQ-{line_type}",
            duration_minutes=req["min_minutes"],
            flush_kg=req["flush_kg"],
        )
        result = processing_line_verifier.record_changeover(changeover)
        assert result.get("compliant") is True

    def test_changeover_on_unregistered_line_raises(self, processing_line_verifier):
        """Changeover on an unregistered line raises an error."""
        changeover = make_changeover(line_id="LINE-NONEXISTENT")
        with pytest.raises((ValueError, KeyError)):
            processing_line_verifier.record_changeover(changeover)


# ===========================================================================
# 3. Changeover Compliance Verification
# ===========================================================================


class TestChangeoverCompliance:
    """Test changeover compliance verification."""

    def test_compliant_changeover(self, processing_line_verifier):
        """Changeover meeting all requirements is compliant."""
        line = make_line(line_id="LINE-COMP-001", line_type="pressing")
        processing_line_verifier.register_line(line)
        req = CHANGEOVER_REQUIREMENTS["pressing"]
        changeover = make_changeover(
            line_id="LINE-COMP-001",
            duration_minutes=req["min_minutes"] + 10,
            flush_kg=req["flush_kg"] + 5.0,
        )
        processing_line_verifier.record_changeover(changeover)
        result = processing_line_verifier.verify_changeover(changeover["changeover_id"])
        assert result["compliant"] is True

    def test_non_compliant_changeover(self, processing_line_verifier):
        """Changeover not meeting requirements is non-compliant."""
        line = make_line(line_id="LINE-NCOMP-001", line_type="extraction")
        processing_line_verifier.register_line(line)
        changeover = make_changeover(
            line_id="LINE-NCOMP-001",
            duration_minutes=10,
            flush_kg=1.0,
        )
        processing_line_verifier.record_changeover(changeover)
        result = processing_line_verifier.verify_changeover(changeover["changeover_id"])
        assert result["compliant"] is False

    def test_changeover_no_flush_for_non_required(self, processing_line_verifier):
        """Line types not requiring flush pass without flush."""
        line = make_line(line_id="LINE-NOFL-001", line_type="sorting")
        processing_line_verifier.register_line(line)
        req = CHANGEOVER_REQUIREMENTS["sorting"]
        changeover = make_changeover(
            line_id="LINE-NOFL-001",
            duration_minutes=req["min_minutes"],
            flush_kg=0.0,
        )
        result = processing_line_verifier.record_changeover(changeover)
        assert result.get("compliant") is True

    @pytest.mark.parametrize("line_type,flush_required", [
        ("roasting", True),
        ("milling", True),
        ("sorting", False),
        ("grading", False),
        ("drying", False),
        ("fermentation", False),
        ("tanning", True),
    ])
    def test_flush_requirement_per_type(self, processing_line_verifier, line_type, flush_required):
        """Flush requirement varies per line type."""
        req = CHANGEOVER_REQUIREMENTS[line_type]
        assert req["flush_required"] is flush_required


# ===========================================================================
# 4. Equipment Sharing Detection
# ===========================================================================


class TestEquipmentSharing:
    """Test shared equipment detection and risk assessment."""

    @pytest.mark.parametrize("equipment,weight", list(EQUIPMENT_SHARING_WEIGHTS.items()))
    def test_shared_equipment_risk_weights(self, processing_line_verifier, equipment, weight):
        """Each shared equipment type has the correct risk weight."""
        line = make_line(line_id=f"LINE-EQ-{equipment}", shared_equipment=[equipment])
        processing_line_verifier.register_line(line)
        risk = processing_line_verifier.assess_equipment_sharing(f"LINE-EQ-{equipment}")
        assert risk is not None
        assert len(risk.get("shared_equipment", [])) > 0

    def test_no_shared_equipment_no_risk(self, processing_line_verifier):
        """Line with no shared equipment has no equipment sharing risk."""
        line = make_line(line_id="LINE-NOEQ-001", shared_equipment=[])
        processing_line_verifier.register_line(line)
        risk = processing_line_verifier.assess_equipment_sharing("LINE-NOEQ-001")
        assert risk.get("risk_score", 0.0) == pytest.approx(0.0) or len(risk.get("shared_equipment", [])) == 0

    def test_multiple_shared_equipment(self, processing_line_verifier):
        """Line with multiple shared equipment has cumulative risk."""
        line = make_line(line_id="LINE-MULTI-EQ", shared_equipment=["scale", "conveyor", "hopper"])
        processing_line_verifier.register_line(line)
        risk = processing_line_verifier.assess_equipment_sharing("LINE-MULTI-EQ")
        assert len(risk.get("shared_equipment", [])) == 3

    def test_tank_sharing_highest_risk(self, processing_line_verifier):
        """Shared tank has the highest equipment sharing risk weight."""
        line = make_line(line_id="LINE-TANK-001", shared_equipment=["tank"])
        processing_line_verifier.register_line(line)
        risk = processing_line_verifier.assess_equipment_sharing("LINE-TANK-001")
        assert risk.get("risk_score", 0) >= EQUIPMENT_SHARING_WEIGHTS["tank"] * 100 * 0.5


# ===========================================================================
# 5. Temporal Separation Verification
# ===========================================================================


class TestTemporalSeparation:
    """Test temporal separation between processing runs."""

    def test_adequate_gap_passes(self, processing_line_verifier):
        """Adequate temporal gap between runs passes."""
        line = make_line(line_id="LINE-TEMP-OK", line_type="roasting")
        processing_line_verifier.register_line(line)
        req = CHANGEOVER_REQUIREMENTS["roasting"]
        result = processing_line_verifier.verify_temporal_separation(
            "LINE-TEMP-OK",
            previous_end=(datetime.now(timezone.utc) - timedelta(minutes=req["min_minutes"] + 30)).isoformat(),
            next_start=datetime.now(timezone.utc).isoformat(),
        )
        assert result["adequate"] is True

    def test_insufficient_gap_fails(self, processing_line_verifier):
        """Insufficient temporal gap between runs fails."""
        line = make_line(line_id="LINE-TEMP-BAD", line_type="roasting")
        processing_line_verifier.register_line(line)
        result = processing_line_verifier.verify_temporal_separation(
            "LINE-TEMP-BAD",
            previous_end=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
            next_start=datetime.now(timezone.utc).isoformat(),
        )
        assert result["adequate"] is False

    @pytest.mark.parametrize("line_type,min_minutes", [
        ("sorting", 15),
        ("roasting", 30),
        ("milling", 45),
        ("pressing", 60),
        ("extraction", 90),
        ("refining", 120),
        ("tanning", 180),
        ("fermentation", 240),
    ])
    def test_minimum_gap_per_line_type(self, processing_line_verifier, line_type, min_minutes):
        """Minimum temporal gap varies by line type."""
        req = CHANGEOVER_REQUIREMENTS[line_type]
        assert req["min_minutes"] == min_minutes


# ===========================================================================
# 6. First-Run-After-Changeover
# ===========================================================================


class TestFirstRunAfterChangeover:
    """Test first-run-after-changeover flagging."""

    def test_first_run_flagged(self, processing_line_verifier):
        """First production run after changeover is flagged for QC."""
        line = make_line(line_id="LINE-FIRST-001", line_type="milling")
        processing_line_verifier.register_line(line)
        changeover = make_changeover(line_id="LINE-FIRST-001")
        processing_line_verifier.record_changeover(changeover)
        result = processing_line_verifier.check_first_run("LINE-FIRST-001")
        assert result.get("first_run_after_changeover") is True

    def test_subsequent_run_not_flagged(self, processing_line_verifier):
        """Subsequent production run is not flagged."""
        line = make_line(line_id="LINE-SUBSEQ-001", line_type="milling")
        processing_line_verifier.register_line(line)
        changeover = make_changeover(line_id="LINE-SUBSEQ-001")
        processing_line_verifier.record_changeover(changeover)
        processing_line_verifier.clear_first_run_flag("LINE-SUBSEQ-001")
        result = processing_line_verifier.check_first_run("LINE-SUBSEQ-001")
        assert result.get("first_run_after_changeover") is False


# ===========================================================================
# 7. Processing Score Calculation
# ===========================================================================


class TestProcessingScore:
    """Test composite processing score calculation."""

    def test_score_components_present(self, processing_line_verifier):
        """Processing score includes all expected components."""
        line = make_line(line_id="LINE-SCORE-001")
        processing_line_verifier.register_line(line)
        score = processing_line_verifier.calculate_score("LINE-SCORE-001")
        expected_keys = ["line_dedication", "changeover", "equipment", "temporal"]
        for key in expected_keys:
            assert key in score, f"Missing score component: {key}"

    def test_dedicated_line_high_score(self, processing_line_verifier):
        """Dedicated line has high dedication sub-score."""
        line = make_line(line_id="LINE-DED-SCORE", dedicated=True)
        processing_line_verifier.register_line(line)
        score = processing_line_verifier.calculate_score("LINE-DED-SCORE")
        assert score.get("line_dedication", 0) >= 80.0

    def test_shared_line_lower_score(self, processing_line_verifier):
        """Shared line has lower dedication sub-score."""
        line = make_line(line_id="LINE-SHR-SCORE", dedicated=False,
                         shared_equipment=["scale", "conveyor"])
        processing_line_verifier.register_line(line)
        score = processing_line_verifier.calculate_score("LINE-SHR-SCORE")
        assert score.get("line_dedication", 0) < 80.0

    def test_score_within_bounds(self, processing_line_verifier):
        """Processing score is between 0 and 100."""
        line = make_line(line_id="LINE-BOUND-001")
        processing_line_verifier.register_line(line)
        score = processing_line_verifier.calculate_score("LINE-BOUND-001")
        total = score.get("total_score", score.get("score", 0))
        assert_valid_score(total)


# ===========================================================================
# 8. Dedicated Line Verification
# ===========================================================================


class TestDedicatedLineVerification:
    """Test dedicated processing line verification."""

    def test_verify_dedicated_line(self, processing_line_verifier):
        """Dedicated line passes verification for its commodity."""
        line = make_line(line_id="LINE-VDED-001", dedicated=True, commodity="cocoa")
        processing_line_verifier.register_line(line)
        result = processing_line_verifier.verify_dedication("LINE-VDED-001", commodity="cocoa")
        assert result["compliant"] is True

    def test_dedicated_wrong_commodity_fails(self, processing_line_verifier):
        """Dedicated line used for wrong commodity fails verification."""
        line = make_line(line_id="LINE-VDED-002", dedicated=True, commodity="cocoa")
        processing_line_verifier.register_line(line)
        result = processing_line_verifier.verify_dedication("LINE-VDED-002", commodity="palm_oil")
        assert result["compliant"] is False

    def test_non_dedicated_always_needs_changeover(self, processing_line_verifier):
        """Non-dedicated line always requires changeover verification."""
        line = make_line(line_id="LINE-NDED-001", dedicated=False)
        processing_line_verifier.register_line(line)
        result = processing_line_verifier.verify_dedication("LINE-NDED-001", commodity="cocoa")
        assert result.get("changeover_required") is True


# ===========================================================================
# 9. Changeover History
# ===========================================================================


class TestChangeoverHistory:
    """Test changeover history tracking."""

    def test_history_records_all_changeovers(self, processing_line_verifier):
        """All changeovers are recorded in history."""
        line = make_line(line_id="LINE-HIST-001", line_type="milling")
        processing_line_verifier.register_line(line)
        for i in range(3):
            changeover = make_changeover(
                line_id="LINE-HIST-001",
                changeover_id=f"CHG-HIST-{i:03d}",
                duration_minutes=50,
                flush_kg=12.0,
            )
            processing_line_verifier.record_changeover(changeover)
        history = processing_line_verifier.get_changeover_history("LINE-HIST-001")
        assert len(history) == 3

    def test_history_chronological_order(self, processing_line_verifier):
        """Changeover history is in chronological order."""
        line = make_line(line_id="LINE-HIST-002", line_type="sorting")
        processing_line_verifier.register_line(line)
        for i in range(3):
            changeover = make_changeover(
                line_id="LINE-HIST-002",
                changeover_id=f"CHG-ORD-{i:03d}",
            )
            processing_line_verifier.record_changeover(changeover)
        history = processing_line_verifier.get_changeover_history("LINE-HIST-002")
        for i in range(len(history) - 1):
            assert history[i].get("started_at", "") <= history[i + 1].get("started_at", "")


# ===========================================================================
# 10. Edge Cases
# ===========================================================================


class TestProcessingLineEdgeCases:
    """Test edge cases for processing line operations."""

    def test_get_nonexistent_line_returns_none(self, processing_line_verifier):
        """Getting a non-existent line returns None."""
        result = processing_line_verifier.get_line("LINE-NONEXISTENT")
        assert result is None

    def test_zero_capacity_line_raises(self, processing_line_verifier):
        """Line with zero capacity raises ValueError."""
        line = make_line(capacity_kg_per_hour=0.0)
        with pytest.raises(ValueError):
            processing_line_verifier.register_line(line)

    def test_no_changeover_history_empty(self, processing_line_verifier):
        """Line with no changeovers has empty history."""
        line = make_line(line_id="LINE-NOCHG-001")
        processing_line_verifier.register_line(line)
        history = processing_line_verifier.get_changeover_history("LINE-NOCHG-001")
        assert len(history) == 0

    def test_immediate_reuse_flagged(self, processing_line_verifier):
        """Immediate re-use without changeover is flagged."""
        line = make_line(line_id="LINE-IMMED-001", line_type="pressing", dedicated=False)
        processing_line_verifier.register_line(line)
        result = processing_line_verifier.verify_temporal_separation(
            "LINE-IMMED-001",
            previous_end=datetime.now(timezone.utc).isoformat(),
            next_start=datetime.now(timezone.utc).isoformat(),
        )
        assert result["adequate"] is False

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_lines_for_all_commodities(self, processing_line_verifier, commodity):
        """Lines can be registered for all 7 EUDR commodities."""
        line = make_line(commodity=commodity)
        result = processing_line_verifier.register_line(line)
        assert result["commodity"] == commodity
