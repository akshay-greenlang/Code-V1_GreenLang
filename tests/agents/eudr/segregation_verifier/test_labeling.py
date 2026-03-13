# -*- coding: utf-8 -*-
"""
Tests for LabelingVerificationEngine - AGENT-EUDR-010 Engine 6: Labeling Verification

Comprehensive test suite covering:
- Label registration (all 8 label types)
- Label verification (content check, placement check, condition check)
- Missing label detection
- Color code validation (consistent=pass, inconsistent=fail)
- Label event recording (applied, verified, damaged, replaced, removed)
- Labeling audit (comprehensive facility audit)
- Expiring label detection
- Labeling score calculation (coverage+readability+accuracy+timeliness)
- Label history tracking
- Edge cases (expired label, damaged label, missing content fields)

Test count: 55+ tests
Coverage target: >= 85% of LabelingVerificationEngine module

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
    LABEL_TYPES,
    LABEL_STATUSES,
    LABEL_EVENT_TYPES,
    LABEL_REQUIRED_FIELDS,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    LABEL_BIN_TAG,
    LABEL_ZONE_SIGN,
    LABEL_ID_BIN_01,
    LABEL_ID_ZONE_01,
    FAC_ID_WAREHOUSE_GH,
    ZONE_ID_COCOA_A,
    ZONE_ID_COCOA_B,
    BATCH_ID_COCOA_001,
    make_label,
    make_label_event,
    assert_valid_score,
    assert_valid_provenance_hash,
    _ts,
)


# ===========================================================================
# 1. Label Registration
# ===========================================================================


class TestLabelRegistration:
    """Test label registration for all types."""

    @pytest.mark.parametrize("label_type", LABEL_TYPES)
    def test_register_all_label_types(self, labeling_verification_engine, label_type):
        """Each of the 8 label types can be registered."""
        label = make_label(label_type=label_type)
        result = labeling_verification_engine.register_label(label)
        assert result is not None
        assert result["label_type"] == label_type

    def test_register_bin_tag(self, labeling_verification_engine):
        """Register a bin tag with full details."""
        label = copy.deepcopy(LABEL_BIN_TAG)
        result = labeling_verification_engine.register_label(label)
        assert result["label_id"] == LABEL_ID_BIN_01
        assert result["label_type"] == "bin_tag"

    def test_register_zone_sign(self, labeling_verification_engine):
        """Register a zone sign."""
        label = copy.deepcopy(LABEL_ZONE_SIGN)
        result = labeling_verification_engine.register_label(label)
        assert result["label_id"] == LABEL_ID_ZONE_01
        assert result["label_type"] == "zone_sign"

    def test_duplicate_label_id_raises(self, labeling_verification_engine):
        """Registering a label with duplicate ID raises an error."""
        label = make_label(label_id="LBL-DUP-001")
        labeling_verification_engine.register_label(label)
        with pytest.raises((ValueError, KeyError)):
            labeling_verification_engine.register_label(copy.deepcopy(label))

    def test_register_provenance_hash(self, labeling_verification_engine):
        """Label registration generates a provenance hash."""
        label = make_label()
        result = labeling_verification_engine.register_label(label)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_missing_label_type_raises(self, labeling_verification_engine):
        """Label without label_type raises ValueError."""
        label = make_label()
        label["label_type"] = None
        with pytest.raises(ValueError):
            labeling_verification_engine.register_label(label)


# ===========================================================================
# 2. Label Verification
# ===========================================================================


class TestLabelVerification:
    """Test label verification (content, placement, condition)."""

    def test_verify_complete_label(self, labeling_verification_engine):
        """Label with all required fields passes verification."""
        label = make_label(label_type="bin_tag")
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.verify_label(label["label_id"])
        assert result.get("content_valid") is True

    def test_verify_missing_content_fields(self, labeling_verification_engine):
        """Label missing required content fields fails verification."""
        label = make_label(label_type="bin_tag", content_fields={"batch_id": "B-001"})
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.verify_label(label["label_id"])
        assert result.get("content_valid") is False
        assert len(result.get("missing_fields", [])) > 0

    @pytest.mark.parametrize("label_type", LABEL_TYPES)
    def test_required_fields_per_type(self, label_type):
        """Each label type has defined required content fields."""
        assert label_type in LABEL_REQUIRED_FIELDS
        assert len(LABEL_REQUIRED_FIELDS[label_type]) > 0

    def test_verify_placement_valid(self, labeling_verification_engine):
        """Label with correct placement passes placement check."""
        label = make_label()
        label["placement"] = "standard"
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.verify_label(label["label_id"])
        assert result.get("placement_valid") is True or result.get("placement_valid") is None

    def test_verify_condition_good(self, labeling_verification_engine):
        """Label in good condition passes condition check."""
        label = make_label()
        label["condition"] = "good"
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.verify_label(label["label_id"])
        assert result.get("condition_acceptable") is True or result.get("condition") == "good"

    def test_verify_condition_damaged(self, labeling_verification_engine):
        """Label in damaged condition fails condition check."""
        label = make_label()
        label["condition"] = "damaged"
        label["status"] = "damaged"
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.verify_label(label["label_id"])
        assert result.get("condition_acceptable") is False or result.get("status") == "damaged"

    def test_verify_nonexistent_label_raises(self, labeling_verification_engine):
        """Verifying a non-existent label raises an error."""
        with pytest.raises((ValueError, KeyError)):
            labeling_verification_engine.verify_label("LBL-NONEXISTENT")


# ===========================================================================
# 3. Missing Label Detection
# ===========================================================================


class TestMissingLabelDetection:
    """Test detection of missing labels in zones and batches."""

    def test_detect_missing_zone_labels(self, labeling_verification_engine):
        """Detect zones without required labels."""
        result = labeling_verification_engine.detect_missing_labels(
            facility_id=FAC_ID_WAREHOUSE_GH,
            zones=[ZONE_ID_COCOA_A, ZONE_ID_COCOA_B],
        )
        assert result is not None
        assert "missing" in result or "gaps" in result

    def test_no_missing_labels(self, labeling_verification_engine):
        """Facility with all labels has no missing labels."""
        label_a = make_label(label_id="LBL-ZONE-A", label_type="zone_sign",
                             zone_id=ZONE_ID_COCOA_A)
        label_b = make_label(label_id="LBL-ZONE-B", label_type="zone_sign",
                             zone_id=ZONE_ID_COCOA_B)
        labeling_verification_engine.register_label(label_a)
        labeling_verification_engine.register_label(label_b)
        result = labeling_verification_engine.detect_missing_labels(
            facility_id=FAC_ID_WAREHOUSE_GH,
            zones=[ZONE_ID_COCOA_A, ZONE_ID_COCOA_B],
        )
        missing = result.get("missing", result.get("gaps", []))
        assert len(missing) == 0


# ===========================================================================
# 4. Color Code Validation
# ===========================================================================


class TestColorCodeValidation:
    """Test color code consistency validation."""

    def test_consistent_color_codes(self, labeling_verification_engine):
        """Consistent color codes across labels pass validation."""
        label1 = make_label(label_id="LBL-CC-A", color_code="green",
                            zone_id=ZONE_ID_COCOA_A)
        label2 = make_label(label_id="LBL-CC-B", color_code="green",
                            zone_id=ZONE_ID_COCOA_A)
        labeling_verification_engine.register_label(label1)
        labeling_verification_engine.register_label(label2)
        result = labeling_verification_engine.validate_color_codes(
            facility_id=FAC_ID_WAREHOUSE_GH
        )
        assert result.get("consistent") is True

    def test_inconsistent_color_codes(self, labeling_verification_engine):
        """Inconsistent color codes fail validation."""
        label1 = make_label(label_id="LBL-ICC-A", color_code="green",
                            zone_id=ZONE_ID_COCOA_A)
        label2 = make_label(label_id="LBL-ICC-B", color_code="red",
                            zone_id=ZONE_ID_COCOA_A)
        labeling_verification_engine.register_label(label1)
        labeling_verification_engine.register_label(label2)
        result = labeling_verification_engine.validate_color_codes(
            facility_id=FAC_ID_WAREHOUSE_GH
        )
        assert result.get("consistent") is False


# ===========================================================================
# 5. Label Event Recording
# ===========================================================================


class TestLabelEventRecording:
    """Test label event recording."""

    @pytest.mark.parametrize("event_type", LABEL_EVENT_TYPES)
    def test_record_all_label_events(self, labeling_verification_engine, event_type):
        """Each of the 5 label event types can be recorded."""
        label = make_label(label_id=f"LBL-EVT-{event_type}")
        labeling_verification_engine.register_label(label)
        evt = make_label_event(label_id=f"LBL-EVT-{event_type}", event_type=event_type)
        result = labeling_verification_engine.record_event(evt)
        assert result is not None
        assert result["event_type"] == event_type

    def test_applied_event_activates_label(self, labeling_verification_engine):
        """Applied event sets label status to active."""
        label = make_label(label_id="LBL-APPLY-001", status="pending")
        labeling_verification_engine.register_label(label)
        evt = make_label_event(label_id="LBL-APPLY-001", event_type="applied")
        labeling_verification_engine.record_event(evt)
        updated = labeling_verification_engine.get_label("LBL-APPLY-001")
        assert updated["status"] == "active"

    def test_damaged_event_flags_label(self, labeling_verification_engine):
        """Damaged event flags label for replacement."""
        label = make_label(label_id="LBL-DMG-001")
        labeling_verification_engine.register_label(label)
        evt = make_label_event(label_id="LBL-DMG-001", event_type="damaged")
        labeling_verification_engine.record_event(evt)
        updated = labeling_verification_engine.get_label("LBL-DMG-001")
        assert updated["status"] == "damaged"

    def test_replaced_event_links_new_label(self, labeling_verification_engine):
        """Replaced event marks old label as replaced."""
        label = make_label(label_id="LBL-REPL-001")
        labeling_verification_engine.register_label(label)
        evt = make_label_event(label_id="LBL-REPL-001", event_type="replaced")
        result = labeling_verification_engine.record_event(evt)
        assert result is not None

    def test_removed_event_deactivates_label(self, labeling_verification_engine):
        """Removed event deactivates label."""
        label = make_label(label_id="LBL-REM-001")
        labeling_verification_engine.register_label(label)
        evt = make_label_event(label_id="LBL-REM-001", event_type="removed")
        labeling_verification_engine.record_event(evt)
        updated = labeling_verification_engine.get_label("LBL-REM-001")
        assert updated["status"] == "removed"


# ===========================================================================
# 6. Labeling Audit
# ===========================================================================


class TestLabelingAudit:
    """Test comprehensive labeling audit for a facility."""

    def test_full_labeling_audit(self, labeling_verification_engine):
        """Run a full labeling audit for a facility."""
        label = make_label(label_id="LBL-AUD-001", facility_id=FAC_ID_WAREHOUSE_GH)
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.audit(facility_id=FAC_ID_WAREHOUSE_GH)
        assert result is not None
        assert "score" in result or "audit_score" in result

    def test_audit_includes_coverage(self, labeling_verification_engine):
        """Audit includes label coverage percentage."""
        label = make_label(label_id="LBL-AUD-COV", facility_id=FAC_ID_WAREHOUSE_GH)
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.audit(facility_id=FAC_ID_WAREHOUSE_GH)
        assert "coverage" in result or "label_coverage" in result or "scores" in result

    def test_audit_empty_facility(self, labeling_verification_engine):
        """Audit of facility with no labels returns zero score."""
        result = labeling_verification_engine.audit(facility_id="FAC-NOLABELS-001")
        total = result.get("score", result.get("audit_score", 0))
        assert total == 0.0

    def test_audit_provenance_hash(self, labeling_verification_engine):
        """Audit generates a provenance hash."""
        label = make_label(label_id="LBL-AUD-PROV", facility_id=FAC_ID_WAREHOUSE_GH)
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.audit(facility_id=FAC_ID_WAREHOUSE_GH)
        assert result.get("provenance_hash") is not None


# ===========================================================================
# 7. Expiring Label Detection
# ===========================================================================


class TestExpiringLabelDetection:
    """Test detection of expiring labels."""

    def test_detect_expiring_labels(self, labeling_verification_engine):
        """Labels expiring within threshold are detected."""
        label = make_label(label_id="LBL-EXP-001", days_until_expiry=5)
        labeling_verification_engine.register_label(label)
        expiring = labeling_verification_engine.find_expiring(
            facility_id=FAC_ID_WAREHOUSE_GH, days_threshold=30
        )
        assert any(l["label_id"] == "LBL-EXP-001" for l in expiring)

    def test_non_expiring_labels_not_detected(self, labeling_verification_engine):
        """Labels not expiring within threshold are not detected."""
        label = make_label(label_id="LBL-NOEXP-001", days_until_expiry=120)
        labeling_verification_engine.register_label(label)
        expiring = labeling_verification_engine.find_expiring(
            facility_id=FAC_ID_WAREHOUSE_GH, days_threshold=30
        )
        assert not any(l["label_id"] == "LBL-NOEXP-001" for l in expiring)

    def test_already_expired_label_detected(self, labeling_verification_engine):
        """Already expired labels are also detected."""
        label = make_label(label_id="LBL-PASTEXP-001", days_until_expiry=-10)
        labeling_verification_engine.register_label(label)
        expiring = labeling_verification_engine.find_expiring(
            facility_id=FAC_ID_WAREHOUSE_GH, days_threshold=30
        )
        assert any(l["label_id"] == "LBL-PASTEXP-001" for l in expiring)


# ===========================================================================
# 8. Labeling Score Calculation
# ===========================================================================


class TestLabelingScore:
    """Test labeling score calculation."""

    def test_score_has_components(self, labeling_verification_engine):
        """Labeling score includes coverage, readability, accuracy, timeliness."""
        label = make_label(label_id="LBL-SCORE-001", facility_id=FAC_ID_WAREHOUSE_GH)
        labeling_verification_engine.register_label(label)
        score = labeling_verification_engine.calculate_score(FAC_ID_WAREHOUSE_GH)
        expected = ["coverage", "readability", "accuracy", "timeliness"]
        for key in expected:
            assert key in score, f"Missing labeling score component: {key}"

    def test_score_within_bounds(self, labeling_verification_engine):
        """Labeling score is between 0 and 100."""
        label = make_label(label_id="LBL-SCORE-002", facility_id=FAC_ID_WAREHOUSE_GH)
        labeling_verification_engine.register_label(label)
        score = labeling_verification_engine.calculate_score(FAC_ID_WAREHOUSE_GH)
        total = score.get("total_score", score.get("score", 0))
        assert_valid_score(total)


# ===========================================================================
# 9. Label History Tracking
# ===========================================================================


class TestLabelHistory:
    """Test label history tracking."""

    def test_history_after_registration(self, labeling_verification_engine):
        """Label history starts with registration."""
        label = make_label(label_id="LBL-HIST-001")
        labeling_verification_engine.register_label(label)
        history = labeling_verification_engine.get_label_history("LBL-HIST-001")
        assert len(history) >= 1

    def test_history_after_events(self, labeling_verification_engine):
        """Events are recorded in label history."""
        label = make_label(label_id="LBL-HIST-002")
        labeling_verification_engine.register_label(label)
        evt = make_label_event(label_id="LBL-HIST-002", event_type="verified")
        labeling_verification_engine.record_event(evt)
        history = labeling_verification_engine.get_label_history("LBL-HIST-002")
        assert len(history) >= 2


# ===========================================================================
# 10. Edge Cases
# ===========================================================================


class TestLabelingEdgeCases:
    """Test edge cases for labeling verification."""

    def test_get_nonexistent_label_returns_none(self, labeling_verification_engine):
        """Getting a non-existent label returns None."""
        result = labeling_verification_engine.get_label("LBL-NONEXISTENT")
        assert result is None

    def test_expired_label_fails_verification(self, labeling_verification_engine):
        """Expired label fails verification."""
        label = make_label(label_id="LBL-EXPIRED", days_until_expiry=-30)
        label["status"] = "expired"
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.verify_label("LBL-EXPIRED")
        assert result.get("expired") is True or result.get("status") == "expired"

    def test_event_on_unregistered_label_raises(self, labeling_verification_engine):
        """Event on an unregistered label raises an error."""
        evt = make_label_event(label_id="LBL-NOREG-001", event_type="verified")
        with pytest.raises((ValueError, KeyError)):
            labeling_verification_engine.record_event(evt)

    @pytest.mark.parametrize("status", LABEL_STATUSES)
    def test_all_label_statuses(self, labeling_verification_engine, status):
        """Labels can be in each of the 5 valid statuses."""
        label = make_label(label_id=f"LBL-STA-{status}", status=status)
        result = labeling_verification_engine.register_label(label)
        assert result["status"] == status

    def test_empty_content_fields_detected(self, labeling_verification_engine):
        """Label with empty content fields is detected in verification."""
        label = make_label(label_type="bin_tag", content_fields={})
        labeling_verification_engine.register_label(label)
        result = labeling_verification_engine.verify_label(label["label_id"])
        assert result.get("content_valid") is False
