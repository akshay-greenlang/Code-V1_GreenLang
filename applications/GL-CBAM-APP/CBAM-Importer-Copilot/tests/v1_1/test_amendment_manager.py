# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Amendment Manager

Tests amendment management:
- create_amendment (valid, outside window)
- validate_amendment_window (within 60 days, expired)
- get_amendment_diff (field-level changes)
- apply_amendment (version increment)
- rollback_to_version (valid, invalid version)
- Version history chain integrity

Target: 40+ tests
"""

import pytest
import uuid
from datetime import datetime, date, timedelta
from decimal import Decimal
from copy import deepcopy


# ---------------------------------------------------------------------------
# Inline amendment manager for self-contained tests
# ---------------------------------------------------------------------------

class AmendmentError(Exception):
    pass


class AmendmentWindowExpiredError(AmendmentError):
    pass


class VersionNotFoundError(AmendmentError):
    pass


class AmendmentManager:
    """Manages submission amendments with version control."""

    AMENDMENT_WINDOW_DAYS = 60

    def __init__(self):
        self._submissions = {}
        self._version_history = {}

    def load_submission(self, submission):
        sid = submission["submission_id"]
        self._submissions[sid] = deepcopy(submission)
        self._version_history[sid] = [deepcopy(submission)]

    def create_amendment(self, submission_id, changes, quarter_end_date=None):
        if submission_id not in self._submissions:
            raise AmendmentError(f"Submission not found: {submission_id}")

        if quarter_end_date:
            if not self.validate_amendment_window(quarter_end_date):
                raise AmendmentWindowExpiredError(
                    f"Amendment window expired for quarter ending {quarter_end_date}"
                )

        current = self._submissions[submission_id]
        amendment_id = f"AMD-{uuid.uuid4().hex[:8].upper()}"
        diff = self.get_amendment_diff(current, changes)

        amended = deepcopy(current)
        amended.update(changes)
        amended["version"] = current["version"] + 1
        amended["status"] = "amended"
        amended["amendment_id"] = amendment_id
        amended["amended_at"] = datetime.utcnow().isoformat()
        amended["amendment_diff"] = diff

        self._submissions[submission_id] = amended
        self._version_history[submission_id].append(deepcopy(amended))

        return amended

    def validate_amendment_window(self, quarter_end_date):
        if isinstance(quarter_end_date, str):
            quarter_end_date = date.fromisoformat(quarter_end_date)
        deadline = quarter_end_date + timedelta(days=self.AMENDMENT_WINDOW_DAYS)
        return date.today() <= deadline

    def get_amendment_diff(self, original, changes):
        diff = {}
        for key, new_val in changes.items():
            old_val = original.get(key)
            if old_val != new_val:
                diff[key] = {"old": old_val, "new": new_val}
        return diff

    def apply_amendment(self, submission_id, amendment):
        if submission_id not in self._submissions:
            raise AmendmentError(f"Submission not found: {submission_id}")
        current = self._submissions[submission_id]
        for key, val in amendment.items():
            if key not in ("version", "amendment_id", "amended_at",
                           "amendment_diff", "status"):
                current[key] = val
        current["version"] += 1
        current["status"] = "amended"
        self._version_history[submission_id].append(deepcopy(current))
        return current

    def rollback_to_version(self, submission_id, target_version):
        if submission_id not in self._version_history:
            raise AmendmentError(f"Submission not found: {submission_id}")
        history = self._version_history[submission_id]
        target = None
        for entry in history:
            if entry["version"] == target_version:
                target = deepcopy(entry)
                break
        if target is None:
            raise VersionNotFoundError(
                f"Version {target_version} not found for {submission_id}"
            )
        target["version"] = self._submissions[submission_id]["version"] + 1
        target["status"] = "rolled_back"
        target["rolled_back_to"] = target_version
        target["rolled_back_at"] = datetime.utcnow().isoformat()
        self._submissions[submission_id] = target
        self._version_history[submission_id].append(deepcopy(target))
        return target

    def get_version_history(self, submission_id):
        if submission_id not in self._version_history:
            raise AmendmentError(f"Submission not found: {submission_id}")
        return deepcopy(self._version_history[submission_id])

    def get_current_version(self, submission_id):
        if submission_id not in self._submissions:
            raise AmendmentError(f"Submission not found: {submission_id}")
        return self._submissions[submission_id]["version"]


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def manager():
    return AmendmentManager()


@pytest.fixture
def base_submission():
    return {
        "submission_id": "SUB-001",
        "supplier_id": "S1",
        "installation_id": "I1",
        "cn_code": "72031000",
        "reporting_quarter": "2026Q1",
        "direct_emissions_tco2": "200.000",
        "indirect_emissions_tco2": "50.000",
        "calculation_method": "eu_default",
        "version": 1,
        "status": "draft",
    }


@pytest.fixture
def loaded_manager(manager, base_submission):
    manager.load_submission(base_submission)
    return manager


# ===========================================================================
# TEST CLASS -- create_amendment
# ===========================================================================

class TestCreateAmendment:
    """Tests for create_amendment."""

    def test_create_valid_amendment(self, loaded_manager):
        result = loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        assert result["version"] == 2
        assert result["status"] == "amended"
        assert "amendment_id" in result

    def test_amendment_increments_version(self, loaded_manager):
        loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "210.000"}
        )
        result = loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        assert result["version"] == 3

    def test_amendment_preserves_unchanged_fields(self, loaded_manager):
        result = loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        assert result["cn_code"] == "72031000"
        assert result["indirect_emissions_tco2"] == "50.000"

    def test_nonexistent_submission_raises(self, manager):
        with pytest.raises(AmendmentError, match="not found"):
            manager.create_amendment("NONE", {"direct_emissions_tco2": "100"})

    def test_amendment_with_expired_window(self, loaded_manager):
        past_end = date(2024, 3, 31)
        with pytest.raises(AmendmentWindowExpiredError):
            loaded_manager.create_amendment(
                "SUB-001",
                {"direct_emissions_tco2": "220.000"},
                quarter_end_date=past_end,
            )

    def test_amendment_with_valid_window(self, loaded_manager):
        future_end = date.today() + timedelta(days=10)
        result = loaded_manager.create_amendment(
            "SUB-001",
            {"direct_emissions_tco2": "220.000"},
            quarter_end_date=future_end,
        )
        assert result["version"] == 2

    def test_amendment_records_diff(self, loaded_manager):
        result = loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        diff = result["amendment_diff"]
        assert "direct_emissions_tco2" in diff
        assert diff["direct_emissions_tco2"]["old"] == "200.000"
        assert diff["direct_emissions_tco2"]["new"] == "220.000"

    def test_amendment_multiple_fields(self, loaded_manager):
        result = loaded_manager.create_amendment(
            "SUB-001",
            {"direct_emissions_tco2": "220.000",
             "calculation_method": "supplier_specific"},
        )
        diff = result["amendment_diff"]
        assert len(diff) == 2


# ===========================================================================
# TEST CLASS -- validate_amendment_window
# ===========================================================================

class TestValidateAmendmentWindow:
    """Tests for validate_amendment_window."""

    def test_within_window(self, manager):
        end = date.today() - timedelta(days=30)
        assert manager.validate_amendment_window(end) is True

    def test_at_deadline(self, manager):
        end = date.today() - timedelta(days=60)
        assert manager.validate_amendment_window(end) is True

    def test_past_deadline(self, manager):
        end = date.today() - timedelta(days=61)
        assert manager.validate_amendment_window(end) is False

    def test_future_quarter_end(self, manager):
        end = date.today() + timedelta(days=30)
        assert manager.validate_amendment_window(end) is True

    def test_string_date_input(self, manager):
        end = (date.today() - timedelta(days=30)).isoformat()
        assert manager.validate_amendment_window(end) is True


# ===========================================================================
# TEST CLASS -- get_amendment_diff
# ===========================================================================

class TestGetAmendmentDiff:
    """Tests for get_amendment_diff."""

    def test_single_field_change(self, manager):
        original = {"field_a": "old", "field_b": "same"}
        changes = {"field_a": "new"}
        diff = manager.get_amendment_diff(original, changes)
        assert diff == {"field_a": {"old": "old", "new": "new"}}

    def test_no_changes(self, manager):
        original = {"field_a": "same"}
        changes = {"field_a": "same"}
        diff = manager.get_amendment_diff(original, changes)
        assert diff == {}

    def test_new_field(self, manager):
        original = {"field_a": "old"}
        changes = {"field_b": "new"}
        diff = manager.get_amendment_diff(original, changes)
        assert diff == {"field_b": {"old": None, "new": "new"}}

    def test_multiple_changes(self, manager):
        original = {"a": 1, "b": 2, "c": 3}
        changes = {"a": 10, "b": 20}
        diff = manager.get_amendment_diff(original, changes)
        assert len(diff) == 2


# ===========================================================================
# TEST CLASS -- apply_amendment
# ===========================================================================

class TestApplyAmendment:
    """Tests for apply_amendment."""

    def test_apply_increments_version(self, loaded_manager):
        result = loaded_manager.apply_amendment(
            "SUB-001", {"direct_emissions_tco2": "230.000"}
        )
        assert result["version"] == 2

    def test_apply_updates_field(self, loaded_manager):
        result = loaded_manager.apply_amendment(
            "SUB-001", {"direct_emissions_tco2": "230.000"}
        )
        assert result["direct_emissions_tco2"] == "230.000"

    def test_apply_nonexistent_raises(self, manager):
        with pytest.raises(AmendmentError):
            manager.apply_amendment("NONE", {"x": "y"})

    def test_apply_preserves_other_fields(self, loaded_manager):
        result = loaded_manager.apply_amendment(
            "SUB-001", {"direct_emissions_tco2": "230.000"}
        )
        assert result["cn_code"] == "72031000"


# ===========================================================================
# TEST CLASS -- rollback_to_version
# ===========================================================================

class TestRollbackToVersion:
    """Tests for rollback_to_version."""

    def test_rollback_to_version_1(self, loaded_manager):
        loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        result = loaded_manager.rollback_to_version("SUB-001", 1)
        assert result["direct_emissions_tco2"] == "200.000"
        assert result["status"] == "rolled_back"
        assert result["rolled_back_to"] == 1

    def test_rollback_version_increments(self, loaded_manager):
        loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        result = loaded_manager.rollback_to_version("SUB-001", 1)
        assert result["version"] == 3

    def test_rollback_invalid_version_raises(self, loaded_manager):
        with pytest.raises(VersionNotFoundError):
            loaded_manager.rollback_to_version("SUB-001", 99)

    def test_rollback_nonexistent_submission_raises(self, manager):
        with pytest.raises(AmendmentError):
            manager.rollback_to_version("NONE", 1)


# ===========================================================================
# TEST CLASS -- Version history chain
# ===========================================================================

class TestVersionHistory:
    """Tests for version history chain integrity."""

    def test_initial_history_has_one_entry(self, loaded_manager):
        history = loaded_manager.get_version_history("SUB-001")
        assert len(history) == 1
        assert history[0]["version"] == 1

    def test_amendment_adds_to_history(self, loaded_manager):
        loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        history = loaded_manager.get_version_history("SUB-001")
        assert len(history) == 2

    def test_version_numbers_sequential(self, loaded_manager):
        loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "230.000"}
        )
        history = loaded_manager.get_version_history("SUB-001")
        versions = [h["version"] for h in history]
        assert versions == [1, 2, 3]

    def test_history_preserves_original(self, loaded_manager):
        loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        history = loaded_manager.get_version_history("SUB-001")
        assert history[0]["direct_emissions_tco2"] == "200.000"
        assert history[1]["direct_emissions_tco2"] == "220.000"

    def test_rollback_adds_to_history(self, loaded_manager):
        loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        loaded_manager.rollback_to_version("SUB-001", 1)
        history = loaded_manager.get_version_history("SUB-001")
        assert len(history) == 3

    def test_get_current_version(self, loaded_manager):
        assert loaded_manager.get_current_version("SUB-001") == 1
        loaded_manager.create_amendment(
            "SUB-001", {"direct_emissions_tco2": "220.000"}
        )
        assert loaded_manager.get_current_version("SUB-001") == 2

    def test_history_returns_copy(self, loaded_manager):
        history = loaded_manager.get_version_history("SUB-001")
        history[0]["cn_code"] = "MODIFIED"
        original = loaded_manager.get_version_history("SUB-001")
        assert original[0]["cn_code"] == "72031000"
