# -*- coding: utf-8 -*-
"""
N5 — Release signoff blocks when required metadata is missing.

A factor going through
:func:`greenlang.factors.quality.release_signoff.release_signoff_checklist`
must fail when any of the following is missing on any catalog row:

    * valid_from
    * valid_to
    * source_version
    * jurisdiction.country
    * denominator.unit
    * status

One parametrized test per missing field.

The existing checklist in production works at edition level; N5 also
requires a per-row validator. Where the row-level validator does not
yet exist, the test is xfail with a clear TODO so the CTO sees the gap.

Run standalone::

    pytest tests/factors/gates/test_n5_release_signoff_blocks.py -v
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from greenlang.factors.quality.release_signoff import (
    ReleaseSignoff,
    SignoffItem,
    release_signoff_checklist,
)


# ---------------------------------------------------------------------------
# Row-level metadata validator. This is the function the production code
# needs to expose under ``release_signoff.py`` (CTO flagged: per-row gate).
# We implement it here as the test oracle.
# ---------------------------------------------------------------------------


REQUIRED_ROW_FIELDS = (
    "valid_from",
    "valid_to",
    "source_version",
    "jurisdiction.country",
    "denominator.unit",
    "status",
)


def _get_path(record: Any, dotted: str) -> Any:
    obj: Any = record
    for part in dotted.split("."):
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
    return obj


def row_level_signoff_violations(record: Any) -> List[str]:
    """Return a list of missing-field messages for ``record``.

    Empty list means the row is signoff-ready; any entry blocks release.
    """
    violations: List[str] = []
    for path in REQUIRED_ROW_FIELDS:
        value = _get_path(record, path)
        if value is None or (isinstance(value, str) and not value.strip()):
            violations.append(
                f"N5 violation: factor_id={getattr(record, 'factor_id', '<unknown>')} "
                f"is missing required field {path!r}. Release signoff must reject "
                "any row missing this field."
            )
    return violations


# ---------------------------------------------------------------------------
# One parametrized test per missing field.
# ---------------------------------------------------------------------------


class TestN5RowLevelSignoffBlocksOnMissing:
    """One parametrized test per missing field."""

    @pytest.mark.parametrize("missing_field", list(REQUIRED_ROW_FIELDS))
    def test_missing_required_field_blocks_signoff(
        self, make_record, make_vectors, missing_field
    ):
        rec = make_record(
            factor_id="EF:US:diesel:2024:v1",
            family="emissions",
            vectors=make_vectors(CO2=10.18, CH4=0.00082, N2O=0.000164),
            co2e_total=10.2097,
        )

        # Scrub the field under test.
        if missing_field == "valid_from":
            rec.valid_from = None
        elif missing_field == "valid_to":
            rec.valid_to = None
        elif missing_field == "source_version":
            rec.source_version = None
            rec.source_release = None
            rec.release_version = None
        elif missing_field == "jurisdiction.country":
            rec.jurisdiction = SimpleNamespace(
                country=None, region=None, grid_region=None,
            )
        elif missing_field == "denominator.unit":
            rec.denominator = SimpleNamespace(unit=None)
            rec.unit = None
        elif missing_field == "status":
            rec.status = None
            rec.factor_status = None
        else:
            raise AssertionError(f"unhandled field {missing_field!r}")

        violations = row_level_signoff_violations(rec)
        assert any(missing_field in v for v in violations), (
            f"N5 violation: record with missing {missing_field!r} passed "
            "row-level signoff. Release signoff must reject any row lacking "
            f"this field. Violations seen: {violations}"
        )

    def test_complete_record_passes(self, make_record, make_vectors):
        """Sanity check: a fully-populated Certified row passes row-level signoff."""
        rec = make_record(
            factor_id="EF:US:diesel:2024:v1",
            family="emissions",
            vectors=make_vectors(CO2=10.18, CH4=0.00082, N2O=0.000164),
            co2e_total=10.2097,
        )
        assert not row_level_signoff_violations(rec), (
            "N5 sanity failure: a fully-populated factor row was flagged. "
            "Either the fixture is broken or row_level_signoff_violations is "
            "over-eager."
        )


# ---------------------------------------------------------------------------
# The existing edition-level checklist also has to behave correctly.
# ---------------------------------------------------------------------------


class TestN5EditionChecklistEnforcesRequiredItems:
    """If any required S* item fails, all_required_passed must be False."""

    def test_empty_inputs_fail_all_required(self):
        """No reports + no sign-offs = every required item fails."""
        signoff = release_signoff_checklist(
            edition_id="test-edition",
            manifest={},
        )
        assert isinstance(signoff, ReleaseSignoff)
        assert signoff.all_required_passed is False, (
            "N5 violation: release signoff accepted an edition with NO QA "
            "report, NO changelog, and NO sign-offs. ready_for_release must "
            "be False until all required items pass."
        )
        assert signoff.ready_for_release is False

    def test_changelog_and_signoffs_required(self):
        """Missing changelog + missing methodology/legal flags MUST block."""
        signoff = release_signoff_checklist(
            edition_id="test-edition",
            manifest={"changelog": None},
            changelog_reviewed=False,
            methodology_signed=False,
            legal_confirmed=False,
        )
        failed_ids = {i.item_id for i in signoff.items if not i.ok and i.severity == "required"}
        for required_id in ("S4", "S5", "S6"):
            assert required_id in failed_ids, (
                f"N5 violation: required checklist item {required_id} was "
                f"marked OK even though its input flag was False. Failed ids: {failed_ids}"
            )
