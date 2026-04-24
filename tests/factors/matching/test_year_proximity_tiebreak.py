# -*- coding: utf-8 -*-
"""Wave 5 regression tests for year-proximity tiebreak (CEA FY27 bug).

Known bug (Wave 4 finding): when two CEA siblings share the same
``source_id``, ``factor_family``, ``geography`` and ``verification``
status, the only distinguishing field is vintage.  The old
``score_time`` helper only consulted ``valid_to`` — as long as the
candidate's window ended on/after the request date, the signal
returned 0.  Both the FY27 factor (2026-04-01 -> 2027-03-31) and
the FY28 sibling (2027-04-01 -> 2028-03-31) satisfied that test
for an FY27 request, so the tie was broken by candidate insertion
order and the resolver sometimes picked the wrong vintage.

The fix: ``score_time`` now awards 0 only when the validity window
CONTAINS the request date; past windows earn ``1 + years_stale``
and future windows earn ``2 + years_ahead`` — so a past-dated
valid sibling is ALWAYS preferred over a future-dated one when
neither contains the request, and the in-window factor beats
either.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import List

import pytest

from greenlang.factors.resolution.tiebreak import (
    build_tiebreak,
    score_time,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _cea_sibling(
    *,
    factor_id: str,
    valid_from: date,
    valid_to: date,
) -> SimpleNamespace:
    """Minimal CEA sibling record that matches on every non-vintage signal."""
    return SimpleNamespace(
        factor_id=factor_id,
        factor_family="grid_intensity",
        fuel_type=None,
        geography="IN",
        valid_from=valid_from,
        valid_to=valid_to,
        source_id="india_cea_co2_baseline",
        redistribution_class="open",
        time_granularity="annual",
        verification=SimpleNamespace(status="regulator_approved"),
        uncertainty_95ci=0.05,
        unit="kWh",
        tags=["electricity", "grid", "india"],
        notes=None,
    )


def _rank(
    records: List[SimpleNamespace],
    *,
    request_date: date,
    request_geo: str = "IN",
) -> List[str]:
    """Return ``factor_id``s ordered by ascending tiebreak score (lowest first)."""
    scored = []
    for rec in records:
        tb = build_tiebreak(
            rec,
            request_geo=request_geo,
            request_date=request_date,
            request_granularity="annual",
            activity_tokens=["electricity"],
        )
        scored.append((tb.score(), rec.factor_id))
    scored.sort(key=lambda pair: pair[0])
    return [fid for _s, fid in scored]


# ---------------------------------------------------------------------
# score_time unit tests
# ---------------------------------------------------------------------


class TestScoreTimeYearProximity:
    """Unit tests for the refined ``score_time`` helper."""

    def test_window_contains_request_returns_zero(self):
        """FY27 request lands inside a FY27 window -> best possible score."""
        assert (
            score_time(
                valid_to=date(2027, 3, 31),
                request_date=date(2026, 9, 15),
                valid_from=date(2026, 4, 1),
            )
            == 0
        )

    def test_future_window_penalised_over_past_window(self):
        """For an FY27 request, FY28 (future) must score strictly worse than
        FY26 (past). This is the core CEA-vintage-mismatch fix."""
        req = date(2026, 9, 15)
        fy26_score = score_time(
            valid_to=date(2026, 3, 31),
            request_date=req,
            valid_from=date(2025, 4, 1),
        )
        fy28_score = score_time(
            valid_to=date(2028, 3, 31),
            request_date=req,
            valid_from=date(2027, 4, 1),
        )
        assert fy26_score < fy28_score, (
            f"past-dated FY26 must beat future-dated FY28; "
            f"got fy26={fy26_score}, fy28={fy28_score}"
        )

    def test_fy27_window_beats_fy28_sibling_for_fy27_request(self):
        """FY27 request: FY27 window scores 0, FY28 window scores > 0."""
        req = date(2026, 9, 15)
        fy27 = score_time(
            valid_to=date(2027, 3, 31),
            request_date=req,
            valid_from=date(2026, 4, 1),
        )
        fy28 = score_time(
            valid_to=date(2028, 3, 31),
            request_date=req,
            valid_from=date(2027, 4, 1),
        )
        assert fy27 == 0
        assert fy28 > 0

    def test_legacy_no_valid_from_stays_backward_compatible(self):
        """SimpleNamespace fixtures with no ``valid_from`` must still resolve."""
        # Past valid_to -> stale (old behaviour preserved).
        stale = score_time(
            valid_to=date(2020, 12, 31),
            request_date=date(2026, 9, 15),
            valid_from=None,
        )
        assert stale > 0
        # Future valid_to and no valid_from -> treated as current (old behaviour).
        current = score_time(
            valid_to=date(2099, 12, 31),
            request_date=date(2026, 9, 15),
            valid_from=None,
        )
        assert current == 0

    def test_score_caps_at_ten(self):
        """Far-future or far-past windows must cap at 10 so vintage cannot
        dominate a semantic miss."""
        req = date(2026, 9, 15)
        far_past = score_time(
            valid_to=date(2000, 3, 31),
            request_date=req,
            valid_from=date(1999, 4, 1),
        )
        far_future = score_time(
            valid_to=date(2060, 3, 31),
            request_date=req,
            valid_from=date(2059, 4, 1),
        )
        assert far_past == 10
        assert far_future == 10


# ---------------------------------------------------------------------
# End-to-end tiebreak ranking tests — CEA sibling scenario
# ---------------------------------------------------------------------


class TestCeaVintageTiebreak:
    """Three regression tests simulating the CEA FY27 bug and its dual."""

    def test_fy27_request_fy27_beats_fy28_sibling(self):
        """FY27 request must pick FY27, not the newer FY28 sibling."""
        fy27 = _cea_sibling(
            factor_id="EF:IN:all_india:2026-27:cea-v20.0",
            valid_from=date(2026, 4, 1),
            valid_to=date(2027, 3, 31),
        )
        fy28 = _cea_sibling(
            factor_id="EF:IN:all_india:2027-28:cea-v20.0",
            valid_from=date(2027, 4, 1),
            valid_to=date(2028, 3, 31),
        )
        # Insertion order deliberately puts FY28 first — this was the
        # insertion-order-wins bug before the fix.
        ranked = _rank([fy28, fy27], request_date=date(2026, 9, 15))
        assert ranked[0] == fy27.factor_id, ranked

    def test_fy28_request_fy28_beats_fy27_sibling(self):
        """Dual: FY28 request must pick FY28, not the older FY27 sibling.

        Guards against an over-correction where past is ALWAYS preferred.
        """
        fy27 = _cea_sibling(
            factor_id="EF:IN:all_india:2026-27:cea-v20.0",
            valid_from=date(2026, 4, 1),
            valid_to=date(2027, 3, 31),
        )
        fy28 = _cea_sibling(
            factor_id="EF:IN:all_india:2027-28:cea-v20.0",
            valid_from=date(2027, 4, 1),
            valid_to=date(2028, 3, 31),
        )
        ranked = _rank([fy27, fy28], request_date=date(2027, 9, 15))
        assert ranked[0] == fy28.factor_id, ranked

    def test_in_window_fy27_beats_out_of_window_fy28_even_near_handover(self):
        """Near the FY26-27 -> FY27-28 handover (e.g. 2027-03-15), the FY27
        window still CONTAINS the date and the FY28 window is still in the
        future. FY27 must win despite being the "older" vintage."""
        fy27 = _cea_sibling(
            factor_id="EF:IN:all_india:2026-27:cea-v20.0",
            valid_from=date(2026, 4, 1),
            valid_to=date(2027, 3, 31),
        )
        fy28 = _cea_sibling(
            factor_id="EF:IN:all_india:2027-28:cea-v20.0",
            valid_from=date(2027, 4, 1),
            valid_to=date(2028, 3, 31),
        )
        ranked = _rank([fy28, fy27], request_date=date(2027, 3, 15))
        assert ranked[0] == fy27.factor_id, ranked

    def test_no_candidate_eliminated_by_year_proximity_signal(self):
        """Non-negotiable: the signal must PENALIZE, never filter. Even a
        far-future candidate keeps a finite score so the engine can still
        resolve when it is the only option."""
        only_future = _cea_sibling(
            factor_id="EF:IN:all_india:2040-41:cea-vX",
            valid_from=date(2040, 4, 1),
            valid_to=date(2041, 3, 31),
        )
        tb = build_tiebreak(
            only_future,
            request_geo="IN",
            request_date=date(2026, 9, 15),
            request_granularity="annual",
            activity_tokens=["electricity"],
        )
        # Score is finite and below a sentinel cap — the engine can still
        # return this factor when no in-window sibling exists.
        assert tb.score() < 100
        assert tb.time_distance == 10  # capped, not infinity


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
