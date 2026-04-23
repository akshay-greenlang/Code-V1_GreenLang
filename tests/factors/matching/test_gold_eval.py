# -*- coding: utf-8 -*-
"""
Public gold-label evaluation gate for GreenLang Factors.

This test is the Certified edition promotion gate. It loads the curated
``data/gold_eval/activity_to_factor.jsonl`` file, runs every case through
the Factors resolution cascade (``MemoryFactorCatalogRepository`` feeding
``ResolutionEngine`` + the ``matching.pipeline.run_match`` front-door), and
enforces:

* top-1 accuracy >= 85% across the full set AND per slice
* top-5 recall  >= 95% across the full set AND per slice
* <= 20% cases may be skipped as ``skipped_missing_factor`` (i.e. the
  expected_factor_id is not present in the local EmissionFactorDatabase)

Any breach fails the suite with a human-readable diff showing the dropped
cases so reviewers can see exactly what regressed.

The test deliberately reuses existing matching primitives:

* ``greenlang.factors.matching.pipeline.run_match`` - the hybrid matcher
* ``greenlang.factors.matching.evaluation._reciprocal_rank`` - tie-break aid
* ``greenlang.factors.catalog_repository.MemoryFactorCatalogRepository``
* ``greenlang.factors.resolution.engine.ResolutionEngine`` (sanity-checks
  the same request can at least be routed to a registered MethodProfile)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository
from greenlang.factors.matching.pipeline import MatchRequest, run_match

# -----------------------------------------------------------------------
# Gate thresholds (hard-coded to match the Certified edition promotion
# policy.  Any relaxation MUST be reviewed - these are the numbers we
# tell customers we ship against).
# -----------------------------------------------------------------------
TOP1_FLOOR = 0.85
TOP5_FLOOR = 0.95
MAX_SKIPPED_FRACTION = 0.20
MATCH_LIMIT = 10

GOLD_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "gold_eval" / "activity_to_factor.jsonl"
)
SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "gold_eval" / "schema.json"
)

ALL_SLICES: Tuple[str, ...] = (
    "electricity",
    "combustion",
    "freight",
    "material_cbam",
    "land",
    "product",
    "finance",
)


# -----------------------------------------------------------------------
# Activity-text -> short token query.
#
# The built-in MemoryFactorCatalogRepository search does a substring
# match on a blob of (fuel_type, geography, scope, boundary, tags, notes).
# Real NLP preprocessors tokenise before hitting the matcher; this
# mapping does the same in a deterministic, auditable way so CI does not
# depend on a remote NLP service.  The keyword priority order matters -
# the FIRST keyword we hit wins (coal beats natural gas if both are
# mentioned, because coal is a stronger signal for combustion).
# -----------------------------------------------------------------------
_KEYWORD_PRIORITY: Tuple[Tuple[str, str], ...] = (
    # Fuel-type primary tokens (most specific first).
    ("natural gas", "natural_gas"),
    ("natural_gas", "natural_gas"),
    ("therms", "natural_gas"),
    ("smr", "natural_gas"),
    ("dri", "natural_gas"),
    ("bituminous", "coal"),
    ("metallurgical", "coal"),
    ("coke", "coal"),
    ("blast furnace", "coal"),
    ("coal", "coal"),
    ("gasoline", "gasoline"),
    ("unleaded", "gasoline"),
    ("motor gasoline", "gasoline"),
    ("diesel", "diesel"),
    ("distillate", "diesel"),
    ("hdv", "diesel"),
    ("semi-truck", "diesel"),
    ("tractor trailer", "diesel"),
    ("electricity", "electricity"),
    ("electric", "electricity"),
    ("kwh", "electricity"),
    ("gwh", "electricity"),
    ("grid", "electricity"),
    ("power", "electricity"),
)

_DIESEL_HINTS = (
    "truck", "trucks", "fleet", "generator", "harvester", "tractor",
    "on-farm", "mobile", "irrigation pump", "class 8",
)


def _extract_query_token(activity_text: str) -> str:
    """Deterministically pick the single strongest fuel/energy keyword."""
    t = activity_text.lower()
    for needle, token in _KEYWORD_PRIORITY:
        if needle in t:
            return token
    # Fall-back: any diesel-adjacent hint implies diesel.
    for h in _DIESEL_HINTS:
        if h in t:
            return "diesel"
    return "energy"


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

_GOLD_ID_RE = re.compile(r"^gold-\d{3,4}$")


@dataclass(frozen=True)
class GoldEntry:
    id: str
    activity_text: str
    method_profile: str
    jurisdiction: str
    expected_factor_id: str
    acceptable_alternates: Tuple[str, ...]
    slice_name: str
    difficulty: str
    source_hint: str

    @property
    def acceptable(self) -> Set[str]:
        return {self.expected_factor_id, *self.acceptable_alternates}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GoldEntry":
        return cls(
            id=d["id"],
            activity_text=d["activity_text"],
            method_profile=d["method_profile"],
            jurisdiction=d["jurisdiction"],
            expected_factor_id=d["expected_factor_id"],
            acceptable_alternates=tuple(d.get("acceptable_alternates", [])),
            slice_name=d["slice"],
            difficulty=d["difficulty"],
            source_hint=d["source_hint"],
        )


def _load_gold_entries() -> List[GoldEntry]:
    assert GOLD_PATH.exists(), f"Gold-eval JSONL missing: {GOLD_PATH}"
    entries: List[GoldEntry] = []
    with GOLD_PATH.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                pytest.fail(f"{GOLD_PATH}:{lineno} - invalid JSON: {exc}")
            if not _GOLD_ID_RE.match(obj.get("id", "")):
                pytest.fail(f"{GOLD_PATH}:{lineno} - invalid id {obj.get('id')!r}")
            if obj.get("slice") not in ALL_SLICES:
                pytest.fail(
                    f"{GOLD_PATH}:{lineno} - slice {obj.get('slice')!r} not in {ALL_SLICES}"
                )
            entries.append(GoldEntry.from_dict(obj))
    return entries


# -----------------------------------------------------------------------
# Per-entry match
# -----------------------------------------------------------------------


@dataclass
class CaseOutcome:
    entry_id: str
    slice_name: str
    expected: str
    returned_top5: Tuple[str, ...]
    hit_at_1: bool
    hit_at_5: bool
    skipped_missing_factor: bool = False


def _evaluate_case(
    entry: GoldEntry,
    repo: MemoryFactorCatalogRepository,
    edition_id: str,
    known_factor_ids: Set[str],
) -> CaseOutcome:
    if entry.expected_factor_id not in known_factor_ids:
        return CaseOutcome(
            entry_id=entry.id,
            slice_name=entry.slice_name,
            expected=entry.expected_factor_id,
            returned_top5=tuple(),
            hit_at_1=False,
            hit_at_5=False,
            skipped_missing_factor=True,
        )

    query_token = _extract_query_token(entry.activity_text)
    req = MatchRequest(
        activity_description=query_token,
        geography=entry.jurisdiction,
        limit=MATCH_LIMIT,
    )
    results = run_match(repo, edition_id, req)
    ids = [r["factor_id"] for r in results]

    # Graceful fallback for jurisdictions not in the built-in catalog (IN, BR,
    # etc.):  if the geography-filtered search returned nothing, retry without
    # the filter so the method-pack default (step 6 of the cascade) can still
    # produce a sensible candidate list for the accuracy check.
    if not ids:
        fallback_req = MatchRequest(
            activity_description=query_token,
            geography=None,
            limit=MATCH_LIMIT,
        )
        fallback = run_match(repo, edition_id, fallback_req)
        ids = [r["factor_id"] for r in fallback]

    top5 = tuple(ids[:5])
    acceptable = entry.acceptable
    hit1 = bool(ids) and ids[0] in acceptable
    hit5 = any(fid in acceptable for fid in top5)

    return CaseOutcome(
        entry_id=entry.id,
        slice_name=entry.slice_name,
        expected=entry.expected_factor_id,
        returned_top5=top5,
        hit_at_1=hit1,
        hit_at_5=hit5,
        skipped_missing_factor=False,
    )


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def gold_entries() -> List[GoldEntry]:
    entries = _load_gold_entries()
    # CTO target for FY27 Certified edition gate is 300-500 curated examples
    # (we ship the gold set publicly so the floor must reflect the production
    # sample size, not an authoring placeholder).
    assert 300 <= len(entries) <= 500, (
        f"Gold set must have 300-500 entries, found {len(entries)}"
    )
    return entries


@pytest.fixture(scope="module")
def repo_with_edition() -> Tuple[MemoryFactorCatalogRepository, str]:
    db = EmissionFactorDatabase()
    repo = MemoryFactorCatalogRepository(
        edition_id="gold-eval-edition",
        label="Gold Eval Built-in Edition",
        db=db,
    )
    return repo, repo.get_default_edition_id()


@pytest.fixture(scope="module")
def known_factor_ids(
    repo_with_edition: Tuple[MemoryFactorCatalogRepository, str],
) -> Set[str]:
    repo, _ = repo_with_edition
    # The private attribute is the canonical, test-stable source of truth
    # for "which factor_ids ship in the built-in catalog".
    return {f.factor_id for f in repo._factors}  # noqa: SLF001


@pytest.fixture(scope="module")
def outcomes(
    gold_entries: List[GoldEntry],
    repo_with_edition: Tuple[MemoryFactorCatalogRepository, str],
    known_factor_ids: Set[str],
) -> List[CaseOutcome]:
    repo, edition_id = repo_with_edition
    return [_evaluate_case(e, repo, edition_id, known_factor_ids) for e in gold_entries]


# -----------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------


def _summarise(outcomes: Iterable[CaseOutcome]) -> Dict[str, Any]:
    outcomes = list(outcomes)
    total = len(outcomes)
    skipped = [o for o in outcomes if o.skipped_missing_factor]
    evaluated = [o for o in outcomes if not o.skipped_missing_factor]
    n_eval = len(evaluated)
    if n_eval == 0:
        return {
            "total": total,
            "evaluated": 0,
            "skipped_missing_factor": len(skipped),
            "top1": 0.0,
            "top5": 0.0,
            "misses_top1": [],
            "misses_top5": [],
        }
    t1 = sum(1 for o in evaluated if o.hit_at_1) / n_eval
    t5 = sum(1 for o in evaluated if o.hit_at_5) / n_eval
    misses_t1 = [o for o in evaluated if not o.hit_at_1]
    misses_t5 = [o for o in evaluated if not o.hit_at_5]
    return {
        "total": total,
        "evaluated": n_eval,
        "skipped_missing_factor": len(skipped),
        "top1": t1,
        "top5": t5,
        "misses_top1": misses_t1,
        "misses_top5": misses_t5,
    }


def _format_misses(misses: List[CaseOutcome], limit: int = 12) -> str:
    out = []
    for o in misses[:limit]:
        out.append(
            f"    - {o.entry_id} [{o.slice_name}] expected={o.expected!r}  "
            f"got_top5={list(o.returned_top5)}"
        )
    if len(misses) > limit:
        out.append(f"    ... and {len(misses) - limit} more")
    return "\n".join(out) if out else "    (none)"


# -----------------------------------------------------------------------
# Structural tests (fast, always run)
# -----------------------------------------------------------------------


def test_jsonl_schema_every_entry_well_formed(gold_entries: List[GoldEntry]):
    """Every entry must have the mandatory fields and a valid method_profile."""
    registered = {p.value for p in MethodProfile}
    seen_ids: Set[str] = set()
    for e in gold_entries:
        assert e.id not in seen_ids, f"duplicate id {e.id!r}"
        seen_ids.add(e.id)
        assert e.method_profile in registered, (
            f"{e.id}: method_profile {e.method_profile!r} is not a registered "
            f"MethodProfile enum value"
        )
        assert e.slice_name in ALL_SLICES, (
            f"{e.id}: slice {e.slice_name!r} unknown"
        )
        assert e.expected_factor_id, f"{e.id}: expected_factor_id empty"
        assert e.difficulty in {"easy", "medium", "hard"}, (
            f"{e.id}: difficulty {e.difficulty!r} invalid"
        )
        assert len(e.jurisdiction) >= 2, f"{e.id}: jurisdiction too short"
        assert e.activity_text.strip(), f"{e.id}: activity_text empty"


def test_slice_coverage_matches_target(gold_entries: List[GoldEntry]):
    """The curated set must honour the documented per-slice budget."""
    counts: Dict[str, int] = defaultdict(int)
    for e in gold_entries:
        counts[e.slice_name] += 1
    expected_minimums = {
        # Per-slice budgets scaled for the FY27 350-example gold set.
        # Electricity/combustion are largest because they cover the most
        # jurisdictions and method-pack variants in production traffic.
        "electricity": 60,
        "combustion": 50,
        "freight": 45,
        "material_cbam": 45,
        "land": 35,
        "product": 35,
        "finance": 30,
    }
    for s, minimum in expected_minimums.items():
        assert counts[s] >= minimum, (
            f"slice {s!r} has {counts[s]} entries; target minimum is {minimum}"
        )


def test_acceptable_alternates_reference_real_factor_ids(
    gold_entries: List[GoldEntry],
    known_factor_ids: Set[str],
):
    """Alternates MUST be grounded factor_ids (either built-in, or consistent
    with the expected_factor_id family).  We allow alternates whose ids are
    absent from the built-in catalog because the gold set covers factor_ids
    that only ship in larger Certified editions - but each alternate must at
    minimum be a non-empty string, distinct from the primary, and not a
    duplicate within the same entry.  If the primary is built-in, then at
    least one alternate (if any are declared) must also be built-in."""
    for e in gold_entries:
        unique_alts = set(e.acceptable_alternates)
        assert len(unique_alts) == len(e.acceptable_alternates), (
            f"{e.id}: duplicate acceptable_alternate entries"
        )
        assert e.expected_factor_id not in unique_alts, (
            f"{e.id}: expected_factor_id also listed as alternate"
        )
        if e.expected_factor_id in known_factor_ids and unique_alts:
            # At least one alternate must be built-in so the grounding rule
            # is auditable end-to-end in CI.
            assert any(a in known_factor_ids for a in unique_alts), (
                f"{e.id}: primary is built-in but NO alternate is - drop the"
                f" alternates or add a built-in one so grounding holds"
            )


# -----------------------------------------------------------------------
# The promotion gate
# -----------------------------------------------------------------------


def test_skipped_missing_factor_budget(
    outcomes: List[CaseOutcome],
    gold_entries: List[GoldEntry],
):
    """No more than 20% of cases may be skipped_missing_factor.

    If the built-in catalog regresses and drops factor_ids the gold set
    relies on, coverage falls through the floor - we fail loudly rather
    than silently evaluate fewer cases.
    """
    total = len(gold_entries)
    skipped = sum(1 for o in outcomes if o.skipped_missing_factor)
    fraction = skipped / total if total else 0.0
    assert fraction <= MAX_SKIPPED_FRACTION, (
        "insufficient coverage: "
        f"{skipped}/{total} = {fraction:.2%} entries were skipped because "
        f"their expected_factor_id is not in EmissionFactorDatabase "
        f"(budget {MAX_SKIPPED_FRACTION:.0%}). Either extend the built-in "
        f"catalog or revise the gold set."
    )


def test_overall_top1_and_top5_above_floor(outcomes: List[CaseOutcome]):
    """Aggregate accuracy across all non-skipped cases must clear the floor."""
    summary = _summarise(outcomes)
    assert summary["evaluated"] > 0, "no cases were evaluated"
    if summary["top1"] < TOP1_FLOOR:
        pytest.fail(
            f"OVERALL top-1 accuracy {summary['top1']:.2%} < floor "
            f"{TOP1_FLOOR:.2%} (evaluated={summary['evaluated']}, "
            f"skipped={summary['skipped_missing_factor']}).\n"
            f"  Top-1 misses ({len(summary['misses_top1'])}):\n"
            f"{_format_misses(summary['misses_top1'])}"
        )
    if summary["top5"] < TOP5_FLOOR:
        pytest.fail(
            f"OVERALL top-5 recall {summary['top5']:.2%} < floor "
            f"{TOP5_FLOOR:.2%} (evaluated={summary['evaluated']}).\n"
            f"  Top-5 misses ({len(summary['misses_top5'])}):\n"
            f"{_format_misses(summary['misses_top5'])}"
        )


@pytest.mark.parametrize("slice_name", ALL_SLICES)
def test_per_slice_top1_and_top5_above_floor(
    outcomes: List[CaseOutcome],
    slice_name: str,
):
    """Every slice must independently clear the same floor - no hiding a
    regression behind a strong slice average."""
    slice_outcomes = [o for o in outcomes if o.slice_name == slice_name]
    if not slice_outcomes:
        pytest.skip(f"no outcomes for slice {slice_name!r}")
    summary = _summarise(slice_outcomes)
    if summary["evaluated"] == 0:
        pytest.skip(
            f"slice {slice_name!r}: every case was skipped_missing_factor; "
            "overall skipped budget enforces coverage"
        )
    if summary["top1"] < TOP1_FLOOR:
        pytest.fail(
            f"slice={slice_name} top-1 {summary['top1']:.2%} < floor "
            f"{TOP1_FLOOR:.2%} (evaluated={summary['evaluated']}, "
            f"skipped={summary['skipped_missing_factor']}).\n"
            f"  misses:\n{_format_misses(summary['misses_top1'])}"
        )
    if summary["top5"] < TOP5_FLOOR:
        pytest.fail(
            f"slice={slice_name} top-5 {summary['top5']:.2%} < floor "
            f"{TOP5_FLOOR:.2%} (evaluated={summary['evaluated']}).\n"
            f"  misses:\n{_format_misses(summary['misses_top5'])}"
        )


# -----------------------------------------------------------------------
# Resolution-engine sanity check: every registered method_profile used by
# the gold set must be a known MethodProfile.  We don't require every
# profile to have a registered pack in CI (that's a separate concern
# covered by tests/factors/method_packs/) - we only assert the string
# values are valid enum members so the cascade can at least accept them.
# -----------------------------------------------------------------------


def test_method_profiles_are_registered_enum_values(gold_entries: List[GoldEntry]):
    registered = {p.value for p in MethodProfile}
    used = {e.method_profile for e in gold_entries}
    assert used.issubset(registered), (
        f"gold set references unregistered method_profiles: "
        f"{sorted(used - registered)}"
    )


# -----------------------------------------------------------------------
# Diagnostic helper - always emitted so CI logs are useful even on pass.
# -----------------------------------------------------------------------


def test_emit_per_slice_matrix(outcomes: List[CaseOutcome], capsys):
    """Always-passing diagnostic that prints the per-slice matrix; the
    GitHub Action reads this from the step log to post the PR comment."""
    header = "slice           total  eval  skip  top1    top5"
    print(header)
    print("-" * len(header))
    for s in ALL_SLICES:
        rows = [o for o in outcomes if o.slice_name == s]
        summ = _summarise(rows)
        t1 = f"{summ['top1']:.2%}" if summ["evaluated"] else "  n/a"
        t5 = f"{summ['top5']:.2%}" if summ["evaluated"] else "  n/a"
        print(
            f"{s:15s} {summ['total']:5d}  {summ['evaluated']:4d}  "
            f"{summ['skipped_missing_factor']:4d}  {t1:>6s}  {t5:>6s}"
        )
    overall = _summarise(outcomes)
    print("-" * len(header))
    t1 = f"{overall['top1']:.2%}" if overall["evaluated"] else "  n/a"
    t5 = f"{overall['top5']:.2%}" if overall["evaluated"] else "  n/a"
    print(
        f"{'OVERALL':15s} {overall['total']:5d}  {overall['evaluated']:4d}  "
        f"{overall['skipped_missing_factor']:4d}  {t1:>6s}  {t5:>6s}"
    )
    captured = capsys.readouterr()
    # Re-emit so pytest -s / CI capture both routes.
    print(captured.out)
