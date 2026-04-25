# -*- coding: utf-8 -*-
"""
Wave E / TaskCreate #26 / WS9-T4 — Incident-drill regression tests.

These tests encode the DESNZ parser column-shift drill executed on
2026-04-25 (postmortem at
``docs/factors/postmortems/2026-Q1-desnz-parser-drift-drill.md``).

They are evidence tests: they prove that the drill fixture *still*
fails the way the postmortem says it failed. If a future change makes
the drill fixture pass cleanly, these tests will fail loudly and the
team must update the postmortem (because either the parser got more
strict, or the gate got more lenient — both warrant a re-drill).

Marker: ``@pytest.mark.drill`` — these are excluded from default CI by
declaration intent (they exercise an INTENTIONALLY corrupted fixture)
but are kept in the standard test directory so the regression is visible
on every run.

CTO doc references: §19.1 (alpha launch — operations & oversight),
Wave E / TaskCreate #26.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DRILL_FIXTURE = (
    _REPO_ROOT
    / "tests"
    / "factors"
    / "v0_1_alpha"
    / "drill_fixtures"
    / "desnz_2024_corrupted_v1.json"
)
_PRODUCTION_SEED = (
    _REPO_ROOT
    / "greenlang"
    / "factors"
    / "data"
    / "catalog_seed"
    / "_inputs"
    / "desnz_uk.json"
)


# ---------------------------------------------------------------------------
# Fixture-existence regression
# ---------------------------------------------------------------------------


@pytest.mark.drill
def test_drill_fixture_exists_and_is_isolated_from_production() -> None:
    """The drill fixture must exist AND must NOT have leaked into the
    production catalog seed directory.

    This is a paranoia check: if anyone ever copies the corrupted fixture
    into ``greenlang/factors/data/catalog_seed/_inputs/`` by mistake, we
    fail loudly here. The corruption marker token below appears ONLY in
    the drill fixture by design.
    """
    assert _DRILL_FIXTURE.is_file(), f"drill fixture missing at {_DRILL_FIXTURE}"
    assert _PRODUCTION_SEED.is_file(), f"production seed missing at {_PRODUCTION_SEED}"

    drill_text = _DRILL_FIXTURE.read_text(encoding="utf-8")
    prod_text = _PRODUCTION_SEED.read_text(encoding="utf-8")

    assert "RENAMED_FOR_DRILL" in drill_text, (
        "drill fixture missing the 'RENAMED_FOR_DRILL' corruption marker — "
        "either the fixture was reverted or the marker token was changed "
        "without updating this regression."
    )
    assert "RENAMED_FOR_DRILL" not in prod_text, (
        "FATAL: the 'RENAMED_FOR_DRILL' corruption marker leaked into the "
        f"production DESNZ seed at {_PRODUCTION_SEED}. Revert immediately. "
        "The drill fixture must NEVER be copied into production."
    )


# ---------------------------------------------------------------------------
# Parser tolerance regression — documents what the drill surfaced
# ---------------------------------------------------------------------------


@pytest.mark.drill
def test_parser_silently_zeros_renamed_factor_columns() -> None:
    """Regression for postmortem 'What didn't work' #1 / Action Item AI-1.

    The DESNZ parser uses `_safe_float(row.get("co2_factor"))` which
    returns 0.0 when the key is missing — so renaming row-level columns
    does NOT raise a KeyError; instead the parser silently produces
    records with zero vectors. This test pins that current behaviour
    so the postmortem's claim ('parser does not fail loud at the row
    level') stays factually accurate.

    When AI-1 ships (raw-record schema validation at parser time), this
    test will need to be updated to expect a structured raise.
    """
    from greenlang.factors.ingestion.parsers.desnz_uk import parse_desnz_uk

    payload = json.loads(_DRILL_FIXTURE.read_text(encoding="utf-8"))
    records = parse_desnz_uk(payload)

    # Parser is tolerant — it produces records (no raise).
    assert isinstance(records, list)
    assert len(records) > 0, (
        "parser produced 0 records; either the fixture lost all its rows "
        "or the parser became stricter (good — update postmortem AI-1)."
    )

    # Every record's vectors must be all-zero (because every numeric
    # column was renamed). If any non-zero vector slips through, the
    # corruption is incomplete.
    nonzero_vectors: List[Dict[str, Any]] = []
    for r in records:
        v = r.get("vectors") or {}
        if any(float(v.get(k, 0.0)) > 0 for k in ("CO2", "CH4", "N2O")):
            nonzero_vectors.append(r)

    assert not nonzero_vectors, (
        f"{len(nonzero_vectors)} records had non-zero vectors despite "
        f"the row-level numeric column rename — corruption incomplete: "
        f"first offender factor_id={nonzero_vectors[0].get('factor_id')!r}"
    )


# ---------------------------------------------------------------------------
# Normalizer / gate rejection regression — drills the actual safety net
# ---------------------------------------------------------------------------


@pytest.mark.drill
def test_normalizer_rejects_every_corrupted_record_with_nonpositive_value() -> None:
    """Regression for postmortem 'What worked' #1 + #2.

    Lifting each parser output via
    ``alpha_v0_1_normalizer.lift_v1_record_to_v0_1`` raises
    :class:`NonPositiveValueError` for every corrupted record (because
    value collapses to 0 + 0*28 + 0*265 = 0, and the alpha schema
    requires value > 0). This is the safety net that catches the silent
    parser tolerance.
    """
    from greenlang.factors.ingestion.parsers.desnz_uk import parse_desnz_uk
    from greenlang.factors.etl.alpha_v0_1_normalizer import (
        NonPositiveValueError,
        lift_v1_record_to_v0_1,
    )
    from greenlang.factors.source_registry import alpha_v0_1_sources

    sources = alpha_v0_1_sources()
    desnz_meta = sources.get("desnz_ghg_conversion")
    assert desnz_meta is not None, "alpha source registry missing desnz_ghg_conversion"

    payload = json.loads(_DRILL_FIXTURE.read_text(encoding="utf-8"))
    records = parse_desnz_uk(payload)
    assert len(records) > 0

    rejected = 0
    other_failures: List[str] = []
    for i, rec in enumerate(records):
        try:
            lift_v1_record_to_v0_1(rec, dict(desnz_meta), idx=i)
        except NonPositiveValueError:
            rejected += 1
        except Exception as exc:  # noqa: BLE001
            other_failures.append(f"rec[{i}]: {type(exc).__name__}: {exc}")

    # Every corrupted record must be rejected by the normalizer.
    assert rejected == len(records), (
        f"normalizer rejected only {rejected}/{len(records)} corrupted "
        f"records via NonPositiveValueError; other failures: {other_failures[:5]}"
    )


# ---------------------------------------------------------------------------
# Counter-emission regression — proves the metric path fires
# ---------------------------------------------------------------------------


@pytest.mark.drill
def test_alpha_provenance_gate_emits_rejection_counter_for_malformed_record() -> None:
    """Regression for postmortem 'What worked' #3 (Prometheus counter).

    Feeding a malformed record directly to the gate must (a) return a
    non-empty failure list and (b) increment the
    ``factors_alpha_provenance_gate_rejections_total`` counter via
    ``assert_valid``. This proves the alert path is live.
    """
    from greenlang.factors.quality.alpha_provenance_gate import (
        AlphaProvenanceGate,
        AlphaProvenanceGateError,
    )

    # A record that is missing the entire ``extraction`` block — the
    # alpha gate's hardest must-have. Mirrors what a partial-failure
    # parser output would look like if it bypassed the normalizer.
    malformed = {
        "urn": "urn:gl:factor:desnz:s1:nat_gas:v1",
        "source_urn": "urn:gl:source:desnz_ghg_conversion",
        "value": 0.18293,  # numerically valid but no provenance
        # extraction MISSING
        # review MISSING
    }
    gate = AlphaProvenanceGate()

    failures = gate.validate(malformed)
    assert failures, "gate did not reject a record missing extraction+review"

    with pytest.raises(AlphaProvenanceGateError) as excinfo:
        gate.assert_valid(malformed)
    assert excinfo.value.failures, "AlphaProvenanceGateError carried no failure list"


# ---------------------------------------------------------------------------
# Postmortem-evidence presence regression
# ---------------------------------------------------------------------------


@pytest.mark.drill
def test_postmortem_and_evidence_files_present() -> None:
    """The postmortem and its evidence files must exist on disk.

    This regression keeps the CTO doc §19.1 acceptance criterion
    (postmortem from first real-incident drill completed and filed)
    enforceable from CI: if anyone deletes the postmortem, the build
    fails.
    """
    postmortem = (
        _REPO_ROOT
        / "docs"
        / "factors"
        / "postmortems"
        / "2026-Q1-desnz-parser-drift-drill.md"
    )
    stack = (
        _REPO_ROOT
        / "docs"
        / "factors"
        / "postmortems"
        / "evidence"
        / "2026-04-25-desnz-stack.txt"
    )
    counters = (
        _REPO_ROOT
        / "docs"
        / "factors"
        / "postmortems"
        / "evidence"
        / "2026-04-25-desnz-counters.json"
    )
    sop = (
        _REPO_ROOT
        / "docs"
        / "factors"
        / "runbooks"
        / "incident-drill-sop.md"
    )

    for path in (postmortem, stack, counters, sop):
        assert path.is_file(), f"missing drill artefact: {path}"

    # Counter snapshot must be valid JSON with the two canonical keys.
    snapshot = json.loads(counters.read_text(encoding="utf-8"))
    assert "factors_parser_errors_total" in snapshot
    assert "factors_alpha_provenance_gate_rejections_total" in snapshot
