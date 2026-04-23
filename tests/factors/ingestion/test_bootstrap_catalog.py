# -*- coding: utf-8 -*-
"""
Tests for the catalog bootstrap orchestrator.

These tests enforce the posture rules recorded in
``docs/legal/source_rights_matrix.md`` — the whole point of Wave 2.5 is
"the default catalog has real factors" WITHOUT accidentally shipping
license-encumbered values.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from greenlang.factors.ingestion.bootstrap import (
    SEED_DIR,
    SOURCE_SPECS,
    bootstrap_catalog,
    count_seed_factors,
    load_seed_envelopes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_bootstrap_once():
    """Idempotent: re-runs the bootstrap and returns the report."""
    return bootstrap_catalog()


def _all_seed_factors() -> List[dict]:
    """Flatten every factor across every envelope on disk."""
    flat = []
    for env in load_seed_envelopes():
        flat.extend(env.get("factors") or [])
    return flat


# ---------------------------------------------------------------------------
# License posture tests
# ---------------------------------------------------------------------------


_LICENSED_EMBEDDED_SOURCES = {
    # Per docs/legal/source_rights_matrix.md — must NOT ship factor values
    # in the default catalog.
    "ghgp_method_refs",
    "tcr_grp_defaults",
    "green_e_residual",
    "green_e_residual_mix",
    "ec3_buildings_epd",
    "glec_framework",
    "pcaf_global_std_v2",
    "pact_pathfinder",
    "lsr_removals",
}


_SAFE_TO_CERTIFY_SOURCES = {
    "epa_hub",
    "egrid",
    "desnz_ghg_conversion",
    "india_cea_co2_baseline",
    "india_ccts_baselines",
    "aib_residual_mix_eu",
    "australia_nga_factors",
    "japan_meti_electric_emission_factors",
    "eu_cbam",
    # IPCC is "Needs-Legal-Review" but ships as preview per spec
    "ipcc_2006_nggi",
}


@pytest.fixture(scope="module", autouse=True)
def _bootstrap_once():
    """Runs the bootstrap once per test module so every test operates on
    real disk state."""
    _run_bootstrap_once()
    yield


def test_bootstrap_excludes_licensed_sources():
    """Licensed-Embedded / Blocked sources must NOT appear as seed files on
    disk. Their parsers stay, but outputs never land in the default catalog."""
    seeded_source_dirs = {p.name for p in SEED_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")}
    leaks = _LICENSED_EMBEDDED_SOURCES & seeded_source_dirs
    assert not leaks, (
        f"Licensed-Embedded sources leaked into default catalog: {leaks}. "
        f"These must stay BYO-connector-only per legal matrix."
    )


def test_bootstrap_includes_open_sources():
    """Every Safe-to-Certify source with an available seed input must
    produce a seed envelope file."""
    seeded_source_dirs = {p.name for p in SEED_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")}
    missing = _SAFE_TO_CERTIFY_SOURCES - seeded_source_dirs
    assert not missing, (
        f"Expected Safe-to-Certify sources are missing from catalog_seed/: "
        f"{missing}. Run `python scripts/bootstrap_catalog.py run`."
    )


def test_bootstrap_envelopes_carry_attribution():
    """Every envelope must carry the attribution string mandated by its
    source's license (OGL-UK-v3, CC-BY-4.0, etc.)."""
    for env in load_seed_envelopes():
        assert env.get("attribution_text"), (
            f"Envelope {env.get('__seed_file')} missing attribution_text"
        )
        assert env.get("license_class"), (
            f"Envelope {env.get('__seed_file')} missing license_class"
        )


# ---------------------------------------------------------------------------
# Catalog size test
# ---------------------------------------------------------------------------


def test_catalog_size_post_bootstrap():
    """The whole point of Wave 2.5: the default catalog has >= 1000 factors
    after bootstrap. Before Wave 2.5 this number was 8."""
    seed_total = count_seed_factors()
    assert seed_total >= 1000, (
        f"catalog_seed factor total is {seed_total} — expected >= 1000 "
        f"after full bootstrap"
    )

    # Also check the EmissionFactorDatabase path
    from greenlang.data.emission_factor_database import EmissionFactorDatabase

    db = EmissionFactorDatabase(enable_cache=False)
    cat_size = db.catalog_size()
    assert cat_size >= 1000, (
        f"EmissionFactorDatabase.catalog_size()={cat_size}, expected >= 1000"
    )


# ---------------------------------------------------------------------------
# License drift gate — no Licensed/Customer-Private/OEM in any factor row
# ---------------------------------------------------------------------------


def test_no_license_class_drift():
    """Every factor in every seed envelope must have
    ``redistribution_class`` set to a value compatible with the default
    catalog — i.e. NOT in (licensed_embedded, customer_private,
    oem_redistributable)."""
    forbidden = {"licensed_embedded", "customer_private", "oem_redistributable", "licensed", "restricted"}
    drift = []
    for env in load_seed_envelopes():
        for rec in env.get("factors") or []:
            rc = rec.get("redistribution_class")
            if rc in forbidden:
                drift.append((env.get("source_id"), rec.get("factor_id"), rc))
    assert not drift, (
        f"Found {len(drift)} records with drift-incompatible "
        f"redistribution_class. First 5: {drift[:5]}"
    )


# ---------------------------------------------------------------------------
# Canonical demo resolves
# ---------------------------------------------------------------------------


def test_canonical_demo_resolves():
    """The canonical Wave 2.5 pain point: resolving
    ``12,500 kWh India FY27 corporate_scope2_location_based`` must return a
    concrete India CEA factor, NOT the tier-7 default placeholder.

    This test is a catalog-content test, not a resolver test — if the
    seed catalog carries the expected India CEA factor_id pattern, we
    consider the demo "resolvable". Actual resolver tuning lives in
    ``tests/factors/matching/test_gold_eval_gate.py``.
    """
    india_cea_factors = [
        rec for rec in _all_seed_factors()
        if (rec.get("source_id") == "india_cea_co2_baseline"
            and rec.get("geography") == "IN")
    ]
    assert india_cea_factors, (
        "No India CEA factors present in catalog_seed. Resolver cannot "
        "possibly answer the canonical India FY27 demo."
    )
    # Must have FY27 coverage (2026-27 or 2027-28 fiscal year)
    fy27_hits = [
        f for f in india_cea_factors
        if "2026-27" in (f.get("factor_id") or "")
        or "2027-28" in (f.get("factor_id") or "")
    ]
    assert fy27_hits, (
        "India CEA factors present but none cover FY27 (2026-27). "
        "Canonical demo requires a factor whose valid_from spans India "
        "FY27 (1 Apr 2026 – 31 Mar 2027)."
    )
    # FY27 factor must be scope 2 + location-based
    for rec in fy27_hits:
        assert rec.get("scope") == "2", (
            f"India CEA FY27 factor {rec.get('factor_id')!r} has "
            f"scope={rec.get('scope')!r} — must be 2"
        )


# ---------------------------------------------------------------------------
# N5 mandatory-field gate
# ---------------------------------------------------------------------------


def test_every_factor_has_n5_required_fields():
    """Every ingested factor must carry the 5 mandatory fields: valid_from,
    source_version, jurisdiction.country (or geography), unit, factor_status.
    The bootstrap already enforces this on write; this test is a regression
    guard against manual edits to seed files."""
    missing_per_source = {}
    for env in load_seed_envelopes():
        src = env.get("source_id")
        for rec in env.get("factors") or []:
            provenance = rec.get("provenance") or {}
            miss = []
            if not rec.get("valid_from"):
                miss.append("valid_from")
            if not (
                rec.get("source_release")
                or rec.get("release_version")
                or provenance.get("version")
            ):
                miss.append("source_version")
            if not rec.get("geography"):
                miss.append("jurisdiction_country/geography")
            if not rec.get("unit"):
                miss.append("denominator_unit")
            if not rec.get("factor_status"):
                miss.append("status")
            if miss:
                missing_per_source.setdefault(src, []).append(
                    (rec.get("factor_id"), miss)
                )
    assert not missing_per_source, (
        f"N5 gate violations detected: "
        f"{ {s: v[:3] for s, v in missing_per_source.items()} }"
    )


# ---------------------------------------------------------------------------
# Offline contract: every registered spec has a clear skip-reason when
# no seed input is on disk.
# ---------------------------------------------------------------------------


def test_every_source_spec_reachable():
    """Smoke check: every registered SourceSpec is either ingested or
    has a clear skip reason — no silent drops."""
    report = bootstrap_catalog()
    known_source_ids = {s.source_id for s in SOURCE_SPECS}
    touched = (
        {r.source_id for r in report.ingested}
        | {r.source_id for r in report.skipped}
        | {r.source_id for r in report.errored}
    )
    missing = known_source_ids - touched
    assert not missing, (
        f"These SourceSpec entries were never visited by bootstrap_catalog: "
        f"{missing}"
    )


def test_stats_command_produces_nonempty_output(capsys):
    """The CLI stats command must produce a non-empty coverage matrix.

    Import the script by file path — not all test environments put
    repo-root/scripts on PYTHONPATH.
    """
    import importlib.util

    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "bootstrap_catalog.py"
    spec = importlib.util.spec_from_file_location("bootstrap_catalog_cli", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    class _Args:
        pass

    rc = mod.cmd_stats(_Args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "factor family coverage" in out.lower()
    assert "jurisdiction coverage" in out.lower()
