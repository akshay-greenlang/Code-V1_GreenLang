# -*- coding: utf-8 -*-
"""Regression test: every catalog-seed factor URN parses canonically.

Pre-Phase-0 audit cleanup. The CTO doc (Section 6.1.1) establishes
``greenlang.factors.ontology.urn.parse`` as the authoritative URN
validator: namespace MUST be lowercase, id segments MUST be lowercase
(except ``T`` / ``Z`` ISO-8601 timestamp markers in the id).

The schema-driven provenance gate uses a regex that historically was
permissive enough to accept uppercase namespace segments like ``IN``,
``CBAM``, ``DESNZ``. This test plugs that gap by running every ``urn``
field in every alpha catalog seed through the canonical parser, so the
gate (schema regex) and the parser stay in lockstep.

Failure surface: if a future seed-generation regression reintroduces
uppercase namespaces (e.g. via a parser that emits country codes raw),
this test will fail loudly with the offending URN and source path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Tuple

import pytest

from greenlang.factors.ontology.urn import InvalidUrnError, parse


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEEDS_DIR = (
    _REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed_v0_1"
)


def _iter_seed_records() -> Iterator[Tuple[str, int, str]]:
    """Yield (source_id, record_index, urn) tuples across every seed."""
    if not _SEEDS_DIR.is_dir():
        return
    for src_dir in sorted(_SEEDS_DIR.iterdir()):
        if not src_dir.is_dir():
            continue
        seed = src_dir / "v1.json"
        if not seed.is_file():
            continue
        try:
            payload = json.loads(seed.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = payload.get("records") or []
        if not isinstance(records, list):
            continue
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            urn = rec.get("urn")
            if isinstance(urn, str) and urn:
                yield (src_dir.name, i, urn)


_SEED_URNS = list(_iter_seed_records())


def test_alpha_seeds_are_present() -> None:
    """Sanity: the seed dir must contain at least one factor URN.

    A regression that empties the seeds would silently neuter the
    parametrised URN-shape test below; this guard fails loudly first.
    """
    assert _SEED_URNS, (
        f"no seed URNs discovered under {_SEEDS_DIR}; alpha catalog "
        "appears to be empty or unreadable"
    )


@pytest.mark.parametrize(
    "source_id,record_index,urn",
    _SEED_URNS,
    ids=[f"{s}:rec{i}" for s, i, _ in _SEED_URNS],
)
def test_seed_urn_parses_canonically(
    source_id: str, record_index: int, urn: str
) -> None:
    """Every catalog-seed factor URN MUST pass the canonical parser.

    The canonical parser enforces lowercase namespace + id (the URN
    spec rule the schema regex did not enforce in the first alpha
    backfill). A failure here means a seed-generation regression has
    re-introduced an invalid URN — fix the upstream generator
    (``greenlang.factors.etl.alpha_v0_1_normalizer.coerce_factor_id_to_urn``)
    rather than relaxing the parser.
    """
    try:
        parsed = parse(urn)
    except InvalidUrnError as exc:
        pytest.fail(
            f"seed URN failed canonical parse — source={source_id} "
            f"record_index={record_index} urn={urn!r}: {exc}"
        )
    assert parsed.kind == "factor", (
        f"expected factor URN, got kind={parsed.kind!r}: {urn!r}"
    )


def test_no_alpha_seed_urn_has_uppercase_namespace() -> None:
    """Explicit guard for the original CTO failure.

    The first batch of alpha seeds shipped URNs like
    ``urn:gl:factor:india-cea-co2-baseline:IN:...`` where the namespace
    segment ``IN`` violates the URN spec. This test asserts the
    namespace segment of every factor URN is strictly lowercase
    (``[a-z0-9._-]+``). Uppercase ASCII letters are explicitly
    forbidden in the namespace.
    """
    offenders: list[tuple[str, int, str]] = []
    for source_id, idx, urn in _SEED_URNS:
        # body after 'urn:gl:factor:' is <source>:<namespace>:<id>:v<version>
        body = urn[len("urn:gl:factor:"):]
        parts = body.split(":")
        if len(parts) < 3:
            offenders.append((source_id, idx, urn))
            continue
        namespace = parts[1]
        if any(ch.isascii() and ch.isupper() for ch in namespace):
            offenders.append((source_id, idx, urn))
    assert not offenders, (
        "alpha seed URNs with uppercase namespace segment "
        f"({len(offenders)} offenders): "
        + "; ".join(
            f"{s}:rec{i}={u}" for s, i, u in offenders[:5]
        )
    )
