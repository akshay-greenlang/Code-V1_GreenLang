# -*- coding: utf-8 -*-
"""Phase 2 Block 1 gate — alpha catalog vs frozen schema + Pydantic mirror.

CTO Phase 2 §2.1 acceptance: every v0.1 alpha factor record validates
against ``config/schemas/factor_record_v0_1.schema.json`` AND the typed
Pydantic mirror :class:`FactorRecordV0_1`. Any divergence between the
two layers is itself a failure (the mirrors must agree on every record).

Coverage requirements
---------------------
1. Catalog non-empty AND covers every alpha source. The CTO brief
   names sources with the slugs ``ipcc-ar6``, ``defra-2025``,
   ``epa-ghg-hub``, ``epa-egrid``, ``india-cea-baseline``,
   ``eu-cbam-defaults``; the actual repo seeds use the registry
   directory names ``ipcc_2006_nggi``, ``desnz_ghg_conversion``,
   ``epa_hub``, ``egrid``, ``india_cea_co2_baseline``,
   ``cbam_default_values`` — these correspond 1:1 to the brief sources
   and are the canonical names in the source registry. We assert each
   directory has at least one record.
2. Every record has populated fields in the 7 required CTO field
   groups (identity / value-unit / context / quality / licence /
   lineage / lifecycle). ``quality`` is special-cased: it is OPTIONAL
   in alpha (CTO doc §19.1 — required from v0.9+), so absence is
   ALLOWED but the field group is still considered "satisfied" because
   the platform tolerates the OptIonal.

CI marker: this is the Phase 2 Block 1 gate. It MUST stay green on
master.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jsonschema
import pytest
from pydantic import ValidationError

from greenlang.factors.schemas.factor_record_v0_1 import (
    FactorRecordV0_1,
    FrozenSchemaPath,
)

logger = logging.getLogger(__name__)


# tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py
#                              -> repo_root
_REPO_ROOT = Path(__file__).resolve().parents[4]

_CATALOG_DIR = _REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed_v0_1"

# Required source directories in the alpha catalog. Every one of these
# MUST have at least one record. The brief's CTO-doc slugs map 1:1 to
# these registry source_ids.
_REQUIRED_SOURCE_DIRS: Tuple[str, ...] = (
    "ipcc_2006_nggi",          # CTO brief: ipcc-ar6
    "desnz_ghg_conversion",    # CTO brief: defra-2025
    "epa_hub",                 # CTO brief: epa-ghg-hub
    "egrid",                   # CTO brief: epa-egrid
    "india_cea_co2_baseline",  # CTO brief: india-cea-baseline
    "cbam_default_values",     # CTO brief: eu-cbam-defaults
)

# CTO Phase 2 §2.1 — 7 required field groups. Each entry maps a group
# name to (always-required fields, optional-but-named fields).
_FIELD_GROUPS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "identity":     (("urn", "source_urn", "factor_pack_urn"), ("factor_id_alias",)),
    "value-unit":   (("value", "unit_urn"), ()),
    "context":      (
        ("name", "description", "category", "geography_urn",
         "methodology_urn", "boundary", "resolution"),
        (),
    ),
    "quality":      ((), ("uncertainty",)),  # uncertainty is optional in alpha
    "licence":      (("licence",), ("licence_constraints",)),
    "lineage":      (("citations", "extraction"), ("tags", "supersedes_urn")),
    "lifecycle":    (("review", "published_at"), ("deprecated_at",)),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_all_records() -> List[Tuple[str, int, Dict[str, Any]]]:
    """Walk the alpha catalog and return ``(source_id, idx, record)`` tuples."""
    out: List[Tuple[str, int, Dict[str, Any]]] = []
    if not _CATALOG_DIR.is_dir():
        return out
    for child in sorted(_CATALOG_DIR.iterdir()):
        if not child.is_dir():
            continue
        seed = child / "v1.json"
        if not seed.is_file():
            continue
        try:
            payload = json.loads(seed.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            pytest.fail(f"{seed} is not valid JSON: {exc}")
        records = payload.get("records") or []
        for idx, rec in enumerate(records):
            if isinstance(rec, dict):
                out.append((child.name, idx, rec))
    return out


def _load_records_by_source() -> Dict[str, List[Dict[str, Any]]]:
    """Return ``{source_id: [record, ...]}`` for the alpha catalog."""
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for src, _, rec in _load_all_records():
        by_src.setdefault(src, []).append(rec)
    return by_src


def _check_field_group(record: Dict[str, Any], group: str) -> List[str]:
    """Return [] if the field group is satisfied, else a list of issues.

    "Satisfied" means every always-required field in the group is
    present and non-empty on the record. Optional-but-named fields are
    allowed to be missing (or null). This mirrors the CTO Phase 2 §2.1
    definition.
    """
    required, _optional = _FIELD_GROUPS[group]
    issues: List[str] = []
    for field_name in required:
        if field_name not in record:
            issues.append(f"missing field {field_name!r}")
            continue
        val = record[field_name]
        if val in (None, ""):
            issues.append(f"empty field {field_name!r}")
            continue
        # Containers must be non-empty.
        if isinstance(val, (list, tuple, dict)) and len(val) == 0:
            issues.append(f"empty container in field {field_name!r}")
    return issues


def _build_validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(FrozenSchemaPath.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_alpha_catalog_is_non_empty_and_covers_every_required_source() -> None:
    """The catalog must cover every alpha source named in the CTO brief.

    The 6 alpha sources in the brief map 1:1 to the registry source
    directory names listed in :data:`_REQUIRED_SOURCE_DIRS`.
    """
    by_src = _load_records_by_source()
    missing = [s for s in _REQUIRED_SOURCE_DIRS if not by_src.get(s)]
    assert not missing, (
        "alpha catalog is missing records for required sources: "
        f"{missing}; catalog dir: "
        f"{_CATALOG_DIR.relative_to(_REPO_ROOT)}"
    )
    total = sum(len(v) for v in by_src.values())
    assert total >= len(_REQUIRED_SOURCE_DIRS), (
        f"catalog has only {total} records across all sources; "
        "expected at least 1 per source"
    )


def test_every_v0_1_record_validates_against_frozen_schema() -> None:
    """Every catalog record passes BOTH the JSON Schema and the Pydantic mirror.

    Either both succeed OR both fail — any divergence is itself a
    failure (it would mean the Pydantic mirror has drifted from the
    frozen contract, which the CI is here to prevent).
    """
    records = _load_all_records()
    if not records:
        pytest.fail(
            f"no records found under {_CATALOG_DIR.relative_to(_REPO_ROOT)}"
        )

    validator = _build_validator()

    schema_failures: List[str] = []
    pydantic_failures: List[str] = []
    divergences: List[str] = []

    for src, idx, rec in records:
        # Layer 1 — JSON Schema.
        schema_errors = list(validator.iter_errors(rec))
        schema_ok = not schema_errors
        if not schema_ok:
            top = "; ".join(
                f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                for e in schema_errors[:3]
            )
            schema_failures.append(f"{src}#{idx}: {top}")

        # Layer 2 — Pydantic mirror.
        try:
            FactorRecordV0_1(**rec)
            pydantic_ok = True
            pydantic_first = ""
        except ValidationError as exc:
            pydantic_ok = False
            errs = exc.errors()
            pydantic_first = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in errs[:3]
            )
            pydantic_failures.append(f"{src}#{idx}: {pydantic_first}")

        # Divergence detection.
        if schema_ok != pydantic_ok:
            divergences.append(
                f"{src}#{idx}: schema_ok={schema_ok}, "
                f"pydantic_ok={pydantic_ok}; "
                f"schema_msg={'; '.join(str(e.message) for e in schema_errors[:2]) or '-'}; "
                f"pydantic_msg={pydantic_first or '-'}"
            )

    # Build a single, dense assertion message.
    msgs: List[str] = []
    if schema_failures:
        msgs.append(
            f"{len(schema_failures)} JSON-Schema failure(s):\n  - "
            + "\n  - ".join(schema_failures[:10])
        )
    if pydantic_failures:
        msgs.append(
            f"{len(pydantic_failures)} Pydantic-mirror failure(s):\n  - "
            + "\n  - ".join(pydantic_failures[:10])
        )
    if divergences:
        msgs.append(
            f"{len(divergences)} divergence(s) between schema and "
            f"Pydantic mirror:\n  - " + "\n  - ".join(divergences[:10])
        )

    assert not msgs, (
        "Phase 2 Block 1 gate FAILED:\n\n" + "\n\n".join(msgs)
    )


def test_every_record_satisfies_seven_required_field_groups() -> None:
    """Every catalog record populates every required CTO field group.

    Field-group definitions follow CTO Phase 2 §2.1. ``quality`` is
    optional in alpha (uncertainty quantification is required from
    v0.9+); the test still iterates it for completeness but tolerates
    absence on alpha records.
    """
    records = _load_all_records()
    failures: List[str] = []
    for src, idx, rec in records:
        for group in _FIELD_GROUPS:
            issues = _check_field_group(rec, group)
            if issues:
                failures.append(
                    f"{src}#{idx}: group={group} -> {'; '.join(issues)}"
                )
    assert not failures, (
        f"{len(failures)} field-group violation(s):\n  - "
        + "\n  - ".join(failures[:20])
    )
