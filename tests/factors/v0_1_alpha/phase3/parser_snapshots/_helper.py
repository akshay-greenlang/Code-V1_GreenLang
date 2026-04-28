# -*- coding: utf-8 -*-
"""Phase 3 — parser snapshot helper.

Provides two surfaces that the per-parser snapshot tests use:

  * :func:`compare_to_snapshot` — load
    ``<snapshot_dir>/<parser_id>__<parser_version>.golden.json``,
    deep-compare with ``parsed_output``, and fail with a precise diff
    message when the parser drifts. The drift detector calls out:

      - changed table-shape (added or removed columns / fields);
      - unit-string drift (case + whitespace + symbol changes);
      - missing citations field;
      - missing licence tag;
      - missing ``row_ref``;
      - missing ``sha256`` / artifact pin.

  * :func:`regenerate_if_env` — when the operator sets
    ``UPDATE_PARSER_SNAPSHOT=1`` (overridable via the ``env_var`` arg),
    write the new golden file and skip the comparison. The skip is
    explicit so a CI run with the env var set is auditable.

Style mirrors ``tests/factors/v0_1_alpha/phase2/_regenerate_openapi_snapshot.py``:
the regen path is a one-shot helper not a normal regression.

Reference: ``docs/factors/PHASE_3_PLAN.md`` §"Wave 1.0 Framework"
("snapshot-test infra"). The helper is the foundation Wave 2 parser
migrations build on; every migrated parser gets one golden + one test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest


__all__ = [
    "compare_to_snapshot",
    "regenerate_if_env",
    "default_snapshot_dir",
    "snapshot_path",
    "DEFAULT_REGEN_ENV_VAR",
    "REQUIRED_PROVENANCE_FIELDS",
]


#: Default environment variable that triggers regeneration. Mirrors the
#: ``UPDATE_OPENAPI_SNAPSHOT`` lever the Phase 2 OpenAPI helper uses.
DEFAULT_REGEN_ENV_VAR = "UPDATE_PARSER_SNAPSHOT"


#: Provenance fields the snapshot helper validates on every parsed row.
#: Drift on any of these is a snapshot failure even if the row payload
#: otherwise matches the golden.
REQUIRED_PROVENANCE_FIELDS: Tuple[str, ...] = (
    "row_ref",
    "licence",
    "raw_artifact_sha256",
    "citations",
)


def default_snapshot_dir() -> Path:
    """Return the canonical golden directory for parser snapshots."""
    return Path(__file__).resolve().parent


def snapshot_path(
    parser_id: str,
    parser_version: str,
    *,
    snapshot_dir: Optional[Path] = None,
) -> Path:
    """Build the canonical golden-file path for a parser+version pair."""
    base = snapshot_dir or default_snapshot_dir()
    safe_id = parser_id.replace("/", "_")
    safe_version = parser_version.replace("/", "_")
    return base / f"{safe_id}__{safe_version}.golden.json"


# ---------------------------------------------------------------------------
# Drift detectors
# ---------------------------------------------------------------------------


def _table_shape(rows: Iterable[Dict[str, Any]]) -> Tuple[Tuple[str, ...], int]:
    """Return (sorted column names, row count) — used to spot shape drift."""
    rows = list(rows)
    if not rows:
        return tuple(), 0
    cols: set = set()
    for row in rows:
        if isinstance(row, dict):
            cols.update(row.keys())
    return tuple(sorted(cols)), len(rows)


def _unit_strings(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """Return all distinct unit strings in the parsed output."""
    out: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        u = row.get("unit") or row.get("unit_symbol") or row.get("unit_urn")
        if u is not None:
            out.append(str(u))
    return out


def _missing_provenance_per_row(
    rows: Iterable[Dict[str, Any]],
    fields: Tuple[str, ...] = REQUIRED_PROVENANCE_FIELDS,
) -> List[Tuple[int, str]]:
    """Return ``[(row_index, missing_field), ...]`` for every drift."""
    misses: List[Tuple[int, str]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        # Provenance can live at the row root OR under ``extraction``.
        bag = dict(row)
        if isinstance(row.get("extraction"), dict):
            for k, v in row["extraction"].items():
                bag.setdefault(k, v)
        for f in fields:
            # ``raw_artifact_sha256`` may be nested under ``extraction.sha256``
            if f == "raw_artifact_sha256":
                if not (
                    bag.get("raw_artifact_sha256")
                    or bag.get("sha256")
                    or bag.get("artifact_sha256")
                ):
                    misses.append((idx, f))
                continue
            if f not in bag or bag[f] in (None, "", [], {}):
                misses.append((idx, f))
    return misses


def _diff_table_shape(
    golden_rows: List[Dict[str, Any]],
    parsed_rows: List[Dict[str, Any]],
) -> Optional[str]:
    """Return a human-readable description of column drift, or None."""
    g_cols, _ = _table_shape(golden_rows)
    p_cols, _ = _table_shape(parsed_rows)
    if g_cols == p_cols:
        return None
    added = sorted(set(p_cols) - set(g_cols))
    removed = sorted(set(g_cols) - set(p_cols))
    parts = []
    if added:
        parts.append(f"added columns: {added}")
    if removed:
        parts.append(f"removed columns: {removed}")
    return "table-shape drift: " + "; ".join(parts)


def _diff_unit_strings(
    golden_rows: List[Dict[str, Any]],
    parsed_rows: List[Dict[str, Any]],
) -> Optional[str]:
    g_units = sorted(set(_unit_strings(golden_rows)))
    p_units = sorted(set(_unit_strings(parsed_rows)))
    if g_units == p_units:
        return None
    return f"unit-string drift: golden={g_units!r} parsed={p_units!r}"


def _diff_missing_provenance(
    parsed_rows: List[Dict[str, Any]],
) -> Optional[str]:
    misses = _missing_provenance_per_row(parsed_rows)
    if not misses:
        return None
    summary: Dict[str, List[int]] = {}
    for idx, field in misses:
        summary.setdefault(field, []).append(idx)
    lines = [f"  - {field}: rows {indices}" for field, indices in summary.items()]
    return "missing required provenance fields:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compare_to_snapshot(
    parser_id: str,
    parser_version: str,
    parsed_output: Any,
    *,
    snapshot_dir: Optional[Path] = None,
) -> None:
    """Compare ``parsed_output`` against the committed golden snapshot.

    Fails the test (via :func:`pytest.fail`) on any drift. The error
    message names the parser, the snapshot path, and the specific drift
    category (table-shape / unit / provenance / payload). On a clean
    match the function returns ``None``.
    """
    path = snapshot_path(parser_id, parser_version, snapshot_dir=snapshot_dir)
    if not path.exists():
        pytest.fail(
            f"snapshot missing for {parser_id}@{parser_version}: "
            f"expected {path}; create one by running the test with "
            f"{DEFAULT_REGEN_ENV_VAR}=1"
        )
    golden_text = path.read_text(encoding="utf-8")
    try:
        golden = json.loads(golden_text)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"golden snapshot at {path} is not valid JSON: {exc!r}"
        )

    # Normalize the comparable shape: both sides serialise to JSON
    # canonically before any deep-compare.
    parsed_canonical = json.loads(
        json.dumps(parsed_output, sort_keys=True, default=str)
    )
    golden_canonical = json.loads(
        json.dumps(golden, sort_keys=True, default=str)
    )

    # Pull the row-list out of either a top-level list or a ``rows`` key.
    def _rows(blob: Any) -> List[Dict[str, Any]]:
        if isinstance(blob, list):
            return [r for r in blob if isinstance(r, dict)]
        if isinstance(blob, dict):
            for key in ("rows", "records", "factors", "data"):
                value = blob.get(key)
                if isinstance(value, list):
                    return [r for r in value if isinstance(r, dict)]
        return []

    g_rows = _rows(golden_canonical)
    p_rows = _rows(parsed_canonical)

    # Run drift detectors before the deep equality check so we surface
    # the most actionable error message.
    drifts: List[str] = []
    for diff_fn in (_diff_table_shape, _diff_unit_strings):
        msg = diff_fn(g_rows, p_rows)
        if msg:
            drifts.append(msg)
    prov_msg = _diff_missing_provenance(p_rows)
    if prov_msg:
        drifts.append(prov_msg)

    if drifts:
        pytest.fail(
            f"parser snapshot drift for {parser_id}@{parser_version} "
            f"({path}):\n  - " + "\n  - ".join(drifts)
            + f"\nRegenerate with {DEFAULT_REGEN_ENV_VAR}=1 if intentional."
        )

    if parsed_canonical != golden_canonical:
        # Build a compact diff message — diff at the row-index level.
        first_drift = None
        for i, (g, p) in enumerate(zip(g_rows, p_rows)):
            if g != p:
                first_drift = (i, g, p)
                break
        msg = f"parser payload drift for {parser_id}@{parser_version}"
        if first_drift is not None:
            i, g, p = first_drift
            msg += (
                f": first divergence at row {i}\n"
                f"  golden: {g!r}\n"
                f"  parsed: {p!r}"
            )
        else:
            msg += (
                f": row counts differ "
                f"(golden={len(g_rows)} parsed={len(p_rows)})"
            )
        pytest.fail(
            msg + f"\nRegenerate with {DEFAULT_REGEN_ENV_VAR}=1 if intentional."
        )


def regenerate_if_env(
    parser_id: str,
    parser_version: str,
    parsed_output: Any,
    *,
    env_var: str = DEFAULT_REGEN_ENV_VAR,
    snapshot_dir: Optional[Path] = None,
) -> None:
    """If ``env_var`` is set in the environment, write the new golden.

    On regeneration the function calls :func:`pytest.skip` so the
    surrounding test is recorded as ``skipped`` (not silently passing
    on a re-write) — this mirrors the Phase 2 OpenAPI regen lever.
    """
    flag = os.environ.get(env_var, "").strip()
    if flag not in {"1", "true", "True", "yes", "YES"}:
        return
    path = snapshot_path(parser_id, parser_version, snapshot_dir=snapshot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(parsed_output, indent=2, sort_keys=True, default=str)
    path.write_text(payload + "\n", encoding="utf-8")
    pytest.skip(
        f"regenerated parser snapshot for {parser_id}@{parser_version} "
        f"-> {path}"
    )
