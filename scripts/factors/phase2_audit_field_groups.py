# -*- coding: utf-8 -*-
"""Phase 2 — alpha catalog field-group coverage audit.

CLI script that prints a coverage matrix of how many records populate
each of the seven CTO Phase 2 §2.1 field groups, broken down by alpha
source. Used by:

* gl-tech-writer when refreshing
  ``docs/factors/PHASE_2_KPI_DASHBOARD.md``.
* gl-spec-guardian when investigating Block 1 gate failures.
* gl-formula-library-curator before authoring new factor seeds.

Usage::

    python scripts/factors/phase2_audit_field_groups.py
    python scripts/factors/phase2_audit_field_groups.py --json
    python scripts/factors/phase2_audit_field_groups.py --catalog-dir /path/to/catalog_seed_v0_1
    python scripts/factors/phase2_audit_field_groups.py --fail-on-missing

Exit codes:
    0 — every required field group is satisfied on every record (or
        the catalog is empty in dev mode).
    1 — at least one record fails a required group AND
        ``--fail-on-missing`` is set.

CTO Phase 2 §2.1 field groups (seven total):
    identity, value-unit, context, quality, licence, lineage,
    lifecycle.

``quality`` is OPTIONAL in alpha (uncertainty quantification is
required from v0.9+ per CTO doc §19.1). The script reports the
populated rate even though absence is not a failure in alpha.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# scripts/factors/phase2_audit_field_groups.py -> repo_root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATALOG_DIR = (
    _REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed_v0_1"
)


# CTO Phase 2 §2.1 — required + optional fields per group.
# Mirrors tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py
_FIELD_GROUPS: "OrderedDict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]" = OrderedDict(
    [
        ("identity",     (("urn", "source_urn", "factor_pack_urn"), ("factor_id_alias",))),
        ("value-unit",   (("value", "unit_urn"), ())),
        ("context",      (
            ("name", "description", "category", "geography_urn",
             "methodology_urn", "boundary", "resolution"),
            (),
        )),
        ("quality",      ((), ("uncertainty",))),  # optional in alpha
        ("licence",      (("licence",), ("licence_constraints",))),
        ("lineage",      (("citations", "extraction"), ("tags", "supersedes_urn"))),
        ("lifecycle",    (("review", "published_at"), ("deprecated_at",))),
    ]
)

# Groups whose required-fields list is empty (i.e. fully optional in
# alpha) — we report on populated rate but never count absence as a
# failure.
_OPTIONAL_IN_ALPHA = {"quality"}


def _check_required(record: Dict[str, Any], group: str) -> bool:
    """Return True iff every always-required field in the group is set."""
    required, _ = _FIELD_GROUPS[group]
    if not required:
        return True
    for field_name in required:
        if field_name not in record:
            return False
        val = record[field_name]
        if val in (None, ""):
            return False
        if isinstance(val, (list, tuple, dict)) and len(val) == 0:
            return False
    return True


def _check_any_populated(record: Dict[str, Any], group: str) -> bool:
    """Return True iff ANY field in the group (required OR optional) is set."""
    required, optional = _FIELD_GROUPS[group]
    for field_name in required + optional:
        val = record.get(field_name)
        if val not in (None, "") and not (
            isinstance(val, (list, tuple, dict)) and len(val) == 0
        ):
            return True
    return False


def _walk_catalog(catalog_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Return ``{source_id: [record, ...]}`` under ``catalog_dir``."""
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    if not catalog_dir.is_dir():
        return by_src
    for child in sorted(catalog_dir.iterdir()):
        if not child.is_dir():
            continue
        seed = child / "v1.json"
        if not seed.is_file():
            continue
        try:
            payload = json.loads(seed.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        records = payload.get("records") or []
        rec_list = [r for r in records if isinstance(r, dict)]
        if rec_list:
            by_src[child.name] = rec_list
    return by_src


def _build_matrix(
    by_src: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Return ``{source_id: {group: {populated, missing, total}}}``."""
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for src, records in by_src.items():
        out[src] = {}
        for group in _FIELD_GROUPS:
            populated = 0
            missing = 0
            for rec in records:
                # For groups with required fields, "populated" means all
                # required fields are populated. For optional-in-alpha
                # groups, "populated" means any field is populated.
                if group in _OPTIONAL_IN_ALPHA:
                    if _check_any_populated(rec, group):
                        populated += 1
                else:
                    if _check_required(rec, group):
                        populated += 1
                    else:
                        missing += 1
            out[src][group] = {
                "total": len(records),
                "populated": populated,
                "missing": missing if group not in _OPTIONAL_IN_ALPHA else 0,
            }
    return out


def _format_human(matrix: Dict[str, Dict[str, Dict[str, int]]]) -> str:
    """Render the matrix as a fixed-width text table."""
    if not matrix:
        return "(no records found in catalog)\n"
    sources = sorted(matrix.keys())
    groups = list(_FIELD_GROUPS.keys())
    src_col = max(8, max(len(s) for s in sources))
    grp_col = 12

    lines: List[str] = []
    header = "source".ljust(src_col) + "  " + "  ".join(
        g.ljust(grp_col) for g in groups
    ) + "  total"
    lines.append(header)
    lines.append("-" * len(header))
    for src in sources:
        row = src.ljust(src_col) + "  "
        cells: List[str] = []
        total = 0
        for grp in groups:
            entry = matrix[src][grp]
            total = entry["total"]
            label = "OPT" if grp in _OPTIONAL_IN_ALPHA else "REQ"
            cell = f"{entry['populated']}/{entry['total']} {label}"
            cells.append(cell.ljust(grp_col))
        row += "  ".join(cells) + f"  {total}"
        lines.append(row)
    lines.append("")
    lines.append(
        "REQ = always-required group (missing => failure). OPT = "
        "optional in alpha (uncertainty)."
    )
    return "\n".join(lines) + "\n"


def _failures(matrix: Dict[str, Dict[str, Dict[str, int]]]) -> List[str]:
    """Return human-readable failure entries (group missing on a record)."""
    out: List[str] = []
    for src, by_grp in sorted(matrix.items()):
        for grp, entry in by_grp.items():
            if grp in _OPTIONAL_IN_ALPHA:
                continue
            if entry["missing"]:
                out.append(
                    f"{src}: {grp} group unsatisfied on "
                    f"{entry['missing']}/{entry['total']} record(s)"
                )
    return out


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print a coverage matrix of CTO Phase 2 §2.1 field groups "
            "across the v0.1 alpha catalog."
        )
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=_DEFAULT_CATALOG_DIR,
        help=(
            "Catalog seed root. Defaults to "
            "greenlang/factors/data/catalog_seed_v0_1/."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON payload instead of the human-readable matrix.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help=(
            "Exit non-zero when any required field group is unsatisfied "
            "on any record."
        ),
    )
    args = parser.parse_args(argv)

    by_src = _walk_catalog(args.catalog_dir)
    matrix = _build_matrix(by_src)
    failures = _failures(matrix)

    if args.json:
        payload = {
            "catalog_dir": str(args.catalog_dir),
            "matrix": matrix,
            "failures": failures,
            "summary": {
                "sources": sorted(matrix.keys()),
                "total_records": sum(
                    sum(g["total"] for g in by_grp.values()) // len(by_grp)
                    if by_grp else 0
                    for by_grp in matrix.values()
                ),
                "fail_count": len(failures),
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        sys.stdout.write(_format_human(matrix))
        if failures:
            sys.stdout.write("\nField-group failures:\n")
            for f in failures:
                sys.stdout.write(f"  - {f}\n")

    if args.fail_on_missing and failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
