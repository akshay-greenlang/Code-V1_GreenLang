#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase F10 — Populate ``factor_family`` on existing YAML factor rows.

Scans ``data/emission_factors*.yaml`` files, infers the canonical
:class:`~greenlang.data.canonical_v2.FactorFamily` for each factor from
its existing tags / activity_type / scope / boundary, and writes the
field back into the YAML.  Idempotent: running twice is a no-op.

Usage::

    python scripts/populate_factor_family.py --dry-run
    python scripts/populate_factor_family.py                # actually writes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml
except ImportError:              # pragma: no cover
    yaml = None


DEFAULT_ROOT = Path("data")

#: Matching rules — highest-precedence first.  Each entry is
#: ``(predicate, factor_family_value)``.  Predicates receive the raw
#: YAML factor dict.
_RULES: List[Tuple[Any, str]] = [
    # Grid electricity
    (lambda f: "grid" in _text(f) or f.get("scope") in ("2", 2), "grid_intensity"),
    # Refrigerants
    (lambda f: "refrigerant" in _text(f), "refrigerant_gwp"),
    # Transport
    (lambda f: any(w in _text(f) for w in ("freight", "truck", "vessel", "flight", "rail")), "transport_lane"),
    # Waste
    (lambda f: any(w in _text(f) for w in ("waste", "landfill", "incineration", "composting")), "waste_treatment"),
    # Materials / embodied
    (lambda f: any(w in _text(f) for w in ("steel", "cement", "aluminium", "aluminum", "plastic", "concrete", "glass", "paper")), "material_embodied"),
    # Heating values — when the YAML entry is explicitly a HV row.
    (lambda f: "hcv_mj_per_kg" in f or "lhv_mj_per_kg" in f, "heating_value"),
    # Land / biomass removals
    (lambda f: any(w in _text(f) for w in ("biochar", "afforestation", "reforestation", "land use", "soil carbon")), "land_use_removals"),
    # Energy conversion
    (lambda f: any(w in _text(f) for w in ("conversion", "density", "btu", "mmbtu"))
     and "electricity" not in _text(f), "energy_conversion"),
    # Fuel combustion (default for Scope 1 factors)
    (lambda f: f.get("scope") in ("1", 1), "emissions"),
]


def _text(f: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("fuel_type", "activity_type", "category", "sub_category"):
        if f.get(key):
            parts.append(str(f[key]).lower())
    for tag_key in ("tags", "activity_tags", "sector_tags"):
        tags = f.get(tag_key) or []
        parts.extend(str(t).lower() for t in tags)
    return " ".join(parts)


def infer_factor_family(factor: Dict[str, Any]) -> str:
    for predicate, family in _RULES:
        try:
            if predicate(factor):
                return family
        except Exception:
            continue
    return "emissions"                  # safe default


def walk_factor_entries(doc: Any) -> Iterable[Dict[str, Any]]:
    """Yield every dict that looks like a factor entry.

    YAML files in ``data/`` mix several schemas; we walk recursively and
    treat any dict with ``factor_id`` OR ``fuel_type`` OR ``activity_type``
    as a factor row.
    """
    if isinstance(doc, dict):
        looks_like_factor = any(
            key in doc
            for key in ("factor_id", "fuel_type", "activity_type")
        )
        if looks_like_factor:
            yield doc
        for v in doc.values():
            yield from walk_factor_entries(v)
    elif isinstance(doc, list):
        for item in doc:
            yield from walk_factor_entries(item)


def process_file(path: Path, *, dry_run: bool) -> Tuple[int, int]:
    """Returns ``(updated, total)`` counts for one file."""
    if yaml is None:                 # pragma: no cover
        raise RuntimeError("PyYAML required: pip install pyyaml")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    updated = 0
    total = 0
    for factor in walk_factor_entries(doc):
        total += 1
        if factor.get("factor_family"):
            continue
        factor["factor_family"] = infer_factor_family(factor)
        updated += 1
    if updated and not dry_run:
        path.write_text(
            yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return updated, total


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.root.exists():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2

    files = sorted(args.root.glob("emission_factors*.yaml"))
    if not files:
        print(f"no emission_factors*.yaml under {args.root}", file=sys.stderr)
        return 1

    grand_updated = 0
    grand_total = 0
    for f in files:
        upd, total = process_file(f, dry_run=args.dry_run)
        grand_updated += upd
        grand_total += total
        marker = " (DRY RUN)" if args.dry_run else ""
        print(f"{f.name}: {upd}/{total} filled in{marker}")

    print(f"\nTotal: {grand_updated}/{grand_total} factors populated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
