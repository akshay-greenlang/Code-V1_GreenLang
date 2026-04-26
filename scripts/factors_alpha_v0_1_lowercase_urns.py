#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot fix: lowercase namespace + id segments in v0.1 alpha catalog
seed factor URNs.

Phase 0 / pre-Phase-0 audit cleanup. The CTO doc Section 6.1.1 (URN
spec) requires namespace segments to be lowercase, and the canonical
URN parser at ``greenlang.factors.ontology.urn.parse`` enforces this.
The first batch of seed JSONs was generated before the
``coerce_factor_id_to_urn`` fix landed and contains uppercase namespace
segments (``IN``, ``CBAM``, ``DESNZ``, ``EPA``, ``IPCC``, ``eGRID``)
plus uppercase id country/grid codes (``CN``, ``UK``, ``US``,
``GLOBAL``, ``AKGD``, ...).

Scope: rewrite the ``urn`` field of every record in each
``greenlang/factors/data/catalog_seed_v0_1/<source>/v1.json`` so that
it passes :func:`greenlang.factors.ontology.urn.parse`. Other URN
fields (``source_urn``, ``geography_urn``, ``methodology_urn``,
``unit_urn``, ``factor_pack_urn``) are spot-checked but not rewritten
here. Pack-version canonicalization is handled separately by ADR-002.

The script is idempotent: re-running on already-lowercased seeds is
a no-op. Existing ``factor_id_alias`` ``EF:...`` strings are preserved
verbatim because the alias schema explicitly permits uppercase.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEEDS_DIR = _REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed_v0_1"


def _lowercase_factor_urn_body(urn: str) -> str:
    """Lowercase namespace + id segments of a factor URN.

    Format: ``urn:gl:factor:<source>:<namespace>:<id-segments...>:v<n>``
    The ``<source>`` slug is already lowercase by construction; the
    trailing ``v<n>`` is preserved verbatim. Everything between the
    source slug and the version is lowercased.

    No-op for non-factor URNs.
    """
    prefix = "urn:gl:factor:"
    if not urn.startswith(prefix):
        return urn
    body = urn[len(prefix):]
    # Last colon separates the version segment.
    last_colon = body.rfind(":")
    if last_colon < 0:
        return urn
    version_seg = body[last_colon + 1:]
    head = body[:last_colon]
    parts = head.split(":")
    if len(parts) < 2:
        return urn
    source = parts[0]  # already lowercase
    middle = [p.lower() for p in parts[1:]]
    rebuilt = ":".join([source] + middle)
    return f"{prefix}{rebuilt}:{version_seg}"


def _process_seed(path: Path) -> Tuple[int, int]:
    """Rewrite ``path`` in place. Returns (records_total, urns_changed)."""
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    records = payload.get("records") or []
    if not isinstance(records, list):
        return 0, 0
    changed = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        urn = rec.get("urn")
        if not isinstance(urn, str):
            continue
        new_urn = _lowercase_factor_urn_body(urn)
        if new_urn != urn:
            rec["urn"] = new_urn
            changed += 1
    if changed:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    return len(records), changed


def main(argv: List[str]) -> int:
    if not _SEEDS_DIR.is_dir():
        print(f"seeds dir missing: {_SEEDS_DIR}", file=sys.stderr)
        return 2
    summary: List[Tuple[str, int, int]] = []
    for src_dir in sorted(_SEEDS_DIR.iterdir()):
        if not src_dir.is_dir():
            continue
        seed_path = src_dir / "v1.json"
        if not seed_path.is_file():
            continue
        total, changed = _process_seed(seed_path)
        summary.append((src_dir.name, total, changed))

    print(f"{'source':30s} {'total':>6s} {'changed':>8s}")
    print("-" * 48)
    for name, total, changed in summary:
        print(f"{name:30s} {total:>6d} {changed:>8d}")
    grand_changed = sum(c for _, _, c in summary)
    grand_total = sum(t for _, t, _ in summary)
    print("-" * 48)
    print(f"{'TOTAL':30s} {grand_total:>6d} {grand_changed:>8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
