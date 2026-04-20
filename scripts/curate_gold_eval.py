#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Curate the Factor-matching gold-label evaluation set.

Audits the full gold set (``tests/factors/fixtures/gold_eval_full.json``),
enriches each case with ``domain``, ``difficulty``, and ``geography`` when
missing, and writes a balanced curated subset to
``tests/factors/fixtures/gold_eval_curated.json``.

Targets (per FY27 plan):

- 350–400 curated cases
- Balanced across US / EU / UK (inclusion: ≥ 30 cases per geo)
- Balanced across fuel types (≥ 10 per fuel type)
- Domain field populated on 100 % of cases
- Difficulty field populated on 100 % of cases, at least 20 % "hard"

Usage::

    python scripts/curate_gold_eval.py               # write curated set
    python scripts/curate_gold_eval.py --audit-only  # print audit, no write
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SOURCE = Path("tests/factors/fixtures/gold_eval_full.json")
OUTPUT = Path("tests/factors/fixtures/gold_eval_curated.json")

# Heuristic vocabulary for enrichment.  Not perfect — but the gold set is
# deliberately curated afterwards, so precision matters less than recall.
_GEO_KEYWORDS = {
    "US": [" us ", "us ", " u.s.", "united states", "america", "california", "ca ", "texas", "ny "],
    "EU": [" eu ", "eu ", "europe", "european", "eu27", "germany", "france", "spain", "italy", "poland", "netherlands"],
    "UK": [" uk ", "uk ", "united kingdom", "britain", "british", "england", "scotland", "defra", "desnz"],
    "IN": [" in ", "india", "indian"],
    "CN": ["china", "chinese"],
}

_DOMAIN_KEYWORDS = {
    "transport": ["fleet", "vehicle", "car ", "truck", "aviation", "shipping", "freight", "rail", "transport"],
    "buildings": ["heating", "cooling", "hvac", "residential", "commercial building", "office"],
    "industry": ["boiler", "furnace", "cement", "steel", "aluminum", "aluminium", "petrochemical"],
    "agriculture": ["livestock", "cattle", "manure", "soil", "fertili", "crop"],
    "regulatory": ["cbam", "csrd", "sb 253", "sb253", "tcfd", "ghg protocol", "sbti"],
    "energy": ["electricity", "grid", "fuel", "combustion", "diesel", "gasoline", "natural_gas", "coal", "propane", "lpg"],
}

_HARD_MARKERS = ["misspell", "abbrev", "cross_geography", "ambiguous", "edge", "multilingual", "scoped", "units", "temporal"]


def audit(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return coverage statistics for a list of cases."""
    return {
        "total": len(cases),
        "by_fuel": dict(
            collections.Counter(c.get("expected_fuel_type") for c in cases).most_common()
        ),
        "by_geography": dict(
            collections.Counter(c.get("geography") or "UNK" for c in cases).most_common()
        ),
        "by_domain": dict(
            collections.Counter(c.get("domain") or "UNK" for c in cases).most_common()
        ),
        "by_difficulty": dict(
            collections.Counter(c.get("difficulty") or "UNK" for c in cases).most_common()
        ),
        "with_domain": sum(1 for c in cases if c.get("domain")),
        "with_difficulty": sum(1 for c in cases if c.get("difficulty")),
        "with_geography": sum(1 for c in cases if c.get("geography")),
    }


def _match_keyword(text: str, groups: Dict[str, List[str]]) -> Optional[str]:
    t = " " + text.lower() + " "
    for key, markers in groups.items():
        for m in markers:
            if m in t:
                return key
    return None


def enrich(case: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``case`` with ``geography``, ``domain``, ``difficulty``
    fields ensured."""
    out = dict(case)

    activity = (out.get("activity") or "").lower()

    # Geography
    if not out.get("geography"):
        inferred = _match_keyword(activity, _GEO_KEYWORDS)
        out["geography"] = inferred or "US"  # default to US (largest bucket)

    # Domain
    if not out.get("domain"):
        inferred = _match_keyword(activity, _DOMAIN_KEYWORDS)
        out["domain"] = inferred or "energy"  # default to energy

    # Difficulty
    if not out.get("difficulty"):
        dom = out.get("domain", "")
        if dom in _HARD_MARKERS or any(mk in activity for mk in _HARD_MARKERS):
            out["difficulty"] = "hard"
        elif len(activity.split()) <= 3:
            out["difficulty"] = "easy"
        else:
            out["difficulty"] = "medium"

    return out


def stratified_sample(
    cases: List[Dict[str, Any]],
    *,
    target: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Balanced sample keeping proportional representation by (fuel, geo).

    Strategy: bucket by ``(expected_fuel_type, geography)``, then take
    ``ceil(target / num_buckets)`` per bucket up to the bucket size.
    """
    buckets: Dict[tuple, List[Dict[str, Any]]] = collections.defaultdict(list)
    for c in cases:
        key = (c.get("expected_fuel_type"), c.get("geography"))
        buckets[key].append(c)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    bucket_cap = max(1, target // max(1, len(buckets)))
    curated: List[Dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        curated.extend(bucket[:bucket_cap])

    # If we fell short of target, top up from the remainder in deterministic order.
    remaining = [c for c in cases if c["id"] not in {x["id"] for x in curated}]
    remaining.sort(key=lambda c: c["id"])
    while len(curated) < target and remaining:
        curated.append(remaining.pop(0))

    # Cap at target.
    curated = curated[:target]

    # Stable output order by id for diffability.
    curated.sort(key=lambda c: c["id"])
    return curated


def write_curated(
    source_path: Path,
    output_path: Path,
    *,
    target: int,
    seed: int,
) -> Dict[str, Any]:
    cases = json.loads(source_path.read_text(encoding="utf-8"))
    enriched = [enrich(c) for c in cases]
    curated = stratified_sample(enriched, target=target, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "generated_from": str(source_path.as_posix()),
        "curation_seed": seed,
        "target_size": target,
        "actual_size": len(curated),
        "cases": curated,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    return audit(curated)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--target", type=int, default=380)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit the source file only; don't write the curated output.",
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"Source file not found: {args.source}", file=sys.stderr)
        return 2

    source_audit = audit(json.loads(args.source.read_text(encoding="utf-8")))
    print("== SOURCE AUDIT ==")
    print(json.dumps(source_audit, indent=2))

    if args.audit_only:
        return 0

    curated_audit = write_curated(
        args.source, args.output, target=args.target, seed=args.seed
    )
    print("\n== CURATED AUDIT ==")
    print(json.dumps(curated_audit, indent=2))
    print(f"\nWrote {curated_audit['total']} cases -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
