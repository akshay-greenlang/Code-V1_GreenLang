#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase F10 — CI invariant check.

Fails if any of the CTO non-negotiables are violated in the catalog:

- Every factor row must have a non-empty ``factor_family``.
- Every ``factor_status = "deprecated"`` row must have a
  ``replacement_factor_id``.
- No factor row may have ``factor_status = "certified"`` without
  ``valid_from`` + source version.

Intended to run in CI (.github/workflows/factors-gold-eval.yml is a good
host) via ``python scripts/check_factor_invariants.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import yaml
except ImportError:              # pragma: no cover
    yaml = None


def _walk(doc: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(doc, dict):
        if any(k in doc for k in ("factor_id", "fuel_type")):
            yield doc
        for v in doc.values():
            yield from _walk(v)
    elif isinstance(doc, list):
        for item in doc:
            yield from _walk(item)


def check_file(path: Path) -> List[str]:
    if yaml is None:            # pragma: no cover
        return [f"{path}: PyYAML not installed"]
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    issues: List[str] = []
    for factor in _walk(doc):
        fid = factor.get("factor_id") or factor.get("fuel_type") or "<anonymous>"
        status = str(factor.get("factor_status", "certified")).lower()
        if not factor.get("factor_family"):
            issues.append(f"{path.name}: {fid} — missing factor_family")
        if status == "deprecated" and not factor.get("replacement_factor_id"):
            issues.append(f"{path.name}: {fid} — deprecated without replacement_factor_id")
        if status == "certified":
            if not factor.get("valid_from"):
                issues.append(f"{path.name}: {fid} — certified without valid_from")
            if not (factor.get("release_version") or factor.get("source_release") or factor.get("factor_version")):
                issues.append(f"{path.name}: {fid} — certified without source/factor version")
    return issues


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--max-issues", type=int, default=25)
    args = parser.parse_args(argv)

    if not args.root.exists():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2

    all_issues: List[str] = []
    for f in sorted(args.root.glob("emission_factors*.yaml")):
        all_issues.extend(check_file(f))

    if not all_issues:
        print("OK — all factor invariants satisfied.")
        return 0

    for issue in all_issues[: args.max_issues]:
        print(issue)
    if len(all_issues) > args.max_issues:
        print(f"... and {len(all_issues) - args.max_issues} more.")
    print(f"\n{len(all_issues)} invariant violation(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
