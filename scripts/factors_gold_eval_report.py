#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/factors_gold_eval_report.py

Human-oriented per-slice reporter for the GreenLang Factors gold-eval set.
Runs the same evaluation pipeline as tests/factors/matching/test_gold_eval.py
but prints a plain-text / markdown / JSON matrix instead of asserting.

Usage:

    # Plain table to stdout
    python scripts/factors_gold_eval_report.py

    # Markdown (for the PR comment used by factors-gold-eval.yml)
    python scripts/factors_gold_eval_report.py --format markdown --out results/matrix.md

    # Raw JSON (for machine-readable gates)
    python scripts/factors_gold_eval_report.py --format json --out results/matrix.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the same primitives the pytest uses so the script and the gate can
# never diverge - if the test harness evolves the matcher, this script
# reports against the same behaviour automatically.
from tests.factors.matching.test_gold_eval import (  # noqa: E402
    ALL_SLICES,
    CaseOutcome,
    GoldEntry,
    MAX_SKIPPED_FRACTION,
    TOP1_FLOOR,
    TOP5_FLOOR,
    _evaluate_case,
    _load_gold_entries,
    _summarise,
)
from greenlang.data.emission_factor_database import EmissionFactorDatabase  # noqa: E402
from greenlang.factors.catalog_repository import (  # noqa: E402
    MemoryFactorCatalogRepository,
)


# -----------------------------------------------------------------------
# Build matrix
# -----------------------------------------------------------------------


@dataclass
class SliceRow:
    slice: str
    total: int
    evaluated: int
    skipped: int
    top1: float
    top5: float
    top1_ok: bool
    top5_ok: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "slice": self.slice,
            "total": self.total,
            "evaluated": self.evaluated,
            "skipped": self.skipped,
            "top1": self.top1,
            "top5": self.top5,
            "top1_ok": self.top1_ok,
            "top5_ok": self.top5_ok,
        }


def _slice_row(slice_name: str, outcomes: Sequence[CaseOutcome]) -> SliceRow:
    subset = [o for o in outcomes if o.slice_name == slice_name]
    summ = _summarise(subset)
    if summ["evaluated"] == 0:
        return SliceRow(
            slice=slice_name,
            total=summ["total"],
            evaluated=0,
            skipped=summ["skipped_missing_factor"],
            top1=0.0,
            top5=0.0,
            top1_ok=True,  # vacuously OK: can't regress zero
            top5_ok=True,
        )
    return SliceRow(
        slice=slice_name,
        total=summ["total"],
        evaluated=summ["evaluated"],
        skipped=summ["skipped_missing_factor"],
        top1=summ["top1"],
        top5=summ["top5"],
        top1_ok=summ["top1"] >= TOP1_FLOOR,
        top5_ok=summ["top5"] >= TOP5_FLOOR,
    )


def build_matrix() -> Dict[str, Any]:
    entries: List[GoldEntry] = _load_gold_entries()
    db = EmissionFactorDatabase()
    repo = MemoryFactorCatalogRepository(
        edition_id="gold-eval-edition",
        label="Gold Eval Built-in Edition",
        db=db,
    )
    edition_id = repo.get_default_edition_id()
    known_factor_ids: Set[str] = {f.factor_id for f in repo._factors}  # noqa: SLF001
    outcomes = [_evaluate_case(e, repo, edition_id, known_factor_ids) for e in entries]

    per_slice = [_slice_row(s, outcomes) for s in ALL_SLICES]
    overall_summ = _summarise(outcomes)
    overall_row = SliceRow(
        slice="OVERALL",
        total=overall_summ["total"],
        evaluated=overall_summ["evaluated"],
        skipped=overall_summ["skipped_missing_factor"],
        top1=overall_summ["top1"],
        top5=overall_summ["top5"],
        top1_ok=(overall_summ["evaluated"] == 0) or (overall_summ["top1"] >= TOP1_FLOOR),
        top5_ok=(overall_summ["evaluated"] == 0) or (overall_summ["top5"] >= TOP5_FLOOR),
    )
    skipped_fraction = (
        overall_summ["skipped_missing_factor"] / overall_summ["total"]
        if overall_summ["total"]
        else 0.0
    )
    coverage_ok = skipped_fraction <= MAX_SKIPPED_FRACTION

    return {
        "floors": {
            "top1": TOP1_FLOOR,
            "top5": TOP5_FLOOR,
            "max_skipped_fraction": MAX_SKIPPED_FRACTION,
        },
        "overall": overall_row.as_dict() | {
            "skipped_fraction": skipped_fraction,
            "coverage_ok": coverage_ok,
        },
        "slices": [r.as_dict() for r in per_slice],
        "gate_passed": (
            overall_row.top1_ok
            and overall_row.top5_ok
            and coverage_ok
            and all(r.top1_ok for r in per_slice)
            and all(r.top5_ok for r in per_slice)
        ),
    }


# -----------------------------------------------------------------------
# Renderers
# -----------------------------------------------------------------------


def _fmt_pct(v: float, n_evaluated: int) -> str:
    if n_evaluated == 0:
        return "   n/a"
    return f"{v*100:5.1f}%"


def render_text(matrix: Dict[str, Any]) -> str:
    lines: List[str] = []
    header = (
        f"{'slice':<15}  {'total':>5}  {'eval':>4}  {'skip':>4}  "
        f"{'top-1':>6}  {'t1?':>4}  {'top-5':>6}  {'t5?':>4}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in matrix["slices"]:
        t1 = _fmt_pct(row["top1"], row["evaluated"])
        t5 = _fmt_pct(row["top5"], row["evaluated"])
        t1_ok = "OK" if row["top1_ok"] else "FAIL"
        t5_ok = "OK" if row["top5_ok"] else "FAIL"
        lines.append(
            f"{row['slice']:<15}  {row['total']:>5}  {row['evaluated']:>4}  "
            f"{row['skipped']:>4}  {t1:>6}  {t1_ok:>4}  {t5:>6}  {t5_ok:>4}"
        )
    lines.append("-" * len(header))
    o = matrix["overall"]
    t1 = _fmt_pct(o["top1"], o["evaluated"])
    t5 = _fmt_pct(o["top5"], o["evaluated"])
    t1_ok = "OK" if o["top1_ok"] else "FAIL"
    t5_ok = "OK" if o["top5_ok"] else "FAIL"
    lines.append(
        f"{'OVERALL':<15}  {o['total']:>5}  {o['evaluated']:>4}  "
        f"{o['skipped']:>4}  {t1:>6}  {t1_ok:>4}  {t5:>6}  {t5_ok:>4}"
    )
    lines.append("")
    lines.append(
        f"Skipped fraction: {o['skipped_fraction']*100:.1f}% "
        f"(budget {matrix['floors']['max_skipped_fraction']*100:.0f}%) "
        f"-> {'OK' if o['coverage_ok'] else 'FAIL'}"
    )
    lines.append(
        f"Floors: top-1 >= {matrix['floors']['top1']*100:.0f}%, "
        f"top-5 >= {matrix['floors']['top5']*100:.0f}%"
    )
    lines.append(
        f"Gate: {'PASS' if matrix['gate_passed'] else 'FAIL'}"
    )
    return "\n".join(lines)


def render_markdown(matrix: Dict[str, Any]) -> str:
    def pct(v: float, n_eval: int) -> str:
        return "n/a" if n_eval == 0 else f"{v*100:.1f}%"

    def tick(ok: bool) -> str:
        return "pass" if ok else "FAIL"

    lines: List[str] = []
    lines.append("| Slice | Total | Evaluated | Skipped | Top-1 | Top-1 Gate | Top-5 | Top-5 Gate |")
    lines.append("|---|---:|---:|---:|---:|:---:|---:|:---:|")
    for row in matrix["slices"]:
        lines.append(
            f"| `{row['slice']}` | {row['total']} | {row['evaluated']} | "
            f"{row['skipped']} | {pct(row['top1'], row['evaluated'])} | "
            f"{tick(row['top1_ok'])} | {pct(row['top5'], row['evaluated'])} | "
            f"{tick(row['top5_ok'])} |"
        )
    o = matrix["overall"]
    lines.append(
        f"| **OVERALL** | **{o['total']}** | **{o['evaluated']}** | "
        f"**{o['skipped']}** | **{pct(o['top1'], o['evaluated'])}** | "
        f"**{tick(o['top1_ok'])}** | **{pct(o['top5'], o['evaluated'])}** | "
        f"**{tick(o['top5_ok'])}** |"
    )
    lines.append("")
    lines.append(
        f"- Skipped fraction: **{o['skipped_fraction']*100:.1f}%** "
        f"(budget {matrix['floors']['max_skipped_fraction']*100:.0f}%) - "
        f"{tick(o['coverage_ok'])}"
    )
    lines.append(
        f"- Overall gate: **{'PASS' if matrix['gate_passed'] else 'FAIL'}**"
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Per-slice gold-eval reporter for GreenLang Factors "
            "(top-1, top-5, skipped, case count)."
        )
    )
    p.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="Output format (default: text).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file; stdout if omitted.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    matrix = build_matrix()

    if args.format == "json":
        rendered = json.dumps(matrix, indent=2, sort_keys=True)
    elif args.format == "markdown":
        rendered = render_markdown(matrix)
    else:
        rendered = render_text(matrix)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)

    # Always return 0 - this is a REPORTER, not a gate.  The pytest is
    # the gate.  CI invokes both so a failing gate still gets a pretty
    # matrix for the PR comment.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
