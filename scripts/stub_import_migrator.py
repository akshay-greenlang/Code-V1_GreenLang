#!/usr/bin/env python3
"""
Stub Import Migrator - Canonicalize old stub imports to greenlang.agents.{cat}.{name}

Usage:
    python scripts/stub_import_migrator.py --scan          # Report old imports
    python scripts/stub_import_migrator.py --dry-run       # Show diffs without writing
    python scripts/stub_import_migrator.py --apply         # Write changes
    python scripts/stub_import_migrator.py --verify        # Confirm 0 old imports remain

Optionally pass --target greenlang/ or --target tests/ to limit scope.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────
# Mapping: old dotted prefix -> new dotted prefix (61 entries)
# ──────────────────────────────────────────────────────────────────────
MAPPING: Dict[str, str] = {
    # Foundation (10)
    "greenlang.access_guard":         "greenlang.agents.foundation.access_guard",
    "greenlang.agent_registry":       "greenlang.agents.foundation.agent_registry",
    "greenlang.assumptions":          "greenlang.agents.foundation.assumptions",
    "greenlang.citations":            "greenlang.agents.foundation.citations",
    "greenlang.normalizer":           "greenlang.agents.foundation.normalizer",
    "greenlang.observability_agent":  "greenlang.agents.foundation.observability_agent",
    "greenlang.orchestrator":         "greenlang.agents.foundation.orchestrator",
    "greenlang.qa_test_harness":      "greenlang.agents.foundation.qa_test_harness",
    "greenlang.reproducibility":      "greenlang.agents.foundation.reproducibility",
    "greenlang.schema":               "greenlang.agents.foundation.schema",
    # Data (20)
    "greenlang.climate_hazard":              "greenlang.agents.data.climate_hazard",
    "greenlang.cross_source_reconciliation": "greenlang.agents.data.cross_source_reconciliation",
    "greenlang.data_freshness_monitor":      "greenlang.agents.data.data_freshness_monitor",
    "greenlang.data_gateway":                "greenlang.agents.data.data_gateway",
    "greenlang.data_lineage_tracker":        "greenlang.agents.data.data_lineage_tracker",
    "greenlang.data_quality_profiler":       "greenlang.agents.data.data_quality_profiler",
    "greenlang.deforestation_satellite":     "greenlang.agents.data.deforestation_satellite",
    "greenlang.duplicate_detector":          "greenlang.agents.data.duplicate_detector",
    "greenlang.erp_connector":               "greenlang.agents.data.erp_connector",
    "greenlang.eudr_traceability":           "greenlang.agents.data.eudr_traceability",
    "greenlang.excel_normalizer":            "greenlang.agents.data.excel_normalizer",
    "greenlang.gis_connector":               "greenlang.agents.data.gis_connector",
    "greenlang.missing_value_imputer":       "greenlang.agents.data.missing_value_imputer",
    "greenlang.outlier_detector":            "greenlang.agents.data.outlier_detector",
    "greenlang.pdf_extractor":               "greenlang.agents.data.pdf_extractor",
    "greenlang.schema_migration":            "greenlang.agents.data.schema_migration",
    "greenlang.spend_categorizer":           "greenlang.agents.data.spend_categorizer",
    "greenlang.supplier_questionnaire":      "greenlang.agents.data.supplier_questionnaire",
    "greenlang.time_series_gap_filler":      "greenlang.agents.data.time_series_gap_filler",
    "greenlang.validation_rule_engine":      "greenlang.agents.data.validation_rule_engine",
    # MRV Scope 1 (8 + 1 flaring)
    "greenlang.stationary_combustion":       "greenlang.agents.mrv.stationary_combustion",
    "greenlang.mobile_combustion":           "greenlang.agents.mrv.mobile_combustion",
    "greenlang.process_emissions":           "greenlang.agents.mrv.process_emissions",
    "greenlang.fugitive_emissions":          "greenlang.agents.mrv.fugitive_emissions",
    "greenlang.refrigerants_fgas":           "greenlang.agents.mrv.refrigerants_fgas",
    "greenlang.land_use_emissions":          "greenlang.agents.mrv.land_use_emissions",
    "greenlang.waste_generated":             "greenlang.agents.mrv.waste_generated",
    "greenlang.waste_treatment_emissions":   "greenlang.agents.mrv.waste_treatment_emissions",
    "greenlang.agricultural_emissions":      "greenlang.agents.mrv.agricultural_emissions",
    "greenlang.flaring":                     "greenlang.agents.mrv.flaring",
    # MRV Scope 2 (5)
    "greenlang.scope2_location":             "greenlang.agents.mrv.scope2_location",
    "greenlang.scope2_market":               "greenlang.agents.mrv.scope2_market",
    "greenlang.steam_heat_purchase":         "greenlang.agents.mrv.steam_heat_purchase",
    "greenlang.cooling_purchase":            "greenlang.agents.mrv.cooling_purchase",
    "greenlang.dual_reporting_reconciliation": "greenlang.agents.mrv.dual_reporting_reconciliation",
    # MRV Scope 3 (15)
    "greenlang.purchased_goods_services":    "greenlang.agents.mrv.purchased_goods_services",
    "greenlang.capital_goods":               "greenlang.agents.mrv.capital_goods",
    "greenlang.fuel_energy_activities":      "greenlang.agents.mrv.fuel_energy_activities",
    "greenlang.upstream_transportation":     "greenlang.agents.mrv.upstream_transportation",
    "greenlang.business_travel":             "greenlang.agents.mrv.business_travel",
    "greenlang.employee_commuting":          "greenlang.agents.mrv.employee_commuting",
    "greenlang.upstream_leased_assets":      "greenlang.agents.mrv.upstream_leased_assets",
    "greenlang.downstream_transportation":   "greenlang.agents.mrv.downstream_transportation",
    "greenlang.downstream_leased_assets":    "greenlang.agents.mrv.downstream_leased_assets",
    "greenlang.end_of_life_treatment":       "greenlang.agents.mrv.end_of_life_treatment",
    "greenlang.use_of_sold_products":        "greenlang.agents.mrv.use_of_sold_products",
    "greenlang.processing_sold_products":    "greenlang.agents.mrv.processing_sold_products",
    "greenlang.franchises":                  "greenlang.agents.mrv.franchises",
    "greenlang.investments":                 "greenlang.agents.mrv.investments",
    # MRV Cross-Cutting (2)
    "greenlang.scope3_category_mapper":      "greenlang.agents.mrv.scope3_category_mapper",
    "greenlang.audit_trail_lineage":         "greenlang.agents.mrv.audit_trail_lineage",
}

# Sort keys longest-first so "greenlang.schema_migration" matches before "greenlang.schema"
SORTED_KEYS = sorted(MAPPING.keys(), key=len, reverse=True)

# Stub directory names (the folder names under greenlang/ to delete)
STUB_DIRS = sorted(set(k.split(".", 1)[1] for k in MAPPING.keys()))

# ──────────────────────────────────────────────────────────────────────
# Directories that ARE stubs (skip when scanning for old imports -
# we don't want to "fix" the stubs themselves, we want to delete them)
# ──────────────────────────────────────────────────────────────────────
def is_stub_file(filepath: Path, root: Path) -> bool:
    """Return True if filepath lives inside a stub directory."""
    try:
        rel = filepath.relative_to(root)
    except ValueError:
        return False
    parts = rel.parts
    # greenlang/<stub_name>/__init__.py
    if len(parts) >= 2 and parts[0] == "greenlang" and parts[1] in STUB_DIRS:
        # But NOT greenlang/agents/... (those are real)
        if parts[1] != "agents":
            return True
    return False


def collect_py_files(target: Path, root: Path, skip_stubs: bool = True) -> List[Path]:
    """Recursively collect .py files under target, optionally skipping stub dirs."""
    files = []
    for dirpath, dirnames, filenames in os.walk(target):
        dp = Path(dirpath)
        # Skip hidden dirs, __pycache__, .git, node_modules
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d != "__pycache__" and d != "node_modules"
        ]
        for fn in filenames:
            if fn.endswith(".py"):
                fp = dp / fn
                if skip_stubs and is_stub_file(fp, root):
                    continue
                files.append(fp)
    return sorted(files)


def migrate_line(line: str) -> Tuple[str, bool]:
    """Attempt to rewrite a single line. Returns (new_line, changed).

    Processes ALL old-path occurrences on the line (not just the first match).
    """
    changed = False
    for old_prefix in SORTED_KEYS:
        new_prefix = MAPPING[old_prefix]
        if old_prefix not in line:
            continue
        # Skip if the new prefix is already present for this specific mapping
        # (but don't skip the whole line - other mappings may still apply)
        if new_prefix in line and old_prefix not in line.replace(new_prefix, ""):
            continue
        new_line = line.replace(old_prefix, new_prefix)
        if new_line != line:
            line = new_line
            changed = True
    return line, changed


def migrate_content(content: str) -> Tuple[str, int]:
    """Migrate all lines. Returns (new_content, num_changes)."""
    lines = content.split("\n")
    changes = 0
    new_lines = []
    for line in lines:
        new_line, changed = migrate_line(line)
        if changed:
            changes += 1
        new_lines.append(new_line)
    return "\n".join(new_lines), changes


def scan_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """Scan a file for old imports. Returns list of (lineno, old_line, new_line)."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    hits = []
    for i, line in enumerate(content.split("\n"), 1):
        new_line, changed = migrate_line(line)
        if changed:
            hits.append((i, line.rstrip(), new_line.rstrip()))
    return hits


def run_scan(targets: List[Path], root: Path) -> int:
    """Scan mode: report old imports without changing anything."""
    total_files = 0
    total_hits = 0
    for target in targets:
        files = collect_py_files(target, root)
        for fp in files:
            hits = scan_file(fp)
            if hits:
                total_files += 1
                total_hits += len(hits)
                rel = fp.relative_to(root)
                print(f"\n{rel} ({len(hits)} occurrence{'s' if len(hits) != 1 else ''}):")
                for lineno, old, new in hits:
                    print(f"  L{lineno}: {old.strip()}")
                    print(f"     -> {new.strip()}")
    print(f"\n{'='*60}")
    print(f"SCAN RESULT: {total_hits} old import(s) in {total_files} file(s)")
    return total_hits


def run_dry_run(targets: List[Path], root: Path) -> int:
    """Dry-run mode: show unified-diff-style output without writing."""
    total_files = 0
    total_hits = 0
    for target in targets:
        files = collect_py_files(target, root)
        for fp in files:
            hits = scan_file(fp)
            if hits:
                total_files += 1
                total_hits += len(hits)
                rel = fp.relative_to(root)
                print(f"\n--- {rel}")
                print(f"+++ {rel}")
                for lineno, old, new in hits:
                    print(f"@@ L{lineno} @@")
                    print(f"- {old.strip()}")
                    print(f"+ {new.strip()}")
    print(f"\n{'='*60}")
    print(f"DRY-RUN: Would change {total_hits} line(s) in {total_files} file(s)")
    return total_hits


def run_apply(targets: List[Path], root: Path) -> int:
    """Apply mode: rewrite files in-place."""
    total_files = 0
    total_changes = 0
    for target in targets:
        files = collect_py_files(target, root)
        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"SKIP {fp}: {e}", file=sys.stderr)
                continue
            new_content, changes = migrate_content(content)
            if changes > 0:
                fp.write_text(new_content, encoding="utf-8")
                total_files += 1
                total_changes += changes
                rel = fp.relative_to(root)
                print(f"  UPDATED {rel} ({changes} change{'s' if changes != 1 else ''})")
    print(f"\n{'='*60}")
    print(f"APPLIED: {total_changes} change(s) in {total_files} file(s)")
    return total_changes


def run_verify(targets: List[Path], root: Path) -> int:
    """Verify mode: confirm 0 old imports remain. Returns count of remaining."""
    remaining = 0
    for target in targets:
        files = collect_py_files(target, root)
        for fp in files:
            hits = scan_file(fp)
            if hits:
                remaining += len(hits)
                rel = fp.relative_to(root)
                print(f"  REMAINING: {rel} ({len(hits)} old import(s))")
                for lineno, old, new in hits:
                    print(f"    L{lineno}: {old.strip()}")
    if remaining == 0:
        print("VERIFY: PASS - 0 old imports remain")
    else:
        print(f"VERIFY: FAIL - {remaining} old import(s) still present")
    return remaining


def main():
    parser = argparse.ArgumentParser(description="Migrate stub imports to canonical paths")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true", help="Report old imports")
    group.add_argument("--dry-run", action="store_true", help="Show diffs without writing")
    group.add_argument("--apply", action="store_true", help="Rewrite files in-place")
    group.add_argument("--verify", action="store_true", help="Confirm 0 old imports remain")
    parser.add_argument(
        "--target", action="append", default=None,
        help="Target directory (can be repeated). Default: greenlang/ and tests/"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.target:
        targets = [root / t for t in args.target]
    else:
        targets = [root / "greenlang", root / "tests"]

    # Validate targets exist
    for t in targets:
        if not t.exists():
            print(f"ERROR: Target {t} does not exist", file=sys.stderr)
            sys.exit(1)

    if args.scan:
        run_scan(targets, root)
    elif args.dry_run:
        run_dry_run(targets, root)
    elif args.apply:
        run_apply(targets, root)
    elif args.verify:
        count = run_verify(targets, root)
        sys.exit(1 if count > 0 else 0)


if __name__ == "__main__":
    main()
