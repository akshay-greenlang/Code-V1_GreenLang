#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze coverage.json and generate module-level statistics."""

import json
from collections import defaultdict
from pathlib import Path

# Load coverage data
with open('coverage.json') as f:
    data = json.load(f)

# Aggregate by module
modules = defaultdict(lambda: {'stmts': 0, 'covered': 0, 'missing': 0})

for filepath, info in data['files'].items():
    if filepath.startswith('greenlang'):
        # Get module name (first directory after greenlang/)
        parts = Path(filepath).parts
        if len(parts) >= 2:
            module = parts[1] if parts[0] == 'greenlang' else parts[0]
            modules[module]['stmts'] += info['summary']['num_statements']
            modules[module]['covered'] += info['summary']['covered_lines']
            modules[module]['missing'] += info['summary']['missing_lines']

# Print results
print("\n" + "="*80)
print("COVERAGE BY MODULE")
print("="*80)
print(f"{'Module':<30} {'Covered':>10} {'Total':>10} {'Coverage':>10}")
print("-"*80)

for module, stats in sorted(modules.items(), key=lambda x: x[1]['covered']/x[1]['stmts'] if x[1]['stmts'] > 0 else 0, reverse=True):
    coverage_pct = (stats['covered'] / stats['stmts'] * 100) if stats['stmts'] > 0 else 0
    print(f"{module:<30} {stats['covered']:>10} {stats['stmts']:>10} {coverage_pct:>9.2f}%")

print("-"*80)
total_covered = sum(m['covered'] for m in modules.values())
total_stmts = sum(m['stmts'] for m in modules.values())
print(f"{'TOTAL':<30} {total_covered:>10} {total_stmts:>10} {total_covered/total_stmts*100:>9.2f}%")
print("="*80)

# Find files with 100% coverage
print("\n" + "="*80)
print("FILES WITH 100% COVERAGE (22 files)")
print("="*80)
full_coverage = []
for filepath, info in data['files'].items():
    if filepath.startswith('greenlang'):
        coverage = info['summary']['percent_covered']
        if coverage == 100.0:
            full_coverage.append(filepath)

for f in sorted(full_coverage):
    print(f"  - {f}")

# Find files with 0% coverage
print("\n" + "="*80)
print("FILES WITH 0% COVERAGE")
print("="*80)
zero_coverage = []
for filepath, info in data['files'].items():
    if filepath.startswith('greenlang'):
        coverage = info['summary']['percent_covered']
        if coverage == 0.0:
            zero_coverage.append(filepath)

print(f"Total: {len(zero_coverage)} files")
for f in sorted(zero_coverage)[:20]:  # Show first 20
    print(f"  - {f}")
if len(zero_coverage) > 20:
    print(f"  ... and {len(zero_coverage) - 20} more")

print("\n" + "="*80)
print("COVERAGE GAPS SUMMARY")
print("="*80)
print(f"Files with 0% coverage: {len(zero_coverage)}")
print(f"Files with 100% coverage: {len(full_coverage)}")
print(f"Files with partial coverage: {len(data['files']) - len(zero_coverage) - len(full_coverage)}")
print(f"Total files analyzed: {len([f for f in data['files'] if f.startswith('greenlang')])}")
print("="*80)
