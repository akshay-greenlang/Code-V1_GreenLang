#!/usr/bin/env python3
"""
Analyze Pack.yaml Schema Usage Patterns

Extracts all dtype and unit usage patterns from pack.yaml files to inform
AgentSpec v2 schema extension design.

Author: Claude Code
Date: 2025-11-06
"""

import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set

PACK_DIRS = [
    "packs/fuel_ai",
    "packs/carbon_ai",
    "packs/grid_factor_ai",
    "packs/boiler_replacement_ai",
    "packs/industrial_process_heat_ai",
    "packs/decarbonization_roadmap_ai",
    "packs/industrial_heat_pump_ai",
    "packs/recommendation_ai",
    "packs/report_ai",
    "packs/anomaly_iforest_ai",
    "packs/forecast_sarima_ai",
    "packs/waste_heat_recovery_ai",
]


def analyze_field(field_data: Dict, field_name: str, section: str, results: Dict):
    """Analyze a single field."""
    dtype = field_data.get("dtype")
    unit = field_data.get("unit")

    # Track dtype usage
    if dtype:
        results["dtypes"][dtype].add(f"{section}.{field_name}")

    # Track unit usage
    if unit:
        results["units"][unit].add(f"{section}.{field_name}")

    # Track missing units for complex types
    if dtype in ["list", "object"] and not unit:
        results["missing_units"].add(f"{section}.{field_name} (dtype={dtype})")

    # Track extra fields
    extra_fields = set(field_data.keys()) - {"dtype", "unit", "description", "required", "default", "ge", "le", "gt", "lt", "enum"}
    if extra_fields:
        for extra in extra_fields:
            results["extra_fields"][extra].add(f"{section}.{field_name}")


def analyze_pack(pack_path: Path, results: Dict):
    """Analyze a single pack.yaml file."""
    yaml_file = pack_path / "pack.yaml"

    if not yaml_file.exists():
        return

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Analyze inputs
    if "compute" in data and "inputs" in data["compute"]:
        for field_name, field_data in data["compute"]["inputs"].items():
            if isinstance(field_data, dict):
                analyze_field(field_data, field_name, "inputs", results)

    # Analyze outputs
    if "compute" in data and "outputs" in data["compute"]:
        for field_name, field_data in data["compute"]["outputs"].items():
            if isinstance(field_data, dict):
                analyze_field(field_data, field_name, "outputs", results)


def main():
    """Analyze all packs."""
    results = {
        "dtypes": defaultdict(set),
        "units": defaultdict(set),
        "missing_units": set(),
        "extra_fields": defaultdict(set)
    }

    print("=" * 70)
    print("Pack.yaml Schema Usage Pattern Analysis")
    print("=" * 70)
    print()

    for pack_dir in PACK_DIRS:
        pack_path = Path(pack_dir)
        analyze_pack(pack_path, results)

    # Print results
    print("1. DTYPE USAGE")
    print("-" * 70)
    for dtype in sorted(results["dtypes"].keys()):
        count = len(results["dtypes"][dtype])
        print(f"{dtype:12s}: {count:3d} fields")
    print()

    print("2. COMPLEX TYPES BREAKDOWN")
    print("-" * 70)
    complex_types = ["list", "object"]
    for dtype in complex_types:
        if dtype in results["dtypes"]:
            print(f"\n{dtype.upper()} dtype ({len(results['dtypes'][dtype])} fields):")
            for field in sorted(results["dtypes"][dtype])[:10]:
                print(f"  - {field}")
            if len(results["dtypes"][dtype]) > 10:
                print(f"  ... and {len(results['dtypes'][dtype]) - 10} more")
    print()

    print("3. UNIT USAGE (Top 20)")
    print("-" * 70)
    unit_counts = [(unit, len(fields)) for unit, fields in results["units"].items()]
    unit_counts.sort(key=lambda x: x[1], reverse=True)
    for unit, count in unit_counts[:20]:
        print(f"{unit:30s}: {count:3d} fields")
    print()

    print("4. NON-STANDARD UNITS (need whitelist expansion)")
    print("-" * 70)
    # Standard units already in whitelist
    standard_units = {
        "1", "kgCO2e", "tCO2e", "kgCO2e/kWh", "kgCO2e/MWh", "gCO2e/kWh",
        "kWh", "MWh", "GWh", "kWh/year", "MWh/year",
        "GJ", "TJ", "m3", "kg", "t"
    }

    non_standard = set(results["units"].keys()) - standard_units
    for unit in sorted(non_standard):
        count = len(results["units"][unit])
        print(f"{unit:30s}: {count:3d} fields")
    print()

    print("5. MISSING UNITS FOR COMPLEX TYPES")
    print("-" * 70)
    print(f"Total: {len(results['missing_units'])} fields")
    for field in sorted(results["missing_units"])[:15]:
        print(f"  - {field}")
    if len(results["missing_units"]) > 15:
        print(f"  ... and {len(results['missing_units']) - 15} more")
    print()

    print("6. EXTRA FIELDS (not in base schema)")
    print("-" * 70)
    for field_name in sorted(results["extra_fields"].keys()):
        count = len(results["extra_fields"][field_name])
        print(f"{field_name:20s}: {count:3d} occurrences")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total dtypes found: {len(results['dtypes'])}")
    print(f"Total units found: {len(results['units'])}")
    print(f"Non-standard units: {len(non_standard)}")
    print(f"Fields missing units: {len(results['missing_units'])}")
    print(f"Extra field types: {len(results['extra_fields'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
