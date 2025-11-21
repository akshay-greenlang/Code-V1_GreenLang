#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix AgentSpec v2 Schema Compliance Issues - Phase 2

After extending the schema to support complex types, this script fixes remaining
violations in all pack.yaml files.

Fixes applied:
1. Add unit="1" to all complex type (list/object) fields
2. Remove extra fields not allowed by schema (items, properties, required, examples)
3. Fix constraint violations (default with required=true)
4. Fix factors field type (dict instead of list)

Author: Claude Code
Date: 2025-11-06
"""

import yaml
from pathlib import Path
from typing import Dict, Any

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

# Extra fields that should be removed
EXTRA_FIELDS_TO_REMOVE = ["items", "properties", "required", "examples"]


def fix_field(field_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix a single input/output field."""
    if not isinstance(field_data, dict):
        return field_data

    # Fix 1: Add unit="1" for complex types missing unit
    dtype = field_data.get("dtype")
    if dtype in ["list", "object"] and "unit" not in field_data:
        field_data["unit"] = "1"

    # Fix 2: Remove extra fields
    for extra_field in EXTRA_FIELDS_TO_REMOVE:
        if extra_field in field_data:
            del field_data[extra_field]

    # Fix 3: Fix constraint violations (default with required=true)
    if "default" in field_data and field_data.get("required", True):
        # If there's a default, required must be false
        field_data["required"] = False

    return field_data


def fix_compute_section(compute: Dict[str, Any]) -> Dict[str, Any]:
    """Fix compute section."""
    # Fix inputs
    if "inputs" in compute:
        for field_name in compute["inputs"]:
            if isinstance(compute["inputs"][field_name], dict):
                compute["inputs"][field_name] = fix_field(compute["inputs"][field_name])

    # Fix outputs
    if "outputs" in compute:
        for field_name in compute["outputs"]:
            if isinstance(compute["outputs"][field_name], dict):
                compute["outputs"][field_name] = fix_field(compute["outputs"][field_name])

    # Fix 4: factors should be dict, not list
    if "factors" in compute:
        if isinstance(compute["factors"], list) and len(compute["factors"]) == 0:
            compute["factors"] = {}

    return compute


def fix_provenance_section(provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Fix provenance section."""
    # Remove extra fields not in schema
    extra_fields = ["data_sources"]
    for field in extra_fields:
        if field in provenance:
            del provenance[field]

    return provenance


def fix_pack_yaml(pack_path: Path) -> bool:
    """Fix a single pack.yaml file. Returns True if changes were made."""
    yaml_file = pack_path / "pack.yaml"

    if not yaml_file.exists():
        print(f"[SKIP] {yaml_file} not found")
        return False

    # Load YAML
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Track if changes were made
    original_data = yaml.dump(data)

    # Apply fixes
    if "compute" in data:
        data["compute"] = fix_compute_section(data["compute"])

    if "provenance" in data:
        data["provenance"] = fix_provenance_section(data["provenance"])

    # Check if changes were made
    fixed_data = yaml.dump(data)
    if fixed_data == original_data:
        print(f"[OK] {pack_path.name}: No changes needed")
        return False

    # Write fixed YAML
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[FIXED] {pack_path.name}: Fixed and saved")
    return True


def main():
    """Fix all pack.yaml files."""
    print("=" * 70)
    print("AgentSpec v2 Schema Compliance Fixer - Phase 2")
    print("=" * 70)
    print()

    fixed_count = 0
    for pack_dir in PACK_DIRS:
        pack_path = Path(pack_dir)
        if fix_pack_yaml(pack_path):
            fixed_count += 1

    print()
    print("=" * 70)
    print(f"Summary: Fixed {fixed_count}/{len(PACK_DIRS)} pack.yaml files")
    print("=" * 70)


if __name__ == "__main__":
    main()
