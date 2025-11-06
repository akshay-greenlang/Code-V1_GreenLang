#!/usr/bin/env python3
"""
Fix AgentSpec v2 Schema Compliance Issues - Phase 3 (Final)

Final cleanup to achieve 100% schema compliance:
1. Normalize unit notation (°C→degC, °F→degF, m²→m2, m³→m3)
2. Fix provenance constraints (add factors or set pin_ef=false)
3. Remove extra top-level fields (priority)
4. Remove extra fields from outputs (enum)
5. Remove extra provenance fields (ml_model)
6. Fix grid_factor_ai factors field type

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

# Unit normalization map
UNIT_NORMALIZATION = {
    "°C": "degC",
    "°F": "degF",
    "m²": "m2",
    "m³": "m3",
    "%": "percent",  # Will still use % as it's now in whitelist, but as fallback
}


def normalize_unit(unit: str) -> str:
    """Normalize unit notation to ASCII-safe format."""
    if unit in UNIT_NORMALIZATION:
        return UNIT_NORMALIZATION[unit]
    return unit


def fix_field_units(field_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize units in a field."""
    if not isinstance(field_data, dict):
        return field_data

    if "unit" in field_data:
        field_data["unit"] = normalize_unit(field_data["unit"])

    # Remove enum from outputs (only allowed on inputs)
    if "enum" in field_data:
        del field_data["enum"]

    return field_data


def fix_compute_section(compute: Dict[str, Any]) -> Dict[str, Any]:
    """Fix compute section."""
    # Fix units in inputs
    if "inputs" in compute:
        for field_name in compute["inputs"]:
            if isinstance(compute["inputs"][field_name], dict):
                compute["inputs"][field_name] = fix_field_units(compute["inputs"][field_name])

    # Fix units in outputs
    if "outputs" in compute:
        for field_name in compute["outputs"]:
            if isinstance(compute["outputs"][field_name], dict):
                compute["outputs"][field_name] = fix_field_units(compute["outputs"][field_name])

    # Ensure factors is a dict
    if "factors" not in compute or not isinstance(compute["factors"], dict):
        compute["factors"] = {}

    return compute


def fix_provenance_section(provenance: Dict[str, Any], has_factors: bool) -> Dict[str, Any]:
    """Fix provenance section."""
    # If pin_ef=true but no factors, set pin_ef=false
    if provenance.get("pin_ef") and not has_factors:
        provenance["pin_ef"] = False

    # Remove ml_model field (not in schema)
    if "ml_model" in provenance:
        del provenance["ml_model"]

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

    # Check if there are any factors
    has_factors = bool(data.get("compute", {}).get("factors", {}))

    if "provenance" in data:
        data["provenance"] = fix_provenance_section(data["provenance"], has_factors)

    # Remove extra top-level fields
    if "priority" in data:
        del data["priority"]

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
    print("AgentSpec v2 Schema Compliance Fixer - Phase 3 (Final)")
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
