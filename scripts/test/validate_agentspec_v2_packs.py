#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Pack.yaml Files Against AgentSpec v2 Pydantic Schema

This script validates all pack.yaml files against the official AgentSpec v2
Pydantic schema to ensure full compliance.

Usage:
    python validate_agentspec_v2_packs.py

Author: Claude Code
Date: 2025-11-06
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from pydantic import ValidationError

# Import the AgentSpec v2 Pydantic schema
from greenlang.specs.agentspec_v2 import AgentSpecV2

# Pack directories to validate
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


def validate_pack_yaml(pack_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate a single pack.yaml file using AgentSpec v2 Pydantic schema.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    yaml_file = pack_path / "pack.yaml"

    if not yaml_file.exists():
        return False, [f"pack.yaml not found at {pack_path}"]

    try:
        # Load YAML
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Validate against Pydantic schema
        AgentSpecV2(**data)

        return True, []

    except ValidationError as e:
        # Parse Pydantic validation errors
        errors = []
        for error in e.errors():
            loc = " -> ".join(str(l) for l in error["loc"])
            msg = error["msg"]
            errors.append(f"  {loc}: {msg}")
        return False, errors

    except yaml.YAMLError as e:
        return False, [f"YAML parsing error: {str(e)}"]

    except Exception as e:
        return False, [f"Unexpected error: {str(e)}"]


def main():
    """Validate all pack.yaml files."""
    print("=" * 70)
    print("AgentSpec v2 Pack.yaml Validation")
    print("=" * 70)
    print()

    results = []

    for pack_dir in PACK_DIRS:
        pack_path = Path(pack_dir)
        pack_name = pack_path.name

        is_valid, errors = validate_pack_yaml(pack_path)
        results.append((pack_name, is_valid, errors))

        if is_valid:
            print(f"[PASS] {pack_name}")
        else:
            print(f"[FAIL] {pack_name}")
            for error in errors:
                print(f"  {error}")

    print()
    print("=" * 70)

    # Summary
    passed = sum(1 for _, is_valid, _ in results if is_valid)
    failed = len(results) - passed

    print(f"Summary: {passed}/{len(results)} packs passed validation")

    if failed > 0:
        print()
        print("Failed packs:")
        for name, is_valid, errors in results:
            if not is_valid:
                print(f"  - {name} ({len(errors)} errors)")

    print("=" * 70)

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
