#!/usr/bin/env python3
"""
Fix AgentSpec v2 Schema Compliance Issues in Pack.yaml Files

This script automatically fixes schema violations in all pack.yaml files
to match the exact AgentSpec v2 Pydantic schema requirements.

Fixes applied:
1. Remove warn_at_usd from ai.budget
2. Fix provenance section (ef_pinning → pin_ef, gwp_set, add record field)
3. Fix realtime section (replay_mode structure → default_mode)
4. Simplify tools to minimal valid schema
5. Remove metadata section (not in AgentSpec v2)

Author: Claude Code
Date: 2025-11-06
"""

import yaml
from pathlib import Path
from typing import Dict, Any

# Pack directories to process
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


def fix_budget_section(ai_section: Dict[str, Any]) -> Dict[str, Any]:
    """Fix AI budget section to match schema."""
    if "budget" in ai_section:
        budget = ai_section["budget"]
        # Remove non-schema fields
        if "warn_at_usd" in budget:
            del budget["warn_at_usd"]
        if "max_usd_per_run" in budget:
            # Rename to correct field name
            budget["max_cost_usd"] = budget.pop("max_usd_per_run")
    return ai_section


def fix_tools_section(ai_section: Dict[str, Any]) -> Dict[str, Any]:
    """Fix AI tools section to have minimal valid schema."""
    if "tools" in ai_section:
        fixed_tools = []
        for tool in ai_section["tools"]:
            # Create minimal valid tool schema
            fixed_tool = {
                "name": tool["name"],
                "description": tool.get("description", f"Tool: {tool['name']}"),
                "schema_in": {"type": "object", "properties": {}},
                "schema_out": {"type": "object", "properties": {}},
                "impl": f"python://greenlang.agents.tools:{tool['name']}",
                "safe": True
            }
            fixed_tools.append(fixed_tool)
        ai_section["tools"] = fixed_tools
    return ai_section


def fix_provenance_section(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix provenance section to match schema."""
    if "provenance" in data:
        prov = data["provenance"]

        # Fix field names
        if "ef_pinning" in prov:
            prov["pin_ef"] = True
            del prov["ef_pinning"]

        # Fix gwp_set value
        if "gwp_set" in prov:
            if prov["gwp_set"] == "AR6":
                prov["gwp_set"] = "AR6GWP100"
        else:
            prov["gwp_set"] = "AR6GWP100"

        # Add required record field
        if "record" not in prov:
            prov["record"] = ["inputs", "outputs", "factors", "code_sha", "seed"]

        # Remove non-schema fields
        for field in ["audit_fields", "citation_required", "determinism_required", "standards"]:
            if field in prov:
                del prov[field]

    return data


def fix_realtime_section(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix realtime section to match schema."""
    if "realtime" in data:
        realtime = data["realtime"]

        # Fix structure
        if "replay_mode" in realtime:
            # Replace nested structure with flat structure
            data["realtime"] = {
                "default_mode": "replay",
                "snapshot_path": None,
                "connectors": []
            }
        elif "default_mode" not in realtime:
            # Ensure default_mode exists
            realtime["default_mode"] = "replay"
            if "connectors" not in realtime:
                realtime["connectors"] = []

    return data


def remove_metadata_section(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove metadata section (not in AgentSpec v2)."""
    if "metadata" in data:
        del data["metadata"]
    return data


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
    if "ai" in data:
        data["ai"] = fix_budget_section(data["ai"])
        data["ai"] = fix_tools_section(data["ai"])

    data = fix_provenance_section(data)
    data = fix_realtime_section(data)
    data = remove_metadata_section(data)

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
    print("AgentSpec v2 Schema Compliance Fixer")
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
