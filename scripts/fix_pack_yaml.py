#!/usr/bin/env python3
"""
Fix all pack.yaml files to comply with GreenLang v1.0 specification
"""

import os
import yaml
import glob
from pathlib import Path
from typing import Dict, List, Any

def fix_pack_yaml(file_path: str) -> Dict[str, Any]:
    """Fix a single pack.yaml file for spec compliance"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse YAML
    try:
        data = yaml.safe_load(content)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {"file": file_path, "status": "error", "error": str(e)}

    if not data:
        return {"file": file_path, "status": "skipped", "reason": "empty file"}

    changes = []

    # 1. Fix schema version field
    if "schema_version" in data:
        if data["schema_version"] == "2.0.0":
            del data["schema_version"]
            data["pack_schema_version"] = "1.0"
            changes.append("replaced schema_version with pack_schema_version")
    elif "pack_schema_version" not in data:
        data["pack_schema_version"] = "1.0"
        changes.append("added pack_schema_version")

    # 2. Add missing kind field
    if "kind" not in data:
        data["kind"] = "pack"
        changes.append("added kind: pack")

    # 3. Remove invalid compute sections
    if "compute" in data:
        del data["compute"]
        changes.append("removed invalid compute section")

    # 4. Fix duplicate license fields
    license_count = 0
    license_value = None

    # Check top level
    if "license" in data:
        license_count += 1
        license_value = data["license"]

    # Check in metadata
    if "metadata" in data and isinstance(data["metadata"], dict):
        if "license" in data["metadata"]:
            license_count += 1
            if not license_value:
                license_value = data["metadata"]["license"]
            del data["metadata"]["license"]
            changes.append("removed duplicate license from metadata")

    # Ensure only one license field at top level
    if license_count > 1:
        data["license"] = license_value
        changes.append("consolidated duplicate license fields")
    elif license_count == 0:
        data["license"] = "MIT"  # Default license
        changes.append("added default MIT license")

    # 5. Add missing author section if needed
    if "author" not in data and "name" in data:
        # Infer author from pack name or use defaults
        data["author"] = {
            "name": "GreenLang Team",
            "email": "support@greenlang.io",
            "organization": "GreenLang"
        }
        changes.append("added default author section")

    # 6. Ensure proper field ordering (optional but good practice)
    ordered_data = {}
    field_order = [
        "name", "version", "kind", "pack_schema_version", "display_name",
        "tagline", "description", "author", "license", "category", "tags",
        "agents", "pipeline", "contents", "dependencies", "data", "rules",
        "schemas", "examples", "usage", "guarantees", "compliance", "metadata"
    ]

    for field in field_order:
        if field in data:
            ordered_data[field] = data[field]

    # Add any remaining fields not in our order
    for key, value in data.items():
        if key not in ordered_data:
            ordered_data[key] = value

    # Write fixed YAML
    if changes:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(ordered_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return {"file": file_path, "status": "fixed", "changes": changes}
    else:
        return {"file": file_path, "status": "unchanged"}

def main():
    """Fix all pack.yaml files in the repository"""

    base_path = r"C:\Users\aksha\Code-V1_GreenLang"

    # Find all pack.yaml files
    pack_files = []
    for root, dirs, files in os.walk(base_path):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        if "pack.yaml" in files:
            pack_files.append(os.path.join(root, "pack.yaml"))

    print(f"Found {len(pack_files)} pack.yaml files")

    results = {
        "fixed": [],
        "unchanged": [],
        "errors": [],
        "skipped": []
    }

    for pack_file in pack_files:
        result = fix_pack_yaml(pack_file)

        if result["status"] == "fixed":
            results["fixed"].append(result)
            print(f"[FIXED] {result['file']}")
            for change in result["changes"]:
                print(f"  - {change}")
        elif result["status"] == "unchanged":
            results["unchanged"].append(result)
            print(f"[OK] {result['file']}")
        elif result["status"] == "error":
            results["errors"].append(result)
            print(f"[ERROR] {result['file']} - {result['error']}")
        elif result["status"] == "skipped":
            results["skipped"].append(result)
            print(f"[SKIP] {result['file']} - {result['reason']}")

    # Summary
    print("\n" + "="*60)
    print("PACK.YAML FIX SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(pack_files)}")
    print(f"Files fixed: {len(results['fixed'])}")
    print(f"Files unchanged: {len(results['unchanged'])}")
    print(f"Files with errors: {len(results['errors'])}")
    print(f"Files skipped: {len(results['skipped'])}")

    return results

if __name__ == "__main__":
    main()