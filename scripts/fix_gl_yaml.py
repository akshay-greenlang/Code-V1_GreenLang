#!/usr/bin/env python3
"""
Fix all gl.yaml files to comply with GreenLang v1.0 specification
"""

import os
import yaml
import glob
from pathlib import Path
from typing import Dict, List, Any

def fix_gl_yaml(file_path: str) -> Dict[str, Any]:
    """Fix a single gl.yaml file for spec compliance"""

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

    # 1. Replace deprecated "hub:" with "registry:"
    if "hub" in data:
        # Move hub contents to registry
        data["registry"] = data.pop("hub")
        changes.append("replaced 'hub' with 'registry'")

    # 2. Fix registry structure if it exists
    if "registry" in data and isinstance(data["registry"], dict):
        # Ensure proper fields in registry
        if "namespace" not in data["registry"] and "pack" in data:
            # Try to infer namespace from pack info
            if isinstance(data["pack"], dict) and "id" in data["pack"]:
                pack_id = data["pack"]["id"]
                # Use a default namespace pattern
                data["registry"]["namespace"] = "greenlang-official"
                changes.append("added default namespace to registry")

        # Fix tags structure if present
        if "tags" in data["registry"] and not isinstance(data["registry"]["tags"], list):
            # Convert to list if not already
            if isinstance(data["registry"]["tags"], str):
                data["registry"]["tags"] = [data["registry"]["tags"]]
                changes.append("converted registry.tags to list")

    # 3. Ensure consistent metadata structure
    if "metadata" in data and isinstance(data["metadata"], dict):
        # Check for common inconsistencies
        if "version" in data["metadata"] and not isinstance(data["metadata"]["version"], str):
            data["metadata"]["version"] = str(data["metadata"]["version"])
            changes.append("converted metadata.version to string")

        # Add schema version if missing
        if "schema_version" not in data["metadata"]:
            data["metadata"]["schema_version"] = "1.0"
            changes.append("added metadata.schema_version")

    # 4. Fix pack structure if present
    if "pack" in data and isinstance(data["pack"], dict):
        # Ensure UUID format
        if "uuid" in data["pack"] and not isinstance(data["pack"]["uuid"], str):
            data["pack"]["uuid"] = str(data["pack"]["uuid"])
            changes.append("converted pack.uuid to string")

        # Fix version fields
        if "version" in data["pack"] and not isinstance(data["pack"]["version"], str):
            data["pack"]["version"] = str(data["pack"]["version"])
            changes.append("converted pack.version to string")

    # 5. Fix dependencies structure
    if "dependencies" in data and isinstance(data["dependencies"], dict):
        # Ensure proper structure for greenlang dependencies
        if "greenlang" in data["dependencies"]:
            if not isinstance(data["dependencies"]["greenlang"], dict):
                # Convert to dict with version key
                gl_version = data["dependencies"]["greenlang"]
                data["dependencies"]["greenlang"] = {"version": str(gl_version)}
                changes.append("fixed greenlang dependency structure")

    # 6. Fix certification structure if present
    if "certification" in data and isinstance(data["certification"], dict):
        # Ensure lists for multiple certifications
        for cert_type in ["compliance", "quality", "security"]:
            if cert_type in data["certification"]:
                if not isinstance(data["certification"][cert_type], list):
                    # Convert single item to list
                    data["certification"][cert_type] = [data["certification"][cert_type]]
                    changes.append(f"converted certification.{cert_type} to list")

    # 7. Ensure proper field ordering (optional but good practice)
    ordered_data = {}
    field_order = [
        "pack", "metadata", "registry", "dependencies", "runtime",
        "deployment", "certification", "monitoring", "support", "documentation"
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
    """Fix all gl.yaml files in the repository"""

    base_path = r"C:\Users\aksha\Code-V1_GreenLang"

    # Find all gl.yaml files
    gl_files = []
    for root, dirs, files in os.walk(base_path):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        if "gl.yaml" in files:
            gl_files.append(os.path.join(root, "gl.yaml"))

    print(f"Found {len(gl_files)} gl.yaml files")

    results = {
        "fixed": [],
        "unchanged": [],
        "errors": [],
        "skipped": []
    }

    for gl_file in gl_files:
        result = fix_gl_yaml(gl_file)

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
    print("GL.YAML FIX SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(gl_files)}")
    print(f"Files fixed: {len(results['fixed'])}")
    print(f"Files unchanged: {len(results['unchanged'])}")
    print(f"Files with errors: {len(results['errors'])}")
    print(f"Files skipped: {len(results['skipped'])}")

    return results

if __name__ == "__main__":
    main()