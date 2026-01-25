#!/usr/bin/env python3
"""
Update all agent specifications from old format to AgentSpec v2.0 format
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any

def convert_to_agentspec_v2(old_spec: Dict) -> Dict:
    """Convert old agent spec format to AgentSpec v2.0"""

    # Create new v2.0 structure
    v2_spec = {
        "apiVersion": "greenlang.io/v2",
        "kind": "AgentSpec",
        "metadata": {},
        "spec": {
            "inputs": [],
            "outputs": [],
            "capabilities": [],
            "requirements": {}
        }
    }

    # Convert metadata
    if "agent_metadata" in old_spec:
        old_meta = old_spec["agent_metadata"]
        v2_spec["metadata"] = {
            "name": old_meta.get("agent_name", ""),
            "version": old_meta.get("version", "1.0.0"),
            "labels": {
                "type": old_meta.get("agent_type", ""),
                "category": old_meta.get("category", ""),
                "id": old_meta.get("agent_id", "")
            },
            "annotations": {
                "description": old_meta.get("description", ""),
                "created_by": old_meta.get("created_by", ""),
                "created_date": old_meta.get("created_date", "")
            }
        }

        # Add tags if present
        if "tags" in old_meta:
            v2_spec["metadata"]["labels"]["tags"] = ",".join(old_meta["tags"])

    # Convert mission to spec
    if "mission" in old_spec:
        mission = old_spec["mission"]
        v2_spec["spec"]["description"] = mission.get("primary_objective", "")

        # Convert capabilities from in_scope
        if "in_scope" in mission:
            v2_spec["spec"]["capabilities"] = [
                {
                    "name": f"capability_{i}",
                    "description": cap
                }
                for i, cap in enumerate(mission["in_scope"], 1)
            ]

        # Add success criteria as requirements
        if "success_criteria" in mission:
            v2_spec["spec"]["requirements"] = {
                "performance": [],
                "quality": []
            }
            for criterion in mission.get("success_criteria", []):
                if "processing" in criterion.lower() or "time" in criterion.lower():
                    v2_spec["spec"]["requirements"]["performance"].append(criterion)
                else:
                    v2_spec["spec"]["requirements"]["quality"].append(criterion)

    # Convert interfaces to inputs/outputs
    if "interfaces" in old_spec:
        interfaces = old_spec["interfaces"]

        # Convert inputs
        if "inputs" in interfaces:
            for input_key, input_data in interfaces["inputs"].items():
                if isinstance(input_data, dict):
                    v2_input = {
                        "name": input_data.get("name", input_key),
                        "type": "data",
                        "format": input_data.get("format") or input_data.get("formats", ["json"])[0] if isinstance(input_data.get("formats"), list) else "json",
                        "schema": input_data.get("schema_reference", ""),
                        "required": input_data.get("required", True),
                        "description": input_data.get("description", "")
                    }

                    # Add validation if fields are specified
                    if "required_fields" in input_data or "optional_fields" in input_data:
                        v2_input["validation"] = {
                            "required_fields": input_data.get("required_fields", []),
                            "optional_fields": input_data.get("optional_fields", [])
                        }

                    v2_spec["spec"]["inputs"].append(v2_input)

        # Convert outputs
        if "outputs" in interfaces:
            for output_key, output_data in interfaces["outputs"].items():
                if isinstance(output_data, dict):
                    v2_output = {
                        "name": output_data.get("name", output_key),
                        "type": "data",
                        "format": output_data.get("format", "json"),
                        "schema": output_data.get("schema_reference", ""),
                        "description": output_data.get("description", "")
                    }

                    if "structure" in output_data:
                        v2_output["structure"] = output_data["structure"]

                    v2_spec["spec"]["outputs"].append(v2_output)

    # Convert processing logic
    if "processing_logic" in old_spec:
        v2_spec["spec"]["processing"] = {
            "steps": []
        }

        if "steps" in old_spec["processing_logic"]:
            for step in old_spec["processing_logic"]["steps"]:
                v2_step = {
                    "id": step.get("step_id", ""),
                    "name": step.get("name", ""),
                    "description": step.get("description", ""),
                    "type": step.get("type", "process")
                }

                if "validation_rules" in step:
                    v2_step["validation"] = step["validation_rules"]

                v2_spec["spec"]["processing"]["steps"].append(v2_step)

    # Convert dependencies
    if "dependencies" in old_spec:
        v2_spec["spec"]["dependencies"] = old_spec["dependencies"]

    # Convert performance targets
    if "performance_targets" in old_spec:
        if "requirements" not in v2_spec["spec"]:
            v2_spec["spec"]["requirements"] = {}
        v2_spec["spec"]["requirements"]["performance_targets"] = old_spec["performance_targets"]

    return v2_spec

def update_agent_spec_file(file_path: str) -> Dict[str, Any]:
    """Update a single agent specification file"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse YAML
    try:
        old_spec = yaml.safe_load(content)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {"file": file_path, "status": "error", "error": str(e)}

    if not old_spec:
        return {"file": file_path, "status": "skipped", "reason": "empty file"}

    # Check if already v2.0 format
    if "apiVersion" in old_spec and "greenlang.io/v2" in old_spec["apiVersion"]:
        return {"file": file_path, "status": "already_v2"}

    # Check if it's an old format agent spec
    if "agent_metadata" not in old_spec and "mission" not in old_spec and "interfaces" not in old_spec:
        return {"file": file_path, "status": "not_agent_spec"}

    # Convert to v2.0
    v2_spec = convert_to_agentspec_v2(old_spec)

    # Write updated YAML
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(v2_spec, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return {"file": file_path, "status": "updated"}

def main():
    """Update all agent specification files in the repository"""

    base_path = r"C:\Users\aksha\Code-V1_GreenLang"

    # Find all potential agent spec files
    spec_files = []
    patterns = ["*agent*.yaml", "*agent*.yml", "*_spec.yaml", "*_spec.yml"]

    for root, dirs, files in os.walk(base_path):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            for pattern in patterns:
                if "agent" in file.lower() and file.endswith(('.yaml', '.yml')):
                    spec_files.append(os.path.join(root, file))
                    break

    # Filter to only actual spec files (in specs/ directories or with spec in name)
    filtered_specs = []
    for spec_file in spec_files:
        if "specs" in spec_file or "_spec" in spec_file or "agent_spec" in spec_file:
            filtered_specs.append(spec_file)

    print(f"Found {len(filtered_specs)} agent specification files")

    results = {
        "updated": [],
        "already_v2": [],
        "not_agent_spec": [],
        "errors": [],
        "skipped": []
    }

    for spec_file in filtered_specs:
        result = update_agent_spec_file(spec_file)

        if result["status"] == "updated":
            results["updated"].append(result)
            print(f"[UPDATED] {result['file']}")
        elif result["status"] == "already_v2":
            results["already_v2"].append(result)
            print(f"[V2] {result['file']}")
        elif result["status"] == "not_agent_spec":
            results["not_agent_spec"].append(result)
            print(f"[SKIP] {result['file']} - not an agent spec")
        elif result["status"] == "error":
            results["errors"].append(result)
            print(f"[ERROR] {result['file']} - {result['error']}")
        elif result["status"] == "skipped":
            results["skipped"].append(result)
            print(f"[SKIP] {result['file']} - {result['reason']}")

    # Summary
    print("\n" + "="*60)
    print("AGENT SPEC UPDATE SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(filtered_specs)}")
    print(f"Files updated to v2.0: {len(results['updated'])}")
    print(f"Files already v2.0: {len(results['already_v2'])}")
    print(f"Files not agent specs: {len(results['not_agent_spec'])}")
    print(f"Files with errors: {len(results['errors'])}")
    print(f"Files skipped: {len(results['skipped'])}")

    return results

if __name__ == "__main__":
    main()