#!/usr/bin/env python3
"""
Comprehensive validation of GreenLang specification implementation.
Tests pack.yaml and gl.yaml files against v1.0 schemas.
"""

import json
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import jsonschema
from jsonschema import Draft202012Validator, ValidationError

def load_yaml_file(filepath: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def load_json_schema(filepath: Path) -> Dict[str, Any]:
    """Load a JSON schema file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any], filename: str) -> Tuple[bool, List[str]]:
    """Validate data against a JSON schema."""
    errors = []
    try:
        validator = Draft202012Validator(schema)
        for error in validator.iter_errors(data):
            error_path = ".".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"{filename} - {error_path}: {error.message}")
        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"{filename}: Schema validation error - {str(e)}"]

def check_required_fields(data: Dict[str, Any], required: List[str], filename: str) -> List[str]:
    """Check if all required fields are present."""
    errors = []
    if data is None:
        errors.append(f"{filename}: File is empty or null")
        return errors
    for field in required:
        if field not in data:
            errors.append(f"{filename}: Missing required field '{field}'")
    return errors

def validate_pack_manifest(pack_file: Path, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a pack.yaml file."""
    errors = []
    warnings = []

    try:
        data = load_yaml_file(pack_file)
    except Exception as e:
        return False, [f"Failed to load {pack_file}: {str(e)}"]

    # Check required fields according to spec
    required = ["name", "version", "kind", "license", "contents"]
    field_errors = check_required_fields(data, required, str(pack_file))
    errors.extend(field_errors)

    # Validate against schema
    valid, schema_errors = validate_against_schema(data, schema, str(pack_file))
    errors.extend(schema_errors)

    # Additional validations
    if "version" in data:
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', str(data["version"])):
            errors.append(f"{pack_file}: Version must be semantic (MAJOR.MINOR.PATCH)")

    if "name" in data:
        import re
        if not re.match(r'^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$', str(data["name"])):
            errors.append(f"{pack_file}: Name must be DNS-safe (lowercase, alphanumeric with hyphens)")

    if "contents" in data and "pipelines" in data["contents"]:
        if not data["contents"]["pipelines"]:
            errors.append(f"{pack_file}: contents.pipelines must have at least one pipeline")

    return len(errors) == 0, errors

def validate_pipeline(pipeline_file: Path, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a gl.yaml file."""
    errors = []

    try:
        data = load_yaml_file(pipeline_file)
    except Exception as e:
        return False, [f"Failed to load {pipeline_file}: {str(e)}"]

    # Check required fields
    required = ["name", "steps"]
    field_errors = check_required_fields(data, required, str(pipeline_file))
    errors.extend(field_errors)

    # Validate against schema
    valid, schema_errors = validate_against_schema(data, schema, str(pipeline_file))
    errors.extend(schema_errors)

    # Additional step validations
    if "steps" in data:
        if not data["steps"]:
            errors.append(f"{pipeline_file}: Pipeline must have at least one step")
        else:
            for i, step in enumerate(data["steps"]):
                if not isinstance(step, dict):
                    errors.append(f"{pipeline_file}: Step {i} is not a dictionary")
                    continue
                if "id" not in step:
                    errors.append(f"{pipeline_file}: Step {i} missing required field 'id'")
                if "agent" not in step:
                    errors.append(f"{pipeline_file}: Step {i} missing required field 'agent'")

                # Check for mutually exclusive fields
                if "in" in step and "inputsRef" in step:
                    errors.append(f"{pipeline_file}: Step {i} has both 'in' and 'inputsRef' (mutually exclusive)")

    return len(errors) == 0, errors

def main():
    """Main validation function."""
    print("=" * 80)
    print("GreenLang Specification v1.0 Validation Report")
    print("=" * 80)

    # Load schemas
    pack_schema_path = Path("schemas/pack.schema.v1.json")
    pipeline_schema_path = Path("schemas/gl_pipeline.schema.v1.json")

    if not pack_schema_path.exists():
        print(f"ERROR: Pack schema not found at {pack_schema_path}")
        return 1

    if not pipeline_schema_path.exists():
        print(f"ERROR: Pipeline schema not found at {pipeline_schema_path}")
        return 1

    pack_schema = load_json_schema(pack_schema_path)
    pipeline_schema = load_json_schema(pipeline_schema_path)

    print(f"\n[PASS] Loaded pack schema v1.0 from {pack_schema_path}")
    print(f"[PASS] Loaded pipeline schema v1.0 from {pipeline_schema_path}")

    # Find and validate pack.yaml files
    print("\n" + "-" * 80)
    print("VALIDATING PACK MANIFESTS")
    print("-" * 80)

    pack_files = list(Path(".").glob("**/pack.yaml"))
    pack_files = [f for f in pack_files if "node_modules" not in str(f) and ".git" not in str(f)]

    pack_results = []
    for pack_file in pack_files:
        valid, errors = validate_pack_manifest(pack_file, pack_schema)
        pack_results.append((pack_file, valid, errors))

        if valid:
            print(f"[PASS] {pack_file}: VALID")
        else:
            print(f"[FAIL] {pack_file}: INVALID")
            for error in errors[:3]:  # Show first 3 errors
                print(f"  - {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more errors")

    # Find and validate gl.yaml files
    print("\n" + "-" * 80)
    print("VALIDATING PIPELINE FILES")
    print("-" * 80)

    pipeline_files = list(Path(".").glob("**/gl.yaml"))
    pipeline_files = [f for f in pipeline_files if "node_modules" not in str(f) and ".git" not in str(f)]

    pipeline_results = []
    for pipeline_file in pipeline_files:
        valid, errors = validate_pipeline(pipeline_file, pipeline_schema)
        pipeline_results.append((pipeline_file, valid, errors))

        if valid:
            print(f"[PASS] {pipeline_file}: VALID")
        else:
            print(f"[FAIL] {pipeline_file}: INVALID")
            for error in errors[:3]:
                print(f"  - {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more errors")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    valid_packs = sum(1 for _, valid, _ in pack_results if valid)
    valid_pipelines = sum(1 for _, valid, _ in pipeline_results if valid)

    print(f"\nPack Manifests: {valid_packs}/{len(pack_results)} valid")
    print(f"Pipeline Files: {valid_pipelines}/{len(pipeline_results)} valid")

    # Check documentation
    print("\n" + "-" * 80)
    print("DOCUMENTATION CHECK")
    print("-" * 80)

    docs = {
        "docs/specs/PACK_SCHEMA_V1.md": Path("docs/specs/PACK_SCHEMA_V1.md"),
        "docs/specs/GL_PIPELINE_SPEC_V1.md": Path("docs/specs/GL_PIPELINE_SPEC_V1.md"),
    }

    for name, path in docs.items():
        if path.exists():
            size = path.stat().st_size
            print(f"[PASS] {name}: EXISTS ({size:,} bytes)")
        else:
            print(f"[FAIL] {name}: MISSING")

    # Check implementation
    print("\n" + "-" * 80)
    print("IMPLEMENTATION CHECK")
    print("-" * 80)

    impl_files = {
        "Pack Manifest Model": Path("greenlang/packs/manifest.py"),
        "Pipeline Spec Model": Path("greenlang/sdk/pipeline_spec.py"),
        "Pipeline Validation": Path("tests/integration/test_pipeline_validation.py"),
    }

    for name, path in impl_files.items():
        if path.exists():
            print(f"[PASS] {name}: {path}")
        else:
            print(f"[FAIL] {name}: MISSING at {path}")

    # Final status
    all_valid = (valid_packs == len(pack_results) and
                 valid_pipelines == len(pipeline_results))

    print("\n" + "=" * 80)
    if all_valid:
        print("[PASS] ALL SPECIFICATIONS FULLY IMPLEMENTED AND VALIDATED")
    else:
        print("[FAIL] SPECIFICATION IMPLEMENTATION INCOMPLETE OR HAS ERRORS")
    print("=" * 80)

    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())