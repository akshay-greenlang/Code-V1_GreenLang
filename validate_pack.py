#!/usr/bin/env python3
"""Validate pack.yaml against GreenLang spec v1.0"""

import yaml
import sys
import re
from pathlib import Path

def validate_pack(pack_path):
    """Validate a pack.yaml file against spec requirements."""

    errors = []
    warnings = []

    # Read pack.yaml
    pack_file = Path(pack_path) / "pack.yaml"
    if not pack_file.exists():
        return False, ["pack.yaml not found"]

    with open(pack_file) as f:
        try:
            pack = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return False, [f"YAML parse error: {e}"]

    # Check required fields
    required_fields = ['name', 'version', 'kind', 'description']
    for field in required_fields:
        if field not in pack:
            errors.append(f"Missing required field: {field}")

    # Validate name (must be a slug)
    if 'name' in pack:
        name = pack['name']
        if not isinstance(name, str):
            errors.append(f"Name must be a string, got {type(name).__name__}")
        elif not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', name):
            errors.append(f"Name '{name}' is not a valid slug (lowercase alphanumeric with hyphens)")

    # Validate version (must be semantic version string)
    if 'version' in pack:
        version = pack['version']
        if not isinstance(version, str):
            errors.append(f"Version must be a string, got {type(version).__name__}")
        elif not re.match(r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?(?:\+[a-zA-Z0-9.-]+)?$', version):
            errors.append(f"Version '{version}' is not a valid semantic version")

    # Validate kind
    if 'kind' in pack:
        if pack['kind'] != 'pack':
            errors.append(f"Kind must be 'pack', got '{pack['kind']}'")

    # Validate contents
    if 'contents' in pack:
        contents = pack['contents']
        if not isinstance(contents, dict):
            errors.append(f"Contents must be a dict, got {type(contents).__name__}")
        else:
            # Check pipelines
            if 'pipelines' in contents:
                pipelines = contents['pipelines']
                if not isinstance(pipelines, list):
                    errors.append(f"Contents.pipelines must be a list, got {type(pipelines).__name__}")
                elif len(pipelines) == 0:
                    warnings.append("Contents.pipelines is empty")
                else:
                    # Check that pipeline files exist
                    for pipeline in pipelines:
                        pipeline_path = Path(pack_path) / pipeline
                        if not pipeline_path.exists():
                            errors.append(f"Pipeline file '{pipeline}' not found")

    # Validate authors
    if 'authors' in pack:
        authors = pack['authors']
        if not isinstance(authors, list):
            errors.append(f"Authors must be a list, got {type(authors).__name__}")
        else:
            for i, author in enumerate(authors):
                if not isinstance(author, dict):
                    errors.append(f"Author {i} must be a dict, got {type(author).__name__}")
                elif 'name' not in author:
                    errors.append(f"Author {i} missing 'name' field")
                elif author.get('name') in ['Your Name', 'TODO', '']:
                    warnings.append(f"Author {i} has placeholder name: '{author['name']}'")

    # Print results
    if errors:
        print("FAILED: Pack validation FAILED\n")
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("PASSED: Pack validation PASSED\n")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    # Show pack info
    print(f"\nPack Info:")
    print(f"  Name: {pack.get('name', 'N/A')}")
    print(f"  Version: {pack.get('version', 'N/A')}")
    print(f"  Description: {pack.get('description', 'N/A')}")
    if 'contents' in pack and 'pipelines' in pack['contents']:
        print(f"  Pipelines: {pack['contents']['pipelines']}")

    return len(errors) == 0, errors

if __name__ == "__main__":
    pack_path = sys.argv[1] if len(sys.argv) > 1 else "."
    success, _ = validate_pack(pack_path)
    sys.exit(0 if success else 1)