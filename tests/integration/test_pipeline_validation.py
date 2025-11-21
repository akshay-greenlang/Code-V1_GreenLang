#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to validate pipelines against v1.0 spec"""

import sys
from pathlib import Path
from greenlang.sdk.pipeline import Pipeline

def validate_pipeline(path: str):
    """Validate a pipeline file against v1.0 spec"""
    print(f"\nValidating: {path}")
    print("-" * 50)

    try:
        # Try to find JSON schema
        schema_path = Path(__file__).parent / "schemas" / "gl_pipeline.schema.v1.json"
        if not schema_path.exists():
            schema_path = None
            print("Note: JSON schema not found, using Pydantic validation only")

        # Load and validate pipeline
        pipeline = Pipeline.from_yaml(path)

        # Additional validation
        validation_errors = pipeline.validate(strict=False)

        # Split into errors and warnings (for now, treat all as errors)
        errors = validation_errors
        warnings = []

        if errors:
            print(f"FAILED - {len(errors)} error(s):")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASSED - Pipeline is valid (v1.0 spec)")

        if warnings:
            print(f"WARNING - {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"  - {warning}")

        return len(errors) == 0

    except Exception as e:
        print(f"FAILED - Exception during validation:")
        print(f"  {e}")
        return False

def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("GreenLang Pipeline v1.0 Spec Validation Test")
    print("=" * 60)

    # Find all yaml files in test_pipelines directory
    test_dir = Path(__file__).parent / "test_pipelines"
    pipelines = list(test_dir.glob("*.yaml"))

    if not pipelines:
        print("No pipelines found in packs directory")
        return 1

    print(f"\nFound {len(pipelines)} pipeline(s) to validate")

    # Validate each pipeline
    results = []
    for pipeline_path in pipelines:
        success = validate_pipeline(str(pipeline_path))
        results.append((pipeline_path.stem, success))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed

    for pack_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {pack_name:20} {status}")

    print("\n" + "-" * 60)
    print(f"Total: {len(results)} pipelines")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())