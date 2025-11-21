#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug script to test a single pipeline"""

from greenlang.sdk.pipeline import Pipeline
import traceback

def test_pipeline(path: str):
    """Test a single pipeline with detailed error output"""
    print(f"Testing: {path}")
    print("-" * 50)

    try:
        pipeline = Pipeline.from_yaml(path)
        print(f"Pipeline loaded: {pipeline.spec.name}")

        validation_errors = pipeline.validate(strict=False)

        if validation_errors:
            print(f"Validation errors: {len(validation_errors)}")
            for error in validation_errors:
                print(f"  - {error}")
        else:
            print("Pipeline is valid!")

    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_pipelines/full_featured.yaml"
    test_pipeline(path)