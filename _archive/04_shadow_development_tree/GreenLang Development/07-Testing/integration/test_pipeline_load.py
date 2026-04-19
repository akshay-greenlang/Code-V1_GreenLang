#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test loading the pipeline with GreenLang SDK."""

import yaml
import sys
from pathlib import Path

def test_pipeline_load(pipeline_path):
    """Test if the pipeline can be loaded and parsed."""

    pipeline_file = Path(pipeline_path)
    if not pipeline_file.exists():
        print(f"ERROR: Pipeline file not found: {pipeline_path}")
        return False

    with open(pipeline_file) as f:
        try:
            pipeline = yaml.safe_load(f)
            print(f"SUCCESS: Pipeline loaded successfully")
            print(f"  Name: {pipeline.get('name')}")
            print(f"  Version: {pipeline.get('version')}")
            print(f"  Steps: {len(pipeline.get('steps', []))}")

            # Check references
            print("\nChecking references:")
            for step in pipeline.get('steps', []):
                step_name = step.get('name')

                # Check input references
                if 'inputs' in step:
                    for key, value in step['inputs'].items():
                        if isinstance(value, str) and value.startswith('$'):
                            print(f"  Step '{step_name}' input '{key}': {value}")

                # Check condition references
                if 'condition' in step:
                    print(f"  Step '{step_name}' condition: {step['condition']}")

                # Check output references
                if 'outputs' in step:
                    for key, value in step['outputs'].items():
                        if isinstance(value, str) and value.startswith('$'):
                            print(f"  Step '{step_name}' output '{key}': {value}")

            # Check output references
            if 'outputs' in pipeline:
                print("\nPipeline outputs:")
                for name, spec in pipeline['outputs'].items():
                    if 'value' in spec and isinstance(spec['value'], str) and spec['value'].startswith('$'):
                        print(f"  {name}: {spec['value']}")

            return True

        except yaml.YAMLError as e:
            print(f"ERROR: Failed to parse YAML: {e}")
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False

if __name__ == "__main__":
    pipeline_path = sys.argv[1] if len(sys.argv) > 1 else "packs/demo-acceptance-test/gl.yaml"
    success = test_pipeline_load(pipeline_path)
    sys.exit(0 if success else 1)