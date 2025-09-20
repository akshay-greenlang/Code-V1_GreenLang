#!/usr/bin/env python3
"""Validate pipeline YAML against GreenLang spec."""

import yaml
import sys
import re
from pathlib import Path

def validate_pipeline(pipeline_path):
    """Validate a pipeline YAML file against spec requirements."""

    errors = []
    warnings = []

    # Read pipeline YAML
    pipeline_file = Path(pipeline_path)
    if not pipeline_file.exists():
        return False, [f"Pipeline file not found: {pipeline_path}"]

    with open(pipeline_file) as f:
        try:
            pipeline = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return False, [f"YAML parse error: {e}"]

    # Check required fields
    required_fields = ['version', 'name', 'steps']
    for field in required_fields:
        if field not in pipeline:
            errors.append(f"Missing required field: {field}")

    # Validate version (must be an integer)
    if 'version' in pipeline:
        version = pipeline['version']
        if not isinstance(version, int):
            errors.append(f"Version must be an integer, got {type(version).__name__}: {version}")

    # Validate name (must be a slug)
    if 'name' in pipeline:
        name = pipeline['name']
        if not isinstance(name, str):
            errors.append(f"Name must be a string, got {type(name).__name__}")
        elif not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', name):
            errors.append(f"Name '{name}' is not a valid slug (lowercase alphanumeric with hyphens)")

    # Validate steps
    if 'steps' in pipeline:
        steps = pipeline['steps']
        if not isinstance(steps, list):
            errors.append(f"Steps must be a list, got {type(steps).__name__}")
        elif len(steps) == 0:
            errors.append("Steps list cannot be empty")
        else:
            for i, step in enumerate(steps):
                step_errors = validate_step(step, i)
                errors.extend(step_errors)

    # Validate inputs (optional)
    if 'inputs' in pipeline:
        inputs = pipeline['inputs']
        if not isinstance(inputs, dict):
            errors.append(f"Inputs must be a dict, got {type(inputs).__name__}")
        else:
            for input_name, input_spec in inputs.items():
                if not isinstance(input_spec, dict):
                    errors.append(f"Input '{input_name}' must be a dict, got {type(input_spec).__name__}")
                elif 'type' not in input_spec:
                    warnings.append(f"Input '{input_name}' missing 'type' field")

    # Validate outputs (optional)
    if 'outputs' in pipeline:
        outputs = pipeline['outputs']
        if not isinstance(outputs, dict):
            errors.append(f"Outputs must be a dict, got {type(outputs).__name__}")
        else:
            for output_name, output_spec in outputs.items():
                if not isinstance(output_spec, dict):
                    errors.append(f"Output '{output_name}' must be a dict, got {type(output_spec).__name__}")

    # Validate vars (optional)
    if 'vars' in pipeline:
        vars = pipeline['vars']
        if not isinstance(vars, dict):
            errors.append(f"Vars must be a dict, got {type(vars).__name__}")

    # Print results
    if errors:
        print("FAILED: Pipeline validation FAILED\n")
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("PASSED: Pipeline validation PASSED\n")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    # Show pipeline info
    print(f"\nPipeline Info:")
    print(f"  Name: {pipeline.get('name', 'N/A')}")
    print(f"  Version: {pipeline.get('version', 'N/A')}")
    print(f"  Description: {pipeline.get('description', 'N/A')}")
    print(f"  Steps: {len(pipeline.get('steps', []))}")
    if 'steps' in pipeline:
        for step in pipeline['steps']:
            print(f"    - {step.get('name', 'unnamed')}: {step.get('agent', 'N/A')}.{step.get('action', 'N/A')}")

    return len(errors) == 0, errors

def validate_step(step, index):
    """Validate a single pipeline step."""
    errors = []

    if not isinstance(step, dict):
        errors.append(f"Step {index} must be a dict, got {type(step).__name__}")
        return errors

    # Required step fields
    required = ['name', 'agent', 'action']
    for field in required:
        if field not in step:
            errors.append(f"Step {index} missing required field: {field}")

    # Validate step name (should be slug)
    if 'name' in step:
        name = step['name']
        if not isinstance(name, str):
            errors.append(f"Step {index} name must be a string, got {type(name).__name__}")
        elif not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', name):
            errors.append(f"Step {index} name '{name}' is not a valid slug")

    # Validate agent
    if 'agent' in step:
        agent = step['agent']
        if not isinstance(agent, str):
            errors.append(f"Step {index} agent must be a string, got {type(agent).__name__}")

    # Validate action
    if 'action' in step:
        action = step['action']
        if not isinstance(action, str):
            errors.append(f"Step {index} action must be a string, got {type(action).__name__}")

    # Validate optional fields
    if 'condition' in step:
        condition = step['condition']
        if not isinstance(condition, str):
            errors.append(f"Step {index} condition must be a string, got {type(condition).__name__}")

    if 'inputs' in step:
        inputs = step['inputs']
        if not isinstance(inputs, dict):
            errors.append(f"Step {index} inputs must be a dict, got {type(inputs).__name__}")

    if 'outputs' in step:
        outputs = step['outputs']
        if not isinstance(outputs, dict):
            errors.append(f"Step {index} outputs must be a dict, got {type(outputs).__name__}")

    if 'on_error' in step:
        on_error = step['on_error']
        valid_policies = ['stop', 'continue', 'fail', 'skip']
        if on_error not in valid_policies:
            errors.append(f"Step {index} on_error must be one of {valid_policies}, got '{on_error}'")

    return errors

if __name__ == "__main__":
    pipeline_path = sys.argv[1] if len(sys.argv) > 1 else "gl.yaml"
    success, _ = validate_pipeline(pipeline_path)
    sys.exit(0 if success else 1)