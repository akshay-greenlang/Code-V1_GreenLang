#!/usr/bin/env python
"""Validate GL-001 configuration files."""

import yaml
import json
import sys
from pathlib import Path

def validate_files():
    """Validate all configuration files."""
    base_path = Path(__file__).parent

    validation_results = {
        "pack.yaml": False,
        "gl.yaml": False,
        "agent_spec.yaml": False,
        "run.json": False,
        "process_heat_orchestrator.py": False
    }

    errors = []

    # Validate pack.yaml
    try:
        with open(base_path / 'pack.yaml', 'r') as f:
            yaml.safe_load(f)
        validation_results["pack.yaml"] = True
        print("VALID: pack.yaml")
    except Exception as e:
        errors.append(f"pack.yaml: {str(e)}")
        print(f"ERROR: pack.yaml - {str(e)}")

    # Validate gl.yaml
    try:
        with open(base_path / 'gl.yaml', 'r') as f:
            yaml.safe_load(f)
        validation_results["gl.yaml"] = True
        print("VALID: gl.yaml")
    except Exception as e:
        errors.append(f"gl.yaml: {str(e)}")
        print(f"ERROR: gl.yaml - {str(e)}")

    # Validate agent_spec.yaml
    try:
        with open(base_path / 'agent_spec.yaml', 'r') as f:
            yaml.safe_load(f)
        validation_results["agent_spec.yaml"] = True
        print("VALID: agent_spec.yaml")
    except Exception as e:
        errors.append(f"agent_spec.yaml: {str(e)}")
        print(f"ERROR: agent_spec.yaml - {str(e)}")

    # Validate run.json
    try:
        with open(base_path / 'run.json', 'r') as f:
            json.load(f)
        validation_results["run.json"] = True
        print("VALID: run.json")
    except Exception as e:
        errors.append(f"run.json: {str(e)}")
        print(f"ERROR: run.json - {str(e)}")

    # Validate Python syntax
    try:
        import py_compile
        py_compile.compile(str(base_path / 'process_heat_orchestrator.py'), doraise=True)
        validation_results["process_heat_orchestrator.py"] = True
        print("VALID: process_heat_orchestrator.py (syntax)")
    except Exception as e:
        errors.append(f"process_heat_orchestrator.py: {str(e)}")
        print(f"ERROR: process_heat_orchestrator.py - {str(e)}")

    # Test imports
    try:
        from process_heat_orchestrator import ProcessHeatOrchestrator, ProcessData
        print("VALID: Module imports work")
    except Exception as e:
        errors.append(f"Imports: {str(e)}")
        print(f"ERROR: Imports - {str(e)}")

    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    for file, status in validation_results.items():
        print(f"{file}: {'PASS' if status else 'FAIL'}")

    if errors:
        print("\nERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\nAll configuration files are valid!")
        return True

if __name__ == "__main__":
    success = validate_files()
    sys.exit(0 if success else 1)
