#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for AgentSpec v2 example

This script validates that the example pack.yaml loads correctly
and demonstrates all features of AgentSpec v2.

Usage:
    python examples/agentspec_v2/test_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from greenlang.specs import from_yaml, to_json_schema


def test_load_example():
    """Test loading the example pack.yaml."""
    print("=" * 70)
    print("Testing AgentSpec v2 Example")
    print("=" * 70)

    # Load the example
    example_path = Path(__file__).parent / "pack.yaml"
    print(f"\n[1] Loading: {example_path}")

    try:
        spec = from_yaml(example_path)
        print(f"    [OK] Loaded successfully")
    except Exception as e:
        print(f"    [ERROR] Failed to load: {e}")
        sys.exit(1)

    # Validate metadata
    print(f"\n[2] Metadata:")
    print(f"    Schema Version: {spec.schema_version}")
    print(f"    ID: {spec.id}")
    print(f"    Name: {spec.name}")
    print(f"    Version: {spec.version}")
    print(f"    Summary: {spec.summary}")
    print(f"    Tags: {', '.join(spec.tags)}")
    print(f"    License: {spec.license}")

    # Validate compute section
    print(f"\n[3] Compute Section:")
    print(f"    Entrypoint: {spec.compute.entrypoint}")
    print(f"    Deterministic: {spec.compute.deterministic}")
    print(f"    Dependencies: {len(spec.compute.dependencies or [])} packages")
    print(f"    Python Version: {spec.compute.python_version}")
    print(f"    Timeout: {spec.compute.timeout_s}s")
    print(f"    Memory Limit: {spec.compute.memory_limit_mb}MB")
    print(f"    Inputs: {', '.join(spec.compute.inputs.keys())}")
    print(f"    Outputs: {', '.join(spec.compute.outputs.keys())}")
    print(f"    Factors: {', '.join(spec.compute.factors.keys())}")

    # Validate AI section
    print(f"\n[4] AI Section:")
    print(f"    JSON Mode: {spec.ai.json_mode}")
    print(f"    System Prompt: {len(spec.ai.system_prompt or '')} chars")
    if spec.ai.budget:
        print(f"    Budget:")
        print(f"      Max Cost: ${spec.ai.budget.max_cost_usd}")
        print(f"      Max Input Tokens: {spec.ai.budget.max_input_tokens}")
        print(f"      Max Output Tokens: {spec.ai.budget.max_output_tokens}")
        print(f"      Max Retries: {spec.ai.budget.max_retries}")
    print(f"    RAG Collections: {', '.join(spec.ai.rag_collections)}")
    print(f"    Tools: {len(spec.ai.tools)} tools")
    for tool in spec.ai.tools:
        print(f"      - {tool.name} (safe={tool.safe})")

    # Validate realtime section
    print(f"\n[5] Realtime Section:")
    print(f"    Default Mode: {spec.realtime.default_mode}")
    print(f"    Snapshot Path: {spec.realtime.snapshot_path}")
    print(f"    Connectors: {len(spec.realtime.connectors)} connectors")
    for conn in spec.realtime.connectors:
        print(f"      - {conn.name} (topic={conn.topic}, required={conn.required})")

    # Validate provenance section
    print(f"\n[6] Provenance Section:")
    print(f"    Pin EF: {spec.provenance.pin_ef}")
    print(f"    GWP Set: {spec.provenance.gwp_set}")
    print(f"    Record Fields: {', '.join(spec.provenance.record)}")

    # Validate security section (if present)
    if spec.security:
        print(f"\n[7] Security Section:")
        allowlist = spec.security.get("allowlist_hosts", [])
        print(f"    Allowlist Hosts: {len(allowlist)} hosts")
        for host in allowlist:
            print(f"      - {host}")
        print(f"    Block on Violation: {spec.security.get('block_on_violation', False)}")

    # Validate tests section (if present)
    if spec.tests:
        print(f"\n[8] Tests Section:")
        golden = spec.tests.get("golden", [])
        properties = spec.tests.get("properties", [])
        print(f"    Golden Tests: {len(golden)} tests")
        for test in golden:
            print(f"      - {test['name']}")
        print(f"    Property Tests: {len(properties)} properties")
        for prop in properties:
            print(f"      - {prop['name']}: {prop['rule']}")

    # Export to JSON Schema
    print(f"\n[9] JSON Schema Export:")
    schema = to_json_schema()
    print(f"    Schema Size: {len(str(schema))} bytes")
    print(f"    Schema ID: {schema.get('$id')}")
    print(f"    Schema Version: {schema.get('$schema')}")

    print(f"\n{'=' * 70}")
    print("[SUCCESS] All validations passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_load_example()
