#!/usr/bin/env python3
"""
End-to-End Test for Agent Generator

This script tests the complete generation pipeline:
1. Parse pack.yaml using YAMLParser
2. Validate spec using SpecValidator
3. Generate agent code using AgentGenerator
4. Generate model code using ModelGenerator
5. Output the results
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.agent_generator.parser.yaml_parser import YAMLParser, ParseError
from backend.agent_generator.parser.spec_validator import SpecValidator
from backend.agent_generator.generators.agent_gen import AgentGenerator
from backend.agent_generator.generators.model_gen import ModelGenerator
from backend.agent_generator.config import GeneratorConfig


def main():
    """Run end-to-end generation test."""
    print("=" * 80)
    print("AGENT FACTORY - END-TO-END GENERATION TEST")
    print("=" * 80)

    # Configuration
    pack_yaml_path = project_root / "specs" / "carbon-emissions" / "pack.yaml"

    print(f"\n[1/5] Loading pack.yaml from: {pack_yaml_path}")

    # Step 1: Parse YAML
    parser = YAMLParser()
    try:
        spec = parser.parse(pack_yaml_path)
        print(f"  SUCCESS: Parsed spec for '{spec.pack.name}' v{spec.pack.version}")
        print(f"  - Agents: {len(spec.agents)}")
        print(f"  - Tools: {len(spec.tools)}")
    except ParseError as e:
        print(f"  FAILED: {e}")
        return 1
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return 1

    # Step 2: Validate spec
    print(f"\n[2/5] Validating specification...")
    validator = SpecValidator()
    result = validator.validate(spec)

    if result.is_valid:
        print(f"  SUCCESS: Spec is valid")
        if result.warnings:
            print(f"  - Warnings: {len(result.warnings)}")
            for w in result.warnings[:3]:
                print(f"    - {w}")
    else:
        print(f"  WARNING: Validation issues found")
        print(f"  - Errors: {len(result.errors)}")
        for e in result.errors[:5]:
            print(f"    - {e}")
        # Continue anyway for demonstration

    # Step 3: Generate agent code
    print(f"\n[3/5] Generating agent code...")
    config = GeneratorConfig()
    agent_generator = AgentGenerator(config)

    if spec.agents:
        agent = spec.agents[0]
        print(f"  Generating for agent: {agent.name}")

        try:
            agent_code = agent_generator.generate_agent(spec, agent)
            print(f"  SUCCESS: Generated {len(agent_code)} characters of agent code")

            # Show first 50 lines
            lines = agent_code.split('\n')[:50]
            print("\n  --- Agent Code Preview (first 50 lines) ---")
            for i, line in enumerate(lines, 1):
                print(f"  {i:3}: {line}")
            print("  --- End Preview ---")

        except Exception as e:
            print(f"  ERROR generating agent: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  SKIPPED: No agents defined in spec")

    # Step 4: Generate model code
    print(f"\n[4/5] Generating model code...")
    model_generator = ModelGenerator(config)

    try:
        model_code = model_generator.generate_models(spec)
        print(f"  SUCCESS: Generated {len(model_code)} characters of model code")

        # Show first 50 lines
        lines = model_code.split('\n')[:50]
        print("\n  --- Model Code Preview (first 50 lines) ---")
        for i, line in enumerate(lines, 1):
            print(f"  {i:3}: {line}")
        print("  --- End Preview ---")

    except Exception as e:
        print(f"  ERROR generating models: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Generate tests
    print(f"\n[5/5] Generating test code...")
    if spec.agents:
        agent = spec.agents[0]
        try:
            test_code = agent_generator.generate_tests(spec, agent)
            print(f"  SUCCESS: Generated {len(test_code)} characters of test code")

            # Show first 30 lines
            lines = test_code.split('\n')[:30]
            print("\n  --- Test Code Preview (first 30 lines) ---")
            for i, line in enumerate(lines, 1):
                print(f"  {i:3}: {line}")
            print("  --- End Preview ---")

        except Exception as e:
            print(f"  ERROR generating tests: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("END-TO-END TEST COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
