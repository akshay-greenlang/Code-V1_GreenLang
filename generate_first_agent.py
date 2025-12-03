#!/usr/bin/env python
"""
Generate First Agent - Integration Test

This script generates the fuel analyzer agent from the AgentSpec and
validates the end-to-end pipeline works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))
sys.path.insert(0, str(project_root / "sdks" / "python"))

from greenlang.generator.spec_parser import AgentSpecParser
from greenlang.generator.code_generator import CodeGenerator, GenerationOptions

def main():
    """Generate fuel analyzer agent."""

    print("=" * 80)
    print("GreenLang Agent Factory - First Agent Generation")
    print("=" * 80)
    print()

    # Paths
    spec_path = project_root / "examples" / "specs" / "fuel_analyzer.yaml"
    output_dir = project_root / "generated" / "fuel_analyzer_agent"

    print(f"Spec file: {spec_path}")
    print(f"Output dir: {output_dir}")
    print()

    # Step 1: Parse AgentSpec
    print("Step 1: Parsing AgentSpec...")
    try:
        parser = AgentSpecParser(strict=True)
        spec = parser.parse(spec_path)

        print(f"  [OK] Parsed: {spec.name} v{spec.version}")
        print(f"  ID: {spec.id}")
        print(f"  Inputs: {len(spec.inputs)}")
        print(f"  Outputs: {len(spec.outputs)}")
        print(f"  Tools: {len(spec.tools)}")

        if spec.tests:
            print(f"  Golden Tests: {len(spec.tests.golden)}")
            print(f"  Property Tests: {len(spec.tests.properties)}")

        if parser.warnings:
            print()
            print("  Warnings:")
            for warning in parser.warnings:
                print(f"    - {warning}")

        print()

    except Exception as e:
        print(f"  [FAIL] Parse failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 2: Generate Code
    print("Step 2: Generating agent code...")
    try:
        options = GenerationOptions(
            output_dir=output_dir,
            overwrite=True,
            generate_tools=True,
            generate_tests=True,
            generate_readme=True,
            generate_init=True,
            use_async=True,
        )

        generator = CodeGenerator(options=options)
        result = generator.generate(spec, output_dir=output_dir)

        print(f"  [OK] Generated {result.total_lines} lines of code")
        print(f"  Files: {len(result.files)}")
        print(f"  Tools: {result.num_tools}")
        print(f"  Tests: {result.num_tests}")
        print()

    except Exception as e:
        print(f"  [FAIL] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Show Generated Files
    print("Step 3: Generated files...")
    for file in result.files:
        file_path = output_dir / file.relative_path / file.filename if file.relative_path else output_dir / file.filename
        file_size = len(file.content)
        print(f"  [OK] {file_path} ({file_size:,} bytes)")
    print()

    # Step 4: Validation Summary
    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print()
    print(f"Agent: {spec.name} v{spec.version}")
    print(f"Location: {output_dir}")
    print()
    print("Next steps:")
    print(f"  1. cd {output_dir}")
    print(f"  2. Inspect generated code")
    print(f"  3. Implement tool logic in tools.py")
    print(f"  4. Run tests: pytest tests/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
