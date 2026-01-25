#!/usr/bin/env python
"""
Generate Agent - Enhanced Script with CLI Arguments

Usage:
    python generate_agent.py --spec path/to/spec.yaml
    python generate_agent.py --spec spec.yaml --output ./my_agent/
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))
sys.path.insert(0, str(project_root / "sdks" / "python"))

from greenlang.generator.spec_parser import AgentSpecParser
from greenlang.generator.code_generator import CodeGenerator, GenerationOptions


def main():
    parser = argparse.ArgumentParser(description="Generate GreenLang agent from AgentSpec")
    parser.add_argument("--spec", "-s", required=True, help="Path to AgentSpec YAML file")
    parser.add_argument("--output", "-o", help="Output directory (default: generated/<agent_name>)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Error: Spec file not found: {spec_path}")
        return 1

    print("=" * 80)
    print(f"Generating agent from: {spec_path}")
    print("=" * 80)

    # Parse spec
    try:
        spec_parser = AgentSpecParser(strict=True)
        spec = spec_parser.parse(spec_path)
        print(f"[OK] Parsed: {spec.name} v{spec.version}")
    except Exception as e:
        print(f"[FAIL] Parse error: {e}")
        return 1

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "generated" / spec.module_name

    print(f"[OK] Output: {output_dir}")

    # Generate
    try:
        options = GenerationOptions(
            output_dir=output_dir,
            overwrite=args.overwrite or True,
            generate_tools=True,
            generate_tests=True,
            generate_readme=True,
        )
        generator = CodeGenerator(options=options)
        result = generator.generate(spec, output_dir=output_dir)

        print(f"[OK] Generated {result.total_lines} lines, {len(result.files)} files")
        print(f"[OK] Location: {output_dir}")
        return 0

    except Exception as e:
        print(f"[FAIL] Generation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
