#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang AgentSpec v2 - JSON Schema Generator

This script generates the JSON Schema (draft-2020-12) from Pydantic models.
Pydantic is the source of truth; JSON Schema is generated, never hand-edited.

Usage:
    # Generate schema
    python scripts/generate_schema.py

    # Generate and write to file
    python scripts/generate_schema.py --output greenlang/specs/agentspec_v2.json

    # CI check mode (verify generated == committed)
    python scripts/generate_schema.py --check

Design:
- Pydantic models → JSON Schema via model_json_schema()
- Pretty-printed JSON (indent=2, sorted keys)
- CI check prevents schema drift

Author: GreenLang Framework Team
Date: October 2025
Spec: FRMW-201 (AgentSpec v2 Schema + Validators)
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import greenlang
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.specs.agentspec_v2 import to_json_schema


def generate_schema() -> dict:
    """
    Generate JSON Schema from Pydantic models.

    Returns:
        JSON Schema dictionary
    """
    print("Generating JSON Schema from Pydantic models...")
    schema = to_json_schema()
    print(f"[OK] Schema generated: {len(json.dumps(schema))} bytes")
    return schema


def write_schema(schema: dict, output_path: Path) -> None:
    """
    Write JSON Schema to file (pretty-printed).

    Args:
        schema: JSON Schema dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")  # Trailing newline for POSIX compliance

    print(f"[OK] Schema written to: {output_path}")


def check_schema(schema: dict, committed_path: Path) -> bool:
    """
    Check if generated schema matches committed schema.

    This is the CI check that prevents schema drift:
    - Pydantic models are modified
    - JSON Schema is NOT regenerated
    - CI fails

    Args:
        schema: Generated schema
        committed_path: Path to committed schema file

    Returns:
        True if schemas match, False otherwise
    """
    if not committed_path.exists():
        print(f"[ERROR] Committed schema not found: {committed_path}")
        print("  Run: python scripts/generate_schema.py --output greenlang/specs/agentspec_v2.json")
        return False

    with open(committed_path, "r", encoding="utf-8") as f:
        committed_schema = json.load(f)

    # Normalize for comparison (pretty-print both)
    generated_json = json.dumps(schema, indent=2, sort_keys=True)
    committed_json = json.dumps(committed_schema, indent=2, sort_keys=True)

    if generated_json == committed_json:
        print(f"[OK] Schema matches committed version: {committed_path}")
        return True
    else:
        print(f"[ERROR] Schema MISMATCH detected!")
        print(f"  Generated schema differs from committed: {committed_path}")
        print(f"  This means Pydantic models were changed but JSON Schema was not regenerated.")
        print(f"  Fix: python scripts/generate_schema.py --output greenlang/specs/agentspec_v2.json")
        print(f"  Then commit the updated schema file.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON Schema for AgentSpec v2 from Pydantic models"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI check mode: verify generated schema matches committed schema"
    )
    args = parser.parse_args()

    # Generate schema
    schema = generate_schema()

    if args.check:
        # CI check mode
        committed_path = Path("greenlang/specs/agentspec_v2.json")
        if check_schema(schema, committed_path):
            print("\n[OK] CI Check PASSED: Schema is up-to-date")
            sys.exit(0)
        else:
            print("\n[ERROR] CI Check FAILED: Schema drift detected")
            sys.exit(1)
    elif args.output:
        # Write to file
        write_schema(schema, args.output)
        print("\n[OK] Schema generation complete")
    else:
        # Print to stdout
        print(json.dumps(schema, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
