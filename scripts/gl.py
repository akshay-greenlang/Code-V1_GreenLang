#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct entry point script for gl command
This ensures the gl command works on Windows after pip install
"""

import sys
import os

# Fix encoding issues on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

def main():
    """Main entry point for the gl command"""
    try:
        from greenlang.cli.main import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"Error: Could not import greenlang CLI: {e}", file=sys.stderr)
        print("\nPlease ensure greenlang-cli is installed:", file=sys.stderr)
        print("  pip install greenlang-cli", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()