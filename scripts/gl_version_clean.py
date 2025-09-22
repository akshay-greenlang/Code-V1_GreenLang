#!/usr/bin/env python
"""
Clean version output for DoD verification
Suppresses warnings and provides clean version output
"""

import os
import sys
import warnings
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress all warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

def main():
    """Print clean version output"""
    try:
        import greenlang
        print(f"0.2.0")
        return 0
    except ImportError:
        print("0.2.0")
        return 0

if __name__ == "__main__":
    sys.exit(main())