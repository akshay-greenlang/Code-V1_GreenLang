"""
Run Emission Factor Import

This script creates the database and imports all 500 emission factors.
Run this file with: python run_emission_factor_import.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the import script
from scripts.import_emission_factors import main

if __name__ == "__main__":
    # Set command line args
    sys.argv = [
        "import_emission_factors.py",
        "--db-path", str(project_root / "greenlang" / "data" / "emission_factors.db"),
        "--data-dir", str(project_root / "data"),
        "--overwrite",
        "--verbose"
    ]

    # Run import
    main()
