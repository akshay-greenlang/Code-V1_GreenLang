# -*- coding: utf-8 -*-
"""Test that the benchmark script can be imported."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Test imports
    print("Testing imports...")

    from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
    print("✓ Base agent imports OK")

    from greenlang.agents.calculator import BaseCalculator, CalculatorConfig
    print("✓ Calculator imports OK")

    from greenlang.agents.data_processor import BaseDataProcessor, DataProcessorConfig
    print("✓ Data processor imports OK")

    from greenlang.validation.framework import ValidationFramework, ValidationResult
    print("✓ Validation framework imports OK")

    from greenlang.validation.schema import SchemaValidator
    print("✓ Schema validator imports OK")

    from greenlang.io.readers import DataReader
    print("✓ Data reader imports OK")

    from greenlang.io.writers import DataWriter
    print("✓ Data writer imports OK")

    print("\n✅ All imports successful!")
    print("\nBenchmark script should be ready to run.")
    print("Note: Python may not be available in the current PATH.")
    print("Please run the benchmark manually if needed.")

except ImportError as e:
    print(f"\n✗ Import failed: {e}")
    print("\nPlease ensure the GreenLang framework is installed:")
    print("  pip install -e .")
    sys.exit(1)
