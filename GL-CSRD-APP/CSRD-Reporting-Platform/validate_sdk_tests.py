#!/usr/bin/env python3
"""
Quick validation script to check if SDK test imports work correctly.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Test imports from SDK
    from sdk.csrd_sdk import (
        CSRDConfig,
        CSRDReport,
        ComplianceStatus,
        ESRSMetrics,
        MaterialityAssessment,
        csrd_assess_materiality,
        csrd_audit_compliance,
        csrd_build_report,
        csrd_calculate_metrics,
        csrd_validate_data,
    )

    print("✅ All SDK imports successful!")
    print(f"   - CSRDConfig: {CSRDConfig}")
    print(f"   - CSRDReport: {CSRDReport}")
    print(f"   - csrd_build_report: {csrd_build_report}")
    print(f"   - csrd_validate_data: {csrd_validate_data}")
    print(f"   - csrd_calculate_metrics: {csrd_calculate_metrics}")
    print()

    # Test CSRDConfig creation
    config = CSRDConfig(
        company_name="Test Corp",
        company_lei="549300TEST1234567",
        reporting_year=2024,
        sector="Technology"
    )
    print(f"✅ CSRDConfig created: {config.company_name}")
    print()

    # Test config serialization
    config_dict = config.to_dict()
    print(f"✅ Config serialization works: {len(config_dict)} fields")
    print()

    # Test imports for test fixtures
    import pandas as pd
    import pytest
    import yaml
    print("✅ All test dependencies imported successfully!")
    print()

    print("=" * 60)
    print("SDK TEST VALIDATION: ALL CHECKS PASSED ✅")
    print("=" * 60)
    print()
    print("The test file is ready to run with pytest:")
    print("  pytest tests/test_sdk.py -v")
    print()

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
