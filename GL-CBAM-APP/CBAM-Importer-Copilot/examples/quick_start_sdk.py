"""
CBAM IMPORTER COPILOT - PYTHON SDK QUICK START
===============================================

This script demonstrates how to use the CBAM Copilot Python SDK
in your applications.

Prerequisites:
    pip install -r requirements.txt

Usage:
    python examples/quick_start_sdk.py

Version: 1.0.0
Author: GreenLang CBAM Team
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import SDK
from sdk import cbam_build_report, cbam_validate_shipments, CBAMConfig

print("=" * 70)
print("CBAM COPILOT - PYTHON SDK QUICK START")
print("=" * 70)
print()

# ============================================================================
# EXAMPLE 1: Basic Report Generation
# ============================================================================

print("EXAMPLE 1: Basic Report Generation")
print("-" * 70)
print("Generating a CBAM report with minimal code...\n")

try:
    report = cbam_build_report(
        input_file="examples/demo_shipments.csv",
        importer_name="Acme Steel EU BV",
        importer_country="NL",
        importer_eori="NL123456789012",
        declarant_name="John Smith",
        declarant_position="Compliance Officer",
        output_json="output/sdk_report_example1.json",
        output_summary="output/sdk_summary_example1.md"
    )

    print("✓ Report generated successfully!\n")
    print(report.summary())
    print()

except FileNotFoundError as e:
    print(f"⚠ File not found: {e}")
    print("  (This is OK if running from a different directory)")
    print()
except Exception as e:
    print(f"✗ Error: {e}")
    print()

# ============================================================================
# EXAMPLE 2: Using Configuration Object
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Using Configuration Object")
print("-" * 70)
print("Store your importer info in a config object for reuse...\n")

# Create config object
config = CBAMConfig(
    importer_name="Acme Steel EU BV",
    importer_country="NL",
    importer_eori="NL123456789012",
    declarant_name="John Smith",
    declarant_position="Compliance Officer"
)

print(f"Config created for: {config.importer_name}")
print(f"Country: {config.importer_country}, EORI: {config.importer_eori}\n")

# Use config for multiple reports
try:
    report = cbam_build_report(
        input_file="examples/demo_shipments.csv",
        config=config,
        output_json="output/sdk_report_example2.json"
    )

    print("✓ Report generated using config object!\n")
    print(f"Report ID: {report.report_id}")
    print(f"Total Emissions: {report.total_emissions_tco2:.2f} tCO2")
    print(f"Valid: {report.is_valid}")
    print()

except Exception as e:
    print(f"⚠ Skipping (demo files not found): {e}\n")

# ============================================================================
# EXAMPLE 3: Loading Config from YAML
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Loading Config from YAML")
print("-" * 70)
print("Load configuration from a YAML file...\n")

try:
    # Load from YAML
    config = CBAMConfig.from_yaml("config/cbam_config.yaml")

    print("⚠ Note: config/cbam_config.yaml is a template")
    print("  You'll need to fill in actual values for production use\n")

except FileNotFoundError:
    print("⚠ config/cbam_config.yaml not found")
    print("  Create it by copying config/cbam_config.yaml.template\n")
except Exception as e:
    print(f"⚠ Could not load config: {e}\n")

# ============================================================================
# EXAMPLE 4: Validation Only (Pre-flight Check)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Validation Only (Pre-flight Check)")
print("-" * 70)
print("Validate shipments before generating the full report...\n")

try:
    result = cbam_validate_shipments(
        input_file="examples/demo_shipments.csv"
    )

    metadata = result['metadata']

    print(f"Total Records: {metadata['total_records']}")
    print(f"Valid Records: {metadata['valid_records']}")
    print(f"Invalid Records: {metadata['invalid_records']}")
    print(f"Warnings: {metadata['warnings']}")

    if metadata['invalid_records'] == 0:
        print("\n✓ All shipments are valid - ready to generate report!")
    else:
        print("\n⚠ Some shipments have errors - review before reporting")

    print()

except Exception as e:
    print(f"⚠ Skipping (demo files not found): {e}\n")

# ============================================================================
# EXAMPLE 5: Working with Pandas DataFrames
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Working with Pandas DataFrames")
print("-" * 70)
print("Use the SDK with pandas DataFrames (great for data pipelines)...\n")

try:
    import pandas as pd

    # Read shipments into DataFrame
    df = pd.read_csv("examples/demo_shipments.csv")

    print(f"Loaded {len(df)} shipments from DataFrame\n")
    print("First 3 shipments:")
    print(df.head(3)[['shipment_id', 'cn_code', 'origin_country', 'net_mass_kg']])
    print()

    # Optional: Clean or transform data
    # df = df[df['net_mass_kg'] > 0]  # Filter out zero mass
    # df['import_date'] = pd.to_datetime(df['import_date'])  # Parse dates

    # Generate report from DataFrame
    report = cbam_build_report(
        input_dataframe=df,  # Pass DataFrame instead of file
        config=CBAMConfig(
            importer_name="Acme Steel EU BV",
            importer_country="NL",
            importer_eori="NL123456789012",
            declarant_name="John Smith",
            declarant_position="Compliance Officer"
        ),
        output_json="output/sdk_report_example5.json"
    )

    print("✓ Report generated from DataFrame!\n")

    # Export detailed goods back to DataFrame
    detailed_df = report.to_dataframe()
    print(f"Exported {len(detailed_df)} detailed goods to DataFrame")
    print()

except ImportError:
    print("⚠ pandas not installed: pip install pandas")
    print()
except Exception as e:
    print(f"⚠ Skipping (demo files not found): {e}\n")

# ============================================================================
# EXAMPLE 6: Accessing Report Data Programmatically
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Accessing Report Data Programmatically")
print("-" * 70)
print("Access report data using convenient properties...\n")

try:
    report = cbam_build_report(
        input_file="examples/demo_shipments.csv",
        config=CBAMConfig(
            importer_name="Acme Steel EU BV",
            importer_country="NL",
            importer_eori="NL123456789012",
            declarant_name="John Smith",
            declarant_position="Compliance Officer"
        )
    )

    # Access via properties
    print("Report Properties:")
    print(f"  Report ID: {report.report_id}")
    print(f"  Quarter: {report.quarter}")
    print(f"  Generated: {report.generated_at}")
    print(f"  Shipments: {report.total_shipments}")
    print(f"  Mass: {report.total_mass_tonnes:.2f} tonnes")
    print(f"  Emissions: {report.total_emissions_tco2:.2f} tCO2")
    print(f"  Valid: {report.is_valid}")
    print()

    # Check for errors
    if report.errors:
        print(f"⚠ Found {len(report.errors)} validation errors:")
        for error in report.errors[:3]:  # Show first 3
            print(f"  - {error.get('message', 'Unknown error')}")
        print()

    # Access raw report dict
    raw = report.to_dict()
    print(f"Raw report contains {len(raw)} top-level keys:")
    print(f"  {', '.join(raw.keys())}")
    print()

    # Save to JSON
    report.save_json("output/sdk_report_example6.json")
    print("✓ Report saved to output/sdk_report_example6.json")

    # Save summary
    report.save_summary("output/sdk_summary_example6.md")
    print("✓ Summary saved to output/sdk_summary_example6.md")
    print()

except Exception as e:
    print(f"⚠ Skipping (demo files not found): {e}\n")

# ============================================================================
# EXAMPLE 7: Integration with ERP Systems
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 7: Integration with ERP Systems")
print("-" * 70)
print("Example of how to integrate CBAM reporting into your ERP...\n")

print("Typical ERP integration workflow:")
print()
print("1. Extract shipments from ERP database:")
print("   SELECT * FROM shipments WHERE import_date >= '2025-10-01'")
print()
print("2. Convert to DataFrame:")
print("   df = pd.read_sql(query, connection)")
print()
print("3. Generate CBAM report:")
print("   report = cbam_build_report(input_dataframe=df, config=config)")
print()
print("4. Store report in ERP:")
print("   INSERT INTO cbam_reports (quarter, report_json, generated_at)")
print("   VALUES ('{report.quarter}', '{report.to_json()}', NOW())")
print()
print("5. Notify compliance team:")
print("   send_email(to='compliance@company.com',")
print("              subject=f'CBAM Report {report.quarter}',")
print("              body=report.summary())")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("QUICK START COMPLETE! 🎉")
print("=" * 70)
print()
print("What you've learned:")
print("  ✓ Basic report generation with cbam_build_report()")
print("  ✓ Using CBAMConfig for reusable configuration")
print("  ✓ Pre-flight validation with cbam_validate_shipments()")
print("  ✓ Working with pandas DataFrames")
print("  ✓ Accessing report data programmatically")
print("  ✓ Saving reports and summaries")
print("  ✓ ERP integration patterns")
print()
print("Next steps:")
print("  1. Import the SDK in your application:")
print("     from cbam_copilot import cbam_build_report, CBAMConfig")
print()
print("  2. Create a CBAMConfig with your company info")
print()
print("  3. Call cbam_build_report() with your shipment data")
print()
print("  4. Review the CBAMReport object and save outputs")
print()
print("For help:")
print("  - Read README.md for full documentation")
print("  - Check sdk/cbam_sdk.py for API reference")
print("  - Email: cbam@greenlang.io")
print()
print("=" * 70)

# ============================================================================
# END OF QUICK START
# ============================================================================
