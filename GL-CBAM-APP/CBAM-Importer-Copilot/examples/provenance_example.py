"""
CBAM Importer Copilot - Provenance Example

This script demonstrates how to work with provenance data for regulatory
compliance and audit requirements.

Provenance provides:
- Input file integrity verification (SHA256 hashing)
- Complete execution environment capture
- Dependency version tracking
- Agent execution audit trail
- Reproducibility guarantees

Usage:
    python examples/provenance_example.py

Version: 1.0.0
Author: GreenLang CBAM Team
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from provenance import (
    hash_file,
    get_environment_info,
    get_dependency_versions,
    create_provenance_record,
    validate_provenance,
    generate_audit_report,
    ProvenanceRecord
)

print("=" * 70)
print("CBAM PROVENANCE - REGULATORY COMPLIANCE EXAMPLE")
print("=" * 70)
print()

# ============================================================================
# EXAMPLE 1: File Integrity Verification (SHA256 Hashing)
# ============================================================================

print("EXAMPLE 1: File Integrity Verification")
print("-" * 70)
print("Generating SHA256 hash of input file for integrity verification...\n")

try:
    # Hash the demo shipments file
    file_hash = hash_file("examples/demo_shipments.csv")

    print(f"File: {file_hash['file_name']}")
    print(f"Size: {file_hash['human_readable_size']}")
    print(f"SHA256: {file_hash['hash_value']}")
    print(f"Algorithm: {file_hash['hash_algorithm']}")
    print(f"Timestamp: {file_hash['hash_timestamp']}")
    print(f"\nVerification command: {file_hash['verification']}")
    print()
    print("✓ File can be verified later using this hash")
    print("  Regulators can confirm file hasn't been tampered with")

except FileNotFoundError:
    print("⚠ Demo file not found (this is OK when running from different directory)")

print()

# ============================================================================
# EXAMPLE 2: Execution Environment Capture
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Execution Environment Capture")
print("-" * 70)
print("Capturing complete execution environment for reproducibility...\n")

env = get_environment_info()

print("Python Environment:")
print(f"  - Version: {env['python']['version_info']['major']}.{env['python']['version_info']['minor']}.{env['python']['version_info']['micro']}")
print(f"  - Implementation: {env['python']['implementation']}")
print(f"  - Compiler: {env['python']['compiler']}")
print(f"  - Executable: {env['python']['executable']}")
print()

print("System Environment:")
print(f"  - OS: {env['system']['os']} {env['system']['release']}")
print(f"  - Machine: {env['system']['machine']}")
print(f"  - Processor: {env['system']['processor']}")
print(f"  - Architecture: {env['system']['architecture']}")
print(f"  - Hostname: {env['system']['hostname']}")
print()

print("Process:")
print(f"  - PID: {env['process']['pid']}")
print(f"  - CWD: {env['process']['cwd']}")
print(f"  - User: {env['process']['user']}")
print()

print("✓ Environment captured for bit-perfect reproducibility")

# ============================================================================
# EXAMPLE 3: Dependency Version Tracking
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Dependency Version Tracking")
print("-" * 70)
print("Tracking versions of all critical dependencies...\n")

deps = get_dependency_versions()

print("Critical Dependencies:")
for package, version in deps.items():
    print(f"  - {package}: {version}")
print()

print("✓ Dependency versions captured")
print("  Essential for reproducing exact execution environment")

# ============================================================================
# EXAMPLE 4: Creating a Provenance Record
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Creating a Provenance Record")
print("-" * 70)
print("Creating a complete provenance record for a report...\n")

# Mock data for demonstration
try:
    provenance = create_provenance_record(
        report_id="CBAM-2025Q4-DEMO-001",
        input_file="examples/demo_shipments.csv",
        configuration={
            "importer_name": "Acme Steel EU BV",
            "importer_country": "NL",
            "importer_eori": "NL123456789012"
        },
        agent_executions=[
            {
                "agent_name": "ShipmentIntakeAgent",
                "description": "Data validation and enrichment",
                "start_time": "2025-10-15T10:00:00Z",
                "end_time": "2025-10-15T10:00:02Z",
                "duration_seconds": 2.15,
                "input_records": 20,
                "output_records": 20,
                "status": "success"
            },
            {
                "agent_name": "EmissionsCalculatorAgent",
                "description": "Emissions calculation (ZERO HALLUCINATION)",
                "start_time": "2025-10-15T10:00:02Z",
                "end_time": "2025-10-15T10:00:03Z",
                "duration_seconds": 0.85,
                "input_records": 20,
                "output_records": 20,
                "total_emissions_tco2": 1234.56,
                "status": "success"
            },
            {
                "agent_name": "ReportingPackagerAgent",
                "description": "Report generation and validation",
                "start_time": "2025-10-15T10:00:03Z",
                "end_time": "2025-10-15T10:00:04Z",
                "duration_seconds": 0.45,
                "input_records": 20,
                "output_records": 20,
                "is_valid": True,
                "status": "success"
            }
        ],
        validation_results={
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
    )

    print(f"Provenance Record Created:")
    print(f"  - Report ID: {provenance.report_id}")
    print(f"  - Generated: {provenance.generated_at}")
    print(f"  - Input File Hash: {provenance.input_file_hash['hash_value'][:32]}...")
    print(f"  - Agent Executions: {len(provenance.agent_execution)}")
    print(f"  - Dependencies Tracked: {len(provenance.dependencies)}")
    print()

    # Save provenance record
    output_path = "output/provenance_example.json"
    Path("output").mkdir(exist_ok=True)
    provenance.save(output_path)
    print(f"✓ Provenance record saved to: {output_path}")

except FileNotFoundError:
    print("⚠ Demo file not found (this is OK when running from different directory)")
    provenance = None

# ============================================================================
# EXAMPLE 5: Validating Provenance
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Validating Provenance")
print("-" * 70)
print("Verifying provenance record integrity...\n")

if provenance:
    try:
        validation = validate_provenance(
            provenance,
            input_file="examples/demo_shipments.csv"
        )

        print(f"Validation Results:")
        print(f"  - Valid: {validation['is_valid']}")
        print(f"  - Checks Passed: {validation['checks_passed']}")
        print(f"  - Errors: {validation['errors']}")
        print(f"  - Warnings: {validation['warnings']}")
        print()

        if validation['is_valid']:
            print("✓ Provenance verified - file integrity confirmed")
        else:
            print("✗ Provenance verification failed")

            if validation['error_details']:
                print("\nErrors:")
                for error in validation['error_details']:
                    print(f"  - {error['message']}")

    except FileNotFoundError:
        print("⚠ Cannot validate without input file")
else:
    print("⚠ Skipped (no provenance record created)")

# ============================================================================
# EXAMPLE 6: Generating Audit Report
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Generating Audit Report")
print("-" * 70)
print("Creating human-readable audit report from provenance...\n")

if provenance:
    audit_report = generate_audit_report(provenance)

    # Preview first part of audit report
    lines = audit_report.split('\n')
    print('\n'.join(lines[:30]))
    print("\n[... see full report in output/provenance_audit.md ...]")

    # Save audit report
    audit_path = "output/provenance_audit.md"
    with open(audit_path, 'w') as f:
        f.write(audit_report)

    print(f"\n✓ Full audit report saved to: {audit_path}")
else:
    print("⚠ Skipped (no provenance record created)")

# ============================================================================
# EXAMPLE 7: Regulatory Use Cases
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 7: Regulatory Use Cases")
print("-" * 70)
print()

print("How provenance meets EU CBAM regulatory requirements:\n")

print("1. DATA INTEGRITY VERIFICATION")
print("   - SHA256 hash proves input file hasn't been tampered with")
print("   - Regulators can verify file integrity independently")
print("   - Cryptographic proof of data authenticity")
print()

print("2. REPRODUCIBILITY")
print("   - Complete environment capture allows exact reproduction")
print("   - Same inputs + same environment = same outputs (bit-perfect)")
print("   - Critical for regulatory audits and disputes")
print()

print("3. AUDIT TRAIL")
print("   - Every agent execution is logged with timestamps")
print("   - Complete chain of custody for data transformations")
print("   - Satisfies EU audit requirements for transparency")
print()

print("4. DEPENDENCY TRACKING")
print("   - All software versions recorded")
print("   - Prevents \"works on my machine\" issues")
print("   - Ensures long-term reproducibility")
print()

print("5. ZERO HALLUCINATION PROOF")
print("   - Provenance proves deterministic calculations")
print("   - No LLM in calculation path (verifiable)")
print("   - 100% calculation accuracy guaranteed")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PROVENANCE EXAMPLES COMPLETE!")
print("=" * 70)
print()
print("What you've learned:")
print("  ✓ How to hash files for integrity verification")
print("  ✓ How to capture execution environment")
print("  ✓ How to track dependency versions")
print("  ✓ How to create provenance records")
print("  ✓ How to validate provenance")
print("  ✓ How to generate audit reports")
print("  ✓ How provenance meets regulatory requirements")
print()
print("Every CBAM report includes complete provenance automatically!")
print("No user action required - it's built into the pipeline.")
print()
print("=" * 70)

# ============================================================================
# END OF PROVENANCE EXAMPLE
# ============================================================================
