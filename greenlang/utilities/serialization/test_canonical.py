"""
Test and demonstration script for canonical JSON serialization.

This script shows the difference between canonical and non-canonical JSON,
and demonstrates how canonical serialization ensures consistent hashing.
"""

import json
import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID
from greenlang.serialization import (
    canonical_dumps,
    canonical_hash,
    canonical_equals,
    diff_canonical,
    CanonicalJSONEncoder,
)


def demonstrate_canonical_vs_standard():
    """Show the difference between canonical and standard JSON."""

    print("=" * 70)
    print("CANONICAL vs STANDARD JSON SERIALIZATION")
    print("=" * 70 + "\n")

    # Create test data with various types
    test_data = {
        "organization": "GreenLang",
        "version": 1.0000,  # Trailing zeros
        "active": True,
        "emissions": {
            "scope1": Decimal("1234.5600"),  # Decimal with trailing zeros
            "scope2": 2345.000,
            "scope3": 3456,
        },
        "metadata": {
            "created_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "uuid": UUID("12345678-1234-5678-1234-567812345678"),
            "tags": ["carbon", "sustainability", "reporting"],
        },
        "values": [3.14159, 2.0000, 1, 0.100],
        "z_last_key": "Should appear last in canonical",
        "a_first_key": "Should appear first in canonical",
    }

    print("1. ORIGINAL DATA STRUCTURE:")
    print("-" * 40)
    print(f"Type: {type(test_data)}")
    print(f"Keys order: {list(test_data.keys())}")
    print()

    # Standard JSON (pretty-printed, unordered)
    print("2. STANDARD JSON (Pretty-printed, unordered):")
    print("-" * 40)
    standard_json = json.dumps(test_data, indent=2, default=str)
    print(standard_json[:500] + "..." if len(standard_json) > 500 else standard_json)
    print(f"\nLength: {len(standard_json)} bytes")
    standard_hash = hashlib.sha256(standard_json.encode()).hexdigest()
    print(f"SHA-256: {standard_hash[:32]}...")
    print()

    # Canonical JSON (minimal, ordered)
    print("3. CANONICAL JSON (Minimal, ordered):")
    print("-" * 40)
    canonical_json = canonical_dumps(test_data)
    print(canonical_json[:500] + "..." if len(canonical_json) > 500 else canonical_json)
    print(f"\nLength: {len(canonical_json)} bytes")
    canonical_hash_value = canonical_hash(test_data)
    print(f"SHA-256: {canonical_hash_value[:32]}...")
    print()

    # Key differences
    print("4. KEY DIFFERENCES:")
    print("-" * 40)
    print(f"* Size reduction: {len(standard_json) - len(canonical_json)} bytes "
          f"({(1 - len(canonical_json)/len(standard_json))*100:.1f}% smaller)")
    print(f"* Whitespace: Removed (no spaces, newlines, or indentation)")
    print(f"* Key ordering: Alphabetical (deterministic)")
    print(f"* Float normalization: Trailing zeros removed (1.0000 -> 1)")
    print(f"* Special types: Handled (Decimal, UUID, datetime)")
    print(f"* Hash consistency: Always same for equivalent data")
    print()


def demonstrate_hash_consistency():
    """Show that canonical hashing is consistent regardless of input order."""

    print("=" * 70)
    print("HASH CONSISTENCY DEMONSTRATION")
    print("=" * 70 + "\n")

    # Create two equivalent objects with different key ordering
    obj1 = {
        "z_key": 3,
        "a_key": 1,
        "m_key": {
            "nested_z": True,
            "nested_a": False
        },
        "values": [1.000, 2.00, 3.0]
    }

    obj2 = {
        "a_key": 1,
        "m_key": {
            "nested_a": False,
            "nested_z": True
        },
        "z_key": 3,
        "values": [1, 2.0, 3.000]
    }

    print("Object 1 (keys: z, a, m):")
    print(json.dumps(obj1, indent=2))
    print()

    print("Object 2 (keys: a, m, z):")
    print(json.dumps(obj2, indent=2))
    print()

    # Standard hashing (different results)
    standard_hash1 = hashlib.sha256(json.dumps(obj1).encode()).hexdigest()
    standard_hash2 = hashlib.sha256(json.dumps(obj2).encode()).hexdigest()

    print("Standard JSON hashes (DIFFERENT due to ordering):")
    print(f"  Object 1: {standard_hash1[:40]}...")
    print(f"  Object 2: {standard_hash2[:40]}...")
    print(f"  Match: {standard_hash1 == standard_hash2}")
    print()

    # Canonical hashing (same results)
    canonical_hash1 = canonical_hash(obj1)
    canonical_hash2 = canonical_hash(obj2)

    print("Canonical JSON hashes (SAME despite different ordering):")
    print(f"  Object 1: {canonical_hash1[:40]}...")
    print(f"  Object 2: {canonical_hash2[:40]}...")
    print(f"  Match: {canonical_hash1 == canonical_hash2}")
    print()

    # Canonical equality
    print("Canonical equality check:")
    print(f"  canonical_equals(obj1, obj2) = {canonical_equals(obj1, obj2)}")
    print()


def demonstrate_diff_functionality():
    """Show how to find differences between objects."""

    print("=" * 70)
    print("OBJECT DIFF DEMONSTRATION")
    print("=" * 70 + "\n")

    baseline = {
        "name": "Project Alpha",
        "emissions": {
            "scope1": 1000,
            "scope2": 2000,
            "scope3": 3000
        },
        "status": "active",
        "tags": ["carbon", "verified"]
    }

    modified = {
        "name": "Project Alpha",
        "emissions": {
            "scope1": 1000,
            "scope2": 2500,  # Changed
            "scope3": 3000,
            "scope4": 100    # Added
        },
        "status": "completed",  # Changed
        # "tags" removed
        "verified": True  # Added
    }

    print("Baseline object:")
    print(json.dumps(baseline, indent=2))
    print()

    print("Modified object:")
    print(json.dumps(modified, indent=2))
    print()

    print("Differences (using diff_canonical):")
    print("-" * 40)

    diff = diff_canonical(baseline, modified)

    if diff['added']:
        print("ADDED:")
        for key, value in diff['added'].items():
            print(f"  + {key}: {value}")
        print()

    if diff['removed']:
        print("REMOVED:")
        for key, value in diff['removed'].items():
            print(f"  - {key}: {value}")
        print()

    if diff['modified']:
        print("MODIFIED:")
        for key, change in diff['modified'].items():
            if isinstance(change, dict) and 'old' in change and 'new' in change:
                print(f"  ~ {key}: {change['old']} -> {change['new']}")
            else:
                print(f"  ~ {key}: [nested changes]")
        print()

    print(f"Objects are equal: {diff['equal']}")
    print()


def demonstrate_provenance_integration():
    """Show how canonical JSON improves provenance tracking."""

    print("=" * 70)
    print("PROVENANCE TRACKING EXAMPLE")
    print("=" * 70 + "\n")

    # Simulate calculation pipeline
    pipeline_data = {
        "input": {
            "activity_data": {
                "fuel_consumption_liters": 1000.00,
                "electricity_kwh": 5000.000,
            },
            "emission_factors": {
                "diesel": 2.68,  # kg CO2 per liter
                "electricity_grid": 0.425,  # kg CO2 per kWh
            }
        },
        "calculation": {
            "timestamp": datetime.now(timezone.utc),
            "method": "IPCC_2006",
            "scope1_emissions": 2680.0,
            "scope2_emissions": 2125.00,
            "total_emissions": 4805.000,
        },
        "metadata": {
            "calculator_version": "1.0.0",
            "data_quality_score": 0.85,
            "uncertainty_range": [4500, 5100]
        }
    }

    print("Pipeline data structure:")
    print(json.dumps(pipeline_data, indent=2, default=str)[:500] + "...")
    print()

    # Generate provenance hash
    provenance_hash = canonical_hash(pipeline_data)

    print("Provenance tracking:")
    print("-" * 40)
    print(f"SHA-256 Hash: {provenance_hash}")
    print(f"Hash prefix: {provenance_hash[:16]}")
    print()

    print("Benefits of canonical hashing for provenance:")
    print("* Deterministic: Same data always produces same hash")
    print("* Audit-ready: Can verify calculations haven't changed")
    print("* Compact: Minimal JSON reduces storage requirements")
    print("* Type-aware: Handles Decimal, datetime, UUID correctly")
    print("* Regulatory compliant: Meets requirements for data integrity")
    print()

    # Demonstrate verification
    print("Verification example:")
    print("-" * 40)

    # Slightly modify the data
    tampered_data = pipeline_data.copy()
    tampered_data["calculation"]["total_emissions"] = 4806  # Changed!

    tampered_hash = canonical_hash(tampered_data)

    print(f"Original hash:  {provenance_hash[:40]}...")
    print(f"Tampered hash:  {tampered_hash[:40]}...")
    print(f"Integrity check: {'FAILED X' if provenance_hash != tampered_hash else 'PASSED OK'}")
    print()


def run_performance_benchmark():
    """Benchmark canonical vs standard JSON performance."""

    import time

    print("=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70 + "\n")

    # Create test dataset
    test_cases = []
    for i in range(1000):
        test_cases.append({
            "id": i,
            "values": [1.000, 2.00, 3.0, 4.0000],
            "nested": {
                "z_key": i * 3,
                "a_key": i * 2,
                "m_key": i
            },
            "timestamp": datetime.now(timezone.utc),
            "decimal": Decimal("123.4500")
        })

    print(f"Test dataset: {len(test_cases)} objects")
    print()

    # Benchmark standard JSON
    start = time.time()
    for obj in test_cases:
        _ = json.dumps(obj, sort_keys=True, default=str)
    standard_time = time.time() - start

    print(f"Standard JSON serialization:")
    print(f"  Time: {standard_time:.3f} seconds")
    print(f"  Rate: {len(test_cases)/standard_time:.0f} objects/second")
    print()

    # Benchmark canonical JSON
    start = time.time()
    for obj in test_cases:
        _ = canonical_dumps(obj)
    canonical_time = time.time() - start

    print(f"Canonical JSON serialization:")
    print(f"  Time: {canonical_time:.3f} seconds")
    print(f"  Rate: {len(test_cases)/canonical_time:.0f} objects/second")
    print()

    # Benchmark hashing
    start = time.time()
    for obj in test_cases:
        _ = canonical_hash(obj)
    hash_time = time.time() - start

    print(f"Canonical hashing (with caching):")
    print(f"  Time: {hash_time:.3f} seconds")
    print(f"  Rate: {len(test_cases)/hash_time:.0f} hashes/second")
    print()

    print("Performance comparison:")
    if canonical_time < standard_time:
        print(f"  Canonical is {standard_time/canonical_time:.1f}x faster")
    else:
        print(f"  Standard is {canonical_time/standard_time:.1f}x faster")
    print()


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_canonical_vs_standard()
    demonstrate_hash_consistency()
    demonstrate_diff_functionality()
    demonstrate_provenance_integration()
    run_performance_benchmark()

    print("=" * 70)
    print("CANONICAL JSON SERIALIZATION TEST COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("* Use canonical JSON for all hash calculations")
    print("* Ensures reproducibility across different systems")
    print("* Meets regulatory requirements for data integrity")
    print("* Reduces storage with minimal representation")
    print("* Provides deterministic output for audits")