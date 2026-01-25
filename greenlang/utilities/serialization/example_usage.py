"""
Simple example demonstrating canonical vs non-canonical JSON serialization.

This example shows the key differences and benefits of using canonical JSON
for consistent hashing in the GreenLang framework.
"""

import json
import hashlib
from decimal import Decimal
from datetime import datetime, timezone
from greenlang.serialization import canonical_dumps, canonical_hash


def main():
    # Sample emissions data
    emissions_data = {
        "report_date": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "company": "Example Corp",
        "emissions": {
            "scope3": 3000.000,  # Note: different order and trailing zeros
            "scope1": 1000.00,
            "scope2": 2000.0,
        },
        "verified": True,
        "confidence_score": Decimal("0.9500"),
    }

    print("CANONICAL vs NON-CANONICAL JSON EXAMPLE")
    print("=" * 50)
    print()

    # Non-canonical JSON (standard Python json module)
    print("NON-CANONICAL JSON:")
    print("-" * 30)
    non_canonical = json.dumps(emissions_data, indent=2, default=str)
    print("Output:")
    print(non_canonical)
    print(f"\nSize: {len(non_canonical)} bytes")

    # Calculate hash of non-canonical JSON
    non_canonical_hash = hashlib.sha256(non_canonical.encode()).hexdigest()
    print(f"SHA-256: {non_canonical_hash[:40]}...")
    print()

    # Canonical JSON (deterministic, minimal)
    print("CANONICAL JSON:")
    print("-" * 30)
    canonical = canonical_dumps(emissions_data)
    print("Output:")
    print(canonical)
    print(f"\nSize: {len(canonical)} bytes")

    # Calculate canonical hash
    canonical_hash_value = canonical_hash(emissions_data)
    print(f"SHA-256: {canonical_hash_value[:40]}...")
    print()

    # Show the differences
    print("KEY DIFFERENCES:")
    print("-" * 30)
    print(f"1. Size: {len(non_canonical)} bytes -> {len(canonical)} bytes "
          f"({(1 - len(canonical)/len(non_canonical))*100:.1f}% smaller)")
    print(f"2. Whitespace: Removed in canonical")
    print(f"3. Key order: Alphabetical in canonical")
    print(f"4. Floats: Trailing zeros removed (2000.0 -> 2000)")
    print(f"5. Special types: Automatically handled (datetime, Decimal)")
    print()

    # Demonstrate hash consistency
    print("HASH CONSISTENCY TEST:")
    print("-" * 30)

    # Create same data with different key order
    reordered_data = {
        "verified": True,  # Different order
        "company": "Example Corp",
        "confidence_score": Decimal("0.95"),  # Different representation
        "emissions": {
            "scope2": 2000.000,  # Different order and trailing zeros
            "scope1": 1000,
            "scope3": 3000.0,
        },
        "report_date": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    }

    # Standard JSON produces different hashes
    standard_hash1 = hashlib.sha256(
        json.dumps(emissions_data, default=str).encode()
    ).hexdigest()
    standard_hash2 = hashlib.sha256(
        json.dumps(reordered_data, default=str).encode()
    ).hexdigest()

    print("Standard JSON hashes:")
    print(f"  Data 1: {standard_hash1[:32]}...")
    print(f"  Data 2: {standard_hash2[:32]}...")
    print(f"  Match: {standard_hash1 == standard_hash2}")
    print()

    # Canonical JSON produces same hash
    canonical_hash1 = canonical_hash(emissions_data)
    canonical_hash2 = canonical_hash(reordered_data)

    print("Canonical JSON hashes:")
    print(f"  Data 1: {canonical_hash1[:32]}...")
    print(f"  Data 2: {canonical_hash2[:32]}...")
    print(f"  Match: {canonical_hash1 == canonical_hash2}")
    print()

    print("SUMMARY:")
    print("-" * 30)
    print("Use canonical JSON for:")
    print("  - Provenance tracking (consistent hashes)")
    print("  - Data integrity verification")
    print("  - Audit trails (reproducible)")
    print("  - Cache keys (deterministic)")
    print("  - Regulatory compliance (CBAM, CSRD)")


if __name__ == "__main__":
    main()