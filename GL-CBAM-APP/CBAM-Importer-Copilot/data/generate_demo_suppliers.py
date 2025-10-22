"""
Generate Synthetic Demo Suppliers for CBAM Importer Copilot

This script generates realistic supplier profiles for demonstration purposes.
The suppliers represent manufacturers/producers in non-EU countries who export CBAM-covered goods to the EU.

Usage:
    python generate_demo_suppliers.py --count 20 --output examples/demo_suppliers.yaml
"""

import argparse
import json
import random
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Set seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================================
# SUPPLIER TEMPLATES
# ============================================================================

# Supplier templates by country and product group
# Format: (company_name_template, country, country_name, product_groups, cn_codes, has_actual_emissions, actual_quality)
SUPPLIER_TEMPLATES = [
    # CHINA - Major exporter of steel, aluminum, cement
    ("Baosteel Group", "CN", "China", ["steel"], ["72031000", "72081000"], True, "high"),
    ("Angang Steel", "CN", "China", ["steel"], ["72031000", "72044900", "72011000"], True, "medium"),
    ("China National Building Material", "CN", "China", ["cement"], ["25232900", "25231000"], False, None),
    ("Chalco (Aluminum Corp of China)", "CN", "China", ["aluminum"], ["76011000", "76012000"], True, "high"),
    ("Shandong Iron & Steel", "CN", "China", ["steel"], ["72031000", "72082500"], False, None),

    # RUSSIA - Steel, aluminum, fertilizers
    ("Severstal", "RU", "Russia", ["steel"], ["72031000", "72011000"], True, "medium"),
    ("Rusal", "RU", "Russia", ["aluminum"], ["76011000", "76012000"], True, "high"),
    ("PhosAgro", "RU", "Russia", ["fertilizers"], ["28141000", "31021000"], False, None),
    ("NLMK (Novolipetsk Steel)", "RU", "Russia", ["steel"], ["72081000", "72044900"], True, "medium"),

    # INDIA - Steel, cement
    ("Tata Steel", "IN", "India", ["steel"], ["72031000", "72081000", "72011000"], True, "high"),
    ("JSW Steel", "IN", "India", ["steel"], ["72031000", "72082500"], True, "medium"),
    ("UltraTech Cement", "IN", "India", ["cement"], ["25232900", "25231000"], False, None),
    ("Hindalco Industries", "IN", "India", ["aluminum"], ["76011000", "76020000"], True, "medium"),

    # TURKEY - Steel, cement
    ("Erdemir (Ereğli Demir Çelik)", "TR", "Turkey", ["steel"], ["72031000", "72081000"], True, "medium"),
    ("Oyak Cement", "TR", "Turkey", ["cement"], ["25232900", "25239000"], False, None),
    ("Assan Alüminyum", "TR", "Turkey", ["aluminum"], ["76012000", "76041000"], False, None),

    # UKRAINE - Steel, fertilizers
    ("Metinvest", "UA", "Ukraine", ["steel"], ["72031000", "72044900"], True, "low"),
    ("OPZ (Odessa Port Plant)", "UA", "Ukraine", ["fertilizers"], ["28141000", "31023000"], False, None),

    # BOSNIA & HERZEGOVINA - Aluminum
    ("Aluminij Mostar", "BA", "Bosnia & Herzegovina", ["aluminum"], ["76011000"], False, None),

    # SERBIA - Aluminum
    ("Impol Seval", "RS", "Serbia", ["aluminum"], ["76012000", "76020000"], True, "medium"),
]


# ============================================================================
# EMISSIONS DATA GENERATION
# ============================================================================

def generate_actual_emissions_data(
    product_group: str,
    quality: str
) -> Dict[str, Any]:
    """
    Generate synthetic "actual" emissions data for suppliers who have EPDs.

    Args:
        product_group: Product group (cement, steel, aluminum, fertilizers)
        quality: Quality of data ("high", "medium", "low")

    Returns:
        Dict with emissions data
    """
    # Base emissions by product group (tCO2/ton)
    base_emissions = {
        "cement": {"direct": 0.766, "indirect": 0.134, "total": 0.900},
        "steel": {"direct": 1.850, "indirect": 0.150, "total": 2.000},
        "aluminum": {"direct": 1.700, "indirect": 9.800, "total": 11.500},
        "fertilizers": {"direct": 2.200, "indirect": 0.300, "total": 2.500},
        "hydrogen": {"direct": 10.000, "indirect": 1.000, "total": 11.000},
    }

    base = base_emissions.get(product_group, {"direct": 1.0, "indirect": 0.2, "total": 1.2})

    # Add variance based on quality
    # High quality: ±5%, Medium: ±15%, Low: ±30%
    variance_map = {
        "high": 0.05,
        "medium": 0.15,
        "low": 0.30,
    }
    variance = variance_map.get(quality, 0.15)

    # Apply random variance
    direct = base["direct"] * (1 + random.uniform(-variance, variance))
    indirect = base["indirect"] * (1 + random.uniform(-variance, variance))
    total = direct + indirect

    # Data completeness
    completeness_map = {
        "high": 95 + random.randint(0, 5),     # 95-100%
        "medium": 75 + random.randint(0, 20),  # 75-95%
        "low": 50 + random.randint(0, 25),     # 50-75%
    }
    completeness = completeness_map.get(quality, 75)

    # Certification/verification
    certifications = {
        "high": ["ISO 14064", "EPD Verified", "Third-party audited"],
        "medium": ["EPD Self-declared", "Internal audit"],
        "low": ["Estimated based on industry averages"],
    }

    return {
        "direct_emissions_tco2_per_ton": round(direct, 3),
        "indirect_emissions_tco2_per_ton": round(indirect, 3),
        "total_emissions_tco2_per_ton": round(total, 3),
        "data_quality": quality,
        "data_completeness_pct": completeness,
        "methodology": "Cradle-to-gate LCA per CBAM guidelines",
        "reporting_year": 2023 if quality == "high" else 2022,
        "certifications": certifications.get(quality, []),
        "boundary": "Cradle-to-gate (raw materials extraction through factory gate)",
        "scope_1_included": True,
        "scope_2_included": True,
        "scope_3_included": False,
        "notes": f"{quality.capitalize()} quality actual emissions data from supplier EPD",
    }


# ============================================================================
# SUPPLIER GENERATOR
# ============================================================================

def generate_suppliers(count: int = 20) -> List[Dict[str, Any]]:
    """
    Generate synthetic supplier profiles.

    Args:
        count: Number of suppliers to generate (default: 20)

    Returns:
        List of supplier dictionaries
    """
    suppliers = []

    # Use templates (cycle if count > len(templates))
    for i in range(count):
        template_idx = i % len(SUPPLIER_TEMPLATES)
        name_template, country_iso, country_name, product_groups, cn_codes, has_actual, quality = SUPPLIER_TEMPLATES[template_idx]

        # Add variation to company name if we're cycling through templates
        if i >= len(SUPPLIER_TEMPLATES):
            cycle_num = (i // len(SUPPLIER_TEMPLATES)) + 1
            company_name = f"{name_template} #{cycle_num}"
        else:
            company_name = name_template

        # Generate supplier ID
        supplier_id = f"SUP-{i+1:04d}"

        # Contact information
        company_email = f"exports@{company_name.lower().replace(' ', '').replace('#', '')}example.com"
        contact_person = f"{random.choice(['Zhang', 'Ivan', 'Rajesh', 'Mehmet', 'Viktor'])} {random.choice(['Wang', 'Petrov', 'Kumar', 'Yilmaz', 'Novak'])}"

        # Production capacity (random but realistic)
        capacity_ranges = {
            "cement": (50000, 500000),
            "steel": (100000, 1000000),
            "aluminum": (20000, 200000),
            "fertilizers": (30000, 300000),
            "hydrogen": (1000, 10000),
        }
        primary_product_group = product_groups[0]
        min_cap, max_cap = capacity_ranges.get(primary_product_group, (10000, 100000))
        production_capacity_tons_per_year = random.randint(min_cap, max_cap)

        # Build supplier record
        supplier = {
            "supplier_id": supplier_id,
            "company_name": company_name,
            "country_iso": country_iso,
            "country_name": country_name,
            "address": {
                "city": random.choice(["Beijing", "Shanghai", "Moscow", "Mumbai", "Istanbul", "Kiev"]),
                "postal_code": f"{random.randint(10000, 99999)}",
                "country": country_name,
            },
            "contact": {
                "person": contact_person,
                "email": company_email,
                "phone": f"+{random.randint(1, 99)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            },
            "product_groups": product_groups,
            "cn_codes_produced": cn_codes,
            "production_capacity_tons_per_year": production_capacity_tons_per_year,
            "certifications": [],
            "actual_emissions_available": has_actual,
        }

        # Add certifications
        if has_actual and quality == "high":
            supplier["certifications"] = ["ISO 14001", "ISO 14064", "EPD Program"]
        elif has_actual and quality == "medium":
            supplier["certifications"] = ["ISO 14001"]

        # Add actual emissions data if available
        if has_actual:
            supplier["actual_emissions_data"] = generate_actual_emissions_data(
                primary_product_group,
                quality
            )
        else:
            supplier["actual_emissions_data"] = None
            supplier["notes"] = "No actual emissions data available. EU importer will use default values."

        # Add metadata
        supplier["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        supplier["data_source"] = "Demo data - synthetic supplier profile"

        suppliers.append(supplier)

    return suppliers


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_to_yaml(suppliers: List[Dict[str, Any]], output_path: Path) -> None:
    """Save suppliers to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "count": len(suppliers),
            "disclaimer": "Synthetic data for demo purposes only",
            "version": "1.0.0-demo",
        },
        "suppliers": suppliers,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"✓ Saved {len(suppliers)} suppliers to: {output_path}")


def save_to_json(suppliers: List[Dict[str, Any]], output_path: Path) -> None:
    """Save suppliers to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "count": len(suppliers),
            "disclaimer": "Synthetic data for demo purposes only",
            "version": "1.0.0-demo",
        },
        "suppliers": suppliers,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved {len(suppliers)} suppliers to: {output_path}")


def print_summary(suppliers: List[Dict[str, Any]]) -> None:
    """Print summary statistics about generated suppliers."""
    if not suppliers:
        print("No suppliers generated!")
        return

    print("\n" + "=" * 80)
    print("SYNTHETIC SUPPLIERS SUMMARY")
    print("=" * 80)

    print(f"\nTotal Suppliers: {len(suppliers)}")

    # Count by country
    countries = {}
    for s in suppliers:
        country = s["country_iso"]
        countries[country] = countries.get(country, 0) + 1

    print(f"\nBy Country:")
    for country, count in sorted(countries.items(), key=lambda x: -x[1]):
        country_name = next((s["country_name"] for s in suppliers if s["country_iso"] == country), country)
        pct = (count / len(suppliers)) * 100
        print(f"  {country} ({country_name:20s}): {count:2d} ({pct:5.1f}%)")

    # Count by product group
    product_groups = {}
    for s in suppliers:
        for pg in s["product_groups"]:
            product_groups[pg] = product_groups.get(pg, 0) + 1

    print(f"\nBy Product Group:")
    for pg, count in sorted(product_groups.items(), key=lambda x: -x[1]):
        pct = (count / len(suppliers)) * 100
        print(f"  {pg:12s}: {count:2d} ({pct:5.1f}%)")

    # Count with actual emissions
    with_actual = sum(1 for s in suppliers if s["actual_emissions_available"])
    pct_actual = (with_actual / len(suppliers)) * 100
    print(f"\nSuppliers with Actual Emissions Data: {with_actual} ({pct_actual:.1f}%)")
    print(f"Suppliers without Actual Data (defaults): {len(suppliers) - with_actual} ({100-pct_actual:.1f}%)")

    # Quality breakdown
    if with_actual > 0:
        quality_counts = {}
        for s in suppliers:
            if s["actual_emissions_available"]:
                quality = s["actual_emissions_data"]["data_quality"]
                quality_counts[quality] = quality_counts.get(quality, 0) + 1

        print(f"\nData Quality Breakdown (for suppliers with actuals):")
        for quality in ["high", "medium", "low"]:
            count = quality_counts.get(quality, 0)
            if count > 0:
                pct = (count / with_actual) * 100
                print(f"  {quality.capitalize():6s} quality: {count} ({pct:.1f}%)")

    # Total production capacity
    total_capacity = sum(s["production_capacity_tons_per_year"] for s in suppliers)
    print(f"\nTotal Production Capacity: {total_capacity:,.0f} tonnes/year")

    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic CBAM supplier data for demo purposes"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of suppliers to generate (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/demo_suppliers.yaml",
        help="Output file path (default: examples/demo_suppliers.yaml)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["yaml", "json", "both"],
        default="yaml",
        help="Output format (default: yaml)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    print(f"Generating {args.count} synthetic suppliers...")
    print(f"Random seed: {args.seed}")

    # Generate suppliers
    suppliers = generate_suppliers(count=args.count)

    # Save outputs
    output_path = Path(args.output)

    if args.format in ["yaml", "both"]:
        save_to_yaml(suppliers, output_path)

    if args.format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        save_to_json(suppliers, json_path)

    # Print summary
    print_summary(suppliers)

    print(f"\n✓ Generation complete! Use this file as input to CBAM Importer Copilot.")
    print(f"  Example: gl cbam report --suppliers {output_path}")


if __name__ == "__main__":
    main()
