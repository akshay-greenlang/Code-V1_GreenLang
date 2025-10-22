"""
Generate Synthetic Demo Shipments for CBAM Importer Copilot

This script generates realistic synthetic shipment data for demonstration purposes.
The data mimics actual EU import patterns but is entirely fictitious.

Usage:
    python generate_demo_shipments.py --count 500 --output examples/demo_shipments.csv
    python generate_demo_shipments.py --quarter 2025Q4 --count 1000
"""

import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Set seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Country origins with realistic import volumes (weight)
# Based on actual EU CBAM-relevant import patterns
COUNTRY_ORIGINS = [
    ("CN", "China", 0.40),          # 40% - Largest exporter to EU
    ("RU", "Russia", 0.15),         # 15% - Major steel, aluminum, fertilizers
    ("IN", "India", 0.15),          # 15% - Steel, cement
    ("TR", "Turkey", 0.10),         # 10% - Steel, cement
    ("UA", "Ukraine", 0.08),        # 8% - Steel, fertilizers
    ("RS", "Serbia", 0.04),         # 4% - Aluminum
    ("BA", "Bosnia Herzegovina", 0.03),  # 3% - Steel, aluminum
    ("EG", "Egypt", 0.02),          # 2% - Fertilizers, cement
    ("MA", "Morocco", 0.02),        # 2% - Fertilizers
    ("OTHER", "Other Countries", 0.01),  # 1% - Misc
]

# Product mix with CN codes, mass ranges, and probabilities
# Format: (cn_code, product_group, product_name, min_mass_kg, max_mass_kg, probability)
PRODUCT_MIX = [
    # STEEL (dominant in CBAM imports - 50%)
    ("72031000", "steel", "Hot-rolled flat steel", 8000, 15000, 0.20),
    ("72081000", "steel", "Hot-rolled steel coils", 10000, 18000, 0.15),
    ("72044900", "steel", "Steel scrap", 5000, 12000, 0.10),
    ("72011000", "steel", "Pig iron", 12000, 20000, 0.05),

    # CEMENT (25%)
    ("25232900", "cement", "Grey portland cement", 15000, 25000, 0.15),
    ("25231000", "cement", "Cement clinker", 18000, 28000, 0.08),
    ("25232100", "cement", "White cement", 8000, 15000, 0.02),

    # ALUMINUM (15%)
    ("76011000", "aluminum", "Unwrought aluminum (primary)", 3000, 8000, 0.08),
    ("76012000", "aluminum", "Aluminum alloys unwrought", 2500, 7000, 0.05),
    ("76020000", "aluminum", "Aluminum scrap", 4000, 10000, 0.02),

    # FERTILIZERS (8%)
    ("28141000", "fertilizers", "Anhydrous ammonia", 10000, 20000, 0.03),
    ("31021000", "fertilizers", "Urea", 12000, 22000, 0.03),
    ("31023000", "fertilizers", "Ammonium nitrate", 8000, 16000, 0.02),

    # HYDROGEN (2%)
    ("28041000", "hydrogen", "Hydrogen", 500, 2000, 0.02),
]

# Importer companies (EU companies receiving goods)
EU_IMPORTERS = [
    ("Acme Steel EU BV", "NL"),
    ("EuroSteel GmbH", "DE"),
    ("Mediterranean Cement SA", "FR"),
    ("Nordic Aluminum AB", "SE"),
    ("Iberia Fertilizers SL", "ES"),
    ("Baltic Metals OÜ", "EE"),
    ("Adriatic Industries Srl", "IT"),
    ("Danube Steel AG", "AT"),
    ("Celtic Materials Ltd", "IE"),
    ("Balkan Cement DOO", "HR"),
]

# Supplier IDs (will be detailed in suppliers generator)
SUPPLIER_IDS = [
    "SUP-0001", "SUP-0002", "SUP-0003", "SUP-0004", "SUP-0005",
    "SUP-0006", "SUP-0007", "SUP-0008", "SUP-0009", "SUP-0010",
    "SUP-0011", "SUP-0012", "SUP-0013", "SUP-0014", "SUP-0015",
    "SUP-0016", "SUP-0017", "SUP-0018", "SUP-0019", "SUP-0020",
]


# ============================================================================
# SHIPMENT GENERATOR
# ============================================================================

def generate_shipments(
    count: int = 500,
    quarter: str = "2025Q4",
    actual_data_ratio: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic shipment data.

    Args:
        count: Number of shipments to generate
        quarter: Quarter identifier (e.g., "2025Q4")
        actual_data_ratio: Percentage of shipments with "actual" emissions data (vs defaults)

    Returns:
        List of shipment dictionaries
    """
    shipments = []

    # Parse quarter
    year, q = quarter.split("Q")
    year = int(year)
    quarter_num = int(q)

    # Determine date range
    if quarter_num == 1:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 3, 31)
    elif quarter_num == 2:
        start_date = datetime(year, 4, 1)
        end_date = datetime(year, 6, 30)
    elif quarter_num == 3:
        start_date = datetime(year, 7, 1)
        end_date = datetime(year, 9, 30)
    else:  # Q4
        start_date = datetime(year, 10, 1)
        end_date = datetime(year, 12, 31)

    days_in_quarter = (end_date - start_date).days

    # Extract weights for random selection
    origin_countries = [x[0] for x in COUNTRY_ORIGINS]
    origin_names = [x[1] for x in COUNTRY_ORIGINS]
    origin_weights = [x[2] for x in COUNTRY_ORIGINS]

    product_data = [(x[0], x[1], x[2], x[3], x[4]) for x in PRODUCT_MIX]
    product_weights = [x[5] for x in PRODUCT_MIX]

    for i in range(count):
        # Select origin country (weighted random)
        origin_idx = random.choices(range(len(origin_countries)), weights=origin_weights)[0]
        origin_iso = origin_countries[origin_idx]
        origin_name = origin_names[origin_idx]

        # Select product (weighted random)
        product_idx = random.choices(range(len(product_data)), weights=product_weights)[0]
        cn_code, product_group, product_name, min_mass, max_mass = product_data[product_idx]

        # Generate mass
        net_mass_kg = random.randint(min_mass, max_mass)

        # Generate import date
        random_days = random.randint(0, days_in_quarter)
        import_date = start_date + timedelta(days=random_days)

        # Select importer
        importer_name, importer_country = random.choice(EU_IMPORTERS)

        # Determine if shipment has actual emissions data
        has_actual_data = random.random() < actual_data_ratio
        supplier_id = random.choice(SUPPLIER_IDS) if has_actual_data else None

        # Generate shipment ID
        shipment_id = f"DEMO-{quarter}-{i+1:05d}"

        # Importer reference (realistic customs reference)
        importer_ref = f"IMP-{year}-{random.randint(100000, 999999)}"

        # Port of entry (common EU ports)
        ports = ["Rotterdam", "Hamburg", "Antwerp", "Le Havre", "Barcelona", "Piraeus", "Genoa"]
        port_of_entry = random.choice(ports)

        # Create shipment record
        shipment = {
            "shipment_id": shipment_id,
            "import_date": import_date.strftime("%Y-%m-%d"),
            "quarter": quarter,
            "cn_code": cn_code,
            "product_group": product_group,
            "product_description": product_name,
            "origin_iso": origin_iso,
            "origin_country": origin_name,
            "importer_name": importer_name,
            "importer_country": importer_country,
            "net_mass_kg": net_mass_kg,
            "port_of_entry": port_of_entry,
            "importer_reference": importer_ref,
            "has_actual_emissions": "YES" if has_actual_data else "NO",
            "supplier_id": supplier_id if has_actual_data else "",
            "notes": "Demo data - synthetic" if not has_actual_data else "Demo data - with supplier actuals",
        }

        shipments.append(shipment)

    return shipments


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_to_csv(shipments: List[Dict[str, Any]], output_path: Path) -> None:
    """Save shipments to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not shipments:
        print("No shipments to save!")
        return

    fieldnames = list(shipments[0].keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shipments)

    print(f"✓ Saved {len(shipments)} shipments to: {output_path}")


def save_to_json(shipments: List[Dict[str, Any]], output_path: Path) -> None:
    """Save shipments to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "count": len(shipments),
                "quarter": shipments[0]["quarter"] if shipments else "N/A",
                "disclaimer": "Synthetic data for demo purposes only",
            },
            "shipments": shipments,
        }, f, indent=2)

    print(f"✓ Saved {len(shipments)} shipments to: {output_path}")


def print_summary(shipments: List[Dict[str, Any]]) -> None:
    """Print summary statistics about generated shipments."""
    if not shipments:
        print("No shipments generated!")
        return

    print("\n" + "=" * 80)
    print("SYNTHETIC SHIPMENTS SUMMARY")
    print("=" * 80)

    # Count by product group
    product_groups = {}
    for s in shipments:
        pg = s["product_group"]
        product_groups[pg] = product_groups.get(pg, 0) + 1

    print(f"\nTotal Shipments: {len(shipments)}")
    print(f"Quarter: {shipments[0]['quarter']}")
    print(f"\nBy Product Group:")
    for pg, count in sorted(product_groups.items(), key=lambda x: -x[1]):
        pct = (count / len(shipments)) * 100
        print(f"  {pg:12s}: {count:4d} ({pct:5.1f}%)")

    # Count by origin
    origins = {}
    for s in shipments:
        origin = s["origin_iso"]
        origins[origin] = origins.get(origin, 0) + 1

    print(f"\nTop 5 Origin Countries:")
    for origin, count in sorted(origins.items(), key=lambda x: -x[1])[:5]:
        pct = (count / len(shipments)) * 100
        origin_name = next((x[1] for x in COUNTRY_ORIGINS if x[0] == origin), origin)
        print(f"  {origin} ({origin_name:20s}): {count:4d} ({pct:5.1f}%)")

    # Count with actual data
    with_actual = sum(1 for s in shipments if s["has_actual_emissions"] == "YES")
    pct_actual = (with_actual / len(shipments)) * 100
    print(f"\nShipments with Actual Emissions Data: {with_actual} ({pct_actual:.1f}%)")
    print(f"Shipments using Default Values: {len(shipments) - with_actual} ({100-pct_actual:.1f}%)")

    # Total mass
    total_mass_tonnes = sum(s["net_mass_kg"] for s in shipments) / 1000
    print(f"\nTotal Mass: {total_mass_tonnes:,.0f} tonnes")

    # Date range
    dates = [datetime.strptime(s["import_date"], "%Y-%m-%d") for s in shipments]
    print(f"Date Range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")

    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic CBAM shipment data for demo purposes"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of shipments to generate (default: 500)"
    )
    parser.add_argument(
        "--quarter",
        type=str,
        default="2025Q4",
        help="Quarter identifier (default: 2025Q4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/demo_shipments.csv",
        help="Output file path (default: examples/demo_shipments.csv)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "both"],
        default="csv",
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--actual-ratio",
        type=float,
        default=0.20,
        help="Ratio of shipments with actual emissions data (default: 0.20)"
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

    print(f"Generating {args.count} synthetic shipments for {args.quarter}...")
    print(f"Actual emissions data ratio: {args.actual_ratio:.0%}")
    print(f"Random seed: {args.seed}")

    # Generate shipments
    shipments = generate_shipments(
        count=args.count,
        quarter=args.quarter,
        actual_data_ratio=args.actual_ratio,
    )

    # Save outputs
    output_path = Path(args.output)

    if args.format in ["csv", "both"]:
        save_to_csv(shipments, output_path)

    if args.format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        save_to_json(shipments, json_path)

    # Print summary
    print_summary(shipments)

    print(f"\n✓ Generation complete! Use this file as input to CBAM Importer Copilot.")
    print(f"  Example: gl cbam report --inputs {output_path} --quarter {args.quarter}")


if __name__ == "__main__":
    main()
