# -*- coding: utf-8 -*-
"""
CSRD Sample Data Generator

This script generates realistic ESG test data:
- Generates CSV, JSON, and Excel formats
- Supports all ESRS standards (E1-E5, S1-S4, G1)
- Creates company profiles
- Generates materiality assessments
- Configurable data sizes (10, 100, 1000, 10000 metrics)
- Realistic value ranges and units
- Proper date formats

Usage:
    python scripts/generate_sample_data.py --size 100 --format csv
    python scripts/generate_sample_data.py --size 1000 --format all --output data/sample
    python scripts/generate_sample_data.py --esrs-standards E1 E3 S1 --format json
    python scripts/generate_sample_data.py --generate-company-profile

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from greenlang.determinism import deterministic_random
from greenlang.determinism import deterministic_uuid, DeterministicClock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

console = Console()


# ============================================================================
# DATA GENERATION CONFIGURATION
# ============================================================================

# ESRS metric templates with realistic ranges
ESRS_METRICS = {
    # Environmental - Climate Change (E1)
    "E1": [
        {"code": "E1-1", "name": "Scope 1 GHG Emissions", "unit": "tCO2e", "range": (1000, 100000), "type": "float"},
        {"code": "E1-2", "name": "Scope 2 GHG Emissions (location-based)", "unit": "tCO2e", "range": (500, 80000), "type": "float"},
        {"code": "E1-3", "name": "Scope 2 GHG Emissions (market-based)", "unit": "tCO2e", "range": (300, 60000), "type": "float"},
        {"code": "E1-4", "name": "Scope 3 GHG Emissions", "unit": "tCO2e", "range": (5000, 500000), "type": "float"},
        {"code": "E1-5", "name": "Total GHG Emissions", "unit": "tCO2e", "range": (10000, 700000), "type": "float"},
        {"code": "E1-6", "name": "Total Energy Consumption", "unit": "GJ", "range": (50000, 1000000), "type": "float"},
        {"code": "E1-7", "name": "Renewable Energy Consumption", "unit": "GJ", "range": (5000, 300000), "type": "float"},
        {"code": "E1-8", "name": "Renewable Energy Percentage", "unit": "%", "range": (5, 80), "type": "float"},
        {"code": "E1-9", "name": "Energy Intensity", "unit": "GJ/revenue", "range": (0.01, 0.5), "type": "float"},
    ],
    # Environmental - Pollution (E2)
    "E2": [
        {"code": "E2-1", "name": "Air Emissions - NOx", "unit": "tonnes", "range": (1, 100), "type": "float"},
        {"code": "E2-2", "name": "Air Emissions - SOx", "unit": "tonnes", "range": (0.5, 50), "type": "float"},
        {"code": "E2-3", "name": "Air Emissions - Particulate Matter", "unit": "tonnes", "range": (0.5, 80), "type": "float"},
        {"code": "E2-4", "name": "Substances of Concern Released", "unit": "tonnes", "range": (0, 10), "type": "float"},
        {"code": "E2-5", "name": "Substances of Very High Concern", "unit": "tonnes", "range": (0, 5), "type": "float"},
    ],
    # Environmental - Water (E3)
    "E3": [
        {"code": "E3-1", "name": "Total Water Withdrawal", "unit": "m3", "range": (10000, 1000000), "type": "float"},
        {"code": "E3-2", "name": "Water Consumption", "unit": "m3", "range": (8000, 800000), "type": "float"},
        {"code": "E3-3", "name": "Water Discharge", "unit": "m3", "range": (2000, 200000), "type": "float"},
        {"code": "E3-4", "name": "Operations in Water-Stressed Areas", "unit": "%", "range": (0, 50), "type": "float"},
        {"code": "E3-5", "name": "Water Intensity", "unit": "m3/revenue", "range": (0.01, 5), "type": "float"},
    ],
    # Environmental - Biodiversity (E4)
    "E4": [
        {"code": "E4-1", "name": "Impact on Biodiversity-Sensitive Areas", "unit": "count", "range": (0, 5), "type": "int"},
        {"code": "E4-2", "name": "Land Degradation", "unit": "hectares", "range": (0, 100), "type": "float"},
        {"code": "E4-3", "name": "Habitat Restoration", "unit": "hectares", "range": (0, 50), "type": "float"},
    ],
    # Environmental - Circular Economy (E5)
    "E5": [
        {"code": "E5-1", "name": "Total Waste Generated", "unit": "tonnes", "range": (500, 50000), "type": "float"},
        {"code": "E5-2", "name": "Waste Diverted from Disposal", "unit": "tonnes", "range": (300, 40000), "type": "float"},
        {"code": "E5-3", "name": "Waste Diversion Rate", "unit": "%", "range": (30, 90), "type": "float"},
        {"code": "E5-4", "name": "Hazardous Waste", "unit": "tonnes", "range": (10, 5000), "type": "float"},
        {"code": "E5-5", "name": "Recycled Content in Products", "unit": "%", "range": (5, 70), "type": "float"},
        {"code": "E5-6", "name": "Material Footprint", "unit": "tonnes", "range": (1000, 100000), "type": "float"},
    ],
    # Social - Own Workforce (S1)
    "S1": [
        {"code": "S1-1", "name": "Total Employees", "unit": "FTE", "range": (50, 50000), "type": "int"},
        {"code": "S1-2", "name": "Female Employees", "unit": "FTE", "range": (20, 25000), "type": "int"},
        {"code": "S1-3", "name": "Male Employees", "unit": "FTE", "range": (30, 25000), "type": "int"},
        {"code": "S1-4", "name": "Non-Binary Employees", "unit": "FTE", "range": (0, 100), "type": "int"},
        {"code": "S1-5", "name": "Employees Under 30", "unit": "FTE", "range": (10, 10000), "type": "int"},
        {"code": "S1-6", "name": "Employees 30-50", "unit": "FTE", "range": (30, 30000), "type": "int"},
        {"code": "S1-7", "name": "Employees Over 50", "unit": "FTE", "range": (10, 10000), "type": "int"},
        {"code": "S1-8", "name": "Employee Turnover Rate", "unit": "%", "range": (5, 25), "type": "float"},
        {"code": "S1-9", "name": "Average Training Hours per Employee", "unit": "hours", "range": (10, 80), "type": "float"},
        {"code": "S1-10", "name": "Work-Related Injuries", "unit": "count", "range": (0, 100), "type": "int"},
        {"code": "S1-11", "name": "Work-Related Fatalities", "unit": "count", "range": (0, 3), "type": "int"},
        {"code": "S1-12", "name": "Lost Time Injury Frequency Rate", "unit": "per million hours", "range": (0.5, 10), "type": "float"},
        {"code": "S1-13", "name": "Average Annual Salary", "unit": "EUR", "range": (30000, 100000), "type": "float"},
        {"code": "S1-14", "name": "Gender Pay Gap", "unit": "%", "range": (0, 20), "type": "float"},
    ],
    # Social - Workers in Value Chain (S2)
    "S2": [
        {"code": "S2-1", "name": "Supplier Audits Conducted", "unit": "count", "range": (10, 200), "type": "int"},
        {"code": "S2-2", "name": "Suppliers with Corrective Actions", "unit": "count", "range": (5, 100), "type": "int"},
        {"code": "S2-3", "name": "Suppliers Terminated for Non-Compliance", "unit": "count", "range": (0, 10), "type": "int"},
        {"code": "S2-4", "name": "Workers in Value Chain Covered by Living Wage", "unit": "%", "range": (30, 100), "type": "float"},
    ],
    # Social - Affected Communities (S3)
    "S3": [
        {"code": "S3-1", "name": "Community Investment", "unit": "EUR", "range": (10000, 5000000), "type": "float"},
        {"code": "S3-2", "name": "Local Employment Percentage", "unit": "%", "range": (40, 95), "type": "float"},
        {"code": "S3-3", "name": "Community Grievances Received", "unit": "count", "range": (0, 50), "type": "int"},
        {"code": "S3-4", "name": "Community Grievances Resolved", "unit": "count", "range": (0, 50), "type": "int"},
    ],
    # Social - Consumers & End-Users (S4)
    "S4": [
        {"code": "S4-1", "name": "Product Safety Incidents", "unit": "count", "range": (0, 20), "type": "int"},
        {"code": "S4-2", "name": "Customer Data Breaches", "unit": "count", "range": (0, 5), "type": "int"},
        {"code": "S4-3", "name": "Customer Satisfaction Score", "unit": "score (1-5)", "range": (3.0, 5.0), "type": "float"},
        {"code": "S4-4", "name": "Customer Complaints", "unit": "count", "range": (10, 1000), "type": "int"},
    ],
    # Governance (G1)
    "G1": [
        {"code": "G1-1", "name": "Board Members", "unit": "count", "range": (5, 15), "type": "int"},
        {"code": "G1-2", "name": "Female Board Members", "unit": "count", "range": (2, 8), "type": "int"},
        {"code": "G1-3", "name": "Independent Board Members", "unit": "count", "range": (3, 12), "type": "int"},
        {"code": "G1-4", "name": "Anti-Corruption Training Completion Rate", "unit": "%", "range": (80, 100), "type": "float"},
        {"code": "G1-5", "name": "Whistleblower Reports Received", "unit": "count", "range": (0, 20), "type": "int"},
        {"code": "G1-6", "name": "Confirmed Corruption Incidents", "unit": "count", "range": (0, 3), "type": "int"},
        {"code": "G1-7", "name": "Suppliers Signed Code of Conduct", "unit": "count", "range": (50, 500), "type": "int"},
        {"code": "G1-8", "name": "Board Meeting Attendance Rate", "unit": "%", "range": (85, 100), "type": "float"},
    ]
}

DATA_QUALITY_OPTIONS = ["high", "medium", "low"]
DATA_QUALITY_WEIGHTS = [0.60, 0.30, 0.10]  # 60% high, 30% medium, 10% low

VERIFICATION_STATUS_OPTIONS = ["verified", "third_party_assured", "unverified"]
VERIFICATION_WEIGHTS = [0.50, 0.30, 0.20]

SOURCE_DOCUMENTS = [
    "SAP ERP System",
    "Energy Management System",
    "HRIS (Workday)",
    "Utility Bills",
    "Environmental Monitoring Report",
    "Health & Safety System",
    "Supplier Audits",
    "Finance Department",
    "Quality Management System",
    "IT Security Logs"
]


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_value(metric_config: Dict[str, Any]) -> Any:
    """Generate a realistic value for a metric."""
    min_val, max_val = metric_config["range"]

    if metric_config["type"] == "int":
        return deterministic_random().randint(int(min_val), int(max_val))
    elif metric_config["type"] == "float":
        value = random.uniform(min_val, max_val)
        # Round to 1-2 decimal places
        return round(value, deterministic_random().choice([1, 2]))
    else:
        return random.uniform(min_val, max_val)


def generate_date_range(year: int = 2024) -> Tuple[str, str]:
    """Generate a reporting period date range."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    return start_date, end_date


def generate_esg_data(
    size: int,
    esrs_standards: Optional[List[str]] = None,
    year: int = 2024
) -> List[Dict[str, Any]]:
    """
    Generate synthetic ESG data points.

    Args:
        size: Number of data points to generate
        esrs_standards: List of ESRS standards to include (e.g., ["E1", "S1"])
        year: Reporting year

    Returns:
        List of ESG data point dictionaries
    """
    console.print(f"[cyan]Generating {size} ESG data points...[/cyan]")

    # Filter metrics by requested standards
    if esrs_standards:
        metrics_pool = []
        for standard in esrs_standards:
            if standard in ESRS_METRICS:
                metrics_pool.extend(ESRS_METRICS[standard])
    else:
        # Use all metrics
        metrics_pool = []
        for metrics in ESRS_METRICS.values():
            metrics_pool.extend(metrics)

    if not metrics_pool:
        raise ValueError("No metrics available for selected ESRS standards")

    data_points = []
    period_start, period_end = generate_date_range(year)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating data points...", total=size)

        for i in range(size):
            # Select metric (cycle through or random)
            metric = metrics_pool[i % len(metrics_pool)]

            # Generate value
            value = generate_value(metric)

            # Create data point
            data_point = {
                "metric_code": metric["code"],
                "metric_name": metric["name"],
                "value": value,
                "unit": metric["unit"],
                "period_start": period_start,
                "period_end": period_end,
                "data_quality": random.choices(DATA_QUALITY_OPTIONS, weights=DATA_QUALITY_WEIGHTS)[0],
                "source_document": deterministic_random().choice(SOURCE_DOCUMENTS),
                "verification_status": random.choices(VERIFICATION_STATUS_OPTIONS, weights=VERIFICATION_WEIGHTS)[0],
                "notes": f"Generated data point {i + 1}"
            }

            data_points.append(data_point)
            progress.update(task, advance=1)

    console.print(f"[green]✓ Generated {len(data_points)} data points[/green]")
    return data_points


def generate_company_profile(company_name: str = "Sample Corporation") -> Dict[str, Any]:
    """Generate a realistic company profile."""

    console.print(f"[cyan]Generating company profile for '{company_name}'...[/cyan]")

    import uuid

    profile = {
        "company_id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "legal_name": f"{company_name} B.V.",
        "commercial_name": company_name,
        "lei_code": f"549300{''.join([str(deterministic_random().randint(0, 9)) for _ in range(12)])}",
        "country": deterministic_random().choice(["NL", "DE", "FR", "ES", "IT", "BE"]),
        "registered_address": {
            "street": f"{deterministic_random().choice(['Main', 'Industrial', 'Technology', 'Business'])} Street {deterministic_random().randint(1, 200)}",
            "city": deterministic_random().choice(["Amsterdam", "Berlin", "Paris", "Madrid", "Brussels"]),
            "postal_code": f"{deterministic_random().randint(1000, 9999)} AB",
            "country": "NL"
        },
        "sector": {
            "nace_code": deterministic_random().choice(["25.11", "26.51", "35.11", "20.14"]),
            "nace_description": "Manufacturing",
            "industry": "Manufacturing",
            "sub_industry": deterministic_random().choice(["Industrial Equipment", "Electronics", "Chemicals", "Food & Beverage"])
        },
        "company_size": {
            "employee_count": deterministic_random().randint(500, 5000),
            "employee_count_fte": deterministic_random().randint(500, 5000),
            "size_category": deterministic_random().choice(["Large", "Medium"]),
            "revenue_eur": deterministic_random().randint(100000000, 1000000000),
            "total_assets_eur": deterministic_random().randint(80000000, 800000000)
        },
        "reporting_period": {
            "fiscal_year": 2024,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "reporting_framework": "CSRD",
            "first_year_csrd": deterministic_random().choice([True, False])
        },
        "csrd_applicability": {
            "csrd_applicable": True,
            "phase": deterministic_random().choice(["Phase 1", "Phase 2"]),
            "reason": "Large company, >500 employees",
            "first_report_due": "2025-04-30"
        },
        "stock_listing": {
            "listed": deterministic_random().choice([True, False]),
            "exchange": "Euronext Amsterdam",
            "ticker": f"{company_name[:4].upper()}.AS",
            "isin": f"NL{''.join([str(deterministic_random().randint(0, 9)) for _ in range(10)])}"
        },
        "sustainability_governance": {
            "sustainability_committee": True,
            "sustainability_officer": {
                "name": "Jane Smith",
                "title": "Chief Sustainability Officer",
                "email": f"sustainability@{company_name.lower().replace(' ', '')}.com"
            },
            "board_oversight": True,
            "esrs_working_group": True
        },
        "previous_reporting": {
            "nfrd_reporting": deterministic_random().choice([True, False]),
            "gri_reporting": deterministic_random().choice([True, False]),
            "tcfd_reporting": deterministic_random().choice([True, False]),
            "cdp_reporting": deterministic_random().choice([True, False])
        },
        "metadata": {
            "created_date": DeterministicClock.now().isoformat(),
            "version": "1.0",
            "data_quality_score": round(random.uniform(0.85, 0.98), 2)
        }
    }

    console.print(f"[green]✓ Company profile generated[/green]")
    return profile


def generate_materiality_assessment() -> Dict[str, Any]:
    """Generate a sample materiality assessment."""

    console.print("[cyan]Generating materiality assessment...[/cyan]")

    # Sample material topics
    material_topics = [
        {
            "topic_id": "E1",
            "topic_name": "Climate Change",
            "impact_materiality_score": round(random.uniform(3.5, 5.0), 1),
            "financial_materiality_score": round(random.uniform(3.5, 5.0), 1),
            "is_material": True
        },
        {
            "topic_id": "E3",
            "topic_name": "Water and Marine Resources",
            "impact_materiality_score": round(random.uniform(2.5, 4.5), 1),
            "financial_materiality_score": round(random.uniform(2.0, 4.0), 1),
            "is_material": deterministic_random().choice([True, False])
        },
        {
            "topic_id": "S1",
            "topic_name": "Own Workforce",
            "impact_materiality_score": round(random.uniform(3.0, 5.0), 1),
            "financial_materiality_score": round(random.uniform(3.0, 5.0), 1),
            "is_material": True
        },
        {
            "topic_id": "G1",
            "topic_name": "Business Conduct",
            "impact_materiality_score": round(random.uniform(3.5, 5.0), 1),
            "financial_materiality_score": round(random.uniform(3.0, 5.0), 1),
            "is_material": True
        }
    ]

    assessment = {
        "assessment_id": f"mat_assess_{int(DeterministicClock.now().timestamp())}",
        "assessment_date": DeterministicClock.now().isoformat(),
        "methodology": "ESRS 1 Double Materiality Assessment",
        "stakeholders_consulted": [
            "Employees",
            "Investors",
            "Customers",
            "Suppliers",
            "Local communities"
        ],
        "material_topics": material_topics,
        "next_assessment_date": (DeterministicClock.now() + timedelta(days=1095)).isoformat()  # 3 years
    }

    console.print(f"[green]✓ Materiality assessment generated with {len(material_topics)} topics[/green]")
    return assessment


# ============================================================================
# FILE EXPORT FUNCTIONS
# ============================================================================

def save_csv(data: List[Dict[str, Any]], output_path: Path):
    """Save data to CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    console.print(f"[green]✓ CSV saved to {output_path}[/green]")


def save_json(data: Any, output_path: Path):
    """Save data to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"[green]✓ JSON saved to {output_path}[/green]")


def save_excel(data: List[Dict[str, Any]], output_path: Path):
    """Save data to Excel file."""
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False, engine='openpyxl')
    console.print(f"[green]✓ Excel saved to {output_path}[/green]")


# ============================================================================
# CLI INTERFACE
# ============================================================================

@click.command()
@click.option(
    '--size',
    type=int,
    default=100,
    help='Number of data points to generate'
)
@click.option(
    '--format',
    'output_format',
    type=click.Choice(['csv', 'json', 'excel', 'all']),
    default='csv',
    help='Output format'
)
@click.option(
    '--output',
    type=click.Path(),
    default='sample_esg_data',
    help='Output file path (without extension)'
)
@click.option(
    '--esrs-standards',
    multiple=True,
    type=click.Choice(['E1', 'E2', 'E3', 'E4', 'E5', 'S1', 'S2', 'S3', 'S4', 'G1']),
    help='ESRS standards to include (can specify multiple)'
)
@click.option(
    '--year',
    type=int,
    default=2024,
    help='Reporting year'
)
@click.option(
    '--generate-company-profile',
    is_flag=True,
    help='Also generate a company profile'
)
@click.option(
    '--generate-materiality',
    is_flag=True,
    help='Also generate a materiality assessment'
)
@click.option(
    '--company-name',
    type=str,
    default='Sample Corporation',
    help='Company name for profile generation'
)
def generate_sample_data(
    size: int,
    output_format: str,
    output: str,
    esrs_standards: Tuple[str],
    year: int,
    generate_company_profile: bool,
    generate_materiality: bool,
    company_name: str
):
    """
    Generate realistic ESG sample data for testing.

    Creates CSV, JSON, or Excel files with synthetic ESG metrics
    following ESRS standards. Optionally generates company profiles
    and materiality assessments.
    """
    console.print("\n[bold cyan]📊 CSRD Sample Data Generator[/bold cyan]\n")

    output_path = Path(output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert tuple to list
    standards_list = list(esrs_standards) if esrs_standards else None

    if standards_list:
        console.print(f"[cyan]ESRS Standards: {', '.join(standards_list)}[/cyan]")
    else:
        console.print(f"[cyan]ESRS Standards: All[/cyan]")

    console.print(f"[cyan]Size: {size} data points[/cyan]")
    console.print(f"[cyan]Year: {year}[/cyan]\n")

    # Generate ESG data
    esg_data = generate_esg_data(size, standards_list, year)

    console.print()

    # Save in requested formats
    if output_format in ['csv', 'all']:
        csv_path = output_path.with_suffix('.csv')
        save_csv(esg_data, csv_path)

    if output_format in ['json', 'all']:
        json_path = output_path.with_suffix('.json')
        save_json({"data_points": esg_data}, json_path)

    if output_format in ['excel', 'all']:
        excel_path = output_path.with_suffix('.xlsx')
        save_excel(esg_data, excel_path)

    console.print()

    # Generate company profile if requested
    if generate_company_profile:
        profile = generate_company_profile(company_name)
        profile_path = output_dir / f"{company_name.lower().replace(' ', '_')}_profile.json"
        save_json(profile, profile_path)
        console.print()

    # Generate materiality assessment if requested
    if generate_materiality:
        materiality = generate_materiality_assessment()
        materiality_path = output_dir / "materiality_assessment.json"
        save_json(materiality, materiality_path)
        console.print()

    console.print("[bold green]✅ Sample data generation complete![/bold green]\n")

    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  - ESG data points: {size}")
    if standards_list:
        console.print(f"  - ESRS standards: {', '.join(standards_list)}")
    else:
        console.print(f"  - ESRS standards: All ({len(ESRS_METRICS)} standards)")
    console.print(f"  - Formats: {output_format}")
    if generate_company_profile:
        console.print(f"  - Company profile: ✓")
    if generate_materiality:
        console.print(f"  - Materiality assessment: ✓")
    console.print()


if __name__ == '__main__':
    generate_sample_data()
