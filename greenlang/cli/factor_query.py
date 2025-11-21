# -*- coding: utf-8 -*-
"""

logger = logging.getLogger(__name__)
Emission Factor CLI Tool

Command-line interface for querying and calculating with emission factors.

Usage:
    greenlang factors list
    greenlang factors search "diesel"
    greenlang factors get FACTOR_ID
    greenlang factors calculate --factor=FACTOR_ID --amount=100 --unit=gallons
    greenlang factors stats
    greenlang factors validate-db
"""

import logging
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from tabulate import tabulate

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from greenlang.sdk.emission_factor_client import (
    EmissionFactorClient,
    EmissionFactorNotFoundError,
    UnitNotAvailableError,
    DatabaseConnectionError
)
from greenlang.models.emission_factor import FactorSearchCriteria
from greenlang.db.emission_factors_schema import validate_database, get_database_info


def format_factor_summary(factor) -> dict:
    """Format factor for summary display."""
    return {
        'ID': factor.factor_id,
        'Name': factor.name,
        'Category': factor.category,
        'Value': f"{factor.emission_factor_kg_co2e:.4f}",
        'Unit': factor.unit,
        'Scope': factor.scope,
        'Geography': factor.geography.geographic_scope,
        'Updated': str(factor.last_updated)
    }


def cmd_list(args):
    """List all emission factors."""
    try:
        client = EmissionFactorClient(db_path=args.db_path)

        # Get filters
        category = args.category if hasattr(args, 'category') else None
        scope = args.scope if hasattr(args, 'scope') else None

        # Build criteria
        criteria = FactorSearchCriteria(
            category=category,
            scope=scope
        )

        factors = client.search_factors(criteria)

        if not factors:
            print("No factors found")
            return

        # Format for display
        rows = []
        for factor in factors:
            rows.append([
                factor.factor_id[:40],
                factor.name[:50],
                factor.category,
                f"{factor.emission_factor_kg_co2e:.4f}",
                factor.unit,
                factor.scope or 'N/A'
            ])

        headers = ['Factor ID', 'Name', 'Category', 'Value', 'Unit', 'Scope']
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print(f"\nTotal: {len(factors)} factors")

        client.close()

    except Exception as e:
        logger.error(f" {e}", file=sys.stderr)
        sys.exit(1)


def cmd_search(args):
    """Search for emission factors."""
    try:
        client = EmissionFactorClient(db_path=args.db_path)

        # Search by name
        factors = client.get_factor_by_name(args.query)

        if not factors:
            print(f"No factors found matching '{args.query}'")
            return

        # Format for display
        rows = []
        for factor in factors:
            rows.append([
                factor.factor_id[:40],
                factor.name[:50],
                factor.category,
                f"{factor.emission_factor_kg_co2e:.6f}",
                factor.unit,
                factor.geography.geographic_scope[:30]
            ])

        headers = ['Factor ID', 'Name', 'Category', 'Value', 'Unit', 'Geography']
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print(f"\nFound: {len(factors)} factors")

        client.close()

    except Exception as e:
        logger.error(f" {e}", file=sys.stderr)
        sys.exit(1)


def cmd_get(args):
    """Get detailed information about a factor."""
    try:
        client = EmissionFactorClient(db_path=args.db_path)

        factor = client.get_factor(args.factor_id)

        # Display detailed information
        print("=" * 70)
        print(f"EMISSION FACTOR: {factor.factor_id}")
        print("=" * 70)
        print(f"Name:           {factor.name}")
        print(f"Category:       {factor.category}")
        if factor.subcategory:
            print(f"Subcategory:    {factor.subcategory}")
        print()

        print("EMISSION FACTOR:")
        print(f"  Value:        {factor.emission_factor_kg_co2e:.6f} kg CO2e/{factor.unit}")
        print(f"  Unit:         {factor.unit}")
        print()

        if factor.additional_units:
            print("ADDITIONAL UNITS:")
            for unit in factor.additional_units:
                print(f"  {unit.emission_factor_value:.6f} kg CO2e/{unit.unit_name}")
            print()

        print("GHG SCOPE:")
        print(f"  {factor.scope}")
        print()

        print("GEOGRAPHY:")
        print(f"  Scope:        {factor.geography.geographic_scope}")
        print(f"  Level:        {factor.geography.geography_level.value}")
        if factor.geography.country_code:
            print(f"  Country:      {factor.geography.country_code}")
        if factor.geography.state_province:
            print(f"  State:        {factor.geography.state_province}")
        print()

        print("SOURCE PROVENANCE:")
        print(f"  Organization: {factor.source.source_org}")
        if factor.source.source_publication:
            print(f"  Publication:  {factor.source.source_publication}")
        print(f"  URI:          {factor.source.source_uri}")
        if factor.source.standard:
            print(f"  Standard:     {factor.source.standard}")
        print()

        print("DATA QUALITY:")
        print(f"  Tier:         {factor.data_quality.tier.value}")
        if factor.data_quality.uncertainty_percent:
            print(f"  Uncertainty:  ±{factor.data_quality.uncertainty_percent}%")
        print()

        print("TEMPORAL:")
        print(f"  Last Updated: {factor.last_updated}")
        if factor.year_applicable:
            print(f"  Year:         {factor.year_applicable}")
        if factor.is_stale():
            print("  ⚠ WARNING: Factor is stale (>3 years old)")
        print()

        if factor.renewable_share:
            print(f"RENEWABLE SHARE: {factor.renewable_share * 100:.1f}%")
            print()

        if factor.gas_vectors:
            print("GAS BREAKDOWN:")
            for gas in factor.gas_vectors:
                print(f"  {gas.gas_type}: {gas.kg_per_unit:.6f} kg/unit", end='')
                if gas.gwp:
                    print(f" (GWP: {gas.gwp})")
                else:
                    print()
            print()

        if factor.notes:
            print("NOTES:")
            print(f"  {factor.notes}")
            print()

        print("PROVENANCE HASH:")
        print(f"  {factor.calculate_provenance_hash()}")
        print()

        client.close()

    except EmissionFactorNotFoundError as e:
        logger.error(f" {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f" {e}", file=sys.stderr)
        sys.exit(1)


def cmd_calculate(args):
    """Calculate emissions."""
    try:
        client = EmissionFactorClient(db_path=args.db_path)

        result = client.calculate_emissions(
            factor_id=args.factor,
            activity_amount=args.amount,
            activity_unit=args.unit,
            geography=args.geography if hasattr(args, 'geography') else None,
            year=args.year if hasattr(args, 'year') else None
        )

        # Display results
        print("=" * 70)
        print("EMISSION CALCULATION RESULT")
        print("=" * 70)
        print()

        print("ACTIVITY:")
        print(f"  Amount:       {result.activity_amount:,.2f} {result.activity_unit}")
        print()

        print("EMISSION FACTOR USED:")
        print(f"  Factor ID:    {result.factor_used.factor_id}")
        print(f"  Name:         {result.factor_used.name}")
        print(f"  Value:        {result.factor_value_applied:.6f} kg CO2e/{result.activity_unit}")
        print(f"  Source:       {result.factor_used.source.source_org}")
        print(f"  Last Updated: {result.factor_used.last_updated}")
        print()

        print("CALCULATION:")
        print(f"  {result.activity_amount:,.2f} {result.activity_unit} × {result.factor_value_applied:.6f} kg CO2e/{result.activity_unit}")
        print(f"  = {result.emissions_kg_co2e:,.2f} kg CO2e")
        print(f"  = {result.emissions_metric_tons_co2e:,.4f} metric tons CO2e")
        print()

        print("AUDIT TRAIL:")
        print(f"  Timestamp:    {result.calculation_timestamp.isoformat()}")
        print(f"  Hash:         {result.audit_trail}")
        print()

        if result.warnings:
            print("WARNINGS:")
            for warning in result.warnings:
                print(f"  ⚠ {warning}")
            print()

        # JSON output if requested
        if hasattr(args, 'json') and args.json:
            print("\nJSON OUTPUT:")
            print(json.dumps(result.to_dict(), indent=2))

        client.close()

    except EmissionFactorNotFoundError as e:
        logger.error(f" {e}", file=sys.stderr)
        print("\nTip: Use 'greenlang factors search' to find available factors")
        sys.exit(1)
    except UnitNotAvailableError as e:
        logger.error(f" {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f" {e}", file=sys.stderr)
        sys.exit(1)


def cmd_stats(args):
    """Show database statistics."""
    try:
        client = EmissionFactorClient(db_path=args.db_path)

        stats = client.get_statistics()

        print("=" * 70)
        print("EMISSION FACTOR DATABASE STATISTICS")
        print("=" * 70)
        print()

        print(f"Total Factors:        {stats['total_factors']:,}")
        print(f"Total Calculations:   {stats['total_calculations']:,}")
        print(f"Stale Factors:        {stats['stale_factors']:,}")
        print()

        print("BY CATEGORY:")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
            print(f"  {category:20s} {count:5d} factors")
        print()

        print("BY SCOPE:")
        for scope, count in sorted(stats['by_scope'].items(), key=lambda x: -x[1]):
            scope_name = scope if scope else 'N/A'
            print(f"  {scope_name:30s} {count:5d} factors")
        print()

        print("BY SOURCE:")
        for source, count in sorted(stats['by_source'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {source:40s} {count:5d} factors")
        print()

        client.close()

    except Exception as e:
        logger.error(f" {e}", file=sys.stderr)
        sys.exit(1)


def cmd_validate_db(args):
    """Validate database integrity."""
    try:
        print("Validating database...")
        print()

        results = validate_database(args.db_path)

        if results['valid']:
            print("✓ Database validation PASSED")
        else:
            print("✗ Database validation FAILED")

        print()
        print("STATISTICS:")
        for key, value in results['statistics'].items():
            print(f"  {key:20s} {value}")
        print()

        if results['warnings']:
            print("WARNINGS:")
            for warning in results['warnings']:
                print(f"  ⚠ {warning}")
            print()

        if results['errors']:
            print("ERRORS:")
            for error in results['errors']:
                print(f"  ✗ {error}")
            print()

        sys.exit(0 if results['valid'] else 1)

    except Exception as e:
        logger.error(f" {e}", file=sys.stderr)
        sys.exit(1)


def cmd_info(args):
    """Show database information."""
    try:
        info = get_database_info(args.db_path)

        print("=" * 70)
        print("DATABASE INFORMATION")
        print("=" * 70)
        print()

        print(f"File:      {info['file_path']}")
        print(f"Size:      {info['file_size_mb']:.2f} MB")
        print()

        print("TABLES:")
        for table, count in info['tables'].items():
            print(f"  {table:30s} {count:8,} rows")
        print()

        print(f"INDEXES:   {len(info['indexes'])}")
        print(f"VIEWS:     {len(info['views'])}")
        print()

    except Exception as e:
        logger.error(f" {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Emission Factor CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--db-path',
        default=None,
        help='Path to emission factors database'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # List command
    list_parser = subparsers.add_parser('list', help='List emission factors')
    list_parser.add_argument('--category', help='Filter by category')
    list_parser.add_argument('--scope', help='Filter by scope')
    list_parser.set_defaults(func=cmd_list)

    # Search command
    search_parser = subparsers.add_parser('search', help='Search emission factors')
    search_parser.add_argument('query', help='Search query')
    search_parser.set_defaults(func=cmd_search)

    # Get command
    get_parser = subparsers.add_parser('get', help='Get factor details')
    get_parser.add_argument('factor_id', help='Factor ID')
    get_parser.set_defaults(func=cmd_get)

    # Calculate command
    calc_parser = subparsers.add_parser('calculate', help='Calculate emissions')
    calc_parser.add_argument('--factor', required=True, help='Factor ID')
    calc_parser.add_argument('--amount', type=float, required=True, help='Activity amount')
    calc_parser.add_argument('--unit', required=True, help='Activity unit')
    calc_parser.add_argument('--geography', help='Geographic scope')
    calc_parser.add_argument('--year', type=int, help='Year')
    calc_parser.add_argument('--json', action='store_true', help='Output JSON')
    calc_parser.set_defaults(func=cmd_calculate)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.set_defaults(func=cmd_stats)

    # Validate command
    validate_parser = subparsers.add_parser('validate-db', help='Validate database')
    validate_parser.set_defaults(func=cmd_validate_db)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show database info')
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Set default database path if not provided
    if not args.db_path:
        args.db_path = str(
            Path(__file__).parent.parent / "data" / "emission_factors.db"
        )

    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
