"""
Import Emission Factors from YAML to SQLite Database

This script parses all emission factor YAML files and imports them into
the SQLite database with full validation and error handling.

Usage:
    python import_emission_factors.py
    python import_emission_factors.py --db-path /path/to/db --overwrite
"""

import sys
import yaml
import sqlite3
import logging
import json
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.db.emission_factors_schema import create_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ImportStatistics:
    """Statistics for import process."""
    total_factors: int = 0
    successful_imports: int = 0
    failed_imports: int = 0
    duplicate_factors: int = 0
    categories: set = None
    sources: set = None
    errors: list = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = set()
        if self.sources is None:
            self.sources = set()
        if self.errors is None:
            self.errors = []


class EmissionFactorImporter:
    """
    Import emission factors from YAML files to SQLite database.

    This importer handles:
    - Multiple YAML file formats
    - Multiple units per factor
    - Gas vector decomposition
    - Data validation
    - Error recovery
    """

    def __init__(self, db_path: str):
        """
        Initialize importer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.stats = ImportStatistics()
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def parse_yaml_file(self, yaml_path: str) -> Dict[str, Any]:
        """
        Parse YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Parsed YAML data
        """
        logger.info(f"Parsing YAML file: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return data

    def normalize_date(self, date_str: Any) -> str:
        """
        Normalize date string to ISO format.

        Args:
            date_str: Date string or date object

        Returns:
            ISO format date string (YYYY-MM-DD)
        """
        if isinstance(date_str, date):
            return date_str.isoformat()

        if isinstance(date_str, str):
            # Try various date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.date().isoformat()
                except ValueError:
                    continue

        # Default to today if parsing fails
        logger.warning(f"Could not parse date '{date_str}', using today's date")
        return datetime.now().date().isoformat()

    def extract_unit_variations(self, factor_data: Dict[str, Any], category: str, factor_id: str) -> tuple[Dict[str, float], str, float]:
        """
        Extract all unit variations from factor data.

        Args:
            factor_data: Factor data dictionary
            category: Factor category
            factor_id: Factor ID

        Returns:
            Tuple of (unit_dict, primary_unit, primary_value)
        """
        unit_variations = {}
        primary_unit = None
        primary_value = None

        # Extract all emission_factor_* fields
        for key, value in factor_data.items():
            if key.startswith('emission_factor_kg_co2e_per_'):
                # Extract unit name
                unit_name = key.replace('emission_factor_kg_co2e_per_', '')

                if isinstance(value, (int, float)) and value > 0:
                    unit_variations[unit_name] = float(value)

                    # Use first unit as primary
                    if primary_unit is None:
                        primary_unit = unit_name
                        primary_value = float(value)

        # Fallback: check for generic emission_factor field
        if not unit_variations:
            if 'emission_factor' in factor_data:
                value = factor_data['emission_factor']
                unit = factor_data.get('unit', 'unit')
                unit_variations[unit] = float(value)
                primary_unit = unit
                primary_value = float(value)
            elif 'emission_factor_value' in factor_data:
                value = factor_data['emission_factor_value']
                unit = factor_data.get('unit', 'unit')
                unit_variations[unit] = float(value)
                primary_unit = unit
                primary_value = float(value)

        if not primary_unit:
            logger.error(f"No emission factor value found for {factor_id}")
            raise ValueError(f"No emission factor value for {factor_id}")

        return unit_variations, primary_unit, primary_value

    def extract_gas_vectors(self, factor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract gas vector breakdown if available.

        Args:
            factor_data: Factor data dictionary

        Returns:
            List of gas vector dictionaries
        """
        gas_vectors = []

        # Check for gas breakdown fields
        gas_keys = ['co2', 'ch4', 'n2o', 'hfcs', 'pfcs', 'sf6', 'nf3']

        for gas in gas_keys:
            key = f'{gas}_kg_per_unit'
            if key in factor_data:
                gas_vectors.append({
                    'gas_type': gas.upper(),
                    'kg_per_unit': float(factor_data[key]),
                    'gwp': factor_data.get(f'{gas}_gwp')
                })

        return gas_vectors

    def import_factor(self, factor_id: str, factor_data: Dict[str, Any], category: str, subcategory: Optional[str] = None) -> bool:
        """
        Import a single emission factor.

        Args:
            factor_id: Unique factor identifier
            factor_data: Factor data dictionary
            category: Primary category
            subcategory: Optional subcategory

        Returns:
            True if successful
        """
        try:
            self.stats.total_factors += 1

            # Extract unit variations
            unit_variations, primary_unit, primary_value = self.extract_unit_variations(
                factor_data, category, factor_id
            )

            # Extract basic fields
            name = factor_data.get('name', factor_id.replace('_', ' ').title())
            scope = factor_data.get('scope', 'Unknown')
            source_org = factor_data.get('source', 'Unknown')
            source_uri = factor_data.get('uri', factor_data.get('source_uri', 'https://example.com'))
            standard = factor_data.get('standard')
            last_updated = self.normalize_date(factor_data.get('last_updated', datetime.now().date()))
            year_applicable = factor_data.get('year', factor_data.get('year_applicable'))

            # Geographic scope
            geographic_scope = factor_data.get('geographic_scope', factor_data.get('region', 'Global'))
            geography_level = factor_data.get('geography_level', 'Global')
            country_code = factor_data.get('country_code')
            state_province = factor_data.get('state_province')
            region = factor_data.get('region')

            # Data quality
            data_quality_tier = factor_data.get('data_quality', factor_data.get('data_quality_tier'))
            uncertainty_str = factor_data.get('uncertainty', '')
            uncertainty_percent = None
            if uncertainty_str and isinstance(uncertainty_str, str):
                # Parse "+/- 5%" format
                uncertainty_str = uncertainty_str.replace('+/-', '').replace('%', '').strip()
                try:
                    uncertainty_percent = float(uncertainty_str)
                except ValueError:
                    pass

            # Optional fields
            renewable_share = factor_data.get('renewable_share')
            notes = factor_data.get('notes')

            # Metadata
            metadata = {
                'source_publication': factor_data.get('source_publication'),
                'year_published': factor_data.get('year_published'),
                'fuel_economy': factor_data.get('fuel_economy_l_per_100km'),
                'vehicle_class': factor_data.get('vehicle_class'),
                'coal_share': factor_data.get('coal_share'),
                'natural_gas_share': factor_data.get('natural_gas_share'),
                'solar_share': factor_data.get('solar_share'),
                'wind_share': factor_data.get('wind_share'),
                'nuclear_share': factor_data.get('nuclear_share'),
                'hcv': factor_data.get('hcv_mj_per_kg')
            }
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadata_json = json.dumps(metadata) if metadata else None

            # Insert main factor record
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO emission_factors (
                    factor_id, name, category, subcategory,
                    emission_factor_value, unit,
                    scope,
                    source_org, source_publication, source_uri, standard,
                    last_updated, year_applicable,
                    geographic_scope, geography_level, country_code, state_province, region,
                    data_quality_tier, uncertainty_percent,
                    renewable_share, notes, metadata_json
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?,
                    ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?
                )
            """, (
                factor_id, name, category, subcategory,
                primary_value, primary_unit,
                scope,
                source_org, metadata.get('source_publication'), source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level, country_code, state_province, region,
                data_quality_tier, uncertainty_percent,
                renewable_share, notes, metadata_json
            ))

            # Insert additional units
            for unit_name, unit_value in unit_variations.items():
                if unit_name != primary_unit:
                    cursor.execute("""
                        INSERT INTO factor_units (
                            factor_id, unit_type, unit_name, emission_factor_value
                        ) VALUES (?, ?, ?, ?)
                    """, (factor_id, 'emission_factor', unit_name, unit_value))

            # Insert gas vectors if available
            gas_vectors = self.extract_gas_vectors(factor_data)
            for gas_vector in gas_vectors:
                cursor.execute("""
                    INSERT INTO factor_gas_vectors (
                        factor_id, gas_type, kg_per_unit, gwp
                    ) VALUES (?, ?, ?, ?)
                """, (
                    factor_id,
                    gas_vector['gas_type'],
                    gas_vector['kg_per_unit'],
                    gas_vector.get('gwp')
                ))

            self.conn.commit()

            self.stats.successful_imports += 1
            self.stats.categories.add(category)
            self.stats.sources.add(source_org)

            logger.debug(f"Imported: {factor_id} ({category})")
            return True

        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            if 'UNIQUE constraint failed' in str(e):
                logger.warning(f"Duplicate factor ID: {factor_id}")
                self.stats.duplicate_factors += 1
            else:
                logger.error(f"Integrity error for {factor_id}: {e}")
                self.stats.failed_imports += 1
                self.stats.errors.append(f"{factor_id}: {str(e)}")
            return False

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to import {factor_id}: {e}", exc_info=True)
            self.stats.failed_imports += 1
            self.stats.errors.append(f"{factor_id}: {str(e)}")
            return False

    def import_yaml_file(self, yaml_path: str) -> int:
        """
        Import all factors from a YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Number of factors imported
        """
        logger.info(f"Importing from: {yaml_path}")

        data = self.parse_yaml_file(yaml_path)

        # Remove metadata section
        if 'metadata' in data:
            del data['metadata']

        imported_count = 0

        # Process each category
        for category, category_data in data.items():
            if not isinstance(category_data, dict):
                continue

            logger.info(f"Processing category: {category}")

            # Process each factor in category
            for factor_key, factor_data in category_data.items():
                if not isinstance(factor_data, dict):
                    continue

                # Generate factor ID
                factor_id = f"{category}_{factor_key}".lower()

                # Import factor
                if self.import_factor(factor_id, factor_data, category, subcategory=factor_key):
                    imported_count += 1

        logger.info(f"Imported {imported_count} factors from {Path(yaml_path).name}")
        return imported_count

    def import_all_yaml_files(self, yaml_paths: List[str]) -> ImportStatistics:
        """
        Import all YAML files.

        Args:
            yaml_paths: List of YAML file paths

        Returns:
            Import statistics
        """
        logger.info(f"Starting import of {len(yaml_paths)} YAML files...")

        self.connect()

        try:
            for yaml_path in yaml_paths:
                if not Path(yaml_path).exists():
                    logger.error(f"File not found: {yaml_path}")
                    continue

                self.import_yaml_file(yaml_path)

            logger.info("=" * 70)
            logger.info("IMPORT COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Total factors processed: {self.stats.total_factors}")
            logger.info(f"Successfully imported: {self.stats.successful_imports}")
            logger.info(f"Failed imports: {self.stats.failed_imports}")
            logger.info(f"Duplicate factors: {self.stats.duplicate_factors}")
            logger.info(f"Unique categories: {len(self.stats.categories)}")
            logger.info(f"Unique sources: {len(self.stats.sources)}")

            if self.stats.errors:
                logger.error(f"\nErrors ({len(self.stats.errors)}):")
                for error in self.stats.errors[:10]:  # Show first 10 errors
                    logger.error(f"  - {error}")
                if len(self.stats.errors) > 10:
                    logger.error(f"  ... and {len(self.stats.errors) - 10} more errors")

            logger.info("\nCategories:")
            for cat in sorted(self.stats.categories):
                logger.info(f"  - {cat}")

            logger.info("\nSources:")
            for src in sorted(self.stats.sources):
                logger.info(f"  - {src}")

        finally:
            self.close()

        return self.stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Import emission factors from YAML to SQLite')
    parser.add_argument(
        '--db-path',
        default='C:/Users/aksha/Code-V1_GreenLang/greenlang/data/emission_factors.db',
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--data-dir',
        default='C:/Users/aksha/Code-V1_GreenLang/data',
        help='Directory containing YAML files'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing database'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create database
    logger.info(f"Database path: {args.db_path}")
    create_database(args.db_path, overwrite=args.overwrite)

    # Find YAML files
    data_dir = Path(args.data_dir)
    yaml_files = [
        str(data_dir / 'emission_factors_registry.yaml'),
        str(data_dir / 'emission_factors_expansion_phase1.yaml'),
        str(data_dir / 'emission_factors_expansion_phase2.yaml')
    ]

    # Import
    importer = EmissionFactorImporter(args.db_path)
    stats = importer.import_all_yaml_files(yaml_files)

    # Success if more imports than failures
    success = stats.successful_imports > stats.failed_imports

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
