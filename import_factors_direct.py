"""
Direct Emission Factor Import - Bypasses SQLAlchemy issues

This script directly imports emission factors using sqlite3 without
going through the greenlang.db module which has SQLAlchemy conflicts.
"""

import sys
import yaml
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database schema (copied from emission_factors_schema.py)
SCHEMA_SQL = """
-- Main emission factors table
CREATE TABLE IF NOT EXISTS emission_factors (
    factor_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT,
    emission_factor_value REAL NOT NULL,
    unit TEXT NOT NULL,
    scope TEXT,
    source_org TEXT NOT NULL,
    source_publication TEXT,
    source_uri TEXT NOT NULL,
    standard TEXT,
    year_published INTEGER,
    last_updated DATE NOT NULL,
    year_applicable INTEGER,
    geographic_scope TEXT,
    geography_level TEXT,
    country_code TEXT,
    state_province TEXT,
    region TEXT,
    data_quality_tier TEXT,
    uncertainty_percent REAL,
    confidence_95ci REAL,
    completeness_score REAL,
    renewable_share REAL,
    notes TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (emission_factor_value > 0),
    CHECK (renewable_share IS NULL OR (renewable_share >= 0 AND renewable_share <= 1)),
    CHECK (uncertainty_percent IS NULL OR (uncertainty_percent >= 0 AND uncertainty_percent <= 100)),
    CHECK (completeness_score IS NULL OR (completeness_score >= 0 AND completeness_score <= 1))
);

CREATE TABLE IF NOT EXISTS factor_units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id TEXT NOT NULL,
    unit_type TEXT NOT NULL,
    unit_name TEXT NOT NULL,
    emission_factor_value REAL NOT NULL,
    conversion_to_base REAL,
    FOREIGN KEY (factor_id) REFERENCES emission_factors(factor_id) ON DELETE CASCADE,
    UNIQUE(factor_id, unit_name),
    CHECK (emission_factor_value > 0)
);

CREATE TABLE IF NOT EXISTS factor_gas_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id TEXT NOT NULL,
    gas_type TEXT NOT NULL,
    kg_per_unit REAL NOT NULL,
    gwp INTEGER,
    FOREIGN KEY (factor_id) REFERENCES emission_factors(factor_id) ON DELETE CASCADE,
    UNIQUE(factor_id, gas_type),
    CHECK (kg_per_unit >= 0),
    CHECK (gas_type IN ('CO2', 'CH4', 'N2O', 'HFCs', 'PFCs', 'SF6', 'NF3'))
);

CREATE TABLE IF NOT EXISTS calculation_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calculation_id TEXT NOT NULL,
    factor_id TEXT NOT NULL,
    activity_amount REAL NOT NULL,
    activity_unit TEXT NOT NULL,
    emissions_kg_co2e REAL NOT NULL,
    audit_hash TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (factor_id) REFERENCES emission_factors(factor_id),
    UNIQUE(calculation_id, timestamp)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_factors_category ON emission_factors(category);
CREATE INDEX IF NOT EXISTS idx_factors_scope ON emission_factors(scope);
CREATE INDEX IF NOT EXISTS idx_factors_geographic ON emission_factors(geographic_scope);
CREATE INDEX IF NOT EXISTS idx_factors_country ON emission_factors(country_code);
CREATE INDEX IF NOT EXISTS idx_factors_quality ON emission_factors(data_quality_tier);
CREATE INDEX IF NOT EXISTS idx_factors_updated ON emission_factors(last_updated);
CREATE INDEX IF NOT EXISTS idx_units_factor ON factor_units(factor_id);
CREATE INDEX IF NOT EXISTS idx_units_name ON factor_units(unit_name);
CREATE INDEX IF NOT EXISTS idx_gas_factor ON factor_gas_vectors(factor_id);
CREATE INDEX IF NOT EXISTS idx_gas_type ON factor_gas_vectors(gas_type);
CREATE INDEX IF NOT EXISTS idx_audit_factor ON calculation_audit_log(factor_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON calculation_audit_log(timestamp);
"""


def create_database(db_path):
    """Create database with schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    logger.info(f"Created database at {db_path}")


def import_yaml_file(conn, yaml_path, stats):
    """Import factors from a single YAML file."""
    logger.info(f"Importing from {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not data:
        logger.warning(f"Empty or invalid YAML file: {yaml_path}")
        return

    # Handle different YAML structures - factors may be nested under categories
    def process_dict(data_dict, parent_category=None):
        """Recursively process nested dictionaries to find factors."""
        for key, value in data_dict.items():
            if not isinstance(value, dict):
                continue

            # Skip metadata entries
            if key in ['metadata', 'validation', 'summary']:
                continue

            # Check if this is a factor (has emission_factor_kg_co2e_per_* field)
            has_emission_factor = any(k.startswith('emission_factor_kg_co2e_per_') for k in value.keys())

            if has_emission_factor:
                # This is an actual emission factor
                import_factor(conn, key, value, parent_category, stats)
            else:
                # This might be a category grouping, recurse into it
                process_dict(value, parent_category=key)

    process_dict(data)


def import_factor(conn, factor_id, factor_data, parent_category, stats):
    """Import a single emission factor."""
    try:
        # Extract primary value and unit
        primary_value = None
        primary_unit = None

        # Find emission_factor_kg_co2e_per_* fields
        for key in factor_data.keys():
            if key.startswith('emission_factor_kg_co2e_per_'):
                primary_unit = key.replace('emission_factor_kg_co2e_per_', '')
                primary_value = float(factor_data[key])
                break

        if primary_value is None or primary_unit is None:
            logger.warning(f"No emission factor value found for {factor_id}")
            return

        # Extract metadata
        name = factor_data.get('name', factor_id)
        category = factor_data.get('category', parent_category or 'unknown')
        source_org = factor_data.get('source', 'Unknown')
        source_uri = factor_data.get('uri', '')
        last_updated = factor_data.get('last_updated', '2024-01-01')

        # Parse uncertainty (might be "+/- X%" format)
        uncertainty = factor_data.get('uncertainty')
        if uncertainty and isinstance(uncertainty, str):
            # Extract number from "+/- X%" format
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', uncertainty)
            if match:
                uncertainty = float(match.group(1))
            else:
                uncertainty = None

        # Normalize data quality tier to enum format
        data_quality = factor_data.get('data_quality', '')
        if data_quality:
            # Extract "Tier 1", "Tier 2", "Tier 3" from strings like "Tier 1 - National Average"
            if 'Tier 1' in data_quality or 'tier 1' in data_quality.lower():
                data_quality = 'Tier 1'
            elif 'Tier 2' in data_quality or 'tier 2' in data_quality.lower():
                data_quality = 'Tier 2'
            elif 'Tier 3' in data_quality or 'tier 3' in data_quality.lower():
                data_quality = 'Tier 3'
            else:
                data_quality = 'Tier 1'  # Default

        # Insert into emission_factors table
        conn.execute("""
            INSERT OR REPLACE INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit,
                scope, source_org, source_uri, standard,
                last_updated, geographic_scope,
                data_quality_tier, uncertainty_percent,
                renewable_share, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            factor_id,
            name,
            category,
            factor_data.get('subcategory'),
            primary_value,
            primary_unit,
            factor_data.get('scope'),
            source_org,
            source_uri,
            factor_data.get('standard'),
            last_updated,
            factor_data.get('geographic_scope'),
            data_quality,
            uncertainty,
            factor_data.get('renewable_share'),
            factor_data.get('notes')
        ))

        stats['successful'] += 1
        stats['categories'].add(category)

    except Exception as e:
        logger.error(f"Error importing {factor_id}: {e}")
        stats['failed'] += 1


def main():
    """Main import function."""
    project_root = Path(__file__).parent
    db_path = project_root / "greenlang" / "data" / "emission_factors.db"
    data_dir = project_root / "data"

    # Create database
    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_database(str(db_path))

    # Import all YAML files
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")

    stats = {
        'successful': 0,
        'failed': 0,
        'categories': set()
    }

    yaml_files = list(data_dir.glob("*.yaml"))
    logger.info(f"Found {len(yaml_files)} YAML files to import")

    for yaml_file in yaml_files:
        import_yaml_file(conn, yaml_file, stats)
        conn.commit()

    conn.close()

    # Print statistics
    logger.info("=" * 60)
    logger.info("IMPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successful imports: {stats['successful']}")
    logger.info(f"Failed imports: {stats['failed']}")
    logger.info(f"Categories: {len(stats['categories'])}")
    logger.info(f"Database: {db_path}")

    # Verify database
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM emission_factors").fetchone()[0]
    logger.info(f"Total factors in database: {count}")
    conn.close()


if __name__ == "__main__":
    main()
