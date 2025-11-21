# -*- coding: utf-8 -*-
"""
Emission Factors Database Schema

This module creates and manages the SQLite database schema for emission factors.
The schema is optimized for fast queries with proper indexing.

Example:
    >>> from greenlang.db.emission_factors_schema import create_database
    >>> create_database("C:/path/to/emission_factors.db")
"""

import sqlite3
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# SQL schema definition
SCHEMA_SQL = """
-- Main emission factors table
CREATE TABLE IF NOT EXISTS emission_factors (
    factor_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT,

    -- Primary emission factor value
    emission_factor_value REAL NOT NULL,
    unit TEXT NOT NULL,

    -- GHG scope and classification
    scope TEXT,

    -- Source provenance
    source_org TEXT NOT NULL,
    source_publication TEXT,
    source_uri TEXT NOT NULL,
    standard TEXT,
    year_published INTEGER,

    -- Temporal information
    last_updated DATE NOT NULL,
    year_applicable INTEGER,

    -- Geographic scope
    geographic_scope TEXT,
    geography_level TEXT,
    country_code TEXT,
    state_province TEXT,
    region TEXT,

    -- Data quality
    data_quality_tier TEXT,
    uncertainty_percent REAL,
    confidence_95ci REAL,
    completeness_score REAL,

    -- Optional attributes
    renewable_share REAL,
    notes TEXT,
    metadata_json TEXT,

    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CHECK (emission_factor_value > 0),
    CHECK (renewable_share IS NULL OR (renewable_share >= 0 AND renewable_share <= 1)),
    CHECK (uncertainty_percent IS NULL OR (uncertainty_percent >= 0 AND uncertainty_percent <= 100)),
    CHECK (completeness_score IS NULL OR (completeness_score >= 0 AND completeness_score <= 1))
);

-- Additional units table (for factors with multiple unit representations)
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

-- Gas vectors table (individual gas contributions)
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

-- Calculation audit log
CREATE TABLE IF NOT EXISTS calculation_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calculation_id TEXT NOT NULL UNIQUE,
    factor_id TEXT NOT NULL,
    activity_amount REAL NOT NULL,
    activity_unit TEXT NOT NULL,
    emissions_kg_co2e REAL NOT NULL,
    factor_value_used REAL NOT NULL,
    calculation_timestamp TIMESTAMP NOT NULL,
    audit_hash TEXT NOT NULL,
    warnings TEXT,
    metadata_json TEXT,

    FOREIGN KEY (factor_id) REFERENCES emission_factors(factor_id),
    CHECK (activity_amount >= 0),
    CHECK (emissions_kg_co2e >= 0)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_category ON emission_factors(category, subcategory);
CREATE INDEX IF NOT EXISTS idx_scope ON emission_factors(scope);
CREATE INDEX IF NOT EXISTS idx_geography ON emission_factors(geographic_scope, geography_level);
CREATE INDEX IF NOT EXISTS idx_geography_country ON emission_factors(country_code);
CREATE INDEX IF NOT EXISTS idx_geography_state ON emission_factors(state_province);
CREATE INDEX IF NOT EXISTS idx_source ON emission_factors(source_org);
CREATE INDEX IF NOT EXISTS idx_updated ON emission_factors(last_updated);
CREATE INDEX IF NOT EXISTS idx_year ON emission_factors(year_applicable);
CREATE INDEX IF NOT EXISTS idx_quality ON emission_factors(data_quality_tier);
CREATE INDEX IF NOT EXISTS idx_name_search ON emission_factors(name);

-- Additional table indexes
CREATE INDEX IF NOT EXISTS idx_factor_units_factor ON factor_units(factor_id);
CREATE INDEX IF NOT EXISTS idx_factor_units_name ON factor_units(unit_name);
CREATE INDEX IF NOT EXISTS idx_gas_vectors_factor ON factor_gas_vectors(factor_id);
CREATE INDEX IF NOT EXISTS idx_gas_vectors_type ON factor_gas_vectors(gas_type);
CREATE INDEX IF NOT EXISTS idx_audit_factor ON calculation_audit_log(factor_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON calculation_audit_log(calculation_timestamp);

-- Full-text search (optional, for advanced search)
-- CREATE VIRTUAL TABLE IF NOT EXISTS emission_factors_fts USING fts5(
--     factor_id, name, category, subcategory, notes,
--     content=emission_factors,
--     content_rowid=rowid
-- );

-- Trigger to update updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_emission_factors_timestamp
AFTER UPDATE ON emission_factors
BEGIN
    UPDATE emission_factors SET updated_at = CURRENT_TIMESTAMP
    WHERE factor_id = NEW.factor_id;
END;
"""

# Database statistics views
VIEWS_SQL = """
-- Statistics view
CREATE VIEW IF NOT EXISTS factor_statistics AS
SELECT
    category,
    subcategory,
    COUNT(*) as factor_count,
    AVG(emission_factor_value) as avg_factor,
    MIN(emission_factor_value) as min_factor,
    MAX(emission_factor_value) as max_factor,
    COUNT(DISTINCT source_org) as source_count,
    MIN(last_updated) as oldest_update,
    MAX(last_updated) as newest_update
FROM emission_factors
GROUP BY category, subcategory;

-- Geography coverage view
CREATE VIEW IF NOT EXISTS geography_coverage AS
SELECT
    geography_level,
    geographic_scope,
    country_code,
    COUNT(*) as factor_count,
    COUNT(DISTINCT category) as category_count
FROM emission_factors
GROUP BY geography_level, geographic_scope, country_code;

-- Data quality summary view
CREATE VIEW IF NOT EXISTS quality_summary AS
SELECT
    data_quality_tier,
    COUNT(*) as factor_count,
    AVG(uncertainty_percent) as avg_uncertainty,
    AVG(completeness_score) as avg_completeness
FROM emission_factors
GROUP BY data_quality_tier;

-- Source provenance view
CREATE VIEW IF NOT EXISTS source_summary AS
SELECT
    source_org,
    standard,
    COUNT(*) as factor_count,
    MIN(last_updated) as oldest_factor,
    MAX(last_updated) as newest_factor,
    COUNT(DISTINCT category) as categories_covered
FROM emission_factors
GROUP BY source_org, standard;

-- Stale factors view (older than 3 years)
CREATE VIEW IF NOT EXISTS stale_factors AS
SELECT
    factor_id,
    name,
    category,
    last_updated,
    CAST((julianday('now') - julianday(last_updated)) / 365.25 AS INTEGER) as years_old,
    source_org
FROM emission_factors
WHERE julianday('now') - julianday(last_updated) > (3 * 365);
"""


def create_database(db_path: str, overwrite: bool = False) -> bool:
    """
    Create emission factors database with schema.

    Args:
        db_path: Path to database file
        overwrite: If True, drop existing tables first

    Returns:
        True if successful

    Raises:
        sqlite3.Error: If database creation fails
    """
    db_file = Path(db_path)

    # Create parent directory if needed
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if database exists
    db_exists = db_file.exists()

    if db_exists and overwrite:
        logger.warning(f"Dropping existing database: {db_path}")
        db_file.unlink()
        db_exists = False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Execute schema
        logger.info("Creating emission factors database schema...")
        cursor.executescript(SCHEMA_SQL)

        # Create views
        logger.info("Creating database views...")
        cursor.executescript(VIEWS_SQL)

        conn.commit()

        # Verify tables created
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name;
        """)
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            'emission_factors',
            'factor_units',
            'factor_gas_vectors',
            'calculation_audit_log'
        ]

        for table in expected_tables:
            if table not in tables:
                raise sqlite3.Error(f"Failed to create table: {table}")

        logger.info(f"Database created successfully: {db_path}")
        logger.info(f"Tables created: {', '.join(tables)}")

        # Get table counts
        # SECURITY FIX: Validate table names against whitelist before SQL execution
        # This prevents SQL injection via malicious table names
        allowed_tables = {
            'emission_factors',
            'factor_units',
            'factor_gas_vectors',
            'calculation_audit_log'
        }

        for table in expected_tables:
            # SECURITY: Whitelist validation - only allow known table names
            if table not in allowed_tables:
                logger.error(f"Table name not in whitelist: {table}")
                raise ValueError(f"Invalid table name: {table}")

            # Safe to use in query after whitelist validation
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"  {table}: {count} rows")

        conn.close()
        return True

    except sqlite3.Error as e:
        logger.error(f"Database creation failed: {e}")
        raise


def validate_database(db_path: str) -> dict:
    """
    Validate database schema and integrity.

    Args:
        db_path: Path to database file

    Returns:
        Dictionary with validation results

    Raises:
        FileNotFoundError: If database doesn't exist
        sqlite3.Error: If validation fails
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    try:
        # Check foreign key integrity
        cursor.execute("PRAGMA foreign_key_check;")
        fk_violations = cursor.fetchall()
        if fk_violations:
            results['valid'] = False
            results['errors'].append(f"Foreign key violations: {len(fk_violations)}")

        # Check for required tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name;
        """)
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            'emission_factors',
            'factor_units',
            'factor_gas_vectors',
            'calculation_audit_log'
        ]

        for table in expected_tables:
            if table not in tables:
                results['valid'] = False
                results['errors'].append(f"Missing table: {table}")

        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM emission_factors")
        results['statistics']['total_factors'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT category) FROM emission_factors")
        results['statistics']['categories'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT source_org) FROM emission_factors")
        results['statistics']['sources'] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM emission_factors
            WHERE julianday('now') - julianday(last_updated) > (3 * 365)
        """)
        stale_count = cursor.fetchone()[0]
        if stale_count > 0:
            results['warnings'].append(f"{stale_count} factors older than 3 years")
        results['statistics']['stale_factors'] = stale_count

        # Check for duplicate factor IDs
        cursor.execute("""
            SELECT factor_id, COUNT(*) as count
            FROM emission_factors
            GROUP BY factor_id
            HAVING count > 1
        """)
        duplicates = cursor.fetchall()
        if duplicates:
            results['valid'] = False
            results['errors'].append(f"Duplicate factor IDs: {len(duplicates)}")

        # Check for invalid emission factors (should be > 0)
        cursor.execute("""
            SELECT COUNT(*) FROM emission_factors
            WHERE emission_factor_value <= 0
        """)
        invalid_factors = cursor.fetchone()[0]
        if invalid_factors > 0:
            results['valid'] = False
            results['errors'].append(f"Invalid emission factors (<=0): {invalid_factors}")

        logger.info(f"Database validation: {'PASSED' if results['valid'] else 'FAILED'}")
        logger.info(f"Total factors: {results['statistics']['total_factors']}")
        logger.info(f"Categories: {results['statistics']['categories']}")
        logger.info(f"Sources: {results['statistics']['sources']}")

        if results['warnings']:
            for warning in results['warnings']:
                logger.warning(warning)

        if results['errors']:
            for error in results['errors']:
                logger.error(error)

    except sqlite3.Error as e:
        results['valid'] = False
        results['errors'].append(str(e))
        logger.error(f"Validation error: {e}")

    finally:
        conn.close()

    return results


def get_database_info(db_path: str) -> dict:
    """
    Get detailed database information.

    Args:
        db_path: Path to database file

    Returns:
        Dictionary with database information
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    info = {
        'file_path': str(db_file.absolute()),
        'file_size_mb': db_file.stat().st_size / (1024 * 1024),
        'tables': {},
        'indexes': [],
        'views': []
    }

    # Get tables
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name;
    """)

    # SECURITY FIX: Whitelist table names to prevent SQL injection
    allowed_tables = {
        'emission_factors',
        'factor_units',
        'factor_gas_vectors',
        'calculation_audit_log',
        'sqlite_sequence'  # System table
    }

    for (table_name,) in cursor.fetchall():
        # SECURITY: Validate table name against whitelist
        if table_name not in allowed_tables:
            logger.warning(f"Skipping unknown table: {table_name}")
            continue

        # Safe to use in query after whitelist validation
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        info['tables'][table_name] = count

    # Get indexes
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='index'
        ORDER BY name;
    """)
    info['indexes'] = [row[0] for row in cursor.fetchall()]

    # Get views
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='view'
        ORDER BY name;
    """)
    info['views'] = [row[0] for row in cursor.fetchall()]

    conn.close()

    return info


if __name__ == "__main__":
    # Example usage
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "C:/Users/aksha/Code-V1_GreenLang/greenlang/data/emission_factors.db"

    print(f"Creating database: {db_path}")
    create_database(db_path, overwrite=True)

    print("\nValidating database...")
    results = validate_database(db_path)

    print("\nDatabase info:")
    info = get_database_info(db_path)
    print(f"File: {info['file_path']}")
    print(f"Size: {info['file_size_mb']:.2f} MB")
    print(f"Tables: {list(info['tables'].keys())}")
    print(f"Indexes: {len(info['indexes'])}")
    print(f"Views: {len(info['views'])}")
