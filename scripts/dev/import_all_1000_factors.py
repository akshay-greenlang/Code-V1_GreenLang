# -*- coding: utf-8 -*-
"""
Import All 1,000 Emission Factors - Production Database Import

This script imports all emission factors from all 6 YAML files into the
production SQLite database.

Files to import:
1. emission_factors_registry.yaml (78 factors)
2. emission_factors_expansion_phase1.yaml (114 factors)
3. emission_factors_expansion_phase2.yaml (308 factors)
4. emission_factors_expansion_phase3_manufacturing_fuels.yaml (70 factors)
5. emission_factors_expansion_phase3b_grids_industry.yaml (175 factors)
6. emission_factors_expansion_phase4.yaml (255 factors)
Total: 1,000 factors

Database: C:/Users/aksha/Code-V1_GreenLang/greenlang/data/emission_factors.db
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from greenlang.determinism import DeterministicClock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from greenlang.db.emission_factors_schema import create_database, validate_database, get_database_info
from scripts.import_emission_factors import EmissionFactorImporter

def print_separator():
    print("=" * 80)

def run_validation_queries(db_path: str):
    """Run validation queries on the database."""
    print_separator()
    print("RUNNING VALIDATION QUERIES")
    print_separator()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Total count
    print("\n1. TOTAL FACTORS:")
    cursor.execute("SELECT COUNT(*) FROM emission_factors")
    total = cursor.fetchone()[0]
    print(f"   Total emission factors: {total:,}")

    # By scope
    print("\n2. FACTORS BY SCOPE:")
    cursor.execute("SELECT scope, COUNT(*) as count FROM emission_factors GROUP BY scope ORDER BY count DESC")
    for scope, count in cursor.fetchall():
        print(f"   {scope}: {count:,}")

    # By category (top 20)
    print("\n3. FACTORS BY CATEGORY (Top 20):")
    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM emission_factors
        GROUP BY category
        ORDER BY count DESC
        LIMIT 20
    """)
    for category, count in cursor.fetchall():
        print(f"   {category}: {count:,}")

    # By source organization (top 20)
    print("\n4. FACTORS BY SOURCE ORGANIZATION (Top 20):")
    cursor.execute("""
        SELECT source_org, COUNT(*) as count
        FROM emission_factors
        GROUP BY source_org
        ORDER BY count DESC
        LIMIT 20
    """)
    for source, count in cursor.fetchall():
        print(f"   {source}: {count:,}")

    # Recent updates
    print("\n5. FACTORS UPDATED SINCE 2024-01-01:")
    cursor.execute("""
        SELECT COUNT(*)
        FROM emission_factors
        WHERE last_updated >= '2024-01-01'
    """)
    recent = cursor.fetchone()[0]
    print(f"   Recent factors: {recent:,}")

    # Missing URIs (should be 0)
    print("\n6. FACTORS WITH MISSING URIs:")
    cursor.execute("""
        SELECT COUNT(*)
        FROM emission_factors
        WHERE source_uri IS NULL OR source_uri = ''
    """)
    missing_uris = cursor.fetchone()[0]
    print(f"   Missing URIs: {missing_uris}")
    if missing_uris > 0:
        print("   WARNING: Some factors are missing source URIs!")

    # Geographic coverage
    print("\n7. GEOGRAPHIC COVERAGE:")
    cursor.execute("""
        SELECT geographic_scope, COUNT(*) as count
        FROM emission_factors
        GROUP BY geographic_scope
        ORDER BY count DESC
        LIMIT 15
    """)
    for geo, count in cursor.fetchall():
        print(f"   {geo}: {count:,}")

    # Data quality distribution
    print("\n8. DATA QUALITY DISTRIBUTION:")
    cursor.execute("""
        SELECT data_quality_tier, COUNT(*) as count
        FROM emission_factors
        GROUP BY data_quality_tier
        ORDER BY count DESC
    """)
    for quality, count in cursor.fetchall():
        quality_str = quality if quality else "(not specified)"
        print(f"   {quality_str}: {count:,}")

    # Additional units
    print("\n9. ADDITIONAL UNIT VARIATIONS:")
    cursor.execute("SELECT COUNT(*) FROM factor_units")
    units_count = cursor.fetchone()[0]
    print(f"   Additional units defined: {units_count:,}")

    # Gas vectors
    print("\n10. GAS VECTOR BREAKDOWNS:")
    cursor.execute("SELECT COUNT(*) FROM factor_gas_vectors")
    gas_count = cursor.fetchone()[0]
    print(f"   Gas vector entries: {gas_count:,}")

    cursor.execute("""
        SELECT gas_type, COUNT(*) as count
        FROM factor_gas_vectors
        GROUP BY gas_type
        ORDER BY count DESC
    """)
    for gas, count in cursor.fetchall():
        print(f"   {gas}: {count:,}")

    conn.close()

def main():
    """Main import execution."""
    print_separator()
    print("EMISSION FACTORS DATABASE IMPORT")
    print("Import All 1,000 Emission Factors into Production Database")
    print_separator()
    print(f"Start time: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Database path
    db_path = str(project_root / "greenlang" / "data" / "emission_factors.db")
    data_dir = project_root / "data"

    # Define all YAML files in order
    yaml_files = [
        ("emission_factors_registry.yaml", 78),
        ("emission_factors_expansion_phase1.yaml", 114),
        ("emission_factors_expansion_phase2.yaml", 308),
        ("emission_factors_expansion_phase3_manufacturing_fuels.yaml", 70),
        ("emission_factors_expansion_phase3b_grids_industry.yaml", 175),
        ("emission_factors_expansion_phase4.yaml", 255),
    ]

    print("YAML FILES TO IMPORT:")
    total_expected = 0
    for i, (filename, count) in enumerate(yaml_files, 1):
        filepath = data_dir / filename
        exists = "✓" if filepath.exists() else "✗"
        print(f"  {i}. {exists} {filename} ({count} factors)")
        total_expected += count
    print(f"\nTotal expected factors: {total_expected:,}")

    # Create database
    print_separator()
    print("STEP 1: CREATE DATABASE SCHEMA")
    print_separator()
    print(f"Database path: {db_path}")
    print("Overwriting existing database: Yes")

    create_database(db_path, overwrite=True)
    print("✓ Database schema created successfully")

    # Import each file
    print_separator()
    print("STEP 2: IMPORT YAML FILES")
    print_separator()

    importer = EmissionFactorImporter(db_path)
    importer.connect()

    import_results = []

    for i, (filename, expected_count) in enumerate(yaml_files, 1):
        yaml_path = str(data_dir / filename)

        print(f"\n[{i}/6] Importing: {filename}")
        print(f"      Expected: {expected_count} factors")

        if not Path(yaml_path).exists():
            print(f"      ERROR: File not found!")
            import_results.append({
                'file': filename,
                'status': 'ERROR',
                'message': 'File not found',
                'imported': 0
            })
            continue

        try:
            before_count = importer.stats.successful_imports
            importer.import_yaml_file(yaml_path)
            after_count = importer.stats.successful_imports
            imported = after_count - before_count

            print(f"      Imported: {imported} factors")
            if imported == expected_count:
                print(f"      Status: ✓ SUCCESS")
            else:
                print(f"      Status: ⚠ WARNING - Expected {expected_count}, got {imported}")

            import_results.append({
                'file': filename,
                'status': 'SUCCESS' if imported == expected_count else 'WARNING',
                'expected': expected_count,
                'imported': imported
            })

        except Exception as e:
            print(f"      Status: ✗ FAILED")
            print(f"      Error: {str(e)}")
            import_results.append({
                'file': filename,
                'status': 'FAILED',
                'message': str(e),
                'imported': 0
            })

    importer.close()

    # Print import summary
    print_separator()
    print("IMPORT SUMMARY")
    print_separator()
    print(f"\nTotal factors processed: {importer.stats.total_factors:,}")
    print(f"Successfully imported: {importer.stats.successful_imports:,}")
    print(f"Failed imports: {importer.stats.failed_imports:,}")
    print(f"Duplicate factors: {importer.stats.duplicate_factors:,}")
    print(f"Unique categories: {len(importer.stats.categories)}")
    print(f"Unique sources: {len(importer.stats.sources)}")

    if importer.stats.errors:
        print(f"\nErrors encountered: {len(importer.stats.errors)}")
        print("First 10 errors:")
        for error in importer.stats.errors[:10]:
            print(f"  - {error}")
        if len(importer.stats.errors) > 10:
            print(f"  ... and {len(importer.stats.errors) - 10} more")

    print("\nFile-by-file results:")
    for result in import_results:
        status_icon = "✓" if result['status'] == 'SUCCESS' else "⚠" if result['status'] == 'WARNING' else "✗"
        print(f"  {status_icon} {result['file']}: {result.get('imported', 0)} factors")

    # Run validation queries
    run_validation_queries(db_path)

    # Database info
    print_separator()
    print("DATABASE INFORMATION")
    print_separator()

    db_info = get_database_info(db_path)
    print(f"\nDatabase file: {db_info['file_path']}")
    print(f"File size: {db_info['file_size_mb']:.2f} MB")

    print("\nTables:")
    for table, count in db_info['tables'].items():
        print(f"  {table}: {count:,} rows")

    print(f"\nIndexes: {len(db_info['indexes'])}")
    print(f"Views: {len(db_info['views'])}")

    # Final validation
    print_separator()
    print("FINAL VALIDATION")
    print_separator()

    validation = validate_database(db_path)

    if validation['valid']:
        print("✓ Database validation: PASSED")
    else:
        print("✗ Database validation: FAILED")
        print("\nErrors:")
        for error in validation['errors']:
            print(f"  - {error}")

    if validation['warnings']:
        print("\nWarnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")

    print("\nStatistics:")
    for key, value in validation['statistics'].items():
        print(f"  {key}: {value:,}")

    # Final status
    print_separator()
    print("IMPORT COMPLETE")
    print_separator()
    print(f"End time: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success = importer.stats.successful_imports >= total_expected * 0.95  # 95% success rate

    if success:
        print("\n✓ IMPORT SUCCESSFUL")
        print(f"  Total factors in database: {importer.stats.successful_imports:,}")
        print(f"  Target: {total_expected:,}")
        sys.exit(0)
    else:
        print("\n✗ IMPORT INCOMPLETE")
        print(f"  Total factors in database: {importer.stats.successful_imports:,}")
        print(f"  Target: {total_expected:,}")
        print(f"  Missing: {total_expected - importer.stats.successful_imports:,}")
        sys.exit(1)

if __name__ == "__main__":
    main()
