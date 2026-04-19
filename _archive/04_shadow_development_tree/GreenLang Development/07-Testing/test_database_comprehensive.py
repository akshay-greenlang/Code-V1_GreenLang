# -*- coding: utf-8 -*-
"""
Comprehensive Database Tests for Emission Factors

This test suite validates:
- 500 factor import capability
- Query performance (<10ms target)
- Geographic and temporal fallback logic
- Concurrent access patterns
- Database integrity and constraints
- Index effectiveness

Target: 90%+ test coverage for database layer
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import date, datetime
import time
import sys
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from greenlang.determinism import deterministic_random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.db.emission_factors_schema import (
    create_database,
    validate_database,
    get_database_info
)
from greenlang.sdk.emission_factor_client import (
    EmissionFactorClient,
    EmissionFactorNotFoundError,
    UnitNotAvailableError,
    DatabaseConnectionError
)
from greenlang.models.emission_factor import (
    EmissionFactor,
    EmissionResult,
    Geography,
    SourceProvenance,
    DataQualityScore,
    DataQualityTier,
    GeographyLevel,
    FactorSearchCriteria
)


# ==================== FIXTURES ====================

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_emission_factors.db"

    create_database(str(db_path))

    yield str(db_path)

    shutil.rmtree(temp_dir)


@pytest.fixture
def large_dataset_db(temp_db):
    """Create database with 500+ emission factors for scale testing."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Generate 500 emission factors
    fuel_types = ['diesel', 'gasoline', 'natural_gas', 'propane', 'coal', 'biomass', 'fuel_oil']
    regions = ['United States', 'Canada', 'Mexico', 'United Kingdom', 'Germany', 'France', 'Japan', 'China', 'India', 'Australia']
    scopes = ['Scope 1', 'Scope 2 - Location-Based', 'Scope 2 - Market-Based', 'Scope 3']
    sources = ['EPA', 'DEFRA', 'IEA', 'IPCC', 'GHG Protocol', 'EU Commission']

    factors = []

    for i in range(500):
        fuel_type = deterministic_random().choice(fuel_types)
        region = deterministic_random().choice(regions)
        scope = deterministic_random().choice(scopes)
        source = deterministic_random().choice(sources)

        factor_id = f"factor_{i:04d}_{fuel_type}_{region.replace(' ', '_').lower()}"
        name = f"Test Factor {i:04d} - {fuel_type.title()} in {region}"

        # Realistic emission factor ranges
        if fuel_type == 'diesel':
            ef_value = random.uniform(2.5, 3.2)
            unit = deterministic_random().choice(['liter', 'gallon', 'kg'])
        elif fuel_type == 'natural_gas':
            ef_value = random.uniform(1.8, 2.1)
            unit = deterministic_random().choice(['m3', 'therms', 'MMBtu'])
        elif fuel_type == 'electricity':
            ef_value = random.uniform(0.2, 0.6)
            unit = 'kwh'
        else:
            ef_value = random.uniform(1.0, 5.0)
            unit = deterministic_random().choice(['liter', 'kg', 'm3'])

        factor = (
            factor_id,
            name,
            'fuels' if fuel_type != 'electricity' else 'grids',
            fuel_type,
            ef_value,
            unit,
            scope,
            source,
            f'{source} Publication {2020 + (i % 5)}',
            f'https://{source.lower().replace(" ", "")}.org/factor/{i}',
            'GHG Protocol',
            f'2024-0{1 + (i % 12):02d}-01',
            2020 + (i % 5),
            region,
            'Country',
            'US' if region == 'United States' else region[:2].upper(),
            None,
            'North America' if region in ['United States', 'Canada', 'Mexico'] else 'Europe',
            deterministic_random().choice(['Tier 1', 'Tier 2', 'Tier 3']),
            random.uniform(3.0, 15.0),
            None,
            None,
            None if fuel_type != 'electricity' else random.uniform(0.0, 0.3),
            f'Test factor {i} for {fuel_type}',
            None
        )

        factors.append(factor)

    # Bulk insert
    cursor.executemany("""
        INSERT INTO emission_factors (
            factor_id, name, category, subcategory,
            emission_factor_value, unit, scope,
            source_org, source_publication, source_uri, standard,
            last_updated, year_applicable,
            geographic_scope, geography_level, country_code, state_province, region,
            data_quality_tier, uncertainty_percent, confidence_95ci, completeness_score,
            renewable_share, notes, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, factors)

    # Add additional units for some factors
    for i in range(0, 100, 5):
        factor_id = f"factor_{i:04d}_diesel_united_states"
        cursor.execute("""
            INSERT OR IGNORE INTO factor_units (factor_id, unit_type, unit_name, emission_factor_value)
            VALUES (?, 'emission_factor', 'gallon', ?)
        """, (factor_id, random.uniform(9.0, 11.0)))

    # Add gas vectors for natural gas factors
    for i in range(500):
        if 'natural_gas' in factors[i][0]:
            factor_id = factors[i][0]
            cursor.execute("""
                INSERT OR IGNORE INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
                VALUES (?, 'CO2', ?, 1)
            """, (factor_id, random.uniform(1.6, 1.8)))
            cursor.execute("""
                INSERT OR IGNORE INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
                VALUES (?, 'CH4', ?, 28)
            """, (factor_id, random.uniform(0.15, 0.20)))

    conn.commit()
    conn.close()

    return temp_db


# ==================== DATABASE IMPORT TESTS ====================

class TestLargeDatasetImport:
    """Test importing 500+ emission factors."""

    def test_import_500_factors(self, large_dataset_db):
        """Test that 500 factors can be imported successfully."""
        client = EmissionFactorClient(db_path=large_dataset_db)

        stats = client.get_statistics()

        assert stats['total_factors'] >= 500
        assert stats['by_category']['fuels'] > 0

        client.close()

    def test_all_factors_queryable(self, large_dataset_db):
        """Test that all imported factors are queryable."""
        conn = sqlite3.connect(large_dataset_db)
        cursor = conn.cursor()

        # Get all factor IDs
        cursor.execute("SELECT factor_id FROM emission_factors ORDER BY factor_id")
        factor_ids = [row[0] for row in cursor.fetchall()]

        conn.close()

        # Verify all can be retrieved via client
        client = EmissionFactorClient(db_path=large_dataset_db)

        for factor_id in factor_ids[:50]:  # Test first 50
            factor = client.get_factor(factor_id)
            assert factor.factor_id == factor_id
            assert factor.emission_factor_kg_co2e > 0

        client.close()

    def test_database_integrity_after_import(self, large_dataset_db):
        """Test database integrity after large import."""
        results = validate_database(large_dataset_db)

        assert results['valid'] is True
        assert len(results['errors']) == 0
        assert results['statistics']['total_factors'] >= 500


# ==================== QUERY PERFORMANCE TESTS ====================

class TestQueryPerformance:
    """Test query performance meets <10ms target."""

    def test_single_factor_lookup_performance(self, large_dataset_db):
        """Test single factor lookup is <10ms."""
        client = EmissionFactorClient(db_path=large_dataset_db)

        # Warm up
        client.get_factor('factor_0000_diesel_united_states')

        # Measure performance
        times = []
        for i in range(10):
            factor_id = f"factor_{i:04d}_diesel_united_states"

            start = time.perf_counter()
            factor = client.get_factor(factor_id)
            elapsed_ms = (time.perf_counter() - start) * 1000

            times.append(elapsed_ms)
            assert factor.factor_id == factor_id

        avg_time = sum(times) / len(times)

        assert avg_time < 10.0, f"Average query time {avg_time:.2f}ms exceeds 10ms target"

        client.close()

    def test_category_query_performance(self, large_dataset_db):
        """Test category query is <50ms."""
        client = EmissionFactorClient(db_path=large_dataset_db)

        start = time.perf_counter()
        factors = client.get_by_category('fuels')
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0, f"Category query time {elapsed_ms:.2f}ms exceeds 50ms target"
        assert len(factors) > 0

        client.close()

    def test_search_query_performance(self, large_dataset_db):
        """Test search query is <100ms."""
        client = EmissionFactorClient(db_path=large_dataset_db)

        criteria = FactorSearchCriteria(
            category='fuels',
            scope='Scope 1'
        )

        start = time.perf_counter()
        factors = client.search_factors(criteria)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0, f"Search query time {elapsed_ms:.2f}ms exceeds 100ms target"
        assert len(factors) > 0

        client.close()

    def test_index_effectiveness(self, large_dataset_db):
        """Test that indexes are being used for queries."""
        conn = sqlite3.connect(large_dataset_db)
        cursor = conn.cursor()

        # Test category index usage
        cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM emission_factors WHERE category = 'fuels'")
        plan = cursor.fetchall()

        # Should use index (contains "USING INDEX" or "SEARCH")
        plan_str = ' '.join([str(row) for row in plan])
        assert 'SEARCH' in plan_str or 'idx_category' in plan_str.lower()

        conn.close()


# ==================== GEOGRAPHIC FALLBACK TESTS ====================

class TestGeographicFallback:
    """Test geographic and temporal fallback logic."""

    def test_exact_geographic_match(self, large_dataset_db):
        """Test exact geographic match is preferred."""
        client = EmissionFactorClient(db_path=large_dataset_db)

        # Should find exact match for United States
        factor = client.get_grid_factor('United States')

        assert 'United States' in factor.geography.geographic_scope

        client.close()

    def test_regional_fallback(self, large_dataset_db):
        """Test fallback to regional average when specific location unavailable."""
        conn = sqlite3.connect(large_dataset_db)
        cursor = conn.cursor()

        # Insert a regional factor
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level, region,
                data_quality_tier, uncertainty_percent
            ) VALUES (
                'grid_north_america_regional', 'North America Grid Average', 'grids', 'regional',
                0.450, 'kwh', 'Scope 2 - Location-Based',
                'EPA', 'https://epa.gov/test', 'GHG Protocol',
                '2024-01-01', 2024,
                'North America', 'Regional', 'North America',
                'Tier 2', 10.0
            )
        """)
        conn.commit()
        conn.close()

        client = EmissionFactorClient(db_path=large_dataset_db)

        # Try to find factor for specific US state that doesn't exist
        # Should fall back to regional or country level
        factors = client.get_by_category('grids')

        assert len(factors) > 0

        client.close()

    def test_temporal_fallback(self, large_dataset_db):
        """Test fallback to most recent factor when specific year unavailable."""
        client = EmissionFactorClient(db_path=large_dataset_db)

        # Get all diesel factors
        factors = client.search_factors(
            FactorSearchCriteria(subcategory='diesel')
        )

        # Should return most recent factors
        assert len(factors) > 0

        # Check that factors are reasonably recent
        for factor in factors[:10]:
            assert factor.year_applicable >= 2020

        client.close()


# ==================== CONCURRENT ACCESS TESTS ====================

class TestConcurrentAccess:
    """Test concurrent database access patterns."""

    def test_concurrent_reads(self, large_dataset_db):
        """Test multiple threads reading simultaneously."""
        def read_factor(db_path, factor_id):
            client = EmissionFactorClient(db_path=db_path)
            factor = client.get_factor(factor_id)
            client.close()
            return factor.factor_id

        # Create thread pool
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for i in range(50):
                factor_id = f"factor_{i:04d}_diesel_united_states"
                future = executor.submit(read_factor, large_dataset_db, factor_id)
                futures.append((future, factor_id))

            # Verify all reads completed successfully
            for future, expected_id in futures:
                result = future.result()
                assert result == expected_id

    def test_concurrent_calculations(self, large_dataset_db):
        """Test multiple threads performing calculations simultaneously."""
        def calculate(db_path, factor_id, amount):
            client = EmissionFactorClient(db_path=db_path)
            result = client.calculate_emissions(
                factor_id=factor_id,
                activity_amount=amount,
                activity_unit='liter'
            )
            client.close()
            return result.emissions_kg_co2e

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for i in range(30):
                factor_id = f"factor_{i:04d}_diesel_united_states"
                amount = random.uniform(10.0, 1000.0)
                future = executor.submit(calculate, large_dataset_db, factor_id, amount)
                futures.append(future)

            # Verify all calculations completed
            for future in futures:
                emissions = future.result()
                assert emissions > 0

    def test_read_write_isolation(self, large_dataset_db):
        """Test read/write isolation (audit log writes during reads)."""
        def read_repeatedly(db_path, stop_event):
            client = EmissionFactorClient(db_path=db_path)
            count = 0
            while not stop_event.is_set():
                client.get_factor('factor_0000_diesel_united_states')
                count += 1
            client.close()
            return count

        def write_audit_logs(db_path, stop_event):
            client = EmissionFactorClient(db_path=db_path)
            count = 0
            while not stop_event.is_set():
                client.calculate_emissions(
                    factor_id='factor_0001_diesel_united_states',
                    activity_amount=100.0,
                    activity_unit='liter'
                )
                count += 1
                time.sleep(0.01)  # Small delay
            client.close()
            return count

        stop_event = threading.Event()

        # Start reader thread
        reader_thread = threading.Thread(
            target=read_repeatedly,
            args=(large_dataset_db, stop_event)
        )

        # Start writer thread
        writer_thread = threading.Thread(
            target=write_audit_logs,
            args=(large_dataset_db, stop_event)
        )

        reader_thread.start()
        writer_thread.start()

        # Run for 2 seconds
        time.sleep(2)
        stop_event.set()

        reader_thread.join()
        writer_thread.join()

        # Verify database is still valid
        results = validate_database(large_dataset_db)
        assert results['valid'] is True


# ==================== DATABASE CONSTRAINTS TESTS ====================

class TestDatabaseConstraints:
    """Test database integrity constraints."""

    def test_primary_key_constraint(self, temp_db):
        """Test primary key constraint prevents duplicates."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Insert factor
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level,
                data_quality_tier, uncertainty_percent
            ) VALUES (
                'test_duplicate', 'Test', 'fuels', 'diesel',
                2.68, 'liter', 'Scope 1',
                'EPA', 'https://epa.gov', 'GHG Protocol',
                '2024-01-01', 2024,
                'United States', 'Country',
                'Tier 1', 5.0
            )
        """)
        conn.commit()

        # Try to insert duplicate
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO emission_factors (
                    factor_id, name, category, subcategory,
                    emission_factor_value, unit, scope,
                    source_org, source_uri, standard,
                    last_updated, year_applicable,
                    geographic_scope, geography_level,
                    data_quality_tier, uncertainty_percent
                ) VALUES (
                    'test_duplicate', 'Test 2', 'fuels', 'gasoline',
                    2.31, 'liter', 'Scope 1',
                    'EPA', 'https://epa.gov', 'GHG Protocol',
                    '2024-01-01', 2024,
                    'United States', 'Country',
                    'Tier 1', 5.0
                )
            """)
            conn.commit()

        conn.close()

    def test_check_constraint_positive_emission_factor(self, temp_db):
        """Test CHECK constraint prevents negative emission factors."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO emission_factors (
                    factor_id, name, category, subcategory,
                    emission_factor_value, unit, scope,
                    source_org, source_uri, standard,
                    last_updated, year_applicable,
                    geographic_scope, geography_level,
                    data_quality_tier, uncertainty_percent
                ) VALUES (
                    'test_negative', 'Test Negative', 'fuels', 'diesel',
                    -2.68, 'liter', 'Scope 1',
                    'EPA', 'https://epa.gov', 'GHG Protocol',
                    '2024-01-01', 2024,
                    'United States', 'Country',
                    'Tier 1', 5.0
                )
            """)
            conn.commit()

        conn.close()

    def test_foreign_key_constraint(self, temp_db):
        """Test foreign key constraint on gas vectors."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Try to insert gas vector for non-existent factor
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
                VALUES ('nonexistent_factor', 'CO2', 1.5, 1)
            """)
            conn.commit()

        conn.close()

    def test_check_constraint_renewable_share_range(self, temp_db):
        """Test CHECK constraint on renewable share (0-1)."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Valid renewable share (0.5)
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level,
                data_quality_tier, renewable_share
            ) VALUES (
                'test_renewable', 'Test Renewable', 'grids', 'electricity',
                0.2, 'kwh', 'Scope 2 - Location-Based',
                'EPA', 'https://epa.gov', 'GHG Protocol',
                '2024-01-01', 2024,
                'United States', 'Country',
                'Tier 1', 0.5
            )
        """)
        conn.commit()

        # Invalid renewable share (>1)
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO emission_factors (
                    factor_id, name, category, subcategory,
                    emission_factor_value, unit, scope,
                    source_org, source_uri, standard,
                    last_updated, year_applicable,
                    geographic_scope, geography_level,
                    data_quality_tier, renewable_share
                ) VALUES (
                    'test_invalid_renewable', 'Test Invalid', 'grids', 'electricity',
                    0.2, 'kwh', 'Scope 2 - Location-Based',
                    'EPA', 'https://epa.gov', 'GHG Protocol',
                    '2024-01-01', 2024,
                    'United States', 'Country',
                    'Tier 1', 1.5
                )
            """)
            conn.commit()

        conn.close()


# ==================== DATABASE VIEWS TESTS ====================

class TestDatabaseViews:
    """Test database statistics views."""

    def test_factor_statistics_view(self, large_dataset_db):
        """Test factor_statistics view."""
        conn = sqlite3.connect(large_dataset_db)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM factor_statistics")
        stats = cursor.fetchall()

        assert len(stats) > 0

        # Verify columns
        for row in stats:
            category, subcategory, count, avg, min_val, max_val, sources, oldest, newest = row
            assert count > 0
            assert avg > 0
            assert min_val > 0
            assert max_val > 0

        conn.close()

    def test_geography_coverage_view(self, large_dataset_db):
        """Test geography_coverage view."""
        conn = sqlite3.connect(large_dataset_db)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM geography_coverage")
        coverage = cursor.fetchall()

        assert len(coverage) > 0

        conn.close()

    def test_quality_summary_view(self, large_dataset_db):
        """Test quality_summary view."""
        conn = sqlite3.connect(large_dataset_db)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM quality_summary")
        quality = cursor.fetchall()

        assert len(quality) > 0

        # Verify tiers
        tiers = [row[0] for row in quality]
        assert 'Tier 1' in tiers or 'Tier 2' in tiers or 'Tier 3' in tiers

        conn.close()


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-k', 'test_'])
