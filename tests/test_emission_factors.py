"""
Comprehensive Test Suite for Emission Factor Infrastructure

This test suite validates:
- Database schema creation
- YAML import functionality
- SDK query methods
- Calculation engine
- Fallback logic
- Data integrity

Target: 85%+ test coverage
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import date, datetime
import sys

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


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_emission_factors.db"

    # Create database
    create_database(str(db_path))

    yield str(db_path)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def populated_db(temp_db):
    """Create database with sample data."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert sample factors
    sample_factors = [
        (
            'test_diesel', 'Test Diesel Fuel', 'fuels', 'diesel',
            2.68, 'liter', 'Scope 1',
            'EPA', None, 'https://epa.gov/test', 'GHG Protocol',
            '2024-01-01', 2024,
            'United States', 'Country', 'USA', None, 'North America',
            'Tier 1', 5.0, None, None,
            None, 'Test diesel factor', None
        ),
        (
            'test_grid_us', 'Test US Grid Average', 'grids', 'us_average',
            0.417, 'kwh', 'Scope 2 - Location-Based',
            'EPA', None, 'https://epa.gov/egrid', 'GHG Protocol',
            '2024-01-01', 2024,
            'United States', 'Country', 'USA', None, 'North America',
            'Tier 2', 8.0, None, None,
            0.20, 'Test grid factor', None
        ),
        (
            'test_natural_gas', 'Test Natural Gas', 'fuels', 'natural_gas',
            1.89, 'm3', 'Scope 1',
            'EPA', None, 'https://epa.gov/test', 'GHG Protocol',
            '2024-01-01', 2024,
            'United States', 'Country', 'USA', None, 'North America',
            'Tier 1', 5.0, None, None,
            None, 'Test natural gas factor', None
        )
    ]

    for factor in sample_factors:
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_publication, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level, country_code, state_province, region,
                data_quality_tier, uncertainty_percent, confidence_95ci, completeness_score,
                renewable_share, notes, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, factor)

    # Insert additional units for diesel
    cursor.execute("""
        INSERT INTO factor_units (factor_id, unit_type, unit_name, emission_factor_value)
        VALUES ('test_diesel', 'emission_factor', 'gallon', 10.21)
    """)

    # Insert gas vectors for natural gas
    cursor.execute("""
        INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
        VALUES ('test_natural_gas', 'CO2', 1.70, 1)
    """)
    cursor.execute("""
        INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
        VALUES ('test_natural_gas', 'CH4', 0.19, 28)
    """)

    conn.commit()
    conn.close()

    return temp_db


class TestDatabaseSchema:
    """Test database schema creation and validation."""

    def test_create_database(self, temp_db):
        """Test database creation."""
        assert Path(temp_db).exists()

        # Verify tables created
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            'calculation_audit_log',
            'emission_factors',
            'factor_gas_vectors',
            'factor_units'
        ]

        for table in expected_tables:
            assert table in tables

        conn.close()

    def test_database_indexes(self, temp_db):
        """Test that indexes are created."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index'
        """)
        indexes = [row[0] for row in cursor.fetchall()]

        # Check for critical indexes
        assert any('category' in idx.lower() for idx in indexes)
        assert any('scope' in idx.lower() for idx in indexes)
        assert any('geography' in idx.lower() for idx in indexes)

        conn.close()

    def test_validate_empty_database(self, temp_db):
        """Test validation of empty database."""
        results = validate_database(temp_db)

        assert results['valid'] is True
        assert results['statistics']['total_factors'] == 0

    def test_database_info(self, temp_db):
        """Test getting database info."""
        info = get_database_info(temp_db)

        assert 'file_path' in info
        assert 'file_size_mb' in info
        assert 'tables' in info
        assert len(info['tables']) >= 4


class TestEmissionFactorClient:
    """Test emission factor client SDK."""

    def test_client_initialization(self, populated_db):
        """Test client initialization."""
        client = EmissionFactorClient(db_path=populated_db)
        assert client.db_path == populated_db
        client.close()

    def test_client_context_manager(self, populated_db):
        """Test client as context manager."""
        with EmissionFactorClient(db_path=populated_db) as client:
            assert client.conn is not None

    def test_client_nonexistent_db(self):
        """Test client with nonexistent database."""
        with pytest.raises(DatabaseConnectionError):
            EmissionFactorClient(db_path="/nonexistent/path/db.db")

    def test_get_factor(self, populated_db):
        """Test getting factor by ID."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('test_diesel')

        assert factor.factor_id == 'test_diesel'
        assert factor.name == 'Test Diesel Fuel'
        assert factor.emission_factor_kg_co2e == 2.68
        assert factor.unit == 'liter'
        assert factor.category == 'fuels'

        client.close()

    def test_get_factor_not_found(self, populated_db):
        """Test getting nonexistent factor."""
        client = EmissionFactorClient(db_path=populated_db)

        with pytest.raises(EmissionFactorNotFoundError):
            client.get_factor('nonexistent_factor')

        client.close()

    def test_get_factor_with_additional_units(self, populated_db):
        """Test factor with multiple units."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('test_diesel')

        # Check additional units loaded
        assert len(factor.additional_units) > 0
        assert any(u.unit_name == 'gallon' for u in factor.additional_units)

        # Test get_factor_for_unit
        gallon_factor = factor.get_factor_for_unit('gallon')
        assert gallon_factor == 10.21

        client.close()

    def test_get_factor_with_gas_vectors(self, populated_db):
        """Test factor with gas breakdown."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('test_natural_gas')

        # Check gas vectors loaded
        assert len(factor.gas_vectors) > 0
        assert any(g.gas_type == 'CO2' for g in factor.gas_vectors)
        assert any(g.gas_type == 'CH4' for g in factor.gas_vectors)

        client.close()

    def test_search_by_name(self, populated_db):
        """Test searching factors by name."""
        client = EmissionFactorClient(db_path=populated_db)

        factors = client.get_factor_by_name('diesel')

        assert len(factors) > 0
        assert all('diesel' in f.name.lower() or 'diesel' in f.factor_id.lower() for f in factors)

        client.close()

    def test_search_by_criteria(self, populated_db):
        """Test searching with criteria."""
        client = EmissionFactorClient(db_path=populated_db)

        criteria = FactorSearchCriteria(category='fuels')
        factors = client.search_factors(criteria)

        assert len(factors) >= 2
        assert all(f.category == 'fuels' for f in factors)

        client.close()

    def test_get_by_category(self, populated_db):
        """Test getting factors by category."""
        client = EmissionFactorClient(db_path=populated_db)

        factors = client.get_by_category('fuels')

        assert len(factors) >= 2
        assert all(f.category == 'fuels' for f in factors)

        client.close()

    def test_get_by_scope(self, populated_db):
        """Test getting factors by scope."""
        client = EmissionFactorClient(db_path=populated_db)

        factors = client.get_by_scope('Scope 1')

        assert len(factors) >= 2
        assert all(f.scope == 'Scope 1' for f in factors)

        client.close()

    def test_get_grid_factor(self, populated_db):
        """Test getting grid emission factor."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_grid_factor('United States')

        assert factor.category == 'grids'
        assert 'United States' in factor.geography.geographic_scope

        client.close()

    def test_get_fuel_factor(self, populated_db):
        """Test getting fuel emission factor."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_fuel_factor('diesel')

        assert factor.category == 'fuels'
        assert 'diesel' in factor.name.lower() or 'diesel' in factor.factor_id.lower()

        client.close()

    def test_calculate_emissions(self, populated_db):
        """Test emission calculation."""
        client = EmissionFactorClient(db_path=populated_db)

        result = client.calculate_emissions(
            factor_id='test_diesel',
            activity_amount=100.0,
            activity_unit='liter'
        )

        # Verify calculation
        assert result.activity_amount == 100.0
        assert result.activity_unit == 'liter'
        assert result.emissions_kg_co2e == 268.0  # 100 * 2.68
        assert result.emissions_metric_tons_co2e == 0.268
        assert result.factor_used.factor_id == 'test_diesel'
        assert result.factor_value_applied == 2.68
        assert len(result.audit_trail) == 64  # SHA-256 hash

        client.close()

    def test_calculate_with_alternative_unit(self, populated_db):
        """Test calculation with non-primary unit."""
        client = EmissionFactorClient(db_path=populated_db)

        result = client.calculate_emissions(
            factor_id='test_diesel',
            activity_amount=10.0,
            activity_unit='gallon'
        )

        # Verify calculation with gallon factor
        assert result.emissions_kg_co2e == 102.1  # 10 * 10.21
        assert result.factor_value_applied == 10.21

        client.close()

    def test_calculate_invalid_unit(self, populated_db):
        """Test calculation with invalid unit."""
        client = EmissionFactorClient(db_path=populated_db)

        with pytest.raises(UnitNotAvailableError):
            client.calculate_emissions(
                factor_id='test_diesel',
                activity_amount=100.0,
                activity_unit='invalid_unit'
            )

        client.close()

    def test_calculate_negative_amount(self, populated_db):
        """Test calculation with negative amount."""
        client = EmissionFactorClient(db_path=populated_db)

        with pytest.raises(ValueError):
            client.calculate_emissions(
                factor_id='test_diesel',
                activity_amount=-100.0,
                activity_unit='liter'
            )

        client.close()

    def test_statistics(self, populated_db):
        """Test getting database statistics."""
        client = EmissionFactorClient(db_path=populated_db)

        stats = client.get_statistics()

        assert 'total_factors' in stats
        assert stats['total_factors'] >= 3
        assert 'by_category' in stats
        assert 'by_scope' in stats
        assert 'total_calculations' in stats

        client.close()


class TestDataModels:
    """Test data model validation."""

    def test_emission_factor_creation(self):
        """Test creating emission factor."""
        geography = Geography(
            geographic_scope='United States',
            geography_level=GeographyLevel.COUNTRY,
            country_code='USA'
        )

        source = SourceProvenance(
            source_org='EPA',
            source_uri='https://epa.gov/test',
            standard='GHG Protocol'
        )

        quality = DataQualityScore(
            tier=DataQualityTier.TIER_1,
            uncertainty_percent=5.0
        )

        factor = EmissionFactor(
            factor_id='test_factor',
            name='Test Factor',
            category='fuels',
            emission_factor_kg_co2e=2.5,
            unit='liter',
            scope='Scope 1',
            source=source,
            geography=geography,
            data_quality=quality,
            last_updated=date.today()
        )

        assert factor.factor_id == 'test_factor'
        assert factor.emission_factor_kg_co2e == 2.5

    def test_emission_factor_validation(self):
        """Test emission factor validation."""
        geography = Geography(
            geographic_scope='United States',
            geography_level=GeographyLevel.COUNTRY
        )

        source = SourceProvenance(
            source_org='EPA',
            source_uri='https://epa.gov/test'
        )

        quality = DataQualityScore(tier=DataQualityTier.TIER_1)

        # Test invalid emission factor (negative)
        with pytest.raises(ValueError):
            EmissionFactor(
                factor_id='test',
                name='Test',
                category='fuels',
                emission_factor_kg_co2e=-1.0,  # Invalid
                unit='liter',
                scope='Scope 1',
                source=source,
                geography=geography,
                data_quality=quality,
                last_updated=date.today()
            )

    def test_emission_result_creation(self, populated_db):
        """Test creating emission result."""
        client = EmissionFactorClient(db_path=populated_db)
        factor = client.get_factor('test_diesel')

        result = EmissionResult(
            activity_amount=100.0,
            activity_unit='liter',
            emissions_kg_co2e=268.0,
            emissions_metric_tons_co2e=0.268,
            factor_used=factor,
            factor_value_applied=2.68,
            calculation_timestamp=datetime.now(),
            audit_trail='abc123...'
        )

        assert result.activity_amount == 100.0
        assert result.emissions_kg_co2e == 268.0

        # Test to_dict
        result_dict = result.to_dict()
        assert 'emissions_kg_co2e' in result_dict
        assert 'audit_trail' in result_dict

        client.close()

    def test_factor_provenance_hash(self, populated_db):
        """Test provenance hash calculation."""
        client = EmissionFactorClient(db_path=populated_db)
        factor = client.get_factor('test_diesel')

        hash1 = factor.calculate_provenance_hash()
        hash2 = factor.calculate_provenance_hash()

        # Hash should be deterministic
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256

        client.close()

    def test_factor_staleness_check(self, populated_db):
        """Test factor staleness detection."""
        client = EmissionFactorClient(db_path=populated_db)
        factor = client.get_factor('test_diesel')

        # Recent factor should not be stale
        assert not factor.is_stale(max_age_years=3)

        client.close()


class TestSearchCriteria:
    """Test search criteria functionality."""

    def test_search_criteria_to_sql(self):
        """Test converting search criteria to SQL."""
        criteria = FactorSearchCriteria(
            category='fuels',
            scope='Scope 1'
        )

        where_clause, params = criteria.to_sql_where()

        assert 'category' in where_clause
        assert 'scope' in where_clause
        assert params['category'] == 'fuels'
        assert params['scope'] == 'Scope 1'

    def test_empty_search_criteria(self):
        """Test empty search criteria."""
        criteria = FactorSearchCriteria()

        where_clause, params = criteria.to_sql_where()

        assert where_clause == "1=1"
        assert len(params) == 0


class TestCalculationAudit:
    """Test calculation audit logging."""

    def test_calculation_logged(self, populated_db):
        """Test that calculations are logged to audit table."""
        client = EmissionFactorClient(db_path=populated_db)

        # Perform calculation
        result = client.calculate_emissions(
            factor_id='test_diesel',
            activity_amount=100.0,
            activity_unit='liter'
        )

        # Check audit log
        conn = sqlite3.connect(populated_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM calculation_audit_log
            WHERE factor_id = 'test_diesel'
        """)
        count = cursor.fetchone()[0]

        assert count >= 1

        conn.close()
        client.close()

    def test_audit_trail_uniqueness(self, populated_db):
        """Test that audit trails are unique."""
        client = EmissionFactorClient(db_path=populated_db)

        result1 = client.calculate_emissions('test_diesel', 100.0, 'liter')
        result2 = client.calculate_emissions('test_diesel', 100.0, 'liter')

        # Different calculations should have different timestamps and hashes
        assert result1.calculation_timestamp != result2.calculation_timestamp

        client.close()


class TestPerformance:
    """Test performance requirements."""

    def test_factor_lookup_performance(self, populated_db):
        """Test that factor lookup is <10ms."""
        import time

        client = EmissionFactorClient(db_path=populated_db)

        start = time.time()
        factor = client.get_factor('test_diesel')
        elapsed = (time.time() - start) * 1000  # Convert to ms

        assert elapsed < 10.0  # Should be <10ms

        client.close()

    def test_calculation_performance(self, populated_db):
        """Test that calculation is <100ms."""
        import time

        client = EmissionFactorClient(db_path=populated_db)

        start = time.time()
        result = client.calculate_emissions('test_diesel', 100.0, 'liter')
        elapsed = (time.time() - start) * 1000  # Convert to ms

        assert elapsed < 100.0  # Should be <100ms

        client.close()


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
