# -*- coding: utf-8 -*-
"""
Integration and End-to-End Tests

This test suite validates:
- End-to-end factor query → calculation → audit trail workflow
- Application integration (CSRD, VCCI, CBAM)
- Import pipeline validation (YAML → Database)
- Multi-factor calculations
- Cross-component data flow
- Real-world usage scenarios

Target: Comprehensive integration test coverage
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
import sys
import yaml
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.db.emission_factors_schema import create_database, validate_database
from greenlang.sdk.emission_factor_client import EmissionFactorClient
from greenlang.models.emission_factor import FactorSearchCriteria


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
def yaml_import_file(tmp_path):
    """Create sample YAML file for import testing."""
    yaml_data = {
        'emission_factors': [
            {
                'factor_id': 'diesel_us_2024',
                'name': 'Diesel Combustion US 2024',
                'category': 'fuels',
                'subcategory': 'diesel',
                'emission_factor_kg_co2e': 10.21,
                'unit': 'gallon',
                'scope': 'Scope 1',
                'source': {
                    'organization': 'EPA',
                    'publication': 'GHG Emission Factors Hub',
                    'uri': 'https://epa.gov/ghg',
                    'standard': 'GHG Protocol'
                },
                'geography': {
                    'scope': 'United States',
                    'level': 'Country',
                    'country_code': 'US'
                },
                'temporal': {
                    'year_applicable': 2024,
                    'last_updated': '2024-01-01'
                },
                'data_quality': {
                    'tier': 'Tier 1',
                    'uncertainty_percent': 5.0
                },
                'gas_vectors': [
                    {'gas_type': 'CO2', 'kg_per_unit': 10.15, 'gwp': 1},
                    {'gas_type': 'CH4', 'kg_per_unit': 0.0004, 'gwp': 28},
                    {'gas_type': 'N2O', 'kg_per_unit': 0.0002, 'gwp': 265}
                ]
            },
            {
                'factor_id': 'electricity_us_avg_2024',
                'name': 'US Grid Average 2024',
                'category': 'grids',
                'subcategory': 'us_average',
                'emission_factor_kg_co2e': 0.385,
                'unit': 'kwh',
                'scope': 'Scope 2 - Location-Based',
                'source': {
                    'organization': 'EPA eGRID',
                    'uri': 'https://epa.gov/egrid',
                    'standard': 'GHG Protocol'
                },
                'geography': {
                    'scope': 'United States',
                    'level': 'Country',
                    'country_code': 'US'
                },
                'temporal': {
                    'year_applicable': 2024,
                    'last_updated': '2024-01-01'
                },
                'data_quality': {
                    'tier': 'Tier 2',
                    'uncertainty_percent': 8.0
                },
                'renewable_share': 0.20
            }
        ]
    }

    yaml_file = tmp_path / "test_factors.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_data, f)

    return yaml_file


# ==================== END-TO-END WORKFLOW TESTS ====================

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_query_calculate_audit_workflow(self, temp_db):
        """Test complete workflow: query factor → calculate → verify audit."""
        # Step 1: Insert test data
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level, country_code,
                data_quality_tier, uncertainty_percent
            ) VALUES (
                'diesel_us_2024', 'Diesel US 2024', 'fuels', 'diesel',
                10.21, 'gallon', 'Scope 1',
                'EPA', 'https://epa.gov', 'GHG Protocol',
                '2024-01-01', 2024,
                'United States', 'Country', 'US',
                'Tier 1', 5.0
            )
        """)
        conn.commit()
        conn.close()

        # Step 2: Query factor
        client = EmissionFactorClient(db_path=temp_db)
        factor = client.get_factor('diesel_us_2024')

        assert factor.factor_id == 'diesel_us_2024'
        assert factor.emission_factor_kg_co2e == 10.21

        # Step 3: Calculate emissions
        result = client.calculate_emissions(
            factor_id='diesel_us_2024',
            activity_amount=1000.0,
            activity_unit='gallon'
        )

        assert result.emissions_kg_co2e == 10210.0
        assert result.emissions_metric_tons_co2e == 10.21

        # Step 4: Verify audit trail
        assert result.audit_trail is not None
        assert len(result.audit_trail) == 64  # SHA-256

        # Step 5: Verify audit log in database
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT factor_id, activity_amount, emissions_kg_co2e
            FROM calculation_audit_log
            WHERE factor_id = 'diesel_us_2024'
            ORDER BY calculation_timestamp DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

        assert row[0] == 'diesel_us_2024'
        assert row[1] == 1000.0
        assert row[2] == 10210.0

        conn.close()
        client.close()

    def test_multi_factor_calculation_workflow(self, temp_db):
        """Test calculating emissions for multiple fuel types."""
        # Insert multiple factors
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        factors_data = [
            ('diesel_us_2024', 'Diesel', 10.21, 'gallon'),
            ('gasoline_us_2024', 'Gasoline', 8.78, 'gallon'),
            ('natural_gas_us_2024', 'Natural Gas', 5.30, 'therms')
        ]

        for factor_id, name, ef_value, unit in factors_data:
            cursor.execute("""
                INSERT INTO emission_factors (
                    factor_id, name, category, subcategory,
                    emission_factor_value, unit, scope,
                    source_org, source_uri, standard,
                    last_updated, year_applicable,
                    geographic_scope, geography_level,
                    data_quality_tier
                ) VALUES (?, ?, 'fuels', ?, ?, ?, 'Scope 1',
                    'EPA', 'https://epa.gov', 'GHG Protocol',
                    '2024-01-01', 2024,
                    'United States', 'Country',
                    'Tier 1')
            """, (factor_id, name, name.split()[0].lower(), ef_value, unit))

        conn.commit()
        conn.close()

        # Calculate emissions for each
        client = EmissionFactorClient(db_path=temp_db)

        calculations = [
            ('diesel_us_2024', 500.0, 'gallon'),
            ('gasoline_us_2024', 300.0, 'gallon'),
            ('natural_gas_us_2024', 1000.0, 'therms')
        ]

        total_emissions = 0.0
        results = []

        for factor_id, amount, unit in calculations:
            result = client.calculate_emissions(factor_id, amount, unit)
            total_emissions += result.emissions_kg_co2e
            results.append(result)

        # Verify results
        assert len(results) == 3
        assert results[0].emissions_kg_co2e == 10.21 * 500  # Diesel
        assert results[1].emissions_kg_co2e == 8.78 * 300   # Gasoline
        assert results[2].emissions_kg_co2e == 5.30 * 1000  # Natural Gas

        # Total emissions
        expected_total = (10.21 * 500) + (8.78 * 300) + (5.30 * 1000)
        assert abs(total_emissions - expected_total) < 0.01

        client.close()


# ==================== YAML IMPORT INTEGRATION TESTS ====================

class TestYAMLImport:
    """Test YAML import pipeline integration."""

    def test_import_yaml_factors(self, temp_db, yaml_import_file):
        """Test importing factors from YAML file."""
        # Load YAML
        with open(yaml_import_file, 'r') as f:
            data = yaml.safe_load(f)

        factors = data['emission_factors']

        # Import to database
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        for factor_data in factors:
            # Insert main factor
            cursor.execute("""
                INSERT INTO emission_factors (
                    factor_id, name, category, subcategory,
                    emission_factor_value, unit, scope,
                    source_org, source_uri, standard,
                    last_updated, year_applicable,
                    geographic_scope, geography_level, country_code,
                    data_quality_tier, uncertainty_percent,
                    renewable_share
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                factor_data['factor_id'],
                factor_data['name'],
                factor_data['category'],
                factor_data['subcategory'],
                factor_data['emission_factor_kg_co2e'],
                factor_data['unit'],
                factor_data['scope'],
                factor_data['source']['organization'],
                factor_data['source']['uri'],
                factor_data['source']['standard'],
                factor_data['temporal']['last_updated'],
                factor_data['temporal']['year_applicable'],
                factor_data['geography']['scope'],
                factor_data['geography']['level'],
                factor_data['geography']['country_code'],
                factor_data['data_quality']['tier'],
                factor_data['data_quality']['uncertainty_percent'],
                factor_data.get('renewable_share')
            ))

            # Insert gas vectors if present
            if 'gas_vectors' in factor_data:
                for gas_vector in factor_data['gas_vectors']:
                    cursor.execute("""
                        INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
                        VALUES (?, ?, ?, ?)
                    """, (
                        factor_data['factor_id'],
                        gas_vector['gas_type'],
                        gas_vector['kg_per_unit'],
                        gas_vector['gwp']
                    ))

        conn.commit()
        conn.close()

        # Verify import
        client = EmissionFactorClient(db_path=temp_db)

        # Check diesel factor
        diesel = client.get_factor('diesel_us_2024')
        assert diesel.name == 'Diesel Combustion US 2024'
        assert diesel.emission_factor_kg_co2e == 10.21
        assert len(diesel.gas_vectors) == 3

        # Check electricity factor
        electricity = client.get_factor('electricity_us_avg_2024')
        assert electricity.emission_factor_kg_co2e == 0.385
        assert electricity.renewable_share == 0.20

        client.close()

    def test_yaml_import_validation(self, temp_db, yaml_import_file):
        """Test that imported data passes validation."""
        # Import data (using same logic as above)
        with open(yaml_import_file, 'r') as f:
            data = yaml.safe_load(f)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        for factor_data in data['emission_factors']:
            cursor.execute("""
                INSERT INTO emission_factors (
                    factor_id, name, category, subcategory,
                    emission_factor_value, unit, scope,
                    source_org, source_uri, standard,
                    last_updated, year_applicable,
                    geographic_scope, geography_level, country_code,
                    data_quality_tier, uncertainty_percent,
                    renewable_share
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                factor_data['factor_id'],
                factor_data['name'],
                factor_data['category'],
                factor_data['subcategory'],
                factor_data['emission_factor_kg_co2e'],
                factor_data['unit'],
                factor_data['scope'],
                factor_data['source']['organization'],
                factor_data['source']['uri'],
                factor_data['source']['standard'],
                factor_data['temporal']['last_updated'],
                factor_data['temporal']['year_applicable'],
                factor_data['geography']['scope'],
                factor_data['geography']['level'],
                factor_data['geography']['country_code'],
                factor_data['data_quality']['tier'],
                factor_data['data_quality']['uncertainty_percent'],
                factor_data.get('renewable_share')
            ))

        conn.commit()
        conn.close()

        # Validate database
        results = validate_database(temp_db)

        assert results['valid'] is True
        assert len(results['errors']) == 0
        assert results['statistics']['total_factors'] == 2


# ==================== APPLICATION INTEGRATION TESTS ====================

class TestApplicationIntegration:
    """Test integration with GreenLang applications (CSRD, VCCI, CBAM)."""

    def test_csrd_reporting_workflow(self, temp_db):
        """Test CSRD reporting workflow (Scope 1+2 emissions)."""
        # Setup: Insert factors for Scope 1 and Scope 2
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Scope 1: Diesel
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level,
                data_quality_tier
            ) VALUES (
                'diesel_eu_2024', 'Diesel EU 2024', 'fuels', 'diesel',
                2.68, 'liter', 'Scope 1',
                'EU Commission', 'https://ec.europa.eu', 'GHG Protocol',
                '2024-01-01', 2024,
                'European Union', 'Regional',
                'Tier 1'
            )
        """)

        # Scope 2: Electricity
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level,
                data_quality_tier
            ) VALUES (
                'electricity_eu_2024', 'EU Grid Average 2024', 'grids', 'eu_average',
                0.233, 'kwh', 'Scope 2 - Location-Based',
                'IEA', 'https://iea.org', 'GHG Protocol',
                '2024-01-01', 2024,
                'European Union', 'Regional',
                'Tier 2'
            )
        """)

        conn.commit()
        conn.close()

        # CSRD Calculation Workflow
        client = EmissionFactorClient(db_path=temp_db)

        # Company data (example)
        company_data = {
            'diesel_consumption_liters': 50000,
            'electricity_consumption_kwh': 1000000
        }

        # Calculate Scope 1
        scope1_result = client.calculate_emissions(
            factor_id='diesel_eu_2024',
            activity_amount=company_data['diesel_consumption_liters'],
            activity_unit='liter'
        )

        # Calculate Scope 2
        scope2_result = client.calculate_emissions(
            factor_id='electricity_eu_2024',
            activity_amount=company_data['electricity_consumption_kwh'],
            activity_unit='kwh'
        )

        # Total emissions
        total_scope1 = scope1_result.emissions_metric_tons_co2e
        total_scope2 = scope2_result.emissions_metric_tons_co2e
        total_emissions = total_scope1 + total_scope2

        # CSRD Report
        csrd_report = {
            'reporting_year': 2024,
            'scope1_emissions_tonnes_co2e': total_scope1,
            'scope2_location_based_tonnes_co2e': total_scope2,
            'total_scope1_and_2_tonnes_co2e': total_emissions,
            'calculations': [
                {
                    'scope': 'Scope 1',
                    'activity': 'Diesel Combustion',
                    'amount': company_data['diesel_consumption_liters'],
                    'unit': 'liters',
                    'emissions_tonnes_co2e': total_scope1,
                    'factor_id': 'diesel_eu_2024',
                    'audit_trail': scope1_result.audit_trail
                },
                {
                    'scope': 'Scope 2 - Location-Based',
                    'activity': 'Electricity Consumption',
                    'amount': company_data['electricity_consumption_kwh'],
                    'unit': 'kWh',
                    'emissions_tonnes_co2e': total_scope2,
                    'factor_id': 'electricity_eu_2024',
                    'audit_trail': scope2_result.audit_trail
                }
            ]
        }

        # Verify report
        assert csrd_report['scope1_emissions_tonnes_co2e'] == 50000 * 2.68 / 1000
        assert csrd_report['scope2_location_based_tonnes_co2e'] == 1000000 * 0.233 / 1000
        assert len(csrd_report['calculations']) == 2

        client.close()

    def test_cbam_import_calculation(self, temp_db):
        """Test CBAM (Carbon Border Adjustment Mechanism) import calculation."""
        # Setup: Insert embedded emissions factor
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level, country_code,
                data_quality_tier
            ) VALUES (
                'steel_cn_2024', 'Steel Production China 2024', 'materials', 'steel',
                1.85, 'kg', 'Scope 1',
                'CBAM Registry', 'https://cbam.ec.europa.eu', 'EU CBAM',
                '2024-01-01', 2024,
                'China', 'Country', 'CN',
                'Tier 3'
            )
        """)

        conn.commit()
        conn.close()

        # CBAM Import Scenario
        client = EmissionFactorClient(db_path=temp_db)

        import_data = {
            'product': 'Steel',
            'weight_tonnes': 500,
            'origin_country': 'China',
            'hs_code': '7208.10'
        }

        # Calculate embedded emissions
        result = client.calculate_emissions(
            factor_id='steel_cn_2024',
            activity_amount=import_data['weight_tonnes'] * 1000,  # Convert to kg
            activity_unit='kg'
        )

        # CBAM Declaration
        cbam_declaration = {
            'import_id': 'CBAM-2024-001',
            'product': import_data['product'],
            'weight_tonnes': import_data['weight_tonnes'],
            'embedded_emissions_tonnes_co2e': result.emissions_metric_tons_co2e,
            'emission_factor_kg_co2e_per_kg': 1.85,
            'origin_country': import_data['origin_country'],
            'data_quality_tier': 'Tier 3',
            'verification_hash': result.audit_trail
        }

        # Verify calculation
        assert cbam_declaration['embedded_emissions_tonnes_co2e'] == 500 * 1.85
        assert cbam_declaration['verification_hash'] is not None

        client.close()


# ==================== SEARCH AND FILTER INTEGRATION TESTS ====================

class TestSearchIntegration:
    """Test search and filter functionality integration."""

    def test_complex_search_criteria(self, temp_db):
        """Test searching with multiple criteria."""
        # Insert diverse factors
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        test_factors = [
            ('diesel_us_2024', 'Diesel US', 'fuels', 'diesel', 10.21, 'gallon', 'Scope 1', 'US', 'Tier 1'),
            ('diesel_uk_2024', 'Diesel UK', 'fuels', 'diesel', 2.68, 'liter', 'Scope 1', 'UK', 'Tier 1'),
            ('gasoline_us_2024', 'Gasoline US', 'fuels', 'gasoline', 8.78, 'gallon', 'Scope 1', 'US', 'Tier 1'),
            ('electricity_us_2024', 'Electricity US', 'grids', 'us_average', 0.385, 'kwh', 'Scope 2 - Location-Based', 'US', 'Tier 2'),
        ]

        for factor_id, name, category, subcategory, ef_value, unit, scope, country, tier in test_factors:
            cursor.execute("""
                INSERT INTO emission_factors (
                    factor_id, name, category, subcategory,
                    emission_factor_value, unit, scope,
                    source_org, source_uri, standard,
                    last_updated, year_applicable,
                    geographic_scope, country_code,
                    data_quality_tier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'Test', 'https://test.com', 'GHG Protocol',
                    '2024-01-01', 2024, ?, ?, ?)
            """, (factor_id, name, category, subcategory, ef_value, unit, scope, country, country, tier))

        conn.commit()
        conn.close()

        # Complex search scenarios
        client = EmissionFactorClient(db_path=temp_db)

        # Scenario 1: All US fuels
        criteria1 = FactorSearchCriteria(
            category='fuels',
            country_code='US'
        )
        results1 = client.search_factors(criteria1)
        assert len(results1) == 2  # Diesel and Gasoline

        # Scenario 2: Only diesel factors
        criteria2 = FactorSearchCriteria(subcategory='diesel')
        results2 = client.search_factors(criteria2)
        assert len(results2) == 2  # US and UK diesel

        # Scenario 3: Scope 1 factors only
        criteria3 = FactorSearchCriteria(scope='Scope 1')
        results3 = client.search_factors(criteria3)
        assert len(results3) == 3  # All fuels

        # Scenario 4: Tier 1 quality factors
        criteria4 = FactorSearchCriteria(data_quality_tier='Tier 1')
        results4 = client.search_factors(criteria4)
        assert len(results4) == 3  # All fuels (not electricity)

        client.close()


# ==================== BATCH PROCESSING INTEGRATION TESTS ====================

class TestBatchProcessingIntegration:
    """Test batch processing integration."""

    def test_batch_calculation_workflow(self, temp_db):
        """Test batch calculation for multiple records."""
        # Setup factors
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, data_quality_tier
            ) VALUES (
                'diesel_us_2024', 'Diesel US', 'fuels', 'diesel',
                10.21, 'gallon', 'Scope 1',
                'EPA', 'https://epa.gov', 'GHG Protocol',
                '2024-01-01', 2024,
                'United States', 'Tier 1'
            )
        """)

        conn.commit()
        conn.close()

        # Batch data (e.g., fleet fuel consumption)
        batch_data = [
            {'vehicle_id': 'VEH-001', 'fuel_type': 'diesel', 'gallons': 150},
            {'vehicle_id': 'VEH-002', 'fuel_type': 'diesel', 'gallons': 200},
            {'vehicle_id': 'VEH-003', 'fuel_type': 'diesel', 'gallons': 175},
            {'vehicle_id': 'VEH-004', 'fuel_type': 'diesel', 'gallons': 180},
            {'vehicle_id': 'VEH-005', 'fuel_type': 'diesel', 'gallons': 165},
        ]

        # Process batch
        client = EmissionFactorClient(db_path=temp_db)

        batch_results = []
        total_emissions = 0.0

        for record in batch_data:
            result = client.calculate_emissions(
                factor_id='diesel_us_2024',
                activity_amount=record['gallons'],
                activity_unit='gallon'
            )

            batch_results.append({
                'vehicle_id': record['vehicle_id'],
                'fuel_gallons': record['gallons'],
                'emissions_kg_co2e': result.emissions_kg_co2e,
                'emissions_tonnes_co2e': result.emissions_metric_tons_co2e,
                'audit_trail': result.audit_trail
            })

            total_emissions += result.emissions_kg_co2e

        # Verify batch results
        assert len(batch_results) == 5
        assert total_emissions == sum(r['gallons'] for r in batch_data) * 10.21

        # Verify each vehicle calculation
        for i, result in enumerate(batch_results):
            expected = batch_data[i]['gallons'] * 10.21
            assert result['emissions_kg_co2e'] == expected

        client.close()


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
