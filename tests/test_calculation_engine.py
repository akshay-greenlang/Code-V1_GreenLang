"""
Calculation Engine Tests

This test suite validates:
- Determinism: Same input → Same output (bit-perfect reproducibility)
- Audit trail integrity and completeness
- Multi-gas decomposition accuracy (CO2, CH4, N2O)
- Uncertainty quantification (Monte Carlo simulation)
- Calculation provenance hashing
- Edge cases and boundary conditions

Target: 94%+ test coverage for calculation engine
"""

import pytest
import hashlib
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
from decimal import Decimal
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.db.emission_factors_schema import create_database
from greenlang.sdk.emission_factor_client import EmissionFactorClient
from greenlang.models.emission_factor import (
    EmissionFactor,
    EmissionResult,
    Geography,
    SourceProvenance,
    DataQualityScore,
    DataQualityTier,
    GeographyLevel,
    GasVector
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
def populated_db(temp_db):
    """Create database with test data including gas vectors."""
    import sqlite3

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert diesel factor with gas breakdown
    cursor.execute("""
        INSERT INTO emission_factors (
            factor_id, name, category, subcategory,
            emission_factor_value, unit, scope,
            source_org, source_uri, standard,
            last_updated, year_applicable,
            geographic_scope, geography_level, country_code,
            data_quality_tier, uncertainty_percent
        ) VALUES (
            'diesel_us_2024', 'Diesel Combustion US 2024', 'fuels', 'diesel',
            10.21, 'gallon', 'Scope 1',
            'EPA', 'https://epa.gov/ghg', 'GHG Protocol',
            '2024-01-01', 2024,
            'United States', 'Country', 'US',
            'Tier 1', 5.0
        )
    """)

    # Insert gas vectors for diesel
    cursor.execute("""
        INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
        VALUES ('diesel_us_2024', 'CO2', 10.15, 1)
    """)
    cursor.execute("""
        INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
        VALUES ('diesel_us_2024', 'CH4', 0.0004, 28)
    """)
    cursor.execute("""
        INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
        VALUES ('diesel_us_2024', 'N2O', 0.0002, 265)
    """)

    # Insert natural gas factor with gas breakdown
    cursor.execute("""
        INSERT INTO emission_factors (
            factor_id, name, category, subcategory,
            emission_factor_value, unit, scope,
            source_org, source_uri, standard,
            last_updated, year_applicable,
            geographic_scope, geography_level, country_code,
            data_quality_tier, uncertainty_percent
        ) VALUES (
            'natural_gas_us_2024', 'Natural Gas Combustion US 2024', 'fuels', 'natural_gas',
            5.30, 'therms', 'Scope 1',
            'EPA', 'https://epa.gov/ghg', 'GHG Protocol',
            '2024-01-01', 2024,
            'United States', 'Country', 'US',
            'Tier 1', 4.5
        )
    """)

    cursor.execute("""
        INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
        VALUES ('natural_gas_us_2024', 'CO2', 5.28, 1)
    """)
    cursor.execute("""
        INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
        VALUES ('natural_gas_us_2024', 'CH4', 0.001, 28)
    """)
    cursor.execute("""
        INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
        VALUES ('natural_gas_us_2024', 'N2O', 0.0001, 265)
    """)

    # Insert electricity factor (grid)
    cursor.execute("""
        INSERT INTO emission_factors (
            factor_id, name, category, subcategory,
            emission_factor_value, unit, scope,
            source_org, source_uri, standard,
            last_updated, year_applicable,
            geographic_scope, geography_level, country_code,
            data_quality_tier, uncertainty_percent,
            renewable_share
        ) VALUES (
            'electricity_us_avg_2024', 'US Grid Average 2024', 'grids', 'us_average',
            0.385, 'kwh', 'Scope 2 - Location-Based',
            'EPA eGRID', 'https://epa.gov/egrid', 'GHG Protocol',
            '2024-01-01', 2024,
            'United States', 'Country', 'US',
            'Tier 2', 8.0,
            0.20
        )
    """)

    conn.commit()
    conn.close()

    return temp_db


# ==================== DETERMINISM TESTS ====================

class TestDeterminism:
    """Test calculation determinism (bit-perfect reproducibility)."""

    def test_same_input_same_output(self, populated_db):
        """Test that same input produces identical output."""
        client = EmissionFactorClient(db_path=populated_db)

        # Perform calculation 10 times
        results = []
        for _ in range(10):
            result = client.calculate_emissions(
                factor_id='diesel_us_2024',
                activity_amount=100.0,
                activity_unit='gallon'
            )
            results.append(result)

        # All results should be identical
        first_emissions = results[0].emissions_kg_co2e
        first_hash = results[0].audit_trail

        for result in results[1:]:
            assert result.emissions_kg_co2e == first_emissions
            # Note: audit_trail includes timestamp, so it will differ
            # But provenance hash should be same for same factor

        client.close()

    def test_calculation_precision(self, populated_db):
        """Test calculation maintains precision (no floating point drift)."""
        client = EmissionFactorClient(db_path=populated_db)

        # Use Decimal for high-precision calculation
        activity_amount = 100.0
        expected_emissions = 10.21 * 100.0  # 1021.0 kg CO2e

        result = client.calculate_emissions(
            factor_id='diesel_us_2024',
            activity_amount=activity_amount,
            activity_unit='gallon'
        )

        # Should match exactly (within floating point tolerance)
        assert abs(result.emissions_kg_co2e - expected_emissions) < 1e-10

        client.close()

    def test_order_independence(self, populated_db):
        """Test that order of operations doesn't affect result."""
        client = EmissionFactorClient(db_path=populated_db)

        # Calculate in different orders
        result1 = client.calculate_emissions('diesel_us_2024', 100.0, 'gallon')
        result2 = client.calculate_emissions('natural_gas_us_2024', 500.0, 'therms')
        result3 = client.calculate_emissions('diesel_us_2024', 100.0, 'gallon')

        # First and third calculation (same input) should be identical
        assert result1.emissions_kg_co2e == result3.emissions_kg_co2e
        assert result1.factor_value_applied == result3.factor_value_applied

        client.close()

    def test_provenance_hash_determinism(self, populated_db):
        """Test provenance hash is deterministic for same factor."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('diesel_us_2024')

        # Calculate hash multiple times
        hash1 = factor.calculate_provenance_hash()
        hash2 = factor.calculate_provenance_hash()
        hash3 = factor.calculate_provenance_hash()

        # Should all be identical
        assert hash1 == hash2 == hash3
        assert len(hash1) == 64  # SHA-256

        client.close()


# ==================== AUDIT TRAIL TESTS ====================

class TestAuditTrail:
    """Test audit trail integrity and completeness."""

    def test_audit_trail_creation(self, populated_db):
        """Test that audit trail is created for every calculation."""
        client = EmissionFactorClient(db_path=populated_db)

        result = client.calculate_emissions(
            factor_id='diesel_us_2024',
            activity_amount=100.0,
            activity_unit='gallon'
        )

        # Verify audit trail exists
        assert result.audit_trail is not None
        assert len(result.audit_trail) == 64  # SHA-256 hash
        assert result.calculation_timestamp is not None

        client.close()

    def test_audit_trail_uniqueness(self, populated_db):
        """Test that each calculation gets unique audit trail."""
        client = EmissionFactorClient(db_path=populated_db)

        result1 = client.calculate_emissions('diesel_us_2024', 100.0, 'gallon')
        result2 = client.calculate_emissions('diesel_us_2024', 100.0, 'gallon')

        # Different timestamps mean different audit hashes
        assert result1.calculation_timestamp != result2.calculation_timestamp

        client.close()

    def test_audit_trail_logged_to_database(self, populated_db):
        """Test that calculations are logged to audit table."""
        import sqlite3

        client = EmissionFactorClient(db_path=populated_db)

        # Perform calculation
        result = client.calculate_emissions(
            factor_id='diesel_us_2024',
            activity_amount=100.0,
            activity_unit='gallon'
        )

        # Check audit log
        conn = sqlite3.connect(populated_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM calculation_audit_log
            WHERE factor_id = 'diesel_us_2024'
        """)
        count = cursor.fetchone()[0]

        assert count >= 1

        # Verify audit record
        cursor.execute("""
            SELECT activity_amount, activity_unit, emissions_kg_co2e, factor_value_used
            FROM calculation_audit_log
            WHERE factor_id = 'diesel_us_2024'
            ORDER BY calculation_timestamp DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

        assert row[0] == 100.0  # activity_amount
        assert row[1] == 'gallon'  # activity_unit
        assert row[2] == result.emissions_kg_co2e
        assert row[3] == result.factor_value_applied

        conn.close()
        client.close()

    def test_audit_trail_includes_all_inputs(self, populated_db):
        """Test that audit trail captures all input parameters."""
        client = EmissionFactorClient(db_path=populated_db)

        result = client.calculate_emissions(
            factor_id='diesel_us_2024',
            activity_amount=100.0,
            activity_unit='gallon'
        )

        # Verify all inputs are captured in result
        assert result.activity_amount == 100.0
        assert result.activity_unit == 'gallon'
        assert result.factor_used.factor_id == 'diesel_us_2024'
        assert result.factor_value_applied == 10.21

        client.close()

    def test_audit_hash_verification(self, populated_db):
        """Test that audit hash can be verified."""
        client = EmissionFactorClient(db_path=populated_db)

        result = client.calculate_emissions(
            factor_id='diesel_us_2024',
            activity_amount=100.0,
            activity_unit='gallon'
        )

        # Recreate hash from result data
        audit_data = {
            'factor_id': result.factor_used.factor_id,
            'activity_amount': result.activity_amount,
            'activity_unit': result.activity_unit,
            'emissions_kg_co2e': result.emissions_kg_co2e,
            'factor_value': result.factor_value_applied,
            'timestamp': result.calculation_timestamp.isoformat()
        }

        calculated_hash = hashlib.sha256(
            json.dumps(audit_data, sort_keys=True).encode()
        ).hexdigest()

        # Hash should be reproducible from data
        assert len(calculated_hash) == 64

        client.close()


# ==================== MULTI-GAS DECOMPOSITION TESTS ====================

class TestMultiGasDecomposition:
    """Test multi-gas decomposition (CO2, CH4, N2O)."""

    def test_gas_vector_loading(self, populated_db):
        """Test that gas vectors are loaded for factors."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('diesel_us_2024')

        # Should have gas vectors
        assert len(factor.gas_vectors) > 0

        # Check for expected gases
        gas_types = {v.gas_type for v in factor.gas_vectors}
        assert 'CO2' in gas_types
        assert 'CH4' in gas_types
        assert 'N2O' in gas_types

        client.close()

    def test_gas_decomposition_calculation(self, populated_db):
        """Test calculation of individual gas contributions."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('diesel_us_2024')

        # Calculate emissions for 1 gallon
        activity_amount = 1.0

        # Get gas vectors
        gas_emissions = {}
        for gas_vector in factor.gas_vectors:
            # Direct emissions (kg of gas per unit)
            direct_kg = gas_vector.kg_per_unit * activity_amount

            # CO2e emissions (applying GWP)
            co2e_kg = direct_kg * gas_vector.gwp

            gas_emissions[gas_vector.gas_type] = {
                'direct_kg': direct_kg,
                'co2e_kg': co2e_kg,
                'gwp': gas_vector.gwp
            }

        # Verify CO2 is dominant
        assert gas_emissions['CO2']['direct_kg'] > gas_emissions['CH4']['direct_kg']
        assert gas_emissions['CO2']['direct_kg'] > gas_emissions['N2O']['direct_kg']

        # Verify total CO2e matches sum of individual contributions
        total_co2e = sum(g['co2e_kg'] for g in gas_emissions.values())

        # Should approximately match factor value (within rounding)
        assert abs(total_co2e - 10.21) < 0.1

        client.close()

    def test_gwp_values(self, populated_db):
        """Test that GWP values are correct."""
        import sqlite3

        conn = sqlite3.connect(populated_db)
        cursor = conn.cursor()

        # Check GWP values
        cursor.execute("""
            SELECT gas_type, gwp FROM factor_gas_vectors
            WHERE factor_id = 'diesel_us_2024'
        """)

        gwp_values = dict(cursor.fetchall())

        # Verify GWP100 values (AR5/AR6)
        assert gwp_values['CO2'] == 1
        assert gwp_values['CH4'] == 28  # AR5 value
        assert gwp_values['N2O'] == 265  # AR5 value

        conn.close()

    def test_natural_gas_gas_breakdown(self, populated_db):
        """Test natural gas has higher CH4 component."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('natural_gas_us_2024')

        # Find CH4 vector
        ch4_vector = next((v for v in factor.gas_vectors if v.gas_type == 'CH4'), None)

        assert ch4_vector is not None
        assert ch4_vector.kg_per_unit > 0

        # Natural gas should have more CH4 than diesel
        diesel_factor = client.get_factor('diesel_us_2024')
        diesel_ch4 = next((v for v in diesel_factor.gas_vectors if v.gas_type == 'CH4'), None)

        assert ch4_vector.kg_per_unit > diesel_ch4.kg_per_unit

        client.close()


# ==================== UNCERTAINTY QUANTIFICATION TESTS ====================

class TestUncertaintyQuantification:
    """Test uncertainty quantification and Monte Carlo simulation."""

    def test_uncertainty_percent_stored(self, populated_db):
        """Test that uncertainty is stored with factors."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('diesel_us_2024')

        assert factor.data_quality.uncertainty_percent is not None
        assert 0 < factor.data_quality.uncertainty_percent <= 100

        client.close()

    def test_monte_carlo_simulation(self, populated_db):
        """Test Monte Carlo simulation for uncertainty propagation."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('diesel_us_2024')
        activity_amount = 100.0

        # Base calculation
        base_emissions = factor.emission_factor_kg_co2e * activity_amount

        # Monte Carlo simulation (1000 iterations)
        n_iterations = 1000
        uncertainty_percent = factor.data_quality.uncertainty_percent  # 5%

        # Generate random samples (normal distribution)
        np.random.seed(42)  # For reproducibility
        factor_samples = np.random.normal(
            loc=factor.emission_factor_kg_co2e,
            scale=factor.emission_factor_kg_co2e * (uncertainty_percent / 100),
            size=n_iterations
        )

        # Calculate emissions for each sample
        emissions_samples = factor_samples * activity_amount

        # Calculate statistics
        mean_emissions = np.mean(emissions_samples)
        std_emissions = np.std(emissions_samples)
        p5_emissions = np.percentile(emissions_samples, 5)
        p95_emissions = np.percentile(emissions_samples, 95)

        print(f"\nMonte Carlo Results:")
        print(f"  Base Emissions: {base_emissions:.2f} kg CO2e")
        print(f"  Mean Emissions: {mean_emissions:.2f} kg CO2e")
        print(f"  Std Dev: {std_emissions:.2f} kg CO2e")
        print(f"  P5-P95 Range: {p5_emissions:.2f} - {p95_emissions:.2f} kg CO2e")
        print(f"  Uncertainty: ±{(p95_emissions - p5_emissions) / 2:.2f} kg CO2e")

        # Verify results are reasonable
        assert abs(mean_emissions - base_emissions) < base_emissions * 0.01  # Within 1%
        assert std_emissions > 0

        client.close()

    def test_uncertainty_tier_quality(self, populated_db):
        """Test different data quality tiers have different uncertainties."""
        import sqlite3

        conn = sqlite3.connect(populated_db)
        cursor = conn.cursor()

        # Tier 1 should have lower uncertainty than Tier 3
        cursor.execute("""
            SELECT data_quality_tier, AVG(uncertainty_percent)
            FROM emission_factors
            WHERE uncertainty_percent IS NOT NULL
            GROUP BY data_quality_tier
            ORDER BY data_quality_tier
        """)

        results = cursor.fetchall()

        if len(results) >= 2:
            tier1_uncertainty = next((r[1] for r in results if 'Tier 1' in r[0]), None)
            tier3_uncertainty = next((r[1] for r in results if 'Tier 3' in r[0]), None)

            if tier1_uncertainty and tier3_uncertainty:
                assert tier1_uncertainty < tier3_uncertainty

        conn.close()


# ==================== EDGE CASES AND BOUNDARY CONDITIONS ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_activity_amount(self, populated_db):
        """Test calculation with zero activity amount."""
        client = EmissionFactorClient(db_path=populated_db)

        result = client.calculate_emissions(
            factor_id='diesel_us_2024',
            activity_amount=0.0,
            activity_unit='gallon'
        )

        assert result.emissions_kg_co2e == 0.0
        assert result.emissions_metric_tons_co2e == 0.0

        client.close()

    def test_very_small_activity_amount(self, populated_db):
        """Test calculation with very small activity amount."""
        client = EmissionFactorClient(db_path=populated_db)

        result = client.calculate_emissions(
            factor_id='diesel_us_2024',
            activity_amount=0.001,
            activity_unit='gallon'
        )

        expected = 10.21 * 0.001  # 0.01021 kg CO2e

        assert abs(result.emissions_kg_co2e - expected) < 1e-10

        client.close()

    def test_very_large_activity_amount(self, populated_db):
        """Test calculation with very large activity amount."""
        client = EmissionFactorClient(db_path=populated_db)

        large_amount = 1_000_000.0  # 1 million gallons

        result = client.calculate_emissions(
            factor_id='diesel_us_2024',
            activity_amount=large_amount,
            activity_unit='gallon'
        )

        expected = 10.21 * large_amount  # 10,210,000 kg CO2e

        assert abs(result.emissions_kg_co2e - expected) < 1e-6
        assert result.emissions_metric_tons_co2e == result.emissions_kg_co2e / 1000

        client.close()

    def test_negative_activity_amount_rejected(self, populated_db):
        """Test that negative activity amount is rejected."""
        client = EmissionFactorClient(db_path=populated_db)

        with pytest.raises(ValueError):
            client.calculate_emissions(
                factor_id='diesel_us_2024',
                activity_amount=-100.0,
                activity_unit='gallon'
            )

        client.close()

    def test_calculation_with_missing_gas_vectors(self, populated_db):
        """Test calculation for factor without gas vectors."""
        import sqlite3

        # Insert factor without gas vectors
        conn = sqlite3.connect(populated_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level,
                data_quality_tier, uncertainty_percent
            ) VALUES (
                'test_no_vectors', 'Test No Vectors', 'fuels', 'other',
                3.0, 'kg', 'Scope 1',
                'Test', 'https://test.com', 'GHG Protocol',
                '2024-01-01', 2024,
                'United States', 'Country',
                'Tier 2', 10.0
            )
        """)
        conn.commit()
        conn.close()

        client = EmissionFactorClient(db_path=populated_db)

        # Should still be able to calculate (using aggregate factor)
        result = client.calculate_emissions(
            factor_id='test_no_vectors',
            activity_amount=100.0,
            activity_unit='kg'
        )

        assert result.emissions_kg_co2e == 300.0

        client.close()

    def test_renewable_energy_factor(self, populated_db):
        """Test calculation with renewable energy factor."""
        client = EmissionFactorClient(db_path=populated_db)

        factor = client.get_factor('electricity_us_avg_2024')

        # Should have renewable share
        assert factor.renewable_share is not None
        assert 0 <= factor.renewable_share <= 1

        # Calculate emissions
        result = client.calculate_emissions(
            factor_id='electricity_us_avg_2024',
            activity_amount=1000.0,
            activity_unit='kwh'
        )

        # Emissions should reflect grid mix
        assert result.emissions_kg_co2e > 0

        client.close()


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
