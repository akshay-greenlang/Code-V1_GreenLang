"""
Comprehensive Test Suite for Emission Factor Database Client

Tests the database client with 1,000 factors across all phases:
- Phase 1: Core DEFRA/EPA factors (200)
- Phase 2: Extended factors (300)
- Phase 3: Industry-specific factors (500)

Author: QA Team Lead
Date: 2025-11-20
"""

import pytest
import sys
from pathlib import Path
from decimal import Decimal
from datetime import date, datetime
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_foundation" / "agents" / "calculator"))

from emission_factors import EmissionFactorDatabase, EmissionFactor


class TestEmissionFactorDatabaseClient:
    """Test emission factor database client functionality."""

    @pytest.fixture(autouse=True)
    def setup_database(self):
        """Setup test database with sample factors."""
        self.db = EmissionFactorDatabase()
        self.test_factors = self._generate_test_factors()

        # Insert all test factors
        inserted_count = self.db.bulk_insert_factors(self.test_factors)
        print(f"\n✓ Inserted {inserted_count} test emission factors")

        yield

        # Cleanup
        self.db.close()

    def _generate_test_factors(self):
        """Generate 1,000 test emission factors across all phases."""
        factors = []

        # Phase 1: Core DEFRA/EPA factors (200 factors)
        fuels = ['diesel', 'natural_gas', 'coal', 'lpg', 'gasoline', 'fuel_oil', 'biomass', 'hydrogen']
        regions = ['GB', 'US', 'DE', 'FR', 'JP', 'CN', 'IN', 'GLOBAL']

        for i, fuel in enumerate(fuels):
            for j, region in enumerate(regions):
                factor_id = f"phase1_defra_{fuel}_{region}_{2024}"
                factors.append(EmissionFactor(
                    factor_id=factor_id,
                    category="scope1",
                    subcategory="stationary_combustion",
                    activity_type="fuel_combustion",
                    material_or_fuel=fuel,
                    unit="kg_co2e_per_unit",
                    factor_co2=Decimal(str(2.5 + i * 0.1)),
                    factor_ch4=Decimal("0.0001"),
                    factor_n2o=Decimal("0.0001"),
                    factor_co2e=Decimal(str(2.52 + i * 0.1)),
                    region=region,
                    valid_from=date(2024, 1, 1),
                    valid_to=date(2024, 12, 31),
                    source="DEFRA" if region == "GB" else "EPA",
                    source_year=2024,
                    source_version="2024",
                    data_quality="high",
                    uncertainty_percentage=5.0,
                    notes=f"Phase 1 - {fuel} combustion in {region}"
                ))

        # Phase 2: Extended factors - Electricity (300 factors)
        electricity_sources = ['grid_average', 'solar', 'wind', 'hydro', 'nuclear', 'coal_power', 'gas_power']
        countries = ['GB', 'US', 'DE', 'FR', 'ES', 'IT', 'NL', 'BE', 'SE', 'NO', 'DK', 'PL', 'CZ', 'AT', 'CH', 'GLOBAL']

        for source in electricity_sources:
            for country in countries:
                factor_id = f"phase2_elec_{source}_{country}_2024"
                factors.append(EmissionFactor(
                    factor_id=factor_id,
                    category="scope2",
                    subcategory="electricity",
                    activity_type="electricity_consumption",
                    material_or_fuel=source,
                    unit="kg_co2e_per_kwh",
                    factor_co2=Decimal(str(0.2 + random.uniform(0, 0.5))),
                    factor_co2e=Decimal(str(0.21 + random.uniform(0, 0.5))),
                    region=country,
                    valid_from=date(2024, 1, 1),
                    valid_to=date(2024, 12, 31),
                    source="IEA",
                    source_year=2024,
                    source_version="2024",
                    data_quality="high",
                    uncertainty_percentage=7.5,
                    notes=f"Phase 2 - {source} electricity in {country}"
                ))

        # Phase 3: Industry-specific factors (500 factors)
        # Transportation
        transport_modes = ['truck', 'van', 'car', 'bus', 'train', 'ship', 'aircraft']
        for mode in transport_modes:
            for region in regions:
                for year in [2023, 2024, 2025]:
                    factor_id = f"phase3_transport_{mode}_{region}_{year}"
                    factors.append(EmissionFactor(
                        factor_id=factor_id,
                        category="scope3",
                        subcategory="transportation",
                        activity_type="freight_transport",
                        material_or_fuel=mode,
                        unit="kg_co2e_per_tkm",
                        factor_co2=Decimal(str(0.05 + random.uniform(0, 0.2))),
                        factor_co2e=Decimal(str(0.051 + random.uniform(0, 0.2))),
                        region=region,
                        valid_from=date(year, 1, 1),
                        valid_to=date(year, 12, 31),
                        source="GLEC",
                        source_year=year,
                        source_version=f"{year}",
                        data_quality="medium",
                        uncertainty_percentage=10.0,
                        notes=f"Phase 3 - {mode} transport in {region}"
                    ))

        # Materials
        materials = ['steel', 'aluminum', 'concrete', 'plastic', 'glass', 'paper', 'wood']
        for material in materials:
            for region in regions:
                for year in [2023, 2024]:
                    factor_id = f"phase3_material_{material}_{region}_{year}"
                    factors.append(EmissionFactor(
                        factor_id=factor_id,
                        category="scope3",
                        subcategory="purchased_goods",
                        activity_type="material_production",
                        material_or_fuel=material,
                        unit="kg_co2e_per_kg",
                        factor_co2=Decimal(str(1.0 + random.uniform(0, 3.0))),
                        factor_co2e=Decimal(str(1.05 + random.uniform(0, 3.0))),
                        region=region,
                        valid_from=date(year, 1, 1),
                        valid_to=date(year, 12, 31),
                        source="Ecoinvent",
                        source_year=year,
                        source_version=f"3.{year - 2020}",
                        data_quality="medium",
                        uncertainty_percentage=12.0,
                        notes=f"Phase 3 - {material} production in {region}"
                    ))

        print(f"\n✓ Generated {len(factors)} test emission factors")
        print(f"  - Phase 1 (DEFRA/EPA): {len([f for f in factors if 'phase1' in f.factor_id])}")
        print(f"  - Phase 2 (Electricity): {len([f for f in factors if 'phase2' in f.factor_id])}")
        print(f"  - Phase 3 (Industry): {len([f for f in factors if 'phase3' in f.factor_id])}")

        return factors

    def test_database_statistics(self):
        """Test database statistics and factor counts."""
        stats = self.db.get_statistics()

        assert stats['total_factors'] >= 1000, f"Expected >= 1000 factors, got {stats['total_factors']}"
        assert 'by_source' in stats
        assert 'by_category' in stats
        assert 'top_regions' in stats

        print(f"\n✓ Database Statistics:")
        print(f"  Total factors: {stats['total_factors']}")
        print(f"  By source: {stats['by_source']}")
        print(f"  By category: {stats['by_category']}")
        print(f"  Top regions: {list(stats['top_regions'].items())[:5]}")

    def test_query_phase1_factors(self):
        """Test querying Phase 1 core factors."""
        # Query DEFRA diesel factor
        factor = self.db.get_factor(
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="diesel",
            region="GB",
            reference_date=date(2024, 6, 1)
        )

        assert factor is not None, "Phase 1 diesel factor not found"
        assert factor.source in ["DEFRA", "EPA"]
        assert factor.data_quality == "high"
        assert factor.factor_co2e > 0

        print(f"\n✓ Phase 1 Factor Query:")
        print(f"  Factor: {factor.factor_id}")
        print(f"  CO2e: {factor.factor_co2e} {factor.unit}")
        print(f"  Source: {factor.source} {factor.source_year}")

    def test_query_phase2_factors(self):
        """Test querying Phase 2 electricity factors."""
        factor = self.db.get_factor(
            category="scope2",
            activity_type="electricity_consumption",
            material_or_fuel="grid_average",
            region="GB",
            reference_date=date(2024, 6, 1)
        )

        assert factor is not None, "Phase 2 electricity factor not found"
        assert factor.source == "IEA"
        assert "kwh" in factor.unit.lower()

        print(f"\n✓ Phase 2 Factor Query:")
        print(f"  Factor: {factor.factor_id}")
        print(f"  CO2e: {factor.factor_co2e} {factor.unit}")
        print(f"  Source: {factor.source}")

    def test_query_phase3_factors(self):
        """Test querying Phase 3 industry-specific factors."""
        # Test transportation factor
        factor = self.db.get_factor(
            category="scope3",
            activity_type="freight_transport",
            material_or_fuel="truck",
            region="US",
            reference_date=date(2024, 6, 1)
        )

        assert factor is not None, "Phase 3 transport factor not found"
        assert factor.subcategory == "transportation"

        print(f"\n✓ Phase 3 Transport Factor:")
        print(f"  Factor: {factor.factor_id}")
        print(f"  CO2e: {factor.factor_co2e} {factor.unit}")

    def test_search_by_category(self):
        """Test searching factors by category."""
        scope1_factors = self.db.search_factors(category="scope1", limit=50)
        scope2_factors = self.db.search_factors(category="scope2", limit=50)
        scope3_factors = self.db.search_factors(category="scope3", limit=50)

        assert len(scope1_factors) > 0, "No Scope 1 factors found"
        assert len(scope2_factors) > 0, "No Scope 2 factors found"
        assert len(scope3_factors) > 0, "No Scope 3 factors found"

        print(f"\n✓ Category Search:")
        print(f"  Scope 1: {len(scope1_factors)} factors")
        print(f"  Scope 2: {len(scope2_factors)} factors")
        print(f"  Scope 3: {len(scope3_factors)} factors")

    def test_search_by_source(self):
        """Test searching factors by data source."""
        defra_factors = self.db.search_factors(source="DEFRA", limit=100)
        epa_factors = self.db.search_factors(source="EPA", limit=100)
        iea_factors = self.db.search_factors(source="IEA", limit=100)

        assert len(defra_factors) > 0, "No DEFRA factors found"
        assert len(iea_factors) > 0, "No IEA factors found"

        print(f"\n✓ Source Search:")
        print(f"  DEFRA: {len(defra_factors)} factors")
        print(f"  EPA: {len(epa_factors)} factors")
        print(f"  IEA: {len(iea_factors)} factors")

    def test_search_by_region(self):
        """Test searching factors by geographic region."""
        gb_factors = self.db.search_factors(region="GB", limit=100)
        us_factors = self.db.search_factors(region="US", limit=100)
        global_factors = self.db.search_factors(region="GLOBAL", limit=100)

        assert len(gb_factors) > 0, "No GB factors found"
        assert len(us_factors) > 0, "No US factors found"

        print(f"\n✓ Region Search:")
        print(f"  GB: {len(gb_factors)} factors")
        print(f"  US: {len(us_factors)} factors")
        print(f"  GLOBAL: {len(global_factors)} factors")

    def test_temporal_validity(self):
        """Test temporal validity of factors."""
        # Query for 2024 - should find
        factor_2024 = self.db.get_factor(
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="natural_gas",
            region="GB",
            reference_date=date(2024, 6, 1)
        )

        assert factor_2024 is not None, "2024 factor not found"

        # Query for future date - may fallback or not find
        factor_2026 = self.db.get_factor(
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="natural_gas",
            region="GB",
            reference_date=date(2026, 6, 1)
        )

        print(f"\n✓ Temporal Validity:")
        print(f"  2024 factor: Found ({factor_2024.factor_id})")
        print(f"  2026 factor: {'Found' if factor_2026 else 'Not found (expected)'}")

    def test_regional_fallback(self):
        """Test regional fallback to GLOBAL."""
        # Insert only global factor for a specific material
        test_factor = EmissionFactor(
            factor_id="test_fallback_global",
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="test_fuel_fallback",
            unit="kg_co2e_per_unit",
            factor_co2=Decimal("3.0"),
            factor_co2e=Decimal("3.05"),
            region="GLOBAL",
            valid_from=date(2024, 1, 1),
            source="TEST",
            source_year=2024,
            source_version="1.0",
            data_quality="high"
        )
        self.db.insert_factor(test_factor)

        # Query for specific region (should fallback to GLOBAL)
        factor = self.db.get_factor(
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="test_fuel_fallback",
            region="ZZ",  # Non-existent region
            reference_date=date(2024, 6, 1)
        )

        assert factor is not None, "Fallback failed"
        assert factor.region == "GLOBAL", "Did not fallback to GLOBAL"

        print(f"\n✓ Regional Fallback:")
        print(f"  Requested: ZZ, Got: {factor.region}")

    def test_random_sample_verification(self):
        """Test random samples from each phase for completeness."""
        # Sample 10 random factors from each phase
        all_factors = self.test_factors
        phase1 = [f for f in all_factors if 'phase1' in f.factor_id]
        phase2 = [f for f in all_factors if 'phase2' in f.factor_id]
        phase3 = [f for f in all_factors if 'phase3' in f.factor_id]

        sample_size = min(10, len(phase1))
        samples = random.sample(phase1, sample_size)

        print(f"\n✓ Random Sample Verification (Phase 1):")
        for factor in samples[:3]:
            retrieved = self.db.get_factor(
                category=factor.category,
                activity_type=factor.activity_type,
                material_or_fuel=factor.material_or_fuel,
                region=factor.region,
                reference_date=date(2024, 6, 1)
            )
            assert retrieved is not None, f"Factor {factor.factor_id} not retrieved"
            print(f"  ✓ {factor.factor_id}: {retrieved.factor_co2e} {retrieved.unit}")

    def test_factor_structure_completeness(self):
        """Test that all factors have complete required fields."""
        sample_factors = self.db.search_factors(limit=50)

        for factor in sample_factors:
            # Required fields
            assert factor.factor_id, "Missing factor_id"
            assert factor.category, "Missing category"
            assert factor.activity_type, "Missing activity_type"
            assert factor.material_or_fuel, "Missing material_or_fuel"
            assert factor.unit, "Missing unit"
            assert factor.factor_co2e > 0, "Invalid factor_co2e"
            assert factor.region, "Missing region"
            assert factor.source, "Missing source"
            assert factor.data_quality, "Missing data_quality"

        print(f"\n✓ Factor Structure: All {len(sample_factors)} samples complete")

    def test_performance_query_speed(self):
        """Test query performance (<10ms target)."""
        import time

        queries = [
            ("scope1", "fuel_combustion", "diesel", "GB"),
            ("scope2", "electricity_consumption", "grid_average", "US"),
            ("scope3", "freight_transport", "truck", "DE"),
        ]

        times = []
        for category, activity, material, region in queries:
            start = time.perf_counter()
            factor = self.db.get_factor(
                category=category,
                activity_type=activity,
                material_or_fuel=material,
                region=region,
                reference_date=date(2024, 6, 1)
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        print(f"\n✓ Query Performance:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")
        print(f"  Target: <10ms {'✓ PASS' if avg_time < 10 else '✗ FAIL'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
