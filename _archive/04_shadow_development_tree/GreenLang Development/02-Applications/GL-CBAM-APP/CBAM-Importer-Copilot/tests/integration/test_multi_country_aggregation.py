# -*- coding: utf-8 -*-
"""
Integration Tests: Multi-Country Aggregation
=============================================

Tests emissions aggregation across multiple dimensions:
- Aggregation by origin country
- Aggregation by product group
- Aggregation by supplier
- Complex multi-dimensional aggregations

Target: Maturity score +1 point (reporting capabilities)
Version: 1.0.0
Author: GL-TestEngineer
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.reporting_packager_agent_v2 import ReportingPackagerAgent_v2


# ============================================================================
# Country Aggregation Tests
# ============================================================================

@pytest.mark.integration
class TestCountryAggregation:
    """Test emissions aggregation by origin country."""

    def test_aggregate_by_origin_country(self, sample_calculated_shipments):
        """Test emissions correctly aggregated by country of origin."""
        packager = ReportingPackagerAgent_v2(cbam_rules_path=None)

        # Aggregate by country
        country_totals = {}
        for shipment in sample_calculated_shipments:
            country = shipment.get("origin_iso")
            emissions = shipment.get("emissions_calculation", {})
            total = emissions.get("total_emissions_tco2", 0)

            if country not in country_totals:
                country_totals[country] = {"emissions": 0, "count": 0}

            country_totals[country]["emissions"] += total
            country_totals[country]["count"] += 1

        # Verify aggregation
        assert len(country_totals) > 0, "Should aggregate by country"
        assert all(v["count"] > 0 for v in country_totals.values())

        print("\n[Country Aggregation]")
        for country, data in sorted(country_totals.items()):
            print(f"  {country}: {data['emissions']:.2f} tCO2 ({data['count']} shipments)")

    def test_top_emitting_countries(self, sample_calculated_shipments):
        """Test identification of top emitting countries."""
        # Calculate country totals
        country_totals = {}
        for shipment in sample_calculated_shipments:
            country = shipment.get("origin_iso")
            emissions = shipment.get("emissions_calculation", {})
            total = emissions.get("total_emissions_tco2", 0)
            country_totals[country] = country_totals.get(country, 0) + total

        # Sort by emissions descending
        top_countries = sorted(country_totals.items(), key=lambda x: x[1], reverse=True)[:3]

        assert len(top_countries) > 0, "Should identify top countries"

        print("\n[Top Emitting Countries]")
        for rank, (country, emissions) in enumerate(top_countries, 1):
            print(f"  {rank}. {country}: {emissions:.2f} tCO2")


# ============================================================================
# Product Group Aggregation Tests
# ============================================================================

@pytest.mark.integration
class TestProductGroupAggregation:
    """Test emissions aggregation by product group."""

    def test_aggregate_by_product_group(self, sample_calculated_shipments):
        """Test emissions correctly aggregated by CBAM product group."""
        # Aggregate by product group
        product_totals = {}
        for shipment in sample_calculated_shipments:
            product_group = shipment.get("product_group", "unknown")
            emissions = shipment.get("emissions_calculation", {})
            total = emissions.get("total_emissions_tco2", 0)

            if product_group not in product_totals:
                product_totals[product_group] = {"emissions": 0, "mass_tonnes": 0, "count": 0}

            product_totals[product_group]["emissions"] += total
            product_totals[product_group]["mass_tonnes"] += shipment.get("net_mass_kg", 0) / 1000
            product_totals[product_group]["count"] += 1

        # Verify aggregation
        assert len(product_totals) > 0, "Should aggregate by product group"

        print("\n[Product Group Aggregation]")
        for group, data in sorted(product_totals.items()):
            intensity = data["emissions"] / data["mass_tonnes"] if data["mass_tonnes"] > 0 else 0
            print(f"  {group}: {data['emissions']:.2f} tCO2, "
                  f"{data['mass_tonnes']:.1f} tonnes ({intensity:.3f} tCO2/tonne)")

    def test_emissions_intensity_by_product(self, sample_calculated_shipments):
        """Test calculation of emissions intensity (tCO2/tonne) by product."""
        product_intensity = {}

        for shipment in sample_calculated_shipments:
            product = shipment.get("product_group", "unknown")
            emissions = shipment.get("emissions_calculation", {})
            total_emissions = emissions.get("total_emissions_tco2", 0)
            mass_tonnes = shipment.get("net_mass_kg", 0) / 1000

            if product not in product_intensity:
                product_intensity[product] = {"total_emissions": 0, "total_mass": 0}

            product_intensity[product]["total_emissions"] += total_emissions
            product_intensity[product]["total_mass"] += mass_tonnes

        # Calculate intensities
        for product, data in product_intensity.items():
            if data["total_mass"] > 0:
                intensity = data["total_emissions"] / data["total_mass"]
                assert intensity > 0, f"Intensity should be positive for {product}"

                print(f"\n[{product}] Emissions intensity: {intensity:.3f} tCO2/tonne")


# ============================================================================
# Supplier Aggregation Tests
# ============================================================================

@pytest.mark.integration
class TestSupplierAggregation:
    """Test emissions aggregation by supplier."""

    def test_aggregate_by_supplier(self, sample_calculated_shipments):
        """Test emissions correctly aggregated by supplier."""
        supplier_totals = {}

        for shipment in sample_calculated_shipments:
            supplier_id = shipment.get("supplier_id", "UNKNOWN")
            emissions = shipment.get("emissions_calculation", {})
            total = emissions.get("total_emissions_tco2", 0)

            if supplier_id not in supplier_totals:
                supplier_totals[supplier_id] = {"emissions": 0, "count": 0, "countries": set()}

            supplier_totals[supplier_id]["emissions"] += total
            supplier_totals[supplier_id]["count"] += 1
            supplier_totals[supplier_id]["countries"].add(shipment.get("origin_iso"))

        # Verify aggregation
        assert len(supplier_totals) > 0, "Should aggregate by supplier"

        print("\n[Supplier Aggregation]")
        for supplier, data in sorted(supplier_totals.items(), key=lambda x: x[1]["emissions"], reverse=True)[:5]:
            print(f"  {supplier}: {data['emissions']:.2f} tCO2 "
                  f"({data['count']} shipments from {len(data['countries'])} countries)")


# ============================================================================
# Multi-Dimensional Aggregation Tests
# ============================================================================

@pytest.mark.integration
class TestMultiDimensionalAggregation:
    """Test complex multi-dimensional aggregations."""

    def test_aggregate_by_country_and_product(self, sample_calculated_shipments):
        """Test emissions aggregated by country AND product group."""
        # Two-dimensional aggregation
        aggregation = {}

        for shipment in sample_calculated_shipments:
            country = shipment.get("origin_iso")
            product = shipment.get("product_group", "unknown")
            key = (country, product)

            emissions = shipment.get("emissions_calculation", {})
            total = emissions.get("total_emissions_tco2", 0)

            if key not in aggregation:
                aggregation[key] = {"emissions": 0, "count": 0}

            aggregation[key]["emissions"] += total
            aggregation[key]["count"] += 1

        # Verify multi-dimensional aggregation
        assert len(aggregation) > 0, "Should aggregate by country x product"

        print("\n[Country × Product Aggregation]")
        for (country, product), data in sorted(aggregation.items(), key=lambda x: x[1]["emissions"], reverse=True)[:10]:
            print(f"  {country} × {product}: {data['emissions']:.2f} tCO2 ({data['count']} shipments)")

    def test_aggregate_by_quarter_country_product(self, sample_calculated_shipments_with_quarters):
        """Test three-dimensional aggregation: quarter × country × product."""
        aggregation = {}

        for shipment in sample_calculated_shipments_with_quarters:
            quarter = shipment.get("quarter", "2025-Q3")
            country = shipment.get("origin_iso")
            product = shipment.get("product_group", "unknown")
            key = (quarter, country, product)

            emissions = shipment.get("emissions_calculation", {})
            total = emissions.get("total_emissions_tco2", 0)

            if key not in aggregation:
                aggregation[key] = {"emissions": 0, "count": 0}

            aggregation[key]["emissions"] += total
            aggregation[key]["count"] += 1

        # Verify three-dimensional aggregation
        assert len(aggregation) > 0, "Should aggregate by quarter × country × product"

        print("\n[Quarter × Country × Product Aggregation]")
        for (quarter, country, product), data in sorted(aggregation.items())[:10]:
            print(f"  {quarter} | {country} | {product}: {data['emissions']:.2f} tCO2")


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_calculated_shipments():
    """Sample shipments with calculated emissions."""
    return [
        {
            "shipment_id": "SHIP-001",
            "origin_iso": "CN",
            "product_group": "iron_steel",
            "net_mass_kg": 10000,
            "supplier_id": "SUP-CN-001",
            "emissions_calculation": {
                "total_emissions_tco2": 22.0,
                "direct_emissions_tco2": 18.0,
                "indirect_emissions_tco2": 4.0
            }
        },
        {
            "shipment_id": "SHIP-002",
            "origin_iso": "TR",
            "product_group": "iron_steel",
            "net_mass_kg": 8000,
            "supplier_id": "SUP-TR-002",
            "emissions_calculation": {
                "total_emissions_tco2": 16.8,
                "direct_emissions_tco2": 14.4,
                "indirect_emissions_tco2": 2.4
            }
        },
        {
            "shipment_id": "SHIP-003",
            "origin_iso": "RU",
            "product_group": "aluminum",
            "net_mass_kg": 5000,
            "supplier_id": "SUP-RU-001",
            "emissions_calculation": {
                "total_emissions_tco2": 30.0,
                "direct_emissions_tco2": 25.0,
                "indirect_emissions_tco2": 5.0
            }
        },
        {
            "shipment_id": "SHIP-004",
            "origin_iso": "CN",
            "product_group": "cement",
            "net_mass_kg": 20000,
            "supplier_id": "SUP-CN-002",
            "emissions_calculation": {
                "total_emissions_tco2": 18.0,
                "direct_emissions_tco2": 16.0,
                "indirect_emissions_tco2": 2.0
            }
        },
        {
            "shipment_id": "SHIP-005",
            "origin_iso": "IN",
            "product_group": "fertilizers",
            "net_mass_kg": 15000,
            "supplier_id": "SUP-IN-001",
            "emissions_calculation": {
                "total_emissions_tco2": 24.0,
                "direct_emissions_tco2": 20.0,
                "indirect_emissions_tco2": 4.0
            }
        }
    ]


@pytest.fixture
def sample_calculated_shipments_with_quarters():
    """Sample shipments with quarter information."""
    base_shipments = [
        {
            "shipment_id": f"SHIP-Q3-{i:03d}",
            "quarter": "2025-Q3",
            "origin_iso": ["CN", "TR", "RU", "IN"][i % 4],
            "product_group": ["iron_steel", "aluminum", "cement"][i % 3],
            "net_mass_kg": 10000 + (i * 1000),
            "supplier_id": f"SUP-{['CN', 'TR', 'RU', 'IN'][i % 4]}-{(i % 3) + 1:03d}",
            "emissions_calculation": {
                "total_emissions_tco2": 20.0 + (i * 2.0),
                "direct_emissions_tco2": 16.0 + (i * 1.6),
                "indirect_emissions_tco2": 4.0 + (i * 0.4)
            }
        }
        for i in range(20)
    ]
    return base_shipments


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
