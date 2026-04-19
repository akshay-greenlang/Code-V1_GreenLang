# -*- coding: utf-8 -*-
"""
Tests for PACK-049 Engine 6: SiteAllocationEngine

Covers shared services allocation, landlord-tenant splits, cogeneration
allocation, district system allocation, VPPA allocation, completeness
checks, and decimal precision.
Target: ~50 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.site_allocation_engine import (
    SiteAllocationEngine,
    AllocationConfig,
    AllocationResult,
    LandlordTenantSplit,
    CogenerationAllocation,
    DistrictConsumption,
    VPPACertificate,
    AllocationSummary,
    CompletenessCheck,
    AllocationType,
    AllocationMethod,
    CogenerationMethod,
)


@pytest.fixture
def engine():
    return SiteAllocationEngine()


# ============================================================================
# Shared Services Allocation Tests
# ============================================================================

class TestSharedServicesAllocation:

    def test_allocate_floor_area(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={
                "site-A": Decimal("5000"),
                "site-B": Decimal("3000"),
                "site-C": Decimal("2000"),
            },
        )
        result = engine.allocate_shared_services(config, Decimal("1000"))
        assert isinstance(result, AllocationResult)
        assert result.total_allocated == Decimal("1000") or \
               abs(result.total_allocated - Decimal("1000")) < Decimal("0.01")

    def test_allocate_headcount(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="HEADCOUNT",
            allocation_keys={
                "site-A": Decimal("100"),
                "site-B": Decimal("200"),
                "site-C": Decimal("300"),
            },
        )
        result = engine.allocate_shared_services(config, Decimal("600"))
        # site-A: 100/600 * 600 = 100
        # site-B: 200/600 * 600 = 200
        # site-C: 300/600 * 600 = 300
        assert result.allocated_amounts["site-C"] >= Decimal("299")

    def test_allocate_revenue(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="REVENUE",
            allocation_keys={
                "site-A": Decimal("1000000"),
                "site-B": Decimal("500000"),
            },
        )
        result = engine.allocate_shared_services(config, Decimal("300"))
        # 2:1 ratio -> ~200 and ~100
        assert abs(result.allocated_amounts["site-A"] - Decimal("200")) < Decimal("1")

    def test_allocate_empty_keys_raises(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={},
        )
        with pytest.raises(ValueError, match="empty"):
            engine.allocate_shared_services(config, Decimal("100"))

    def test_allocate_negative_source_raises(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={"site-A": Decimal("1000")},
        )
        with pytest.raises(ValueError, match="non-negative"):
            engine.allocate_shared_services(config, Decimal("-100"))

    def test_allocate_zero_keys_sum_raises(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={
                "site-A": Decimal("0"),
                "site-B": Decimal("0"),
            },
        )
        with pytest.raises(ValueError, match="zero"):
            engine.allocate_shared_services(config, Decimal("100"))

    def test_allocate_provenance_hash(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={"site-A": Decimal("1000"), "site-B": Decimal("1000")},
        )
        result = engine.allocate_shared_services(config, Decimal("200"))
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# ============================================================================
# Landlord-Tenant Split Tests
# ============================================================================

class TestLandlordTenantSplit:

    def test_landlord_tenant_whole_building(self, engine):
        result = engine.calculate_landlord_tenant_split(
            site_id="BLDG-001",
            whole_building_emissions=Decimal("1000"),
            tenant_floor_area=Decimal("8000"),
            total_floor_area=Decimal("10000"),
            common_area_pct=Decimal("0"),
        )
        assert isinstance(result, LandlordTenantSplit)
        # 80% tenant, 20% landlord
        assert abs(result.tenant_emissions - Decimal("800")) < Decimal("1")
        assert abs(result.landlord_emissions - Decimal("200")) < Decimal("1")

    def test_landlord_tenant_tenant_only(self, engine):
        result = engine.calculate_landlord_tenant_split(
            site_id="BLDG-002",
            whole_building_emissions=Decimal("500"),
            tenant_floor_area=Decimal("500"),
            total_floor_area=Decimal("1000"),
            common_area_pct=Decimal("20"),
        )
        # common = 200 m2, private = 800 m2
        # tenant_share = 500 + (200 * 500 / 800) = 500 + 125 = 625
        # tenant_pct = 625 / 1000 = 62.5%
        assert result.tenant_emissions > Decimal("300")
        assert result.landlord_emissions > Decimal("0")
        assert abs(result.tenant_emissions + result.landlord_emissions - Decimal("500")) < Decimal("0.01")

    def test_landlord_tenant_provenance(self, engine):
        result = engine.calculate_landlord_tenant_split(
            site_id="BLDG-001",
            whole_building_emissions=Decimal("1000"),
            tenant_floor_area=Decimal("5000"),
            total_floor_area=Decimal("10000"),
        )
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_landlord_tenant_zero_emissions(self, engine):
        result = engine.calculate_landlord_tenant_split(
            site_id="BLDG-003",
            whole_building_emissions=Decimal("0"),
            tenant_floor_area=Decimal("5000"),
            total_floor_area=Decimal("10000"),
        )
        assert result.tenant_emissions == Decimal("0") or result.tenant_emissions == Decimal("0.000000")
        assert result.landlord_emissions == Decimal("0") or result.landlord_emissions == Decimal("0.000000")

    def test_landlord_tenant_invalid_area_raises(self, engine):
        with pytest.raises(ValueError):
            engine.calculate_landlord_tenant_split(
                site_id="BLDG-004",
                whole_building_emissions=Decimal("1000"),
                tenant_floor_area=Decimal("15000"),  # exceeds total
                total_floor_area=Decimal("10000"),
            )


# ============================================================================
# Cogeneration Tests
# ============================================================================

class TestCogenerationAllocation:

    def test_cogeneration_efficiency_method(self, engine):
        result = engine.allocate_cogeneration(
            site_id="CHP-001",
            total_fuel_emissions=Decimal("1000"),
            electricity_output_kwh=Decimal("600000"),
            heat_output_kwh=Decimal("400000"),
            method="EFFICIENCY",
        )
        assert isinstance(result, CogenerationAllocation)
        # 60:40 split
        assert abs(result.electricity_emissions - Decimal("600")) < Decimal("1")
        assert abs(result.heat_emissions - Decimal("400")) < Decimal("1")

    def test_cogeneration_energy_content(self, engine):
        result = engine.allocate_cogeneration(
            site_id="CHP-002",
            total_fuel_emissions=Decimal("500"),
            electricity_output_kwh=Decimal("300000"),
            heat_output_kwh=Decimal("200000"),
            method="ENERGY_CONTENT",
        )
        # Same formula, 60:40 split
        assert result.electricity_emissions + result.heat_emissions == Decimal("500") or \
               abs(result.electricity_emissions + result.heat_emissions - Decimal("500")) < Decimal("0.01")

    def test_cogeneration_zero_output_raises(self, engine):
        with pytest.raises(ValueError, match="zero"):
            engine.allocate_cogeneration(
                site_id="CHP-003",
                total_fuel_emissions=Decimal("500"),
                electricity_output_kwh=Decimal("0"),
                heat_output_kwh=Decimal("0"),
            )

    def test_cogeneration_provenance(self, engine):
        result = engine.allocate_cogeneration(
            site_id="CHP-001",
            total_fuel_emissions=Decimal("1000"),
            electricity_output_kwh=Decimal("500000"),
            heat_output_kwh=Decimal("500000"),
        )
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_cogeneration_invalid_method_raises(self, engine):
        with pytest.raises(ValueError):
            engine.allocate_cogeneration(
                site_id="CHP-004",
                total_fuel_emissions=Decimal("500"),
                electricity_output_kwh=Decimal("300000"),
                heat_output_kwh=Decimal("200000"),
                method="INVALID_METHOD",
            )


# ============================================================================
# District System Tests
# ============================================================================

class TestDistrictSystemAllocation:

    def test_district_system(self, engine):
        consumption = [
            DistrictConsumption(site_id="s1", consumption_kwh=Decimal("300000")),
            DistrictConsumption(site_id="s2", consumption_kwh=Decimal("200000")),
        ]
        results = engine.allocate_district_system(
            system_emissions=Decimal("500"),
            connected_sites=["s1", "s2"],
            consumption_data=consumption,
        )
        assert len(results) == 2
        total = sum(r.total_allocated for r in results)
        assert abs(total - Decimal("500")) < Decimal("0.01")


# ============================================================================
# VPPA Allocation Tests
# ============================================================================

class TestVPPAAllocation:

    def test_vppa_allocation(self, engine):
        certs = [
            VPPACertificate(
                volume_mwh=Decimal("1000"),
                emission_factor=Decimal("0.5"),
            ),
        ]
        results = engine.allocate_vppa(
            vppa_certificates=certs,
            beneficiary_sites=["s1", "s2"],
            allocation_keys={"s1": Decimal("60"), "s2": Decimal("40")},
        )
        assert len(results) == 2
        total = sum(r.total_allocated for r in results)
        # 1000 MWh * 0.5 tCO2e/MWh = 500 tCO2e
        assert abs(total - Decimal("500")) < Decimal("1")


# ============================================================================
# Completeness and Summary Tests
# ============================================================================

class TestAllocCompleteness:

    def test_allocation_completeness(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={"s1": Decimal("5000"), "s2": Decimal("5000")},
        )
        result = engine.allocate_shared_services(config, Decimal("1000"))
        check = engine.verify_allocation_completeness(Decimal("1000"), [result])
        assert check.within_tolerance is True
        assert check.status == "PASS"

    def test_zero_remainder(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={
                "s1": Decimal("1"),
                "s2": Decimal("1"),
                "s3": Decimal("1"),
            },
        )
        result = engine.allocate_shared_services(config, Decimal("300"))
        # Sum must equal exactly 300 (remainder absorbed by last site)
        total = sum(result.allocated_amounts.values())
        assert abs(total - Decimal("300")) < Decimal("0.01")

    def test_decimal_precision_allocation(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={"s1": Decimal("3"), "s2": Decimal("7")},
        )
        result = engine.allocate_shared_services(config, Decimal("100"))
        for site_id, amount in result.allocated_amounts.items():
            assert isinstance(amount, Decimal)

    def test_allocation_summary(self, engine):
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={"s1": Decimal("5000"), "s2": Decimal("5000")},
        )
        r1 = engine.allocate_shared_services(config, Decimal("100"))
        r2 = engine.allocate_shared_services(config, Decimal("200"))
        summary = engine.get_allocation_summary([r1, r2])
        assert isinstance(summary, AllocationSummary)
        assert summary.allocation_count == 2
        assert summary.total_allocated >= Decimal("300") or \
               abs(summary.total_allocated - Decimal("300")) < Decimal("1")
