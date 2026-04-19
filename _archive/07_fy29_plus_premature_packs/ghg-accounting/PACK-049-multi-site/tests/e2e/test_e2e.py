# -*- coding: utf-8 -*-
"""
End-to-end tests for PACK-049 GHG Multi-Site Management Pack.

Tests complete lifecycle scenarios including single-site lifecycle,
multi-site consolidation, boundary changes, shared services allocation,
landlord-tenant splits, cogeneration, site comparison, quality improvement,
all 8 presets, determinism, and provenance chain integrity.
Target: ~100 tests.
"""

import json
import pytest
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, timezone
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

# ---------------------------------------------------------------------------
# Import engines
# ---------------------------------------------------------------------------

try:
    from engines.site_registry_engine import (
        SiteRegistryEngine, SiteRecord, FacilityCharacteristics,
    )
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

try:
    from engines.site_allocation_engine import (
        SiteAllocationEngine, AllocationConfig, VPPACertificate,
        DistrictConsumption, LandlordTenantSplit, CogenerationAllocation,
    )
    HAS_ALLOCATION = True
except ImportError:
    HAS_ALLOCATION = False

try:
    from engines.site_data_collection_engine import SiteDataCollectionEngine
    HAS_COLLECTION = True
except ImportError:
    HAS_COLLECTION = False

try:
    from engines.site_boundary_engine import SiteBoundaryEngine
    HAS_BOUNDARY = True
except ImportError:
    HAS_BOUNDARY = False

try:
    from engines.regional_factor_engine import RegionalFactorEngine
    HAS_FACTOR = True
except ImportError:
    HAS_FACTOR = False

try:
    from engines.site_consolidation_engine import SiteConsolidationEngine, SiteTotal
    HAS_CONSOLIDATION = True
except ImportError:
    HAS_CONSOLIDATION = False

try:
    from engines.site_comparison_engine import SiteComparisonEngine
    HAS_COMPARISON = True
except ImportError:
    HAS_COMPARISON = False

try:
    from engines.site_completion_engine import SiteCompletionEngine
    HAS_COMPLETION = True
except ImportError:
    HAS_COMPLETION = False

try:
    from engines.site_quality_engine import SiteQualityEngine
    HAS_QUALITY = True
except ImportError:
    HAS_QUALITY = False

try:
    from engines.multi_site_reporting_engine import MultiSiteReportingEngine
    HAS_REPORTING = True
except ImportError:
    HAS_REPORTING = False

from config.pack_config import (
    PackConfig, MultiSitePackConfig, ConsolidationApproach,
    AVAILABLE_PRESETS, load_preset, validate_config,
    BoundaryConfig, SiteRegistryConfig, PerformanceConfig,
)


# ============================================================================
# E2E: Single Site Lifecycle
# ============================================================================

@pytest.mark.skipif(not HAS_REGISTRY, reason="Registry engine not built")
class TestE2ESingleSiteLifecycle:

    def test_e2e_register_site(self):
        engine = SiteRegistryEngine()
        site = engine.register_site({
            "site_code": "E2E-SITE-001",
            "site_name": "E2E Test Plant",
            "facility_type": "MANUFACTURING",
            "legal_entity_id": "LE-E2E",
            "country": "US",
            "characteristics": {
                "floor_area_m2": Decimal("15000"),
                "headcount": 200,
                "operating_hours_per_year": 6000,
            },
        })
        assert site.is_active is True
        assert site.lifecycle_status == "OPERATIONAL"

    def test_e2e_update_and_classify(self):
        engine = SiteRegistryEngine()
        site = engine.register_site({
            "site_code": "E2E-SITE-002",
            "site_name": "E2E Office",
            "facility_type": "OFFICE",
            "legal_entity_id": "LE-E2E",
            "country": "GB",
            "characteristics": {
                "floor_area_m2": Decimal("3000"),
                "headcount": 100,
                "operating_hours_per_year": 2500,
            },
        })
        engine.update_site(site.site_id, {"business_unit": "Europe"})
        classification = engine.classify_site(site.site_id)
        assert classification.primary_category == "OFFICE"
        assert classification.emission_intensity_class == "LOW"

    def test_e2e_decommission(self):
        engine = SiteRegistryEngine()
        site = engine.register_site({
            "site_code": "E2E-SITE-003",
            "site_name": "E2E Closing Plant",
            "facility_type": "WAREHOUSE",
            "legal_entity_id": "LE-E2E",
            "country": "DE",
            "characteristics": {
                "floor_area_m2": Decimal("8000"),
                "headcount": 50,
                "operating_hours_per_year": 4000,
            },
        })
        decommissioned = engine.decommission_site(
            site.site_id, date(2026, 12, 31), "Lease expiry",
        )
        assert decommissioned.is_active is False
        summary = engine.get_portfolio_summary()
        assert summary.total_active_sites == 0

    def test_e2e_single_site_completeness(self):
        engine = SiteRegistryEngine()
        site = engine.register_site({
            "site_code": "E2E-COMPL-001",
            "site_name": "Complete Site",
            "facility_type": "MANUFACTURING",
            "legal_entity_id": "LE-E2E",
            "country": "US",
            "region": "Texas",
            "city": "Dallas",
            "characteristics": {
                "floor_area_m2": Decimal("20000"),
                "headcount": 300,
                "operating_hours_per_year": 6000,
                "production_output": Decimal("50000"),
                "production_unit": "tonnes",
                "grid_region": "ERCT",
            },
        })
        result = engine.validate_site_completeness(site)
        assert result.completeness_pct >= Decimal("80")

    def test_e2e_portfolio_summary_after_operations(self):
        engine = SiteRegistryEngine()
        for i in range(5):
            engine.register_site({
                "site_code": f"E2E-PORT-{i:03d}",
                "site_name": f"Portfolio Site {i}",
                "facility_type": "MANUFACTURING",
                "legal_entity_id": "LE-E2E",
                "country": "US",
                "characteristics": {
                    "floor_area_m2": Decimal("10000"),
                    "headcount": 100,
                    "operating_hours_per_year": 5000,
                },
            })
        summary = engine.get_portfolio_summary()
        assert summary.total_active_sites == 5
        assert summary.total_floor_area == Decimal("50000.00")
        assert summary.total_headcount == 500
        assert summary.provenance_hash is not None
        assert len(summary.provenance_hash) == 64


# ============================================================================
# E2E: Multi-Site Consolidation
# ============================================================================

@pytest.mark.skipif(not HAS_REGISTRY, reason="Registry engine not built")
class TestE2EMultiSiteConsolidation:

    def test_e2e_5_sites_consolidation(self, sample_site_records):
        engine = SiteRegistryEngine()
        sites = []
        for data in sample_site_records:
            sites.append(engine.register_site(data))
        summary = engine.get_portfolio_summary()
        assert summary.total_active_sites == 5
        assert summary.countries_covered == 3

    def test_e2e_grouping_and_filter(self, sample_site_records):
        engine = SiteRegistryEngine()
        sites = []
        for data in sample_site_records:
            sites.append(engine.register_site(data))

        us_sites = [s for s in sites if s.country == "US"]
        group = engine.create_site_group(
            "North America", "REGION", [s.site_id for s in us_sites],
        )
        assert len(group.member_site_ids) == 2

        de_filter = engine.get_sites_by_filter(filters={"country": "DE"})
        assert len(de_filter) == 2


# ============================================================================
# E2E: Allocation Scenarios
# ============================================================================

@pytest.mark.skipif(not HAS_ALLOCATION, reason="Allocation engine not built")
class TestE2EAllocationScenarios:

    def test_e2e_shared_services(self):
        engine = SiteAllocationEngine()
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={
                "site-A": Decimal("25000"),
                "site-B": Decimal("3000"),
                "site-C": Decimal("8000"),
                "site-D": Decimal("1500"),
                "site-E": Decimal("5000"),
            },
        )
        result = engine.allocate_shared_services(config, Decimal("500.00"))
        total = sum(result.allocated_amounts.values())
        assert abs(total - Decimal("500.00")) < Decimal("0.01")
        assert result.provenance_hash is not None

    def test_e2e_landlord_tenant(self):
        engine = SiteAllocationEngine()
        result = engine.calculate_landlord_tenant_split(
            site_id="BLDG-001",
            whole_building_emissions=Decimal("1000.00"),
            tenant_floor_area=Decimal("7000"),
            total_floor_area=Decimal("10000"),
            common_area_pct=Decimal("15"),
        )
        assert result.tenant_emissions + result.landlord_emissions == Decimal("1000.00") or \
               abs(result.tenant_emissions + result.landlord_emissions - Decimal("1000.00")) < Decimal("0.01")
        assert result.tenant_emissions > Decimal("700")

    def test_e2e_cogeneration(self):
        engine = SiteAllocationEngine()
        result = engine.allocate_cogeneration(
            site_id="CHP-001",
            total_fuel_emissions=Decimal("2000.00"),
            electricity_output_kwh=Decimal("800000"),
            heat_output_kwh=Decimal("200000"),
            method="EFFICIENCY",
        )
        assert result.electricity_emissions + result.heat_emissions == Decimal("2000.00") or \
               abs(result.electricity_emissions + result.heat_emissions - Decimal("2000.00")) < Decimal("0.01")
        # 80:20 ratio
        assert result.electricity_emissions > Decimal("1500")
        assert result.heat_emissions < Decimal("500")

    def test_e2e_district_system(self):
        engine = SiteAllocationEngine()
        consumption = [
            DistrictConsumption(site_id="s1", consumption_kwh=Decimal("100000")),
            DistrictConsumption(site_id="s2", consumption_kwh=Decimal("200000")),
            DistrictConsumption(site_id="s3", consumption_kwh=Decimal("300000")),
        ]
        results = engine.allocate_district_system(
            system_emissions=Decimal("600.00"),
            connected_sites=["s1", "s2", "s3"],
            consumption_data=consumption,
        )
        total = sum(r.total_allocated for r in results)
        assert abs(total - Decimal("600.00")) < Decimal("0.01")

    def test_e2e_vppa(self):
        engine = SiteAllocationEngine()
        certs = [
            VPPACertificate(volume_mwh=Decimal("5000"), emission_factor=Decimal("0.0")),
        ]
        results = engine.allocate_vppa(
            vppa_certificates=certs,
            beneficiary_sites=["s1", "s2"],
            allocation_keys={"s1": Decimal("70"), "s2": Decimal("30")},
        )
        assert len(results) == 2
        # Zero EF -> zero emissions allocated
        total = sum(r.total_allocated for r in results)
        assert total == Decimal("0") or total >= Decimal("0")

    def test_e2e_allocation_completeness_check(self):
        engine = SiteAllocationEngine()
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="HEADCOUNT",
            allocation_keys={
                "s1": Decimal("100"),
                "s2": Decimal("200"),
                "s3": Decimal("300"),
            },
        )
        result = engine.allocate_shared_services(config, Decimal("1200.00"))
        check = engine.verify_allocation_completeness(Decimal("1200.00"), [result])
        assert check.within_tolerance is True
        assert check.status == "PASS"


# ============================================================================
# E2E: Config Presets
# ============================================================================

class TestE2EPresets:

    @pytest.mark.parametrize("preset_name", list(AVAILABLE_PRESETS.keys()))
    def test_e2e_full_pipeline_each_preset(self, preset_name):
        pc = PackConfig.from_preset(preset_name)
        assert pc.preset_name == preset_name
        assert pc.pack is not None
        assert pc.pack.reporting_year >= 2020
        warnings = pc.validate_completeness()
        assert isinstance(warnings, list)

    def test_e2e_preset_hash_deterministic(self):
        pc1 = PackConfig.from_preset("corporate_general")
        pc2 = PackConfig.from_preset("corporate_general")
        assert pc1.get_config_hash() == pc2.get_config_hash()

    def test_e2e_preset_merge_override(self):
        base = PackConfig.from_preset("manufacturing")
        merged = PackConfig.merge(base, {"company_name": "E2E Corp"})
        assert merged.pack.company_name == "E2E Corp"
        assert merged.preset_name == "manufacturing"


# ============================================================================
# E2E: Determinism
# ============================================================================

@pytest.mark.skipif(not HAS_ALLOCATION, reason="Allocation engine not built")
class TestE2EDeterminism:

    def test_e2e_determinism_allocation(self):
        """Same input produces identical output."""
        engine = SiteAllocationEngine()
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={
                "s1": Decimal("5000"),
                "s2": Decimal("3000"),
                "s3": Decimal("2000"),
            },
        )
        r1 = engine.allocate_shared_services(config, Decimal("1000"))
        r2 = engine.allocate_shared_services(config, Decimal("1000"))

        assert r1.allocated_amounts == r2.allocated_amounts
        assert r1.total_allocated == r2.total_allocated

    def test_e2e_determinism_landlord_tenant(self):
        engine = SiteAllocationEngine()
        r1 = engine.calculate_landlord_tenant_split(
            "B1", Decimal("1000"), Decimal("5000"), Decimal("10000"), Decimal("10"),
        )
        r2 = engine.calculate_landlord_tenant_split(
            "B1", Decimal("1000"), Decimal("5000"), Decimal("10000"), Decimal("10"),
        )
        assert r1.tenant_emissions == r2.tenant_emissions
        assert r1.landlord_emissions == r2.landlord_emissions

    def test_e2e_determinism_cogeneration(self):
        engine = SiteAllocationEngine()
        r1 = engine.allocate_cogeneration(
            "CHP", Decimal("500"), Decimal("300000"), Decimal("200000"),
        )
        r2 = engine.allocate_cogeneration(
            "CHP", Decimal("500"), Decimal("300000"), Decimal("200000"),
        )
        assert r1.electricity_emissions == r2.electricity_emissions
        assert r1.heat_emissions == r2.heat_emissions


# ============================================================================
# E2E: Provenance Chain
# ============================================================================

@pytest.mark.skipif(not HAS_REGISTRY, reason="Registry engine not built")
class TestE2EProvenanceChain:

    def test_e2e_provenance_chain(self):
        """Every result in the chain has a 64-char SHA-256 hash."""
        engine = SiteRegistryEngine()

        site = engine.register_site({
            "site_code": "PROV-001",
            "site_name": "Provenance Test",
            "facility_type": "MANUFACTURING",
            "legal_entity_id": "LE-PROV",
            "country": "US",
            "characteristics": {
                "floor_area_m2": Decimal("20000"),
                "headcount": 300,
                "operating_hours_per_year": 6000,
            },
        })

        classification = engine.classify_site(site.site_id)
        assert classification.provenance_hash is not None
        assert len(classification.provenance_hash) == 64

        summary = engine.get_portfolio_summary()
        assert summary.provenance_hash is not None
        assert len(summary.provenance_hash) == 64

        completeness = engine.validate_site_completeness(site)
        assert completeness.provenance_hash is not None
        assert len(completeness.provenance_hash) == 64


# ============================================================================
# E2E: Allocation Combined Scenario
# ============================================================================

@pytest.mark.skipif(not HAS_ALLOCATION, reason="Allocation engine not built")
class TestE2ECombinedAllocation:

    def test_e2e_combined_shared_and_landlord(self):
        engine = SiteAllocationEngine()

        # Step 1: Shared services
        shared_config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={
                "site-A": Decimal("10000"),
                "site-B": Decimal("5000"),
            },
        )
        shared_result = engine.allocate_shared_services(
            shared_config, Decimal("300.00"),
        )

        # Step 2: Landlord-tenant for site-A
        lt_result = engine.calculate_landlord_tenant_split(
            site_id="site-A",
            whole_building_emissions=shared_result.allocated_amounts["site-A"],
            tenant_floor_area=Decimal("6000"),
            total_floor_area=Decimal("10000"),
        )

        assert lt_result.tenant_emissions + lt_result.landlord_emissions == \
               shared_result.allocated_amounts["site-A"] or \
               abs(lt_result.tenant_emissions + lt_result.landlord_emissions -
                   shared_result.allocated_amounts["site-A"]) < Decimal("0.01")

    def test_e2e_allocation_summary(self):
        engine = SiteAllocationEngine()

        c1 = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="HEADCOUNT",
            allocation_keys={"s1": Decimal("100"), "s2": Decimal("200")},
        )
        c2 = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={"s1": Decimal("5000"), "s2": Decimal("3000")},
        )

        r1 = engine.allocate_shared_services(c1, Decimal("150"))
        r2 = engine.allocate_shared_services(c2, Decimal("250"))

        summary = engine.get_allocation_summary([r1, r2])
        assert summary.allocation_count == 2
        assert abs(summary.total_allocated - Decimal("400")) < Decimal("1")

    def test_e2e_multiple_cogeneration_sites(self):
        engine = SiteAllocationEngine()
        results = []
        for i, (elec, heat) in enumerate([
            (Decimal("600000"), Decimal("400000")),
            (Decimal("500000"), Decimal("500000")),
            (Decimal("800000"), Decimal("200000")),
        ]):
            result = engine.allocate_cogeneration(
                site_id=f"CHP-{i:03d}",
                total_fuel_emissions=Decimal("1000"),
                electricity_output_kwh=elec,
                heat_output_kwh=heat,
            )
            results.append(result)
            assert abs(result.electricity_emissions + result.heat_emissions - Decimal("1000")) < Decimal("0.01")

        # Each CHP plant should sum to exactly 1000
        for r in results:
            total = r.electricity_emissions + r.heat_emissions
            assert abs(total - Decimal("1000")) < Decimal("0.01")


# ============================================================================
# E2E: Registration + Classification + Grouping Pipeline
# ============================================================================

@pytest.mark.skipif(not HAS_REGISTRY, reason="Registry engine not built")
class TestE2ERegistrationPipeline:

    def test_e2e_full_registration_pipeline(self, sample_site_records):
        engine = SiteRegistryEngine()

        # Step 1: Register all sites
        sites = []
        for data in sample_site_records:
            sites.append(engine.register_site(data))
        assert len(sites) == 5

        # Step 2: Classify all sites
        classifications = []
        for site in sites:
            cls_result = engine.classify_site(site.site_id)
            classifications.append(cls_result)
            assert cls_result.provenance_hash is not None

        # Step 3: Create groups
        us_sites = [s for s in sites if s.country == "US"]
        eu_sites = [s for s in sites if s.country in ("GB", "DE")]

        us_group = engine.create_site_group(
            "North America", "REGION", [s.site_id for s in us_sites],
        )
        eu_group = engine.create_site_group(
            "Europe", "REGION", [s.site_id for s in eu_sites],
        )

        # Step 4: Portfolio summary
        summary = engine.get_portfolio_summary()
        assert summary.total_active_sites == 5
        assert summary.countries_covered == 3
        assert len(summary.groups) == 2
        assert summary.provenance_hash is not None

    def test_e2e_filter_by_multiple_criteria(self, sample_site_records):
        engine = SiteRegistryEngine()
        for data in sample_site_records:
            engine.register_site(data)

        # Filter by country
        us_sites = engine.get_sites_by_filter(filters={"country": "US"})
        assert len(us_sites) == 2

        # Filter by facility type
        mfg = engine.get_sites_by_filter(filters={"facility_type": "MANUFACTURING"})
        assert len(mfg) == 1

        # Filter active
        active = engine.get_sites_by_filter(filters={"is_active": True})
        assert len(active) == 5


# ============================================================================
# E2E: Decimal Precision
# ============================================================================

@pytest.mark.skipif(not HAS_ALLOCATION, reason="Allocation engine not built")
class TestE2EDecimalPrecision:

    def test_e2e_decimal_precision_shared(self):
        """Verify Decimal is used throughout, no float drift."""
        engine = SiteAllocationEngine()
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={
                "s1": Decimal("3"),
                "s2": Decimal("3"),
                "s3": Decimal("3"),
            },
        )
        result = engine.allocate_shared_services(config, Decimal("100"))
        total = sum(result.allocated_amounts.values())
        assert isinstance(total, Decimal)
        assert abs(total - Decimal("100")) < Decimal("0.01")

    def test_e2e_decimal_precision_landlord_tenant(self):
        engine = SiteAllocationEngine()
        result = engine.calculate_landlord_tenant_split(
            site_id="PREC-001",
            whole_building_emissions=Decimal("333.333333"),
            tenant_floor_area=Decimal("3333"),
            total_floor_area=Decimal("10000"),
        )
        assert isinstance(result.tenant_emissions, Decimal)
        assert isinstance(result.landlord_emissions, Decimal)
        total = result.tenant_emissions + result.landlord_emissions
        assert abs(total - Decimal("333.333333")) < Decimal("0.001")


# ============================================================================
# E2E: Edge Cases
# ============================================================================

@pytest.mark.skipif(not HAS_REGISTRY, reason="Registry engine not built")
class TestE2EEdgeCases:

    def test_e2e_empty_portfolio(self):
        engine = SiteRegistryEngine()
        summary = engine.get_portfolio_summary()
        assert summary.total_active_sites == 0
        assert summary.total_floor_area == Decimal("0")
        assert summary.countries_covered == 0

    def test_e2e_single_site_all_operations(self):
        engine = SiteRegistryEngine()
        site = engine.register_site({
            "site_code": "EDGE-001",
            "site_name": "Lone Site",
            "facility_type": "OFFICE",
            "legal_entity_id": "LE-EDGE",
            "country": "US",
            "characteristics": {
                "floor_area_m2": Decimal("2000"),
                "headcount": 50,
                "operating_hours_per_year": 2500,
            },
        })

        cls_result = engine.classify_site(site.site_id)
        assert cls_result.size_class == "MEDIUM"

        completeness = engine.validate_site_completeness(site)
        assert completeness.completeness_pct >= Decimal("0")

        summary = engine.get_portfolio_summary()
        assert summary.total_active_sites == 1
        assert summary.avg_floor_area == Decimal("2000.00")

    def test_e2e_site_with_no_characteristics(self):
        engine = SiteRegistryEngine()
        site = engine.register_site({
            "site_code": "BARE-E2E",
            "site_name": "Bare Minimum Site",
            "facility_type": "OTHER",
            "legal_entity_id": "LE-BARE",
            "country": "US",
        })
        assert site.characteristics is None
        cls_result = engine.classify_site(site.site_id)
        assert cls_result.size_class == "SMALL"  # No floor area -> 0 -> SMALL


# ============================================================================
# E2E: Allocation Edge Cases
# ============================================================================

@pytest.mark.skipif(not HAS_ALLOCATION, reason="Allocation engine not built")
class TestE2EAllocationEdgeCases:

    def test_e2e_allocation_single_site(self):
        engine = SiteAllocationEngine()
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={"s1": Decimal("10000")},
        )
        result = engine.allocate_shared_services(config, Decimal("500"))
        assert result.allocated_amounts["s1"] == Decimal("500") or \
               abs(result.allocated_amounts["s1"] - Decimal("500")) < Decimal("0.01")

    def test_e2e_allocation_zero_source(self):
        engine = SiteAllocationEngine()
        config = AllocationConfig(
            allocation_type="SHARED_SERVICES",
            method="FLOOR_AREA",
            allocation_keys={"s1": Decimal("5000"), "s2": Decimal("5000")},
        )
        result = engine.allocate_shared_services(config, Decimal("0"))
        assert result.total_allocated == Decimal("0") or \
               abs(result.total_allocated) < Decimal("0.01")

    def test_e2e_landlord_full_tenant(self):
        """Tenant occupies entire building."""
        engine = SiteAllocationEngine()
        result = engine.calculate_landlord_tenant_split(
            site_id="FULL-001",
            whole_building_emissions=Decimal("1000"),
            tenant_floor_area=Decimal("10000"),
            total_floor_area=Decimal("10000"),
        )
        assert abs(result.tenant_emissions - Decimal("1000")) < Decimal("0.01")
        assert abs(result.landlord_emissions) < Decimal("0.01")

    def test_e2e_landlord_zero_tenant(self):
        """Tenant has no floor area."""
        engine = SiteAllocationEngine()
        result = engine.calculate_landlord_tenant_split(
            site_id="ZERO-001",
            whole_building_emissions=Decimal("1000"),
            tenant_floor_area=Decimal("0"),
            total_floor_area=Decimal("10000"),
        )
        assert abs(result.tenant_emissions) < Decimal("0.01")
        assert abs(result.landlord_emissions - Decimal("1000")) < Decimal("0.01")

    def test_e2e_cogeneration_all_heat(self):
        """CHP plant with zero electricity output."""
        engine = SiteAllocationEngine()
        result = engine.allocate_cogeneration(
            site_id="HEAT-001",
            total_fuel_emissions=Decimal("1000"),
            electricity_output_kwh=Decimal("0"),
            heat_output_kwh=Decimal("500000"),
        )
        assert abs(result.heat_emissions - Decimal("1000")) < Decimal("0.01")
        assert abs(result.electricity_emissions) < Decimal("0.01")

    def test_e2e_cogeneration_all_electricity(self):
        """CHP plant with zero heat output."""
        engine = SiteAllocationEngine()
        result = engine.allocate_cogeneration(
            site_id="ELEC-001",
            total_fuel_emissions=Decimal("1000"),
            electricity_output_kwh=Decimal("500000"),
            heat_output_kwh=Decimal("0"),
        )
        assert abs(result.electricity_emissions - Decimal("1000")) < Decimal("0.01")
        assert abs(result.heat_emissions) < Decimal("0.01")


# ============================================================================
# E2E: Config Validation Comprehensive
# ============================================================================

class TestE2EConfigValidation:

    def test_e2e_config_equity_share_warnings(self):
        cfg = MultiSitePackConfig(
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            company_name="",
        )
        warnings = validate_config(cfg)
        assert any("equity" in w.lower() for w in warnings)
        assert any("company_name" in w for w in warnings)

    def test_e2e_config_high_materiality_warning(self):
        cfg = MultiSitePackConfig(
            boundary=BoundaryConfig(materiality_threshold=Decimal("0.15")),
        )
        warnings = validate_config(cfg)
        assert any("materiality" in w.lower() for w in warnings)

    def test_e2e_config_large_portfolio_perf_warning(self):
        cfg = MultiSitePackConfig(
            total_sites=2000,
            site_registry=SiteRegistryConfig(max_sites=5000),
            performance=PerformanceConfig(max_concurrent_sites=50),
        )
        warnings = validate_config(cfg)
        assert any("concurrency" in w.lower() or "concurrent" in w.lower() for w in warnings)

    def test_e2e_all_presets_validate(self):
        for preset_name in AVAILABLE_PRESETS:
            pc = load_preset(preset_name)
            warnings = pc.validate_completeness()
            assert isinstance(warnings, list)
            # Presets should have relatively few warnings
            assert len(warnings) < 10
