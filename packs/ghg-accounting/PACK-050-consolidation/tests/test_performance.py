# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Performance Tests

Tests engine performance benchmarks, consolidation with multiple entities,
report generation speed, and memory/throughput targets.

Target: 15-25 tests.
"""

import time
import pytest
from decimal import Decimal
from datetime import date

from engines.entity_registry_engine import EntityRegistryEngine
from engines.ownership_structure_engine import OwnershipStructureEngine, _round2
from engines.intercompany_elimination_engine import IntercompanyEliminationEngine
from engines.acquisition_divestiture_engine import AcquisitionDivestitureEngine
from engines.consolidation_adjustment_engine import ConsolidationAdjustmentEngine
from engines.group_reporting_engine import GroupReportingEngine
from engines.consolidation_audit_engine import ConsolidationAuditEngine


class TestEntityRegistryPerformance:
    """Test entity registry performance."""

    def test_register_100_entities_under_1s(self):
        engine = EntityRegistryEngine()
        start = time.perf_counter()
        for i in range(100):
            engine.register_entity({
                "entity_id": f"ENT-{i:04d}",
                "entity_name": f"Entity {i}",
                "entity_type": "SUBSIDIARY",
                "country": "US",
            })
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Registration took {elapsed:.3f}s (target <1.0s)"

    def test_register_500_entities_under_5s(self):
        engine = EntityRegistryEngine()
        start = time.perf_counter()
        for i in range(500):
            engine.register_entity({
                "entity_id": f"ENT-{i:04d}",
                "entity_name": f"Entity {i}",
                "entity_type": "SUBSIDIARY",
                "country": "DE" if i % 3 == 0 else "US",
            })
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Registration took {elapsed:.3f}s (target <5.0s)"

    def test_entity_search_under_100ms(self):
        engine = EntityRegistryEngine()
        for i in range(200):
            engine.register_entity({
                "entity_id": f"ENT-{i:04d}",
                "entity_name": f"Entity {i}",
                "entity_type": "SUBSIDIARY" if i % 2 == 0 else "ASSOCIATE",
                "country": "US",
            })
        start = time.perf_counter()
        results = engine.search_entities(entity_type="SUBSIDIARY")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"Search took {elapsed:.3f}s (target <0.1s)"
        assert len(results) == 100


class TestOwnershipPerformance:
    """Test ownership chain resolution performance."""

    def test_resolve_100_chains_under_1s(self):
        engine = OwnershipStructureEngine()
        parent = "PARENT"
        for i in range(100):
            engine.set_ownership({
                "owner_entity_id": parent,
                "target_entity_id": f"SUB-{i:04d}",
                "ownership_pct": Decimal("80"),
                "manages_operations": True,
            })

        start = time.perf_counter()
        for i in range(100):
            engine.resolve_equity_chain(parent, f"SUB-{i:04d}")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"100 chains took {elapsed:.3f}s (target <1.0s)"

    def test_multi_tier_chain_under_50ms(self):
        engine = OwnershipStructureEngine()
        # 5-tier chain: A -> B -> C -> D -> E
        engine.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("90")})
        engine.set_ownership({"owner_entity_id": "B", "target_entity_id": "C", "ownership_pct": Decimal("80")})
        engine.set_ownership({"owner_entity_id": "C", "target_entity_id": "D", "ownership_pct": Decimal("70")})
        engine.set_ownership({"owner_entity_id": "D", "target_entity_id": "E", "ownership_pct": Decimal("60")})

        start = time.perf_counter()
        for _ in range(100):
            engine.resolve_equity_chain("A", "E")
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"Average chain resolution: {avg_ms:.1f}ms (target <50ms)"

    def test_control_assessment_under_1s(self):
        engine = OwnershipStructureEngine()
        parent = "PARENT"
        for i in range(100):
            engine.set_ownership({
                "owner_entity_id": parent,
                "target_entity_id": f"SUB-{i:04d}",
                "ownership_pct": Decimal("80"),
                "manages_operations": True,
            })

        start = time.perf_counter()
        for i in range(100):
            engine.assess_control(parent, f"SUB-{i:04d}")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"100 assessments took {elapsed:.3f}s (target <1.0s)"


class TestEliminationPerformance:
    """Test elimination processing performance."""

    def test_register_100_transfers_under_1s(self):
        engine = IntercompanyEliminationEngine()
        start = time.perf_counter()
        for i in range(100):
            engine.register_transfer({
                "seller_entity_id": f"ENT-{i:04d}",
                "buyer_entity_id": f"ENT-{(i+1):04d}",
                "transfer_type": "ELECTRICITY",
                "seller_emissions_tco2e": Decimal("100"),
                "buyer_emissions_tco2e": Decimal("100"),
                "intra_group_pct": Decimal("100"),
            })
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"100 transfers took {elapsed:.3f}s (target <1.0s)"


class TestMnAPerformance:
    """Test M&A event processing performance."""

    def test_register_and_prorate_50_events_under_2s(self):
        engine = AcquisitionDivestitureEngine()
        start = time.perf_counter()
        for i in range(50):
            event = engine.register_event({
                "event_type": "ACQUISITION" if i % 2 == 0 else "DIVESTITURE",
                "entity_id": f"ENT-{i:04d}",
                "effective_date": date(2025, 7, 1),
                "reporting_year": 2025,
                "annual_emissions_tco2e": Decimal("10000"),
                "scope1_tco2e": Decimal("4000"),
                "scope2_location_tco2e": Decimal("3000"),
                "scope3_tco2e": Decimal("3000"),
            })
            engine.calculate_prorate(event.event_id)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"50 events took {elapsed:.3f}s (target <2.0s)"


class TestReportGenerationPerformance:
    """Test report generation performance."""

    def test_report_with_100_entities_under_5s(self):
        engine = GroupReportingEngine()
        entity_data = []
        for i in range(100):
            entity_data.append({
                "entity_id": f"ENT-{i:04d}",
                "entity_name": f"Entity {i}",
                "scope1": str(1000 + i * 10),
                "scope2_location": str(500 + i * 5),
                "scope2_market": str(480 + i * 5),
                "scope3": str(300 + i * 3),
                "country": ["US", "DE", "GB", "FR", "JP"][i % 5],
                "region": ["AMERICAS", "EUROPE", "EUROPE", "EUROPE", "APAC"][i % 5],
                "sector": "MANUFACTURING",
            })

        start = time.perf_counter()
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            organisation_name="Large Corp",
            intensity_denominators={"revenue_m": "5000", "employees": "50000"},
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Report generation took {elapsed:.3f}s (target <5.0s)"
        assert report.entity_count == 100
        assert report.scope_breakdown.total > Decimal("0")

    def test_framework_mapping_under_1s(self):
        engine = GroupReportingEngine()
        entity_data = [
            {
                "entity_id": "ENT-001",
                "scope1": "10000",
                "scope2_location": "5000",
                "scope3": "3000",
                "country": "US",
            },
        ]
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )

        start = time.perf_counter()
        frameworks = [
            "CSRD_ESRS_E1", "CDP", "GRI_305", "TCFD",
            "SEC_CLIMATE", "SBTI", "IFRS_S2", "UK_SECR", "ISO_14064",
        ]
        for fw in frameworks:
            engine.map_to_framework(report, fw)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"9 mappings took {elapsed:.3f}s (target <1.0s)"


class TestAuditPerformance:
    """Test audit engine performance."""

    def test_record_1000_entries_under_2s(self):
        engine = ConsolidationAuditEngine()
        start = time.perf_counter()
        for i in range(1000):
            engine.record_step(
                reporting_year=2025,
                step_type="DATA_RECEIPT",
                description=f"Step {i}",
                entity_id=f"ENT-{i:04d}",
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"1000 entries took {elapsed:.3f}s (target <2.0s)"

    def test_assurance_package_under_1s(self):
        engine = ConsolidationAuditEngine()
        for i in range(100):
            engine.record_step(2025, "DATA_RECEIPT", f"Step {i}", entity_id=f"ENT-{i:03d}")
        engine.reconcile(2025, Decimal("50000"), Decimal("50000"))
        engine.check_completeness(
            2025,
            [f"ENT-{i:03d}" for i in range(100)],
            [f"ENT-{i:03d}" for i in range(100)],
        )
        engine.record_signoff(2025, "GROUP", "cfo@corp.com")

        start = time.perf_counter()
        package = engine.generate_assurance_package(2025, consolidated_total_tco2e=Decimal("50000"))
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Assurance package took {elapsed:.3f}s (target <1.0s)"
        assert package.total_audit_entries > 0


class TestDecimalPerformance:
    """Test Decimal arithmetic performance."""

    def test_1000_equity_calculations_under_500ms(self):
        start = time.perf_counter()
        for i in range(1000):
            pct = Decimal(str(50 + (i % 50)))
            emissions = Decimal(str(10000 + i))
            allocated = _round2(emissions * pct / Decimal("100"))
            assert isinstance(allocated, Decimal)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"1000 calcs took {elapsed:.3f}s (target <0.5s)"
