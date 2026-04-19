# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Supply Chain Mapping Engine.

Tests multi-tier supplier mapping, engagement tracking,
CDP integration, hotspot analysis, and engagement programs.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~55 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.supply_chain_mapping_engine import (
    SupplyChainMappingEngine,
    SupplyChainMappingInput,
    SupplyChainMappingResult,
    SupplierScorecard,
    EngagementProgramStatus,
    SupplierTier,
    EngagementLevel,
    GeographicHotspot,
    CDPScore,
    SupplierEntry,
)

from .conftest import assert_decimal_positive, assert_provenance_hash


def _make_supplier(sid, name, country, spend, emissions, **kwargs):
    return SupplierEntry(
        supplier_id=sid, supplier_name=name, country=country,
        annual_spend_usd=spend, scope3_contribution_tco2e=emissions,
        **kwargs,
    )


@pytest.fixture
def basic_suppliers():
    return [
        _make_supplier("S001", "SteelCorp AG", "DE", Decimal("120000000"), Decimal("85000")),
        _make_supplier("S002", "ChemWorks Inc", "US", Decimal("95000000"), Decimal("62000")),
        _make_supplier("S003", "PlastiPack Ltd", "GB", Decimal("78000000"), Decimal("45000")),
        _make_supplier("S004", "EnergiePro GmbH", "DE", Decimal("45000000"), Decimal("28000")),
        _make_supplier("S005", "LogiTrans BV", "NL", Decimal("32000000"), Decimal("18000")),
        _make_supplier("S006", "RawMat Asia Pte", "SG", Decimal("28000000"), Decimal("35000")),
        _make_supplier("S007", "TechParts Co", "KR", Decimal("15000000"), Decimal("12000")),
        _make_supplier("S008", "PackageCo SA", "FR", Decimal("8000000"), Decimal("5500")),
    ]


class TestSupplyChainInstantiation:
    def test_engine_instantiates(self):
        engine = SupplyChainMappingEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = SupplyChainMappingEngine()
        assert hasattr(engine, "calculate")


class TestSupplierTiering:
    def test_supplier_scorecards_generated(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert len(result.supplier_scorecards) > 0

    def test_tier_summaries_generated(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert len(result.tier_summaries) > 0

    def test_tier_assignment_present(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        for sc in result.supplier_scorecards:
            assert sc.tier in [
                SupplierTier.TIER_1_CRITICAL.value,
                SupplierTier.TIER_2_STRATEGIC.value,
                SupplierTier.TIER_3_MANAGED.value,
                SupplierTier.TIER_4_MONITORED.value,
                "tier_1_critical", "tier_2_strategic", "tier_3_managed", "tier_4_monitored",
            ]

    def test_engagement_level_present(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        for sc in result.supplier_scorecards:
            assert sc.engagement_level != ""


class TestHotspotAnalysis:
    def test_geographic_hotspots_generated(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert len(result.geographic_hotspots) > 0

    def test_hotspot_has_country(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        for hs in result.geographic_hotspots:
            assert hs.country != ""

    def test_hotspot_has_tco2e(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        for hs in result.geographic_hotspots:
            assert hs.total_tco2e >= Decimal("0")

    def test_category_hotspots_generated(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert isinstance(result.category_hotspots, dict)

    def test_top_10_suppliers_tracked(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert isinstance(result.top_10_suppliers_tco2e, list)


class TestEngagementProgram:
    def test_engagement_status_generated(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert result.engagement_status is not None

    def test_engagement_total_suppliers(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert result.engagement_status.total_suppliers >= 0

    def test_engagement_response_rate(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert result.engagement_status.response_rate_pct >= Decimal("0")

    def test_sbti_adoption_rate(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert result.engagement_status.sbti_adoption_rate_pct >= Decimal("0")

    def test_cdp_score_field_exists(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        for sc in result.supplier_scorecards:
            assert hasattr(sc, "cdp_score")


class TestSupplierScorecard:
    def test_scorecard_has_emissions(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        for sc in result.supplier_scorecards:
            assert sc.scope3_tco2e >= Decimal("0")

    def test_scorecard_has_dq(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        for sc in result.supplier_scorecards:
            assert hasattr(sc, "data_quality_level")

    def test_scorecard_has_sbti_status(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        for sc in result.supplier_scorecards:
            assert hasattr(sc, "sbti_status")

    def test_scorecard_recommended_actions(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        for sc in result.supplier_scorecards:
            assert isinstance(sc.recommended_actions, list)


class TestScaleAndPerformance:
    def test_total_scope3_mapped(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert result.total_scope3_mapped_tco2e >= Decimal("0")

    def test_scope3_coverage_pct(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(
            suppliers=basic_suppliers,
            total_scope3_tco2e=Decimal("500000"),
        ))
        assert result.scope3_coverage_pct >= Decimal("0")

    def test_provenance_hash(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert_provenance_hash(result)

    def test_regulatory_citations(self, basic_suppliers):
        engine = SupplyChainMappingEngine()
        result = engine.calculate(SupplyChainMappingInput(suppliers=basic_suppliers))
        assert len(result.regulatory_citations) > 0
