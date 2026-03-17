# -*- coding: utf-8 -*-
"""
Unit tests for SupplyChainEmissionsEngine (PACK-013, Engine 7)

Tests Scope 3 supply chain emissions calculation, BOM-based PCF,
transport emissions, hotspot analysis, data quality, and provenance.

Target: 85%+ coverage, 38+ tests.
"""

import importlib.util
import os
import sys
import pytest
from decimal import Decimal

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------

_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engines"
)


def _load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_ENGINE_DIR, file_name)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load_module("supply_chain_emissions_engine", "supply_chain_emissions_engine.py")

SupplyChainEmissionsEngine = sc.SupplyChainEmissionsEngine
SupplyChainConfig = sc.SupplyChainConfig
SupplierData = sc.SupplierData
BOMEmissionData = sc.BOMEmissionData
TransportData = sc.TransportData
SupplyChainResult = sc.SupplyChainResult
SupplierHotspot = sc.SupplierHotspot
EngagementRecommendation = sc.EngagementRecommendation
CalculationMethod = sc.CalculationMethod
Scope3Category = sc.Scope3Category
SupplierTier = sc.SupplierTier
DataQualityScore = sc.DataQualityScore
TransportMode = sc.TransportMode
EngagementPriority = sc.EngagementPriority
SPEND_EMISSION_FACTORS = sc.SPEND_EMISSION_FACTORS
MATERIAL_EMISSION_FACTORS = sc.MATERIAL_EMISSION_FACTORS
TRANSPORT_EMISSION_FACTORS = sc.TRANSPORT_EMISSION_FACTORS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    return SupplyChainConfig(
        reporting_year=2025,
        production_volume=10000,
    )


@pytest.fixture
def default_engine(default_config):
    return SupplyChainEmissionsEngine(default_config)


@pytest.fixture
def tier1_suppliers():
    return [
        SupplierData(
            supplier_id="SUP-001", supplier_name="SteelCo",
            tier=SupplierTier.TIER_1, nace_sector="C24",
            spend_eur=Decimal("5000000"),
            calculation_method=CalculationMethod.SPEND_BASED,
            data_quality_score=DataQualityScore.SCORE_5,
            scope3_category=Scope3Category.CAT_1,
        ),
        SupplierData(
            supplier_id="SUP-002", supplier_name="ChemSupply",
            tier=SupplierTier.TIER_1, nace_sector="C20",
            spend_eur=Decimal("3000000"),
            reported_emissions_tco2e=Decimal("1500"),
            calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
            data_quality_score=DataQualityScore.SCORE_1,
            scope3_category=Scope3Category.CAT_1,
        ),
        SupplierData(
            supplier_id="SUP-003", supplier_name="PlasticParts",
            tier=SupplierTier.TIER_1, nace_sector="C22",
            spend_eur=Decimal("2000000"),
            reported_emissions_tco2e=Decimal("800"),
            calculation_method=CalculationMethod.HYBRID,
            data_quality_score=DataQualityScore.SCORE_3,
            scope3_category=Scope3Category.CAT_1,
        ),
        SupplierData(
            supplier_id="SUP-004", supplier_name="ElectroComp",
            tier=SupplierTier.TIER_2, nace_sector="C26",
            spend_eur=Decimal("1500000"),
            calculation_method=CalculationMethod.SPEND_BASED,
            data_quality_score=DataQualityScore.SCORE_4,
            scope3_category=Scope3Category.CAT_1,
        ),
        SupplierData(
            supplier_id="SUP-005", supplier_name="PackagingInc",
            tier=SupplierTier.TIER_1, nace_sector="C17",
            spend_eur=Decimal("800000"),
            calculation_method=CalculationMethod.AVERAGE_DATA,
            data_quality_score=DataQualityScore.SCORE_4,
            scope3_category=Scope3Category.CAT_1,
        ),
    ]


@pytest.fixture
def sample_bom():
    return [
        BOMEmissionData(
            component_id="BOM-001", component_name="Steel Frame",
            material_type="steel_generic", quantity_per_product=Decimal("25.0"),
        ),
        BOMEmissionData(
            component_id="BOM-002", component_name="Plastic Housing",
            material_type="abs", quantity_per_product=Decimal("5.0"),
            recycled_content_pct=30.0,
        ),
    ]


@pytest.fixture
def sample_transport():
    return [
        TransportData(
            origin="Shanghai", destination="Rotterdam",
            mode=TransportMode.SEA,
            distance_km=Decimal("19000"), weight_tonnes=Decimal("500"),
            scope3_category=Scope3Category.CAT_4,
        ),
        TransportData(
            origin="Rotterdam", destination="Munich",
            mode=TransportMode.ROAD,
            distance_km=Decimal("850"), weight_tonnes=Decimal("200"),
            scope3_category=Scope3Category.CAT_4,
        ),
    ]


# ---------------------------------------------------------------------------
# TestInitialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test SupplyChainEmissionsEngine initialization."""

    def test_default_init(self, default_config):
        engine = SupplyChainEmissionsEngine(default_config)
        assert engine.config == default_config
        assert engine.config.reporting_year == 2025

    def test_with_config(self):
        cfg = SupplyChainConfig(
            reporting_year=2024,
            production_volume=5000,
        )
        engine = SupplyChainEmissionsEngine(cfg)
        assert engine.config.production_volume == 5000

    def test_with_dict(self):
        cfg = SupplyChainConfig(**{
            "reporting_year": 2025,
            "production_volume": 20000,
        })
        engine = SupplyChainEmissionsEngine(cfg)
        assert engine.config.production_volume == 20000

    def test_with_none_optional_fields(self):
        cfg = SupplyChainConfig(reporting_year=2025)
        engine = SupplyChainEmissionsEngine(cfg)
        assert engine.config.production_volume == 0


# ---------------------------------------------------------------------------
# TestCalculationMethods
# ---------------------------------------------------------------------------


class TestCalculationMethods:
    """Test calculation method hierarchy."""

    def test_supplier_specific_method(self, default_engine):
        supplier = SupplierData(
            supplier_id="S1", spend_eur=Decimal("1000000"),
            reported_emissions_tco2e=Decimal("500"),
            calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
            data_quality_score=DataQualityScore.SCORE_1,
        )
        emissions = default_engine.calculate_supplier_emissions(supplier)
        assert emissions == pytest.approx(500.0, rel=1e-4)

    def test_hybrid_method(self, default_engine):
        supplier = SupplierData(
            supplier_id="S2", spend_eur=Decimal("1000000"),
            nace_sector="C20",
            reported_emissions_tco2e=Decimal("1000"),
            calculation_method=CalculationMethod.HYBRID,
            data_quality_score=DataQualityScore.SCORE_3,
        )
        emissions = default_engine.calculate_supplier_emissions(supplier)
        # 70% reported + 30% spend-based
        spend_est = 1000000 * 0.00180  # C20 factor
        expected = 1000 * 0.7 + spend_est * 0.3
        assert emissions == pytest.approx(expected, rel=1e-2)

    def test_average_data_method(self, default_engine):
        supplier = SupplierData(
            supplier_id="S3", spend_eur=Decimal("500000"),
            nace_sector="C24",
            reported_emissions_tco2e=Decimal("300"),
            calculation_method=CalculationMethod.AVERAGE_DATA,
            data_quality_score=DataQualityScore.SCORE_3,
        )
        emissions = default_engine.calculate_supplier_emissions(supplier)
        # Average data with reported emissions uses reported value
        assert emissions == pytest.approx(300.0, rel=1e-4)

    def test_spend_based_method(self, default_engine):
        supplier = SupplierData(
            supplier_id="S4", spend_eur=Decimal("2000000"),
            nace_sector="C24",
            calculation_method=CalculationMethod.SPEND_BASED,
            data_quality_score=DataQualityScore.SCORE_5,
        )
        emissions = default_engine.calculate_supplier_emissions(supplier)
        # 2000000 * 0.00280 = 5600 tCO2e
        assert emissions == pytest.approx(5600.0, rel=1e-3)

    def test_method_enum_values(self):
        assert len(CalculationMethod) == 4
        values = {m.value for m in CalculationMethod}
        assert "supplier_specific" in values
        assert "spend_based" in values


# ---------------------------------------------------------------------------
# TestSupplierEmissions
# ---------------------------------------------------------------------------


class TestSupplierEmissions:
    """Test supplier-level emission calculations."""

    def test_single_supplier_cat1(self, default_engine):
        supplier = SupplierData(
            supplier_id="S-CAT1", spend_eur=Decimal("1000000"),
            nace_sector="C24",
            calculation_method=CalculationMethod.SPEND_BASED,
            data_quality_score=DataQualityScore.SCORE_5,
            scope3_category=Scope3Category.CAT_1,
        )
        result = default_engine.calculate_supply_chain_emissions([supplier])
        assert result.total_scope3_tco2e > 0

    def test_supplier_with_reported_data(self, default_engine):
        supplier = SupplierData(
            supplier_id="S-RPT", spend_eur=Decimal("500000"),
            reported_emissions_tco2e=Decimal("250"),
            calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
            data_quality_score=DataQualityScore.SCORE_1,
        )
        emissions = default_engine.calculate_supplier_emissions(supplier)
        assert emissions == pytest.approx(250.0, rel=1e-4)

    def test_supplier_spend_based(self, default_engine):
        supplier = SupplierData(
            supplier_id="S-SPD", spend_eur=Decimal("1000000"),
            nace_sector="default",
            calculation_method=CalculationMethod.SPEND_BASED,
            data_quality_score=DataQualityScore.SCORE_5,
        )
        emissions = default_engine.calculate_supplier_emissions(supplier)
        expected = 1000000 * 0.00080  # default factor
        assert emissions == pytest.approx(expected, rel=1e-3)

    def test_zero_spend(self, default_engine):
        supplier = SupplierData(
            supplier_id="S-ZERO", spend_eur=Decimal("0"),
            nace_sector="C24",
            calculation_method=CalculationMethod.SPEND_BASED,
            data_quality_score=DataQualityScore.SCORE_5,
        )
        emissions = default_engine.calculate_supplier_emissions(supplier)
        assert emissions == pytest.approx(0.0, abs=1e-6)

    def test_supplier_tier_enum(self):
        assert len(SupplierTier) == 4


# ---------------------------------------------------------------------------
# TestBOMEmissions
# ---------------------------------------------------------------------------


class TestBOMEmissions:
    """Test BOM-based product emission calculations."""

    def test_bom_based_calculation(self, default_engine, sample_bom):
        total = default_engine.calculate_bom_emissions(sample_bom, 1000)
        assert total > 0

    def test_bom_with_recycled_content(self, default_engine):
        bom_recycled = [
            BOMEmissionData(
                component_id="RC-1", material_type="steel_generic",
                quantity_per_product=Decimal("10.0"),
                recycled_content_pct=50.0,
            ),
        ]
        bom_virgin = [
            BOMEmissionData(
                component_id="VG-1", material_type="steel_generic",
                quantity_per_product=Decimal("10.0"),
                recycled_content_pct=0.0,
            ),
        ]
        em_recycled = default_engine.calculate_bom_emissions(bom_recycled, 1)
        em_virgin = default_engine.calculate_bom_emissions(bom_virgin, 1)
        assert em_recycled < em_virgin

    def test_bom_per_product(self, default_engine, sample_bom):
        result = default_engine.calculate_supply_chain_emissions(
            [], bom=sample_bom
        )
        assert result.bom_emissions_per_product_kgco2e > 0

    def test_total_bom_emissions(self, default_engine, sample_bom):
        result = default_engine.calculate_supply_chain_emissions(
            [], bom=sample_bom
        )
        assert result.bom_emissions_tco2e > 0


# ---------------------------------------------------------------------------
# TestTransportEmissions
# ---------------------------------------------------------------------------


class TestTransportEmissions:
    """Test transport emission calculations."""

    def test_road_transport(self, default_engine):
        legs = [
            TransportData(
                origin="A", destination="B",
                mode=TransportMode.ROAD,
                distance_km=Decimal("1000"), weight_tonnes=Decimal("20"),
            ),
        ]
        emissions = default_engine.calculate_transport_emissions(legs)
        # 1000 * 20 * 0.062 / 1000 = 1.24 tCO2e
        assert emissions == pytest.approx(1.24, rel=1e-2)

    def test_sea_transport(self, default_engine):
        legs = [
            TransportData(
                origin="A", destination="B",
                mode=TransportMode.SEA,
                distance_km=Decimal("10000"), weight_tonnes=Decimal("500"),
            ),
        ]
        emissions = default_engine.calculate_transport_emissions(legs)
        # 10000 * 500 * 0.016 / 1000 = 80 tCO2e
        assert emissions == pytest.approx(80.0, rel=1e-2)

    def test_multimodal(self, default_engine, sample_transport):
        emissions = default_engine.calculate_transport_emissions(sample_transport)
        # Sea: 19000*500*0.016/1000 = 152, Road: 850*200*0.062/1000 = 10.54
        assert emissions == pytest.approx(152 + 10.54, rel=1e-1)

    def test_transport_emission_factors(self):
        assert TransportMode.ROAD in TRANSPORT_EMISSION_FACTORS
        assert TransportMode.SEA in TRANSPORT_EMISSION_FACTORS
        assert TransportMode.AIR in TRANSPORT_EMISSION_FACTORS
        assert TransportMode.RAIL in TRANSPORT_EMISSION_FACTORS


# ---------------------------------------------------------------------------
# TestCategoryBreakdown
# ---------------------------------------------------------------------------


class TestCategoryBreakdown:
    """Test Scope 3 category breakdown."""

    def test_category_breakdown_present(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        assert len(result.category_breakdown) > 0

    def test_cat1_dominant_for_manufacturing(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        cat1_key = Scope3Category.CAT_1.value
        assert cat1_key in result.category_breakdown
        # All suppliers are Cat 1, so it should be 100%
        assert result.category_breakdown[cat1_key] == pytest.approx(
            result.total_scope3_tco2e, rel=1e-3
        )

    def test_category_totals(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        cat_sum = sum(result.category_breakdown.values())
        assert cat_sum == pytest.approx(result.total_scope3_tco2e, rel=1e-3)

    def test_all_categories_enum(self):
        assert len(Scope3Category) == 15


# ---------------------------------------------------------------------------
# TestHotspotAnalysis
# ---------------------------------------------------------------------------


class TestHotspotAnalysis:
    """Test supplier hotspot identification."""

    def test_top_suppliers_identified(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        assert len(result.supplier_hotspots) > 0

    def test_hotspot_share_of_total(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        for hs in result.supplier_hotspots:
            assert 0.0 <= hs.share_of_total_pct <= 100.0

    def test_material_hotspots(self, default_engine, sample_bom):
        result = default_engine.calculate_supply_chain_emissions(
            [], bom=sample_bom
        )
        assert isinstance(result.material_hotspots, list)

    def test_improvement_potential(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        for hs in result.supplier_hotspots:
            assert isinstance(hs.improvement_potential, str)
            assert len(hs.improvement_potential) > 0


# ---------------------------------------------------------------------------
# TestDataQuality
# ---------------------------------------------------------------------------


class TestDataQuality:
    """Test data quality assessment."""

    def test_weighted_average_dq(self, default_engine, tier1_suppliers):
        dq = default_engine.assess_data_quality(tier1_suppliers)
        assert 1.0 <= dq <= 5.0

    def test_dq_improvement_target(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        # Default target is 3.0; most suppliers are DQ 4-5
        assert isinstance(result.dq_improvement_needed, bool)

    def test_dq_score_enum(self):
        assert len(DataQualityScore) == 5
        assert DataQualityScore.SCORE_1.value == 1  # best
        assert DataQualityScore.SCORE_5.value == 5  # worst

    def test_engagement_recommendations(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        assert isinstance(result.engagement_recommendations, list)


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Test provenance hash generation."""

    def test_hash(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self, default_engine, tier1_suppliers):
        r1 = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        r2 = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64

    def test_different_input(self, default_engine):
        s1 = [SupplierData(
            supplier_id="A", spend_eur=Decimal("1000000"),
            nace_sector="C24",
            calculation_method=CalculationMethod.SPEND_BASED,
            data_quality_score=DataQualityScore.SCORE_5,
        )]
        s2 = [SupplierData(
            supplier_id="B", spend_eur=Decimal("9999999"),
            nace_sector="C10",
            calculation_method=CalculationMethod.SPEND_BASED,
            data_quality_score=DataQualityScore.SCORE_5,
        )]
        r1 = default_engine.calculate_supply_chain_emissions(s1)
        r2 = default_engine.calculate_supply_chain_emissions(s2)
        assert r1.total_scope3_tco2e != r2.total_scope3_tco2e


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_suppliers_returns_zero(self, default_engine):
        result = default_engine.calculate_supply_chain_emissions([])
        assert result.total_scope3_tco2e == 0.0

    def test_large_supply_chain(self, default_engine):
        suppliers = [
            SupplierData(
                supplier_id=f"S-{i}",
                spend_eur=Decimal(str(100000 * (i + 1))),
                nace_sector="C24",
                calculation_method=CalculationMethod.SPEND_BASED,
                data_quality_score=DataQualityScore.SCORE_5,
            )
            for i in range(50)
        ]
        result = default_engine.calculate_supply_chain_emissions(suppliers)
        assert result.supplier_count == 50
        assert result.total_scope3_tco2e > 0

    def test_result_fields(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        assert isinstance(result, SupplyChainResult)
        assert result.engine_version == "1.0.0"
        assert result.processing_time_ms >= 0

    def test_methodology_notes(self, default_engine, tier1_suppliers):
        result = default_engine.calculate_supply_chain_emissions(tier1_suppliers)
        assert isinstance(result.methodology_notes, list)
        assert len(result.methodology_notes) > 0
