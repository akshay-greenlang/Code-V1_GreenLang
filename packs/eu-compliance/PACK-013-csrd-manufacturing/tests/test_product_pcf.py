# -*- coding: utf-8 -*-
"""
Unit tests for ProductCarbonFootprintEngine - PACK-013 CSRD Manufacturing Engine 3

Tests all methods of ProductCarbonFootprintEngine with 85%+ coverage.
Validates lifecycle scope logic, allocation methods, BOM hotspot analysis,
distribution and use-phase calculations, DPP generation, data quality scoring,
and provenance hashing.

Target: 40+ tests across 11 test classes.
"""

import importlib.util
import os
import sys
import pytest
from unittest.mock import patch

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


pcf = _load_module("product_carbon_footprint_engine", "product_carbon_footprint_engine.py")

ProductCarbonFootprintEngine = pcf.ProductCarbonFootprintEngine
PCFConfig = pcf.PCFConfig
LifecycleScope = pcf.LifecycleScope
AllocationMethod = pcf.AllocationMethod
LifecycleStage = pcf.LifecycleStage
DataQualityLevel = pcf.DataQualityLevel
ProductData = pcf.ProductData
BOMComponent = pcf.BOMComponent
ManufacturingProcess = pcf.ManufacturingProcess
DistributionData = pcf.DistributionData
UsePhaseData = pcf.UsePhaseData
EndOfLifeData = pcf.EndOfLifeData
DPPData = pcf.DPPData
DataQualityScore = pcf.DataQualityScore
PCFResult = pcf.PCFResult
MATERIAL_EMISSION_FACTORS = pcf.MATERIAL_EMISSION_FACTORS
TRANSPORT_EMISSION_FACTORS = pcf.TRANSPORT_EMISSION_FACTORS
END_OF_LIFE_FACTORS = pcf.END_OF_LIFE_FACTORS
_round3 = pcf._round3
_round2 = pcf._round2
_compute_hash = pcf._compute_hash
_safe_divide = pcf._safe_divide


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_engine():
    """Create a ProductCarbonFootprintEngine with default configuration."""
    return ProductCarbonFootprintEngine()


@pytest.fixture
def sample_product():
    """Create a sample product definition."""
    return ProductData(
        product_id="prod-001",
        product_name="Industrial Pump",
        functional_unit="1 unit",
        annual_production=10000.0,
        product_weight_kg=25.0,
        product_category="industrial_equipment",
    )


@pytest.fixture
def sample_bom():
    """Create a sample BOM with steel, plastic, and glass components."""
    return [
        BOMComponent(
            component_name="Steel Housing",
            material_type="steel_primary",
            quantity_per_unit=15.0,
            recycled_content_pct=0.0,
            data_quality_score=DataQualityLevel.SCORE_2,
            supplier_name="Steel Corp",
        ),
        BOMComponent(
            component_name="Plastic Seal",
            material_type="plastics_pp",
            quantity_per_unit=3.0,
            recycled_content_pct=20.0,
            data_quality_score=DataQualityLevel.SCORE_3,
            supplier_name="Poly Ltd",
        ),
        BOMComponent(
            component_name="Glass Viewport",
            material_type="glass",
            quantity_per_unit=2.0,
            recycled_content_pct=0.0,
            data_quality_score=DataQualityLevel.SCORE_4,
            supplier_name="Glass Works",
        ),
    ]


@pytest.fixture
def sample_manufacturing_process():
    """Create sample manufacturing processes."""
    return [
        ManufacturingProcess(
            process_name="CNC Machining",
            energy_consumption_kwh_per_unit=5.0,
            process_emissions_kgco2e_per_unit=0.5,
        ),
        ManufacturingProcess(
            process_name="Assembly",
            energy_consumption_kwh_per_unit=2.0,
            process_emissions_kgco2e_per_unit=0.1,
        ),
    ]


@pytest.fixture
def sample_distribution():
    """Create sample distribution data."""
    return DistributionData(
        transport_mode="road_truck",
        distance_km=500.0,
        load_factor_pct=70.0,
    )


@pytest.fixture
def sample_use_phase():
    """Create sample use phase data."""
    return UsePhaseData(
        energy_consumption_kwh_per_use=2.0,
        uses_per_lifetime=5000.0,
        lifetime_years=10.0,
        emission_factor=0.4,
    )


@pytest.fixture
def sample_end_of_life():
    """Create sample end-of-life data."""
    return EndOfLifeData(
        recyclable_pct=60.0,
        landfill_pct=20.0,
        incineration_pct=20.0,
        primary_material="steel",
    )


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test engine initialization."""

    def test_default_init(self):
        """Engine initializes with default PCFConfig."""
        engine = ProductCarbonFootprintEngine()
        assert engine.config is not None
        assert isinstance(engine.config, PCFConfig)
        assert engine.config.lifecycle_scope == LifecycleScope.CRADLE_TO_GATE
        assert engine.engine_version == "1.0.0"

    def test_init_with_config(self):
        """Engine initializes with explicit PCFConfig."""
        config = PCFConfig(
            lifecycle_scope=LifecycleScope.CRADLE_TO_GRAVE,
            allocation_method=AllocationMethod.ECONOMIC,
            dpp_enabled=True,
        )
        engine = ProductCarbonFootprintEngine(config)
        assert engine.config.lifecycle_scope == LifecycleScope.CRADLE_TO_GRAVE
        assert engine.config.allocation_method == AllocationMethod.ECONOMIC
        assert engine.config.dpp_enabled is True

    def test_init_with_dict(self):
        """Engine initializes from a dictionary."""
        engine = ProductCarbonFootprintEngine({
            "lifecycle_scope": "gate_to_gate",
            "reporting_year": 2024,
        })
        assert engine.config.lifecycle_scope == LifecycleScope.GATE_TO_GATE
        assert engine.config.reporting_year == 2024

    def test_init_with_none(self):
        """Engine initializes with None (defaults)."""
        engine = ProductCarbonFootprintEngine(None)
        assert engine.config.allocation_method == AllocationMethod.MASS


class TestLifecycleScopes:
    """Test lifecycle scope boundary logic."""

    def test_cradle_to_gate(self, sample_product, sample_bom, sample_manufacturing_process):
        """Cradle-to-gate includes raw material + manufacturing only."""
        engine = ProductCarbonFootprintEngine(
            PCFConfig(lifecycle_scope=LifecycleScope.CRADLE_TO_GATE)
        )
        result = engine.calculate_product_pcf(
            product=sample_product,
            bom=sample_bom,
            manufacturing=sample_manufacturing_process,
        )
        assert LifecycleStage.RAW_MATERIAL.value in result.lifecycle_breakdown
        assert LifecycleStage.MANUFACTURING.value in result.lifecycle_breakdown
        assert LifecycleStage.DISTRIBUTION.value not in result.lifecycle_breakdown
        assert LifecycleStage.USE.value not in result.lifecycle_breakdown
        assert LifecycleStage.END_OF_LIFE.value not in result.lifecycle_breakdown

    def test_gate_to_gate(self, sample_product, sample_bom, sample_manufacturing_process):
        """Gate-to-gate includes manufacturing only."""
        engine = ProductCarbonFootprintEngine(
            PCFConfig(lifecycle_scope=LifecycleScope.GATE_TO_GATE)
        )
        result = engine.calculate_product_pcf(
            product=sample_product,
            bom=sample_bom,
            manufacturing=sample_manufacturing_process,
        )
        assert LifecycleStage.RAW_MATERIAL.value not in result.lifecycle_breakdown
        assert LifecycleStage.MANUFACTURING.value in result.lifecycle_breakdown

    def test_cradle_to_grave(
        self, sample_product, sample_bom, sample_manufacturing_process,
        sample_distribution, sample_use_phase, sample_end_of_life
    ):
        """Cradle-to-grave includes all 5 stages."""
        engine = ProductCarbonFootprintEngine(
            PCFConfig(lifecycle_scope=LifecycleScope.CRADLE_TO_GRAVE)
        )
        result = engine.calculate_product_pcf(
            product=sample_product,
            bom=sample_bom,
            manufacturing=sample_manufacturing_process,
            distribution=sample_distribution,
            use_phase=sample_use_phase,
            end_of_life=sample_end_of_life,
        )
        assert LifecycleStage.RAW_MATERIAL.value in result.lifecycle_breakdown
        assert LifecycleStage.MANUFACTURING.value in result.lifecycle_breakdown
        assert LifecycleStage.DISTRIBUTION.value in result.lifecycle_breakdown
        assert LifecycleStage.USE.value in result.lifecycle_breakdown
        assert LifecycleStage.END_OF_LIFE.value in result.lifecycle_breakdown

    def test_scope_enum_values(self):
        """LifecycleScope enum has exactly 3 values."""
        assert len(LifecycleScope) == 3
        assert LifecycleScope.CRADLE_TO_GATE.value == "cradle_to_gate"
        assert LifecycleScope.GATE_TO_GATE.value == "gate_to_gate"
        assert LifecycleScope.CRADLE_TO_GRAVE.value == "cradle_to_grave"


class TestAllocationMethods:
    """Test allocation method application."""

    def test_mass_allocation(self, default_engine):
        """Mass allocation applies correct share."""
        allocated = default_engine.apply_allocation(1000.0, AllocationMethod.MASS, 0.4)
        assert allocated == pytest.approx(400.0, rel=1e-6)

    def test_economic_allocation(self, default_engine):
        """Economic allocation applies correct share."""
        allocated = default_engine.apply_allocation(
            1000.0, AllocationMethod.ECONOMIC, 0.75
        )
        assert allocated == pytest.approx(750.0, rel=1e-6)

    def test_physical_causality(self, default_engine):
        """Physical causality allocation applies correct share."""
        allocated = default_engine.apply_allocation(
            2000.0, AllocationMethod.PHYSICAL_CAUSALITY, 0.5
        )
        assert allocated == pytest.approx(1000.0, rel=1e-6)

    def test_system_expansion(self, default_engine):
        """System expansion allocation at 100% share returns total."""
        allocated = default_engine.apply_allocation(
            500.0, AllocationMethod.SYSTEM_EXPANSION, 1.0
        )
        assert allocated == pytest.approx(500.0, rel=1e-6)

    def test_allocation_enum_values(self):
        """AllocationMethod enum has exactly 4 values."""
        assert len(AllocationMethod) == 4
        assert AllocationMethod.MASS.value == "mass"
        assert AllocationMethod.ECONOMIC.value == "economic"
        assert AllocationMethod.PHYSICAL_CAUSALITY.value == "physical_causality"
        assert AllocationMethod.SYSTEM_EXPANSION.value == "system_expansion"


class TestRawMaterialStage:
    """Test raw material stage emission calculations."""

    def test_steel_component(self, default_engine):
        """Steel primary component emission = qty * factor."""
        bom = [
            BOMComponent(
                component_name="Steel Part",
                material_type="steel_primary",
                quantity_per_unit=10.0,
            ),
        ]
        total_co2, biogenic, hotspots = default_engine.calculate_raw_material_stage(bom)
        expected = 10.0 * MATERIAL_EMISSION_FACTORS["steel_primary"]["factor_kgco2e_per_kg"]
        assert total_co2 == pytest.approx(expected, rel=1e-6)
        assert biogenic == 0.0

    def test_plastic_component(self, default_engine):
        """Plastic PP component emission uses correct factor."""
        bom = [
            BOMComponent(
                component_name="PP Part",
                material_type="plastics_pp",
                quantity_per_unit=5.0,
            ),
        ]
        total_co2, _, _ = default_engine.calculate_raw_material_stage(bom)
        expected = 5.0 * MATERIAL_EMISSION_FACTORS["plastics_pp"]["factor_kgco2e_per_kg"]
        assert total_co2 == pytest.approx(expected, rel=1e-6)

    def test_recycled_content_reduction(self, default_engine):
        """Recycled content reduces effective emission factor."""
        bom_virgin = [
            BOMComponent(
                component_name="Steel A",
                material_type="steel_primary",
                quantity_per_unit=10.0,
                recycled_content_pct=0.0,
            ),
        ]
        bom_recycled = [
            BOMComponent(
                component_name="Steel B",
                material_type="steel_primary",
                quantity_per_unit=10.0,
                recycled_content_pct=50.0,
            ),
        ]
        co2_virgin, _, _ = default_engine.calculate_raw_material_stage(bom_virgin)
        co2_recycled, _, _ = default_engine.calculate_raw_material_stage(bom_recycled)
        assert co2_recycled < co2_virgin

    def test_multiple_components(self, default_engine, sample_bom):
        """Multiple BOM components sum correctly."""
        total_co2, _, hotspots = default_engine.calculate_raw_material_stage(sample_bom)
        assert total_co2 > 0.0
        assert len(hotspots) == 3

    def test_bom_hotspots(self, default_engine, sample_bom):
        """Hotspots are sorted by contribution (highest first)."""
        _, _, hotspots = default_engine.calculate_raw_material_stage(sample_bom)
        if len(hotspots) >= 2:
            assert hotspots[0]["co2_kgco2e"] >= hotspots[1]["co2_kgco2e"]
        assert hotspots[0]["component_name"] == "Steel Housing"


class TestManufacturingStage:
    """Test manufacturing stage emission calculations."""

    def test_energy_per_unit(self, default_engine):
        """Energy emissions: kwh * grid_ef (0.4 kgCO2e/kWh)."""
        processes = [
            ManufacturingProcess(
                process_name="Machining",
                energy_consumption_kwh_per_unit=10.0,
                process_emissions_kgco2e_per_unit=0.0,
            ),
        ]
        result = default_engine.calculate_manufacturing_stage(processes)
        expected = 10.0 * 0.4
        assert result == pytest.approx(expected, rel=1e-6)

    def test_process_emissions_per_unit(self, default_engine):
        """Direct process emissions added per unit."""
        processes = [
            ManufacturingProcess(
                process_name="Welding",
                energy_consumption_kwh_per_unit=0.0,
                process_emissions_kgco2e_per_unit=2.5,
            ),
        ]
        result = default_engine.calculate_manufacturing_stage(processes)
        assert result == pytest.approx(2.5, rel=1e-6)

    def test_waste_per_unit(self, default_engine):
        """Waste field is tracked but does not add to emissions directly."""
        processes = [
            ManufacturingProcess(
                process_name="Cutting",
                energy_consumption_kwh_per_unit=3.0,
                process_emissions_kgco2e_per_unit=0.0,
                waste_generated_kg_per_unit=0.5,
            ),
        ]
        result = default_engine.calculate_manufacturing_stage(processes)
        expected = 3.0 * 0.4
        assert result == pytest.approx(expected, rel=1e-6)

    def test_total_manufacturing(self, default_engine, sample_manufacturing_process):
        """Total manufacturing sums energy + process across all steps."""
        result = default_engine.calculate_manufacturing_stage(sample_manufacturing_process)
        expected = (5.0 * 0.4 + 0.5) + (2.0 * 0.4 + 0.1)
        assert result == pytest.approx(expected, rel=1e-6)


class TestDistributionStage:
    """Test distribution / transport emission calculations."""

    def test_road_transport(self, default_engine, sample_distribution):
        """Road transport: weight_t * distance * ef / load_factor."""
        co2 = default_engine.calculate_distribution_stage(
            sample_distribution, product_weight_kg=25.0
        )
        expected = 0.025 * 500.0 * 0.062 * (100.0 / 70.0)
        assert co2 == pytest.approx(expected, rel=1e-4)

    def test_multimodal(self, default_engine):
        """Different transport modes use correct emission factors."""
        dist_sea = DistributionData(
            transport_mode="sea_container",
            distance_km=10000.0,
            load_factor_pct=80.0,
        )
        co2_sea = default_engine.calculate_distribution_stage(dist_sea, 25.0)
        dist_air = DistributionData(
            transport_mode="air_freight",
            distance_km=10000.0,
            load_factor_pct=80.0,
        )
        co2_air = default_engine.calculate_distribution_stage(dist_air, 25.0)
        assert co2_air > co2_sea

    def test_zero_distance(self, default_engine):
        """Zero distance yields zero distribution emissions."""
        dist = DistributionData(
            transport_mode="road_truck",
            distance_km=0.0,
        )
        co2 = default_engine.calculate_distribution_stage(dist, 25.0)
        assert co2 == 0.0


class TestUseStage:
    """Test use phase emission calculations."""

    def test_energy_consuming_product(self, default_engine, sample_use_phase):
        """Use phase: energy_per_use * uses * emission_factor."""
        co2 = default_engine.calculate_use_stage(sample_use_phase)
        expected = 2.0 * 5000.0 * 0.4
        assert co2 == pytest.approx(expected, rel=1e-6)

    def test_zero_use_emissions(self, default_engine):
        """Product with zero energy use has zero use-phase emissions."""
        use = UsePhaseData(
            energy_consumption_kwh_per_use=0.0,
            uses_per_lifetime=1000.0,
            lifetime_years=5.0,
        )
        co2 = default_engine.calculate_use_stage(use)
        assert co2 == 0.0

    def test_lifetime_calculation(self, default_engine):
        """Lifetime-based total energy is correctly computed."""
        use = UsePhaseData(
            energy_consumption_kwh_per_use=1.0,
            uses_per_lifetime=365.0,
            lifetime_years=1.0,
            emission_factor=0.5,
        )
        co2 = default_engine.calculate_use_stage(use)
        expected = 1.0 * 365.0 * 0.5
        assert co2 == pytest.approx(expected, rel=1e-6)


class TestEndOfLifeStage:
    """Test end-of-life emission calculations."""

    def test_recyclable_product(self, default_engine, sample_end_of_life):
        """Steel product with high recyclability gets recycling credit."""
        co2 = default_engine.calculate_end_of_life_stage(
            sample_end_of_life, product_weight_kg=25.0
        )
        landfill = 25.0 * 0.20 * END_OF_LIFE_FACTORS["landfill"]["steel"]
        incin = 25.0 * 0.20 * END_OF_LIFE_FACTORS["incineration"]["steel"]
        recycling = 25.0 * 0.60 * END_OF_LIFE_FACTORS["recycling_credit"]["steel"]
        expected = landfill + incin + recycling
        assert co2 == pytest.approx(expected, rel=1e-4)
        assert co2 < 0.0

    def test_landfill_product(self, default_engine):
        """100% landfill produces positive emissions."""
        eol = EndOfLifeData(
            recyclable_pct=0.0,
            landfill_pct=100.0,
            incineration_pct=0.0,
            primary_material="plastics",
        )
        co2 = default_engine.calculate_end_of_life_stage(eol, 10.0)
        expected = 10.0 * 1.0 * END_OF_LIFE_FACTORS["landfill"]["plastics"]
        assert co2 == pytest.approx(expected, rel=1e-4)
        assert co2 > 0.0

    def test_recycling_credit(self, default_engine):
        """100% recycling produces negative emissions (credit)."""
        eol = EndOfLifeData(
            recyclable_pct=100.0,
            landfill_pct=0.0,
            incineration_pct=0.0,
            primary_material="aluminum",
        )
        co2 = default_engine.calculate_end_of_life_stage(eol, 5.0)
        expected = 5.0 * END_OF_LIFE_FACTORS["recycling_credit"]["aluminum"]
        assert co2 == pytest.approx(expected, rel=1e-4)
        assert co2 < 0.0


class TestDPP:
    """Test Digital Product Passport generation."""

    def test_dpp_generation(
        self, sample_product, sample_bom, sample_manufacturing_process
    ):
        """DPP is generated when dpp_enabled=True."""
        engine = ProductCarbonFootprintEngine(
            PCFConfig(dpp_enabled=True)
        )
        result = engine.calculate_product_pcf(
            product=sample_product,
            bom=sample_bom,
            manufacturing=sample_manufacturing_process,
        )
        assert result.dpp_data is not None
        assert isinstance(result.dpp_data, DPPData)

    def test_dpp_fields(
        self, sample_product, sample_bom, sample_manufacturing_process
    ):
        """DPP contains carbon footprint and material composition."""
        engine = ProductCarbonFootprintEngine(
            PCFConfig(dpp_enabled=True)
        )
        result = engine.calculate_product_pcf(
            product=sample_product,
            bom=sample_bom,
            manufacturing=sample_manufacturing_process,
        )
        dpp = result.dpp_data
        assert dpp.carbon_footprint_per_unit > 0.0
        assert len(dpp.material_composition) > 0
        assert dpp.product_passport_id is not None

    def test_dpp_carbon_footprint(
        self, sample_product, sample_bom, sample_manufacturing_process
    ):
        """DPP carbon footprint matches total PCF."""
        engine = ProductCarbonFootprintEngine(
            PCFConfig(dpp_enabled=True)
        )
        result = engine.calculate_product_pcf(
            product=sample_product,
            bom=sample_bom,
            manufacturing=sample_manufacturing_process,
        )
        assert result.dpp_data.carbon_footprint_per_unit == pytest.approx(
            result.total_pcf_kgco2e, rel=1e-3
        )


class TestDataQuality:
    """Test data quality assessment."""

    def test_score_assessment(self, default_engine, sample_bom):
        """Data quality score is computed from BOM."""
        dq = default_engine.assess_data_quality(sample_bom)
        assert isinstance(dq, DataQualityScore)
        assert 1.0 <= dq.overall_score <= 5.0
        assert dq.components_assessed == 3

    def test_weighted_average(self, default_engine):
        """Weighted average uses mass contribution."""
        bom = [
            BOMComponent(
                component_name="Heavy",
                material_type="steel_primary",
                quantity_per_unit=100.0,
                data_quality_score=DataQualityLevel.SCORE_1,
            ),
            BOMComponent(
                component_name="Light",
                material_type="plastics_pp",
                quantity_per_unit=1.0,
                data_quality_score=DataQualityLevel.SCORE_5,
            ),
        ]
        dq = default_engine.assess_data_quality(bom)
        assert dq.overall_score < 2.0

    def test_dq_levels(self):
        """DataQualityLevel enum has 5 values."""
        assert len(DataQualityLevel) == 5
        assert DataQualityLevel.SCORE_1.value == "score_1"
        assert DataQualityLevel.SCORE_5.value == "score_5"


class TestProvenance:
    """Test provenance hash generation."""

    def test_hash(
        self, default_engine, sample_product, sample_bom,
        sample_manufacturing_process
    ):
        """Result has a 64-character provenance hash."""
        result = default_engine.calculate_product_pcf(
            product=sample_product,
            bom=sample_bom,
            manufacturing=sample_manufacturing_process,
        )
        assert len(result.provenance_hash) == 64

    def test_deterministic(self):
        """Same data produces the same hash."""
        data = {"product": "pump", "co2": 123.456}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2

    def test_different_input(self):
        """Different data produces different hashes."""
        h1 = _compute_hash({"co2": 100.0})
        h2 = _compute_hash({"co2": 200.0})
        assert h1 != h2
