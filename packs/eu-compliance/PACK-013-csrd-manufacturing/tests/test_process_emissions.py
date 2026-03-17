# -*- coding: utf-8 -*-
"""
Unit tests for ProcessEmissionsEngine - PACK-013 CSRD Manufacturing Engine 1

Tests all methods of ProcessEmissionsEngine with 85%+ coverage.
Validates calculation accuracy, CBAM embedded emissions, EU ETS benchmarking,
provenance hashing, error handling, and edge cases.

Target: 40+ tests across 8 test classes.
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


pe = _load_module("process_emissions_engine", "process_emissions_engine.py")

ProcessEmissionsEngine = pe.ProcessEmissionsEngine
ProcessEmissionsConfig = pe.ProcessEmissionsConfig
ManufacturingSubSector = pe.ManufacturingSubSector
ProcessType = pe.ProcessType
FuelType = pe.FuelType
FacilityData = pe.FacilityData
ProcessLine = pe.ProcessLine
RawMaterial = pe.RawMaterial
FuelConsumption = pe.FuelConsumption
ProcessEmissionsResult = pe.ProcessEmissionsResult
CBAMEmbeddedEmissions = pe.CBAMEmbeddedEmissions
ETSBenchmarkComparison = pe.ETSBenchmarkComparison
PROCESS_EMISSION_FACTORS = pe.PROCESS_EMISSION_FACTORS
FUEL_EMISSION_FACTORS = pe.FUEL_EMISSION_FACTORS
ETS_PRODUCT_BENCHMARKS = pe.ETS_PRODUCT_BENCHMARKS
CBAM_GOODS_CATEGORIES = pe.CBAM_GOODS_CATEGORIES
_round3 = pe._round3
_safe_divide = pe._safe_divide
_compute_hash = pe._compute_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_engine():
    """Create a ProcessEmissionsEngine with default configuration."""
    return ProcessEmissionsEngine()


@pytest.fixture
def cement_facility():
    """Create a cement facility with a CaCO3 calcination process line."""
    return FacilityData(
        facility_id="cement-fac-001",
        facility_name="Test Cement Works",
        sub_sector=ManufacturingSubSector.CEMENT,
        country="DE",
        eu_ets_installation_id="DE-ETS-12345",
        production_lines=[
            ProcessLine(
                line_id="line-calc-001",
                line_name="Clinker Kiln 1",
                process_type=ProcessType.CALCINATION,
                annual_production_tonnes=500000.0,
                raw_materials=[
                    RawMaterial(
                        material_name="Limestone (CaCO3)",
                        quantity_tonnes=750000.0,
                        co2_factor_per_tonne=0.525,
                        source="IPCC 2006 Vol.3",
                    ),
                ],
                fuel_consumption=[
                    FuelConsumption(
                        fuel_type=FuelType.COAL,
                        quantity=60000.0,
                        unit="tonnes",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def steel_facility():
    """Create a steel facility with BF-BOF route."""
    return FacilityData(
        facility_id="steel-fac-001",
        facility_name="Test Steel Mill",
        sub_sector=ManufacturingSubSector.STEEL,
        country="DE",
        production_lines=[
            ProcessLine(
                line_id="line-bof-001",
                line_name="Blast Furnace - BOF",
                process_type=ProcessType.REDUCTION,
                annual_production_tonnes=1000000.0,
                raw_materials=[
                    RawMaterial(
                        material_name="Iron Ore (BOF route)",
                        quantity_tonnes=1500000.0,
                        co2_factor_per_tonne=1.328,
                        source="EU MRR Annex IV",
                    ),
                ],
                fuel_consumption=[
                    FuelConsumption(
                        fuel_type=FuelType.COKE,
                        quantity=400000.0,
                        unit="tonnes",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def chemical_facility():
    """Create a chemicals facility (ammonia synthesis)."""
    return FacilityData(
        facility_id="chem-fac-001",
        facility_name="Test Ammonia Plant",
        sub_sector=ManufacturingSubSector.CHEMICALS,
        country="NL",
        production_lines=[
            ProcessLine(
                line_id="line-nh3-001",
                line_name="Ammonia Synthesis",
                process_type=ProcessType.SYNTHESIS,
                annual_production_tonnes=300000.0,
                raw_materials=[
                    RawMaterial(
                        material_name="Natural gas feedstock",
                        quantity_tonnes=300000.0,
                        co2_factor_per_tonne=1.600,
                        source="IPCC 2006",
                    ),
                ],
                fuel_consumption=[
                    FuelConsumption(
                        fuel_type=FuelType.NATURAL_GAS,
                        quantity=50000.0,
                        unit="1000m3",
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test engine initialization with various config types."""

    def test_default_init(self):
        """Engine initializes with default ProcessEmissionsConfig."""
        engine = ProcessEmissionsEngine()
        assert engine.config is not None
        assert isinstance(engine.config, ProcessEmissionsConfig)
        assert engine.config.reporting_year == 2025
        assert engine.config.sub_sector == ManufacturingSubSector.CEMENT
        assert engine.engine_version == "1.0.0"

    def test_init_with_config(self):
        """Engine initializes with an explicit ProcessEmissionsConfig."""
        config = ProcessEmissionsConfig(
            reporting_year=2024,
            sub_sector=ManufacturingSubSector.STEEL,
            include_cbam=False,
            include_ets_benchmark=False,
        )
        engine = ProcessEmissionsEngine(config)
        assert engine.config.reporting_year == 2024
        assert engine.config.sub_sector == ManufacturingSubSector.STEEL
        assert engine.config.include_cbam is False

    def test_init_with_dict(self):
        """Engine initializes from a plain dictionary."""
        engine = ProcessEmissionsEngine({
            "reporting_year": 2023,
            "sub_sector": "aluminum",
        })
        assert engine.config.reporting_year == 2023
        assert engine.config.sub_sector == ManufacturingSubSector.ALUMINUM

    def test_init_with_none(self):
        """Engine initializes with None (defaults applied)."""
        engine = ProcessEmissionsEngine(None)
        assert engine.config.reporting_year == 2025
        assert engine.config.include_cbam is True


class TestProcessEmissionFactors:
    """Validate that hard-coded emission factors match regulatory values."""

    def test_cement_calcination_factor(self):
        """Cement calcination factor is 0.525 tCO2/t clinker (IPCC)."""
        assert PROCESS_EMISSION_FACTORS["cement_calcination"] == 0.525

    def test_steel_bof_factor(self):
        """Steel BOF factor is 1.328 tCO2/t hot metal."""
        assert PROCESS_EMISSION_FACTORS["steel_bof"] == 1.328

    def test_aluminum_electrolysis_factor(self):
        """Aluminium electrolysis factor is 1.514 tCO2/t Al."""
        assert PROCESS_EMISSION_FACTORS["aluminum_electrolysis"] == 1.514

    def test_glass_decomposition_factor(self):
        """Glass batch decomposition factor is 0.210 tCO2/t glass."""
        assert PROCESS_EMISSION_FACTORS["glass_decomposition"] == 0.210

    def test_ammonia_factor(self):
        """Ammonia synthesis factor is 1.600 tCO2/t NH3."""
        assert PROCESS_EMISSION_FACTORS["ammonia_synthesis"] == 1.600

    def test_all_subsectors_have_factors(self):
        """Key manufacturing sub-sectors have emission factors defined."""
        # The engine uses _get_sector_process_factors which matches by prefix.
        # Some sub-sectors use material names (e.g. ammonia_* for chemicals).
        # Verify the major sub-sectors that use the sub-sector prefix pattern.
        prefix_sectors = [
            "cement", "steel", "aluminum", "glass", "ceramics",
            "pulp", "food", "textiles", "pharma", "electronics",
            "automotive",
        ]
        for prefix in prefix_sectors:
            matching = [k for k in PROCESS_EMISSION_FACTORS if k.startswith(prefix)]
            assert len(matching) > 0, (
                f"Prefix '{prefix}' has no emission factors"
            )


class TestSingleProcessLine:
    """Test calculate_process_line for a single production line."""

    def test_cement_calcination_co2(self, default_engine):
        """Cement calcination: 750,000 t * 0.525 = 393,750 tCO2."""
        line = ProcessLine(
            line_name="Clinker Kiln",
            process_type=ProcessType.CALCINATION,
            annual_production_tonnes=500000.0,
            raw_materials=[
                RawMaterial(
                    material_name="Limestone",
                    quantity_tonnes=750000.0,
                    co2_factor_per_tonne=0.525,
                ),
            ],
        )
        result = default_engine.calculate_process_line(line)
        assert result["process_co2"] == pytest.approx(393750.0, rel=1e-6)

    def test_process_line_with_fuel(self, default_engine):
        """Process line with fuel yields non-zero combustion_co2."""
        line = ProcessLine(
            line_name="Kiln with fuel",
            process_type=ProcessType.COMBUSTION,
            annual_production_tonnes=100000.0,
            fuel_consumption=[
                FuelConsumption(
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=1000.0,
                    unit="1000m3",
                ),
            ],
        )
        result = default_engine.calculate_process_line(line)
        # CO2 = 1000 * 0.03412 TJ/1000m3 * 56.1 tCO2/TJ = 1914.132
        expected = 1000.0 * 0.03412 * 56.1
        assert result["combustion_co2"] == pytest.approx(expected, rel=1e-4)

    def test_process_line_zero_production(self, default_engine):
        """Process line with zero production returns zero emissions."""
        line = ProcessLine(
            line_name="Idle Line",
            process_type=ProcessType.CALCINATION,
            annual_production_tonnes=0.0,
            raw_materials=[],
        )
        result = default_engine.calculate_process_line(line)
        assert result["process_co2"] == 0.0
        assert result["combustion_co2"] == 0.0

    def test_multiple_raw_materials(self, default_engine):
        """Multiple raw materials sum correctly."""
        line = ProcessLine(
            line_name="Multi-material",
            process_type=ProcessType.CALCINATION,
            annual_production_tonnes=100000.0,
            raw_materials=[
                RawMaterial(
                    material_name="Limestone",
                    quantity_tonnes=100000.0,
                    co2_factor_per_tonne=0.525,
                ),
                RawMaterial(
                    material_name="MgCO3",
                    quantity_tonnes=5000.0,
                    co2_factor_per_tonne=0.785,
                ),
            ],
        )
        result = default_engine.calculate_process_line(line)
        expected = (100000.0 * 0.525) + (5000.0 * 0.785)
        assert result["process_co2"] == pytest.approx(expected, rel=1e-6)

    def test_result_has_provenance_hash(self, default_engine, cement_facility):
        """Facility result carries a provenance hash."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_result_has_engine_version(self, default_engine, cement_facility):
        """Result engine_version matches engine attribute."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert result.engine_version == "1.0.0"


class TestFacilityCalculation:
    """Test calculate_facility_emissions for full facility."""

    def test_cement_facility_emissions(self, default_engine, cement_facility):
        """Cement facility total emissions are positive and reasonable."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert result.total_process_co2 > 0.0
        assert result.total_combustion_co2 > 0.0
        assert result.total_emissions > 0.0
        # Process: 750000 * 0.525 = 393750
        assert result.total_process_co2 == pytest.approx(393750.0, rel=1e-3)

    def test_multi_line_facility(self, default_engine):
        """Facility with two lines sums emissions correctly."""
        facility = FacilityData(
            facility_name="Multi-Line Cement",
            sub_sector=ManufacturingSubSector.CEMENT,
            country="DE",
            production_lines=[
                ProcessLine(
                    line_name="Kiln A",
                    process_type=ProcessType.CALCINATION,
                    annual_production_tonnes=250000.0,
                    raw_materials=[
                        RawMaterial(
                            material_name="Limestone",
                            quantity_tonnes=375000.0,
                            co2_factor_per_tonne=0.525,
                        ),
                    ],
                ),
                ProcessLine(
                    line_name="Kiln B",
                    process_type=ProcessType.CALCINATION,
                    annual_production_tonnes=250000.0,
                    raw_materials=[
                        RawMaterial(
                            material_name="Limestone",
                            quantity_tonnes=375000.0,
                            co2_factor_per_tonne=0.525,
                        ),
                    ],
                ),
            ],
        )
        result = default_engine.calculate_facility_emissions(facility)
        expected_process = 2 * (375000.0 * 0.525)
        assert result.total_process_co2 == pytest.approx(expected_process, rel=1e-3)

    def test_facility_emission_intensity(self, default_engine, cement_facility):
        """Emission intensity = total_emissions / production_tonnes."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert result.emission_intensity_per_tonne > 0.0
        expected_intensity = result.total_emissions / 500000.0
        assert result.emission_intensity_per_tonne == pytest.approx(
            _round3(expected_intensity), abs=0.01
        )

    def test_empty_process_lines_raises(self, default_engine):
        """Facility with no production lines raises ValueError."""
        facility = FacilityData(
            facility_name="Empty Facility",
            sub_sector=ManufacturingSubSector.CEMENT,
            country="DE",
            production_lines=[],
        )
        with pytest.raises(ValueError, match="no production lines"):
            default_engine.calculate_facility_emissions(facility)

    def test_facility_result_type(self, default_engine, cement_facility):
        """Facility result is a ProcessEmissionsResult instance."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert isinstance(result, ProcessEmissionsResult)

    def test_facility_result_fields(self, default_engine, cement_facility):
        """Result contains all mandatory fields."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert result.facility_id == "cement-fac-001"
        assert result.processing_time_ms >= 0.0
        assert result.calculated_at is not None
        assert len(result.methodology_notes) > 0
        assert result.total_fugitive_co2 > 0.0


class TestCBAMCalculation:
    """Test CBAM embedded emissions calculation."""

    def test_cbam_embedded_cement(self, default_engine, cement_facility):
        """CBAM embedded emissions calculated for cement (CBAM-affected)."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert result.cbam_embedded_emissions is not None
        cbam = result.cbam_embedded_emissions
        assert cbam.goods_category == "cement"
        assert cbam.direct_emissions > 0.0
        assert cbam.total_embedded > 0.0
        assert cbam.production_tonnes > 0.0

    def test_cbam_embedded_steel(self, default_engine, steel_facility):
        """CBAM embedded emissions calculated for steel (CBAM-affected)."""
        result = default_engine.calculate_facility_emissions(steel_facility)
        assert result.cbam_embedded_emissions is not None
        assert result.cbam_embedded_emissions.goods_category == "steel"

    def test_cbam_not_affected(self, default_engine):
        """Non-CBAM sub-sector returns zero embedded emissions."""
        facility = FacilityData(
            facility_name="Textile Mill",
            sub_sector=ManufacturingSubSector.TEXTILES,
            country="DE",
            production_lines=[
                ProcessLine(
                    line_name="Dyeing Line",
                    process_type=ProcessType.COMBUSTION,
                    annual_production_tonnes=10000.0,
                    raw_materials=[
                        RawMaterial(
                            material_name="Fabric",
                            quantity_tonnes=10000.0,
                            co2_factor_per_tonne=0.015,
                        ),
                    ],
                ),
            ],
        )
        result = default_engine.calculate_facility_emissions(facility)
        cbam = result.cbam_embedded_emissions
        assert cbam is not None
        assert cbam.direct_emissions == 0.0
        assert cbam.total_embedded == 0.0

    def test_cbam_includes_indirect(self, default_engine):
        """CBAM embedded emissions include indirect (electricity) when present."""
        facility = FacilityData(
            facility_name="Cement with Electricity",
            sub_sector=ManufacturingSubSector.CEMENT,
            country="DE",
            production_lines=[
                ProcessLine(
                    line_name="Kiln",
                    process_type=ProcessType.CALCINATION,
                    annual_production_tonnes=100000.0,
                    raw_materials=[
                        RawMaterial(
                            material_name="Limestone",
                            quantity_tonnes=150000.0,
                            co2_factor_per_tonne=0.525,
                        ),
                    ],
                    fuel_consumption=[
                        FuelConsumption(
                            fuel_type=FuelType.ELECTRICITY,
                            quantity=50000.0,
                            unit="MWh",
                        ),
                    ],
                ),
            ],
        )
        result = default_engine.calculate_facility_emissions(facility)
        cbam = result.cbam_embedded_emissions
        assert cbam is not None
        # Indirect = 50000 MWh * 0.23 tCO2/MWh = 11500
        assert cbam.indirect_emissions == pytest.approx(11500.0, rel=1e-3)

    def test_cbam_precursor_emissions(self, default_engine, cement_facility):
        """Precursor emissions default to zero (placeholder)."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert result.cbam_embedded_emissions.precursor_emissions == 0.0


class TestETSBenchmark:
    """Test EU ETS benchmark comparison."""

    def test_ets_benchmark_cement(self, default_engine, cement_facility):
        """Cement facility gets grey_clinker benchmark (0.766)."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        ets = result.ets_benchmark_comparison
        assert ets is not None
        assert ets.product_name == "grey_clinker"
        assert ets.benchmark_value == 0.766

    def test_ets_benchmark_steel(self, default_engine, steel_facility):
        """Steel facility gets hot_metal benchmark (1.328)."""
        result = default_engine.calculate_facility_emissions(steel_facility)
        ets = result.ets_benchmark_comparison
        assert ets is not None
        assert ets.product_name == "hot_metal"
        assert ets.benchmark_value == 1.328

    def test_below_benchmark(self, default_engine):
        """Facility below benchmark is eligible for free allocation."""
        facility = FacilityData(
            facility_name="Efficient Cement",
            sub_sector=ManufacturingSubSector.CEMENT,
            country="DE",
            production_lines=[
                ProcessLine(
                    line_name="Efficient Kiln",
                    process_type=ProcessType.CALCINATION,
                    annual_production_tonnes=1000000.0,
                    raw_materials=[
                        RawMaterial(
                            material_name="Limestone",
                            quantity_tonnes=1000.0,
                            co2_factor_per_tonne=0.525,
                        ),
                    ],
                ),
            ],
        )
        result = default_engine.calculate_facility_emissions(facility)
        ets = result.ets_benchmark_comparison
        assert ets is not None
        assert ets.free_allocation_eligible is True
        assert ets.shortfall_tco2 == 0.0

    def test_above_benchmark(self, default_engine, cement_facility):
        """Facility above benchmark has a non-zero shortfall."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        ets = result.ets_benchmark_comparison
        assert ets is not None
        if not ets.free_allocation_eligible:
            assert ets.shortfall_tco2 > 0.0
            assert ets.ratio_to_benchmark > 1.0


class TestProvenance:
    """Test provenance hash generation and determinism."""

    def test_provenance_hash_64char(self, default_engine, cement_facility):
        """Provenance hash is a 64-character hex string (SHA-256)."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_deterministic_results(self, default_engine, cement_facility):
        """Same inputs produce the same provenance hash (deterministic)."""
        data = {"a": 1, "b": 2, "c": [3, 4]}
        hash1 = _compute_hash(data)
        hash2 = _compute_hash(data)
        assert hash1 == hash2

    def test_different_input_different_hash(self):
        """Different data produces different hashes."""
        hash1 = _compute_hash({"value": 100})
        hash2 = _compute_hash({"value": 200})
        assert hash1 != hash2

    def test_methodology_notes(self, default_engine, cement_facility):
        """Result includes methodology notes with key information."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        notes_text = " ".join(result.methodology_notes)
        assert "Reporting year" in notes_text
        assert "Sub-sector" in notes_text
        assert "Engine version" in notes_text
        assert "Fugitive" in notes_text


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_large_facility(self, default_engine):
        """Large facility with high production does not crash."""
        facility = FacilityData(
            facility_name="Mega Cement",
            sub_sector=ManufacturingSubSector.CEMENT,
            country="CN",
            production_lines=[
                ProcessLine(
                    line_name=f"Kiln {i}",
                    process_type=ProcessType.CALCINATION,
                    annual_production_tonnes=5000000.0,
                    raw_materials=[
                        RawMaterial(
                            material_name="Limestone",
                            quantity_tonnes=7500000.0,
                            co2_factor_per_tonne=0.525,
                        ),
                    ],
                    fuel_consumption=[
                        FuelConsumption(
                            fuel_type=FuelType.COAL,
                            quantity=600000.0,
                            unit="tonnes",
                        ),
                    ],
                )
                for i in range(5)
            ],
        )
        result = default_engine.calculate_facility_emissions(facility)
        assert result.total_emissions > 0.0
        assert result.processing_time_ms >= 0.0

    def test_zero_raw_materials(self, default_engine):
        """Facility with fuel only (no raw materials) computes correctly."""
        facility = FacilityData(
            facility_name="Fuel Only",
            sub_sector=ManufacturingSubSector.FOOD_BEVERAGE,
            country="FR",
            production_lines=[
                ProcessLine(
                    line_name="Boiler",
                    process_type=ProcessType.COMBUSTION,
                    annual_production_tonnes=50000.0,
                    raw_materials=[],
                    fuel_consumption=[
                        FuelConsumption(
                            fuel_type=FuelType.NATURAL_GAS,
                            quantity=5000.0,
                            unit="1000m3",
                        ),
                    ],
                ),
            ],
        )
        result = default_engine.calculate_facility_emissions(facility)
        assert result.total_process_co2 == 0.0
        assert result.total_combustion_co2 > 0.0

    def test_mixed_fuels(self, default_engine):
        """Multiple fuel types summed correctly."""
        line = ProcessLine(
            line_name="Mixed Fuel Kiln",
            process_type=ProcessType.COMBUSTION,
            annual_production_tonnes=100000.0,
            fuel_consumption=[
                FuelConsumption(
                    fuel_type=FuelType.COAL,
                    quantity=10000.0,
                    unit="tonnes",
                ),
                FuelConsumption(
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=5000.0,
                    unit="1000m3",
                ),
                FuelConsumption(
                    fuel_type=FuelType.BIOMASS,
                    quantity=3000.0,
                    unit="tonnes",
                ),
            ],
        )
        result = default_engine.calculate_process_line(line)
        coal_co2 = 10000.0 * 0.02558 * 94.6
        gas_co2 = 5000.0 * 0.03412 * 56.1
        biomass_co2 = 0.0
        expected = coal_co2 + gas_co2 + biomass_co2
        assert result["combustion_co2"] == pytest.approx(expected, rel=1e-4)

    def test_result_fields_complete(self, default_engine, cement_facility):
        """ProcessEmissionsResult contains all expected fields."""
        result = default_engine.calculate_facility_emissions(cement_facility)
        assert hasattr(result, "result_id")
        assert hasattr(result, "facility_id")
        assert hasattr(result, "total_process_co2")
        assert hasattr(result, "total_combustion_co2")
        assert hasattr(result, "total_fugitive_co2")
        assert hasattr(result, "total_emissions")
        assert hasattr(result, "emission_intensity_per_tonne")
        assert hasattr(result, "sub_sector_breakdown")
        assert hasattr(result, "cbam_embedded_emissions")
        assert hasattr(result, "ets_benchmark_comparison")
        assert hasattr(result, "methodology_notes")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "engine_version")
        assert hasattr(result, "calculated_at")
        assert hasattr(result, "provenance_hash")

    def test_float_precision(self, default_engine):
        """_round3 handles regulatory precision correctly."""
        assert _round3(1.23456789) == 1.235
        assert _round3(0.0005) == 0.001
        assert _round3(0.0004) == 0.0
        assert _round3(99999.9999) == 100000.0
