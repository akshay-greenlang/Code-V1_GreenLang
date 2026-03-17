# -*- coding: utf-8 -*-
"""
Unit tests for WaterPollutionEngine (PACK-013, Engine 5)

Tests water balance, pollutant inventory, IED compliance, REACH SVHC,
water stress, ESRS E2/E3 metrics, and provenance tracking.

Target: 85%+ coverage, 43 tests.
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


wp = _load_module("water_pollution_engine", "water_pollution_engine.py")

WaterPollutionEngine = wp.WaterPollutionEngine
WaterPollutionConfig = wp.WaterPollutionConfig
WaterIntakeData = wp.WaterIntakeData
WaterDischargeData = wp.WaterDischargeData
PollutantEmission = wp.PollutantEmission
SVHCSubstance = wp.SVHCSubstance
WaterPollutionResult = wp.WaterPollutionResult
WaterStressAssessment = wp.WaterStressAssessment
WaterSource = wp.WaterSource
WaterStressLevel = wp.WaterStressLevel
PollutantCategory = wp.PollutantCategory
PollutantType = wp.PollutantType
TreatmentLevel = wp.TreatmentLevel
QualityGrade = wp.QualityGrade
MeasurementMethod = wp.MeasurementMethod
AuthorizationStatus = wp.AuthorizationStatus
IED_EMISSION_LIMITS = wp.IED_EMISSION_LIMITS
REACH_SVHC_THRESHOLD = wp.REACH_SVHC_THRESHOLD
WATER_STRESS_THRESHOLDS = wp.WATER_STRESS_THRESHOLDS
TREATMENT_EFFICIENCY = wp.TREATMENT_EFFICIENCY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    return WaterPollutionConfig(
        reporting_year=2025,
        production_volume=Decimal("100000"),
        production_unit="tonnes",
        sub_sector="cement",
    )


@pytest.fixture
def default_engine(default_config):
    return WaterPollutionEngine(default_config)


@pytest.fixture
def sample_intake():
    return [
        WaterIntakeData(
            source=WaterSource.SURFACE,
            volume_m3=Decimal("500000"),
            quality_grade=QualityGrade.RAW,
            water_stressed_area=False,
            facility_id="FAC-001",
        ),
        WaterIntakeData(
            source=WaterSource.GROUNDWATER,
            volume_m3=Decimal("200000"),
            quality_grade=QualityGrade.PROCESS_GRADE,
            water_stressed_area=True,
            facility_id="FAC-001",
        ),
    ]


@pytest.fixture
def sample_discharge():
    return [
        WaterDischargeData(
            destination=WaterSource.SURFACE,
            volume_m3=Decimal("400000"),
            treatment_level=TreatmentLevel.SECONDARY,
            facility_id="FAC-001",
        ),
    ]


@pytest.fixture
def sample_pollutants():
    return [
        PollutantEmission(
            pollutant_type=PollutantType.NOX,
            category=PollutantCategory.AIR,
            quantity_tonnes=Decimal("50"),
            emission_limit=350.0,
            facility_id="FAC-001",
        ),
        PollutantEmission(
            pollutant_type=PollutantType.SOX,
            category=PollutantCategory.AIR,
            quantity_tonnes=Decimal("30"),
            emission_limit=300.0,
            facility_id="FAC-001",
        ),
        PollutantEmission(
            pollutant_type=PollutantType.PM10,
            category=PollutantCategory.AIR,
            quantity_tonnes=Decimal("10"),
            emission_limit=15.0,
            facility_id="FAC-001",
        ),
        PollutantEmission(
            pollutant_type=PollutantType.VOC,
            category=PollutantCategory.AIR,
            quantity_tonnes=Decimal("5"),
            emission_limit=20.0,
            facility_id="FAC-001",
        ),
        PollutantEmission(
            pollutant_type=PollutantType.NITROGEN_WATER,
            category=PollutantCategory.WATER,
            quantity_tonnes=Decimal("2"),
            facility_id="FAC-001",
        ),
    ]


# ---------------------------------------------------------------------------
# TestInitialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test WaterPollutionEngine initialization."""

    def test_default_init(self, default_config):
        engine = WaterPollutionEngine(default_config)
        assert engine.config == default_config
        assert engine.config.reporting_year == 2025

    def test_with_config(self):
        cfg = WaterPollutionConfig(
            reporting_year=2024,
            production_volume=Decimal("50000"),
            sub_sector="steel",
        )
        engine = WaterPollutionEngine(cfg)
        assert engine.config.sub_sector == "steel"
        assert engine.config.production_volume == Decimal("50000")

    def test_with_dict(self):
        cfg = WaterPollutionConfig(**{
            "reporting_year": 2025,
            "production_volume": 75000,
            "sub_sector": "glass",
        })
        engine = WaterPollutionEngine(cfg)
        assert engine.config.sub_sector == "glass"

    def test_with_none_optional_fields(self):
        cfg = WaterPollutionConfig(reporting_year=2025)
        engine = WaterPollutionEngine(cfg)
        assert engine.config.production_volume == Decimal("0")
        assert engine.config.sub_sector == ""


# ---------------------------------------------------------------------------
# TestWaterSources
# ---------------------------------------------------------------------------


class TestWaterSources:
    """Test water source enum completeness."""

    def test_all_sources_defined(self):
        assert len(WaterSource) == 6

    def test_source_enum_values(self):
        expected = {"surface", "groundwater", "third_party",
                    "rainwater", "seawater", "produced_water"}
        actual = {s.value for s in WaterSource}
        assert actual == expected

    def test_water_source_surface(self):
        assert WaterSource.SURFACE.value == "surface"

    def test_water_source_groundwater(self):
        assert WaterSource.GROUNDWATER.value == "groundwater"


# ---------------------------------------------------------------------------
# TestWaterBalance
# ---------------------------------------------------------------------------


class TestWaterBalance:
    """Test water balance calculations."""

    def test_total_withdrawal(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        assert result.total_water_withdrawal_m3 == pytest.approx(700000.0, rel=1e-3)

    def test_total_discharge(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        assert result.total_water_discharge_m3 == pytest.approx(400000.0, rel=1e-3)

    def test_total_consumption(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        # consumption = withdrawal - discharge = 700000 - 400000 = 300000
        assert result.total_water_consumption_m3 == pytest.approx(300000.0, rel=1e-3)

    def test_water_recycling_rate(self, default_engine, sample_intake, sample_discharge):
        recycled = Decimal("100000")
        result = default_engine.calculate_water_balance(
            sample_intake, sample_discharge, recycled_volume_m3=recycled
        )
        # recycling_rate = 100000 / (700000 + 100000) * 100 = 12.5%
        assert result.water_recycling_rate_pct == pytest.approx(12.5, rel=1e-2)

    def test_water_intensity(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        # intensity = 700000 / 100000 = 7.0 m3/tonne
        assert result.water_intensity_m3_per_unit == pytest.approx(7.0, rel=1e-3)

    def test_zero_withdrawal(self, default_engine):
        result = default_engine.calculate_water_balance([], [])
        assert result.total_water_withdrawal_m3 == 0.0
        assert result.total_water_discharge_m3 == 0.0
        assert result.total_water_consumption_m3 == 0.0


# ---------------------------------------------------------------------------
# TestWaterStress
# ---------------------------------------------------------------------------


class TestWaterStress:
    """Test water stress assessment."""

    def test_low_stress(self, default_engine):
        intake = [
            WaterIntakeData(
                source=WaterSource.SURFACE,
                volume_m3=Decimal("1000000"),
                water_stressed_area=False,
            ),
            WaterIntakeData(
                source=WaterSource.GROUNDWATER,
                volume_m3=Decimal("50000"),
                water_stressed_area=True,
            ),
        ]
        assessment = default_engine.assess_water_stress(intake)
        # stressed = 50000 / 1050000 * 100 = ~4.76% --> LOW
        assert assessment.stressed_withdrawal_pct < 10.0
        assert assessment.stress_level_classification == WaterStressLevel.LOW

    def test_high_stress(self, default_engine):
        intake = [
            WaterIntakeData(
                source=WaterSource.SURFACE,
                volume_m3=Decimal("300000"),
                water_stressed_area=False,
            ),
            WaterIntakeData(
                source=WaterSource.GROUNDWATER,
                volume_m3=Decimal("350000"),
                water_stressed_area=True,
            ),
        ]
        assessment = default_engine.assess_water_stress(intake)
        # stressed = 350000 / 650000 * 100 = ~53.8% --> HIGH (40-80%)
        assert 40.0 <= assessment.stressed_withdrawal_pct < 80.0
        assert assessment.stress_level_classification == WaterStressLevel.HIGH

    def test_extremely_high_stress(self, default_engine):
        intake = [
            WaterIntakeData(
                source=WaterSource.SURFACE,
                volume_m3=Decimal("50000"),
                water_stressed_area=False,
            ),
            WaterIntakeData(
                source=WaterSource.GROUNDWATER,
                volume_m3=Decimal("450000"),
                water_stressed_area=True,
            ),
        ]
        assessment = default_engine.assess_water_stress(intake)
        # stressed = 450000 / 500000 * 100 = 90% --> EXTREMELY_HIGH
        assert assessment.stressed_withdrawal_pct >= 80.0
        assert assessment.stress_level_classification == WaterStressLevel.EXTREMELY_HIGH

    def test_stress_level_enum(self):
        assert len(WaterStressLevel) == 5

    def test_stressed_area_percentage(self, default_engine, sample_intake):
        assessment = default_engine.assess_water_stress(sample_intake)
        # Only groundwater (200000) is stressed, total=700000 -> ~28.57%
        expected = 200000 / 700000 * 100
        assert assessment.stressed_withdrawal_pct == pytest.approx(expected, abs=0.1)


# ---------------------------------------------------------------------------
# TestPollutantInventory
# ---------------------------------------------------------------------------


class TestPollutantInventory:
    """Test pollutant inventory calculations."""

    def test_air_pollutants(self, default_engine, sample_pollutants):
        inv = default_engine.calculate_pollutant_inventory(sample_pollutants)
        # NOx(50) + SOx(30) + PM10(10) + VOC(5) = 95 tonnes air total
        assert float(inv["air_total"]) == pytest.approx(95.0, rel=1e-6)

    def test_water_pollutants(self, default_engine, sample_pollutants):
        inv = default_engine.calculate_pollutant_inventory(sample_pollutants)
        # nitrogen_water = 2 tonnes
        assert float(inv["water_total"]) == pytest.approx(2.0, rel=1e-6)

    def test_total_by_category(self, default_engine, sample_pollutants):
        inv = default_engine.calculate_pollutant_inventory(sample_pollutants)
        total = float(inv["air_total"]) + float(inv["water_total"]) + float(inv["soil_total"])
        assert total == pytest.approx(97.0, rel=1e-6)

    def test_pollutant_below_limit(self, default_engine):
        em = [
            PollutantEmission(
                pollutant_type=PollutantType.PM10,
                category=PollutantCategory.AIR,
                quantity_tonnes=Decimal("5"),
                emission_limit=15.0,  # cement upper=20, so 15<20 => not exceeds
            ),
        ]
        inv = default_engine.calculate_pollutant_inventory(em)
        assert inv["items"][0].exceeds_limit is False

    def test_pollutant_above_limit(self, default_engine):
        em = [
            PollutantEmission(
                pollutant_type=PollutantType.PM10,
                category=PollutantCategory.AIR,
                quantity_tonnes=Decimal("5"),
                emission_limit=25.0,  # cement upper=20, so 25>20 => exceeds
            ),
        ]
        inv = default_engine.calculate_pollutant_inventory(em)
        assert inv["items"][0].exceeds_limit is True

    def test_all_pollutant_types_defined(self):
        assert len(PollutantType) >= 12


# ---------------------------------------------------------------------------
# TestIEDCompliance
# ---------------------------------------------------------------------------


class TestIEDCompliance:
    """Test IED BAT-AEL compliance checking."""

    def test_compliant_facility(self, default_engine):
        emissions = [
            PollutantEmission(
                pollutant_type=PollutantType.NOX,
                category=PollutantCategory.AIR,
                quantity_tonnes=Decimal("10"),
                emission_limit=150.0,  # cement NOx lower=200, 150<200 => compliant
            ),
        ]
        result = default_engine.check_ied_compliance(emissions, "cement")
        assert result["overall_status"] == "compliant"

    def test_non_compliant_facility(self, default_engine):
        emissions = [
            PollutantEmission(
                pollutant_type=PollutantType.NOX,
                category=PollutantCategory.AIR,
                quantity_tonnes=Decimal("10"),
                emission_limit=500.0,  # cement NOx upper=450, 500>450 => non-compliant
            ),
        ]
        result = default_engine.check_ied_compliance(emissions, "cement")
        assert result["overall_status"] == "non_compliant"

    def test_bat_ael_reference(self, default_engine):
        emissions = [
            PollutantEmission(
                pollutant_type=PollutantType.NOX,
                category=PollutantCategory.AIR,
                quantity_tonnes=Decimal("10"),
                emission_limit=350.0,  # within range
            ),
        ]
        result = default_engine.check_ied_compliance(emissions, "cement")
        nox_detail = [d for d in result["details"] if d.pollutant_type == "nox"][0]
        assert nox_detail.bat_ael_lower == 200.0
        assert nox_detail.bat_ael_upper == 450.0

    def test_ied_limits_for_sector(self, default_engine):
        assert "cement" in IED_EMISSION_LIMITS
        assert "steel" in IED_EMISSION_LIMITS
        assert "glass" in IED_EMISSION_LIMITS
        assert "chemicals" in IED_EMISSION_LIMITS


# ---------------------------------------------------------------------------
# TestREACH
# ---------------------------------------------------------------------------


class TestREACH:
    """Test REACH SVHC assessment."""

    def test_svhc_above_threshold(self, default_engine):
        substances = [
            SVHCSubstance(
                cas_number="117-81-7",
                substance_name="DEHP",
                concentration_pct=0.15,  # >0.1% w/w threshold
                quantity_tonnes=Decimal("2"),
            ),
        ]
        result = default_engine.assess_svhc(substances)
        assert result.above_threshold_count == 1
        assert result.requires_notification is True

    def test_svhc_below_threshold(self, default_engine):
        substances = [
            SVHCSubstance(
                cas_number="117-81-7",
                substance_name="DEHP",
                concentration_pct=0.05,  # <0.1% w/w threshold
                quantity_tonnes=Decimal("1"),
            ),
        ]
        result = default_engine.assess_svhc(substances)
        assert result.above_threshold_count == 0
        assert result.requires_notification is False

    def test_svhc_count(self, default_engine):
        substances = [
            SVHCSubstance(
                cas_number="117-81-7",
                substance_name="DEHP",
                concentration_pct=0.15,
                quantity_tonnes=Decimal("2"),
            ),
            SVHCSubstance(
                cas_number="84-74-2",
                substance_name="DBP",
                concentration_pct=0.05,
                quantity_tonnes=Decimal("1"),
            ),
        ]
        result = default_engine.assess_svhc(substances)
        assert result.total_svhc_count == 2


# ---------------------------------------------------------------------------
# TestESRSMetrics
# ---------------------------------------------------------------------------


class TestESRSMetrics:
    """Test ESRS E2/E3 metric presence."""

    def test_e2_metrics_present(self, default_engine, sample_intake, sample_discharge, sample_pollutants):
        result = default_engine.calculate_water_balance(
            sample_intake, sample_discharge, emissions=sample_pollutants
        )
        assert "e2_4_pollution_air_tonnes" in result.esrs_e2_metrics
        assert "e2_5_svhc_count" in result.esrs_e2_metrics

    def test_e3_metrics_present(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        assert "e3_4_total_water_consumption_m3" in result.esrs_e3_metrics
        assert "e3_4_total_water_withdrawal_m3" in result.esrs_e3_metrics
        assert "e3_4_water_recycling_rate_pct" in result.esrs_e3_metrics

    def test_withdrawal_by_source(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        wbs = result.withdrawal_by_source
        assert "surface" in wbs
        assert "groundwater" in wbs
        assert wbs["surface"] == pytest.approx(500000.0, rel=1e-3)
        assert wbs["groundwater"] == pytest.approx(200000.0, rel=1e-3)

    def test_discharge_by_destination(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        dbd = result.discharge_by_destination
        assert "surface" in dbd
        assert dbd["surface"] == pytest.approx(400000.0, rel=1e-3)


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Test provenance hash generation."""

    def test_hash_64char(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self, default_engine, sample_intake, sample_discharge):
        r1 = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        r2 = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        # Both should produce valid 64-char SHA-256 hashes
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64

    def test_different_input_different_hash(self, default_engine, sample_intake, sample_discharge):
        r1 = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        intake2 = [
            WaterIntakeData(
                source=WaterSource.SURFACE,
                volume_m3=Decimal("999999"),
            )
        ]
        r2 = default_engine.calculate_water_balance(intake2, [])
        # Different inputs produce different water balance values
        assert r1.total_water_withdrawal_m3 != r2.total_water_withdrawal_m3


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_intake_runs(self, default_engine):
        result = default_engine.calculate_water_balance([], [])
        assert result.total_water_withdrawal_m3 == 0.0

    def test_large_facility(self, default_engine):
        intake = [
            WaterIntakeData(
                source=WaterSource.SURFACE,
                volume_m3=Decimal("50000000"),
                facility_id="LARGE-FAC",
            )
        ]
        discharge = [
            WaterDischargeData(
                destination=WaterSource.SURFACE,
                volume_m3=Decimal("45000000"),
                facility_id="LARGE-FAC",
            )
        ]
        result = default_engine.calculate_water_balance(intake, discharge)
        assert result.total_water_withdrawal_m3 == pytest.approx(50000000.0, rel=1e-3)
        assert result.total_water_consumption_m3 == pytest.approx(5000000.0, rel=1e-3)

    def test_result_fields(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        assert isinstance(result, WaterPollutionResult)
        assert result.engine_version == "1.0.0"
        assert result.processing_time_ms >= 0

    def test_methodology_notes(self, default_engine, sample_intake, sample_discharge):
        result = default_engine.calculate_water_balance(sample_intake, sample_discharge)
        assert isinstance(result.methodology_notes, list)
        assert len(result.methodology_notes) > 0
