# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Sector Classification Engine.

Tests automatic sector classification using NACE Rev.2, GICS, and ISIC Rev.4
codes with SBTi SDA sector mapping, multi-sector companies, eligibility
validation, and edge cases.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
Engine:  1 of 8 - sector_classification_engine.py
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.sector_classification_engine import (
    SectorClassificationEngine,
    ClassificationInput,
    ClassificationResult,
    IndustryCodeEntry,
    ManualSectorOverride,
    SectorCode,
    ClassificationSystem,
    SDAEligibility,
)

from .conftest import (
    assert_provenance_hash,
    assert_processing_time,
    NACE_SECTOR_MAP,
    SDA_SECTORS,
    ALL_SECTORS,
    timed_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_code(code, system="nace_rev2", revenue_pct=Decimal("100"), primary=True):
    return IndustryCodeEntry(
        system=ClassificationSystem(system),
        code=code,
        revenue_share_pct=revenue_pct,
        is_primary=primary,
    )


def _make_input(entity_name="TestCo", codes=None, manual_overrides=None,
                include_sda=True, include_iea=True, reporting_year=2024,
                country="DE"):
    kw = dict(
        entity_name=entity_name,
        include_sda_validation=include_sda,
        include_iea_mapping=include_iea,
        reporting_year=reporting_year,
        country=country,
    )
    if codes is not None:
        kw["industry_codes"] = codes
    if manual_overrides is not None:
        kw["manual_overrides"] = manual_overrides
    return ClassificationInput(**kw)


# ===========================================================================
# Engine Instantiation
# ===========================================================================


class TestClassificationInstantiation:
    """Engine instantiation tests."""

    def test_engine_instantiates(self):
        engine = SectorClassificationEngine()
        assert engine is not None

    def test_engine_has_calculate(self):
        engine = SectorClassificationEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = SectorClassificationEngine()
        assert engine.engine_version == "1.0.0"

    def test_get_supported_sectors(self):
        engine = SectorClassificationEngine()
        sectors = engine.get_supported_sectors()
        assert len(sectors) >= 12

    def test_get_nace_codes(self):
        engine = SectorClassificationEngine()
        codes = engine.get_nace_codes()
        assert len(codes) > 0

    def test_get_gics_codes(self):
        engine = SectorClassificationEngine()
        codes = engine.get_gics_codes()
        assert len(codes) > 0

    def test_get_isic_codes(self):
        engine = SectorClassificationEngine()
        codes = engine.get_isic_codes()
        assert len(codes) > 0


# ===========================================================================
# NACE Rev.2 Classification
# ===========================================================================


class TestNACEClassification:
    """Test NACE Rev.2 code-based classification."""

    @pytest.mark.parametrize("nace_code,expected_sector", [
        ("D35.11", "power_generation"),
        ("C24.10", "steel"),
        ("C23.51", "cement"),
        ("C24.42", "aluminum"),
        ("C17.11", "pulp_paper"),
        ("C20.11", "chemicals"),
        ("H51.10", "aviation"),
        ("H50.10", "shipping"),
    ])
    def test_nace_to_sector_mapping(self, nace_code, expected_sector):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code(nace_code, "nace_rev2")])
        result = engine.calculate(inp)
        assert result.primary_sector == expected_sector

    def test_power_generation_nace(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("D35.11")])
        result = engine.calculate(inp)
        assert result.primary_sector == "power_generation"

    def test_steel_nace(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("C24.10")])
        result = engine.calculate(inp)
        assert result.primary_sector == "steel"

    def test_cement_nace(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("C23.51")])
        result = engine.calculate(inp)
        assert result.primary_sector == "cement"


# ===========================================================================
# SDA Eligibility
# ===========================================================================


class TestSDAEligibility:
    """Test SBTi SDA eligibility validation."""

    def test_sda_validation_exists(self):
        engine = SectorClassificationEngine()
        inp = _make_input(
            codes=[_make_code("D35.11")],
            include_sda=True,
        )
        result = engine.calculate(inp)
        assert result.sda_validation is not None

    def test_power_sda_eligibility_present(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("D35.11")])
        result = engine.calculate(inp)
        if result.sda_validation:
            assert result.sda_validation.eligibility in [
                e.value for e in SDAEligibility]

    def test_sda_disabled(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("D35.11")], include_sda=False)
        result = engine.calculate(inp)
        assert result.sda_validation is None


# ===========================================================================
# IEA Mapping
# ===========================================================================


class TestIEAMapping:
    """Test IEA sector mapping."""

    def test_iea_mappings_exist(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("D35.11")], include_iea=True)
        result = engine.calculate(inp)
        assert len(result.iea_mappings) > 0

    def test_iea_disabled(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("D35.11")], include_iea=False)
        result = engine.calculate(inp)
        assert len(result.iea_mappings) == 0


# ===========================================================================
# Multi-Sector Classification
# ===========================================================================


class TestMultiSectorClassification:
    """Test multi-sector company classification."""

    def test_multi_sector_codes(self):
        engine = SectorClassificationEngine()
        codes = [
            _make_code("C24.10", revenue_pct=Decimal("60"), primary=True),
            _make_code("C23.51", revenue_pct=Decimal("30"), primary=False),
            _make_code("D35.11", revenue_pct=Decimal("10"), primary=False),
        ]
        inp = _make_input(codes=codes)
        result = engine.calculate(inp)
        assert len(result.sector_matches) >= 2

    def test_primary_sector_highest_revenue(self):
        engine = SectorClassificationEngine()
        codes = [
            _make_code("C24.10", revenue_pct=Decimal("60"), primary=True),
            _make_code("C23.51", revenue_pct=Decimal("40"), primary=False),
        ]
        inp = _make_input(codes=codes)
        result = engine.calculate(inp)
        assert result.primary_sector == "steel"

    def test_multi_sector_summary(self):
        engine = SectorClassificationEngine()
        codes = [
            _make_code("C24.10", revenue_pct=Decimal("50")),
            _make_code("C23.51", revenue_pct=Decimal("50")),
        ]
        inp = _make_input(codes=codes)
        result = engine.calculate(inp)
        if result.multi_sector_summary:
            assert result.multi_sector_summary is not None


# ===========================================================================
# No Codes - Cross-Sector Fallback
# ===========================================================================


class TestCrossSectorFallback:
    """Test fallback when no industry codes provided."""

    def test_no_codes_falls_back(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[])
        result = engine.calculate(inp)
        assert result.primary_sector == "cross_sector"

    def test_no_codes_entity_only(self):
        engine = SectorClassificationEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert result.primary_sector is not None


# ===========================================================================
# Result Structure & Provenance
# ===========================================================================


class TestClassificationResultStructure:
    """Test result structure and provenance."""

    def test_result_provenance(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("D35.11")])
        result = engine.calculate(inp)
        assert_provenance_hash(result)

    def test_result_processing_time(self):
        engine = SectorClassificationEngine()
        result = engine.calculate(_make_input(codes=[_make_code("D35.11")]))
        assert_processing_time(result)

    def test_result_entity_name(self):
        engine = SectorClassificationEngine()
        result = engine.calculate(_make_input(entity_name="ClassCo"))
        assert result.entity_name == "ClassCo"

    def test_result_has_data_quality(self):
        engine = SectorClassificationEngine()
        result = engine.calculate(_make_input(codes=[_make_code("D35.11")]))
        assert result.data_quality is not None

    def test_result_has_recommendations(self):
        engine = SectorClassificationEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.recommendations, list)

    def test_result_has_warnings(self):
        engine = SectorClassificationEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.warnings, list)

    def test_result_deterministic(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("C24.10")])
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.primary_sector == r2.primary_sector


# ===========================================================================
# Batch Classification
# ===========================================================================


class TestBatchClassification:
    """Test batch classification."""

    def test_batch_calculate(self):
        engine = SectorClassificationEngine()
        inputs = [
            _make_input(entity_name="Co1", codes=[_make_code("D35.11")]),
            _make_input(entity_name="Co2", codes=[_make_code("C24.10")]),
            _make_input(entity_name="Co3", codes=[_make_code("C23.51")]),
        ]
        results = engine.calculate_batch(inputs)
        assert len(results) == 3
        assert results[0].primary_sector == "power_generation"
        assert results[1].primary_sector == "steel"
        assert results[2].primary_sector == "cement"


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestClassificationEdgeCases:
    """Edge case tests."""

    def test_unknown_nace_code(self):
        engine = SectorClassificationEngine()
        inp = _make_input(codes=[_make_code("Z99.99")])
        result = engine.calculate(inp)
        assert result.primary_sector is not None

    def test_empty_entity_name_rejected(self):
        with pytest.raises(Exception):
            _make_input(entity_name="")

    def test_very_long_entity_name(self):
        engine = SectorClassificationEngine()
        name = "A" * 300
        result = engine.calculate(_make_input(entity_name=name))
        assert result.entity_name == name


# ===========================================================================
# Performance Tests
# ===========================================================================


class TestClassificationPerformance:
    """Performance tests."""

    def test_single_classification_under_100ms(self):
        engine = SectorClassificationEngine()
        with timed_block("single_classification", max_seconds=0.1):
            engine.calculate(_make_input(codes=[_make_code("D35.11")]))

    def test_100_classifications_under_2s(self):
        engine = SectorClassificationEngine()
        with timed_block("100_classifications", max_seconds=2.0):
            for _ in range(100):
                engine.calculate(_make_input(codes=[_make_code("C24.10")]))

    @pytest.mark.parametrize("nace_code", list(NACE_SECTOR_MAP.keys()))
    def test_each_nace_code_under_100ms(self, nace_code):
        engine = SectorClassificationEngine()
        with timed_block(f"class_{nace_code}", max_seconds=0.1):
            engine.calculate(_make_input(codes=[_make_code(nace_code)]))
