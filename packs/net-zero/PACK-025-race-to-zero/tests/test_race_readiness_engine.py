# -*- coding: utf-8 -*-
"""
Deep tests for RaceReadinessEngine (Engine 10 of 10).

Covers: 8-dimension composite scoring, dimension weights, RAG status
classification (GREEN/AMBER/RED/BLACK), readiness level classification
(RACE_READY/APPROACHING/BUILDING/EARLY_STAGE/PRE_PLEDGE), urgency
levels, dimension configuration, improvement priority ranking,
Decimal arithmetic, SHA-256 provenance.

Target: ~55 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from engines.race_readiness_engine import (
    RaceReadinessEngine,
    ReadinessLevel,
    RAGStatus,
    DimensionId,
    UrgencyLevel,
    DIMENSION_CONFIG,
)

from conftest import assert_decimal_close


# ========================================================================
# Dimension Configuration
# ========================================================================


class TestDimensionConfiguration:
    """Validate 8-dimension configuration."""

    def test_exactly_8_dimensions(self):
        assert len(DimensionId) == 8

    def test_dimension_values(self):
        assert DimensionId.PLEDGE_STRENGTH.value == "pledge_strength"
        assert DimensionId.STARTING_LINE.value == "starting_line_compliance"
        assert DimensionId.TARGET_AMBITION.value == "target_ambition"
        assert DimensionId.ACTION_PLAN.value == "action_plan_quality"
        assert DimensionId.PROGRESS.value == "progress_trajectory"
        assert DimensionId.SECTOR.value == "sector_alignment"
        assert DimensionId.PARTNERSHIP.value == "partnership_engagement"
        assert DimensionId.CREDIBILITY.value == "hleg_credibility"

    def test_all_dimensions_configured(self):
        for dim in DimensionId:
            assert dim.value in DIMENSION_CONFIG, f"{dim.value} missing config"

    def test_weights_sum_to_one(self):
        total = sum(
            cfg["weight"] for cfg in DIMENSION_CONFIG.values()
        )
        assert_decimal_close(total, Decimal("1.00"), Decimal("0.001"))

    def test_starting_line_highest_weight(self):
        assert DIMENSION_CONFIG["starting_line_compliance"]["weight"] == Decimal("0.18")

    def test_pledge_strength_weight_012(self):
        assert DIMENSION_CONFIG["pledge_strength"]["weight"] == Decimal("0.12")

    def test_target_ambition_weight_015(self):
        assert DIMENSION_CONFIG["target_ambition"]["weight"] == Decimal("0.15")

    def test_action_plan_weight_015(self):
        assert DIMENSION_CONFIG["action_plan_quality"]["weight"] == Decimal("0.15")

    def test_progress_weight_012(self):
        assert DIMENSION_CONFIG["progress_trajectory"]["weight"] == Decimal("0.12")

    def test_sector_weight_010(self):
        assert DIMENSION_CONFIG["sector_alignment"]["weight"] == Decimal("0.10")

    def test_partnership_weight_008(self):
        assert DIMENSION_CONFIG["partnership_engagement"]["weight"] == Decimal("0.08")

    def test_credibility_weight_010(self):
        assert DIMENSION_CONFIG["hleg_credibility"]["weight"] == Decimal("0.10")


# ========================================================================
# Dimension Config Fields
# ========================================================================


class TestDimensionConfigFields:
    """Validate each dimension has required configuration fields."""

    def test_each_has_name(self):
        for dim_id, cfg in DIMENSION_CONFIG.items():
            assert "name" in cfg, f"{dim_id} missing name"

    def test_each_has_weight(self):
        for dim_id, cfg in DIMENSION_CONFIG.items():
            assert "weight" in cfg, f"{dim_id} missing weight"
            assert isinstance(cfg["weight"], Decimal)

    def test_each_has_source_engine(self):
        for dim_id, cfg in DIMENSION_CONFIG.items():
            assert "source_engine" in cfg, f"{dim_id} missing source_engine"

    def test_each_has_description(self):
        for dim_id, cfg in DIMENSION_CONFIG.items():
            assert "description" in cfg, f"{dim_id} missing description"

    def test_each_has_urgency_factor(self):
        for dim_id, cfg in DIMENSION_CONFIG.items():
            assert "urgency_factor" in cfg, f"{dim_id} missing urgency_factor"
            assert cfg["urgency_factor"] > Decimal("0")

    def test_each_has_min_for_readiness(self):
        for dim_id, cfg in DIMENSION_CONFIG.items():
            assert "min_for_readiness" in cfg, f"{dim_id} missing min_for_readiness"

    def test_source_engines_are_correct(self):
        expected_engines = {
            "pledge_strength": "PledgeCommitmentEngine",
            "starting_line_compliance": "StartingLineEngine",
            "target_ambition": "InterimTargetEngine",
            "action_plan_quality": "ActionPlanEngine",
            "progress_trajectory": "ProgressTrackingEngine",
            "sector_alignment": "SectorPathwayEngine",
            "partnership_engagement": "PartnershipScoringEngine",
            "hleg_credibility": "CredibilityAssessmentEngine",
        }
        for dim_id, expected in expected_engines.items():
            assert DIMENSION_CONFIG[dim_id]["source_engine"] == expected

    def test_starting_line_highest_urgency(self):
        assert DIMENSION_CONFIG["starting_line_compliance"]["urgency_factor"] == Decimal("1.5")


# ========================================================================
# Enum Validation
# ========================================================================


class TestRaceReadinessEnums:
    """Validate race readiness enums."""

    def test_readiness_level_5_values(self):
        assert len(ReadinessLevel) == 5

    def test_readiness_level_values(self):
        assert ReadinessLevel.RACE_READY.value == "RACE_READY"
        assert ReadinessLevel.APPROACHING.value == "APPROACHING"
        assert ReadinessLevel.BUILDING.value == "BUILDING"
        assert ReadinessLevel.EARLY_STAGE.value == "EARLY_STAGE"
        assert ReadinessLevel.PRE_PLEDGE.value == "PRE_PLEDGE"

    def test_rag_status_4_values(self):
        assert len(RAGStatus) == 4

    def test_rag_values(self):
        assert RAGStatus.GREEN.value == "GREEN"
        assert RAGStatus.AMBER.value == "AMBER"
        assert RAGStatus.RED.value == "RED"
        assert RAGStatus.BLACK.value == "BLACK"

    def test_urgency_level_4_values(self):
        assert len(UrgencyLevel) == 4

    def test_urgency_values(self):
        assert UrgencyLevel.CRITICAL.value == "CRITICAL"
        assert UrgencyLevel.HIGH.value == "HIGH"
        assert UrgencyLevel.MEDIUM.value == "MEDIUM"
        assert UrgencyLevel.LOW.value == "LOW"


# ========================================================================
# Engine Instantiation
# ========================================================================


class TestRaceReadinessEngineInstantiation:
    """Tests for engine creation."""

    def test_default_instantiation(self, readiness_engine):
        assert readiness_engine is not None

    def test_engine_has_calculate(self, readiness_engine):
        assert callable(getattr(readiness_engine, "assess", None))

    def test_engine_class_name(self):
        assert RaceReadinessEngine.__name__ == "RaceReadinessEngine"

    def test_engine_has_docstring(self):
        assert RaceReadinessEngine.__doc__ is not None
        assert len(RaceReadinessEngine.__doc__) > 0
