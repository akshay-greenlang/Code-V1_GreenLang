# -*- coding: utf-8 -*-
"""
Unit tests for PACK-022 engines/__init__.py lazy-loading module.

Tests module-level metadata, lazy engine imports, dynamic __all__,
get_loaded_engines(), get_engine_count(), get_loaded_engine_count(),
and individual engine symbol accessibility.
"""

import sys
from pathlib import Path

import pytest

# Ensure pack root is on the Python path
PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

import engines


# ---------------------------------------------------------------------------
# Module-Level Metadata
# ---------------------------------------------------------------------------


class TestModuleMetadata:

    def test_version(self):
        assert engines.__version__ == "1.0.0"

    def test_pack_id(self):
        assert engines.__pack__ == "PACK-022"

    def test_pack_name(self):
        assert engines.__pack_name__ == "Net Zero Acceleration Pack"

    def test_engines_count(self):
        assert engines.__engines_count__ == 10

    def test_all_is_list(self):
        assert isinstance(engines.__all__, list)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


class TestUtilityFunctions:

    def test_get_engine_count(self):
        assert engines.get_engine_count() == 10

    def test_get_loaded_engine_count_is_int(self):
        count = engines.get_loaded_engine_count()
        assert isinstance(count, int)
        assert count >= 0
        assert count <= 10

    def test_get_loaded_engines_is_list(self):
        loaded = engines.get_loaded_engines()
        assert isinstance(loaded, list)

    def test_loaded_engines_are_strings(self):
        loaded = engines.get_loaded_engines()
        for name in loaded:
            assert isinstance(name, str)

    def test_loaded_count_matches_list_length(self):
        loaded = engines.get_loaded_engines()
        count = engines.get_loaded_engine_count()
        assert len(loaded) == count


# ---------------------------------------------------------------------------
# Engine 6-10 Lazy Loading (these are the engines built for PACK-022 part 2)
# ---------------------------------------------------------------------------


class TestEngine6Loading:
    """Verify TemperatureScoringEngine and its symbols are importable."""

    def test_engine_class_exists(self):
        from engines.temperature_scoring_engine import TemperatureScoringEngine
        assert TemperatureScoringEngine is not None

    def test_config_class_exists(self):
        from engines.temperature_scoring_engine import TemperatureScoringConfig
        assert TemperatureScoringConfig is not None

    def test_enums_exist(self):
        from engines.temperature_scoring_engine import (
            TargetScope, TargetTimeframe, ScoreType, TemperatureBand,
            AggregationMethod, TargetValidityStatus,
        )
        assert TargetScope.S1S2 is not None
        assert ScoreType.WATS is not None

    def test_models_exist(self):
        from engines.temperature_scoring_engine import (
            EmissionsTarget, PortfolioEntity, TemperatureResult,
            EntityTemperatureScore, PortfolioTemperatureScore,
            ContributionEntry, WhatIfScenario, WhatIfResult,
        )
        assert EmissionsTarget is not None
        assert TemperatureResult is not None

    def test_in_loaded_engines(self):
        loaded = engines.get_loaded_engines()
        assert "TemperatureScoringEngine" in loaded


class TestEngine7Loading:
    """Verify VarianceDecompositionEngine and its symbols are importable."""

    def test_engine_class_exists(self):
        from engines.variance_decomposition_engine import VarianceDecompositionEngine
        assert VarianceDecompositionEngine is not None

    def test_enums_exist(self):
        from engines.variance_decomposition_engine import (
            DecompositionMethod, DecompositionEffect,
            ScopeFilter, ForecastHorizon, AlertSeverity,
        )
        assert DecompositionMethod.LMDI_I is not None
        assert AlertSeverity.RED is not None

    def test_models_exist(self):
        from engines.variance_decomposition_engine import (
            SegmentData, YearDecomposition, DriverAttribution,
            ForecastPoint, EarlyWarningAlert, CumulativeEffect, VarianceResult,
        )
        assert SegmentData is not None
        assert VarianceResult is not None

    def test_in_loaded_engines(self):
        loaded = engines.get_loaded_engines()
        assert "VarianceDecompositionEngine" in loaded


class TestEngine8Loading:
    """Verify MultiEntityEngine and its symbols are importable."""

    def test_engine_class_exists(self):
        from engines.multi_entity_engine import MultiEntityEngine
        assert MultiEntityEngine is not None

    def test_enums_exist(self):
        from engines.multi_entity_engine import (
            ConsolidationMethod, EntityType, EliminationType,
            TargetAllocationType, StructuralChangeType, ReportingStatus,
        )
        assert ConsolidationMethod.OPERATIONAL_CONTROL is not None

    def test_models_exist(self):
        from engines.multi_entity_engine import (
            EntityEmissions, IntercompanyElimination, StructuralChange,
            GroupEmissions, EntityTargetAllocation, BaseYearRecalculation,
            ConsolidationResult,
        )
        assert EntityEmissions is not None
        assert ConsolidationResult is not None

    def test_in_loaded_engines(self):
        loaded = engines.get_loaded_engines()
        assert "MultiEntityEngine" in loaded


class TestEngine9Loading:
    """Verify VCMIValidationEngine and its symbols are importable."""

    def test_engine_class_exists(self):
        from engines.vcmi_validation_engine import VCMIValidationEngine
        assert VCMIValidationEngine is not None

    def test_enums_exist(self):
        from engines.vcmi_validation_engine import (
            VCMITier, CriterionStatus, EvidenceStrength,
            CreditQualityLevel, GreenwashingRiskLevel,
        )
        assert VCMITier.SILVER is not None
        assert GreenwashingRiskLevel.CRITICAL is not None

    def test_models_exist(self):
        from engines.vcmi_validation_engine import (
            EmissionsData, CarbonCreditPortfolio, VCMIResult,
            FoundationalCriterionResult, TierEligibility,
            ICVCMAssessment, ISOComparison, GapToNextTier, GreenwashingFlag,
        )
        assert EmissionsData is not None
        assert VCMIResult is not None

    def test_in_loaded_engines(self):
        loaded = engines.get_loaded_engines()
        assert "VCMIValidationEngine" in loaded


class TestEngine10Loading:
    """Verify AssuranceWorkpaperEngine and its symbols are importable."""

    def test_engine_class_exists(self):
        from engines.assurance_workpaper_engine import AssuranceWorkpaperEngine
        assert AssuranceWorkpaperEngine is not None

    def test_enums_exist(self):
        from engines.assurance_workpaper_engine import (
            AssuranceLevel, WorkpaperSection, MaterialityBasis,
            DataSourceType, CalculationMethod, ExceptionSeverity,
            CrossCheckStatus,
        )
        assert AssuranceLevel.LIMITED is not None
        assert WorkpaperSection is not None

    def test_models_exist(self):
        from engines.assurance_workpaper_engine import (
            AssuranceResult, EngagementSummary, MethodologyEntry,
            DataLineageEntry, CalculationStep, CalculationTrace,
            ControlEvidence, CompletenessEntry, CrossCheckResult,
            ExceptionEntry, ChangeEntry,
        )
        assert AssuranceResult is not None
        assert MethodologyEntry is not None

    def test_in_loaded_engines(self):
        loaded = engines.get_loaded_engines()
        assert "AssuranceWorkpaperEngine" in loaded


# ---------------------------------------------------------------------------
# __all__ Composition
# ---------------------------------------------------------------------------


class TestAllComposition:

    def test_all_contains_engine_6_symbols(self):
        expected = [
            "TemperatureScoringEngine", "TemperatureScoringConfig",
            "EmissionsTarget", "PortfolioEntity", "TemperatureResult",
        ]
        for sym in expected:
            assert sym in engines.__all__, f"{sym} not in __all__"

    def test_all_contains_engine_7_symbols(self):
        expected = [
            "VarianceDecompositionEngine", "VarianceDecompositionConfig",
            "SegmentData", "VarianceResult",
        ]
        for sym in expected:
            assert sym in engines.__all__, f"{sym} not in __all__"

    def test_all_contains_engine_8_symbols(self):
        expected = [
            "MultiEntityEngine", "MultiEntityConfig",
            "EntityEmissions", "ConsolidationResult",
        ]
        for sym in expected:
            assert sym in engines.__all__, f"{sym} not in __all__"

    def test_all_contains_engine_9_symbols(self):
        expected = [
            "VCMIValidationEngine", "VCMIValidationConfig",
            "EmissionsData", "CarbonCreditPortfolio", "VCMIResult",
        ]
        for sym in expected:
            assert sym in engines.__all__, f"{sym} not in __all__"

    def test_all_contains_engine_10_symbols(self):
        expected = [
            "AssuranceWorkpaperEngine", "AssuranceWorkpaperConfig",
            "AssuranceResult", "MethodologyEntry",
        ]
        for sym in expected:
            assert sym in engines.__all__, f"{sym} not in __all__"

    def test_all_has_at_least_engine_6_10_symbols(self):
        # Engines 6-10 have 16+14+15+16+20 = 81 symbols total
        # At minimum the engines 6-10 should contribute these
        assert len(engines.__all__) >= 81
