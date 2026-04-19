# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Lifecycle Assessment Engine Tests
=======================================================================

Unit tests for LifecycleAssessmentEngine covering enums (ImpactCategory 16
values, LifecyclePhase, DataQualityRating, SystemBoundaryType), models
(LifecycleImpact, LCAResult, DataQualityResult, SystemBoundaryResult),
constants (PEF_WEIGHTING_FACTORS, PEF_CATEGORY_UNITS, DATA_QUALITY_SCORES),
and engine methods (calculate_lifecycle_impacts, identify_hotspots,
calculate_pef_score, assess_data_quality, validate_system_boundary).

Target: ~50 tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine, ENGINES_DIR


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the Lifecycle Assessment engine module."""
    return _load_engine("lifecycle_assessment")


@pytest.fixture
def engine(mod):
    """Create a fresh LifecycleAssessmentEngine instance."""
    return mod.LifecycleAssessmentEngine()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestLifecycleAssessmentEnums:
    """Tests for Lifecycle Assessment engine enums."""

    def test_impact_category_count(self, mod):
        """ImpactCategory has exactly 16 values (PEF categories)."""
        assert len(mod.ImpactCategory) == 16

    def test_impact_category_climate_change(self, mod):
        """ImpactCategory includes CLIMATE_CHANGE."""
        assert mod.ImpactCategory.CLIMATE_CHANGE.value == "climate_change"

    def test_impact_category_ozone_depletion(self, mod):
        """ImpactCategory includes OZONE_DEPLETION."""
        assert mod.ImpactCategory.OZONE_DEPLETION.value == "ozone_depletion"

    def test_impact_category_acidification(self, mod):
        """ImpactCategory includes ACIDIFICATION."""
        assert mod.ImpactCategory.ACIDIFICATION.value == "acidification"

    def test_impact_category_water_use(self, mod):
        """ImpactCategory includes WATER_USE."""
        assert mod.ImpactCategory.WATER_USE.value == "water_use"

    def test_impact_category_land_use(self, mod):
        """ImpactCategory includes LAND_USE."""
        assert mod.ImpactCategory.LAND_USE.value == "land_use"

    def test_impact_category_particulate_matter(self, mod):
        """ImpactCategory includes PARTICULATE_MATTER."""
        assert mod.ImpactCategory.PARTICULATE_MATTER.value == "particulate_matter"

    def test_impact_category_ionising_radiation(self, mod):
        """ImpactCategory includes IONISING_RADIATION."""
        assert mod.ImpactCategory.IONISING_RADIATION.value == "ionising_radiation"

    def test_impact_category_ecotoxicity(self, mod):
        """ImpactCategory includes ECOTOXICITY."""
        assert mod.ImpactCategory.ECOTOXICITY.value == "ecotoxicity"

    def test_lifecycle_phase_count(self, mod):
        """LifecyclePhase has exactly 6 values."""
        assert len(mod.LifecyclePhase) == 6

    def test_lifecycle_phase_values(self, mod):
        """LifecyclePhase covers full product lifecycle."""
        values = {m.value for m in mod.LifecyclePhase}
        expected = {
            "raw_materials", "manufacturing", "transportation",
            "distribution", "use", "end_of_life",
        }
        assert values == expected

    def test_data_quality_rating_count(self, mod):
        """DataQualityRating has exactly 5 values."""
        assert len(mod.DataQualityRating) == 5

    def test_data_quality_rating_values(self, mod):
        """DataQualityRating spans excellent to poor."""
        values = {m.value for m in mod.DataQualityRating}
        expected = {"excellent", "very_good", "good", "fair", "poor"}
        assert values == expected

    def test_system_boundary_type_count(self, mod):
        """SystemBoundaryType has exactly 4 values."""
        assert len(mod.SystemBoundaryType) == 4

    def test_system_boundary_type_values(self, mod):
        """SystemBoundaryType includes cradle-to-grave etc."""
        values = {m.value for m in mod.SystemBoundaryType}
        expected = {
            "cradle_to_grave", "cradle_to_gate",
            "gate_to_gate", "gate_to_grave",
        }
        assert values == expected


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestLifecycleAssessmentConstants:
    """Tests for Lifecycle Assessment engine constants."""

    def test_pef_weighting_factors_count(self, mod):
        """PEF_WEIGHTING_FACTORS has 16 entries for all impact categories."""
        assert len(mod.PEF_WEIGHTING_FACTORS) == 16

    def test_pef_weighting_factors_sum_near_100(self, mod):
        """PEF_WEIGHTING_FACTORS sum to approximately 100."""
        total = sum(mod.PEF_WEIGHTING_FACTORS.values())
        assert Decimal("99") <= total <= Decimal("101")

    def test_pef_weighting_factors_climate_change(self, mod):
        """Climate change has the highest PEF weighting factor (21.06%)."""
        cc_weight = mod.PEF_WEIGHTING_FACTORS["climate_change"]
        assert cc_weight == Decimal("21.06")

    def test_pef_weighting_factors_are_decimal(self, mod):
        """PEF_WEIGHTING_FACTORS values are all Decimal."""
        for val in mod.PEF_WEIGHTING_FACTORS.values():
            assert isinstance(val, Decimal)

    def test_pef_category_units_count(self, mod):
        """PEF_CATEGORY_UNITS has 16 entries."""
        assert len(mod.PEF_CATEGORY_UNITS) == 16

    def test_pef_category_units_climate_change(self, mod):
        """Climate change unit is kg CO2 eq."""
        assert mod.PEF_CATEGORY_UNITS["climate_change"] == "kg CO2 eq"

    def test_data_quality_scores_count(self, mod):
        """DATA_QUALITY_SCORES has 5 entries."""
        assert len(mod.DATA_QUALITY_SCORES) == 5

    def test_data_quality_scores_excellent(self, mod):
        """Excellent data quality scores 100."""
        assert mod.DATA_QUALITY_SCORES["excellent"] == Decimal("100")

    def test_boundary_required_phases_exist(self, mod):
        """BOUNDARY_REQUIRED_PHASES has 4 entries."""
        assert len(mod.BOUNDARY_REQUIRED_PHASES) == 4

    def test_minimum_dqr_threshold(self, mod):
        """MINIMUM_DQR_THRESHOLD is 70."""
        assert mod.MINIMUM_DQR_THRESHOLD == Decimal("70")


# ===========================================================================
# Model Tests
# ===========================================================================


class TestLifecycleImpactModel:
    """Tests for LifecycleImpact Pydantic model."""

    def test_create_valid_impact(self, mod):
        """Create a valid LifecycleImpact with required fields."""
        impact = mod.LifecycleImpact(
            phase=mod.LifecyclePhase.RAW_MATERIALS,
            category=mod.ImpactCategory.CLIMATE_CHANGE,
            value=Decimal("12.5"),
        )
        assert impact.phase == mod.LifecyclePhase.RAW_MATERIALS
        assert impact.category == mod.ImpactCategory.CLIMATE_CHANGE
        assert impact.value == Decimal("12.5")

    def test_impact_has_auto_id(self, mod):
        """LifecycleImpact auto-generates impact_id."""
        impact = mod.LifecycleImpact(
            phase=mod.LifecyclePhase.MANUFACTURING,
            category=mod.ImpactCategory.WATER_USE,
            value=Decimal("100"),
        )
        assert impact.impact_id is not None

    def test_impact_default_quality_rating(self, mod):
        """LifecycleImpact defaults data_quality_rating to FAIR."""
        impact = mod.LifecycleImpact(
            phase=mod.LifecyclePhase.USE,
            category=mod.ImpactCategory.LAND_USE,
            value=Decimal("25"),
        )
        assert impact.data_quality_rating == mod.DataQualityRating.FAIR

    def test_impact_default_primary_data(self, mod):
        """LifecycleImpact defaults is_primary_data to False."""
        impact = mod.LifecycleImpact(
            phase=mod.LifecyclePhase.END_OF_LIFE,
            category=mod.ImpactCategory.ACIDIFICATION,
            value=Decimal("5"),
        )
        assert impact.is_primary_data is False


class TestLCAResultModel:
    """Tests for LCAResult Pydantic model."""

    def test_create_lca_result(self, mod):
        """Create an LCAResult with required fields."""
        result = mod.LCAResult(product_name="Test Product")
        assert result.product_name == "Test Product"

    def test_lca_result_has_provenance(self, mod):
        """LCAResult has provenance_hash field."""
        result = mod.LCAResult(product_name="Test")
        assert hasattr(result, "provenance_hash")


# ===========================================================================
# Engine Method Tests
# ===========================================================================


class TestLifecycleAssessmentEngine:
    """Tests for LifecycleAssessmentEngine methods."""

    def test_engine_instantiation(self, mod):
        """Engine can be instantiated."""
        engine = mod.LifecycleAssessmentEngine()
        assert engine is not None

    def test_engine_has_calculate_lifecycle_impacts(self, engine):
        """Engine has calculate_lifecycle_impacts method."""
        assert hasattr(engine, "calculate_lifecycle_impacts")
        assert callable(engine.calculate_lifecycle_impacts)

    def test_engine_has_identify_hotspots(self, engine):
        """Engine has identify_hotspots method."""
        assert hasattr(engine, "identify_hotspots")
        assert callable(engine.identify_hotspots)

    def test_engine_has_calculate_pef_score(self, engine):
        """Engine has calculate_pef_score method."""
        assert hasattr(engine, "calculate_pef_score")
        assert callable(engine.calculate_pef_score)

    def test_engine_has_assess_data_quality(self, engine):
        """Engine has assess_data_quality method."""
        assert hasattr(engine, "assess_data_quality")
        assert callable(engine.assess_data_quality)

    def test_engine_has_validate_system_boundary(self, engine):
        """Engine has validate_system_boundary method."""
        assert hasattr(engine, "validate_system_boundary")
        assert callable(engine.validate_system_boundary)

    def test_engine_has_docstring(self, mod):
        """LifecycleAssessmentEngine class has a docstring."""
        assert mod.LifecycleAssessmentEngine.__doc__ is not None


# ===========================================================================
# Provenance and Source Checks
# ===========================================================================


class TestLifecycleAssessmentProvenance:
    """Tests for source file characteristics and provenance."""

    def test_engine_source_has_sha256(self):
        """Engine source uses SHA-256 for provenance."""
        source = (ENGINES_DIR / "lifecycle_assessment_engine.py").read_text(
            encoding="utf-8"
        )
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        """Engine source uses Decimal arithmetic."""
        source = (ENGINES_DIR / "lifecycle_assessment_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        """Engine source uses Pydantic BaseModel."""
        source = (ENGINES_DIR / "lifecycle_assessment_engine.py").read_text(
            encoding="utf-8"
        )
        assert "BaseModel" in source

    def test_engine_source_has_logging(self):
        """Engine source uses logging."""
        source = (ENGINES_DIR / "lifecycle_assessment_engine.py").read_text(
            encoding="utf-8"
        )
        assert "logging" in source

    def test_engine_source_references_pef(self):
        """Engine source references PEF methodology."""
        source = (ENGINES_DIR / "lifecycle_assessment_engine.py").read_text(
            encoding="utf-8"
        )
        assert "PEF" in source

    def test_engine_source_references_iso14040(self):
        """Engine source references ISO 14040."""
        source = (ENGINES_DIR / "lifecycle_assessment_engine.py").read_text(
            encoding="utf-8"
        )
        assert "14040" in source or "14044" in source

    def test_engine_file_exists(self):
        """Engine source file exists on disk."""
        assert (ENGINES_DIR / "lifecycle_assessment_engine.py").exists()
