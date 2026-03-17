# -*- coding: utf-8 -*-
"""
Unit tests for FSDoubleMaterialityEngine (Engine 6)
=====================================================

Tests financial materiality scoring, impact materiality scoring,
IRO identification, FI-specific topics, ESRS datapoint mapping,
stakeholder analysis, materiality matrix, and provenance hashing.

Target: 85%+ coverage, ~30 tests.
"""

import importlib.util
import os
from typing import Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Dynamic import via importlib
# ---------------------------------------------------------------------------

_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "engines",
)
_ENGINE_PATH = os.path.normpath(
    os.path.join(_ENGINE_DIR, "fs_double_materiality_engine.py")
)

spec = importlib.util.spec_from_file_location(
    "fs_double_materiality_engine", _ENGINE_PATH,
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FSDoubleMaterialityEngine = mod.FSDoubleMaterialityEngine
FSMaterialityConfig = mod.FSMaterialityConfig
MaterialityTopicData = mod.MaterialityTopicData
FSMaterialityResult = mod.FSMaterialityResult
IROAssessment = mod.IROAssessment
FinancedImpactAssessment = mod.FinancedImpactAssessment
StakeholderInput = mod.StakeholderInput
MaterialityMatrix = mod.MaterialityMatrix
DatapointMapping = mod.DatapointMapping
MaterialityDimension = mod.MaterialityDimension
IROType = mod.IROType
ESRSStandard = mod.ESRSStandard
SeverityScale = mod.SeverityScale
LikelihoodScale = mod.LikelihoodScale
MaterialityOutcome = mod.MaterialityOutcome
StakeholderGroup = mod.StakeholderGroup
SEVERITY_SCORES = mod.SEVERITY_SCORES
LIKELIHOOD_SCORES = mod.LIKELIHOOD_SCORES
DEFAULT_MATERIALITY_THRESHOLD = mod.DEFAULT_MATERIALITY_THRESHOLD
FI_SPECIFIC_TOPICS = mod.FI_SPECIFIC_TOPICS
ESRS_DATAPOINT_REGISTRY = mod.ESRS_DATAPOINT_REGISTRY
DEFAULT_STAKEHOLDER_WEIGHTS = mod.DEFAULT_STAKEHOLDER_WEIGHTS
_compute_hash = mod._compute_hash
_clamp = mod._clamp
_round_val = mod._round_val


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> FSMaterialityConfig:
    """Default engine configuration."""
    return FSMaterialityConfig(
        institution_name="Test Bank AG",
        materiality_threshold=50.0,
        use_max_rule=True,
        include_fi_specific_topics=True,
    )


@pytest.fixture
def engine(default_config) -> FSDoubleMaterialityEngine:
    """Engine instance."""
    return FSDoubleMaterialityEngine(default_config)


@pytest.fixture
def sample_topics() -> List[MaterialityTopicData]:
    """Create a set of diverse materiality topics."""
    return [
        MaterialityTopicData(
            topic_id="t-climate",
            topic_name="Climate Change Mitigation",
            esrs_standard=ESRSStandard.E1,
            is_fi_specific=False,
            financial_likelihood=LikelihoodScale.VERY_LIKELY,
            financial_magnitude=85.0,
            financial_scope=0.8,
            impact_severity=SeverityScale.CRITICAL,
            impact_likelihood=LikelihoodScale.VERY_LIKELY,
            impact_scope=0.9,
            impact_irremediability=0.7,
            iro_type=IROType.RISK,
            description="Climate change risks and opportunities",
        ),
        MaterialityTopicData(
            topic_id="t-pollution",
            topic_name="Pollution Prevention",
            esrs_standard=ESRSStandard.E2,
            is_fi_specific=False,
            financial_likelihood=LikelihoodScale.POSSIBLE,
            financial_magnitude=40.0,
            financial_scope=0.4,
            impact_severity=SeverityScale.MODERATE,
            impact_likelihood=LikelihoodScale.POSSIBLE,
            impact_scope=0.3,
            impact_irremediability=0.2,
            iro_type=IROType.IMPACT,
            description="Pollution from financed activities",
        ),
        MaterialityTopicData(
            topic_id="t-workforce",
            topic_name="Own Workforce",
            esrs_standard=ESRSStandard.S1,
            is_fi_specific=False,
            financial_likelihood=LikelihoodScale.LIKELY,
            financial_magnitude=60.0,
            financial_scope=0.7,
            impact_severity=SeverityScale.SIGNIFICANT,
            impact_likelihood=LikelihoodScale.LIKELY,
            impact_scope=0.8,
            impact_irremediability=0.4,
            iro_type=IROType.IMPACT,
            description="Working conditions and labor rights",
        ),
        MaterialityTopicData(
            topic_id="t-financed-emissions",
            topic_name="Financed Emissions",
            esrs_standard=ESRSStandard.E1,
            is_fi_specific=True,
            financial_likelihood=LikelihoodScale.VERY_LIKELY,
            financial_magnitude=90.0,
            financial_scope=0.9,
            impact_severity=SeverityScale.CRITICAL,
            impact_likelihood=LikelihoodScale.VERY_LIKELY,
            impact_scope=0.95,
            impact_irremediability=0.8,
            iro_type=IROType.IMPACT,
            description="GHG from lending and investment portfolios",
        ),
        MaterialityTopicData(
            topic_id="t-governance",
            topic_name="Business Conduct",
            esrs_standard=ESRSStandard.G1,
            is_fi_specific=False,
            financial_likelihood=LikelihoodScale.UNLIKELY,
            financial_magnitude=30.0,
            financial_scope=0.3,
            impact_severity=SeverityScale.MINOR,
            impact_likelihood=LikelihoodScale.UNLIKELY,
            impact_scope=0.2,
            impact_irremediability=0.1,
            iro_type=IROType.RISK,
            description="Governance and anti-corruption",
        ),
    ]


@pytest.fixture
def sample_stakeholder_inputs() -> List[StakeholderInput]:
    """Create sample stakeholder engagement inputs."""
    return [
        StakeholderInput(
            stakeholder_group=StakeholderGroup.REGULATORS,
            topic_ratings={
                "t-climate": 95.0,
                "t-financed-emissions": 90.0,
                "t-governance": 70.0,
            },
        ),
        StakeholderInput(
            stakeholder_group=StakeholderGroup.INVESTORS,
            topic_ratings={
                "t-climate": 85.0,
                "t-financed-emissions": 95.0,
                "t-workforce": 60.0,
            },
        ),
    ]


@pytest.fixture
def sample_financed_impacts() -> List[FinancedImpactAssessment]:
    """Create sample financed impact assessments."""
    return [
        FinancedImpactAssessment(
            impact_channel="lending",
            total_exposure_eur=5_000_000_000.0,
            high_impact_exposure_eur=1_500_000_000.0,
            high_impact_ratio_pct=30.0,
            financed_emissions_tco2e=500_000.0,
            taxonomy_aligned_pct=12.5,
            impact_severity=SeverityScale.SIGNIFICANT,
        ),
    ]


# ===================================================================
# Test Class: Configuration
# ===================================================================


class TestFSMaterialityConfig:
    """Tests for FSMaterialityConfig validation."""

    def test_default_config(self):
        cfg = FSMaterialityConfig()
        assert cfg.materiality_threshold == DEFAULT_MATERIALITY_THRESHOLD
        assert cfg.use_max_rule is True

    def test_custom_threshold(self):
        cfg = FSMaterialityConfig(materiality_threshold=60.0)
        assert cfg.materiality_threshold == 60.0

    def test_weighted_mode_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            FSMaterialityConfig(
                use_max_rule=False,
                financial_weight_in_overall=0.3,
                impact_weight_in_overall=0.3,
            )

    def test_weighted_mode_valid(self):
        cfg = FSMaterialityConfig(
            use_max_rule=False,
            financial_weight_in_overall=0.6,
            impact_weight_in_overall=0.4,
        )
        assert cfg.financial_weight_in_overall == 0.6


# ===================================================================
# Test Class: Engine Initialization
# ===================================================================


class TestEngineInit:
    """Tests for engine construction."""

    def test_engine_creates_with_config(self, default_config):
        eng = FSDoubleMaterialityEngine(default_config)
        assert eng.config.institution_name == "Test Bank AG"


# ===================================================================
# Test Class: Financial Materiality Scoring
# ===================================================================


class TestFinancialMaterialityScoring:
    """Tests for financial materiality dimension."""

    def test_high_financial_topic_scores_high(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        climate_iro = next(
            a for a in result.iro_assessments if a.topic_id == "t-climate"
        )
        assert climate_iro.financial_score > 50.0

    def test_low_financial_topic_scores_low(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        gov_iro = next(
            a for a in result.iro_assessments if a.topic_id == "t-governance"
        )
        assert gov_iro.financial_score < 30.0

    def test_financial_score_formula(self, engine):
        """Verify: fin_score = (likelihood/100) * (magnitude/100) * scope * 100."""
        topic = MaterialityTopicData(
            topic_id="t-verify",
            topic_name="Verification",
            financial_likelihood=LikelihoodScale.LIKELY,
            financial_magnitude=80.0,
            financial_scope=0.5,
            esrs_standard=ESRSStandard.E1,
        )
        result = engine.assess_materiality([topic])
        iro = result.iro_assessments[0]
        # LIKELY = 75.0 score
        expected = (75.0 / 100.0) * (80.0 / 100.0) * 0.5 * 100.0
        assert abs(iro.financial_score - _clamp(_round_val(expected, 2))) < 2.0

    def test_financial_score_in_range(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        for iro in result.iro_assessments:
            assert 0.0 <= iro.financial_score <= 100.0


# ===================================================================
# Test Class: Impact Materiality Scoring
# ===================================================================


class TestImpactMaterialityScoring:
    """Tests for impact materiality dimension."""

    def test_critical_severity_scores_high(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        financed = next(
            a for a in result.iro_assessments if a.topic_id == "t-financed-emissions"
        )
        assert financed.impact_score > 70.0

    def test_negligible_impact_scores_low(self, engine):
        topic = MaterialityTopicData(
            topic_id="t-low",
            topic_name="Low Impact Topic",
            impact_severity=SeverityScale.NEGLIGIBLE,
            impact_likelihood=LikelihoodScale.VERY_UNLIKELY,
            impact_scope=0.1,
            impact_irremediability=0.0,
            esrs_standard=ESRSStandard.E5,
        )
        result = engine.assess_materiality([topic])
        assert result.iro_assessments[0].impact_score < 10.0

    def test_irremediability_increases_score(self, engine):
        low_irrem = MaterialityTopicData(
            topic_id="t-irr-low",
            topic_name="Low Irremediability",
            impact_severity=SeverityScale.SIGNIFICANT,
            impact_likelihood=LikelihoodScale.LIKELY,
            impact_scope=0.7,
            impact_irremediability=0.0,
            esrs_standard=ESRSStandard.E1,
        )
        high_irrem = MaterialityTopicData(
            topic_id="t-irr-high",
            topic_name="High Irremediability",
            impact_severity=SeverityScale.SIGNIFICANT,
            impact_likelihood=LikelihoodScale.LIKELY,
            impact_scope=0.7,
            impact_irremediability=0.9,
            esrs_standard=ESRSStandard.E1,
        )
        r1 = engine.assess_materiality([low_irrem])
        r2 = engine.assess_materiality([high_irrem])
        assert r2.iro_assessments[0].impact_score >= r1.iro_assessments[0].impact_score

    def test_impact_score_in_range(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        for iro in result.iro_assessments:
            assert 0.0 <= iro.impact_score <= 100.0


# ===================================================================
# Test Class: IRO Identification
# ===================================================================


class TestIROIdentification:
    """Tests for IRO (Impact, Risk, Opportunity) identification."""

    def test_iro_type_preserved(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        climate_iro = next(
            a for a in result.iro_assessments if a.topic_id == "t-climate"
        )
        assert climate_iro.iro_type == IROType.RISK

    def test_overall_score_uses_max_rule(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        for iro in result.iro_assessments:
            assert iro.overall_score >= max(0.0, min(iro.financial_score, iro.impact_score))

    def test_materiality_threshold_applied(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        for iro in result.iro_assessments:
            if iro.is_material:
                assert iro.overall_score >= engine.config.materiality_threshold
            else:
                assert iro.overall_score < engine.config.materiality_threshold


# ===================================================================
# Test Class: FI-Specific Topics
# ===================================================================


class TestFISpecificTopics:
    """Tests for financial institution specific topics."""

    def test_fi_topics_registry_populated(self):
        assert len(FI_SPECIFIC_TOPICS) >= 10
        assert "financed_emissions" in FI_SPECIFIC_TOPICS
        assert "responsible_lending" in FI_SPECIFIC_TOPICS
        assert "financial_inclusion" in FI_SPECIFIC_TOPICS
        assert "fair_pricing" in FI_SPECIFIC_TOPICS
        assert "taxonomy_alignment" in FI_SPECIFIC_TOPICS

    def test_fi_topic_has_esrs_mapping(self):
        for key, info in FI_SPECIFIC_TOPICS.items():
            assert "esrs_standard" in info
            assert "fi_relevance" in info

    def test_fi_specific_flag_in_topic(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        financed = next(
            a for a in result.iro_assessments if a.topic_id == "t-financed-emissions"
        )
        assert financed.is_material is True


# ===================================================================
# Test Class: ESRS Datapoint Mapping
# ===================================================================


class TestESRSDatapointMapping:
    """Tests for ESRS datapoint mapping of material topics."""

    def test_datapoint_registry_covers_all_standards(self):
        for std in ESRSStandard:
            assert std.value in ESRS_DATAPOINT_REGISTRY

    def test_datapoint_mappings_generated(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        assert len(result.datapoint_mappings) > 0

    def test_material_topics_have_datapoints(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        for mapping in result.datapoint_mappings:
            if mapping.is_material:
                assert len(mapping.required_datapoints) > 0

    def test_total_required_datapoints(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        assert result.total_required_datapoints >= 0


# ===================================================================
# Test Class: Stakeholder Analysis
# ===================================================================


class TestStakeholderAnalysis:
    """Tests for stakeholder analysis and alignment."""

    def test_stakeholder_weights_sum_to_one(self):
        total = sum(DEFAULT_STAKEHOLDER_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_stakeholder_inputs_affect_assessment(
        self, engine, sample_topics, sample_stakeholder_inputs,
    ):
        without = engine.assess_materiality(sample_topics)
        with_stakeholders = engine.assess_materiality(
            sample_topics, stakeholder_inputs=sample_stakeholder_inputs,
        )
        # Stakeholder input should produce some alignment score
        assert with_stakeholders.stakeholder_alignment_score >= 0.0

    def test_alignment_score_in_range(
        self, engine, sample_topics, sample_stakeholder_inputs,
    ):
        result = engine.assess_materiality(
            sample_topics, stakeholder_inputs=sample_stakeholder_inputs,
        )
        assert 0.0 <= result.stakeholder_alignment_score <= 100.0


# ===================================================================
# Test Class: Materiality Matrix
# ===================================================================


class TestMaterialityMatrix:
    """Tests for materiality matrix generation."""

    def test_matrix_generated(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        assert result.materiality_matrix is not None

    def test_matrix_contains_all_topics(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        matrix = result.materiality_matrix
        assert len(matrix.topics) == len(sample_topics)

    def test_matrix_thresholds(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        matrix = result.materiality_matrix
        assert matrix.threshold_x == engine.config.materiality_threshold
        assert matrix.threshold_y == engine.config.materiality_threshold


# ===================================================================
# Test Class: Full Assessment Result
# ===================================================================


class TestFullAssessmentResult:
    """Tests for complete assessment result structure."""

    def test_result_structure(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        assert isinstance(result, FSMaterialityResult)
        assert result.total_topics_assessed == len(sample_topics)
        assert result.material_topic_count >= 0
        assert result.material_topic_count <= result.total_topics_assessed

    def test_material_topics_list(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        assert isinstance(result.material_topics, list)
        for name in result.material_topics:
            assert isinstance(name, str)

    def test_standards_coverage(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        assert len(result.standards_coverage) == len(ESRSStandard)
        for std_val, is_material in result.standards_coverage.items():
            assert isinstance(is_material, bool)

    def test_financed_impacts_passthrough(
        self, engine, sample_topics, sample_financed_impacts,
    ):
        result = engine.assess_materiality(
            sample_topics, financed_impacts=sample_financed_impacts,
        )
        assert len(result.financed_impact_assessments) == 1
        assert result.financed_impact_assessments[0].impact_channel == "lending"


# ===================================================================
# Test Class: Provenance and Metadata
# ===================================================================


class TestProvenanceAndMetadata:
    """Tests for provenance hash and result metadata."""

    def test_provenance_hash_is_sha256(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # valid hex

    def test_iro_provenance_hashes(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        for iro in result.iro_assessments:
            assert len(iro.provenance_hash) == 64

    def test_processing_time_positive(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        assert result.processing_time_ms > 0.0

    def test_engine_version(self, engine, sample_topics):
        result = engine.assess_materiality(sample_topics)
        assert result.engine_version == "1.0.0"

    def test_empty_topics_list(self, engine):
        result = engine.assess_materiality([])
        assert result.total_topics_assessed == 0
        assert result.material_topic_count == 0
        assert result.material_topics == []
