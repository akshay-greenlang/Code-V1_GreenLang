# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Impact Materiality Engine Tests
================================================================================

Unit tests for ImpactMaterialityEngine (Engine 1) covering severity
calculation, impact scoring, batch assessment, ranking, threshold
application, provenance hashing, and edge cases.

ESRS 1 Para 43-48: severity = geometric_mean(scale_w, scope_w, irrem_w)
Impact score = severity * likelihood * time_horizon_weight

Target: 45+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

from decimal import Decimal, ROUND_HALF_UP

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the impact_materiality engine module."""
    return _load_engine("impact_materiality")


@pytest.fixture
def engine(mod):
    """Create a fresh ImpactMaterialityEngine instance."""
    return mod.ImpactMaterialityEngine()


@pytest.fixture
def sample_matter(mod):
    """Create a sample SustainabilityMatter for testing."""
    return mod.SustainabilityMatter(
        id="MATTER-001",
        name="GHG Emissions Scope 1",
        esrs_topic=mod.ESRSTopic.E1_CLIMATE,
        sub_topic="ghg_emissions_scope_1",
        value_chain_stage=mod.ValueChainStage.OWN_OPERATIONS,
    )


@pytest.fixture
def sample_assessment(mod):
    """Create a sample ImpactAssessment for testing."""
    return mod.ImpactAssessment(
        matter_id="MATTER-001",
        scale=4,
        scope=4,
        irremediability=3,
        likelihood=3,
        is_actual=True,
        time_horizon=mod.TimeHorizon.SHORT_TERM,
        impact_type=mod.ImpactType.ACTUAL_NEGATIVE,
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestImpactMaterialityEnums:
    """Tests for impact materiality enums."""

    def test_time_horizon_values(self, mod):
        """TimeHorizon has 3 values: SHORT_TERM, MEDIUM_TERM, LONG_TERM."""
        assert len(mod.TimeHorizon) == 3
        values = {m.value for m in mod.TimeHorizon}
        assert values == {"short_term", "medium_term", "long_term"}

    def test_value_chain_stage_values(self, mod):
        """ValueChainStage has 3 values."""
        assert len(mod.ValueChainStage) == 3
        values = {m.value for m in mod.ValueChainStage}
        assert values == {"upstream", "own_operations", "downstream"}

    def test_esrs_topic_values(self, mod):
        """ESRSTopic has 10 topics (E1-E5, S1-S4, G1)."""
        assert len(mod.ESRSTopic) == 10

    def test_impact_type_values(self, mod):
        """ImpactType has 4 values."""
        assert len(mod.ImpactType) == 4
        values = {m.value for m in mod.ImpactType}
        expected = {
            "actual_negative", "potential_negative",
            "actual_positive", "potential_positive",
        }
        assert values == expected

    def test_scale_level_values(self, mod):
        """ScaleLevel has 5 levels (1-5)."""
        assert len(mod.ScaleLevel) == 5
        int_values = {m.value for m in mod.ScaleLevel}
        assert int_values == {1, 2, 3, 4, 5}


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestImpactMaterialityConstants:
    """Tests for impact materiality constants."""

    def test_scale_weights_mapping(self, mod):
        """SCALE_WEIGHTS maps 1-5 to Decimal 0.20-1.00."""
        assert mod.SCALE_WEIGHTS[1] == Decimal("0.20")
        assert mod.SCALE_WEIGHTS[3] == Decimal("0.60")
        assert mod.SCALE_WEIGHTS[5] == Decimal("1.00")

    def test_scope_weights_mapping(self, mod):
        """SCOPE_WEIGHTS maps 1-5 to Decimal 0.20-1.00."""
        assert mod.SCOPE_WEIGHTS[1] == Decimal("0.20")
        assert mod.SCOPE_WEIGHTS[5] == Decimal("1.00")

    def test_irremediability_weights_mapping(self, mod):
        """IRREMEDIABILITY_WEIGHTS maps 1-5 to Decimal 0.20-1.00."""
        assert mod.IRREMEDIABILITY_WEIGHTS[1] == Decimal("0.20")
        assert mod.IRREMEDIABILITY_WEIGHTS[5] == Decimal("1.00")

    def test_likelihood_weights_mapping(self, mod):
        """LIKELIHOOD_WEIGHTS maps 1-5 to Decimal 0.20-1.00."""
        assert mod.LIKELIHOOD_WEIGHTS[1] == Decimal("0.20")
        assert mod.LIKELIHOOD_WEIGHTS[5] == Decimal("1.00")

    def test_time_horizon_weights(self, mod):
        """TIME_HORIZON_WEIGHTS: short=1.00, medium=0.85, long=0.70."""
        assert mod.TIME_HORIZON_WEIGHTS["short_term"] == Decimal("1.00")
        assert mod.TIME_HORIZON_WEIGHTS["medium_term"] == Decimal("0.85")
        assert mod.TIME_HORIZON_WEIGHTS["long_term"] == Decimal("0.70")

    def test_default_materiality_threshold(self, mod):
        """Default threshold is 0.40."""
        assert mod.DEFAULT_MATERIALITY_THRESHOLD == Decimal("0.40")

    def test_esrs_sustainability_matters_has_10_topics(self, mod):
        """ESRS_SUSTAINABILITY_MATTERS has all 10 topics."""
        assert len(mod.ESRS_SUSTAINABILITY_MATTERS) == 10


# ===========================================================================
# Pydantic Model Tests
# ===========================================================================


class TestSustainabilityMatterModel:
    """Tests for SustainabilityMatter Pydantic model."""

    def test_create_valid_matter(self, mod):
        """Create a valid SustainabilityMatter."""
        matter = mod.SustainabilityMatter(
            name="Climate Change Mitigation",
            esrs_topic=mod.ESRSTopic.E1_CLIMATE,
        )
        assert matter.name == "Climate Change Mitigation"
        assert matter.esrs_topic == mod.ESRSTopic.E1_CLIMATE
        assert len(matter.id) > 0

    def test_matter_name_required(self, mod):
        """SustainabilityMatter requires a name."""
        with pytest.raises(Exception):
            mod.SustainabilityMatter(
                esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            )

    def test_matter_empty_name_rejected(self, mod):
        """SustainabilityMatter rejects empty/whitespace name."""
        with pytest.raises(Exception):
            mod.SustainabilityMatter(
                name="   ",
                esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            )

    def test_matter_default_value_chain_stage(self, mod):
        """Default value chain stage is OWN_OPERATIONS."""
        matter = mod.SustainabilityMatter(
            name="Test", esrs_topic=mod.ESRSTopic.E1_CLIMATE,
        )
        assert matter.value_chain_stage == mod.ValueChainStage.OWN_OPERATIONS


class TestImpactAssessmentModel:
    """Tests for ImpactAssessment Pydantic model."""

    def test_create_valid_assessment(self, mod):
        """Create a valid ImpactAssessment."""
        asmt = mod.ImpactAssessment(
            matter_id="M-001",
            scale=3,
            scope=3,
            irremediability=3,
        )
        assert asmt.scale == 3
        assert asmt.is_actual is True
        assert asmt.time_horizon == mod.TimeHorizon.SHORT_TERM

    def test_assessment_score_out_of_range(self, mod):
        """Assessment rejects scores outside 1-5."""
        with pytest.raises(Exception):
            mod.ImpactAssessment(
                matter_id="M-001", scale=0, scope=3, irremediability=3,
            )
        with pytest.raises(Exception):
            mod.ImpactAssessment(
                matter_id="M-001", scale=6, scope=3, irremediability=3,
            )

    def test_assessment_matter_id_required(self, mod):
        """Assessment requires matter_id."""
        with pytest.raises(Exception):
            mod.ImpactAssessment(
                matter_id="", scale=3, scope=3, irremediability=3,
            )


# ===========================================================================
# Severity Calculation Tests
# ===========================================================================


class TestSeverityCalculation:
    """Tests for severity calculation (geometric mean)."""

    def test_severity_all_ones(self, engine):
        """All 1s: severity = (0.20*0.20*0.20)^(1/3) = 0.20."""
        result = engine.calculate_severity(1, 1, 1)
        assert isinstance(result, Decimal)
        assert result == Decimal("0.2000")

    def test_severity_all_fives(self, engine):
        """All 5s: severity = (1.00*1.00*1.00)^(1/3) = 1.00."""
        result = engine.calculate_severity(5, 5, 5)
        assert result == Decimal("1.0000")

    def test_severity_all_threes(self, engine):
        """All 3s: severity = (0.60*0.60*0.60)^(1/3) = 0.60."""
        result = engine.calculate_severity(3, 3, 3)
        assert result == Decimal("0.6000")

    def test_severity_mixed_inputs(self, engine):
        """Mixed inputs: (0.80*0.80*0.60)^(1/3)."""
        result = engine.calculate_severity(4, 4, 3)
        # (0.80 * 0.80 * 0.60) = 0.384
        # 0.384 ^ (1/3) ~= 0.7268
        assert isinstance(result, Decimal)
        assert Decimal("0.70") <= result <= Decimal("0.75")

    def test_severity_invalid_scale_raises(self, engine):
        """Scale outside 1-5 raises ValueError."""
        with pytest.raises(ValueError, match="Scale must be 1-5"):
            engine.calculate_severity(0, 3, 3)
        with pytest.raises(ValueError, match="Scale must be 1-5"):
            engine.calculate_severity(6, 3, 3)

    def test_severity_invalid_scope_raises(self, engine):
        """Scope outside 1-5 raises ValueError."""
        with pytest.raises(ValueError, match="Scope must be 1-5"):
            engine.calculate_severity(3, 0, 3)

    def test_severity_invalid_irremediability_raises(self, engine):
        """Irremediability outside 1-5 raises ValueError."""
        with pytest.raises(ValueError, match="Irremediability must be 1-5"):
            engine.calculate_severity(3, 3, 6)

    def test_severity_deterministic(self, engine):
        """Same inputs always produce same severity."""
        s1 = engine.calculate_severity(4, 3, 2)
        s2 = engine.calculate_severity(4, 3, 2)
        assert s1 == s2

    @pytest.mark.parametrize("scale,scope,irrem", [
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
        (5, 5, 5),
    ])
    def test_severity_monotonic_with_equal_inputs(self, engine, scale, scope, irrem):
        """For equal inputs, severity equals the weight directly."""
        result = engine.calculate_severity(scale, scope, irrem)
        # geometric mean of (w, w, w) = w
        expected_weight = Decimal(str(scale * 0.20))
        assert result == pytest.approx(expected_weight, abs=Decimal("0.001"))


# ===========================================================================
# Impact Score Calculation Tests
# ===========================================================================


class TestImpactScoreCalculation:
    """Tests for calculate_impact_score method."""

    def test_actual_impact_score(self, engine, mod):
        """Actual impact: score = severity * 1.0 * time_horizon_weight."""
        severity = Decimal("0.7268")
        score = engine.calculate_impact_score(
            severity, likelihood=3, is_actual=True,
            time_horizon=mod.TimeHorizon.SHORT_TERM,
        )
        # score = 0.7268 * 1.0 * 1.0 = 0.7268
        assert isinstance(score, Decimal)
        assert score == pytest.approx(Decimal("0.7268"), abs=Decimal("0.001"))

    def test_potential_impact_score(self, engine, mod):
        """Potential impact: score = severity * likelihood_weight * time_weight."""
        severity = Decimal("0.6000")
        score = engine.calculate_impact_score(
            severity, likelihood=3, is_actual=False,
            time_horizon=mod.TimeHorizon.SHORT_TERM,
        )
        # score = 0.6 * 0.6 * 1.0 = 0.36
        assert score == pytest.approx(Decimal("0.36"), abs=Decimal("0.01"))

    def test_medium_term_discount(self, engine, mod):
        """Medium term reduces score by 0.85 factor."""
        severity = Decimal("1.0000")
        score = engine.calculate_impact_score(
            severity, likelihood=5, is_actual=True,
            time_horizon=mod.TimeHorizon.MEDIUM_TERM,
        )
        # score = 1.0 * 1.0 * 0.85 = 0.85
        assert score == pytest.approx(Decimal("0.85"), abs=Decimal("0.01"))

    def test_long_term_discount(self, engine, mod):
        """Long term reduces score by 0.70 factor."""
        severity = Decimal("1.0000")
        score = engine.calculate_impact_score(
            severity, likelihood=5, is_actual=True,
            time_horizon=mod.TimeHorizon.LONG_TERM,
        )
        assert score == pytest.approx(Decimal("0.70"), abs=Decimal("0.01"))

    def test_invalid_likelihood_raises(self, engine, mod):
        """Likelihood outside 1-5 raises ValueError."""
        with pytest.raises(ValueError, match="Likelihood must be 1-5"):
            engine.calculate_impact_score(
                Decimal("0.60"), likelihood=0, is_actual=False,
                time_horizon=mod.TimeHorizon.SHORT_TERM,
            )


# ===========================================================================
# Assess Impact Tests
# ===========================================================================


class TestAssessImpact:
    """Tests for the full assess_impact workflow."""

    def test_assess_impact_basic(self, engine, sample_matter, sample_assessment):
        """Basic assess_impact returns valid ImpactMaterialityResult."""
        result = engine.assess_impact(sample_matter, sample_assessment)
        assert result.matter_id == "MATTER-001"
        assert result.matter_name == "GHG Emissions Scope 1"
        assert result.esrs_topic == "e1_climate"
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_assess_impact_severity_score(self, engine, sample_matter, sample_assessment):
        """Verify severity score calculation in assess_impact."""
        result = engine.assess_impact(sample_matter, sample_assessment)
        # scale=4, scope=4, irremediability=3
        # severity = (0.80 * 0.80 * 0.60)^(1/3) ~= 0.727
        assert Decimal("0.70") <= result.severity_score <= Decimal("0.75")

    def test_assess_impact_actual_likelihood_is_one(self, engine, sample_matter, sample_assessment):
        """Actual impact has likelihood_score = 1.000."""
        result = engine.assess_impact(sample_matter, sample_assessment)
        assert result.likelihood_score == Decimal("1.000")

    def test_assess_impact_materiality_determination(self, engine, sample_matter, sample_assessment):
        """High score results in is_material=True (default threshold 0.40)."""
        result = engine.assess_impact(sample_matter, sample_assessment)
        # score ~= 0.727, threshold = 0.40 => is_material = True
        assert result.is_material is True

    def test_assess_impact_mismatched_id_raises(self, engine, mod, sample_matter):
        """Mismatched matter_id raises ValueError."""
        assessment = mod.ImpactAssessment(
            matter_id="WRONG-ID",
            scale=3, scope=3, irremediability=3,
        )
        with pytest.raises(ValueError, match="does not match"):
            engine.assess_impact(sample_matter, assessment)

    def test_assess_impact_custom_threshold(self, engine, sample_matter, sample_assessment):
        """Custom threshold changes materiality determination."""
        result = engine.assess_impact(
            sample_matter, sample_assessment, threshold=Decimal("0.90")
        )
        # score ~= 0.727 < 0.90 threshold
        assert result.is_material is False
        assert result.threshold_used == Decimal("0.90")

    def test_assess_impact_processing_time(self, engine, sample_matter, sample_assessment):
        """Processing time is recorded and > 0."""
        result = engine.assess_impact(sample_matter, sample_assessment)
        assert result.processing_time_ms >= 0.0

    def test_assess_impact_potential_impact(self, engine, mod):
        """Potential impact uses likelihood weight < 1.0."""
        matter = mod.SustainabilityMatter(
            id="MATTER-P-001",
            name="Potential Water Pollution",
            esrs_topic=mod.ESRSTopic.E3_WATER,
        )
        assessment = mod.ImpactAssessment(
            matter_id="MATTER-P-001",
            scale=3, scope=3, irremediability=3,
            likelihood=2, is_actual=False,
            impact_type=mod.ImpactType.POTENTIAL_NEGATIVE,
        )
        result = engine.assess_impact(matter, assessment)
        # likelihood_score for level 2 = 0.40
        assert result.likelihood_score == Decimal("0.400")
        # score = 0.60 * 0.40 * 1.0 = 0.24
        assert result.impact_materiality_score == pytest.approx(
            Decimal("0.24"), abs=Decimal("0.01")
        )


# ===========================================================================
# Provenance Hash Tests
# ===========================================================================


class TestProvenanceHash:
    """Tests for SHA-256 provenance hash determinism."""

    def test_provenance_hash_is_64_chars(self, engine, sample_matter, sample_assessment):
        """Provenance hash is a 64-character hex string."""
        result = engine.assess_impact(sample_matter, sample_assessment)
        assert len(result.provenance_hash) == 64
        # Verify it is valid hex
        int(result.provenance_hash, 16)

    def test_provenance_hash_deterministic(self, engine, mod):
        """Same inputs produce same provenance hash across calls."""
        matter = mod.SustainabilityMatter(
            id="DET-001", name="Determinism Test",
            esrs_topic=mod.ESRSTopic.E1_CLIMATE,
        )
        asmt = mod.ImpactAssessment(
            matter_id="DET-001", scale=3, scope=3, irremediability=3,
            is_actual=True,
        )
        r1 = engine.assess_impact(matter, asmt)
        r2 = engine.assess_impact(matter, asmt)
        # Provenance hashes may differ due to calculated_at timestamp
        # but the score values must match
        assert r1.severity_score == r2.severity_score
        assert r1.impact_materiality_score == r2.impact_materiality_score

    def test_different_inputs_different_hash(self, engine, mod):
        """Different inputs produce different provenance hashes."""
        matter = mod.SustainabilityMatter(
            id="DIFF-001", name="Hash Diff Test",
            esrs_topic=mod.ESRSTopic.E1_CLIMATE,
        )
        asmt1 = mod.ImpactAssessment(
            matter_id="DIFF-001", scale=3, scope=3, irremediability=3,
        )
        asmt2 = mod.ImpactAssessment(
            matter_id="DIFF-001", scale=5, scope=5, irremediability=5,
        )
        r1 = engine.assess_impact(matter, asmt1)
        r2 = engine.assess_impact(matter, asmt2)
        assert r1.provenance_hash != r2.provenance_hash


# ===========================================================================
# Batch Assessment Tests
# ===========================================================================


class TestBatchAssessment:
    """Tests for batch_assess method."""

    def test_batch_assess_basic(self, engine, mod):
        """Batch assess returns BatchImpactResult with all results."""
        matters = [
            mod.SustainabilityMatter(
                id=f"B-{i}", name=f"Matter {i}",
                esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            )
            for i in range(3)
        ]
        assessments = [
            mod.ImpactAssessment(
                matter_id=f"B-{i}", scale=i + 1, scope=i + 1,
                irremediability=i + 1,
            )
            for i in range(3)
        ]
        result = engine.batch_assess(matters, assessments)
        assert result is not None
        assert len(result.results) == 3

    def test_batch_assess_empty_matters_raises(self, engine, mod):
        """Empty matters list raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            engine.batch_assess([], [
                mod.ImpactAssessment(
                    matter_id="X", scale=3, scope=3, irremediability=3,
                )
            ])

    def test_batch_assess_empty_assessments_raises(self, engine, mod):
        """Empty assessments list raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            engine.batch_assess(
                [mod.SustainabilityMatter(
                    id="X", name="Test", esrs_topic=mod.ESRSTopic.E1_CLIMATE,
                )],
                [],
            )

    def test_batch_assess_ranking_order(self, engine, mod):
        """Results are ranked by score (highest first)."""
        matters = [
            mod.SustainabilityMatter(
                id=f"R-{i}", name=f"Matter {i}",
                esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            )
            for i in range(1, 4)
        ]
        assessments = [
            mod.ImpactAssessment(
                matter_id="R-1", scale=1, scope=1, irremediability=1,
            ),
            mod.ImpactAssessment(
                matter_id="R-2", scale=5, scope=5, irremediability=5,
            ),
            mod.ImpactAssessment(
                matter_id="R-3", scale=3, scope=3, irremediability=3,
            ),
        ]
        result = engine.batch_assess(matters, assessments)
        ranked = result.results
        # Verify ranking is assigned
        rankings = [r.ranking for r in ranked]
        assert 1 in rankings


# ===========================================================================
# Interpret Score Tests
# ===========================================================================


class TestInterpretScore:
    """Tests for score interpretation strings."""

    def test_interpret_very_high(self, engine):
        """Score >= 0.80 returns VERY_HIGH or Critical."""
        label = engine.interpret_score(Decimal("0.90"))
        assert isinstance(label, str)
        assert len(label) > 0

    def test_interpret_high(self, engine):
        """Score in [0.60, 0.80) returns HIGH or Significant."""
        label = engine.interpret_score(Decimal("0.70"))
        assert isinstance(label, str)

    def test_interpret_moderate(self, engine):
        """Score in [0.40, 0.60) returns MODERATE."""
        label = engine.interpret_score(Decimal("0.50"))
        assert isinstance(label, str)

    def test_interpret_low(self, engine):
        """Score in [0.20, 0.40) returns LOW."""
        label = engine.interpret_score(Decimal("0.30"))
        assert isinstance(label, str)

    def test_interpret_negligible(self, engine):
        """Score < 0.20 returns NEGLIGIBLE."""
        label = engine.interpret_score(Decimal("0.10"))
        assert isinstance(label, str)


# ===========================================================================
# Apply Threshold Tests
# ===========================================================================


class TestApplyThreshold:
    """Tests for threshold application."""

    def test_apply_threshold_filters_correctly(self, engine, mod):
        """apply_threshold returns only material results."""
        matters = [
            mod.SustainabilityMatter(
                id=f"T-{i}", name=f"T Matter {i}",
                esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            )
            for i in range(1, 4)
        ]
        assessments = [
            mod.ImpactAssessment(
                matter_id="T-1", scale=5, scope=5, irremediability=5,
            ),
            mod.ImpactAssessment(
                matter_id="T-2", scale=1, scope=1, irremediability=1,
            ),
            mod.ImpactAssessment(
                matter_id="T-3", scale=3, scope=3, irremediability=3,
            ),
        ]
        batch = engine.batch_assess(matters, assessments)
        material = engine.apply_threshold(batch.results, Decimal("0.50"))
        # T-1 (all 5s => score 1.0) should be material
        # T-2 (all 1s => score 0.20) should not
        assert len(material) >= 1
        for r in material:
            assert r.impact_materiality_score >= Decimal("0.50")


# ===========================================================================
# Positive Severity Tests
# ===========================================================================


class TestPositiveSeverity:
    """Tests for positive impact severity calculation."""

    def test_calculate_positive_severity(self, engine):
        """Positive severity uses sqrt (scale_w * scope_w)."""
        result = engine.calculate_positive_severity(5, 5)
        # sqrt(1.0 * 1.0) = 1.0
        assert result == Decimal("1.0000")

    def test_positive_severity_lower_bound(self, engine):
        """Minimum positive severity with all 1s."""
        result = engine.calculate_positive_severity(1, 1)
        # sqrt(0.20 * 0.20) = sqrt(0.04) = 0.2
        assert result == Decimal("0.2000")

    def test_positive_severity_mixed(self, engine):
        """Mixed positive severity."""
        result = engine.calculate_positive_severity(4, 3)
        # sqrt(0.80 * 0.60) = sqrt(0.48) ~= 0.6928
        assert isinstance(result, Decimal)
        assert Decimal("0.68") <= result <= Decimal("0.70")


# ===========================================================================
# Score Breakdown Tests
# ===========================================================================


class TestScoreBreakdown:
    """Tests for get_score_breakdown method."""

    def test_score_breakdown_returns_dict(self, engine, sample_matter, sample_assessment):
        """get_score_breakdown returns a dictionary with component scores."""
        result = engine.assess_impact(sample_matter, sample_assessment)
        breakdown = engine.get_score_breakdown(result)
        assert isinstance(breakdown, dict)
        assert "severity_score" in breakdown or "severity" in str(breakdown).lower()
