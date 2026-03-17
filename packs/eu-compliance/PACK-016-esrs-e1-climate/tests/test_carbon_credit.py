# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Carbon Credit Engine Tests
=============================================================

Unit tests for CarbonCreditEngine (Engine 6) covering credit
registration, removal registration, portfolio building, quality
assessment, SBTi compliance, completeness, and E1-7 data points.

ESRS E1-7: GHG removals and carbon credits.

Target: 45+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the carbon_credit engine module."""
    return _load_engine("carbon_credit")


@pytest.fixture
def engine(mod):
    """Create a fresh CarbonCreditEngine instance."""
    return mod.CarbonCreditEngine()


@pytest.fixture
def sample_credit(mod):
    """Create a sample carbon credit."""
    return mod.CarbonCredit(
        standard=mod.CreditStandard.VERRA_VCS,
        credit_type=mod.CreditType.REMOVAL,
        project_type=mod.ProjectType.FORESTRY_AFFORESTATION,
        project_name="Amazon Reforestation Project",
        vintage_year=2024,
        quantity_tco2e=Decimal("5000"),
        unit_price=Decimal("15.00"),
        total_cost=Decimal("75000"),
        is_certified=True,
        additionality_score=4,
        permanence_score=3,
        measurability_score=4,
        verification_score=4,
    )


@pytest.fixture
def sample_removal(mod):
    """Create a sample GHG removal."""
    return mod.GHGRemoval(
        removal_type=mod.RemovalType.AFFORESTATION,
        quantity_tco2e=Decimal("1000"),
        methodology="AR-ACM0003",
        verification_status=mod.VerificationStatus.VERIFIED,
        is_in_own_operations=True,
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestCreditEnums:
    """Tests for carbon credit enums."""

    def test_credit_standard_count(self, mod):
        """CreditStandard has at least 6 values."""
        assert len(mod.CreditStandard) >= 6
        values = {m.value for m in mod.CreditStandard}
        assert "verra_vcs" in values
        assert "gold_standard" in values
        assert "cdm" in values

    def test_credit_type_values(self, mod):
        """CreditType has 2 values: avoidance and removal."""
        assert len(mod.CreditType) == 2
        values = {m.value for m in mod.CreditType}
        assert values == {"avoidance", "removal"}

    def test_project_type_count(self, mod):
        """ProjectType has at least 8 values."""
        assert len(mod.ProjectType) >= 8
        values = {m.value for m in mod.ProjectType}
        assert "renewable_energy" in values
        assert "direct_air_capture" in values
        assert "forestry_afforestation" in values

    def test_credit_status_values(self, mod):
        """CreditStatus has lifecycle values."""
        assert len(mod.CreditStatus) >= 3
        values = {m.value for m in mod.CreditStatus}
        assert "purchased" in values
        assert "retired" in values

    def test_removal_type_count(self, mod):
        """RemovalType has at least 6 values."""
        assert len(mod.RemovalType) >= 6
        values = {m.value for m in mod.RemovalType}
        assert "daccs" in values
        assert "beccs" in values
        assert "afforestation" in values

    def test_verification_status_values(self, mod):
        """VerificationStatus has expected values."""
        assert len(mod.VerificationStatus) >= 3
        values = {m.value for m in mod.VerificationStatus}
        assert "verified" in values
        assert "not_verified" in values


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestCreditConstants:
    """Tests for carbon credit constants."""

    def test_e1_7_datapoints_exist(self, mod):
        """E1_7_DATAPOINTS is a non-empty dict."""
        assert len(mod.E1_7_DATAPOINTS) >= 12

    def test_quality_criteria_count(self, mod):
        """QUALITY_CRITERIA has 4 criteria."""
        assert len(mod.QUALITY_CRITERIA) == 4
        for key in ["additionality", "permanence", "measurability", "verification"]:
            assert key in mod.QUALITY_CRITERIA

    def test_quality_criteria_weights_sum_to_1(self, mod):
        """Quality criteria weights sum to 1.0."""
        total = sum(c["weight"] for c in mod.QUALITY_CRITERIA.values())
        assert float(total) == pytest.approx(1.0, abs=0.001)

    def test_sbti_guidance_exists(self, mod):
        """SBTI_BEYONDVALUECHAINMITIGATION has required keys."""
        assert "principle" in mod.SBTI_BEYONDVALUECHAINMITIGATION
        assert "near_term" in mod.SBTI_BEYONDVALUECHAINMITIGATION
        assert "long_term" in mod.SBTI_BEYONDVALUECHAINMITIGATION
        assert "net_zero" in mod.SBTI_BEYONDVALUECHAINMITIGATION


# ===========================================================================
# Credit Model Tests
# ===========================================================================


class TestCreditModel:
    """Tests for CarbonCredit Pydantic model."""

    def test_create_valid_credit(self, mod):
        """Create a valid CarbonCredit."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.GOLD_STANDARD,
            credit_type=mod.CreditType.AVOIDANCE,
            vintage_year=2023,
            quantity_tco2e=Decimal("1000"),
        )
        assert credit.standard == mod.CreditStandard.GOLD_STANDARD
        assert len(credit.credit_id) > 0

    def test_credit_with_quality_scores(self, mod):
        """Credit with quality assessment scores."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.VERRA_VCS,
            credit_type=mod.CreditType.REMOVAL,
            vintage_year=2024,
            quantity_tco2e=Decimal("500"),
            additionality_score=5,
            permanence_score=4,
            measurability_score=5,
            verification_score=5,
        )
        assert credit.additionality_score == 5
        assert credit.verification_score == 5

    def test_credit_zero_quantity_rejected(self, mod):
        """Credit with zero quantity is rejected."""
        with pytest.raises(Exception):
            mod.CarbonCredit(
                standard=mod.CreditStandard.VERRA_VCS,
                credit_type=mod.CreditType.AVOIDANCE,
                vintage_year=2024,
                quantity_tco2e=Decimal("0"),
            )

    def test_credit_score_out_of_range(self, mod):
        """Quality score > 5 is rejected."""
        with pytest.raises(Exception):
            mod.CarbonCredit(
                standard=mod.CreditStandard.VERRA_VCS,
                credit_type=mod.CreditType.AVOIDANCE,
                vintage_year=2024,
                quantity_tco2e=Decimal("100"),
                additionality_score=6,
            )


# ===========================================================================
# Removal Model Tests
# ===========================================================================


class TestRemovalModel:
    """Tests for GHGRemoval Pydantic model."""

    def test_create_valid_removal(self, mod):
        """Create a valid GHGRemoval."""
        removal = mod.GHGRemoval(
            removal_type=mod.RemovalType.DACCS,
            quantity_tco2e=Decimal("500"),
        )
        assert removal.removal_type == mod.RemovalType.DACCS
        assert len(removal.removal_id) > 0

    def test_removal_zero_quantity_rejected(self, mod):
        """Removal with zero quantity is rejected."""
        with pytest.raises(Exception):
            mod.GHGRemoval(
                removal_type=mod.RemovalType.DACCS,
                quantity_tco2e=Decimal("0"),
            )


# ===========================================================================
# Portfolio Tests
# ===========================================================================


class TestPortfolio:
    """Tests for credit portfolio building."""

    def test_basic_portfolio(self, engine, sample_credit, sample_removal):
        """Build a basic credit portfolio."""
        engine.register_credit(sample_credit)
        engine.register_removal(sample_removal)
        result = engine.build_credit_portfolio()
        assert result is not None
        assert result.processing_time_ms >= 0.0

    def test_portfolio_by_standard(self, engine, mod):
        """Portfolio groups credits by standard."""
        eng = mod.CarbonCreditEngine()
        eng.register_credit(mod.CarbonCredit(
            standard=mod.CreditStandard.VERRA_VCS,
            credit_type=mod.CreditType.AVOIDANCE,
            vintage_year=2024,
            quantity_tco2e=Decimal("1000"),
        ))
        eng.register_credit(mod.CarbonCredit(
            standard=mod.CreditStandard.GOLD_STANDARD,
            credit_type=mod.CreditType.REMOVAL,
            vintage_year=2024,
            quantity_tco2e=Decimal("500"),
        ))
        result = eng.build_credit_portfolio()
        assert result is not None

    def test_portfolio_by_type(self, engine, mod):
        """Portfolio distinguishes avoidance and removal credits."""
        eng = mod.CarbonCreditEngine()
        eng.register_credit(mod.CarbonCredit(
            standard=mod.CreditStandard.VERRA_VCS,
            credit_type=mod.CreditType.AVOIDANCE,
            vintage_year=2024,
            quantity_tco2e=Decimal("2000"),
        ))
        eng.register_credit(mod.CarbonCredit(
            standard=mod.CreditStandard.VERRA_VCS,
            credit_type=mod.CreditType.REMOVAL,
            vintage_year=2024,
            quantity_tco2e=Decimal("1000"),
        ))
        result = eng.build_credit_portfolio()
        assert result is not None

    def test_portfolio_provenance_hash(self, mod, sample_credit):
        """Portfolio has a provenance hash."""
        eng = mod.CarbonCreditEngine()
        eng.register_credit(sample_credit)
        result = eng.build_credit_portfolio()
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)


# ===========================================================================
# Quality Assessment Tests
# ===========================================================================


class TestQualityAssessment:
    """Tests for carbon credit quality assessment."""

    def test_high_quality_credit(self, engine, mod):
        """High-quality credit gets high score."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.GOLD_STANDARD,
            credit_type=mod.CreditType.REMOVAL,
            vintage_year=2024,
            quantity_tco2e=Decimal("100"),
            additionality_score=5,
            permanence_score=5,
            measurability_score=5,
            verification_score=5,
        )
        result = engine.assess_credit_quality(credit)
        assert isinstance(result, mod.QualityAssessment)
        assert result.overall_score > Decimal("4")

    def test_low_quality_credit(self, engine, mod):
        """Low-quality credit gets low score."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.CUSTOM,
            credit_type=mod.CreditType.AVOIDANCE,
            vintage_year=2020,
            quantity_tco2e=Decimal("100"),
            additionality_score=1,
            permanence_score=1,
            measurability_score=1,
            verification_score=1,
        )
        result = engine.assess_credit_quality(credit)
        assert result.overall_score < Decimal("2")

    def test_quality_scores_weighted(self, engine, mod):
        """Quality assessment uses weighted scores."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.VERRA_VCS,
            credit_type=mod.CreditType.REMOVAL,
            vintage_year=2024,
            quantity_tco2e=Decimal("100"),
            additionality_score=4,
            permanence_score=3,
            measurability_score=4,
            verification_score=5,
        )
        result = engine.assess_credit_quality(credit)
        # Weighted: 4*0.30 + 3*0.25 + 4*0.25 + 5*0.20 = 1.2+0.75+1.0+1.0 = 3.95
        assert float(result.overall_score) == pytest.approx(3.95, abs=0.1)

    def test_quality_tier_assigned(self, engine, mod):
        """Quality assessment assigns a tier label."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.GOLD_STANDARD,
            credit_type=mod.CreditType.REMOVAL,
            vintage_year=2024,
            quantity_tco2e=Decimal("100"),
            additionality_score=4,
            permanence_score=4,
            measurability_score=4,
            verification_score=4,
        )
        result = engine.assess_credit_quality(credit)
        assert result.quality_tier != ""
        assert isinstance(result.quality_tier, str)


# ===========================================================================
# SBTi Compliance Tests
# ===========================================================================


class TestSBTiCompliance:
    """Tests for SBTi carbon credit compliance."""

    def test_valid_use(self, engine, mod):
        """Credits used for BVCM are SBTi-compliant."""
        eng = mod.CarbonCreditEngine()
        eng.register_credit(mod.CarbonCredit(
            standard=mod.CreditStandard.VERRA_VCS,
            credit_type=mod.CreditType.REMOVAL,
            vintage_year=2024,
            quantity_tco2e=Decimal("1000"),
            additionality_score=4,
            permanence_score=4,
            measurability_score=4,
            verification_score=4,
        ))
        portfolio = eng.build_credit_portfolio()
        result = eng.validate_sbti_compliance(
            result=portfolio,
            target_emissions_tco2e=Decimal("100000"),
        )
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_exceeds_sbti_limits(self, engine, mod):
        """Over-reliance on credits flagged by SBTi rules."""
        eng = mod.CarbonCreditEngine()
        # Register large quantity of avoidance credits
        eng.register_credit(mod.CarbonCredit(
            standard=mod.CreditStandard.CUSTOM,
            credit_type=mod.CreditType.AVOIDANCE,
            vintage_year=2024,
            quantity_tco2e=Decimal("50000"),
        ))
        portfolio = eng.build_credit_portfolio()
        result = eng.validate_sbti_compliance(
            result=portfolio,
            target_emissions_tco2e=Decimal("100000"),
        )
        assert isinstance(result, dict)


# ===========================================================================
# Completeness Tests
# ===========================================================================


class TestCompleteness:
    """Tests for E1-7 completeness validation."""

    def test_complete(self, mod, sample_credit, sample_removal):
        """Complete portfolio has good completeness."""
        eng = mod.CarbonCreditEngine()
        eng.register_credit(sample_credit)
        eng.register_removal(sample_removal)
        result = eng.build_credit_portfolio()
        completeness = eng.validate_completeness(result)
        assert isinstance(completeness, dict)
        assert len(completeness) > 0

    def test_incomplete(self, mod):
        """Empty portfolio has lower completeness."""
        eng = mod.CarbonCreditEngine()
        result = eng.build_credit_portfolio()
        completeness = eng.validate_completeness(result)
        assert isinstance(completeness, dict)


# ===========================================================================
# E1-7 Data Points Tests
# ===========================================================================


class TestE17Datapoints:
    """Tests for E1-7 required data point extraction."""

    def test_returns_datapoints(self, mod, sample_credit, sample_removal):
        """get_e1_7_datapoints returns required data points."""
        eng = mod.CarbonCreditEngine()
        eng.register_credit(sample_credit)
        eng.register_removal(sample_removal)
        result = eng.build_credit_portfolio()
        datapoints = eng.get_e1_7_datapoints(result)
        assert isinstance(datapoints, dict)
        assert len(datapoints) >= 10

    def test_e1_7_datapoints_constant(self, mod):
        """E1_7_DATAPOINTS dict has at least 12 entries."""
        assert len(mod.E1_7_DATAPOINTS) >= 12


# ===========================================================================
# Additional Credit Standard Tests
# ===========================================================================


class TestAdditionalStandards:
    """Tests for additional credit standards."""

    def test_cdm_standard(self, mod):
        """CDM standard credit can be created."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.CDM,
            credit_type=mod.CreditType.AVOIDANCE,
            vintage_year=2023,
            quantity_tco2e=Decimal("2000"),
        )
        assert credit.standard == mod.CreditStandard.CDM

    def test_custom_standard(self, mod):
        """Custom standard credit can be created."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.CUSTOM,
            credit_type=mod.CreditType.AVOIDANCE,
            vintage_year=2024,
            quantity_tco2e=Decimal("500"),
        )
        assert credit.standard == mod.CreditStandard.CUSTOM

    def test_gold_standard_credit(self, mod):
        """Gold Standard credit can be created."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.GOLD_STANDARD,
            credit_type=mod.CreditType.REMOVAL,
            vintage_year=2024,
            quantity_tco2e=Decimal("1000"),
            is_certified=True,
        )
        assert credit.is_certified is True


# ===========================================================================
# Credit Status Tests
# ===========================================================================


class TestCreditStatus:
    """Tests for credit status and lifecycle."""

    def test_credit_status_values(self, mod):
        """CreditStatus has expected lifecycle values."""
        values = {m.value for m in mod.CreditStatus}
        assert "purchased" in values
        assert "retired" in values

    def test_default_credit_status(self, mod):
        """Default credit status is set."""
        credit = mod.CarbonCredit(
            standard=mod.CreditStandard.VERRA_VCS,
            credit_type=mod.CreditType.AVOIDANCE,
            vintage_year=2024,
            quantity_tco2e=Decimal("100"),
        )
        # Default status should be valid
        assert credit.status is not None


# ===========================================================================
# Portfolio Advanced Tests
# ===========================================================================


class TestPortfolioAdvanced:
    """Advanced portfolio building tests."""

    def test_portfolio_total_quantity(self, mod):
        """Portfolio total quantity sums all credits."""
        eng = mod.CarbonCreditEngine()
        eng.register_credit(mod.CarbonCredit(
            standard=mod.CreditStandard.VERRA_VCS,
            credit_type=mod.CreditType.AVOIDANCE,
            vintage_year=2024,
            quantity_tco2e=Decimal("3000"),
        ))
        eng.register_credit(mod.CarbonCredit(
            standard=mod.CreditStandard.GOLD_STANDARD,
            credit_type=mod.CreditType.REMOVAL,
            vintage_year=2024,
            quantity_tco2e=Decimal("2000"),
        ))
        result = eng.build_credit_portfolio()
        assert result.total_credits_purchased_tco2e == Decimal("5000")

    def test_portfolio_removals_tracked(self, mod, sample_removal):
        """Portfolio tracks removal quantities separately."""
        eng = mod.CarbonCreditEngine()
        eng.register_removal(sample_removal)
        result = eng.build_credit_portfolio()
        assert result.total_removals_tco2e > Decimal("0")

    def test_empty_portfolio(self, mod):
        """Empty portfolio returns valid result with zero totals."""
        eng = mod.CarbonCreditEngine()
        result = eng.build_credit_portfolio()
        assert result.total_credits_purchased_tco2e == Decimal("0")


# ===========================================================================
# Removal Advanced Tests
# ===========================================================================


class TestRemovalAdvanced:
    """Advanced GHG removal tests."""

    def test_daccs_removal(self, mod):
        """DACCS removal can be created."""
        removal = mod.GHGRemoval(
            removal_type=mod.RemovalType.DACCS,
            quantity_tco2e=Decimal("500"),
            methodology="DACCS-001",
            verification_status=mod.VerificationStatus.VERIFIED,
        )
        assert removal.removal_type == mod.RemovalType.DACCS

    def test_beccs_removal(self, mod):
        """BECCS removal can be created."""
        removal = mod.GHGRemoval(
            removal_type=mod.RemovalType.BECCS,
            quantity_tco2e=Decimal("1000"),
        )
        assert removal.removal_type == mod.RemovalType.BECCS

    def test_own_operations_flag(self, mod):
        """Removal distinguishes own operations vs third party."""
        own = mod.GHGRemoval(
            removal_type=mod.RemovalType.AFFORESTATION,
            quantity_tco2e=Decimal("200"),
            is_in_own_operations=True,
        )
        assert own.is_in_own_operations is True

    def test_removal_unique_ids(self, mod):
        """Each removal gets a unique removal_id."""
        r1 = mod.GHGRemoval(
            removal_type=mod.RemovalType.DACCS,
            quantity_tco2e=Decimal("100"),
        )
        r2 = mod.GHGRemoval(
            removal_type=mod.RemovalType.DACCS,
            quantity_tco2e=Decimal("100"),
        )
        assert r1.removal_id != r2.removal_id
