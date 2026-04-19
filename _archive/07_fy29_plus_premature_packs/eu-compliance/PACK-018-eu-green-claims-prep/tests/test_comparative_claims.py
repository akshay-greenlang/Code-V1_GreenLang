# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Comparative Claims Engine Tests
=====================================================================

Unit tests for ComparativeClaimsEngine covering enums (ComparisonType,
FutureClaimStatus, MethodologyStatus, ClaimCompliance), models
(ComparativeClaim, FutureClaimInput, MethodologyAssessment), constants,
and engine methods (validate_comparative_claim, assess_future_claim,
calculate_improvement, validate_methodology).

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
    """Load the Comparative Claims engine module."""
    return _load_engine("comparative_claims")


@pytest.fixture
def engine(mod):
    """Create a fresh ComparativeClaimsEngine instance."""
    return mod.ComparativeClaimsEngine()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestComparativeClaimsEnums:
    """Tests for Comparative Claims engine enums."""

    def test_comparison_type_count(self, mod):
        """ComparisonType has exactly 5 values."""
        assert len(mod.ComparisonType) == 5

    def test_comparison_type_values(self, mod):
        """ComparisonType contains expected comparison types."""
        values = {m.value for m in mod.ComparisonType}
        expected = {
            "year_over_year", "product_vs_product",
            "vs_industry_average", "vs_regulatory_baseline",
            "improvement_over_time",
        }
        assert values == expected

    def test_comparison_type_year_over_year(self, mod):
        """ComparisonType includes YEAR_OVER_YEAR."""
        assert mod.ComparisonType.YEAR_OVER_YEAR.value == "year_over_year"

    def test_comparison_type_product_vs_product(self, mod):
        """ComparisonType includes PRODUCT_VS_PRODUCT."""
        assert mod.ComparisonType.PRODUCT_VS_PRODUCT.value == "product_vs_product"

    def test_future_claim_status_count(self, mod):
        """FutureClaimStatus has exactly 4 values."""
        assert len(mod.FutureClaimStatus) == 4

    def test_future_claim_status_values(self, mod):
        """FutureClaimStatus contains expected statuses."""
        values = {m.value for m in mod.FutureClaimStatus}
        expected = {"validated", "conditional", "unsubstantiated", "prohibited"}
        assert values == expected

    def test_methodology_status_count(self, mod):
        """MethodologyStatus has exactly 4 values."""
        assert len(mod.MethodologyStatus) == 4

    def test_methodology_status_values(self, mod):
        """MethodologyStatus contains expected values."""
        values = {m.value for m in mod.MethodologyStatus}
        expected = {
            "equivalent", "partially_equivalent",
            "non_equivalent", "insufficient_data",
        }
        assert values == expected

    def test_claim_compliance_count(self, mod):
        """ClaimCompliance has exactly 4 values."""
        assert len(mod.ClaimCompliance) == 4

    def test_claim_compliance_values(self, mod):
        """ClaimCompliance contains expected values."""
        values = {m.value for m in mod.ClaimCompliance}
        expected = {
            "compliant", "partially_compliant",
            "non_compliant", "requires_amendment",
        }
        assert values == expected


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestComparativeClaimsConstants:
    """Tests for Comparative Claims engine constants."""

    def test_comparison_requirements_exist(self, mod):
        """COMPARISON_REQUIREMENTS dict exists."""
        assert hasattr(mod, "COMPARISON_REQUIREMENTS")
        assert len(mod.COMPARISON_REQUIREMENTS) >= 4

    def test_comparison_requirements_has_year_over_year(self, mod):
        """COMPARISON_REQUIREMENTS has year_over_year entry."""
        assert "year_over_year" in mod.COMPARISON_REQUIREMENTS

    def test_comparison_requirements_has_product_vs_product(self, mod):
        """COMPARISON_REQUIREMENTS has product_vs_product entry."""
        assert "product_vs_product" in mod.COMPARISON_REQUIREMENTS


# ===========================================================================
# Model Tests
# ===========================================================================


class TestComparativeClaimModel:
    """Tests for ComparativeClaim Pydantic model."""

    def test_create_valid_claim(self, mod):
        """Create a valid ComparativeClaim with required fields."""
        claim = mod.ComparativeClaim(
            claim_text="30% less CO2 than 2020 baseline",
            comparison_type=mod.ComparisonType.YEAR_OVER_YEAR,
            baseline_value=Decimal("100"),
            current_value=Decimal("70"),
            unit="kg CO2 eq",
            methodology="ISO 14067",
        )
        assert claim.comparison_type == mod.ComparisonType.YEAR_OVER_YEAR
        assert claim.baseline_value == Decimal("100")
        assert claim.current_value == Decimal("70")

    def test_claim_has_auto_id(self, mod):
        """ComparativeClaim auto-generates claim_id."""
        claim = mod.ComparativeClaim(
            claim_text="Improvement over time test claim",
            comparison_type=mod.ComparisonType.IMPROVEMENT_OVER_TIME,
            baseline_value=Decimal("50"),
            current_value=Decimal("40"),
            unit="kg CO2 eq",
            methodology="GHG Protocol",
        )
        assert claim.claim_id is not None

    def test_claim_stores_methodology(self, mod):
        """ComparativeClaim stores methodology reference."""
        claim = mod.ComparativeClaim(
            claim_text="Year over year comparison claim",
            comparison_type=mod.ComparisonType.YEAR_OVER_YEAR,
            baseline_value=Decimal("100"),
            current_value=Decimal("70"),
            unit="kg CO2 eq",
            methodology="ISO 14067",
        )
        assert claim.methodology == "ISO 14067"

    def test_claim_stores_years(self, mod):
        """ComparativeClaim stores baseline and current year."""
        claim = mod.ComparativeClaim(
            claim_text="Year over year emission reduction",
            comparison_type=mod.ComparisonType.YEAR_OVER_YEAR,
            baseline_value=Decimal("100"),
            current_value=Decimal("70"),
            unit="kg CO2 eq",
            methodology="ISO 14067",
            baseline_year=2020,
            current_year=2025,
        )
        assert claim.baseline_year == 2020
        assert claim.current_year == 2025


class TestFutureClaimInputModel:
    """Tests for FutureClaimInput Pydantic model."""

    def test_create_future_claim(self, mod):
        """Create a valid FutureClaimInput."""
        future = mod.FutureClaimInput(
            claim_text="We will be carbon neutral by 2030",
        )
        assert "carbon neutral" in future.claim_text

    def test_future_claim_has_fields(self, mod):
        """FutureClaimInput has expected fields."""
        future = mod.FutureClaimInput(
            claim_text="Test future claim",
        )
        assert hasattr(future, "claim_text")


# ===========================================================================
# Engine Method Tests
# ===========================================================================


class TestComparativeClaimsEngine:
    """Tests for ComparativeClaimsEngine methods."""

    def test_engine_instantiation(self, mod):
        """Engine can be instantiated."""
        engine = mod.ComparativeClaimsEngine()
        assert engine is not None

    def test_engine_has_validate_comparative_claim(self, engine):
        """Engine has validate_comparative_claim method."""
        assert hasattr(engine, "validate_comparative_claim")
        assert callable(engine.validate_comparative_claim)

    def test_engine_has_assess_future_claim(self, engine):
        """Engine has assess_future_claim method."""
        assert hasattr(engine, "assess_future_claim")
        assert callable(engine.assess_future_claim)

    def test_engine_has_calculate_improvement(self, engine):
        """Engine has calculate_improvement method."""
        assert hasattr(engine, "calculate_improvement")
        assert callable(engine.calculate_improvement)

    def test_engine_has_validate_methodology(self, engine):
        """Engine has validate_methodology method."""
        assert hasattr(engine, "validate_methodology")
        assert callable(engine.validate_methodology)

    def test_engine_has_docstring(self, mod):
        """ComparativeClaimsEngine class has a docstring."""
        assert mod.ComparativeClaimsEngine.__doc__ is not None
        assert "comparative" in mod.ComparativeClaimsEngine.__doc__.lower()


# ===========================================================================
# Provenance and Source Checks
# ===========================================================================


class TestComparativeClaimsProvenance:
    """Tests for source file characteristics and provenance."""

    def test_engine_source_has_sha256(self):
        """Engine source uses SHA-256 for provenance."""
        source = (ENGINES_DIR / "comparative_claims_engine.py").read_text(
            encoding="utf-8"
        )
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        """Engine source uses Decimal arithmetic."""
        source = (ENGINES_DIR / "comparative_claims_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        """Engine source uses Pydantic BaseModel."""
        source = (ENGINES_DIR / "comparative_claims_engine.py").read_text(
            encoding="utf-8"
        )
        assert "BaseModel" in source

    def test_engine_source_has_logging(self):
        """Engine source uses logging."""
        source = (ENGINES_DIR / "comparative_claims_engine.py").read_text(
            encoding="utf-8"
        )
        assert "logging" in source

    def test_engine_source_references_article3_4(self):
        """Engine source references Article 3(4) of Green Claims Directive."""
        source = (ENGINES_DIR / "comparative_claims_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Article 3(4)" in source or "Article 5" in source

    def test_engine_file_exists(self):
        """Engine source file exists on disk."""
        assert (ENGINES_DIR / "comparative_claims_engine.py").exists()


# ===========================================================================
# Sample Data Tests
# ===========================================================================


class TestComparativeClaimsSampleData:
    """Tests using sample_comparative_claims fixture from conftest."""

    def test_sample_comparative_claims_count(self, sample_comparative_claims):
        """sample_comparative_claims has at least 2 entries."""
        assert len(sample_comparative_claims) >= 2

    def test_sample_claims_have_claim_id(self, sample_comparative_claims):
        """All sample comparative claims have claim_id."""
        for claim in sample_comparative_claims:
            assert "claim_id" in claim

    def test_sample_claims_have_comparison_type(self, sample_comparative_claims):
        """All sample comparative claims have comparison_type."""
        for claim in sample_comparative_claims:
            assert "comparison_type" in claim

    def test_sample_claims_have_baseline(self, sample_comparative_claims):
        """All sample comparative claims have baseline_value."""
        for claim in sample_comparative_claims:
            assert "baseline_value" in claim

    def test_sample_claims_have_current(self, sample_comparative_claims):
        """All sample comparative claims have current_value."""
        for claim in sample_comparative_claims:
            assert "current_value" in claim

    def test_sample_claims_values_are_decimal(self, sample_comparative_claims):
        """Baseline and current values are Decimal."""
        for claim in sample_comparative_claims:
            assert isinstance(claim["baseline_value"], Decimal)
            assert isinstance(claim["current_value"], Decimal)

    def test_sample_improvement_claim(self, sample_comparative_claims):
        """Second sample is an improvement_over_time claim."""
        assert sample_comparative_claims[1]["comparison_type"] == "IMPROVEMENT_OVER_TIME"
