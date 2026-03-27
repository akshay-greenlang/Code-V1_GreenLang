# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (AGENT-MRV-029, Engine 6)

60 tests covering 8 regulatory frameworks plus multi-framework assessment:
- GHG Protocol Scope 3 Standard (5 tests)
- CSRD / ESRS E1 (5 tests)
- SBTi (5 tests)
- SB 253 (California Climate Accountability) (5 tests)
- ISO 14064-1 (5 tests)
- CDP Climate Change (5 tests)
- SEC Climate Disclosure (5 tests)
- EU Taxonomy / ISSB S2 (5 tests)
- Multi-framework assessment, scoring, recommendations (20 tests)

The ComplianceCheckerEngine evaluates Scope 3 reporting against each
framework's specific requirements and produces findings with severity,
recommendations, and a compliance score per framework.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.scope3_category_mapper.compliance_checker import (
        ComplianceCheckerEngine,
    )
    COMPLIANCE_AVAILABLE = True
except (ImportError, AttributeError):
    COMPLIANCE_AVAILABLE = False

try:
    from greenlang.agents.mrv.scope3_category_mapper.models import (
        ALL_SCOPE3_CATEGORIES,
        CategoryCompletenessEntry,
        CategoryRelevance,
        CompanyType,
        ComplianceFramework,
        ComplianceFinding,
        ComplianceSeverity,
        ComplianceStatus,
        CompletenessReport,
        DataQualityTier,
        DetailedComplianceAssessment,
        Scope3Category,
        ScreeningResult,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

try:
    from greenlang.agents.mrv.scope3_category_mapper.completeness_screener import (
        CompletenessScreenerEngine,
        COMPANY_TYPE_RELEVANCE,
    )
    SCREENER_AVAILABLE = True
except ImportError:
    SCREENER_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (COMPLIANCE_AVAILABLE and MODELS_AVAILABLE and SCREENER_AVAILABLE),
    reason="ComplianceCheckerEngine or dependencies not available",
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine():
    """Create a fresh ComplianceCheckerEngine."""
    if not COMPLIANCE_AVAILABLE:
        pytest.skip("ComplianceCheckerEngine not available")
    ComplianceCheckerEngine.reset_instance()
    return ComplianceCheckerEngine.get_instance()


@pytest.fixture
def screener():
    """Create a fresh CompletenessScreenerEngine."""
    if not SCREENER_AVAILABLE:
        pytest.skip("CompletenessScreenerEngine not available")
    CompletenessScreenerEngine.reset_instance()
    return CompletenessScreenerEngine.get_instance()


def _build_full_completeness_report(
    screener_engine,
    company_type=CompanyType.MANUFACTURER,
    categories_reported=None,
    data_by_category=None,
) -> "CompletenessReport":
    """Build a complete CompletenessReport using the actual screener engine."""
    if categories_reported is None:
        categories_reported = list(ALL_SCOPE3_CATEGORIES)
    return screener_engine.screen_completeness(
        company_type=company_type,
        categories_reported=categories_reported,
        data_by_category=data_by_category,
    )


def _build_minimal_completeness_report(
    screener_engine,
    company_type=CompanyType.MANUFACTURER,
) -> "CompletenessReport":
    """Build a minimal CompletenessReport with very few categories."""
    return screener_engine.screen_completeness(
        company_type=company_type,
        categories_reported=[
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ],
    )


def _build_full_emission_pcts() -> Dict[Scope3Category, Decimal]:
    """
    Build emission percentages covering all 15 categories, summing to 100%.

    Used to satisfy SBTi's 67% coverage verification requirement.
    """
    cats = list(ALL_SCOPE3_CATEGORIES)
    n = len(cats)
    # First category gets the bulk, remaining split equally, last gets remainder
    share = Decimal("5.00")
    pcts: Dict[Scope3Category, Decimal] = {}
    for i, cat in enumerate(cats):
        if i == 0:
            pcts[cat] = Decimal("30.00")
        elif i < n - 1:
            pcts[cat] = share
        else:
            # Last gets remainder so total is exactly 100
            pcts[cat] = Decimal("100.00") - sum(pcts.values())
    return pcts


# ==============================================================================
# GHG PROTOCOL SCOPE 3 STANDARD
# ==============================================================================


@_SKIP
class TestGHGProtocol:
    """Test GHG Protocol Scope 3 Standard compliance checks."""

    def test_compliant_with_all_categories(self, engine, screener):
        """Fully compliant report (all 15 categories) passes GHG Protocol."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.GHG_PROTOCOL,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.status == ComplianceStatus.PASS
        assert result.score >= Decimal("90")

    def test_material_categories_required(self, engine, screener):
        """Missing material categories -> FAIL or WARNING."""
        partial = [Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES]
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.GHG_PROTOCOL,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=partial,
            completeness_report=report,
        )
        assert result.status in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_non_compliant_missing_material(self, engine, screener):
        """Missing all material categories -> FAIL."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.GHG_PROTOCOL,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        assert result.status == ComplianceStatus.FAIL

    def test_findings_have_rule_code(self, engine, screener):
        """Findings from GHG Protocol assessment have rule codes."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.GHG_PROTOCOL,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        for finding in result.findings:
            assert finding.rule_code is not None
            assert len(finding.rule_code) > 0

    def test_total_checks_positive(self, engine, screener):
        """Assessment always has >0 total checks."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.GHG_PROTOCOL,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.total_checks > 0


# ==============================================================================
# CSRD / ESRS E1
# ==============================================================================


@_SKIP
class TestCSRD:
    """Test CSRD ESRS E1 compliance checks."""

    def test_compliant_full_report(self, engine, screener):
        """Fully compliant CSRD report passes."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CSRD_ESRS,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.status == ComplianceStatus.PASS

    def test_missing_categories_warning(self, engine, screener):
        """Missing material/relevant categories triggers warning."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CSRD_ESRS,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        assert result.status in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_findings_include_csrd_context(self, engine, screener):
        """CSRD findings reference the framework."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CSRD_ESRS,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        for finding in result.findings:
            assert finding.framework == "csrd_esrs"

    def test_score_below_100_for_partial(self, engine, screener):
        """Partial report yields score < 100."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CSRD_ESRS,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        assert result.score < Decimal("100")

    def test_framework_description_populated(self, engine, screener):
        """CSRD assessment has framework description."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CSRD_ESRS,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.framework_description != ""


# ==============================================================================
# SBTi (Science Based Targets Initiative)
# ==============================================================================


@_SKIP
class TestSBTi:
    """Test SBTi compliance checks."""

    def test_compliant_with_full_coverage(self, engine, screener):
        """Full coverage meets SBTi requirements (with emission_pcts)."""
        report = _build_full_completeness_report(screener)
        # SBTi requires emission_pcts to verify 67% coverage; omitting
        # them triggers a WARNING.  Supply pcts that sum to 100%.
        emission_pcts = _build_full_emission_pcts()
        result = engine.assess_compliance(
            framework=ComplianceFramework.SBTI,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
            emission_pcts=emission_pcts,
        )
        assert result.status == ComplianceStatus.PASS

    def test_missing_categories_fails(self, engine, screener):
        """Missing categories -> non-PASS."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SBTI,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        assert result.status in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_sbti_coverage_check_pass(self, engine):
        """SBTi coverage check passes at >= 67%."""
        emission_pcts = {
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES: Decimal("60.00"),
            Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS: Decimal("15.00"),
        }
        result = engine.check_sbti_coverage(
            categories_reported=[
                Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
                Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS,
            ],
            emission_pcts=emission_pcts,
        )
        assert result is True  # 75% >= 67%

    def test_sbti_coverage_check_fail(self, engine):
        """SBTi coverage check fails below 67%."""
        emission_pcts = {
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES: Decimal("30.00"),
            Scope3Category.CAT_6_BUSINESS_TRAVEL: Decimal("0.50"),
        }
        result = engine.check_sbti_coverage(
            categories_reported=[
                Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
                Scope3Category.CAT_6_BUSINESS_TRAVEL,
            ],
            emission_pcts=emission_pcts,
        )
        assert result is False  # 30.50% < 67%

    def test_findings_have_recommendations(self, engine, screener):
        """SBTi findings include recommendations."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SBTI,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        failing = [f for f in result.findings if f.status == ComplianceStatus.FAIL]
        for f in failing:
            assert f.recommendation is not None


# ==============================================================================
# SB 253 (California Climate Accountability Act)
# ==============================================================================


@_SKIP
class TestSB253:
    """Test SB 253 compliance checks."""

    def test_compliant_full_report(self, engine, screener):
        """Fully compliant SB 253 report passes."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SB_253,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.status == ComplianceStatus.PASS

    def test_missing_material_fails(self, engine, screener):
        """Missing material categories -> FAIL or WARNING."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SB_253,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        assert result.status in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_score_high_for_full_report(self, engine, screener):
        """Full report -> high SB 253 score."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SB_253,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.score >= Decimal("90")

    def test_provenance_hash_present(self, engine, screener):
        """SB 253 assessment includes provenance hash."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SB_253,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert len(result.provenance_hash) == 64

    def test_check_counts_balance(self, engine, screener):
        """passed + failed + warning = total for SB 253."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SB_253,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert (
            result.passed_checks + result.failed_checks + result.warning_checks
            == result.total_checks
        )


# ==============================================================================
# ISO 14064-1
# ==============================================================================


@_SKIP
class TestISO14064:
    """Test ISO 14064-1 compliance checks."""

    def test_compliant_full_report(self, engine, screener):
        """Fully compliant ISO 14064 report passes."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.ISO_14064,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.status == ComplianceStatus.PASS

    def test_missing_categories_issue(self, engine, screener):
        """Missing categories produces findings."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.ISO_14064,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        assert result.status in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_score_positive(self, engine, screener):
        """ISO 14064 assessment has a valid score."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.ISO_14064,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert Decimal("0") <= result.score <= Decimal("100")

    def test_assessed_at_populated(self, engine, screener):
        """Assessment timestamp is populated."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.ISO_14064,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.assessed_at != ""
        assert "T" in result.assessed_at

    def test_processing_time_positive(self, engine, screener):
        """Processing time is >= 0."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.ISO_14064,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.processing_time_ms >= 0


# ==============================================================================
# CDP CLIMATE CHANGE
# ==============================================================================


@_SKIP
class TestCDP:
    """Test CDP Climate Change compliance checks."""

    def test_compliant_full_report(self, engine, screener):
        """Fully compliant CDP report passes."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CDP,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.status == ComplianceStatus.PASS

    def test_missing_categories_issue(self, engine, screener):
        """Missing categories triggers findings."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CDP,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        non_pass = [f for f in result.findings if f.status != ComplianceStatus.PASS]
        assert len(non_pass) > 0

    def test_framework_tag(self, engine, screener):
        """CDP findings are tagged with cdp framework."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CDP,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        for finding in result.findings:
            assert finding.framework == "cdp"

    def test_score_high_for_complete(self, engine, screener):
        """Complete report gets high CDP score."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CDP,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.score >= Decimal("90")

    def test_framework_field_is_cdp(self, engine, screener):
        """Assessment framework is CDP."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.CDP,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.framework == ComplianceFramework.CDP


# ==============================================================================
# SEC CLIMATE DISCLOSURE
# ==============================================================================


@_SKIP
class TestSECClimate:
    """Test SEC Climate Disclosure compliance checks."""

    def test_compliant_full_report(self, engine, screener):
        """Fully compliant SEC report passes."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SEC_CLIMATE,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.status == ComplianceStatus.PASS

    def test_minimal_report_status(self, engine, screener):
        """Minimal report still produces structured assessment."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SEC_CLIMATE,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        assert result.total_checks > 0

    def test_score_range(self, engine, screener):
        """SEC score is in valid range."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SEC_CLIMATE,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert Decimal("0") <= result.score <= Decimal("100")

    def test_framework_is_sec(self, engine, screener):
        """Assessment framework is SEC_CLIMATE."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SEC_CLIMATE,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.framework == ComplianceFramework.SEC_CLIMATE

    def test_provenance_hash(self, engine, screener):
        """SEC assessment has provenance hash."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.SEC_CLIMATE,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert len(result.provenance_hash) == 64


# ==============================================================================
# EU TAXONOMY (ISSB S2 mapped)
# ==============================================================================


@_SKIP
class TestEUTaxonomyISSB:
    """Test EU Taxonomy / ISSB S2 compliance checks."""

    def test_compliant_full_report(self, engine, screener):
        """Fully compliant report passes EU Taxonomy/ISSB."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.EU_TAXONOMY,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.status == ComplianceStatus.PASS

    def test_missing_material_issue(self, engine, screener):
        """Missing material categories triggers findings."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.EU_TAXONOMY,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        assert result.status in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_score_high_for_full(self, engine, screener):
        """Full report yields high score."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.EU_TAXONOMY,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.score >= Decimal("90")

    def test_findings_have_severity(self, engine, screener):
        """Findings have valid severity levels."""
        report = _build_minimal_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.EU_TAXONOMY,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        for f in result.findings:
            assert isinstance(f.severity, ComplianceSeverity)

    def test_framework_description(self, engine, screener):
        """Assessment has framework description."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.EU_TAXONOMY,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert result.framework_description != ""


# ==============================================================================
# MULTI-FRAMEWORK ASSESSMENT TESTS
# ==============================================================================


@_SKIP
class TestMultiFrameworkAssessment:
    """Test multi-framework compliance assessment."""

    def test_assess_all_frameworks(self, engine, screener):
        """Assess all frameworks at once."""
        report = _build_full_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert len(results) >= 7  # At least 7 frameworks

    def test_all_frameworks_pass_full_report(self, engine, screener):
        """Fully compliant report passes all frameworks."""
        report = _build_full_completeness_report(screener)
        emission_pcts = _build_full_emission_pcts()
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
            emission_pcts=emission_pcts,
        )
        for assessment in results:
            assert assessment.status == ComplianceStatus.PASS, (
                f"Framework {assessment.framework} failed with status "
                f"{assessment.status}"
            )

    def test_improvement_recommendations(self, engine, screener):
        """Non-compliant report generates improvement recommendations."""
        report = _build_minimal_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        recs = engine.get_improvement_recommendations(results)
        assert len(recs) > 0

    def test_compliance_score_high_for_full(self, engine, screener):
        """Full compliance -> all scores >= 90."""
        report = _build_full_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        for assessment in results:
            assert assessment.score >= Decimal("90"), (
                f"Framework {assessment.framework} score {assessment.score} < 90"
            )

    def test_compliance_score_low_minimal(self, engine, screener):
        """Minimal report -> lower scores."""
        report = _build_minimal_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        low_count = sum(1 for a in results if a.score < Decimal("90"))
        assert low_count > 0

    def test_assessment_has_provenance(self, engine, screener):
        """Each assessment includes a provenance hash."""
        report = _build_full_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        for assessment in results:
            assert len(assessment.provenance_hash) == 64

    def test_assessment_has_timestamp(self, engine, screener):
        """Each assessment includes ISO timestamp."""
        report = _build_full_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        for assessment in results:
            assert assessment.assessed_at != ""
            assert "T" in assessment.assessed_at

    def test_assessment_processing_time(self, engine, screener):
        """Processing time is populated."""
        report = _build_full_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        for assessment in results:
            assert assessment.processing_time_ms >= 0

    def test_finding_severity_levels(self, engine, screener):
        """Findings use correct severity levels for minimal report."""
        report = _build_minimal_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        severities_found = set()
        for assessment in results:
            for finding in assessment.findings:
                severities_found.add(finding.severity)
        assert len(severities_found) > 0

    def test_assessment_check_counts(self, engine, screener):
        """Assessment tracks passed/failed/warning/total check counts."""
        report = _build_full_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        for assessment in results:
            assert assessment.total_checks > 0
            assert (
                assessment.passed_checks
                + assessment.failed_checks
                + assessment.warning_checks
                == assessment.total_checks
            )

    def test_deterministic_assessment(self, engine, screener):
        """Same input produces same assessment scores."""
        report = _build_full_completeness_report(screener)
        cats = list(ALL_SCOPE3_CATEGORIES)
        r1 = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=cats,
            completeness_report=report,
        )
        r2 = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=cats,
            completeness_report=report,
        )
        for a1, a2 in zip(r1, r2):
            assert a1.score == a2.score

    def test_findings_have_recommendation(self, engine, screener):
        """Failed findings include a recommendation."""
        report = _build_minimal_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES],
            completeness_report=report,
        )
        for assessment in results:
            for finding in assessment.findings:
                if finding.status == ComplianceStatus.FAIL:
                    assert finding.recommendation is not None
                    assert len(finding.recommendation) > 0

    def test_findings_have_rule_code(self, engine, screener):
        """Each finding has a non-empty rule code."""
        report = _build_full_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        for assessment in results:
            for finding in assessment.findings:
                assert finding.rule_code is not None
                assert len(finding.rule_code) > 0

    def test_findings_have_framework_tag(self, engine, screener):
        """Each finding is tagged with its framework."""
        report = _build_full_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        for assessment in results:
            for finding in assessment.findings:
                assert finding.framework is not None

    def test_single_framework_assessment(self, engine, screener):
        """Can assess a single framework in isolation."""
        report = _build_full_completeness_report(screener)
        result = engine.assess_compliance(
            framework=ComplianceFramework.GHG_PROTOCOL,
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        assert isinstance(result, DetailedComplianceAssessment)
        assert result.framework == ComplianceFramework.GHG_PROTOCOL

    def test_framework_description_populated(self, engine, screener):
        """Assessment includes human-readable framework description."""
        report = _build_full_completeness_report(screener)
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(ALL_SCOPE3_CATEGORIES),
            completeness_report=report,
        )
        for assessment in results:
            assert assessment.framework_description != ""

    def test_assessment_empty_categories(self, engine, screener):
        """Empty categories list still produces structured assessment."""
        report = screener.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[],
        )
        results = engine.assess_all_frameworks(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=[],
            completeness_report=report,
        )
        assert len(results) >= 7
        for assessment in results:
            assert assessment.total_checks > 0

    def test_framework_enum_completeness(self):
        """ComplianceFramework enum has all 8 expected frameworks."""
        if not MODELS_AVAILABLE:
            pytest.skip("Models not available")
        frameworks = list(ComplianceFramework)
        assert len(frameworks) == 8
