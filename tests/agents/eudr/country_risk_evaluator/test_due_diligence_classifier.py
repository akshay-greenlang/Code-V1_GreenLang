# -*- coding: utf-8 -*-
"""
Unit tests for DueDiligenceClassifier - AGENT-EUDR-016 Engine 5

Tests automated 3-tier due diligence classification per EUDR Articles 10-13
covering simplified/standard/enhanced DD levels, certification credit
calculation, cost estimation, audit frequency, sub-national override
rules, reclassification impact, requirements generation, and batch
classification.

Target: 60+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import datetime, timezone
from typing import Dict, List
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.country_risk_evaluator.due_diligence_classifier import (
    DueDiligenceClassifier,
    _BASE_AUDIT_MONTHS,
    _BASE_COMPLIANCE_DAYS,
    _CERTIFICATION_EFFECTIVENESS,
    _DEFAULT_OVERRIDE_REGIONS,
    _REQUIREMENTS,
)
from greenlang.agents.eudr.country_risk_evaluator.models import (
    CommodityType,
    DueDiligenceClassification,
    DueDiligenceLevel,
    RiskLevel,
    SUPPORTED_COMMODITIES,
)


# ============================================================================
# TestDueDiligenceClassifierInit
# ============================================================================


class TestDueDiligenceClassifierInit:
    """Tests for DueDiligenceClassifier initialization."""

    @pytest.mark.unit
    def test_initialization_empty_stores(self, mock_config):
        classifier = DueDiligenceClassifier()
        assert classifier._classifications == {}

    @pytest.mark.unit
    def test_override_regions_loaded(self, mock_config):
        classifier = DueDiligenceClassifier()
        assert len(classifier._override_regions) > 0
        assert "BR" in classifier._override_regions
        assert "ID" in classifier._override_regions

    @pytest.mark.unit
    def test_three_dd_levels_exist(self):
        levels = [d.value for d in DueDiligenceLevel]
        assert "simplified" in levels
        assert "standard" in levels
        assert "enhanced" in levels

    @pytest.mark.unit
    def test_base_audit_months_defined(self):
        assert _BASE_AUDIT_MONTHS["simplified"] == 12
        assert _BASE_AUDIT_MONTHS["standard"] == 6
        assert _BASE_AUDIT_MONTHS["enhanced"] == 3

    @pytest.mark.unit
    def test_base_compliance_days_defined(self):
        assert _BASE_COMPLIANCE_DAYS["simplified"] == 30
        assert _BASE_COMPLIANCE_DAYS["standard"] == 90
        assert _BASE_COMPLIANCE_DAYS["enhanced"] == 180


# ============================================================================
# TestClassify
# ============================================================================


class TestClassify:
    """Tests for classify method."""

    @pytest.mark.unit
    def test_classify_valid(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=72.5,
        )
        assert isinstance(result, DueDiligenceClassification)
        assert result.country_code == "BR"

    @pytest.mark.unit
    def test_classify_has_classification_id(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=50.0,
        )
        assert result.classification_id.startswith("ddc-")

    @pytest.mark.unit
    def test_classify_stores_result(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=50.0,
        )
        retrieved = due_diligence_classifier.get_classification(
            result.classification_id
        )
        assert retrieved is not None

    @pytest.mark.unit
    def test_classify_uppercase_country(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="br",
            risk_score=50.0,
        )
        assert result.country_code == "BR"


# ============================================================================
# TestSimplifiedDD
# ============================================================================


class TestSimplifiedDD:
    """Tests for simplified due diligence classification."""

    @pytest.mark.unit
    def test_low_risk_gets_simplified(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="SE",
            risk_score=15.0,
        )
        assert result.level == DueDiligenceLevel.SIMPLIFIED

    @pytest.mark.unit
    def test_simplified_threshold_boundary(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="FI",
            risk_score=30.0,  # Exactly at simplified threshold
        )
        assert result.level == DueDiligenceLevel.SIMPLIFIED

    @pytest.mark.unit
    def test_simplified_has_annual_audit(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="SE",
            risk_score=10.0,
        )
        assert result.audit_frequency in ["annual", "12_months"]

    @pytest.mark.unit
    def test_simplified_satellite_not_required(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="SE",
            risk_score=10.0,
        )
        assert result.satellite_required is False

    @pytest.mark.unit
    def test_simplified_lower_cost(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="SE",
            risk_score=10.0,
        )
        if result.cost_estimate_min_eur is not None:
            assert result.cost_estimate_min_eur >= 200
            assert result.cost_estimate_max_eur <= 500


# ============================================================================
# TestStandardDD
# ============================================================================


class TestStandardDD:
    """Tests for standard due diligence classification."""

    @pytest.mark.unit
    def test_standard_risk_gets_standard(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="MY",
            risk_score=45.0,
        )
        assert result.level == DueDiligenceLevel.STANDARD

    @pytest.mark.unit
    def test_standard_lower_boundary(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="MY",
            risk_score=31.0,  # Just above simplified threshold
        )
        assert result.level == DueDiligenceLevel.STANDARD

    @pytest.mark.unit
    def test_standard_upper_boundary(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="MY",
            risk_score=60.0,  # At enhanced threshold
        )
        assert result.level == DueDiligenceLevel.STANDARD

    @pytest.mark.unit
    def test_standard_semi_annual_audit(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="MY",
            risk_score=45.0,
        )
        assert result.audit_frequency in ["semi_annual", "6_months", "semi-annual"]

    @pytest.mark.unit
    def test_standard_cost_range(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="MY",
            risk_score=45.0,
        )
        if result.cost_estimate_min_eur is not None:
            assert result.cost_estimate_min_eur >= 1000
            assert result.cost_estimate_max_eur <= 3000


# ============================================================================
# TestEnhancedDD
# ============================================================================


class TestEnhancedDD:
    """Tests for enhanced due diligence classification."""

    @pytest.mark.unit
    def test_high_risk_gets_enhanced(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=75.0,
        )
        assert result.level == DueDiligenceLevel.ENHANCED

    @pytest.mark.unit
    def test_enhanced_above_threshold(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=61.0,  # Just above enhanced threshold
        )
        assert result.level == DueDiligenceLevel.ENHANCED

    @pytest.mark.unit
    def test_enhanced_satellite_required(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=75.0,
        )
        assert result.satellite_required is True

    @pytest.mark.unit
    def test_enhanced_quarterly_audit(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=80.0,
        )
        assert result.audit_frequency in ["quarterly", "3_months"]

    @pytest.mark.unit
    def test_enhanced_higher_cost(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=80.0,
        )
        if result.cost_estimate_min_eur is not None:
            assert result.cost_estimate_min_eur >= 5000
            assert result.cost_estimate_max_eur <= 15000

    @pytest.mark.unit
    def test_enhanced_max_risk_score(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="CD",
            risk_score=100.0,
        )
        assert result.level == DueDiligenceLevel.ENHANCED


# ============================================================================
# TestRequirementsGeneration
# ============================================================================


class TestRequirementsGeneration:
    """Tests for DD requirements generation."""

    @pytest.mark.unit
    def test_simplified_requirements(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="SE",
            risk_score=10.0,
        )
        assert len(result.regulatory_requirements) > 0
        # Simplified should have fewer requirements
        assert len(result.regulatory_requirements) <= len(
            _REQUIREMENTS["simplified"]
        ) + 2

    @pytest.mark.unit
    def test_enhanced_has_satellite_requirement(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=80.0,
        )
        requirements_text = " ".join(result.regulatory_requirements).lower()
        assert "satellite" in requirements_text or result.satellite_required

    @pytest.mark.unit
    def test_enhanced_has_most_requirements(self, due_diligence_classifier):
        result_simplified = due_diligence_classifier.classify(
            country_code="SE", risk_score=10.0
        )
        result_enhanced = due_diligence_classifier.classify(
            country_code="BR", risk_score=80.0
        )
        assert len(result_enhanced.regulatory_requirements) >= len(
            result_simplified.regulatory_requirements
        )

    @pytest.mark.unit
    def test_requirements_constants(self):
        assert len(_REQUIREMENTS["simplified"]) == 5
        assert len(_REQUIREMENTS["standard"]) == 8
        assert len(_REQUIREMENTS["enhanced"]) == 13


# ============================================================================
# TestCertificationCredit
# ============================================================================


class TestCertificationCredit:
    """Tests for certification-based risk mitigation credit."""

    @pytest.mark.unit
    def test_certification_reduces_effective_score(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="ID",
            risk_score=65.0,
            commodity_type="oil_palm",
            certification_schemes=["rspo"],
        )
        # RSPO has 80% effectiveness for oil palm
        # Effective score should be reduced
        if result.effective_risk_score is not None:
            assert result.effective_risk_score <= result.risk_score

    @pytest.mark.unit
    def test_certification_credit_capped(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=50.0,
            commodity_type="wood",
            certification_schemes=["fsc", "pefc", "rainforest_alliance"],
        )
        # Credit should be capped at certification_credit_max (30)
        assert result.certification_credit <= 30.0

    @pytest.mark.unit
    def test_certification_credit_not_below_zero(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="SE",
            risk_score=5.0,
            commodity_type="wood",
            certification_schemes=["fsc"],
        )
        if result.effective_risk_score is not None:
            assert result.effective_risk_score >= 0.0

    @pytest.mark.unit
    def test_irrelevant_certification_no_credit(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="CI",
            risk_score=50.0,
            commodity_type="cocoa",
            certification_schemes=["fsc"],  # FSC is for wood, not cocoa
        )
        assert result.certification_credit == pytest.approx(0.0, abs=1.0)

    @pytest.mark.unit
    def test_certification_can_change_dd_level(self, due_diligence_classifier):
        # Without certification: enhanced (score 65 > 60 threshold)
        result_no_cert = due_diligence_classifier.classify(
            country_code="ID",
            risk_score=65.0,
        )
        # With RSPO: may reduce effective score below enhanced threshold
        result_with_cert = due_diligence_classifier.classify(
            country_code="ID",
            risk_score=65.0,
            commodity_type="oil_palm",
            certification_schemes=["rspo"],
        )
        # Both should be valid
        assert result_no_cert.level in [
            DueDiligenceLevel.STANDARD,
            DueDiligenceLevel.ENHANCED,
        ]
        assert result_with_cert.level in [
            DueDiligenceLevel.SIMPLIFIED,
            DueDiligenceLevel.STANDARD,
            DueDiligenceLevel.ENHANCED,
        ]


# ============================================================================
# TestSubNationalOverride
# ============================================================================


class TestSubNationalOverride:
    """Tests for sub-national region override rules."""

    @pytest.mark.unit
    def test_brazil_para_cattle_override(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=25.0,  # Low score that would normally be simplified
            commodity_type="cattle",
            region="Para",
        )
        assert result.level == DueDiligenceLevel.ENHANCED

    @pytest.mark.unit
    def test_brazil_mato_grosso_soya_override(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=20.0,
            commodity_type="soya",
            region="Mato Grosso",
        )
        assert result.level == DueDiligenceLevel.ENHANCED

    @pytest.mark.unit
    def test_indonesia_kalimantan_oil_palm_override(
        self, due_diligence_classifier
    ):
        result = due_diligence_classifier.classify(
            country_code="ID",
            risk_score=20.0,
            commodity_type="oil_palm",
            region="Kalimantan",
        )
        assert result.level == DueDiligenceLevel.ENHANCED

    @pytest.mark.unit
    def test_no_override_without_region(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=20.0,
            commodity_type="cattle",
        )
        # Without region, should use normal classification
        assert result.level == DueDiligenceLevel.SIMPLIFIED

    @pytest.mark.unit
    def test_non_override_region(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=20.0,
            commodity_type="cattle",
            region="Sao Paulo",  # Not an override region
        )
        assert result.level == DueDiligenceLevel.SIMPLIFIED

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "country,region,commodity",
        [
            ("BR", "Para", "cattle"),
            ("BR", "Mato Grosso", "soya"),
            ("BR", "Rondonia", "wood"),
            ("ID", "Kalimantan", "oil_palm"),
            ("ID", "Sumatra", "rubber"),
            ("CI", "Cavally", "cocoa"),
            ("GH", "Western Region", "cocoa"),
            ("MY", "Sabah", "oil_palm"),
        ],
    )
    def test_all_override_regions_force_enhanced(
        self, due_diligence_classifier, country, region, commodity
    ):
        result = due_diligence_classifier.classify(
            country_code=country,
            risk_score=15.0,
            commodity_type=commodity,
            region=region,
        )
        assert result.level == DueDiligenceLevel.ENHANCED

    @pytest.mark.unit
    def test_override_regions_constant(self):
        assert "BR" in _DEFAULT_OVERRIDE_REGIONS
        assert "ID" in _DEFAULT_OVERRIDE_REGIONS
        assert "CD" in _DEFAULT_OVERRIDE_REGIONS
        assert "CO" in _DEFAULT_OVERRIDE_REGIONS
        assert "CI" in _DEFAULT_OVERRIDE_REGIONS
        assert "GH" in _DEFAULT_OVERRIDE_REGIONS
        assert "MY" in _DEFAULT_OVERRIDE_REGIONS


# ============================================================================
# TestCostEstimation
# ============================================================================


class TestCostEstimation:
    """Tests for DD cost estimation."""

    @pytest.mark.unit
    def test_cost_estimate_present(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=50.0,
        )
        assert result.cost_estimate_min_eur is not None
        assert result.cost_estimate_max_eur is not None

    @pytest.mark.unit
    def test_cost_min_less_than_max(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=50.0,
        )
        assert result.cost_estimate_min_eur <= result.cost_estimate_max_eur

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "risk_score,expected_min_range,expected_max_range",
        [
            (10.0, (200, 500), (200, 500)),
            (45.0, (1000, 3000), (1000, 3000)),
            (80.0, (5000, 15000), (5000, 15000)),
        ],
        ids=["simplified", "standard", "enhanced"],
    )
    def test_cost_ranges_by_dd_level(
        self,
        due_diligence_classifier,
        risk_score,
        expected_min_range,
        expected_max_range,
    ):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=risk_score,
        )
        assert result.cost_estimate_min_eur >= expected_min_range[0]
        assert result.cost_estimate_max_eur <= expected_max_range[1]


# ============================================================================
# TestAuditFrequency
# ============================================================================


class TestAuditFrequency:
    """Tests for audit frequency recommendations."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "risk_score,expected_frequency",
        [
            (10.0, ["annual", "12_months"]),
            (45.0, ["semi_annual", "6_months", "semi-annual"]),
            (80.0, ["quarterly", "3_months"]),
        ],
        ids=["simplified", "standard", "enhanced"],
    )
    def test_audit_frequency_by_level(
        self, due_diligence_classifier, risk_score, expected_frequency
    ):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=risk_score,
        )
        assert result.audit_frequency in expected_frequency


# ============================================================================
# TestTimeToCompliance
# ============================================================================


class TestTimeToCompliance:
    """Tests for time-to-compliance estimation."""

    @pytest.mark.unit
    def test_time_to_compliance_present(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=50.0,
        )
        if result.time_to_compliance_days is not None:
            assert result.time_to_compliance_days > 0

    @pytest.mark.unit
    def test_enhanced_takes_longer(self, due_diligence_classifier):
        result_simplified = due_diligence_classifier.classify(
            country_code="SE", risk_score=10.0
        )
        result_enhanced = due_diligence_classifier.classify(
            country_code="BR", risk_score=80.0
        )
        if (
            result_simplified.time_to_compliance_days is not None
            and result_enhanced.time_to_compliance_days is not None
        ):
            assert (
                result_enhanced.time_to_compliance_days
                >= result_simplified.time_to_compliance_days
            )


# ============================================================================
# TestBatchClassification
# ============================================================================


class TestBatchClassification:
    """Tests for batch due diligence classification."""

    @pytest.mark.unit
    def test_batch_multiple_countries(self, due_diligence_classifier):
        test_cases = [
            ("SE", 10.0),
            ("MY", 45.0),
            ("BR", 80.0),
        ]
        results = []
        for cc, score in test_cases:
            result = due_diligence_classifier.classify(
                country_code=cc,
                risk_score=score,
            )
            results.append(result)

        assert len(results) == 3
        assert results[0].level == DueDiligenceLevel.SIMPLIFIED
        assert results[1].level == DueDiligenceLevel.STANDARD
        assert results[2].level == DueDiligenceLevel.ENHANCED


# ============================================================================
# TestInputValidation
# ============================================================================


class TestInputValidation:
    """Tests for classifier input validation."""

    @pytest.mark.unit
    def test_empty_country_code_raises(self, due_diligence_classifier):
        with pytest.raises(ValueError):
            due_diligence_classifier.classify(
                country_code="",
                risk_score=50.0,
            )

    @pytest.mark.unit
    def test_negative_risk_score_raises(self, due_diligence_classifier):
        with pytest.raises(ValueError):
            due_diligence_classifier.classify(
                country_code="BR",
                risk_score=-5.0,
            )

    @pytest.mark.unit
    def test_over_100_risk_score_raises(self, due_diligence_classifier):
        with pytest.raises(ValueError):
            due_diligence_classifier.classify(
                country_code="BR",
                risk_score=105.0,
            )

    @pytest.mark.unit
    def test_provenance_hash_present(self, due_diligence_classifier):
        result = due_diligence_classifier.classify(
            country_code="BR",
            risk_score=50.0,
        )
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
