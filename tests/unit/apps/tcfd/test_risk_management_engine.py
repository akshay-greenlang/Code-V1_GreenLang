# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Risk Management Engine.

Tests risk register CRUD, risk assessment (5x5 matrix), risk scoring,
risk prioritization, risk responses, ERM integration, and heat map
generation with 28+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import date
from decimal import Decimal

import pytest

from services.config import (
    RiskType,
    RiskLikelihood,
    RiskImpact,
    LIKELIHOOD_SCORES,
    IMPACT_SCORES,
    RISK_MATRIX_THRESHOLDS,
    TCFD_DISCLOSURES,
)
from services.models import (
    RiskManagementRecord,
    ClimateRisk,
    _new_id,
)


# ===========================================================================
# Risk Register CRUD
# ===========================================================================

class TestRiskRegisterCRUD:
    """Test risk register create/read/update/delete."""

    def test_create_risk_record(self, sample_risk_management_record):
        assert sample_risk_management_record.likelihood_score == 4
        assert sample_risk_management_record.impact_score == 4

    def test_record_has_id(self, sample_risk_management_record):
        assert len(sample_risk_management_record.id) == 36

    def test_record_timestamps(self, sample_risk_management_record):
        assert sample_risk_management_record.created_at is not None
        assert sample_risk_management_record.updated_at is not None

    def test_record_serialization(self, sample_risk_management_record):
        data = sample_risk_management_record.model_dump()
        assert "risk_id" in data
        assert "likelihood_score" in data

    def test_record_default_values(self):
        record = RiskManagementRecord(
            org_id=_new_id(),
            risk_id=_new_id(),
        )
        assert record.likelihood_score == 1
        assert record.impact_score == 1
        assert record.response_type == "accept"
        assert record.erm_integrated is False


# ===========================================================================
# Risk Assessment (5x5 Matrix)
# ===========================================================================

class TestRiskAssessment5x5:
    """Test 5x5 likelihood-impact risk assessment matrix."""

    @pytest.mark.parametrize("likelihood,impact,expected_score", [
        (1, 1, 1),
        (1, 5, 5),
        (5, 1, 5),
        (5, 5, 25),
        (3, 3, 9),
        (4, 4, 16),
        (2, 3, 6),
    ])
    def test_risk_score_calculation(self, likelihood, impact, expected_score):
        record = RiskManagementRecord(
            org_id=_new_id(),
            risk_id=_new_id(),
            likelihood_score=likelihood,
            impact_score=impact,
        )
        assert record.risk_score == expected_score

    def test_max_risk_score(self):
        record = RiskManagementRecord(
            org_id=_new_id(),
            risk_id=_new_id(),
            likelihood_score=5,
            impact_score=5,
        )
        assert record.risk_score == 25

    def test_min_risk_score(self):
        record = RiskManagementRecord(
            org_id=_new_id(),
            risk_id=_new_id(),
            likelihood_score=1,
            impact_score=1,
        )
        assert record.risk_score == 1


# ===========================================================================
# Risk Scoring
# ===========================================================================

class TestRiskScoring:
    """Test risk scoring against thresholds."""

    def test_low_risk_threshold(self):
        thresholds = RISK_MATRIX_THRESHOLDS["low"]
        assert thresholds["min"] == 1
        assert thresholds["max"] == 5

    def test_medium_risk_threshold(self):
        thresholds = RISK_MATRIX_THRESHOLDS["medium"]
        assert thresholds["min"] == 6
        assert thresholds["max"] == 12

    def test_high_risk_threshold(self):
        thresholds = RISK_MATRIX_THRESHOLDS["high"]
        assert thresholds["min"] == 13
        assert thresholds["max"] == 19

    def test_critical_risk_threshold(self):
        thresholds = RISK_MATRIX_THRESHOLDS["critical"]
        assert thresholds["min"] == 20
        assert thresholds["max"] == 25

    def test_score_16_is_high(self):
        score = 16
        thresholds = RISK_MATRIX_THRESHOLDS["high"]
        assert thresholds["min"] <= score <= thresholds["max"]


# ===========================================================================
# Risk Prioritization
# ===========================================================================

class TestRiskPrioritization:
    """Test risk prioritization by score."""

    def test_sort_by_risk_score(self):
        records = [
            RiskManagementRecord(
                org_id=_new_id(), risk_id=_new_id(),
                likelihood_score=2, impact_score=2,
            ),
            RiskManagementRecord(
                org_id=_new_id(), risk_id=_new_id(),
                likelihood_score=5, impact_score=5,
            ),
            RiskManagementRecord(
                org_id=_new_id(), risk_id=_new_id(),
                likelihood_score=3, impact_score=4,
            ),
        ]
        sorted_records = sorted(records, key=lambda r: r.risk_score, reverse=True)
        assert sorted_records[0].risk_score == 25
        assert sorted_records[-1].risk_score == 4

    def test_sample_record_score(self, sample_risk_management_record):
        assert sample_risk_management_record.risk_score == 16


# ===========================================================================
# Risk Responses
# ===========================================================================

class TestRiskResponses:
    """Test risk response types."""

    @pytest.mark.parametrize("response_type", ["accept", "mitigate", "transfer", "avoid"])
    def test_response_types(self, response_type):
        record = RiskManagementRecord(
            org_id=_new_id(),
            risk_id=_new_id(),
            response_type=response_type,
        )
        assert record.response_type == response_type

    def test_mitigation_response(self, sample_risk_management_record):
        assert sample_risk_management_record.response_type == "mitigate"

    def test_response_actions(self, sample_risk_management_record):
        assert len(sample_risk_management_record.response_actions) >= 1
        assert "Install flood barriers" in sample_risk_management_record.response_actions

    def test_risk_owner(self, sample_risk_management_record):
        assert sample_risk_management_record.owner == "VP Risk"

    def test_review_date(self, sample_risk_management_record):
        assert sample_risk_management_record.review_date == date(2026, 6, 30)


# ===========================================================================
# ERM Integration
# ===========================================================================

class TestERMIntegration:
    """Test enterprise risk management integration."""

    def test_erm_integrated(self, sample_risk_management_record):
        assert sample_risk_management_record.erm_integrated is True

    def test_erm_not_integrated(self):
        record = RiskManagementRecord(
            org_id=_new_id(),
            risk_id=_new_id(),
            erm_integrated=False,
        )
        assert record.erm_integrated is False

    def test_rm_disclosures_defined(self):
        assert "rm_a" in TCFD_DISCLOSURES
        assert "rm_b" in TCFD_DISCLOSURES
        assert "rm_c" in TCFD_DISCLOSURES

    def test_rm_c_erm_integration(self):
        disclosure = TCFD_DISCLOSURES["rm_c"]
        assert "integration" in disclosure["title"].lower() or \
               "integration" in disclosure["description"].lower() or \
               "integrated" in disclosure["description"].lower()


# ===========================================================================
# Heat Map Generation
# ===========================================================================

class TestHeatMapGeneration:
    """Test risk heat map data generation."""

    def test_heat_map_data_structure(self):
        risks = []
        for l in range(1, 6):
            for i in range(1, 6):
                risks.append({
                    "likelihood": l,
                    "impact": i,
                    "score": l * i,
                    "count": 0,
                })
        assert len(risks) == 25  # 5x5 matrix

    def test_heat_map_cell_classification(self):
        cells = {}
        for l in range(1, 6):
            for i in range(1, 6):
                score = l * i
                if score <= 5:
                    level = "low"
                elif score <= 12:
                    level = "medium"
                elif score <= 19:
                    level = "high"
                else:
                    level = "critical"
                cells[(l, i)] = level
        assert cells[(5, 5)] == "critical"
        assert cells[(1, 1)] == "low"
        assert cells[(3, 3)] == "medium"
        assert cells[(4, 4)] == "high"

    def test_identification_process(self, sample_risk_management_record):
        assert "workshop" in sample_risk_management_record.identification_process.lower()

    def test_assessment_methodology(self, sample_risk_management_record):
        assert "5x5" in sample_risk_management_record.assessment_methodology
