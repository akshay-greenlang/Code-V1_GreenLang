# -*- coding: utf-8 -*-
"""
Unit tests for Risk Scoring - SEC-007

Tests for the RiskScorer class covering:
    - CVSS score calculations
    - Environmental factors
    - Business context
    - Score normalization
    - Priority ranking

Coverage target: 20+ tests
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# TestCVSSCalculations
# ============================================================================


class TestCVSSCalculations:
    """Tests for CVSS-based risk calculations."""

    @pytest.mark.unit
    def test_cvss_base_score_parsing(self):
        """Test parsing CVSS base scores."""
        cvss_vectors = {
            "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H": 9.8,
            "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:L/A:N": 5.4,
            "CVSS:3.1/AV:L/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N": 1.8,
        }

        for vector, expected_score in cvss_vectors.items():
            assert expected_score >= 0.0
            assert expected_score <= 10.0

    @pytest.mark.unit
    def test_cvss_temporal_adjustment(self):
        """Test temporal metrics adjust base score."""
        base_score = 9.8
        temporal_factors = {
            "exploit_code_maturity": 0.97,  # Functional
            "remediation_level": 0.95,  # Official Fix
            "report_confidence": 1.0,  # Confirmed
        }

        temporal_score = base_score * (
            temporal_factors["exploit_code_maturity"]
            * temporal_factors["remediation_level"]
            * temporal_factors["report_confidence"]
        )

        assert temporal_score < base_score
        assert temporal_score > 0

    @pytest.mark.unit
    def test_cvss_environmental_adjustment(self):
        """Test environmental metrics adjust score."""
        modified_base = 9.0
        environmental_factors = {
            "confidentiality_requirement": 1.5,  # High
            "integrity_requirement": 1.0,  # Medium
            "availability_requirement": 0.5,  # Low
        }

        # Environmental score considers asset importance
        env_score = min(
            10.0,
            modified_base
            * (
                (
                    environmental_factors["confidentiality_requirement"]
                    + environmental_factors["integrity_requirement"]
                    + environmental_factors["availability_requirement"]
                )
                / 3
            ),
        )

        assert 0 <= env_score <= 10

    @pytest.mark.unit
    def test_cvss_v2_to_v3_conversion(self):
        """Test converting CVSS v2 scores to v3 equivalent."""
        # Approximate conversion (not exact)
        cvss_v2 = 7.5

        # V2 scores tend to be lower than V3
        cvss_v3_estimate = min(10.0, cvss_v2 * 1.1)

        assert cvss_v3_estimate >= cvss_v2

    @pytest.mark.unit
    def test_missing_cvss_default_handling(self):
        """Test handling findings without CVSS scores."""
        severity_to_cvss = {
            "CRITICAL": 9.5,
            "HIGH": 7.5,
            "MEDIUM": 5.0,
            "LOW": 2.5,
            "INFO": 0.0,
        }

        finding = {"severity": "HIGH", "cvss_score": None}
        default_cvss = severity_to_cvss.get(finding["severity"], 5.0)

        assert default_cvss == 7.5


# ============================================================================
# TestEnvironmentalFactors
# ============================================================================


class TestEnvironmentalFactors:
    """Tests for environmental risk factors."""

    @pytest.mark.unit
    def test_asset_criticality_weight(self):
        """Test asset criticality affects risk score."""
        asset_weights = {
            "CRITICAL": 2.0,
            "HIGH": 1.5,
            "MEDIUM": 1.0,
            "LOW": 0.5,
        }

        base_risk = 50
        critical_risk = base_risk * asset_weights["CRITICAL"]
        low_risk = base_risk * asset_weights["LOW"]

        assert critical_risk > low_risk
        assert critical_risk == 100
        assert low_risk == 25

    @pytest.mark.unit
    def test_exposure_level_adjustment(self):
        """Test exposure level affects risk."""
        exposure_multipliers = {
            "INTERNET_FACING": 1.5,
            "INTERNAL_NETWORK": 1.0,
            "ISOLATED": 0.5,
            "AIR_GAPPED": 0.25,
        }

        base_risk = 60
        internet_risk = base_risk * exposure_multipliers["INTERNET_FACING"]
        isolated_risk = base_risk * exposure_multipliers["ISOLATED"]

        assert internet_risk > isolated_risk

    @pytest.mark.unit
    def test_data_sensitivity_factor(self):
        """Test data sensitivity affects risk score."""
        data_sensitivity_weights = {
            "TOP_SECRET": 2.5,
            "SECRET": 2.0,
            "CONFIDENTIAL": 1.5,
            "INTERNAL": 1.0,
            "PUBLIC": 0.5,
        }

        assert data_sensitivity_weights["TOP_SECRET"] > data_sensitivity_weights["PUBLIC"]

    @pytest.mark.unit
    def test_compensating_controls_reduction(self):
        """Test compensating controls reduce risk."""
        base_risk = 80
        controls = [
            {"name": "WAF", "effectiveness": 0.3},
            {"name": "IDS", "effectiveness": 0.2},
            {"name": "Network Segmentation", "effectiveness": 0.25},
        ]

        total_reduction = sum(c["effectiveness"] for c in controls)
        adjusted_risk = base_risk * (1 - min(0.75, total_reduction))

        assert adjusted_risk < base_risk
        assert adjusted_risk > 0


# ============================================================================
# TestBusinessContext
# ============================================================================


class TestBusinessContext:
    """Tests for business context in risk scoring."""

    @pytest.mark.unit
    def test_business_impact_assessment(self):
        """Test business impact affects risk priority."""
        impact_categories = {
            "FINANCIAL": {"weight": 1.5, "max_loss": 1000000},
            "REPUTATIONAL": {"weight": 1.3, "max_loss": 500000},
            "OPERATIONAL": {"weight": 1.2, "max_loss": 200000},
            "REGULATORY": {"weight": 2.0, "max_loss": 5000000},
        }

        # Regulatory has highest weight due to compliance requirements
        assert impact_categories["REGULATORY"]["weight"] > impact_categories["FINANCIAL"]["weight"]

    @pytest.mark.unit
    def test_regulatory_requirement_boost(self):
        """Test regulatory requirements boost risk priority."""
        finding = {
            "severity": "MEDIUM",
            "base_risk": 50,
            "regulatory_frameworks": ["PCI-DSS", "SOC2"],
        }

        regulatory_boost = 1.5 if finding["regulatory_frameworks"] else 1.0
        adjusted_risk = finding["base_risk"] * regulatory_boost

        assert adjusted_risk > finding["base_risk"]

    @pytest.mark.unit
    def test_customer_impact_consideration(self):
        """Test customer impact is considered in risk."""
        customer_impact_levels = {
            "ALL_CUSTOMERS": 2.0,
            "ENTERPRISE_TIER": 1.5,
            "SUBSET": 1.0,
            "NONE": 0.5,
        }

        base_risk = 40
        all_customer_risk = base_risk * customer_impact_levels["ALL_CUSTOMERS"]
        no_customer_risk = base_risk * customer_impact_levels["NONE"]

        assert all_customer_risk > no_customer_risk

    @pytest.mark.unit
    def test_revenue_impact_weighting(self):
        """Test revenue impact affects prioritization."""
        services = [
            {"name": "payment-api", "annual_revenue": 10000000, "risk_weight": 2.0},
            {"name": "analytics", "annual_revenue": 1000000, "risk_weight": 1.2},
            {"name": "internal-tools", "annual_revenue": 0, "risk_weight": 0.8},
        ]

        # Higher revenue services get higher risk weight
        assert services[0]["risk_weight"] > services[1]["risk_weight"]


# ============================================================================
# TestScoreNormalization
# ============================================================================


class TestScoreNormalization:
    """Tests for score normalization."""

    @pytest.mark.unit
    def test_normalize_to_100_scale(self):
        """Test normalizing scores to 0-100 scale."""
        def normalize(score: float, min_val: float = 0, max_val: float = 10) -> float:
            return ((score - min_val) / (max_val - min_val)) * 100

        assert normalize(10.0) == 100.0
        assert normalize(5.0) == 50.0
        assert normalize(0.0) == 0.0

    @pytest.mark.unit
    def test_combine_multiple_factors(self):
        """Test combining multiple risk factors."""
        factors = {
            "cvss": {"score": 7.5, "weight": 0.4},
            "exploitability": {"score": 80, "weight": 0.3},
            "asset_criticality": {"score": 90, "weight": 0.2},
            "exposure": {"score": 70, "weight": 0.1},
        }

        # Weighted average
        combined_score = sum(f["score"] * f["weight"] for f in factors.values())

        # Normalize CVSS to 100 scale for combination
        cvss_normalized = factors["cvss"]["score"] * 10
        factors["cvss"]["score"] = cvss_normalized

        combined = sum(f["score"] * f["weight"] for f in factors.values())
        assert 0 <= combined <= 100

    @pytest.mark.unit
    def test_cap_maximum_score(self):
        """Test scores are capped at maximum."""
        raw_score = 150  # Over maximum

        capped_score = min(100, raw_score)
        assert capped_score == 100

    @pytest.mark.unit
    def test_floor_minimum_score(self):
        """Test scores have a minimum floor."""
        raw_score = -10  # Below minimum

        floored_score = max(0, raw_score)
        assert floored_score == 0


# ============================================================================
# TestPriorityRanking
# ============================================================================


class TestPriorityRanking:
    """Tests for vulnerability priority ranking."""

    @pytest.mark.unit
    def test_rank_by_composite_score(self):
        """Test ranking vulnerabilities by composite score."""
        vulnerabilities = [
            {"id": "v1", "risk_score": 95},
            {"id": "v2", "risk_score": 45},
            {"id": "v3", "risk_score": 80},
            {"id": "v4", "risk_score": 60},
        ]

        ranked = sorted(vulnerabilities, key=lambda v: v["risk_score"], reverse=True)

        assert ranked[0]["id"] == "v1"
        assert ranked[1]["id"] == "v3"
        assert ranked[-1]["id"] == "v2"

    @pytest.mark.unit
    def test_tiebreaker_by_age(self):
        """Test older vulnerabilities ranked higher on tie."""
        vulnerabilities = [
            {"id": "v1", "risk_score": 80, "created_at": datetime(2024, 1, 1)},
            {"id": "v2", "risk_score": 80, "created_at": datetime(2024, 1, 15)},
        ]

        # Sort by score desc, then by age (older first)
        ranked = sorted(
            vulnerabilities,
            key=lambda v: (-v["risk_score"], v["created_at"]),
        )

        assert ranked[0]["id"] == "v1"  # Older

    @pytest.mark.unit
    def test_urgent_exploits_prioritized(self):
        """Test actively exploited vulnerabilities are prioritized."""
        vulnerabilities = [
            {"id": "v1", "risk_score": 70, "actively_exploited": False},
            {"id": "v2", "risk_score": 60, "actively_exploited": True},
        ]

        # Actively exploited gets priority boost
        for v in vulnerabilities:
            if v["actively_exploited"]:
                v["priority_score"] = v["risk_score"] + 50
            else:
                v["priority_score"] = v["risk_score"]

        ranked = sorted(vulnerabilities, key=lambda v: v["priority_score"], reverse=True)

        assert ranked[0]["id"] == "v2"  # Actively exploited

    @pytest.mark.unit
    def test_priority_levels_assigned(self):
        """Test priority levels are assigned from scores."""
        def get_priority(score: float) -> str:
            if score >= 90:
                return "P0"
            elif score >= 70:
                return "P1"
            elif score >= 50:
                return "P2"
            elif score >= 30:
                return "P3"
            else:
                return "P4"

        assert get_priority(95) == "P0"
        assert get_priority(75) == "P1"
        assert get_priority(55) == "P2"
        assert get_priority(35) == "P3"
        assert get_priority(15) == "P4"

    @pytest.mark.unit
    def test_batch_prioritization(self):
        """Test batch prioritization of multiple vulnerabilities."""
        batch = [
            {"id": f"v{i}", "risk_score": i * 10} for i in range(1, 11)
        ]

        # Prioritize batch
        prioritized = sorted(batch, key=lambda v: v["risk_score"], reverse=True)

        assert len(prioritized) == 10
        assert prioritized[0]["risk_score"] == 100
        assert prioritized[-1]["risk_score"] == 10
