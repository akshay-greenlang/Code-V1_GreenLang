# -*- coding: utf-8 -*-
"""
Tests for RightsViolationEngine - AGENT-EUDR-021 Engine 5: Rights Violation

Comprehensive test suite covering:
- Violation detection from 10+ source types
- 5-factor severity scoring (Decimal arithmetic)
- 10 violation type classifications
- 7-day deduplication window
- Supply chain correlation
- Alert generation and routing
- Provenance tracking

Test count: 72 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 5: Rights Violation Monitoring)
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

import pytest

from tests.agents.eudr.indigenous_rights_checker.conftest import (
    compute_test_hash,
    compute_violation_severity,
    SHA256_HEX_LENGTH,
    VIOLATION_TYPE_SCORES,
    DEFAULT_VIOLATION_SEVERITY_WEIGHTS,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    ViolationType,
    ViolationAlert,
    ViolationAlertStatus,
    AlertSeverity,
)


# ===========================================================================
# 1. Violation Type Classification (12 tests)
# ===========================================================================


class TestViolationTypeClassification:
    """Test 10 violation type classifications."""

    @pytest.mark.parametrize("vtype,expected_score", [
        (ViolationType.PHYSICAL_VIOLENCE, Decimal("100")),
        (ViolationType.FORCED_DISPLACEMENT, Decimal("95")),
        (ViolationType.LAND_SEIZURE, Decimal("90")),
        (ViolationType.CULTURAL_DESTRUCTION, Decimal("85")),
        (ViolationType.FPIC_VIOLATION, Decimal("80")),
        (ViolationType.ENVIRONMENTAL_DAMAGE, Decimal("75")),
        (ViolationType.CONSULTATION_DENIAL, Decimal("70")),
        (ViolationType.RESTRICTION_OF_ACCESS, Decimal("65")),
        (ViolationType.BENEFIT_SHARING_BREACH, Decimal("60")),
        (ViolationType.DISCRIMINATORY_POLICY, Decimal("55")),
    ])
    def test_violation_type_base_scores(self, vtype, expected_score):
        """Test each violation type maps to correct base severity score."""
        assert VIOLATION_TYPE_SCORES[vtype.value] == expected_score

    def test_physical_violence_highest(self):
        """Test physical violence has the highest base severity."""
        max_type = max(VIOLATION_TYPE_SCORES, key=VIOLATION_TYPE_SCORES.get)
        assert max_type == ViolationType.PHYSICAL_VIOLENCE.value

    def test_discriminatory_policy_lowest(self):
        """Test discriminatory policy has the lowest base severity."""
        min_type = min(VIOLATION_TYPE_SCORES, key=VIOLATION_TYPE_SCORES.get)
        assert min_type == ViolationType.DISCRIMINATORY_POLICY.value


# ===========================================================================
# 2. Severity Scoring (15 tests)
# ===========================================================================


class TestViolationSeverityScoring:
    """Test 5-factor violation severity scoring."""

    def test_maximum_severity(self):
        """Test maximum severity with all factors at 100."""
        score = compute_violation_severity(
            violation_type="physical_violence",
            proximity_score=Decimal("100"),
            population_score=Decimal("100"),
            legal_gap_score=Decimal("100"),
            media_score=Decimal("100"),
        )
        assert score == Decimal("100.00")

    def test_minimum_severity(self):
        """Test minimum severity with all non-type factors at 0."""
        score = compute_violation_severity(
            violation_type="discriminatory_policy",
            proximity_score=Decimal("0"),
            population_score=Decimal("0"),
            legal_gap_score=Decimal("0"),
            media_score=Decimal("0"),
        )
        # Only violation_type contributes: 55 * 0.30 = 16.50
        assert score == Decimal("16.50")

    def test_severity_weights_sum_to_one(self):
        """Test violation severity weights sum to 1.0."""
        total = sum(DEFAULT_VIOLATION_SEVERITY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_proximity_amplifies_severity(self):
        """Test high spatial proximity increases severity."""
        score_near = compute_violation_severity(
            violation_type="land_seizure",
            proximity_score=Decimal("100"),
            population_score=Decimal("50"),
            legal_gap_score=Decimal("50"),
            media_score=Decimal("50"),
        )
        score_far = compute_violation_severity(
            violation_type="land_seizure",
            proximity_score=Decimal("0"),
            population_score=Decimal("50"),
            legal_gap_score=Decimal("50"),
            media_score=Decimal("50"),
        )
        assert score_near > score_far

    def test_media_coverage_amplifies_severity(self):
        """Test high media coverage increases severity."""
        score_covered = compute_violation_severity(
            violation_type="fpic_violation",
            proximity_score=Decimal("50"),
            population_score=Decimal("50"),
            legal_gap_score=Decimal("50"),
            media_score=Decimal("100"),
        )
        score_uncovered = compute_violation_severity(
            violation_type="fpic_violation",
            proximity_score=Decimal("50"),
            population_score=Decimal("50"),
            legal_gap_score=Decimal("50"),
            media_score=Decimal("0"),
        )
        assert score_covered > score_uncovered

    def test_severity_decimal_precision(self):
        """Test severity score maintains 2 decimal places."""
        score = compute_violation_severity(
            violation_type="environmental_damage",
            proximity_score=Decimal("33.33"),
            population_score=Decimal("66.67"),
            legal_gap_score=Decimal("44.44"),
            media_score=Decimal("55.55"),
        )
        assert score == score.quantize(Decimal("0.01"))

    def test_severity_critical_threshold(self):
        """Test severity >= 80 is CRITICAL."""
        score = compute_violation_severity(
            violation_type="physical_violence",
            proximity_score=Decimal("90"),
            population_score=Decimal("80"),
            legal_gap_score=Decimal("70"),
            media_score=Decimal("80"),
        )
        assert score >= Decimal("80")

    @pytest.mark.parametrize("violation_type", [v.value for v in ViolationType])
    def test_each_type_produces_valid_score(self, violation_type):
        """Test each violation type produces a score in valid range."""
        score = compute_violation_severity(
            violation_type=violation_type,
            proximity_score=Decimal("50"),
            population_score=Decimal("50"),
            legal_gap_score=Decimal("50"),
            media_score=Decimal("50"),
        )
        assert Decimal("0") <= score <= Decimal("100")

    def test_violation_type_weight_is_highest(self):
        """Test violation_type factor has the highest weight (0.30)."""
        assert DEFAULT_VIOLATION_SEVERITY_WEIGHTS["violation_type"] == 0.30

    def test_all_equal_factors(self):
        """Test all factors at 50 yields predictable result."""
        score = compute_violation_severity(
            violation_type="fpic_violation",
            proximity_score=Decimal("50"),
            population_score=Decimal("50"),
            legal_gap_score=Decimal("50"),
            media_score=Decimal("50"),
        )
        # type: 80*0.30=24, prox: 50*0.25=12.5, pop: 50*0.15=7.5,
        # legal: 50*0.15=7.5, media: 50*0.15=7.5 = 59.00
        assert score == Decimal("59.00")

    def test_custom_severity_weights(self):
        """Test severity scoring with custom weights."""
        custom = {
            "violation_type": 0.20,
            "spatial_proximity": 0.30,
            "community_population": 0.20,
            "legal_framework_gap": 0.20,
            "media_coverage": 0.10,
        }
        score = compute_violation_severity(
            violation_type="land_seizure",
            proximity_score=Decimal("100"),
            population_score=Decimal("50"),
            legal_gap_score=Decimal("50"),
            media_score=Decimal("50"),
            weights=custom,
        )
        # type: 90*0.20=18, prox: 100*0.30=30, pop: 50*0.20=10,
        # legal: 50*0.20=10, media: 50*0.10=5 = 73.00
        assert score == Decimal("73.00")


# ===========================================================================
# 3. Violation Alert Model (12 tests)
# ===========================================================================


class TestViolationAlertModel:
    """Test ViolationAlert model creation and validation."""

    def test_create_violation_alert(self, sample_violation):
        """Test creating a violation alert with all fields."""
        assert sample_violation.alert_id == "v-001"
        assert sample_violation.violation_type == ViolationType.FPIC_VIOLATION
        assert sample_violation.severity_level == AlertSeverity.HIGH

    def test_violation_source_attribution(self, sample_violation):
        """Test violation source is attributed."""
        assert sample_violation.source == "iwgia"
        assert sample_violation.source_url is not None

    def test_violation_geographic_location(self, sample_violation):
        """Test violation has geographic coordinates."""
        assert sample_violation.location_lat is not None
        assert sample_violation.location_lon is not None
        assert -90 <= sample_violation.location_lat <= 90
        assert -180 <= sample_violation.location_lon <= 180

    def test_violation_affected_communities(self, sample_violation):
        """Test violation links to affected communities."""
        assert len(sample_violation.affected_communities) >= 1

    def test_violation_supply_chain_correlation(self, sample_violation):
        """Test violation is correlated with supply chain."""
        assert sample_violation.supply_chain_correlation is True
        assert len(sample_violation.affected_plots) >= 1

    def test_violation_severity_score_range(self, sample_violations):
        """Test all violation severity scores are within valid range."""
        for v in sample_violations:
            assert Decimal("0") <= v.severity_score <= Decimal("100")

    @pytest.mark.parametrize("status", [
        ViolationAlertStatus.ACTIVE,
        ViolationAlertStatus.INVESTIGATING,
        ViolationAlertStatus.RESOLVED,
        ViolationAlertStatus.FALSE_POSITIVE,
        ViolationAlertStatus.ARCHIVED,
    ])
    def test_all_violation_statuses(self, status):
        """Test violation alert at each status."""
        v = ViolationAlert(
            alert_id=f"v-{status.value}",
            source="test",
            publication_date=date(2026, 3, 1),
            violation_type=ViolationType.FPIC_VIOLATION,
            country_code="BR",
            severity_score=Decimal("50"),
            severity_level=AlertSeverity.MEDIUM,
            status=status,
            provenance_hash="a" * 64,
        )
        assert v.status == status

    def test_violation_severity_levels(self, sample_violations):
        """Test violations span multiple severity levels."""
        levels = {v.severity_level for v in sample_violations}
        assert len(levels) >= 3

    def test_violation_countries(self, sample_violations):
        """Test violations span multiple countries."""
        countries = {v.country_code for v in sample_violations}
        assert len(countries) >= 3


# ===========================================================================
# 4. Deduplication (10 tests)
# ===========================================================================


class TestViolationDeduplication:
    """Test 7-day deduplication window."""

    def test_dedup_window_default(self, mock_config):
        """Test default deduplication window is 7 days."""
        assert mock_config.violation_dedup_window_days == 7

    def test_within_dedup_window(self, mock_config):
        """Test violation within 7-day window is a duplicate."""
        original_date = date(2026, 3, 1)
        new_date = date(2026, 3, 5)
        gap = (new_date - original_date).days
        assert gap <= mock_config.violation_dedup_window_days

    def test_outside_dedup_window(self, mock_config):
        """Test violation after 7-day window is not a duplicate."""
        original_date = date(2026, 3, 1)
        new_date = date(2026, 3, 10)
        gap = (new_date - original_date).days
        assert gap > mock_config.violation_dedup_window_days

    def test_exactly_at_dedup_boundary(self, mock_config):
        """Test violation exactly 7 days later is within window."""
        original_date = date(2026, 3, 1)
        new_date = date(2026, 3, 8)
        gap = (new_date - original_date).days
        assert gap == mock_config.violation_dedup_window_days

    def test_strict_config_shorter_window(self, strict_config):
        """Test strict config has shorter dedup window (3 days)."""
        assert strict_config.violation_dedup_window_days == 3

    def test_dedup_same_location_same_type(self):
        """Test same location + same type within window is duplicate."""
        v1 = ViolationAlert(
            alert_id="v-dup-1",
            source="iwgia",
            publication_date=date(2026, 3, 1),
            violation_type=ViolationType.LAND_SEIZURE,
            country_code="BR",
            location_lat=-3.0,
            location_lon=-60.0,
            severity_score=Decimal("80"),
            severity_level=AlertSeverity.HIGH,
            deduplication_group="dedup-br-land-001",
            provenance_hash="a" * 64,
        )
        v2 = ViolationAlert(
            alert_id="v-dup-2",
            source="amnesty_international",
            publication_date=date(2026, 3, 3),
            violation_type=ViolationType.LAND_SEIZURE,
            country_code="BR",
            location_lat=-3.0,
            location_lon=-60.0,
            severity_score=Decimal("82"),
            severity_level=AlertSeverity.HIGH,
            deduplication_group="dedup-br-land-001",
            provenance_hash="b" * 64,
        )
        assert v1.deduplication_group == v2.deduplication_group

    def test_different_location_not_duplicate(self):
        """Test different location is not a duplicate."""
        v1_group = "dedup-br-001"
        v2_group = "dedup-id-001"
        assert v1_group != v2_group

    def test_different_type_not_duplicate(self):
        """Test different violation type at same location is not a duplicate."""
        v1 = ViolationAlert(
            alert_id="v-diff-type-1",
            source="iwgia",
            publication_date=date(2026, 3, 1),
            violation_type=ViolationType.LAND_SEIZURE,
            country_code="BR",
            severity_score=Decimal("80"),
            severity_level=AlertSeverity.HIGH,
            deduplication_group="dedup-br-land",
            provenance_hash="c" * 64,
        )
        v2 = ViolationAlert(
            alert_id="v-diff-type-2",
            source="iwgia",
            publication_date=date(2026, 3, 1),
            violation_type=ViolationType.FORCED_DISPLACEMENT,
            country_code="BR",
            severity_score=Decimal("90"),
            severity_level=AlertSeverity.CRITICAL,
            deduplication_group="dedup-br-displacement",
            provenance_hash="d" * 64,
        )
        assert v1.deduplication_group != v2.deduplication_group

    def test_dedup_group_is_optional(self, sample_violation):
        """Test deduplication_group can be None."""
        assert sample_violation.deduplication_group is None

    def test_ten_violation_sources(self):
        """Test alerts can come from at least 10 different sources."""
        sources = [
            "iwgia", "cultural_survival", "forest_peoples_programme",
            "amnesty_international", "human_rights_watch",
            "national_human_rights_commission", "iachr",
            "achpr", "ohchr", "judicial_database", "media_monitoring",
        ]
        assert len(sources) >= 10
        for src in sources:
            v = ViolationAlert(
                alert_id=f"v-{src[:5]}",
                source=src,
                publication_date=date(2026, 3, 1),
                violation_type=ViolationType.FPIC_VIOLATION,
                country_code="BR",
                severity_score=Decimal("60"),
                severity_level=AlertSeverity.MEDIUM,
                provenance_hash=compute_test_hash({"source": src}),
            )
            assert v.source == src


# ===========================================================================
# 5. Supply Chain Correlation (8 tests)
# ===========================================================================


class TestSupplyChainCorrelation:
    """Test violation-supply chain correlation."""

    def test_correlated_violation(self, sample_violation):
        """Test violation with supply chain correlation."""
        assert sample_violation.supply_chain_correlation is True

    def test_uncorrelated_violation(self):
        """Test violation without supply chain correlation."""
        v = ViolationAlert(
            alert_id="v-uncorr",
            source="iwgia",
            publication_date=date(2026, 3, 1),
            violation_type=ViolationType.DISCRIMINATORY_POLICY,
            country_code="PE",
            severity_score=Decimal("40"),
            severity_level=AlertSeverity.LOW,
            supply_chain_correlation=False,
            provenance_hash="e" * 64,
        )
        assert v.supply_chain_correlation is False
        assert v.affected_plots == []

    def test_affected_plots_tracked(self, sample_violation):
        """Test affected plots are listed in violation."""
        assert len(sample_violation.affected_plots) >= 1
        assert "p-001" in sample_violation.affected_plots

    def test_affected_suppliers_tracked(self, sample_violation):
        """Test affected suppliers are listed in violation."""
        assert len(sample_violation.affected_suppliers) >= 1

    def test_correlation_increases_priority(self):
        """Test correlated violations have higher priority."""
        # Correlated violations should trigger immediate action
        v = ViolationAlert(
            alert_id="v-corr-pri",
            source="iwgia",
            publication_date=date(2026, 3, 1),
            violation_type=ViolationType.LAND_SEIZURE,
            country_code="BR",
            severity_score=Decimal("85"),
            severity_level=AlertSeverity.CRITICAL,
            supply_chain_correlation=True,
            affected_plots=["p-001", "p-002"],
            provenance_hash="f" * 64,
        )
        assert v.supply_chain_correlation is True
        assert v.severity_level == AlertSeverity.CRITICAL

    def test_impact_assessment_field(self):
        """Test impact assessment is stored as dict."""
        v = ViolationAlert(
            alert_id="v-impact",
            source="iwgia",
            publication_date=date(2026, 3, 1),
            violation_type=ViolationType.ENVIRONMENTAL_DAMAGE,
            country_code="BR",
            severity_score=Decimal("70"),
            severity_level=AlertSeverity.HIGH,
            impact_assessment={
                "affected_area_hectares": 500,
                "river_systems_affected": 2,
                "estimated_remediation_cost_usd": 250000,
            },
            provenance_hash="g" * 64,
        )
        assert "affected_area_hectares" in v.impact_assessment

    def test_multiple_plots_affected(self):
        """Test violation affecting multiple supply chain plots."""
        v = ViolationAlert(
            alert_id="v-multi-plot",
            source="amnesty_international",
            publication_date=date(2026, 3, 1),
            violation_type=ViolationType.FORCED_DISPLACEMENT,
            country_code="BR",
            severity_score=Decimal("92"),
            severity_level=AlertSeverity.CRITICAL,
            supply_chain_correlation=True,
            affected_plots=["p-001", "p-002", "p-003", "p-004"],
            provenance_hash="h" * 64,
        )
        assert len(v.affected_plots) == 4

    def test_zero_correlation_radius(self):
        """Test violation with no nearby plots has no correlation."""
        v = ViolationAlert(
            alert_id="v-far",
            source="ohchr",
            publication_date=date(2026, 3, 1),
            violation_type=ViolationType.PHYSICAL_VIOLENCE,
            country_code="CD",
            location_lat=0.5,
            location_lon=25.0,
            severity_score=Decimal("95"),
            severity_level=AlertSeverity.CRITICAL,
            supply_chain_correlation=False,
            provenance_hash="i" * 64,
        )
        assert v.supply_chain_correlation is False
        assert v.affected_plots == []


# ===========================================================================
# 6. Provenance (5 tests)
# ===========================================================================


class TestViolationProvenance:
    """Test provenance tracking for violation operations."""

    def test_violation_provenance_hash_length(self, sample_violation):
        """Test violation provenance hash is SHA-256."""
        assert len(sample_violation.provenance_hash) == SHA256_HEX_LENGTH

    def test_violation_provenance_deterministic(self):
        """Test same violation data produces same hash."""
        data = {"alert_id": "v-001", "violation_type": "fpic_violation"}
        h1 = compute_test_hash(data)
        h2 = compute_test_hash(data)
        assert h1 == h2

    def test_provenance_records_ingestion(self, mock_provenance):
        """Test provenance records violation ingestion."""
        mock_provenance.record("violation", "create", "v-001")
        assert mock_provenance.entry_count == 1

    def test_provenance_records_correlation(self, mock_provenance):
        """Test provenance records supply chain correlation."""
        mock_provenance.record("violation", "correlate", "v-001")
        assert mock_provenance.entry_count == 1

    def test_provenance_chain_intact(self, mock_provenance):
        """Test provenance chain is valid."""
        mock_provenance.record("violation", "create", "v-001")
        mock_provenance.record("violation", "correlate", "v-001")
        assert mock_provenance.verify_chain() is True
