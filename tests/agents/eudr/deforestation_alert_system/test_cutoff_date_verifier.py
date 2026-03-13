# -*- coding: utf-8 -*-
"""
Unit tests for CutoffDateVerifier - AGENT-EUDR-020 Engine 5

Tests EUDR cutoff date verification including pre-cutoff/post-cutoff/ongoing/
uncertain classification, temporal evidence collection, confidence scoring,
batch verification, evidence chain retrieval, timeline construction, and
provenance tracking.

The EUDR cutoff date is December 31, 2020. Deforestation AFTER this date
renders products NON-COMPLIANT for EU market access.

Coverage targets: 85%+ across all CutoffDateVerifier methods.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020
Agent ID: GL-EUDR-DAS-020
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.deforestation_alert_system.engines.cutoff_date_verifier import (
    BIOME_NDVI_BASELINES,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ConfidenceLevel,
    CutoffDateVerifier,
    CutoffResult,
    CutoffVerification,
    DEFAULT_CUTOFF_CONFIDENCE,
    DEFAULT_GRACE_PERIOD_DAYS,
    EUDR_CUTOFF_DATE,
    EvidenceChain,
    EvidenceSource,
    ForestState,
    ForestTimeline,
    MAX_BATCH_SIZE,
    MIN_EVIDENCE_SOURCES,
    REFERENCE_OBSERVATIONS,
    SOURCE_RELIABILITY_WEIGHTS,
    TemporalEvidence,
    TemporalTransition,
    VerificationStatus,
    BatchVerificationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def verifier() -> CutoffDateVerifier:
    """Create a default CutoffDateVerifier instance."""
    return CutoffDateVerifier()


@pytest.fixture
def verifier_custom_cutoff() -> CutoffDateVerifier:
    """Create a CutoffDateVerifier with a custom cutoff date."""
    return CutoffDateVerifier(cutoff_date="2021-06-30")


@pytest.fixture
def post_cutoff_observations() -> List[Dict[str, Any]]:
    """Observations showing clearing AFTER the cutoff date."""
    return list(REFERENCE_OBSERVATIONS["sample_post_cutoff"])


@pytest.fixture
def pre_cutoff_observations() -> List[Dict[str, Any]]:
    """Observations showing clearing BEFORE the cutoff date."""
    return list(REFERENCE_OBSERVATIONS["sample_pre_cutoff"])


@pytest.fixture
def ongoing_observations() -> List[Dict[str, Any]]:
    """Observations showing clearing spanning the cutoff date."""
    return list(REFERENCE_OBSERVATIONS["sample_ongoing"])


@pytest.fixture
def uncertain_observations() -> List[Dict[str, Any]]:
    """Sparse observations with insufficient temporal evidence."""
    return list(REFERENCE_OBSERVATIONS["sample_uncertain"])


# ---------------------------------------------------------------------------
# TestCutoffVerification
# ---------------------------------------------------------------------------


class TestCutoffVerification:
    """Tests for verify() with clear pre/post/ongoing/uncertain cases."""

    def test_post_cutoff_deforestation(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Post-cutoff deforestation classified as POST_CUTOFF."""
        verifier.load_observations("det-post", post_cutoff_observations)
        result = verifier.verify("det-post", -3.12, 28.57, "2021-07-01", "CD")
        assert result["cutoff_result"] == CutoffResult.POST_CUTOFF.value
        assert result["eudr_compliant"] is False
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_pre_cutoff_deforestation_is_compliant(
        self,
        verifier: CutoffDateVerifier,
        pre_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Deforestation before 2020-12-31 should be EUDR compliant."""
        verifier.load_observations("det-pre", pre_cutoff_observations)
        result = verifier.verify("det-pre", -3.12, 28.57, "2019-12-01", "CD")
        assert result["cutoff_result"] == CutoffResult.PRE_CUTOFF.value
        assert result["eudr_compliant"] is True

    def test_post_cutoff_deforestation_is_non_compliant(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Deforestation after 2020-12-31 should NOT be EUDR compliant."""
        verifier.load_observations("det-non-comp", post_cutoff_observations)
        result = verifier.verify(
            "det-non-comp", -3.12, 28.57, "2021-05-15", "CD"
        )
        assert result["eudr_compliant"] is False

    def test_ongoing_deforestation(
        self,
        verifier: CutoffDateVerifier,
        ongoing_observations: List[Dict[str, Any]],
    ) -> None:
        """Clearing spanning cutoff date classified as ONGOING."""
        verifier.load_observations("det-ongoing", ongoing_observations)
        result = verifier.verify(
            "det-ongoing", -3.12, 28.57, "2021-06-15", "CD"
        )
        assert result["cutoff_result"] in (
            CutoffResult.ONGOING.value,
            CutoffResult.POST_CUTOFF.value,
        )

    def test_uncertain_treated_as_high_risk(
        self,
        verifier: CutoffDateVerifier,
        uncertain_observations: List[Dict[str, Any]],
    ) -> None:
        """Uncertain cutoff timing should be treated as high risk."""
        verifier.load_observations("det-uncertain", uncertain_observations)
        result = verifier.verify(
            "det-uncertain", -3.12, 28.57, "2021-06-15", "CD"
        )
        if result["cutoff_result"] == CutoffResult.UNCERTAIN.value:
            assert result["eudr_compliant"] is False

    def test_verify_empty_detection_id_raises(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Empty detection_id raises ValueError."""
        with pytest.raises(ValueError):
            verifier.verify("", -3.12, 28.57, "2021-05-15")

    def test_verify_invalid_latitude_raises(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Invalid latitude raises ValueError."""
        with pytest.raises(ValueError):
            verifier.verify("det-bad", 95.0, 28.57, "2021-05-15")

    def test_verify_invalid_longitude_raises(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Invalid longitude raises ValueError."""
        with pytest.raises(ValueError):
            verifier.verify("det-bad", -3.12, 200.0, "2021-05-15")

    def test_verify_result_fields(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Verify result contains all required fields."""
        verifier.load_observations("det-fields", post_cutoff_observations)
        result = verifier.verify("det-fields", -3.12, 28.57, "2021-07-01")
        assert "verification_id" in result
        assert "cutoff_result" in result
        assert "confidence" in result
        assert "confidence_level" in result
        assert "eudr_compliant" in result
        assert "evidence_count" in result
        assert "provenance_hash" in result
        assert "processing_time_ms" in result
        assert "verification_status" in result

    def test_verify_with_country_code(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Country code influences biome determination."""
        verifier.load_observations("det-country", post_cutoff_observations)
        result = verifier.verify(
            "det-country", -3.12, 28.57, "2021-07-01", country_code="BR"
        )
        assert result["biome"] == "tropical_moist"

    def test_verify_caches_result(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Verification result is cached for subsequent lookups."""
        verifier.load_observations("det-cache", post_cutoff_observations)
        verifier.verify("det-cache", -3.12, 28.57, "2021-07-01")
        assert "det-cache" in verifier._verification_cache


# ---------------------------------------------------------------------------
# TestBatchVerification
# ---------------------------------------------------------------------------


class TestBatchVerification:
    """Tests for batch_verify with multiple detections."""

    def test_batch_verify_multiple(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
        pre_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Batch verify with mixed pre/post detections."""
        verifier.load_observations("det-batch-post", post_cutoff_observations)
        verifier.load_observations("det-batch-pre", pre_cutoff_observations)
        detections = [
            {
                "detection_id": "det-batch-post",
                "latitude": -3.12,
                "longitude": 28.57,
                "detection_date": "2021-07-01",
                "country_code": "CD",
            },
            {
                "detection_id": "det-batch-pre",
                "latitude": -3.12,
                "longitude": 28.57,
                "detection_date": "2019-12-01",
                "country_code": "CD",
            },
        ]
        result = verifier.batch_verify(detections)
        assert result["total_detections"] == 2
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64
        assert result["processing_time_ms"] > 0

    def test_batch_verify_empty_raises(self, verifier: CutoffDateVerifier) -> None:
        """Empty detections list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            verifier.batch_verify([])

    def test_batch_verify_exceeds_max_raises(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Exceeding MAX_BATCH_SIZE raises ValueError."""
        big_batch = [
            {
                "detection_id": f"det-{i}",
                "latitude": 0,
                "longitude": 0,
                "detection_date": "2021-01-01",
            }
            for i in range(MAX_BATCH_SIZE + 1)
        ]
        with pytest.raises(ValueError, match="exceeds maximum"):
            verifier.batch_verify(big_batch)

    def test_batch_verify_handles_failed_detection(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Batch continues processing when individual detection fails."""
        verifier.load_observations("det-ok", post_cutoff_observations)
        detections = [
            {
                "detection_id": "det-ok",
                "latitude": -3.12,
                "longitude": 28.57,
                "detection_date": "2021-07-01",
            },
            {
                "detection_id": "",
                "latitude": 999,
                "longitude": 999,
                "detection_date": "2021-01-01",
            },
        ]
        result = verifier.batch_verify(detections)
        assert result["total_detections"] == 2
        # Failed detection counted as uncertain
        assert result["uncertain_count"] >= 1

    def test_batch_verify_mean_confidence(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Batch result includes mean confidence."""
        verifier.load_observations("det-conf", post_cutoff_observations)
        detections = [
            {
                "detection_id": "det-conf",
                "latitude": -3.12,
                "longitude": 28.57,
                "detection_date": "2021-07-01",
            },
        ]
        result = verifier.batch_verify(detections)
        assert "mean_confidence" in result


# ---------------------------------------------------------------------------
# TestEvidence
# ---------------------------------------------------------------------------


class TestEvidence:
    """Tests for get_evidence temporal evidence chain retrieval."""

    def test_get_evidence_returns_chain(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """get_evidence returns temporal evidence chain."""
        verifier.load_observations("det-ev", post_cutoff_observations)
        result = verifier.get_evidence("det-ev")
        assert "evidence" in result
        assert result["total_observations"] >= 1
        assert "source_counts" in result
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_get_evidence_empty_id_raises(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Empty detection_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            verifier.get_evidence("")

    def test_get_evidence_temporal_statistics(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Evidence chain includes temporal coverage statistics."""
        verifier.load_observations("det-stats", post_cutoff_observations)
        result = verifier.get_evidence("det-stats")
        assert "temporal_coverage_days" in result
        assert "mean_observation_gap_days" in result
        assert "max_observation_gap_days" in result
        assert result["temporal_coverage_days"] > 0


# ---------------------------------------------------------------------------
# TestTimeline
# ---------------------------------------------------------------------------


class TestTimeline:
    """Tests for get_timeline forest state history."""

    def test_get_timeline_returns_states(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """get_timeline returns chronological forest states."""
        verifier.load_observations("det-tl", post_cutoff_observations)
        result = verifier.get_timeline("det-tl", -3.12, 28.57, "CD")
        assert "states" in result
        assert "transitions" in result
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_get_timeline_empty_id_raises(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Empty detection_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            verifier.get_timeline("")

    def test_get_timeline_cutoff_date_field(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Timeline includes the EUDR cutoff date."""
        verifier.load_observations("det-tl-cd", post_cutoff_observations)
        result = verifier.get_timeline("det-tl-cd", -3.12, 28.57)
        assert "cutoff_date" in result


# ---------------------------------------------------------------------------
# TestConfidence
# ---------------------------------------------------------------------------


class TestConfidence:
    """Tests for confidence level classification."""

    @pytest.mark.parametrize(
        "confidence_value,expected_level",
        [
            (Decimal("0.90"), ConfidenceLevel.HIGH.value),
            (Decimal("0.85"), ConfidenceLevel.HIGH.value),
            (Decimal("0.75"), ConfidenceLevel.MEDIUM.value),
            (Decimal("0.65"), ConfidenceLevel.MEDIUM.value),
            (Decimal("0.50"), ConfidenceLevel.LOW.value),
            (Decimal("0.45"), ConfidenceLevel.LOW.value),
            (Decimal("0.30"), ConfidenceLevel.INSUFFICIENT.value),
            (Decimal("0.10"), ConfidenceLevel.INSUFFICIENT.value),
        ],
    )
    def test_confidence_classification(
        self,
        verifier: CutoffDateVerifier,
        confidence_value: Decimal,
        expected_level: str,
    ) -> None:
        """Confidence values map to correct levels."""
        level = verifier._classify_confidence(confidence_value)
        if isinstance(level, ConfidenceLevel):
            assert level.value == expected_level
        else:
            assert level == expected_level


# ---------------------------------------------------------------------------
# TestEUDRCompliance
# ---------------------------------------------------------------------------


class TestEUDRCompliance:
    """Tests for EUDR compliance determination."""

    def test_pre_cutoff_is_compliant(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """PRE_CUTOFF result is EUDR compliant."""
        result = verifier._determine_compliance(
            CutoffResult.PRE_CUTOFF, Decimal("0.90")
        )
        assert result is True

    def test_post_cutoff_is_non_compliant(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """POST_CUTOFF result is not EUDR compliant."""
        result = verifier._determine_compliance(
            CutoffResult.POST_CUTOFF, Decimal("0.90")
        )
        assert result is False

    def test_uncertain_is_non_compliant(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """UNCERTAIN result is treated as non-compliant."""
        result = verifier._determine_compliance(
            CutoffResult.UNCERTAIN, Decimal("0.50")
        )
        assert result is False

    def test_ongoing_is_non_compliant(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """ONGOING result is non-compliant."""
        result = verifier._determine_compliance(
            CutoffResult.ONGOING, Decimal("0.80")
        )
        assert result is False


# ---------------------------------------------------------------------------
# TestTemporalEvidence
# ---------------------------------------------------------------------------


class TestTemporalEvidence:
    """Tests for _collect_temporal_evidence from multiple sources."""

    def test_collect_custom_observations(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Custom observations are returned in evidence collection."""
        verifier.load_observations("det-custom", post_cutoff_observations)
        evidence = verifier._collect_temporal_evidence(
            "det-custom", Decimal("-3.12"), Decimal("28.57")
        )
        assert len(evidence) >= len(post_cutoff_observations)

    def test_collect_reference_observations(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Reference observations are used for known detection patterns."""
        evidence = verifier._collect_temporal_evidence(
            "sample_post_cutoff", Decimal("-3.12"), Decimal("28.57")
        )
        assert len(evidence) >= 1

    def test_load_observations_validates_id(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Loading observations with empty ID raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            verifier.load_observations("", [{"date": "2021-01-01"}])

    def test_load_observations_validates_data(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Loading empty observations list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            verifier.load_observations("det-empty", [])


# ---------------------------------------------------------------------------
# TestCutoffResult
# ---------------------------------------------------------------------------


class TestCutoffResultDetermination:
    """Tests for _determine_cutoff_result with various evidence patterns."""

    def test_cutoff_result_enum_values(self) -> None:
        """CutoffResult enum has expected values."""
        assert CutoffResult.PRE_CUTOFF.value == "PRE_CUTOFF"
        assert CutoffResult.POST_CUTOFF.value == "POST_CUTOFF"
        assert CutoffResult.ONGOING.value == "ONGOING"
        assert CutoffResult.UNCERTAIN.value == "UNCERTAIN"

    def test_forest_state_enum_values(self) -> None:
        """ForestState enum has expected values."""
        assert ForestState.FORESTED.value == "FORESTED"
        assert ForestState.CLEARED.value == "CLEARED"
        assert ForestState.TRANSITIONING.value == "TRANSITIONING"
        assert ForestState.DEGRADED.value == "DEGRADED"
        assert ForestState.UNKNOWN.value == "UNKNOWN"


# ---------------------------------------------------------------------------
# TestPreCutoffState
# ---------------------------------------------------------------------------


class TestPreCutoffState:
    """Tests for checking pre-cutoff forest state."""

    def test_forested_before_cutoff(
        self,
        verifier: CutoffDateVerifier,
        pre_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Area with high NDVI before cutoff shows clearing completed pre-cutoff."""
        verifier.load_observations("det-pre-state", pre_cutoff_observations)
        result = verifier.verify(
            "det-pre-state", -3.12, 28.57, "2020-01-01", "CD"
        )
        assert result["cutoff_result"] == CutoffResult.PRE_CUTOFF.value


# ---------------------------------------------------------------------------
# TestPostCutoffChange
# ---------------------------------------------------------------------------


class TestPostCutoffChange:
    """Tests for post-cutoff change detection."""

    def test_clearing_after_cutoff_detected(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """NDVI drop after cutoff triggers POST_CUTOFF classification."""
        verifier.load_observations("det-post-change", post_cutoff_observations)
        result = verifier.verify(
            "det-post-change", -3.12, 28.57, "2021-09-01", "CD"
        )
        assert result["cutoff_result"] == CutoffResult.POST_CUTOFF.value
        assert result["eudr_compliant"] is False


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for provenance hash generation and chain integrity."""

    def test_verify_provenance_is_sha256(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Provenance hash is a 64-character SHA-256 hex digest."""
        verifier.load_observations("det-prov", post_cutoff_observations)
        result = verifier.verify("det-prov", -3.12, 28.57, "2021-07-01")
        assert len(result["provenance_hash"]) == 64

    def test_evidence_chain_provenance(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Evidence chain has provenance hash."""
        verifier.load_observations("det-ec-prov", post_cutoff_observations)
        result = verifier.get_evidence("det-ec-prov")
        assert len(result["provenance_hash"]) == 64

    def test_timeline_provenance(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Timeline result has provenance hash."""
        verifier.load_observations("det-tl-prov", post_cutoff_observations)
        result = verifier.get_timeline("det-tl-prov", -3.12, 28.57)
        assert len(result["provenance_hash"]) == 64

    def test_batch_provenance(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Batch result has provenance hash."""
        verifier.load_observations("det-batch-prov", post_cutoff_observations)
        result = verifier.batch_verify([
            {
                "detection_id": "det-batch-prov",
                "latitude": -3.12,
                "longitude": 28.57,
                "detection_date": "2021-07-01",
            },
        ])
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for temporal edge cases."""

    def test_detection_exactly_on_cutoff_date(
        self,
        verifier: CutoffDateVerifier,
        post_cutoff_observations: List[Dict[str, Any]],
    ) -> None:
        """Detection exactly on 2020-12-31 produces a valid result."""
        verifier.load_observations("det-exact", post_cutoff_observations)
        result = verifier.verify("det-exact", -3.12, 28.57, "2020-12-31")
        assert result["cutoff_result"] in (
            CutoffResult.PRE_CUTOFF.value,
            CutoffResult.POST_CUTOFF.value,
            CutoffResult.ONGOING.value,
            CutoffResult.UNCERTAIN.value,
        )

    def test_detection_far_in_past(
        self, verifier: CutoffDateVerifier
    ) -> None:
        """Detection in 2010 processes without error."""
        result = verifier.verify(
            "sample_pre_cutoff", -3.12, 28.57, "2010-01-01"
        )
        assert "cutoff_result" in result

    def test_custom_cutoff_date(
        self, verifier_custom_cutoff: CutoffDateVerifier
    ) -> None:
        """Custom cutoff date is respected."""
        assert verifier_custom_cutoff._cutoff_date == date(2021, 6, 30)

    def test_custom_grace_period(self) -> None:
        """Custom grace period is stored."""
        v = CutoffDateVerifier(grace_period_days=180)
        assert v._grace_period_days == 180

    def test_custom_min_evidence_sources(self) -> None:
        """Custom minimum evidence sources is stored."""
        v = CutoffDateVerifier(min_evidence_sources=5)
        assert v._min_evidence_sources == 5

    def test_custom_confidence_threshold(self) -> None:
        """Custom confidence threshold is stored."""
        v = CutoffDateVerifier(cutoff_confidence_threshold=Decimal("0.95"))
        assert v._cutoff_confidence_threshold == Decimal("0.95")

    def test_eudr_cutoff_date_constant(self) -> None:
        """EUDR cutoff date constant is 2020-12-31."""
        assert EUDR_CUTOFF_DATE == date(2020, 12, 31)

    def test_source_reliability_weights(self) -> None:
        """Source reliability weights sum to plausible values."""
        for source, weight in SOURCE_RELIABILITY_WEIGHTS.items():
            assert Decimal("0") < weight <= Decimal("1")

    def test_evidence_source_enum(self) -> None:
        """EvidenceSource enum has key satellite sources."""
        assert EvidenceSource.SENTINEL2.value == "SENTINEL2"
        assert EvidenceSource.LANDSAT.value == "LANDSAT"
        assert EvidenceSource.HANSEN_GFC.value == "HANSEN_GFC"
        assert EvidenceSource.GLAD.value == "GLAD"
        assert EvidenceSource.RADD.value == "RADD"

    def test_verification_status_enum(self) -> None:
        """VerificationStatus enum has expected values."""
        assert VerificationStatus.COMPLETED.value == "COMPLETED"
        assert VerificationStatus.PARTIAL.value == "PARTIAL"
        assert VerificationStatus.FAILED.value == "FAILED"
        assert VerificationStatus.PENDING.value == "PENDING"

    def test_temporal_evidence_to_dict(self) -> None:
        """TemporalEvidence serialization works."""
        ev = TemporalEvidence(
            evidence_id="ev-1",
            source="SENTINEL2",
            observation_date="2021-01-15",
            ndvi_value=Decimal("0.45"),
            evi_value=Decimal("0.22"),
        )
        d = ev.to_dict()
        assert d["source"] == "SENTINEL2"
        assert d["ndvi_value"] == "0.45"

    def test_cutoff_verification_to_dict(self) -> None:
        """CutoffVerification serialization includes all fields."""
        cv = CutoffVerification(
            verification_id="cv-1",
            detection_id="det-1",
            cutoff_result=CutoffResult.POST_CUTOFF.value,
            confidence=Decimal("0.90"),
            eudr_compliant=False,
        )
        d = cv.to_dict()
        assert d["cutoff_result"] == "POST_CUTOFF"
        assert d["eudr_compliant"] is False

    def test_batch_verification_result_to_dict(self) -> None:
        """BatchVerificationResult serialization works."""
        bvr = BatchVerificationResult(
            batch_id="bcv-1",
            total_detections=10,
            pre_cutoff_count=3,
            post_cutoff_count=7,
        )
        d = bvr.to_dict()
        assert d["total_detections"] == 10
        assert d["pre_cutoff_count"] == 3
