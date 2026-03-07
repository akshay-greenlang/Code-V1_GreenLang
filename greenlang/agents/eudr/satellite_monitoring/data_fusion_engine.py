# -*- coding: utf-8 -*-
"""
DataFusionEngine - AGENT-EUDR-003 Feature 5: Multi-Source Satellite Data Fusion

Combines analysis results from multiple satellite sources (Sentinel-2, Landsat,
Global Forest Watch, SAR) into a unified, high-confidence deforestation
assessment. Implements weighted consensus scoring with configurable source
weights, agreement metrics, compliance determination, NDVI series fusion,
and data quality assessment.

Source Weight Model (Default):
    Sentinel-2:  0.50 (10m resolution, 5-day revisit, primary optical)
    Landsat:     0.30 (30m resolution, 16-day revisit, long archive)
    GFW:         0.20 (30m tree cover, annual updates, Hansen et al.)

When sources are missing, weights are re-normalized across available sources
to maintain a valid probability distribution.

Decision Thresholds:
    Deforestation Score > 0.5:          DEFORESTATION_DETECTED
    Deforestation Score 0.2 - 0.5:      POTENTIAL_DEFORESTATION
    Deforestation Score < 0.2:          NO_DEFORESTATION

Zero-Hallucination Guarantees:
    - All fusion calculations are deterministic weighted arithmetic.
    - No ML/LLM involvement in scoring, consensus, or compliance decisions.
    - SHA-256 provenance hashes on all fusion results.
    - Agreement score is computed via standard deviation of binary indicators.

Performance Targets:
    - Single fusion (4 sources): <5ms
    - NDVI series fusion (100 dates x 4 sources): <50ms

Regulatory References:
    - EUDR Article 2(1): Deforestation-free definition.
    - EUDR Article 9: Geolocation requirements.
    - EUDR Article 10: Risk assessment via satellite monitoring.
    - EUDR Cutoff Date: 31 December 2020.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003, Feature 5
Agent ID: GL-EUDR-SAT-003
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other serializable).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Module version for provenance tracking.
_MODULE_VERSION: str = "1.0.0"

#: Default source weights for fusion consensus.
DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = {
    "sentinel2": 0.50,
    "landsat": 0.30,
    "gfw": 0.20,
}

#: Extended source weights including SAR.
EXTENDED_SOURCE_WEIGHTS: Dict[str, float] = {
    "sentinel2": 0.40,
    "landsat": 0.25,
    "gfw": 0.15,
    "sar": 0.20,
}

#: Deforestation score thresholds for decision-making.
DEFORESTATION_DETECTED_THRESHOLD: float = 0.5
POTENTIAL_DEFORESTATION_THRESHOLD: float = 0.2

#: Compliance confidence thresholds.
COMPLIANCE_HIGH_CONFIDENCE: float = 0.7
COMPLIANCE_LOW_CONFIDENCE: float = 0.5

#: Minimum number of sources for reliable fusion.
MIN_SOURCES_FOR_FUSION: int = 2

#: Data quality tier thresholds.
QUALITY_EXCELLENT_THRESHOLD: float = 0.85
QUALITY_GOOD_THRESHOLD: float = 0.70
QUALITY_FAIR_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SourceResult:
    """Analysis result from a single satellite source.

    Attributes:
        source_name: Source identifier (sentinel2, landsat, gfw, sar).
        deforestation_detected: Whether deforestation was detected by this source.
        deforestation_score: Deforestation confidence score (0.0-1.0).
        affected_area_ha: Estimated affected area in hectares.
        confidence: Overall analysis confidence (0.0-1.0).
        ndvi_baseline: Baseline NDVI value (if available).
        ndvi_latest: Latest NDVI value (if available).
        ndvi_change: NDVI change from baseline (if available).
        cloud_cover_pct: Cloud cover percentage for this observation.
        observation_date: Date of the satellite observation.
        data_quality: Data quality indicator (0.0-1.0).
        metadata: Additional source-specific metadata.
    """

    source_name: str = ""
    deforestation_detected: bool = False
    deforestation_score: float = 0.0
    affected_area_ha: float = 0.0
    confidence: float = 0.0
    ndvi_baseline: Optional[float] = None
    ndvi_latest: Optional[float] = None
    ndvi_change: Optional[float] = None
    cloud_cover_pct: float = 0.0
    observation_date: Optional[str] = None
    data_quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "source_name": self.source_name,
            "deforestation_detected": self.deforestation_detected,
            "deforestation_score": round(self.deforestation_score, 4),
            "affected_area_ha": round(self.affected_area_ha, 4),
            "confidence": round(self.confidence, 4),
            "ndvi_baseline": self.ndvi_baseline,
            "ndvi_latest": self.ndvi_latest,
            "ndvi_change": self.ndvi_change,
            "cloud_cover_pct": round(self.cloud_cover_pct, 2),
            "observation_date": self.observation_date,
            "data_quality": round(self.data_quality, 4),
            "metadata": self.metadata,
        }


@dataclass
class FusionResult:
    """Result of multi-source data fusion.

    Attributes:
        fusion_id: Unique fusion result identifier.
        fused_at: UTC timestamp of fusion computation.
        sources_used: List of source names that contributed to this fusion.
        source_count: Number of sources used.
        weighted_deforestation_score: Weighted consensus deforestation score (0.0-1.0).
        weighted_affected_area_ha: Weighted consensus affected area in hectares.
        weighted_confidence: Weighted consensus confidence (0.0-1.0).
        agreement_score: Inter-source agreement (0.0-1.0, 1.0 = full agreement).
        decision: Fusion decision string.
        source_results: Per-source analysis results.
        weights_applied: Weights applied to each source.
        provenance_hash: SHA-256 hash for tamper detection.
        processing_time_ms: Fusion computation time in milliseconds.
    """

    fusion_id: str = field(default_factory=lambda: _generate_id("FUS"))
    fused_at: datetime = field(default_factory=_utcnow)
    sources_used: List[str] = field(default_factory=list)
    source_count: int = 0
    weighted_deforestation_score: float = 0.0
    weighted_affected_area_ha: float = 0.0
    weighted_confidence: float = 0.0
    agreement_score: float = 0.0
    decision: str = "NO_DEFORESTATION"
    source_results: List[SourceResult] = field(default_factory=list)
    weights_applied: Dict[str, float] = field(default_factory=dict)
    provenance_hash: str = ""
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "fusion_id": self.fusion_id,
            "fused_at": self.fused_at.isoformat(),
            "sources_used": self.sources_used,
            "source_count": self.source_count,
            "weighted_deforestation_score": round(
                self.weighted_deforestation_score, 4
            ),
            "weighted_affected_area_ha": round(
                self.weighted_affected_area_ha, 4
            ),
            "weighted_confidence": round(self.weighted_confidence, 4),
            "agreement_score": round(self.agreement_score, 4),
            "decision": self.decision,
            "source_results": [s.to_dict() for s in self.source_results],
            "weights_applied": {
                k: round(v, 4) for k, v in self.weights_applied.items()
            },
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class DataQualityAssessment:
    """Assessment of fused data quality.

    Attributes:
        assessment_id: Unique assessment identifier.
        assessed_at: UTC timestamp of assessment.
        overall_quality: Overall quality score (0.0-1.0).
        quality_tier: Quality tier label (excellent, good, fair, poor).
        source_count: Number of contributing sources.
        temporal_coverage: Temporal coverage score (0.0-1.0).
        spatial_coverage: Spatial coverage score (0.0-1.0).
        confidence_score: Average source confidence.
        cloud_impact: Cloud cover impact factor (0.0-1.0, 0.0 = no impact).
        agreement_factor: Source agreement factor (0.0-1.0).
        recommendations: List of quality improvement recommendations.
        provenance_hash: SHA-256 hash for tamper detection.
    """

    assessment_id: str = field(default_factory=lambda: _generate_id("DQA"))
    assessed_at: datetime = field(default_factory=_utcnow)
    overall_quality: float = 0.0
    quality_tier: str = "poor"
    source_count: int = 0
    temporal_coverage: float = 0.0
    spatial_coverage: float = 0.0
    confidence_score: float = 0.0
    cloud_impact: float = 0.0
    agreement_factor: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "assessment_id": self.assessment_id,
            "assessed_at": self.assessed_at.isoformat(),
            "overall_quality": round(self.overall_quality, 4),
            "quality_tier": self.quality_tier,
            "source_count": self.source_count,
            "temporal_coverage": round(self.temporal_coverage, 4),
            "spatial_coverage": round(self.spatial_coverage, 4),
            "confidence_score": round(self.confidence_score, 4),
            "cloud_impact": round(self.cloud_impact, 4),
            "agreement_factor": round(self.agreement_factor, 4),
            "recommendations": self.recommendations,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# DataFusionEngine
# ---------------------------------------------------------------------------


class DataFusionEngine:
    """Multi-source satellite data fusion engine for EUDR compliance.

    Combines analysis results from multiple satellite sources into a unified,
    high-confidence deforestation assessment using weighted consensus scoring.
    Supports configurable source weights, agreement metrics, compliance
    determination, NDVI series fusion, and data quality assessment.

    All fusion calculations are deterministic with zero LLM/ML involvement.

    Attributes:
        _source_weights: Configured source weight map.
        _fusion_store: In-memory store of fusion results by fusion_id.

    Example::

        engine = DataFusionEngine()

        s2 = SourceResult(source_name="sentinel2", deforestation_score=0.8,
                          confidence=0.9, affected_area_ha=12.5,
                          deforestation_detected=True)
        ls = SourceResult(source_name="landsat", deforestation_score=0.7,
                          confidence=0.8, affected_area_ha=11.0,
                          deforestation_detected=True)
        gfw = SourceResult(source_name="gfw", deforestation_score=0.6,
                           confidence=0.7, affected_area_ha=10.0,
                           deforestation_detected=True)

        result = engine.fuse_sources(
            sentinel2_result=s2, landsat_result=ls, gfw_result=gfw
        )
        assert result.decision == "DEFORESTATION_DETECTED"
        assert result.provenance_hash != ""
    """

    def __init__(
        self,
        source_weights: Optional[Dict[str, float]] = None,
        config: Any = None,
    ) -> None:
        """Initialize the DataFusionEngine.

        Args:
            source_weights: Optional custom source weight mapping. Keys are
                source names (sentinel2, landsat, gfw, sar), values are
                weights that must sum to 1.0 before re-normalization.
                If None, DEFAULT_SOURCE_WEIGHTS is used.
            config: Optional configuration object. Reserved for future use.
        """
        self._source_weights: Dict[str, float] = (
            dict(source_weights) if source_weights is not None
            else dict(DEFAULT_SOURCE_WEIGHTS)
        )
        self._fusion_store: Dict[str, FusionResult] = {}

        logger.info(
            "DataFusionEngine initialized: weights=%s",
            {k: round(v, 3) for k, v in self._source_weights.items()},
        )

    # ------------------------------------------------------------------
    # Public API: Multi-Source Fusion
    # ------------------------------------------------------------------

    def fuse_sources(
        self,
        sentinel2_result: Optional[SourceResult] = None,
        landsat_result: Optional[SourceResult] = None,
        gfw_result: Optional[SourceResult] = None,
        sar_result: Optional[SourceResult] = None,
    ) -> FusionResult:
        """Fuse results from multiple satellite sources into a consensus.

        Applies configurable weighted scoring. When sources are missing,
        weights are re-normalized across available sources. The fusion
        produces a weighted deforestation score, weighted area, weighted
        confidence, agreement score, and a decision classification.

        Args:
            sentinel2_result: Sentinel-2 analysis result (or None).
            landsat_result: Landsat analysis result (or None).
            gfw_result: Global Forest Watch analysis result (or None).
            sar_result: SAR (Sentinel-1) analysis result (or None).

        Returns:
            FusionResult with consensus scoring and provenance hash.

        Raises:
            ValueError: If no source results are provided.
        """
        start_time = time.monotonic()

        # Collect available sources
        source_map: Dict[str, SourceResult] = {}
        if sentinel2_result is not None:
            source_map["sentinel2"] = sentinel2_result
        if landsat_result is not None:
            source_map["landsat"] = landsat_result
        if gfw_result is not None:
            source_map["gfw"] = gfw_result
        if sar_result is not None:
            source_map["sar"] = sar_result

        if not source_map:
            raise ValueError(
                "At least one source result must be provided for fusion"
            )

        # Re-normalize weights for available sources
        active_weights = self._renormalize_weights(list(source_map.keys()))

        # Calculate weighted consensus metrics
        weighted_score = self._calculate_weighted_metric(
            source_map, active_weights, "deforestation_score"
        )
        weighted_area = self._calculate_weighted_metric(
            source_map, active_weights, "affected_area_ha"
        )
        weighted_confidence = self._calculate_weighted_metric(
            source_map, active_weights, "confidence"
        )

        # Calculate agreement score
        agreement = self.calculate_agreement(list(source_map.values()))

        # Determine decision
        decision = self._classify_decision(weighted_score)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        result = FusionResult(
            sources_used=list(source_map.keys()),
            source_count=len(source_map),
            weighted_deforestation_score=round(weighted_score, 4),
            weighted_affected_area_ha=round(weighted_area, 4),
            weighted_confidence=round(weighted_confidence, 4),
            agreement_score=round(agreement, 4),
            decision=decision,
            source_results=list(source_map.values()),
            weights_applied=active_weights,
            processing_time_ms=round(elapsed_ms, 2),
        )

        # Compute provenance hash
        result.provenance_hash = _compute_hash(result)

        # Store result
        self._fusion_store[result.fusion_id] = result

        logger.info(
            "Fusion %s completed: sources=%d (%s), score=%.3f, "
            "area=%.2f ha, confidence=%.3f, agreement=%.3f, "
            "decision=%s, elapsed=%.2fms",
            result.fusion_id,
            result.source_count,
            ", ".join(result.sources_used),
            weighted_score,
            weighted_area,
            weighted_confidence,
            agreement,
            decision,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Agreement Calculation
    # ------------------------------------------------------------------

    def calculate_agreement(
        self,
        results: List[SourceResult],
    ) -> float:
        """Measure agreement between multiple satellite sources.

        Agreement is computed as 1 minus the standard deviation of
        per-source binary deforestation indicators (0 or 1). Full
        agreement (all sources agree) yields 1.0. Maximum disagreement
        (half detect, half do not) yields approximately 0.5.

        For a single source, agreement is defined as 1.0 (trivial).

        Args:
            results: List of SourceResult from different sources.

        Returns:
            Agreement score between 0.0 and 1.0.
        """
        if not results:
            return 0.0

        if len(results) == 1:
            return 1.0

        # Binary indicators: 1.0 if deforestation detected, 0.0 otherwise
        binary_indicators = [
            1.0 if r.deforestation_detected else 0.0
            for r in results
        ]

        n = len(binary_indicators)
        mean = sum(binary_indicators) / n
        variance = sum((x - mean) ** 2 for x in binary_indicators) / n
        std_dev = math.sqrt(variance)

        # Agreement = 1 - std_dev (std_dev of binary is max 0.5)
        agreement = max(0.0, min(1.0, 1.0 - std_dev))
        return round(agreement, 4)

    # ------------------------------------------------------------------
    # Public API: Compliance Determination
    # ------------------------------------------------------------------

    def determine_compliance(
        self,
        fusion_result: FusionResult,
    ) -> str:
        """Determine EUDR compliance status from a fusion result.

        Decision rules (deterministic):
            COMPLIANT:             No deforestation AND confidence > 0.7
            NON_COMPLIANT:         Deforestation detected AND confidence > 0.7
            INSUFFICIENT_DATA:     Confidence < 0.5
            MANUAL_REVIEW_REQUIRED: All other cases (ambiguous results)

        Args:
            fusion_result: FusionResult from fuse_sources().

        Returns:
            Compliance status string: COMPLIANT, NON_COMPLIANT,
            INSUFFICIENT_DATA, or MANUAL_REVIEW_REQUIRED.
        """
        score = fusion_result.weighted_deforestation_score
        confidence = fusion_result.weighted_confidence

        if confidence < COMPLIANCE_LOW_CONFIDENCE:
            status = "INSUFFICIENT_DATA"
        elif (
            score >= DEFORESTATION_DETECTED_THRESHOLD
            and confidence >= COMPLIANCE_HIGH_CONFIDENCE
        ):
            status = "NON_COMPLIANT"
        elif (
            score < POTENTIAL_DEFORESTATION_THRESHOLD
            and confidence >= COMPLIANCE_HIGH_CONFIDENCE
        ):
            status = "COMPLIANT"
        else:
            status = "MANUAL_REVIEW_REQUIRED"

        logger.info(
            "Compliance determination for fusion %s: score=%.3f, "
            "confidence=%.3f -> %s",
            fusion_result.fusion_id,
            score,
            confidence,
            status,
        )

        return status

    # ------------------------------------------------------------------
    # Public API: NDVI Series Fusion
    # ------------------------------------------------------------------

    def fuse_ndvi_series(
        self,
        series_by_source: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Merge NDVI time series from multiple sources into a unified series.

        For each unique date across all sources, computes a weighted average
        NDVI value using the configured source weights. When only a subset
        of sources has data for a given date, weights are re-normalized.

        Each entry in a source series must be a dict with at least:
            - ``date``: ISO 8601 date string (YYYY-MM-DD).
            - ``ndvi``: NDVI value (float).

        Args:
            series_by_source: Mapping from source name to list of
                ``{date, ndvi, ...}`` observations.

        Returns:
            Unified NDVI time series sorted chronologically, each entry
            containing: date, ndvi, source_count, sources, provenance_hash.
        """
        start_time = time.monotonic()

        if not series_by_source:
            return []

        # Collect all observations grouped by date
        date_source_map: Dict[str, Dict[str, float]] = {}
        for source_name, series in series_by_source.items():
            for obs in series:
                date_str = str(obs.get("date", ""))
                ndvi_val = float(obs.get("ndvi", 0.0))
                if date_str:
                    date_source_map.setdefault(date_str, {})[source_name] = (
                        ndvi_val
                    )

        # Fuse per date
        fused_series: List[Dict[str, Any]] = []
        for date_str in sorted(date_source_map.keys()):
            sources_for_date = date_source_map[date_str]
            active_weights = self._renormalize_weights(
                list(sources_for_date.keys())
            )

            # Weighted average NDVI
            weighted_ndvi = sum(
                sources_for_date[src] * active_weights.get(src, 0.0)
                for src in sources_for_date
            )

            entry = {
                "date": date_str,
                "ndvi": round(weighted_ndvi, 4),
                "source_count": len(sources_for_date),
                "sources": list(sources_for_date.keys()),
                "provenance_hash": _compute_hash({
                    "date": date_str,
                    "ndvi": round(weighted_ndvi, 4),
                    "sources": sorted(sources_for_date.keys()),
                }),
            }
            fused_series.append(entry)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "NDVI series fusion completed: %d dates from %d sources, "
            "elapsed=%.2fms",
            len(fused_series),
            len(series_by_source),
            elapsed_ms,
        )

        return fused_series

    # ------------------------------------------------------------------
    # Public API: Data Quality Assessment
    # ------------------------------------------------------------------

    def assess_fusion_quality(
        self,
        fusion_result: FusionResult,
    ) -> DataQualityAssessment:
        """Assess overall data quality of a fused analysis result.

        Quality is computed from five deterministic factors:
            1. Source count factor (more sources = higher quality).
            2. Temporal coverage (observation recency and overlap).
            3. Spatial coverage (effective resolution proxy).
            4. Confidence score (average source confidence).
            5. Cloud impact (average cloud cover penalty).

        Quality Tiers:
            excellent: >= 0.85
            good:      >= 0.70
            fair:      >= 0.50
            poor:      <  0.50

        Args:
            fusion_result: FusionResult to assess.

        Returns:
            DataQualityAssessment with overall quality and recommendations.
        """
        start_time = time.monotonic()

        source_count = fusion_result.source_count
        sources = fusion_result.source_results

        # Factor 1: Source count (2 sources = 0.5, 3 = 0.75, 4 = 1.0)
        source_factor = min(1.0, source_count / 4.0)

        # Factor 2: Temporal coverage
        temporal_coverage = self._assess_temporal_coverage(sources)

        # Factor 3: Spatial coverage (resolution proxy based on source mix)
        spatial_coverage = self._assess_spatial_coverage(
            fusion_result.sources_used
        )

        # Factor 4: Confidence score (average)
        confidence_score = self._calculate_mean_confidence(sources)

        # Factor 5: Cloud impact
        cloud_impact = self._calculate_cloud_impact(sources)

        # Agreement factor
        agreement_factor = fusion_result.agreement_score

        # Overall quality: weighted combination of factors
        overall_quality = (
            source_factor * 0.20
            + temporal_coverage * 0.15
            + spatial_coverage * 0.15
            + confidence_score * 0.20
            + (1.0 - cloud_impact) * 0.15
            + agreement_factor * 0.15
        )
        overall_quality = max(0.0, min(1.0, round(overall_quality, 4)))

        # Determine quality tier
        quality_tier = self._classify_quality_tier(overall_quality)

        # Generate recommendations
        recommendations = self._generate_quality_recommendations(
            source_count=source_count,
            temporal_coverage=temporal_coverage,
            spatial_coverage=spatial_coverage,
            confidence_score=confidence_score,
            cloud_impact=cloud_impact,
            agreement_factor=agreement_factor,
        )

        assessment = DataQualityAssessment(
            overall_quality=overall_quality,
            quality_tier=quality_tier,
            source_count=source_count,
            temporal_coverage=round(temporal_coverage, 4),
            spatial_coverage=round(spatial_coverage, 4),
            confidence_score=round(confidence_score, 4),
            cloud_impact=round(cloud_impact, 4),
            agreement_factor=round(agreement_factor, 4),
            recommendations=recommendations,
        )

        assessment.provenance_hash = _compute_hash(assessment)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Quality assessment %s: overall=%.3f (%s), sources=%d, "
            "cloud_impact=%.3f, agreement=%.3f, elapsed=%.2fms",
            assessment.assessment_id,
            overall_quality,
            quality_tier,
            source_count,
            cloud_impact,
            agreement_factor,
            elapsed_ms,
        )

        return assessment

    # ------------------------------------------------------------------
    # Public API: Fusion Result Retrieval
    # ------------------------------------------------------------------

    def get_fusion_result(self, fusion_id: str) -> Optional[FusionResult]:
        """Retrieve a stored fusion result by ID.

        Args:
            fusion_id: Fusion result identifier.

        Returns:
            FusionResult if found, else None.
        """
        return self._fusion_store.get(fusion_id)

    def get_all_fusion_results(self) -> List[FusionResult]:
        """Retrieve all stored fusion results.

        Returns:
            List of all FusionResult objects in storage.
        """
        return list(self._fusion_store.values())

    # ------------------------------------------------------------------
    # Internal: Weight Re-normalization
    # ------------------------------------------------------------------

    def _renormalize_weights(
        self,
        available_sources: List[str],
    ) -> Dict[str, float]:
        """Re-normalize source weights for available sources only.

        When some sources are missing, the weights of available sources
        are scaled proportionally so they sum to 1.0.

        Args:
            available_sources: List of source names that have data.

        Returns:
            Dict mapping source name to re-normalized weight.
        """
        raw_weights = {
            src: self._source_weights.get(src, 0.0)
            for src in available_sources
            if self._source_weights.get(src, 0.0) > 0.0
        }

        total_weight = sum(raw_weights.values())
        if total_weight <= 0.0:
            # Equal weighting fallback
            n = len(available_sources)
            if n == 0:
                return {}
            equal_weight = 1.0 / n
            return {src: round(equal_weight, 6) for src in available_sources}

        return {
            src: round(w / total_weight, 6)
            for src, w in raw_weights.items()
        }

    # ------------------------------------------------------------------
    # Internal: Weighted Metric Calculation
    # ------------------------------------------------------------------

    def _calculate_weighted_metric(
        self,
        source_map: Dict[str, SourceResult],
        weights: Dict[str, float],
        metric_name: str,
    ) -> float:
        """Calculate a weighted average of a named metric across sources.

        Args:
            source_map: Mapping from source name to SourceResult.
            weights: Re-normalized weights per source.
            metric_name: Attribute name on SourceResult to aggregate.

        Returns:
            Weighted average value.
        """
        total = 0.0
        for source_name, result in source_map.items():
            weight = weights.get(source_name, 0.0)
            value = getattr(result, metric_name, 0.0)
            if value is None:
                value = 0.0
            total += float(value) * weight
        return total

    # ------------------------------------------------------------------
    # Internal: Decision Classification
    # ------------------------------------------------------------------

    def _classify_decision(self, score: float) -> str:
        """Classify the fusion decision based on deforestation score.

        Decision rules:
            score > 0.5:     DEFORESTATION_DETECTED
            0.2 <= score <= 0.5: POTENTIAL_DEFORESTATION
            score < 0.2:     NO_DEFORESTATION

        Args:
            score: Weighted deforestation score (0.0-1.0).

        Returns:
            Decision string.
        """
        if score > DEFORESTATION_DETECTED_THRESHOLD:
            return "DEFORESTATION_DETECTED"
        elif score >= POTENTIAL_DEFORESTATION_THRESHOLD:
            return "POTENTIAL_DEFORESTATION"
        else:
            return "NO_DEFORESTATION"

    # ------------------------------------------------------------------
    # Internal: Quality Assessment Helpers
    # ------------------------------------------------------------------

    def _assess_temporal_coverage(
        self,
        sources: List[SourceResult],
    ) -> float:
        """Assess temporal coverage from source observation dates.

        Sources with recent observations score higher. Sources without
        observation dates are penalized.

        Args:
            sources: List of SourceResult entries.

        Returns:
            Temporal coverage score (0.0-1.0).
        """
        if not sources:
            return 0.0

        has_date_count = sum(
            1 for s in sources if s.observation_date is not None
        )

        # Base score: proportion of sources with observation dates
        date_coverage = has_date_count / len(sources) if sources else 0.0

        # Recency bonus: check if any source has data from the last 30 days
        recency_bonus = 0.0
        now = _utcnow()
        for source in sources:
            if source.observation_date is not None:
                try:
                    obs_date = datetime.fromisoformat(
                        source.observation_date.replace("Z", "+00:00")
                    )
                    if hasattr(obs_date, "tzinfo") and obs_date.tzinfo is None:
                        obs_date = obs_date.replace(tzinfo=timezone.utc)
                    days_ago = (now - obs_date).days
                    if days_ago <= 30:
                        recency_bonus = max(recency_bonus, 0.2)
                    elif days_ago <= 90:
                        recency_bonus = max(recency_bonus, 0.1)
                except (ValueError, TypeError):
                    pass

        return min(1.0, date_coverage + recency_bonus)

    def _assess_spatial_coverage(
        self,
        source_names: List[str],
    ) -> float:
        """Assess spatial coverage based on the source mix.

        Higher-resolution sources (Sentinel-2: 10m) score better than
        lower-resolution sources (Landsat: 30m). SAR provides all-weather
        capability which adds a bonus.

        Args:
            source_names: List of source name strings.

        Returns:
            Spatial coverage score (0.0-1.0).
        """
        if not source_names:
            return 0.0

        # Source resolution scoring
        resolution_scores: Dict[str, float] = {
            "sentinel2": 1.0,   # 10m resolution
            "sar": 0.85,        # 10m, all-weather
            "landsat": 0.7,     # 30m resolution
            "gfw": 0.6,         # 30m, derived product
        }

        total_score = sum(
            resolution_scores.get(src, 0.5) for src in source_names
        )
        return min(1.0, total_score / max(len(source_names), 1))

    def _calculate_mean_confidence(
        self,
        sources: List[SourceResult],
    ) -> float:
        """Calculate mean confidence across all sources.

        Args:
            sources: List of SourceResult entries.

        Returns:
            Mean confidence (0.0-1.0).
        """
        if not sources:
            return 0.0
        total = sum(s.confidence for s in sources)
        return total / len(sources)

    def _calculate_cloud_impact(
        self,
        sources: List[SourceResult],
    ) -> float:
        """Calculate cloud cover impact factor.

        Higher cloud cover reduces data quality. The impact is the
        weighted average cloud cover percentage across sources,
        normalized to 0.0-1.0.

        Args:
            sources: List of SourceResult entries.

        Returns:
            Cloud impact factor (0.0-1.0, where 0.0 = no cloud impact).
        """
        if not sources:
            return 1.0

        total_cloud = sum(s.cloud_cover_pct for s in sources)
        avg_cloud = total_cloud / len(sources)
        return min(1.0, avg_cloud / 100.0)

    def _classify_quality_tier(self, quality: float) -> str:
        """Classify quality score into a named tier.

        Args:
            quality: Overall quality score (0.0-1.0).

        Returns:
            Quality tier string: excellent, good, fair, or poor.
        """
        if quality >= QUALITY_EXCELLENT_THRESHOLD:
            return "excellent"
        elif quality >= QUALITY_GOOD_THRESHOLD:
            return "good"
        elif quality >= QUALITY_FAIR_THRESHOLD:
            return "fair"
        else:
            return "poor"

    def _generate_quality_recommendations(
        self,
        source_count: int,
        temporal_coverage: float,
        spatial_coverage: float,
        confidence_score: float,
        cloud_impact: float,
        agreement_factor: float,
    ) -> List[str]:
        """Generate actionable quality improvement recommendations.

        Args:
            source_count: Number of satellite sources used.
            temporal_coverage: Temporal coverage score.
            spatial_coverage: Spatial coverage score.
            confidence_score: Average confidence score.
            cloud_impact: Cloud impact factor.
            agreement_factor: Source agreement factor.

        Returns:
            List of human-readable recommendation strings.
        """
        recommendations: List[str] = []

        if source_count < MIN_SOURCES_FOR_FUSION:
            recommendations.append(
                f"Add at least {MIN_SOURCES_FOR_FUSION - source_count} more "
                f"satellite source(s) for reliable multi-source fusion."
            )

        if source_count < 3:
            recommendations.append(
                "Consider adding SAR (Sentinel-1) data for all-weather "
                "monitoring capability."
            )

        if temporal_coverage < 0.5:
            recommendations.append(
                "Improve temporal coverage by acquiring more recent "
                "satellite imagery (within the last 30 days)."
            )

        if spatial_coverage < 0.7:
            recommendations.append(
                "Improve spatial coverage by including higher-resolution "
                "sources (e.g., Sentinel-2 at 10m)."
            )

        if confidence_score < 0.6:
            recommendations.append(
                "Low average confidence detected. Consider re-running "
                "analysis with clearer imagery or additional sources."
            )

        if cloud_impact > 0.3:
            recommendations.append(
                "Significant cloud cover impact detected. Use SAR fusion "
                "or temporal compositing to fill cloudy gaps."
            )

        if agreement_factor < 0.7:
            recommendations.append(
                "Low inter-source agreement. Manual review recommended "
                "to resolve conflicting deforestation assessments."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def source_weights(self) -> Dict[str, float]:
        """Return the current source weight configuration."""
        return dict(self._source_weights)

    @property
    def fusion_count(self) -> int:
        """Return the number of stored fusion results."""
        return len(self._fusion_store)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Engine
    "DataFusionEngine",
    # Data classes
    "SourceResult",
    "FusionResult",
    "DataQualityAssessment",
    # Constants
    "DEFAULT_SOURCE_WEIGHTS",
    "EXTENDED_SOURCE_WEIGHTS",
    "DEFORESTATION_DETECTED_THRESHOLD",
    "POTENTIAL_DEFORESTATION_THRESHOLD",
    "COMPLIANCE_HIGH_CONFIDENCE",
    "COMPLIANCE_LOW_CONFIDENCE",
    "MIN_SOURCES_FOR_FUSION",
]
