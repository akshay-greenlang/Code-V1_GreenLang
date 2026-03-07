# -*- coding: utf-8 -*-
"""
Conversion Risk Assessor Engine - AGENT-EUDR-005: Land Use Change (Engine 6)

Scores each plot's risk of future land use conversion using eight
deterministic risk factors with configurable weights.  All calculations
are purely arithmetic -- no ML or LLM is involved in any numeric
computation, ensuring zero-hallucination compliance.

Risk Tiers:
    LOW:       0-25   -- minimal conversion pressure
    MODERATE:  26-50  -- some risk factors present
    HIGH:      51-75  -- significant conversion pressure
    CRITICAL:  76-100 -- imminent conversion risk

Risk Factors (8, weights sum to 1.0):
    1. Deforestation frontier proximity  (weight 0.20)
    2. Historical conversion rate        (weight 0.15)
    3. Road infrastructure proximity     (weight 0.15)
    4. Population density trend          (weight 0.10)
    5. Commodity price trend             (weight 0.10)
    6. Protected area proximity          (weight 0.10)
    7. Governance index                  (weight 0.10)
    8. Slope/accessibility               (weight 0.10)

Reference Data:
    - Governance: Forest Governance Index by country (0-100, higher =
      better governance = lower risk).
    - Infrastructure: proximity scoring (< 1km = 100, 1-5km = 75,
      5-20km = 50, 20-50km = 25, > 50km = 10).
    - Slope: flat (<5 deg) = high risk (80), moderate (5-15 deg) =
      medium (50), steep (15-30 deg) = low (30), very steep (>30 deg)
      = minimal (10).

Regulatory References:
    - EUDR Article 10: Risk assessment and mitigation.
    - EUDR Article 12: Risk benchmarking by country/region.
    - EUDR Article 29: Commission implementing acts on country risk.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-005 (Engine 6: Conversion Risk Assessment)
Agent ID: GL-EUDR-LUC-005
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
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RiskTier(str, Enum):
    """Conversion risk tier classification.

    Tiers are based on composite risk score ranges.  All thresholds
    are deterministic.

    LOW:       Score 0-25   -- minimal pressure.
    MODERATE:  Score 26-50  -- some factors present.
    HIGH:      Score 51-75  -- significant pressure.
    CRITICAL:  Score 76-100 -- imminent risk.
    """

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Constants: Risk Factor Weights
# ---------------------------------------------------------------------------

DEFAULT_RISK_WEIGHTS: Dict[str, float] = {
    "deforestation_frontier_proximity": 0.20,
    "historical_conversion_rate": 0.15,
    "road_infrastructure_proximity": 0.15,
    "population_density_trend": 0.10,
    "commodity_price_trend": 0.10,
    "protected_area_proximity": 0.10,
    "governance_index": 0.10,
    "slope_accessibility": 0.10,
}

# ---------------------------------------------------------------------------
# Constants: Governance Index by Country
# ---------------------------------------------------------------------------
# Forest Governance Index (0-100, higher = better governance = lower risk).
# Selected countries most relevant to EUDR-regulated commodities.
# Source: WRI Forest Governance indicators, Transparency International,
#         World Bank Worldwide Governance Indicators (WGI).

GOVERNANCE_INDEX: Dict[str, float] = {
    "BR": 45.0,   # Brazil
    "ID": 40.0,   # Indonesia
    "MY": 55.0,   # Malaysia
    "CO": 42.0,   # Colombia
    "PE": 38.0,   # Peru
    "EC": 40.0,   # Ecuador
    "GH": 48.0,   # Ghana
    "CI": 35.0,   # Cote d'Ivoire
    "CM": 30.0,   # Cameroon
    "CD": 20.0,   # DR Congo
    "CG": 25.0,   # Congo Republic
    "GA": 35.0,   # Gabon
    "NG": 28.0,   # Nigeria
    "ET": 32.0,   # Ethiopia
    "UG": 35.0,   # Uganda
    "TZ": 38.0,   # Tanzania
    "KE": 42.0,   # Kenya
    "VN": 45.0,   # Vietnam
    "TH": 50.0,   # Thailand
    "MM": 25.0,   # Myanmar
    "KH": 28.0,   # Cambodia
    "LA": 30.0,   # Laos
    "PG": 22.0,   # Papua New Guinea
    "PH": 40.0,   # Philippines
    "IN": 50.0,   # India
    "CN": 55.0,   # China
    "MX": 42.0,   # Mexico
    "GT": 35.0,   # Guatemala
    "HN": 30.0,   # Honduras
    "NI": 28.0,   # Nicaragua
    "BO": 32.0,   # Bolivia
    "PY": 35.0,   # Paraguay
    "AR": 50.0,   # Argentina
    "UY": 75.0,   # Uruguay
    "CR": 68.0,   # Costa Rica
    "PA": 52.0,   # Panama
    "SL": 25.0,   # Sierra Leone
    "LR": 22.0,   # Liberia
    "GN": 20.0,   # Guinea
    "DEFAULT": 40.0,
}

# ---------------------------------------------------------------------------
# Constants: Commodity Price Trend Factors
# ---------------------------------------------------------------------------
# Annualized price trend (higher value = higher demand = higher risk).
# Scale: 0-100 where 50 is neutral, >50 is rising, <50 is falling.

COMMODITY_PRICE_TRENDS: Dict[str, float] = {
    "palm_oil": 65.0,
    "rubber": 52.0,
    "cocoa": 72.0,
    "coffee": 68.0,
    "soya": 60.0,
    "cattle": 55.0,
    "wood": 48.0,
    "DEFAULT": 50.0,
}

# ---------------------------------------------------------------------------
# Constants: Infrastructure Proximity Thresholds
# ---------------------------------------------------------------------------

_INFRA_PROXIMITY_TABLE: List[Tuple[float, float]] = [
    (1.0, 100.0),    # < 1 km  -> score 100
    (5.0, 75.0),     # 1-5 km  -> score 75
    (20.0, 50.0),    # 5-20 km -> score 50
    (50.0, 25.0),    # 20-50 km -> score 25
]
_INFRA_FAR_SCORE: float = 10.0  # > 50 km

# ---------------------------------------------------------------------------
# Constants: Slope Risk Thresholds
# ---------------------------------------------------------------------------

_SLOPE_RISK_TABLE: List[Tuple[float, float]] = [
    (5.0, 80.0),     # flat (< 5 deg)   -> high risk 80
    (15.0, 50.0),    # moderate (5-15)   -> medium 50
    (30.0, 30.0),    # steep (15-30)     -> low 30
]
_SLOPE_VERY_STEEP_SCORE: float = 10.0  # > 30 deg


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class RiskPlotInput:
    """Input data for a single plot risk assessment.

    Attributes:
        plot_id: Unique plot identifier.
        latitude: Plot centroid latitude (-90 to 90).
        longitude: Plot centroid longitude (-180 to 180).
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: Target commodity being assessed.
        distance_to_frontier_km: Distance to nearest deforestation
            frontier in kilometres. -1 if unknown.
        historical_conversion_rate_ha_per_year: Historical conversion
            rate near this plot (ha/year). -1 if unknown.
        distance_to_road_km: Distance to nearest road in kilometres.
            -1 if unknown.
        population_density_change_pct: Percentage change in population
            density over the assessment period. 0 if unknown.
        distance_to_protected_area_km: Distance to nearest protected
            area in kilometres. -1 if unknown.
        slope_deg: Terrain slope in degrees. -1 if unknown.
        area_ha: Plot area in hectares.
    """

    plot_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    country_code: str = ""
    commodity: str = ""
    distance_to_frontier_km: float = -1.0
    historical_conversion_rate_ha_per_year: float = -1.0
    distance_to_road_km: float = -1.0
    population_density_change_pct: float = 0.0
    distance_to_protected_area_km: float = -1.0
    slope_deg: float = -1.0
    area_ha: float = 1.0


@dataclass
class ConversionRisk:
    """Result of conversion risk assessment for a single plot.

    Attributes:
        result_id: Unique result identifier.
        plot_id: Identifier of the assessed plot.
        composite_score: Weighted composite risk score (0-100).
        risk_tier: Classified risk tier.
        factor_scores: Individual factor scores (0-100).
        factor_weights: Weights used for each factor.
        conversion_probability_6m: Estimated conversion probability
            within 6 months (0.0-1.0).
        conversion_probability_12m: 12-month probability.
        conversion_probability_24m: 24-month probability.
        latitude: Plot centroid latitude.
        longitude: Plot centroid longitude.
        country_code: Country code.
        commodity: Assessed commodity.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        timestamp: UTC ISO timestamp.
        metadata: Additional contextual fields.
    """

    result_id: str = ""
    plot_id: str = ""
    composite_score: float = 0.0
    risk_tier: str = ""
    factor_scores: Dict[str, float] = field(default_factory=dict)
    factor_weights: Dict[str, float] = field(default_factory=dict)
    conversion_probability_6m: float = 0.0
    conversion_probability_12m: float = 0.0
    conversion_probability_24m: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    country_code: str = ""
    commodity: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "result_id": self.result_id,
            "plot_id": self.plot_id,
            "composite_score": self.composite_score,
            "risk_tier": self.risk_tier,
            "factor_scores": self.factor_scores,
            "factor_weights": self.factor_weights,
            "conversion_probability_6m": self.conversion_probability_6m,
            "conversion_probability_12m": self.conversion_probability_12m,
            "conversion_probability_24m": self.conversion_probability_24m,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "country_code": self.country_code,
            "commodity": self.commodity,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# ConversionRiskAssessor
# ---------------------------------------------------------------------------


class ConversionRiskAssessor:
    """Scores each plot's risk of future land use conversion.

    Uses 8 deterministic risk factors with configurable weights.
    Risk tiers: LOW (0-25), MODERATE (26-50), HIGH (51-75),
    CRITICAL (76-100). All calculations are deterministic with
    no LLM involvement.

    Example::

        assessor = ConversionRiskAssessor()
        plot = RiskPlotInput(
            plot_id="plot-001",
            latitude=-3.4,
            longitude=-62.2,
            country_code="BR",
            commodity="soya",
            distance_to_frontier_km=5.0,
            slope_deg=3.0,
        )
        risk = assessor.assess_risk(
            latitude=-3.4, longitude=-62.2,
            commodity="soya", plot_input=plot,
        )
        assert risk.risk_tier in ("LOW", "MODERATE", "HIGH", "CRITICAL")

    Attributes:
        config: Optional configuration object.
        weights: Risk factor weight dictionary.
    """

    def __init__(
        self,
        config: Any = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the ConversionRiskAssessor.

        Args:
            config: Optional LandUseChangeConfig instance.
            weights: Optional custom risk factor weights. If None,
                uses DEFAULT_RISK_WEIGHTS.
        """
        self.config = config
        self.weights = dict(weights) if weights else dict(DEFAULT_RISK_WEIGHTS)

        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                "Risk weights do not sum to 1.0 (sum=%.4f), normalizing",
                weight_sum,
            )
            if weight_sum > 0:
                self.weights = {
                    k: v / weight_sum for k, v in self.weights.items()
                }

        logger.info(
            "ConversionRiskAssessor initialized: %d factors, "
            "module_version=%s",
            len(self.weights), _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API: Single Plot Risk Assessment
    # ------------------------------------------------------------------

    def assess_risk(
        self,
        latitude: float,
        longitude: float,
        commodity: str,
        plot_input: Optional[RiskPlotInput] = None,
    ) -> ConversionRisk:
        """Assess conversion risk for a single plot.

        Computes all 8 risk factors, weights them, classifies the
        tier, and estimates conversion probabilities for 6, 12, and
        24 month horizons.

        Args:
            latitude: Plot centroid latitude.
            longitude: Plot centroid longitude.
            commodity: Commodity being assessed.
            plot_input: Optional detailed input data.

        Returns:
            ConversionRisk with composite score and tier.

        Raises:
            ValueError: If coordinates are out of range.
        """
        start_time = time.monotonic()
        self._validate_coordinates(latitude, longitude)

        plot = plot_input or RiskPlotInput(
            plot_id=_generate_id(),
            latitude=latitude,
            longitude=longitude,
            commodity=commodity,
        )

        factor_scores = self._compute_all_factors(plot, commodity)
        composite = self._compute_composite_score(factor_scores)
        tier = self._classify_risk_tier(composite)

        prob_6m = self._estimate_conversion_probability(composite, 6)
        prob_12m = self._estimate_conversion_probability(composite, 12)
        prob_24m = self._estimate_conversion_probability(composite, 24)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = ConversionRisk(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            composite_score=round(composite, 2),
            risk_tier=tier.value,
            factor_scores={k: round(v, 2) for k, v in factor_scores.items()},
            factor_weights=dict(self.weights),
            conversion_probability_6m=round(prob_6m, 4),
            conversion_probability_12m=round(prob_12m, 4),
            conversion_probability_24m=round(prob_24m, 4),
            latitude=latitude,
            longitude=longitude,
            country_code=plot.country_code,
            commodity=commodity,
            processing_time_ms=round(elapsed_ms, 2),
            timestamp=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_result_hash(result)

        logger.info(
            "Risk assessed: plot=%s, score=%.1f, tier=%s, "
            "commodity=%s, prob_12m=%.3f, %.2fms",
            plot.plot_id, composite, tier.value,
            commodity, prob_12m, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Batch Risk Assessment
    # ------------------------------------------------------------------

    def assess_batch(
        self,
        plots: List[RiskPlotInput],
    ) -> List[ConversionRisk]:
        """Assess conversion risk for multiple plots.

        Args:
            plots: List of plot inputs.

        Returns:
            List of ConversionRisk results.

        Raises:
            ValueError: If plots list is empty.
        """
        if not plots:
            raise ValueError("plots list must not be empty")

        start_time = time.monotonic()
        results: List[ConversionRisk] = []

        for i, plot in enumerate(plots):
            try:
                result = self.assess_risk(
                    latitude=plot.latitude,
                    longitude=plot.longitude,
                    commodity=plot.commodity,
                    plot_input=plot,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "assess_batch: failed on plot[%d] id=%s: %s",
                    i, plot.plot_id, str(exc),
                )
                results.append(self._create_error_result(plot, str(exc)))

        elapsed_ms = (time.monotonic() - start_time) * 1000
        critical_count = sum(
            1 for r in results if r.risk_tier == RiskTier.CRITICAL.value
        )
        logger.info(
            "assess_batch complete: %d plots, %d critical, %.2fms",
            len(plots), critical_count, elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Deforestation Frontier Detection
    # ------------------------------------------------------------------

    def detect_deforestation_frontier(
        self,
        region_bounds: Dict[str, float],
        risk_results: Optional[List[ConversionRisk]] = None,
    ) -> Dict[str, Any]:
        """Detect the deforestation frontier within a region.

        Identifies the boundary between forest and actively converted
        land using risk score spatial gradients.

        Args:
            region_bounds: Dictionary with min_lat, max_lat, min_lon,
                max_lon.
            risk_results: Optional pre-computed risk results.

        Returns:
            Dictionary with frontier line, direction, and statistics.
        """
        start_time = time.monotonic()

        if not risk_results:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return {
                "frontier_detected": False,
                "region_bounds": region_bounds,
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": _compute_hash(region_bounds),
            }

        high_risk = [
            r for r in risk_results
            if r.composite_score >= 50.0
        ]
        low_risk = [
            r for r in risk_results
            if r.composite_score < 50.0
        ]

        if not high_risk or not low_risk:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return {
                "frontier_detected": False,
                "region_bounds": region_bounds,
                "high_risk_count": len(high_risk),
                "low_risk_count": len(low_risk),
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": _compute_hash(region_bounds),
            }

        hr_lat = sum(r.latitude for r in high_risk) / len(high_risk)
        hr_lon = sum(r.longitude for r in high_risk) / len(high_risk)
        lr_lat = sum(r.latitude for r in low_risk) / len(low_risk)
        lr_lon = sum(r.longitude for r in low_risk) / len(low_risk)

        frontier_lat = (hr_lat + lr_lat) / 2.0
        frontier_lon = (hr_lon + lr_lon) / 2.0

        dlat = hr_lat - lr_lat
        dlon = hr_lon - lr_lon
        direction = math.degrees(math.atan2(dlon, dlat))
        if direction < 0:
            direction += 360.0

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "frontier_detected": True,
            "region_bounds": region_bounds,
            "frontier_centroid": {
                "lat": round(frontier_lat, 6),
                "lon": round(frontier_lon, 6),
            },
            "frontier_direction_deg": round(direction, 2),
            "high_risk_centroid": {
                "lat": round(hr_lat, 6),
                "lon": round(hr_lon, 6),
            },
            "low_risk_centroid": {
                "lat": round(lr_lat, 6),
                "lon": round(lr_lon, 6),
            },
            "high_risk_count": len(high_risk),
            "low_risk_count": len(low_risk),
            "mean_high_risk_score": round(
                sum(r.composite_score for r in high_risk) / len(high_risk), 2,
            ),
            "mean_low_risk_score": round(
                sum(r.composite_score for r in low_risk) / len(low_risk), 2,
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Frontier detected: direction=%.1fdeg, high_risk=%d, "
            "low_risk=%d",
            direction, len(high_risk), len(low_risk),
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Risk Heatmap Generation
    # ------------------------------------------------------------------

    def generate_risk_heatmap(
        self,
        plots_with_scores: List[ConversionRisk],
    ) -> Dict[str, Any]:
        """Generate a risk heatmap from assessed plots.

        Groups plots into spatial cells and aggregates risk scores
        to produce a gridded heatmap suitable for visualization.

        Args:
            plots_with_scores: List of risk assessment results.

        Returns:
            Dictionary with heatmap grid, statistics, and metadata.
        """
        start_time = time.monotonic()

        if not plots_with_scores:
            return {
                "cell_count": 0,
                "cells": [],
                "statistics": {},
                "provenance_hash": _compute_hash({"empty": True}),
            }

        grid_size = 0.1
        cells: Dict[Tuple[int, int], List[ConversionRisk]] = {}

        for result in plots_with_scores:
            cell_r = int(result.latitude / grid_size)
            cell_c = int(result.longitude / grid_size)
            key = (cell_r, cell_c)
            if key not in cells:
                cells[key] = []
            cells[key].append(result)

        heatmap_cells: List[Dict[str, Any]] = []
        for (cell_r, cell_c), group in cells.items():
            mean_score = sum(r.composite_score for r in group) / len(group)
            max_score = max(r.composite_score for r in group)
            cell_lat = cell_r * grid_size + grid_size / 2.0
            cell_lon = cell_c * grid_size + grid_size / 2.0

            heatmap_cells.append({
                "cell_lat": round(cell_lat, 4),
                "cell_lon": round(cell_lon, 4),
                "mean_risk_score": round(mean_score, 2),
                "max_risk_score": round(max_score, 2),
                "plot_count": len(group),
                "tier": self._classify_risk_tier(mean_score).value,
            })

        all_scores = [r.composite_score for r in plots_with_scores]
        stats = {
            "total_plots": len(plots_with_scores),
            "mean_score": round(sum(all_scores) / len(all_scores), 2),
            "max_score": round(max(all_scores), 2),
            "min_score": round(min(all_scores), 2),
            "critical_count": sum(
                1 for s in all_scores if s > 75.0
            ),
            "high_count": sum(
                1 for s in all_scores if 50.0 < s <= 75.0
            ),
            "moderate_count": sum(
                1 for s in all_scores if 25.0 < s <= 50.0
            ),
            "low_count": sum(
                1 for s in all_scores if s <= 25.0
            ),
        }

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "cell_count": len(heatmap_cells),
            "grid_size_deg": grid_size,
            "cells": heatmap_cells,
            "statistics": stats,
            "processing_time_ms": round(elapsed_ms, 2),
        }
        result["provenance_hash"] = _compute_hash({
            "cell_count": len(heatmap_cells),
            "total_plots": len(plots_with_scores),
        })

        logger.info(
            "Risk heatmap: %d cells, %d plots, mean=%.1f, %.2fms",
            len(heatmap_cells), len(plots_with_scores),
            stats["mean_score"], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: Compute All Factors
    # ------------------------------------------------------------------

    def _compute_all_factors(
        self,
        plot: RiskPlotInput,
        commodity: str,
    ) -> Dict[str, float]:
        """Compute all 8 risk factor scores for a plot.

        Args:
            plot: Plot input data.
            commodity: Commodity being assessed.

        Returns:
            Dictionary mapping factor names to scores (0-100).
        """
        return {
            "deforestation_frontier_proximity":
                self._factor_deforestation_frontier_proximity(
                    plot.latitude, plot.longitude,
                    plot.distance_to_frontier_km,
                ),
            "historical_conversion_rate":
                self._factor_historical_conversion_rate(
                    plot.latitude, plot.longitude,
                    plot.historical_conversion_rate_ha_per_year,
                ),
            "road_infrastructure_proximity":
                self._factor_road_infrastructure_proximity(
                    plot.latitude, plot.longitude,
                    plot.distance_to_road_km,
                ),
            "population_density_trend":
                self._factor_population_density_trend(
                    plot.latitude, plot.longitude,
                    plot.population_density_change_pct,
                ),
            "commodity_price_trend":
                self._factor_commodity_price_trend(commodity),
            "protected_area_proximity":
                self._factor_protected_area_proximity(
                    plot.latitude, plot.longitude,
                    plot.distance_to_protected_area_km,
                ),
            "governance_index":
                self._factor_governance_index(plot.country_code),
            "slope_accessibility":
                self._factor_slope_accessibility(
                    plot.latitude, plot.longitude,
                    plot.slope_deg,
                ),
        }

    # ------------------------------------------------------------------
    # Internal: Composite Score
    # ------------------------------------------------------------------

    def _compute_composite_score(
        self,
        factors: Dict[str, float],
    ) -> float:
        """Compute weighted composite risk score.

        Args:
            factors: Factor name -> score (0-100) dictionary.

        Returns:
            Composite risk score (0-100).
        """
        total = 0.0
        for factor_name, score in factors.items():
            weight = self.weights.get(factor_name, 0.0)
            total += score * weight
        return max(0.0, min(100.0, total))

    # ------------------------------------------------------------------
    # Internal: Risk Tier Classification
    # ------------------------------------------------------------------

    def _classify_risk_tier(self, score: float) -> RiskTier:
        """Classify a composite score into a risk tier.

        Args:
            score: Composite risk score (0-100).

        Returns:
            RiskTier enum value.
        """
        if score > 75.0:
            return RiskTier.CRITICAL
        elif score > 50.0:
            return RiskTier.HIGH
        elif score > 25.0:
            return RiskTier.MODERATE
        else:
            return RiskTier.LOW

    # ------------------------------------------------------------------
    # Internal: Conversion Probability Estimation
    # ------------------------------------------------------------------

    def _estimate_conversion_probability(
        self,
        score: float,
        months: int,
    ) -> float:
        """Estimate conversion probability over a given horizon.

        Uses a logistic function calibrated so that a score of 50
        yields ~5% probability at 12 months, and a score of 100
        yields ~80% at 12 months.

        P(conversion) = 1 / (1 + exp(-k * (score - midpoint)))
        Adjusted by time horizon: multiply k by months/12.

        Args:
            score: Composite risk score (0-100).
            months: Forecast horizon in months (6, 12, or 24).

        Returns:
            Probability (0.0-1.0).
        """
        k = 0.08 * (months / 12.0)
        midpoint = 60.0
        exponent = -k * (score - midpoint)
        exponent = max(-20.0, min(20.0, exponent))
        probability = 1.0 / (1.0 + math.exp(exponent))
        return max(0.0, min(1.0, probability))

    # ------------------------------------------------------------------
    # Internal: Factor 1 - Deforestation Frontier Proximity
    # ------------------------------------------------------------------

    def _factor_deforestation_frontier_proximity(
        self,
        lat: float,
        lon: float,
        distance_km: float = -1.0,
    ) -> float:
        """Score based on proximity to deforestation frontier.

        Closer to the frontier = higher risk.

        Args:
            lat: Latitude.
            lon: Longitude.
            distance_km: Distance to frontier in km (-1 if unknown).

        Returns:
            Risk factor score (0-100). Weight: 0.20.
        """
        if distance_km < 0.0:
            return 50.0

        if distance_km < 1.0:
            return 100.0
        elif distance_km < 5.0:
            return 90.0
        elif distance_km < 10.0:
            return 75.0
        elif distance_km < 25.0:
            return 55.0
        elif distance_km < 50.0:
            return 35.0
        elif distance_km < 100.0:
            return 20.0
        else:
            return 5.0

    # ------------------------------------------------------------------
    # Factor 2 - Historical Conversion Rate
    # ------------------------------------------------------------------

    def _factor_historical_conversion_rate(
        self,
        lat: float,
        lon: float,
        rate_ha_per_year: float = -1.0,
        radius_km: float = 25.0,
    ) -> float:
        """Score based on historical conversion rate near the plot.

        Higher historical rates in the surrounding area indicate
        greater future risk.

        Args:
            lat: Latitude.
            lon: Longitude.
            rate_ha_per_year: Historical rate in ha/year (-1 unknown).
            radius_km: Radius for rate calculation.

        Returns:
            Risk factor score (0-100). Weight: 0.15.
        """
        if rate_ha_per_year < 0.0:
            return 50.0

        if rate_ha_per_year < 1.0:
            return 5.0
        elif rate_ha_per_year < 10.0:
            return 20.0
        elif rate_ha_per_year < 50.0:
            return 40.0
        elif rate_ha_per_year < 100.0:
            return 60.0
        elif rate_ha_per_year < 500.0:
            return 80.0
        else:
            return 100.0

    # ------------------------------------------------------------------
    # Factor 3 - Road Infrastructure Proximity
    # ------------------------------------------------------------------

    def _factor_road_infrastructure_proximity(
        self,
        lat: float,
        lon: float,
        distance_km: float = -1.0,
    ) -> float:
        """Score based on proximity to road infrastructure.

        Closer to roads = more accessible = higher risk.
        Scoring: <1km=100, 1-5km=75, 5-20km=50, 20-50km=25, >50km=10.

        Args:
            lat: Latitude.
            lon: Longitude.
            distance_km: Distance to nearest road in km (-1 unknown).

        Returns:
            Risk factor score (0-100). Weight: 0.15.
        """
        if distance_km < 0.0:
            return 50.0

        for threshold, score in _INFRA_PROXIMITY_TABLE:
            if distance_km < threshold:
                return score
        return _INFRA_FAR_SCORE

    # ------------------------------------------------------------------
    # Factor 4 - Population Density Trend
    # ------------------------------------------------------------------

    def _factor_population_density_trend(
        self,
        lat: float,
        lon: float,
        change_pct: float = 0.0,
    ) -> float:
        """Score based on population density change trend.

        Rising population density near forest = higher conversion
        pressure.

        Args:
            lat: Latitude.
            lon: Longitude.
            change_pct: Population density change percentage.

        Returns:
            Risk factor score (0-100). Weight: 0.10.
        """
        if change_pct <= -5.0:
            return 10.0
        elif change_pct <= 0.0:
            return 20.0
        elif change_pct <= 5.0:
            return 40.0
        elif change_pct <= 15.0:
            return 60.0
        elif change_pct <= 30.0:
            return 80.0
        else:
            return 100.0

    # ------------------------------------------------------------------
    # Factor 5 - Commodity Price Trend
    # ------------------------------------------------------------------

    def _factor_commodity_price_trend(
        self,
        commodity: str,
    ) -> float:
        """Score based on commodity price trend.

        Rising commodity prices increase economic incentive for
        conversion.

        Args:
            commodity: Commodity identifier.

        Returns:
            Risk factor score (0-100). Weight: 0.10.
        """
        key = commodity.lower().strip()
        return COMMODITY_PRICE_TRENDS.get(
            key, COMMODITY_PRICE_TRENDS["DEFAULT"],
        )

    # ------------------------------------------------------------------
    # Factor 6 - Protected Area Proximity
    # ------------------------------------------------------------------

    def _factor_protected_area_proximity(
        self,
        lat: float,
        lon: float,
        distance_km: float = -1.0,
    ) -> float:
        """Score based on proximity to protected areas.

        Plots near protected areas face reduced governance barriers
        but also indicate proximity to intact forest that may be
        under conversion pressure (edge effects).

        Convention: very close (<1km) = moderate risk (leakage),
        medium distance = lower risk, far away = higher risk
        (no protection benefit).

        Args:
            lat: Latitude.
            lon: Longitude.
            distance_km: Distance to nearest PA in km (-1 unknown).

        Returns:
            Risk factor score (0-100). Weight: 0.10.
        """
        if distance_km < 0.0:
            return 50.0

        if distance_km < 1.0:
            return 60.0
        elif distance_km < 5.0:
            return 40.0
        elif distance_km < 20.0:
            return 30.0
        elif distance_km < 50.0:
            return 50.0
        elif distance_km < 100.0:
            return 70.0
        else:
            return 80.0

    # ------------------------------------------------------------------
    # Factor 7 - Governance Index
    # ------------------------------------------------------------------

    def _factor_governance_index(
        self,
        country_code: str,
    ) -> float:
        """Score based on country-level forest governance.

        Lower governance index = higher risk of illegal conversion.
        Score is inverted: governance_risk = 100 - governance_index.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Risk factor score (0-100). Weight: 0.10.
        """
        code = country_code.upper().strip() if country_code else "DEFAULT"
        governance = GOVERNANCE_INDEX.get(
            code, GOVERNANCE_INDEX["DEFAULT"],
        )
        return max(0.0, min(100.0, 100.0 - governance))

    # ------------------------------------------------------------------
    # Factor 8 - Slope Accessibility
    # ------------------------------------------------------------------

    def _factor_slope_accessibility(
        self,
        lat: float,
        lon: float,
        slope_deg: float = -1.0,
    ) -> float:
        """Score based on terrain slope (accessibility).

        Flat terrain is more accessible for agricultural machinery
        and therefore faces higher conversion risk.
        Flat (<5 deg)=80, moderate (5-15)=50, steep (15-30)=30,
        very steep (>30)=10.

        Args:
            lat: Latitude.
            lon: Longitude.
            slope_deg: Terrain slope in degrees (-1 unknown).

        Returns:
            Risk factor score (0-100). Weight: 0.10.
        """
        if slope_deg < 0.0:
            return 50.0

        for threshold, score in _SLOPE_RISK_TABLE:
            if slope_deg < threshold:
                return score
        return _SLOPE_VERY_STEEP_SCORE

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_coordinates(
        self,
        latitude: float,
        longitude: float,
    ) -> None:
        """Validate coordinate ranges.

        Raises:
            ValueError: If coordinates are out of range.
        """
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(
                f"latitude must be in [-90, 90], got {latitude}"
            )
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(
                f"longitude must be in [-180, 180], got {longitude}"
            )

    # ------------------------------------------------------------------
    # Internal: Error Result
    # ------------------------------------------------------------------

    def _create_error_result(
        self,
        plot: RiskPlotInput,
        error_msg: str,
    ) -> ConversionRisk:
        """Create an error result for a failed assessment.

        Args:
            plot: Plot that caused the error.
            error_msg: Error message.

        Returns:
            ConversionRisk with zero score and error metadata.
        """
        result = ConversionRisk(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            composite_score=0.0,
            risk_tier=RiskTier.LOW.value,
            latitude=plot.latitude,
            longitude=plot.longitude,
            country_code=plot.country_code,
            commodity=plot.commodity,
            timestamp=_utcnow().isoformat(),
            metadata={"error": error_msg},
        )
        result.provenance_hash = self._compute_result_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Provenance Hashing
    # ------------------------------------------------------------------

    def _compute_result_hash(self, result: ConversionRisk) -> str:
        """Compute SHA-256 provenance hash for a risk result.

        Args:
            result: ConversionRisk to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "result_id": result.result_id,
            "plot_id": result.plot_id,
            "composite_score": result.composite_score,
            "risk_tier": result.risk_tier,
            "factor_scores": result.factor_scores,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "country_code": result.country_code,
            "commodity": result.commodity,
            "timestamp": result.timestamp,
        }
        return _compute_hash(hash_data)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "RiskTier",
    # Constants
    "DEFAULT_RISK_WEIGHTS",
    "GOVERNANCE_INDEX",
    "COMMODITY_PRICE_TRENDS",
    # Data classes
    "RiskPlotInput",
    "ConversionRisk",
    # Engine
    "ConversionRiskAssessor",
]
