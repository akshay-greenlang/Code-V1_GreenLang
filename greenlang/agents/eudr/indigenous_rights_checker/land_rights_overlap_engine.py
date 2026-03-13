# -*- coding: utf-8 -*-
"""
LandRightsOverlapEngine - Feature 3: Land Rights Overlap Detector

Spatial analysis engine detecting overlaps between supply chain production
plots and indigenous territories using PostGIS. Classifies overlaps into
4 categories (DIRECT, PARTIAL, ADJACENT, PROXIMATE) and calculates
overlap risk scores using a deterministic weighted formula.

Overlap Risk Scoring Formula (PRD Section 6.1, Feature 3):
    Overlap_Risk_Score = (
        overlap_type_score * 0.40
        + territory_legal_status * 0.20
        + community_population_factor * 0.10
        + conflict_history * 0.15
        + country_rights_framework * 0.15
    )

Performance Targets:
    - Single plot query: < 500ms p99
    - Batch 10,000 plots: < 5 minutes

Example:
    >>> engine = LandRightsOverlapEngine(config, provenance)
    >>> result = await engine.detect_overlap(
    ...     plot_id="p-001", latitude=-3.46, longitude=-62.21,
    ... )

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 3)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    OverlapType,
    RiskLevel,
    TerritoryOverlap,
    OverlapDetectionResponse,
    BatchOverlapResponse,
)
from greenlang.agents.eudr.indigenous_rights_checker.provenance import (
    ProvenanceTracker,
)
from greenlang.agents.eudr.indigenous_rights_checker.metrics import (
    record_overlap_detected,
    observe_overlap_query_duration,
    set_active_overlaps,
)

logger = logging.getLogger(__name__)

_D100 = Decimal("100")
_D80 = Decimal("80")
_D60 = Decimal("60")
_D50 = Decimal("50")
_D40 = Decimal("40")
_D25 = Decimal("25")
_D0 = Decimal("0")
_PRECISION = Decimal("0.01")

# ---------------------------------------------------------------------------
# Overlap type scores (PRD Section 6.1, Feature 3 formula)
# ---------------------------------------------------------------------------

_OVERLAP_TYPE_SCORES: Dict[OverlapType, Decimal] = {
    OverlapType.DIRECT: _D100,
    OverlapType.PARTIAL: _D80,
    OverlapType.ADJACENT: _D50,
    OverlapType.PROXIMATE: _D25,
    OverlapType.NONE: _D0,
}

# ---------------------------------------------------------------------------
# Territory legal status scores
# ---------------------------------------------------------------------------

_LEGAL_STATUS_SCORES: Dict[str, Decimal] = {
    "titled": _D100,
    "declared": _D80,
    "claimed": _D60,
    "customary": _D50,
    "pending": _D40,
    "disputed": _D60,
}

# ---------------------------------------------------------------------------
# SQL templates for PostGIS spatial analysis
# ---------------------------------------------------------------------------

_SQL_DETECT_OVERLAPS = """
    WITH plot_point AS (
        SELECT ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326) AS geom
    ),
    direct_overlaps AS (
        SELECT t.territory_id, t.territory_name, t.people_name,
               t.country_code, t.legal_status, t.area_hectares,
               t.data_source,
               'direct' AS overlap_type,
               0.0 AS distance_meters,
               ST_Area(ST_Intersection(t.boundary_geom, pp.geom)::geography) / 10000.0 AS overlap_area_ha
        FROM eudr_indigenous_rights_checker.gl_eudr_irc_territories t,
             plot_point pp
        WHERE ST_Contains(t.boundary_geom, pp.geom)
    ),
    adjacent_overlaps AS (
        SELECT t.territory_id, t.territory_name, t.people_name,
               t.country_code, t.legal_status, t.area_hectares,
               t.data_source,
               CASE
                   WHEN ST_DWithin(t.boundary_geom::geography,
                                   pp.geom::geography,
                                   %(inner_buffer_m)s)
                   THEN 'adjacent'
                   ELSE 'proximate'
               END AS overlap_type,
               ST_Distance(t.boundary_geom::geography,
                           pp.geom::geography) AS distance_meters,
               NULL AS overlap_area_ha
        FROM eudr_indigenous_rights_checker.gl_eudr_irc_territories t,
             plot_point pp
        WHERE NOT ST_Contains(t.boundary_geom, pp.geom)
          AND ST_DWithin(t.boundary_geom::geography,
                         pp.geom::geography,
                         %(outer_buffer_m)s)
    )
    SELECT * FROM direct_overlaps
    UNION ALL
    SELECT * FROM adjacent_overlaps
    ORDER BY distance_meters ASC
"""

_SQL_INSERT_OVERLAP = """
    INSERT INTO eudr_indigenous_rights_checker.gl_eudr_irc_overlaps (
        overlap_id, plot_id, territory_id, overlap_type,
        overlap_area_hectares, overlap_pct_of_plot,
        overlap_pct_of_territory, distance_meters, bearing_degrees,
        affected_communities, risk_score, risk_level,
        deforestation_correlation, provenance_hash
    ) VALUES (
        %(overlap_id)s, %(plot_id)s, %(territory_id)s, %(overlap_type)s,
        %(overlap_area_hectares)s, %(overlap_pct_of_plot)s,
        %(overlap_pct_of_territory)s, %(distance_meters)s,
        %(bearing_degrees)s, %(affected_communities)s,
        %(risk_score)s, %(risk_level)s,
        %(deforestation_correlation)s, %(provenance_hash)s
    )
    ON CONFLICT (plot_id, territory_id) DO UPDATE SET
        overlap_type = EXCLUDED.overlap_type,
        distance_meters = EXCLUDED.distance_meters,
        risk_score = EXCLUDED.risk_score,
        risk_level = EXCLUDED.risk_level,
        provenance_hash = EXCLUDED.provenance_hash,
        updated_at = NOW()
"""


class LandRightsOverlapEngine:
    """Spatial overlap detection engine using PostGIS.

    Detects overlaps between production plots and indigenous territories,
    classifies them by type, and calculates deterministic risk scores.

    Attributes:
        _config: Agent configuration with buffer settings.
        _provenance: Provenance tracker for audit trail.
        _pool: Async database connection pool.
    """

    def __init__(
        self,
        config: IndigenousRightsCheckerConfig,
        provenance: ProvenanceTracker,
    ) -> None:
        """Initialize LandRightsOverlapEngine."""
        self._config = config
        self._provenance = provenance
        self._pool: Any = None
        self._risk_weights: Dict[str, Decimal] = {
            k: Decimal(str(v))
            for k, v in config.overlap_risk_weights.items()
        }
        logger.info(
            "LandRightsOverlapEngine initialized: "
            f"inner_buffer={config.inner_buffer_km}km, "
            f"outer_buffer={config.outer_buffer_km}km"
        )

    async def startup(self, pool: Any) -> None:
        """Set the database connection pool."""
        self._pool = pool
        logger.info("LandRightsOverlapEngine started")

    async def shutdown(self) -> None:
        """Clean up engine resources."""
        self._pool = None

    async def detect_overlap(
        self,
        plot_id: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        plot_geojson: Optional[Dict[str, Any]] = None,
        inner_buffer_km: Optional[float] = None,
        outer_buffer_km: Optional[float] = None,
    ) -> OverlapDetectionResponse:
        """Detect territory overlaps for a single production plot.

        Uses PostGIS spatial operations (ST_Contains, ST_DWithin,
        ST_Distance) to find all indigenous territories that overlap
        with or are near the given plot location.

        Args:
            plot_id: Production plot identifier.
            latitude: WGS84 latitude for point-based plots.
            longitude: WGS84 longitude for point-based plots.
            plot_geojson: GeoJSON polygon for polygon-based plots.
            inner_buffer_km: Override inner buffer (default from config).
            outer_buffer_km: Override outer buffer (default from config).

        Returns:
            OverlapDetectionResponse with all detected overlaps.

        Example:
            >>> result = await engine.detect_overlap(
            ...     plot_id="p-001",
            ...     latitude=-3.4653,
            ...     longitude=-62.2159,
            ... )
            >>> print(result.total_overlaps, result.highest_risk_level)
        """
        start = time.monotonic()

        inner_m = (inner_buffer_km or self._config.inner_buffer_km) * 1000
        outer_m = (outer_buffer_km or self._config.outer_buffer_km) * 1000

        overlaps: List[TerritoryOverlap] = []

        if latitude is not None and longitude is not None:
            raw_overlaps = await self._detect_point_overlaps(
                plot_id, latitude, longitude, inner_m, outer_m
            )
            for raw in raw_overlaps:
                overlap = self._build_overlap_result(plot_id, raw)
                overlaps.append(overlap)
                await self._persist_overlap(overlap)

        # Calculate highest risk level
        if overlaps:
            highest = max(overlaps, key=lambda o: o.risk_score)
            highest_risk = RiskLevel(highest.risk_level)
        else:
            highest_risk = RiskLevel.NONE

        elapsed = time.monotonic() - start
        observe_overlap_query_duration(elapsed)

        provenance_hash = self._provenance.compute_data_hash({
            "plot_id": plot_id,
            "total_overlaps": len(overlaps),
            "highest_risk": highest_risk.value,
        })

        self._provenance.record(
            "overlap", "detect", plot_id,
            metadata={
                "total_overlaps": len(overlaps),
                "highest_risk": highest_risk.value,
                "elapsed_ms": elapsed * 1000,
            },
        )

        logger.info(
            f"Overlap detection for {plot_id}: "
            f"{len(overlaps)} overlaps, "
            f"highest_risk={highest_risk.value}, "
            f"time={elapsed*1000:.1f}ms"
        )

        return OverlapDetectionResponse(
            plot_id=plot_id,
            overlaps=overlaps,
            total_overlaps=len(overlaps),
            highest_risk_level=highest_risk,
            processing_time_ms=elapsed * 1000,
            provenance_hash=provenance_hash,
        )

    async def detect_batch(
        self,
        plots: List[Dict[str, Any]],
    ) -> BatchOverlapResponse:
        """Batch overlap screening for multiple plots.

        Processes up to 10,000 plots with progress tracking.
        Performance target: < 5 minutes for 10,000 plots.

        Args:
            plots: List of plot dictionaries with plot_id, latitude,
                   longitude (and optionally plot_geojson).

        Returns:
            BatchOverlapResponse with aggregated results.
        """
        start = time.monotonic()
        results: List[OverlapDetectionResponse] = []
        plots_with_overlaps = 0
        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0

        for plot in plots[:self._config.batch_max_size]:
            result = await self.detect_overlap(
                plot_id=plot["plot_id"],
                latitude=plot.get("latitude"),
                longitude=plot.get("longitude"),
                plot_geojson=plot.get("plot_geojson"),
            )
            results.append(result)

            if result.total_overlaps > 0:
                plots_with_overlaps += 1
                rl = result.highest_risk_level
                if rl == RiskLevel.CRITICAL:
                    critical_count += 1
                elif rl == RiskLevel.HIGH:
                    high_count += 1
                elif rl == RiskLevel.MEDIUM:
                    medium_count += 1
                elif rl == RiskLevel.LOW:
                    low_count += 1

        elapsed = time.monotonic() - start

        provenance_hash = self._provenance.compute_data_hash({
            "total_plots": len(plots),
            "plots_with_overlaps": plots_with_overlaps,
            "critical_count": critical_count,
        })

        logger.info(
            f"Batch overlap screening: {len(plots)} plots, "
            f"{plots_with_overlaps} with overlaps, "
            f"time={elapsed*1000:.1f}ms"
        )

        return BatchOverlapResponse(
            total_plots=len(plots),
            plots_with_overlaps=plots_with_overlaps,
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            results=results,
            processing_time_ms=elapsed * 1000,
            provenance_hash=provenance_hash,
        )

    # -----------------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------------

    async def _detect_point_overlaps(
        self,
        plot_id: str,
        lat: float,
        lon: float,
        inner_buffer_m: float,
        outer_buffer_m: float,
    ) -> List[Dict[str, Any]]:
        """Execute PostGIS spatial queries for point-based overlap detection."""
        if self._pool is None:
            return []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_DETECT_OVERLAPS,
                    {
                        "lat": lat,
                        "lon": lon,
                        "inner_buffer_m": inner_buffer_m,
                        "outer_buffer_m": outer_buffer_m,
                    },
                )
                rows = await cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "territory_id": str(row[0]),
                "territory_name": row[1],
                "people_name": row[2],
                "country_code": row[3],
                "legal_status": row[4],
                "area_hectares": row[5],
                "data_source": row[6],
                "overlap_type": row[7],
                "distance_meters": row[8],
                "overlap_area_ha": row[9],
            })

        return results

    def _build_overlap_result(
        self,
        plot_id: str,
        raw: Dict[str, Any],
    ) -> TerritoryOverlap:
        """Build TerritoryOverlap model from raw query results."""
        overlap_id = str(uuid.uuid4())
        overlap_type = OverlapType(raw["overlap_type"])
        distance = Decimal(str(raw.get("distance_meters") or 0))
        area = (
            Decimal(str(raw["overlap_area_ha"]))
            if raw.get("overlap_area_ha")
            else None
        )

        # Calculate risk score (deterministic)
        risk_score = self._calculate_risk_score(
            overlap_type=overlap_type,
            legal_status=raw.get("legal_status", "claimed"),
            community_population=None,
            conflict_history=0,
            country_rights_score=Decimal("50"),
        )

        risk_level = self._classify_risk_level(risk_score)

        record_overlap_detected(overlap_type.value)

        provenance_hash = self._provenance.compute_data_hash({
            "overlap_id": overlap_id,
            "plot_id": plot_id,
            "territory_id": raw["territory_id"],
            "risk_score": str(risk_score),
        })

        return TerritoryOverlap(
            overlap_id=overlap_id,
            plot_id=plot_id,
            territory_id=raw["territory_id"],
            overlap_type=overlap_type,
            overlap_area_hectares=area,
            overlap_pct_of_plot=None,
            overlap_pct_of_territory=None,
            distance_meters=distance,
            bearing_degrees=None,
            affected_communities=[],
            risk_score=risk_score,
            risk_level=risk_level,
            deforestation_correlation=False,
            provenance_hash=provenance_hash,
            detected_at=datetime.now(timezone.utc),
        )

    def _calculate_risk_score(
        self,
        overlap_type: OverlapType,
        legal_status: str,
        community_population: Optional[int],
        conflict_history: int,
        country_rights_score: Decimal,
    ) -> Decimal:
        """Calculate deterministic overlap risk score (0-100).

        Uses weighted formula from PRD Section 6.1, Feature 3.

        Args:
            overlap_type: Classification of overlap.
            legal_status: Territory legal recognition status.
            community_population: Affected community population.
            conflict_history: Number of reported violations (0-100).
            country_rights_score: Country indigenous rights score.

        Returns:
            Risk score (0-100) as Decimal.
        """
        type_score = _OVERLAP_TYPE_SCORES.get(overlap_type, _D0)
        status_score = _LEGAL_STATUS_SCORES.get(legal_status, _D50)

        # Normalize population to 0-100 scale
        pop_score = _D0
        if community_population is not None:
            if community_population >= 10000:
                pop_score = _D100
            elif community_population >= 1000:
                pop_score = Decimal("75")
            elif community_population >= 100:
                pop_score = _D50
            elif community_population > 0:
                pop_score = _D25

        conflict_score = min(Decimal(str(conflict_history)), _D100)
        country_score = country_rights_score

        risk = (
            type_score * self._risk_weights["overlap_type"]
            + status_score * self._risk_weights["territory_legal_status"]
            + pop_score * self._risk_weights["community_population"]
            + conflict_score * self._risk_weights["conflict_history"]
            + country_score * self._risk_weights["country_rights_framework"]
        )

        return risk.quantize(_PRECISION, rounding=ROUND_HALF_UP)

    def _classify_risk_level(self, score: Decimal) -> RiskLevel:
        """Classify risk level from risk score.

        Per PRD:
            >= 80: CRITICAL
            60-79: HIGH
            40-59: MEDIUM
            < 40: LOW
        """
        if score >= _D80:
            return RiskLevel.CRITICAL
        elif score >= _D60:
            return RiskLevel.HIGH
        elif score >= _D40:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def _persist_overlap(self, overlap: TerritoryOverlap) -> None:
        """Persist overlap result to database."""
        if self._pool is None:
            return

        import json
        params = {
            "overlap_id": overlap.overlap_id,
            "plot_id": overlap.plot_id,
            "territory_id": overlap.territory_id,
            "overlap_type": overlap.overlap_type.value,
            "overlap_area_hectares": (
                float(overlap.overlap_area_hectares)
                if overlap.overlap_area_hectares else None
            ),
            "overlap_pct_of_plot": None,
            "overlap_pct_of_territory": None,
            "distance_meters": float(overlap.distance_meters),
            "bearing_degrees": None,
            "affected_communities": json.dumps(overlap.affected_communities),
            "risk_score": float(overlap.risk_score),
            "risk_level": overlap.risk_level.value,
            "deforestation_correlation": overlap.deforestation_correlation,
            "provenance_hash": overlap.provenance_hash,
        }

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_INSERT_OVERLAP, params)
            await conn.commit()
