# -*- coding: utf-8 -*-
"""
Spatial Overlap Detection Engine - AGENT-EUDR-022 (Feature 2)

PostGIS-powered spatial analysis engine that detects overlaps between
supply chain production plots and protected area boundaries. Performs
polygon intersection (ST_Intersects, ST_Intersection, ST_Area), IUCN-
category-aware risk scoring, multi-designation overlap detection, and
batch screening of 10,000+ plots against the full WDPA database.

Zero-Hallucination: All spatial calculations use deterministic PostGIS
functions. Risk scores use Decimal arithmetic. No LLM involvement.

Performance Targets:
    - Single plot: < 500ms p99
    - Batch 10,000 plots: < 5 minutes

Example:
    >>> engine = SpatialOverlapEngine(config)
    >>> overlap = await engine.check_single_plot(
    ...     plot_id="plot-001", latitude=Decimal("-3.1234"),
    ...     longitude=Decimal("28.5678"), pool=pool,
    ... )
    >>> print(overlap.highest_risk_level)

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.protected_area_validator.config import ProtectedAreaValidatorConfig
from greenlang.agents.eudr.protected_area_validator.models import (
    ProtectedAreaOverlap,
    PAOverlapType,
    RiskLevel,
    IUCNCategory,
    DesignationLevel,
    CheckOverlapResponse,
    BatchScreeningJob,
)
from greenlang.agents.eudr.protected_area_validator.provenance import get_tracker
from greenlang.agents.eudr.protected_area_validator.reference_data.iucn_categories import (
    get_iucn_risk_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL Templates
# ---------------------------------------------------------------------------

_SQL_OVERLAP_CHECK = """
    WITH plot_geom AS (
        SELECT ST_SetSRID(ST_MakePoint(%s, %s), 4326) AS geom
    ),
    nearby_pas AS (
        SELECT pa.wdpa_id, pa.name, pa.iucn_category,
               pa.designation_level, pa.country_code,
               pa.area_hectares, pa.legal_status,
               pa.governance_type, pa.mett_score,
               pa.is_world_heritage, pa.is_ramsar, pa.is_kba, pa.is_aze,
               pa.boundary_geom,
               ST_Distance(
                   pa.boundary_geom::geography,
                   (SELECT geom FROM plot_geom)::geography
               ) AS distance_meters
        FROM eudr_protected_area_validator.gl_eudr_pav_protected_areas pa
        WHERE ST_DWithin(
            pa.boundary_geom::geography,
            (SELECT geom FROM plot_geom)::geography,
            %s
        )
    )
    SELECT wdpa_id, name, iucn_category, designation_level, country_code,
           area_hectares, legal_status, governance_type, mett_score,
           is_world_heritage, is_ramsar, is_kba, is_aze,
           distance_meters,
           ST_Contains(boundary_geom, (SELECT geom FROM plot_geom)) AS is_inside
    FROM nearby_pas
    ORDER BY distance_meters ASC
"""

_SQL_POLYGON_OVERLAP = """
    SELECT pa.wdpa_id, pa.name, pa.iucn_category,
           pa.designation_level, pa.country_code,
           pa.area_hectares, pa.legal_status, pa.governance_type,
           pa.mett_score,
           pa.is_world_heritage, pa.is_ramsar, pa.is_kba, pa.is_aze,
           ST_Area(ST_Intersection(pa.boundary_geom::geography,
               ST_GeomFromText(%s, 4326)::geography)) AS overlap_area_m2,
           ST_Area(ST_GeomFromText(%s, 4326)::geography) AS plot_area_m2,
           ST_Area(pa.boundary_geom::geography) AS pa_area_m2,
           ST_Distance(pa.boundary_geom::geography,
               ST_GeomFromText(%s, 4326)::geography) AS distance_meters,
           ST_Contains(pa.boundary_geom, ST_GeomFromText(%s, 4326)) AS is_inside,
           ST_Intersects(pa.boundary_geom, ST_GeomFromText(%s, 4326)) AS does_intersect,
           ST_Touches(pa.boundary_geom, ST_GeomFromText(%s, 4326)) AS is_touching
    FROM eudr_protected_area_validator.gl_eudr_pav_protected_areas pa
    WHERE ST_DWithin(
        pa.boundary_geom::geography,
        ST_GeomFromText(%s, 4326)::geography,
        %s
    )
    ORDER BY distance_meters ASC
"""

# ---------------------------------------------------------------------------
# Designation level scores (PRD Section 6.1 Feature 2)
# ---------------------------------------------------------------------------

DESIGNATION_LEVEL_SCORES: Dict[str, int] = {
    "international": 100,
    "national": 70,
    "regional": 50,
    "local": 30,
    "proposed": 20,
}

# ---------------------------------------------------------------------------
# Overlap type multipliers (PRD Section 6.1 Feature 2)
# ---------------------------------------------------------------------------

OVERLAP_TYPE_MULTIPLIERS: Dict[str, Decimal] = {
    "inside": Decimal("1.0"),
    "partial": Decimal("0.8"),
    "boundary": Decimal("0.6"),
    "buffer": Decimal("0.3"),
    "clear": Decimal("0.0"),
}


class SpatialOverlapEngine:
    """PostGIS spatial overlap detection engine (Feature 2).

    Detects overlaps between production plots and protected areas using
    deterministic PostGIS spatial functions. Applies IUCN-category-weighted
    risk scoring with configurable overlap type multipliers.

    All calculations use Decimal arithmetic for bit-perfect reproducibility.

    Attributes:
        _config: Agent configuration.
        _tracker: Provenance tracker.
    """

    def __init__(self, config: ProtectedAreaValidatorConfig) -> None:
        """Initialize the spatial overlap engine.

        Args:
            config: Agent configuration with spatial settings.
        """
        self._config = config
        self._tracker = get_tracker()
        logger.info("SpatialOverlapEngine initialized")

    async def check_single_plot(
        self,
        plot_id: str,
        latitude: Decimal,
        longitude: Decimal,
        geometry_wkt: Optional[str] = None,
        buffer_radius_m: Decimal = Decimal("50000"),
        country_enforcement_score: Decimal = Decimal("50"),
        pool: Any = None,
    ) -> CheckOverlapResponse:
        """Check a single plot for protected area overlaps.

        Performs point-in-polygon or polygon-polygon intersection against
        all protected areas within the search radius using PostGIS.

        Args:
            plot_id: Production plot identifier.
            latitude: WGS84 latitude.
            longitude: WGS84 longitude.
            geometry_wkt: Optional WKT polygon for the plot.
            buffer_radius_m: Search radius in meters (default 50km).
            country_enforcement_score: Enforcement score from EUDR-016.
            pool: Database connection pool.

        Returns:
            CheckOverlapResponse with all detected overlaps and risk levels.

        Example:
            >>> resp = await engine.check_single_plot(
            ...     "plot-001", Decimal("-3.1234"), Decimal("28.5678"), pool=pool,
            ... )
            >>> print(resp.total_overlaps, resp.highest_risk_level)
        """
        start = time.monotonic()
        overlaps: List[ProtectedAreaOverlap] = []

        if pool is None:
            logger.warning("No database pool; returning empty response")
            elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
            return CheckOverlapResponse(
                processing_time_ms=elapsed_ms,
            )

        # Determine query type: polygon or point
        if geometry_wkt:
            rows = await self._query_polygon_overlap(
                geometry_wkt, buffer_radius_m, pool
            )
            overlaps = self._process_polygon_results(
                plot_id, rows, country_enforcement_score
            )
        else:
            rows = await self._query_point_overlap(
                latitude, longitude, buffer_radius_m, pool
            )
            overlaps = self._process_point_results(
                plot_id, rows, buffer_radius_m, country_enforcement_score
            )

        # Determine highest risk level
        highest_risk = RiskLevel.CLEAR
        for ovl in overlaps:
            if self._risk_level_rank(ovl.risk_level) > self._risk_level_rank(highest_risk):
                highest_risk = ovl.risk_level

        elapsed_ms = Decimal(str(
            round((time.monotonic() - start) * 1000, 2)
        ))

        # Record provenance
        self._tracker.record(
            "overlap", "detect", plot_id,
            actor="spatial_overlap_engine",
            metadata={
                "total_overlaps": len(overlaps),
                "highest_risk": highest_risk.value,
                "processing_time_ms": str(elapsed_ms),
            },
        )

        return CheckOverlapResponse(
            overlaps=overlaps,
            highest_risk_level=highest_risk,
            total_overlaps=len(overlaps),
            processing_time_ms=elapsed_ms,
        )

    def compute_risk_score(
        self,
        iucn_category: str,
        overlap_type: str,
        designation_level: str,
        mett_score: Optional[Decimal],
        country_enforcement_score: Decimal,
    ) -> Tuple[Decimal, RiskLevel]:
        """Compute the deterministic protected area risk score.

        Formula (PRD Section 6.1 Feature 2):
            Protected_Area_Risk = (
                iucn_category_score * overlap_type_multiplier * 0.50
                + designation_level_score * 0.20
                + management_effectiveness_gap * 0.15
                + country_enforcement_gap * 0.15
            )

        All arithmetic uses Decimal for bit-perfect reproducibility.

        Args:
            iucn_category: IUCN category string.
            overlap_type: Overlap type (inside/partial/boundary/buffer/clear).
            designation_level: Designation level string.
            mett_score: METT effectiveness score (0-100) or None.
            country_enforcement_score: Enforcement score from EUDR-016.

        Returns:
            Tuple of (risk_score, risk_level).

        Example:
            >>> score, level = engine.compute_risk_score(
            ...     "Ia", "inside", "international", None, Decimal("50"),
            ... )
            >>> assert score >= Decimal("80")
            >>> assert level == RiskLevel.CRITICAL
        """
        # IUCN category score
        iucn_score = Decimal(str(get_iucn_risk_score(iucn_category)))

        # Overlap type multiplier
        multiplier = OVERLAP_TYPE_MULTIPLIERS.get(
            overlap_type, Decimal("0.0")
        )

        # Designation level score
        designation_score = Decimal(str(
            DESIGNATION_LEVEL_SCORES.get(designation_level, 50)
        ))

        # Management effectiveness gap (100 - METT score)
        effective_mett = mett_score if mett_score is not None else Decimal("50")
        management_gap = Decimal("100") - effective_mett

        # Country enforcement gap
        enforcement_gap = Decimal("100") - country_enforcement_score

        # Weighted components
        w_iucn = self._config.risk_weight_iucn_category
        w_desig = self._config.risk_weight_designation_level
        w_mgmt = self._config.risk_weight_management_gap
        w_enf = self._config.risk_weight_enforcement_gap

        risk_score = (
            iucn_score * multiplier * w_iucn
            + designation_score * w_desig
            + management_gap * w_mgmt
            + enforcement_gap * w_enf
        )

        # Clamp to 0-100
        risk_score = max(Decimal("0"), min(Decimal("100"), risk_score))
        risk_score = risk_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Classify risk level
        risk_level = self._classify_risk_level(risk_score)

        return risk_score, risk_level

    def classify_overlap_type(
        self,
        is_inside: bool,
        does_intersect: bool,
        is_touching: bool,
        distance_meters: Decimal,
        buffer_radius_m: Decimal,
    ) -> PAOverlapType:
        """Classify the overlap type between a plot and a protected area.

        Args:
            is_inside: Plot entirely within PA (ST_Contains).
            does_intersect: Plot intersects PA (ST_Intersects).
            is_touching: Plot touches PA boundary (ST_Touches).
            distance_meters: Distance from plot to PA (meters).
            buffer_radius_m: Configured buffer radius (meters).

        Returns:
            PAOverlapType classification.
        """
        if is_inside:
            return PAOverlapType.INSIDE
        if does_intersect and not is_touching:
            return PAOverlapType.PARTIAL
        if is_touching:
            return PAOverlapType.BOUNDARY
        if distance_meters <= buffer_radius_m:
            return PAOverlapType.BUFFER
        return PAOverlapType.CLEAR

    async def _query_point_overlap(
        self,
        latitude: Decimal,
        longitude: Decimal,
        radius_m: Decimal,
        pool: Any,
    ) -> List[Dict[str, Any]]:
        """Execute point-based overlap query."""
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_OVERLAP_CHECK,
                    (
                        float(longitude), float(latitude),
                        float(radius_m),
                    ),
                )
                rows = await cur.fetchall()
                columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]

    async def _query_polygon_overlap(
        self,
        geometry_wkt: str,
        radius_m: Decimal,
        pool: Any,
    ) -> List[Dict[str, Any]]:
        """Execute polygon-based overlap query."""
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_POLYGON_OVERLAP,
                    (
                        geometry_wkt, geometry_wkt, geometry_wkt,
                        geometry_wkt, geometry_wkt, geometry_wkt,
                        geometry_wkt, float(radius_m),
                    ),
                )
                rows = await cur.fetchall()
                columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]

    def _process_point_results(
        self,
        plot_id: str,
        rows: List[Dict[str, Any]],
        buffer_radius_m: Decimal,
        country_enforcement_score: Decimal,
    ) -> List[ProtectedAreaOverlap]:
        """Process point query results into overlap models."""
        overlaps: List[ProtectedAreaOverlap] = []
        for row in rows:
            distance_m = Decimal(str(row.get("distance_meters", 0)))
            is_inside = bool(row.get("is_inside", False))

            if is_inside:
                overlap_type = PAOverlapType.INSIDE
            elif distance_m <= Decimal("0"):
                overlap_type = PAOverlapType.BOUNDARY
            elif distance_m <= buffer_radius_m:
                overlap_type = PAOverlapType.BUFFER
            else:
                overlap_type = PAOverlapType.CLEAR

            iucn_cat = str(row.get("iucn_category", "NR"))
            desig_level = str(row.get("designation_level", "national"))
            mett = Decimal(str(row["mett_score"])) if row.get("mett_score") else None

            risk_score, risk_level = self.compute_risk_score(
                iucn_category=iucn_cat,
                overlap_type=overlap_type.value,
                designation_level=desig_level,
                mett_score=mett,
                country_enforcement_score=country_enforcement_score,
            )

            overlaps.append(ProtectedAreaOverlap(
                plot_id=plot_id,
                wdpa_id=int(row["wdpa_id"]),
                protected_area_name=str(row.get("name", "Unknown")),
                iucn_category=iucn_cat,
                designation_level=desig_level,
                overlap_type=overlap_type,
                distance_meters=distance_m.quantize(Decimal("0.01")),
                risk_score=risk_score,
                risk_level=risk_level,
            ))

        return overlaps

    def _process_polygon_results(
        self,
        plot_id: str,
        rows: List[Dict[str, Any]],
        country_enforcement_score: Decimal,
    ) -> List[ProtectedAreaOverlap]:
        """Process polygon query results into overlap models."""
        overlaps: List[ProtectedAreaOverlap] = []
        for row in rows:
            is_inside = bool(row.get("is_inside", False))
            does_intersect = bool(row.get("does_intersect", False))
            is_touching = bool(row.get("is_touching", False))
            distance_m = Decimal(str(row.get("distance_meters", 0)))
            overlap_area_m2 = Decimal(str(row.get("overlap_area_m2", 0)))
            plot_area_m2 = Decimal(str(row.get("plot_area_m2", 1)))
            pa_area_m2 = Decimal(str(row.get("pa_area_m2", 1)))

            # Convert m2 to hectares (1 ha = 10000 m2)
            overlap_ha = (overlap_area_m2 / Decimal("10000")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            overlap_pct_plot = (
                (overlap_area_m2 / plot_area_m2 * Decimal("100"))
                if plot_area_m2 > 0 else Decimal("0")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            overlap_pct_pa = (
                (overlap_area_m2 / pa_area_m2 * Decimal("100"))
                if pa_area_m2 > 0 else Decimal("0")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            overlap_type = self.classify_overlap_type(
                is_inside, does_intersect, is_touching,
                distance_m, self._config.max_buffer_km * Decimal("1000"),
            )

            iucn_cat = str(row.get("iucn_category", "NR"))
            desig_level = str(row.get("designation_level", "national"))
            mett = Decimal(str(row["mett_score"])) if row.get("mett_score") else None

            risk_score, risk_level = self.compute_risk_score(
                iucn_category=iucn_cat,
                overlap_type=overlap_type.value,
                designation_level=desig_level,
                mett_score=mett,
                country_enforcement_score=country_enforcement_score,
            )

            overlaps.append(ProtectedAreaOverlap(
                plot_id=plot_id,
                wdpa_id=int(row["wdpa_id"]),
                protected_area_name=str(row.get("name", "Unknown")),
                iucn_category=iucn_cat,
                designation_level=desig_level,
                overlap_type=overlap_type,
                overlap_area_hectares=overlap_ha,
                overlap_pct_of_plot=min(overlap_pct_plot, Decimal("100")),
                overlap_pct_of_pa=min(overlap_pct_pa, Decimal("100")),
                distance_meters=distance_m.quantize(Decimal("0.01")),
                risk_score=risk_score,
                risk_level=risk_level,
            ))

        return overlaps

    @staticmethod
    def _classify_risk_level(score: Decimal) -> RiskLevel:
        """Classify risk level from score (PRD Section 6.1 Feature 2)."""
        if score >= Decimal("80"):
            return RiskLevel.CRITICAL
        if score >= Decimal("60"):
            return RiskLevel.HIGH
        if score >= Decimal("40"):
            return RiskLevel.MEDIUM
        if score >= Decimal("20"):
            return RiskLevel.LOW
        return RiskLevel.CLEAR

    @staticmethod
    def _risk_level_rank(level: RiskLevel) -> int:
        """Get numeric rank for risk level comparison."""
        ranks = {
            RiskLevel.CLEAR: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }
        return ranks.get(level, 0)
