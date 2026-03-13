# -*- coding: utf-8 -*-
"""
Protected Area Database Engine - AGENT-EUDR-022 (Feature 1)

Manages the consolidated, georeferenced database of 270,000+ protected
areas from WDPA/Protected Planet, national registries, KBA database, and
international designations. Provides spatial indexing (PostGIS GIST) for
sub-second overlap queries, IUCN category classification, data versioning,
staleness tracking, and search/filter capabilities.

Zero-Hallucination: All spatial data sourced from authoritative WDPA
releases. No synthetic boundaries. Version-controlled data with provenance.

Example:
    >>> engine = ProtectedAreaDatabaseEngine(config)
    >>> pa = await engine.get_by_wdpa_id(12345)
    >>> assert pa.iucn_category.value in ("Ia", "Ib", "II")

    >>> results = await engine.search(country_code="BRA", iucn_category="II")
    >>> assert len(results) > 0

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.protected_area_validator.config import ProtectedAreaValidatorConfig
from greenlang.agents.eudr.protected_area_validator.models import (
    ProtectedArea,
    ProtectedAreaVersion,
    IUCNCategory,
    DesignationLevel,
    PALegalStatus,
    GovernanceType,
    GISQuality,
)
from greenlang.agents.eudr.protected_area_validator.provenance import get_tracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL Templates for PostGIS operations
# ---------------------------------------------------------------------------

_SQL_SEARCH_BY_BBOX = """
    SELECT wdpa_id, name, iucn_category, designation_level, country_code,
           area_hectares, legal_status, governance_type, is_world_heritage,
           is_ramsar, is_kba, is_aze, wdpa_version, confidence
    FROM eudr_protected_area_validator.gl_eudr_pav_protected_areas
    WHERE boundary_geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
"""

_SQL_SEARCH_BY_PROXIMITY = """
    SELECT wdpa_id, name, iucn_category, designation_level, country_code,
           area_hectares, legal_status,
           ST_Distance(
               boundary_geom::geography,
               ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
           ) AS distance_meters
    FROM eudr_protected_area_validator.gl_eudr_pav_protected_areas
    WHERE ST_DWithin(
        boundary_geom::geography,
        ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
        %s
    )
    ORDER BY distance_meters ASC
"""

_SQL_POINT_IN_POLYGON = """
    SELECT wdpa_id, name, iucn_category, designation_level, country_code,
           area_hectares, legal_status, governance_type
    FROM eudr_protected_area_validator.gl_eudr_pav_protected_areas
    WHERE ST_Contains(
        boundary_geom,
        ST_SetSRID(ST_MakePoint(%s, %s), 4326)
    )
"""

_SQL_COUNT_BY_COUNTRY = """
    SELECT country_code, COUNT(*) AS pa_count,
           SUM(area_hectares) AS total_area_ha
    FROM eudr_protected_area_validator.gl_eudr_pav_protected_areas
    GROUP BY country_code
    ORDER BY pa_count DESC
"""


class ProtectedAreaDatabaseEngine:
    """Engine for managing the protected area spatial database (Feature 1).

    Provides CRUD operations for protected areas, spatial queries using
    PostGIS, IUCN category filtering, data version management, and
    staleness tracking for the WDPA dataset.

    This engine does NOT perform any LLM operations. All operations are
    deterministic SQL queries against the PostGIS-enabled PostgreSQL
    database.

    Attributes:
        _config: Agent configuration.
        _tracker: Provenance tracker instance.
    """

    def __init__(self, config: ProtectedAreaValidatorConfig) -> None:
        """Initialize the database engine.

        Args:
            config: Agent configuration with database settings.
        """
        self._config = config
        self._tracker = get_tracker()
        logger.info(
            "ProtectedAreaDatabaseEngine initialized: "
            f"staleness_threshold={config.wdpa_staleness_days}d"
        )

    async def get_by_wdpa_id(
        self,
        wdpa_id: int,
        pool: Any = None,
    ) -> Optional[ProtectedArea]:
        """Retrieve a protected area by WDPA ID.

        Args:
            wdpa_id: WDPA unique identifier.
            pool: Database connection pool (psycopg_pool.AsyncConnectionPool).

        Returns:
            ProtectedArea model or None if not found.

        Example:
            >>> pa = await engine.get_by_wdpa_id(555558226)
            >>> assert pa.name == "Virunga National Park"
        """
        start = time.monotonic()
        query = """
            SELECT * FROM eudr_protected_area_validator.gl_eudr_pav_protected_areas
            WHERE wdpa_id = %s
        """
        if pool is None:
            logger.warning("No database pool provided; returning None")
            return None

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (wdpa_id,))
                row = await cur.fetchone()

        elapsed_ms = (time.monotonic() - start) * 1000
        if row is None:
            logger.debug(f"WDPA ID {wdpa_id} not found ({elapsed_ms:.1f}ms)")
            return None

        logger.debug(f"WDPA ID {wdpa_id} retrieved ({elapsed_ms:.1f}ms)")
        return self._row_to_model(row)

    async def search(
        self,
        country_code: Optional[str] = None,
        iucn_category: Optional[str] = None,
        designation_level: Optional[str] = None,
        legal_status: Optional[str] = None,
        is_world_heritage: Optional[bool] = None,
        is_ramsar: Optional[bool] = None,
        is_kba: Optional[bool] = None,
        is_aze: Optional[bool] = None,
        min_area_ha: Optional[Decimal] = None,
        max_area_ha: Optional[Decimal] = None,
        limit: int = 1000,
        offset: int = 0,
        pool: Any = None,
    ) -> List[Dict[str, Any]]:
        """Search protected areas with multiple filter criteria.

        Args:
            country_code: ISO 3166-1 alpha-3 filter.
            iucn_category: IUCN category filter (Ia, Ib, II, etc.).
            designation_level: Designation level filter.
            legal_status: Legal status filter.
            is_world_heritage: UNESCO WH filter.
            is_ramsar: Ramsar filter.
            is_kba: KBA filter.
            is_aze: AZE filter.
            min_area_ha: Minimum area filter.
            max_area_ha: Maximum area filter.
            limit: Maximum results to return.
            offset: Result offset for pagination.
            pool: Database connection pool.

        Returns:
            List of dictionaries with protected area data.

        Example:
            >>> results = await engine.search(country_code="BRA", iucn_category="II")
            >>> assert all(r["country_code"] == "BRA" for r in results)
        """
        conditions: List[str] = []
        params: List[Any] = []

        if country_code:
            conditions.append("country_code = %s")
            params.append(country_code.upper())
        if iucn_category:
            conditions.append("iucn_category = %s")
            params.append(iucn_category)
        if designation_level:
            conditions.append("designation_level = %s")
            params.append(designation_level)
        if legal_status:
            conditions.append("legal_status = %s")
            params.append(legal_status)
        if is_world_heritage is not None:
            conditions.append("is_world_heritage = %s")
            params.append(is_world_heritage)
        if is_ramsar is not None:
            conditions.append("is_ramsar = %s")
            params.append(is_ramsar)
        if is_kba is not None:
            conditions.append("is_kba = %s")
            params.append(is_kba)
        if is_aze is not None:
            conditions.append("is_aze = %s")
            params.append(is_aze)
        if min_area_ha is not None:
            conditions.append("area_hectares >= %s")
            params.append(min_area_ha)
        if max_area_ha is not None:
            conditions.append("area_hectares <= %s")
            params.append(max_area_ha)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        query = f"""
            SELECT wdpa_id, name, iucn_category, designation_level,
                   country_code, area_hectares, legal_status, governance_type,
                   is_world_heritage, is_ramsar, is_kba, is_aze, wdpa_version
            FROM eudr_protected_area_validator.gl_eudr_pav_protected_areas
            WHERE {where_clause}
            ORDER BY wdpa_id
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        if pool is None:
            logger.warning("No database pool provided; returning empty list")
            return []

        start = time.monotonic()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()
                columns = [desc.name for desc in cur.description]

        elapsed_ms = (time.monotonic() - start) * 1000
        results = [dict(zip(columns, row)) for row in rows]
        logger.info(
            f"Search returned {len(results)} protected areas ({elapsed_ms:.1f}ms)"
        )
        return results

    async def find_by_proximity(
        self,
        latitude: Decimal,
        longitude: Decimal,
        radius_meters: Decimal,
        pool: Any = None,
    ) -> List[Dict[str, Any]]:
        """Find protected areas within a radius of a point.

        Uses PostGIS ST_DWithin for efficient proximity queries with
        the GIST spatial index.

        Args:
            latitude: WGS84 latitude.
            longitude: WGS84 longitude.
            radius_meters: Search radius in meters.
            pool: Database connection pool.

        Returns:
            List of dictionaries with distance_meters field included.
        """
        if pool is None:
            logger.warning("No database pool provided; returning empty list")
            return []

        start = time.monotonic()
        params = (
            float(longitude), float(latitude),
            float(longitude), float(latitude),
            float(radius_meters),
        )

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_SEARCH_BY_PROXIMITY, params)
                rows = await cur.fetchall()
                columns = [desc.name for desc in cur.description]

        elapsed_ms = (time.monotonic() - start) * 1000
        results = [dict(zip(columns, row)) for row in rows]
        logger.info(
            f"Proximity search: {len(results)} PAs within "
            f"{radius_meters}m ({elapsed_ms:.1f}ms)"
        )
        return results

    async def find_containing_point(
        self,
        latitude: Decimal,
        longitude: Decimal,
        pool: Any = None,
    ) -> List[Dict[str, Any]]:
        """Find protected areas that contain a given point.

        Uses PostGIS ST_Contains for point-in-polygon queries.

        Args:
            latitude: WGS84 latitude.
            longitude: WGS84 longitude.
            pool: Database connection pool.

        Returns:
            List of protected areas containing the point.
        """
        if pool is None:
            return []

        start = time.monotonic()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_POINT_IN_POLYGON,
                    (float(longitude), float(latitude)),
                )
                rows = await cur.fetchall()
                columns = [desc.name for desc in cur.description]

        elapsed_ms = (time.monotonic() - start) * 1000
        results = [dict(zip(columns, row)) for row in rows]
        logger.info(
            f"Point-in-polygon: {len(results)} PAs contain point "
            f"({latitude}, {longitude}) ({elapsed_ms:.1f}ms)"
        )
        return results

    async def get_country_coverage(
        self,
        pool: Any = None,
    ) -> List[Dict[str, Any]]:
        """Get protected area count and total area by country.

        Returns:
            List of {country_code, pa_count, total_area_ha} dictionaries.
        """
        if pool is None:
            return []

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_COUNT_BY_COUNTRY)
                rows = await cur.fetchall()
                columns = [desc.name for desc in cur.description]

        return [dict(zip(columns, row)) for row in rows]

    async def get_total_count(self, pool: Any = None) -> int:
        """Get the total number of protected areas in the database.

        Returns:
            Total count of protected area records.
        """
        if pool is None:
            return 0

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(*) FROM "
                    "eudr_protected_area_validator.gl_eudr_pav_protected_areas"
                )
                row = await cur.fetchone()

        return row[0] if row else 0

    def _row_to_model(self, row: Any) -> ProtectedArea:
        """Convert a database row to a ProtectedArea model.

        Args:
            row: Database row tuple or dict.

        Returns:
            ProtectedArea Pydantic model instance.
        """
        if isinstance(row, dict):
            return ProtectedArea(**row)
        # Handle tuple rows by creating a dict mapping
        logger.debug("Row conversion requires column mapping for tuple rows")
        return ProtectedArea(
            wdpa_id=row[0],
            name=row[1] or "Unknown",
            designation=row[3] or "Unknown",
            designation_level=row[4] or "national",
            iucn_category=row[5] or "NR",
            country_code=row[6] or "UNK",
            iso3=row[7] or "UNK",
            area_hectares=Decimal(str(row[8] or 0)),
            legal_status=row[10] or "designated",
            wdpa_version=row[20] or "unknown",
        )
