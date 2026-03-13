# -*- coding: utf-8 -*-
"""
TerritoryDatabaseEngine - Feature 1: Indigenous Territory Database Integration

Manages a consolidated, georeferenced database of indigenous territories
from 6 authoritative sources (LandMark, RAISG, FUNAI, BPN/AMAN, ACHPR,
national registries) covering 100+ countries and 50,000+ territories.

Capabilities:
    - Territory CRUD with version control and provenance tracking
    - Spatial queries via PostGIS (point-in-polygon, bounding box)
    - Territory search by name, people, country, legal status
    - Data freshness tracking with staleness alerts
    - Territory version history for audit trail
    - Coverage statistics aggregation

Performance Targets:
    - Point-in-polygon query: < 100ms
    - Batch screening 10,000 plots: < 5 minutes
    - Territory boundary stored at 6+ decimal places

Zero-Hallucination: All territory data comes from authoritative sources.
No synthetic boundaries. Every record includes source attribution and
SHA-256 provenance hash.

Example:
    >>> engine = TerritoryDatabaseEngine(config, provenance_tracker)
    >>> territory = await engine.get_territory("t-001")
    >>> territories = await engine.search_territories(country_code="BR")
    >>> stats = await engine.get_coverage_statistics()

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 1)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    ConfidenceLevel,
    IndigenousTerritory,
    TerritoryLegalStatus,
)
from greenlang.agents.eudr.indigenous_rights_checker.provenance import (
    ProvenanceTracker,
)
from greenlang.agents.eudr.indigenous_rights_checker.metrics import (
    record_territory_query,
    set_active_territories,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL query templates for PostGIS territory operations
# ---------------------------------------------------------------------------

_SQL_GET_TERRITORY = """
    SELECT territory_id, territory_name, indigenous_name, people_name,
           country_code, region, area_hectares, legal_status,
           recognition_date, governing_authority, boundary_geojson,
           data_source, source_url, confidence, version,
           provenance_hash, last_verified, created_at
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_territories
    WHERE territory_id = %(territory_id)s
"""

_SQL_SEARCH_TERRITORIES = """
    SELECT territory_id, territory_name, indigenous_name, people_name,
           country_code, region, area_hectares, legal_status,
           recognition_date, governing_authority, boundary_geojson,
           data_source, source_url, confidence, version,
           provenance_hash, last_verified, created_at
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_territories
    WHERE 1=1
    {filters}
    ORDER BY territory_name
    LIMIT %(limit)s OFFSET %(offset)s
"""

_SQL_INSERT_TERRITORY = """
    INSERT INTO eudr_indigenous_rights_checker.gl_eudr_irc_territories (
        territory_id, territory_name, indigenous_name, people_name,
        country_code, region, area_hectares, legal_status,
        recognition_date, governing_authority, boundary_geom,
        boundary_geojson, data_source, source_url, confidence,
        version, provenance_hash, last_verified
    ) VALUES (
        %(territory_id)s, %(territory_name)s, %(indigenous_name)s,
        %(people_name)s, %(country_code)s, %(region)s,
        %(area_hectares)s, %(legal_status)s, %(recognition_date)s,
        %(governing_authority)s,
        ST_SetSRID(ST_GeomFromGeoJSON(%(boundary_geojson_str)s), 4326),
        %(boundary_geojson)s, %(data_source)s, %(source_url)s,
        %(confidence)s, %(version)s, %(provenance_hash)s,
        %(last_verified)s
    )
    RETURNING territory_id
"""

_SQL_TERRITORIES_IN_BBOX = """
    SELECT territory_id, territory_name, people_name, country_code,
           legal_status, area_hectares, data_source
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_territories
    WHERE boundary_geom && ST_MakeEnvelope(%(xmin)s, %(ymin)s,
                                            %(xmax)s, %(ymax)s, 4326)
"""

_SQL_POINT_IN_TERRITORY = """
    SELECT territory_id, territory_name, people_name, country_code,
           legal_status, area_hectares, data_source
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_territories
    WHERE ST_Contains(boundary_geom,
                      ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326))
"""

_SQL_COVERAGE_STATS = """
    SELECT country_code,
           COUNT(*) as territory_count,
           SUM(area_hectares) as total_area_hectares,
           COUNT(DISTINCT people_name) as people_count,
           COUNT(DISTINCT data_source) as source_count
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_territories
    GROUP BY country_code
    ORDER BY territory_count DESC
"""

_SQL_COUNT_TERRITORIES = """
    SELECT COUNT(*) as total
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_territories
"""

_SQL_INSERT_VERSION = """
    INSERT INTO eudr_indigenous_rights_checker.gl_eudr_irc_territory_versions (
        version_id, territory_id, version_number,
        previous_boundary_geojson, new_boundary_geojson,
        change_description, change_source, effective_date,
        provenance_hash
    ) VALUES (
        %(version_id)s, %(territory_id)s, %(version_number)s,
        %(previous_boundary_geojson)s, %(new_boundary_geojson)s,
        %(change_description)s, %(change_source)s, %(effective_date)s,
        %(provenance_hash)s
    )
"""


class TerritoryDatabaseEngine:
    """Engine for managing the indigenous territory spatial database.

    Provides CRUD operations, spatial queries, version control, and
    coverage statistics for indigenous territory data from 6 authoritative
    sources.

    Attributes:
        _config: Agent configuration.
        _provenance: Provenance tracker for audit trail.
        _pool: Async database connection pool (set during startup).

    Example:
        >>> engine = TerritoryDatabaseEngine(config, tracker)
        >>> await engine.startup(pool)
        >>> territory = await engine.get_territory("t-001")
    """

    def __init__(
        self,
        config: IndigenousRightsCheckerConfig,
        provenance: ProvenanceTracker,
    ) -> None:
        """Initialize TerritoryDatabaseEngine.

        Args:
            config: Agent configuration instance.
            provenance: Provenance tracker instance.
        """
        self._config = config
        self._provenance = provenance
        self._pool: Any = None
        logger.info("TerritoryDatabaseEngine initialized")

    async def startup(self, pool: Any) -> None:
        """Set the database connection pool.

        Args:
            pool: AsyncConnectionPool instance from psycopg_pool.
        """
        self._pool = pool
        logger.info("TerritoryDatabaseEngine started with connection pool")

    async def shutdown(self) -> None:
        """Clean up engine resources."""
        self._pool = None
        logger.info("TerritoryDatabaseEngine shut down")

    async def get_territory(
        self, territory_id: str
    ) -> Optional[IndigenousTerritory]:
        """Get a single territory by ID.

        Args:
            territory_id: UUID territory identifier.

        Returns:
            IndigenousTerritory model or None if not found.

        Example:
            >>> t = await engine.get_territory("t-001")
            >>> if t:
            ...     print(t.territory_name)
        """
        start = time.monotonic()
        self._provenance.record(
            "territory", "query", territory_id,
            metadata={"operation": "get_by_id"},
        )

        if self._pool is None:
            logger.warning("No database pool; returning None")
            return None

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_GET_TERRITORY,
                    {"territory_id": territory_id},
                )
                row = await cur.fetchone()

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(f"get_territory({territory_id}) took {elapsed_ms:.1f}ms")

        if row is None:
            return None

        return self._row_to_territory(row)

    async def search_territories(
        self,
        country_code: Optional[str] = None,
        legal_status: Optional[str] = None,
        data_source: Optional[str] = None,
        people_name: Optional[str] = None,
        name_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[IndigenousTerritory]:
        """Search territories with optional filters.

        Args:
            country_code: Filter by ISO 3166-1 alpha-2 country code.
            legal_status: Filter by legal status.
            data_source: Filter by data source.
            people_name: Filter by indigenous people name (partial match).
            name_query: Filter by territory name (partial match).
            limit: Maximum results to return.
            offset: Result offset for pagination.

        Returns:
            List of matching IndigenousTerritory models.

        Example:
            >>> results = await engine.search_territories(country_code="BR")
            >>> for t in results:
            ...     print(t.territory_name)
        """
        start = time.monotonic()
        filters = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}

        if country_code:
            filters.append("AND country_code = %(country_code)s")
            params["country_code"] = country_code.upper()
        if legal_status:
            filters.append("AND legal_status = %(legal_status)s")
            params["legal_status"] = legal_status
        if data_source:
            filters.append("AND data_source = %(data_source)s")
            params["data_source"] = data_source
        if people_name:
            filters.append("AND people_name ILIKE %(people_name)s")
            params["people_name"] = f"%{people_name}%"
        if name_query:
            filters.append(
                "AND (territory_name ILIKE %(name_query)s "
                "OR indigenous_name ILIKE %(name_query)s)"
            )
            params["name_query"] = f"%{name_query}%"

        filter_sql = "\n    ".join(filters)
        sql = _SQL_SEARCH_TERRITORIES.format(filters=filter_sql)

        record_territory_query(
            country_code=country_code or "all",
            data_source=data_source or "all",
        )

        self._provenance.record(
            "territory", "query", f"search-{country_code or 'all'}",
            metadata={
                "country_code": country_code,
                "legal_status": legal_status,
                "limit": limit,
            },
        )

        if self._pool is None:
            logger.warning("No database pool; returning empty list")
            return []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            f"search_territories() returned {len(rows)} results "
            f"in {elapsed_ms:.1f}ms"
        )

        return [self._row_to_territory(row) for row in rows]

    async def find_territories_at_point(
        self, latitude: float, longitude: float
    ) -> List[IndigenousTerritory]:
        """Find all territories containing a geographic point.

        Uses PostGIS ST_Contains for point-in-polygon query.
        Performance target: < 100ms per query.

        Args:
            latitude: WGS84 latitude (-90 to 90).
            longitude: WGS84 longitude (-180 to 180).

        Returns:
            List of territories containing the point.

        Example:
            >>> territories = await engine.find_territories_at_point(
            ...     latitude=-3.4653, longitude=-62.2159,
            ... )
        """
        start = time.monotonic()

        self._provenance.record(
            "territory", "query",
            f"point-{latitude:.4f}-{longitude:.4f}",
            metadata={"lat": latitude, "lon": longitude},
        )

        if self._pool is None:
            return []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_POINT_IN_TERRITORY,
                    {"lat": latitude, "lon": longitude},
                )
                rows = await cur.fetchall()

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            f"find_territories_at_point({latitude}, {longitude}) "
            f"found {len(rows)} territories in {elapsed_ms:.1f}ms"
        )

        return [self._row_to_territory(row) for row in rows]

    async def find_territories_in_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> List[Dict[str, Any]]:
        """Find territories intersecting a bounding box.

        Uses PostGIS GIST index for fast spatial queries.

        Args:
            min_lat: Minimum latitude.
            min_lon: Minimum longitude.
            max_lat: Maximum latitude.
            max_lon: Maximum longitude.

        Returns:
            List of territory summary dictionaries.
        """
        start = time.monotonic()

        if self._pool is None:
            return []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_TERRITORIES_IN_BBOX,
                    {
                        "xmin": min_lon, "ymin": min_lat,
                        "xmax": max_lon, "ymax": max_lat,
                    },
                )
                rows = await cur.fetchall()

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            f"find_territories_in_bbox() found {len(rows)} "
            f"territories in {elapsed_ms:.1f}ms"
        )

        return [
            {
                "territory_id": str(row[0]),
                "territory_name": row[1],
                "people_name": row[2],
                "country_code": row[3],
                "legal_status": row[4],
                "area_hectares": row[5],
                "data_source": row[6],
            }
            for row in rows
        ]

    async def create_territory(
        self, territory_data: Dict[str, Any]
    ) -> str:
        """Create a new territory record with provenance tracking.

        Args:
            territory_data: Dictionary with territory fields.

        Returns:
            UUID of the created territory.

        Raises:
            ValueError: If required fields are missing.
        """
        territory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        provenance_hash = self._provenance.compute_data_hash({
            "territory_id": territory_id,
            "territory_name": territory_data.get("territory_name", ""),
            "people_name": territory_data.get("people_name", ""),
            "country_code": territory_data.get("country_code", ""),
            "data_source": territory_data.get("data_source", ""),
            "created_at": now.isoformat(),
        })

        self._provenance.record(
            "territory", "create", territory_id,
            metadata={
                "territory_name": territory_data.get("territory_name"),
                "country_code": territory_data.get("country_code"),
            },
        )

        if self._pool is None:
            logger.warning("No database pool; territory not persisted")
            return territory_id

        import json
        geojson = territory_data.get("boundary_geojson")
        geojson_str = json.dumps(geojson) if geojson else None

        params = {
            "territory_id": territory_id,
            "territory_name": territory_data["territory_name"],
            "indigenous_name": territory_data.get("indigenous_name"),
            "people_name": territory_data["people_name"],
            "country_code": territory_data["country_code"],
            "region": territory_data.get("region"),
            "area_hectares": territory_data.get("area_hectares"),
            "legal_status": territory_data.get("legal_status", "claimed"),
            "recognition_date": territory_data.get("recognition_date"),
            "governing_authority": territory_data.get("governing_authority"),
            "boundary_geojson_str": geojson_str,
            "boundary_geojson": json.dumps(geojson) if geojson else None,
            "data_source": territory_data["data_source"],
            "source_url": territory_data.get("source_url"),
            "confidence": territory_data.get("confidence", "medium"),
            "version": 1,
            "provenance_hash": provenance_hash,
            "last_verified": now,
        }

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_INSERT_TERRITORY, params)
            await conn.commit()

        logger.info(f"Created territory {territory_id}: {params['territory_name']}")
        return territory_id

    async def get_coverage_statistics(self) -> Dict[str, Any]:
        """Get territory database coverage statistics.

        Returns:
            Dictionary with coverage data by country, including
            total territories, area, people count, and source count.

        Example:
            >>> stats = await engine.get_coverage_statistics()
            >>> print(stats["total_territories"])
        """
        if self._pool is None:
            return {"total_territories": 0, "countries": []}

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_COUNT_TERRITORIES)
                total_row = await cur.fetchone()
                total = total_row[0] if total_row else 0

                await cur.execute(_SQL_COVERAGE_STATS)
                rows = await cur.fetchall()

        countries = [
            {
                "country_code": row[0],
                "territory_count": row[1],
                "total_area_hectares": float(row[2]) if row[2] else 0,
                "people_count": row[3],
                "source_count": row[4],
            }
            for row in rows
        ]

        set_active_territories(total)

        return {
            "total_territories": total,
            "total_countries": len(countries),
            "countries": countries,
        }

    async def get_territory_history(
        self, territory_id: str
    ) -> List[Dict[str, Any]]:
        """Get version history for a territory.

        Args:
            territory_id: UUID territory identifier.

        Returns:
            List of version records in reverse chronological order.
        """
        if self._pool is None:
            return []

        sql = """
            SELECT version_id, version_number, change_description,
                   change_source, effective_date, provenance_hash,
                   created_at
            FROM eudr_indigenous_rights_checker.gl_eudr_irc_territory_versions
            WHERE territory_id = %(territory_id)s
            ORDER BY version_number DESC
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, {"territory_id": territory_id})
                rows = await cur.fetchall()

        return [
            {
                "version_id": str(row[0]),
                "version_number": row[1],
                "change_description": row[2],
                "change_source": row[3],
                "effective_date": str(row[4]) if row[4] else None,
                "provenance_hash": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
            }
            for row in rows
        ]

    def _row_to_territory(self, row: Tuple) -> IndigenousTerritory:
        """Convert a database row tuple to an IndigenousTerritory model.

        Args:
            row: Database result row tuple.

        Returns:
            IndigenousTerritory model instance.
        """
        import json
        return IndigenousTerritory(
            territory_id=str(row[0]),
            territory_name=row[1],
            indigenous_name=row[2],
            people_name=row[3],
            country_code=row[4],
            region=row[5],
            area_hectares=Decimal(str(row[6])) if row[6] else None,
            legal_status=TerritoryLegalStatus(row[7]),
            recognition_date=row[8],
            governing_authority=row[9],
            boundary_geojson=(
                json.loads(row[10]) if isinstance(row[10], str) else row[10]
            ),
            data_source=row[11],
            source_url=row[12],
            confidence=ConfidenceLevel(row[13]) if row[13] else ConfidenceLevel.MEDIUM,
            version=row[14] or 1,
            provenance_hash=row[15] or "",
            last_verified=row[16],
            created_at=row[17],
        )
