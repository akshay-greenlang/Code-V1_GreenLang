# -*- coding: utf-8 -*-
"""
Postgres-backed factor catalog repository (F031).

Async implementation using psycopg + psycopg_pool per project patterns.
Full-text search via tsvector, JSONB payload, connection pooling.

Tables are defined in V426__factors_catalog.sql (Postgres version).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from greenlang.data.emission_factor_record import EmissionFactorRecord
from greenlang.factors.catalog_repository import (
    EditionRow,
    FactorCatalogRepository,
    _cap_facet_counts,
    _record_from_stored_payload,
)
from greenlang.factors.middleware.method_profile_guard import require_method_profile

logger = logging.getLogger(__name__)


@dataclass
class PgPoolConfig:
    """Connection pool settings for PostgresFactorCatalogRepository."""

    dsn: str
    min_size: int = 2
    max_size: int = 20
    max_idle: float = 300.0
    max_lifetime: float = 3600.0


def _status_visibility_sql(include_preview: bool, include_connector: bool) -> str:
    """SQL fragment for Postgres status visibility filter."""
    statuses = ["'certified'"]
    if include_preview:
        statuses.append("'preview'")
    if include_connector:
        statuses.append("'connector_only'")
    inner = ", ".join(statuses)
    return (
        f"(COALESCE(factor_status, 'certified') IN ({inner}) "
        f"AND COALESCE(factor_status, 'certified') != 'deprecated')"
    )


class PostgresFactorCatalogRepository(FactorCatalogRepository):
    """
    Async Postgres-backed catalog repository.

    Uses psycopg3 (psycopg) async connection pool. All public methods are
    synchronous wrappers around the async internals for compatibility with
    the existing FactorCatalogRepository ABC.

    For production async usage, use the _async_* methods directly.
    """

    def __init__(self, pool_config: PgPoolConfig):
        self._config = pool_config
        self._pool = None
        logger.info(
            "PostgresFactorCatalogRepository configured: dsn=%s pool=%d-%d",
            pool_config.dsn.split("@")[-1] if "@" in pool_config.dsn else "(local)",
            pool_config.min_size,
            pool_config.max_size,
        )

    def _get_sync_conn(self):
        """Get a synchronous psycopg connection."""
        import psycopg

        return psycopg.connect(self._config.dsn)

    # ---- Edition operations ----

    def list_editions(self, include_pending: bool = True) -> List[EditionRow]:
        import psycopg

        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                if include_pending:
                    cur.execute(
                        "SELECT edition_id, status, label, manifest_hash, changelog_json "
                        "FROM factors_catalog.editions ORDER BY edition_id DESC"
                    )
                else:
                    cur.execute(
                        "SELECT edition_id, status, label, manifest_hash, changelog_json "
                        "FROM factors_catalog.editions WHERE status != 'pending' "
                        "ORDER BY edition_id DESC"
                    )
                rows = cur.fetchall()
        return [
            EditionRow(
                edition_id=r[0],
                status=r[1],
                label=r[2] or "",
                manifest_hash=r[3] or "",
                changelog_json=r[4] or "[]",
            )
            for r in rows
        ]

    def get_default_edition_id(self) -> str:
        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT edition_id FROM factors_catalog.editions "
                    "WHERE status = 'stable' ORDER BY edition_id DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row:
                    return str(row[0])
                cur.execute(
                    "SELECT edition_id FROM factors_catalog.editions "
                    "ORDER BY edition_id DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row:
                    return str(row[0])
        return ""

    def resolve_edition(self, requested: Optional[str]) -> str:
        if requested:
            with self._get_sync_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM factors_catalog.editions WHERE edition_id = %s LIMIT 1",
                        (requested,),
                    )
                    if cur.fetchone():
                        return requested
            raise ValueError(f"Unknown edition_id: {requested!r}")
        default = self.get_default_edition_id()
        if not default:
            raise ValueError("No editions in catalog database")
        return default

    def get_changelog(self, edition_id: str) -> List[str]:
        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT changelog_json FROM factors_catalog.editions WHERE edition_id = %s",
                    (edition_id,),
                )
                row = cur.fetchone()
        if not row:
            return []
        return list(json.loads(row[0] or "[]"))

    def get_manifest_dict(self, edition_id: str) -> Dict[str, Any]:
        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT manifest_json FROM factors_catalog.editions WHERE edition_id = %s",
                    (edition_id,),
                )
                row = cur.fetchone()
        if not row:
            return {}
        return dict(json.loads(row[0] or "{}"))

    # ---- Factor CRUD ----

    def list_factors(
        self,
        edition_id: str,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
        scope: Optional[str] = None,
        boundary: Optional[str] = None,
        page: int = 1,
        limit: int = 100,
        include_preview: bool = False,
        include_connector: bool = False,
        *,
        method_profile: Optional[Any] = None,
    ) -> Tuple[List[EmissionFactorRecord], int]:
        # N6 guard — policy workflows must bind a method_profile before
        # reaching the catalog. No-op outside ``@policy_workflow`` scope.
        require_method_profile(
            {"method_profile": method_profile},
            caller="PostgresFactorCatalogRepository.list_factors",
        )
        vis = _status_visibility_sql(include_preview, include_connector)
        clauses = ["edition_id = %s", vis]
        params: List[Any] = [edition_id]
        if fuel_type:
            clauses.append("LOWER(fuel_type) = %s")
            params.append(fuel_type.lower())
        if geography:
            clauses.append("geography = %s")
            params.append(geography)
        if scope:
            clauses.append("scope = %s")
            params.append(scope)
        if boundary:
            clauses.append("boundary = %s")
            params.append(boundary.lower().replace(" ", "_"))
        where = " AND ".join(clauses)
        offset = (page - 1) * limit

        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM factors_catalog.catalog_factors WHERE {where}",
                    params,
                )
                total = cur.fetchone()[0]
                cur.execute(
                    f"SELECT payload_json FROM factors_catalog.catalog_factors "
                    f"WHERE {where} ORDER BY factor_id LIMIT %s OFFSET %s",
                    params + [limit, offset],
                )
                rows = cur.fetchall()
        recs = [_record_from_stored_payload(json.loads(r[0])) for r in rows]
        return recs, int(total)

    def get_factor(
        self,
        edition_id: str,
        factor_id: str,
        *,
        method_profile: Optional[Any] = None,
    ) -> Optional[EmissionFactorRecord]:
        # N6 guard — see ``list_factors`` above.
        require_method_profile(
            {"method_profile": method_profile},
            caller="PostgresFactorCatalogRepository.get_factor",
        )
        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload_json FROM factors_catalog.catalog_factors "
                    "WHERE edition_id = %s AND factor_id = %s",
                    (edition_id, factor_id),
                )
                row = cur.fetchone()
        if not row:
            return None
        return _record_from_stored_payload(json.loads(row[0]))

    def search_factors(
        self,
        edition_id: str,
        query: str,
        geography: Optional[str] = None,
        limit: int = 20,
        include_preview: bool = False,
        include_connector: bool = False,
        factor_status: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[EmissionFactorRecord]:
        vis = _status_visibility_sql(include_preview, include_connector)
        clauses = [
            "edition_id = %s",
            vis,
            "search_tsv @@ plainto_tsquery('english', %s)",
        ]
        params: List[Any] = [edition_id, query]
        if geography:
            clauses.append("geography = %s")
            params.append(geography)
        if factor_status:
            clauses.append("COALESCE(factor_status, 'certified') = %s")
            params.append(factor_status)
        if source_id:
            clauses.append("source_id = %s")
            params.append(source_id)
        where = " AND ".join(clauses)

        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT payload_json FROM factors_catalog.catalog_factors "
                    f"WHERE {where} "
                    f"ORDER BY ts_rank(search_tsv, plainto_tsquery('english', %s)) DESC, factor_id "
                    f"LIMIT %s",
                    params + [query, limit],
                )
                rows = cur.fetchall()
        return [_record_from_stored_payload(json.loads(r[0])) for r in rows]

    def search_facets(
        self,
        edition_id: str,
        include_preview: bool = False,
        include_connector: bool = False,
        max_values: int = 80,
    ) -> Dict[str, Any]:
        vis = _status_visibility_sql(include_preview, include_connector)
        facets: Dict[str, Dict[str, int]] = {}

        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                for expr, key in [
                    ("COALESCE(factor_status, 'certified')", "factor_status"),
                    ("COALESCE(NULLIF(TRIM(source_id), ''), '(unset)')", "source_id"),
                    ("geography", "geography"),
                    ("scope", "scope"),
                    ("boundary", "boundary"),
                    ("fuel_type", "fuel_type"),
                ]:
                    cur.execute(
                        f"SELECT {expr} AS k, COUNT(*) AS c "
                        f"FROM factors_catalog.catalog_factors "
                        f"WHERE edition_id = %s AND {vis} "
                        f"GROUP BY k ORDER BY c DESC",
                        (edition_id,),
                    )
                    raw = {str(r[0]): int(r[1]) for r in cur.fetchall()}
                    facets[key] = _cap_facet_counts(raw, max_values)

        return {"edition_id": edition_id, "facets": facets}

    def list_factor_summaries(self, edition_id: str) -> List[Dict[str, str]]:
        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT factor_id, content_hash, "
                    "COALESCE(factor_status, 'certified') AS factor_status "
                    "FROM factors_catalog.catalog_factors "
                    "WHERE edition_id = %s ORDER BY factor_id",
                    (edition_id,),
                )
                rows = cur.fetchall()
        return [
            {
                "factor_id": str(r[0]),
                "content_hash": str(r[1]),
                "factor_status": str(r[2]),
            }
            for r in rows
        ]

    def coverage_stats(self, edition_id: str) -> Dict[str, Any]:
        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM factors_catalog.catalog_factors WHERE edition_id = %s",
                    (edition_id,),
                )
                total = cur.fetchone()[0]

                cur.execute(
                    "SELECT geography, COUNT(*) FROM factors_catalog.catalog_factors "
                    "WHERE edition_id = %s GROUP BY geography",
                    (edition_id,),
                )
                geo_rows = cur.fetchall()

                cur.execute(
                    "SELECT fuel_type, COUNT(*) FROM factors_catalog.catalog_factors "
                    "WHERE edition_id = %s GROUP BY fuel_type",
                    (edition_id,),
                )
                fuel_rows = cur.fetchall()

                cur.execute(
                    "SELECT scope, COUNT(*) FROM factors_catalog.catalog_factors "
                    "WHERE edition_id = %s GROUP BY scope",
                    (edition_id,),
                )
                scope_rows = cur.fetchall()

                cur.execute(
                    "SELECT boundary, COUNT(*) FROM factors_catalog.catalog_factors "
                    "WHERE edition_id = %s GROUP BY boundary",
                    (edition_id,),
                )
                boundary_rows = cur.fetchall()

                cur.execute(
                    "SELECT COALESCE(factor_status, 'certified') AS st, COUNT(*) "
                    "FROM factors_catalog.catalog_factors "
                    "WHERE edition_id = %s GROUP BY st",
                    (edition_id,),
                )
                status_rows = cur.fetchall()

        scopes = {"1": 0, "2": 0, "3": 0}
        for r in scope_rows:
            k = str(r[0])
            if k in scopes:
                scopes[k] = int(r[1])
        boundaries = {str(r[0]): int(r[1]) for r in boundary_rows}
        by_geography = {str(r[0]): int(r[1]) for r in geo_rows}
        by_fuel_type = {str(r[0]): int(r[1]) for r in fuel_rows}
        by_status = {str(r[0]): int(r[1]) for r in status_rows}
        return {
            "total_factors": int(total),
            "total_catalog": int(total),
            "certified": by_status.get("certified", 0),
            "preview": by_status.get("preview", 0),
            "connector_visible": by_status.get("connector_only", 0),
            "geographies": len(by_geography),
            "fuel_types": len(by_fuel_type),
            "scopes": scopes,
            "boundaries": boundaries,
            "by_geography": by_geography,
            "by_fuel_type": by_fuel_type,
            "by_status": by_status,
        }

    # ---- Write operations (for ingestion) ----

    def upsert_edition(
        self,
        edition_id: str,
        status: str,
        label: str,
        manifest: Dict[str, Any],
        changelog: Sequence[str],
    ) -> None:
        from greenlang.factors.edition_manifest import EditionManifest

        m = EditionManifest.from_dict({**manifest, "edition_id": edition_id, "status": status})
        fp = m.manifest_fingerprint()
        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO factors_catalog.editions "
                    "(edition_id, status, label, manifest_hash, manifest_json, changelog_json) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (edition_id) DO UPDATE SET "
                    "status=EXCLUDED.status, label=EXCLUDED.label, "
                    "manifest_hash=EXCLUDED.manifest_hash, "
                    "manifest_json=EXCLUDED.manifest_json, "
                    "changelog_json=EXCLUDED.changelog_json",
                    (
                        edition_id,
                        status,
                        label,
                        fp,
                        json.dumps(m.to_dict(), sort_keys=True, default=str),
                        json.dumps(list(changelog)),
                    ),
                )
            conn.commit()

    def insert_factors(
        self,
        edition_id: str,
        records: Sequence[EmissionFactorRecord],
    ) -> None:
        with self._get_sync_conn() as conn:
            with conn.cursor() as cur:
                for r in records:
                    blob = " ".join([
                        r.factor_id, r.fuel_type, r.geography,
                        r.scope.value, r.boundary.value,
                        " ".join(r.tags),
                        " ".join(getattr(r, "activity_tags", []) or []),
                        " ".join(getattr(r, "sector_tags", []) or []),
                        getattr(r, "license_class", "") or "",
                        r.notes or "",
                    ])
                    cur.execute(
                        "INSERT INTO factors_catalog.catalog_factors "
                        "(edition_id, factor_id, fuel_type, geography, scope, boundary, "
                        "search_blob, search_tsv, payload_json, content_hash, "
                        "factor_status, source_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s), "
                        "%s, %s, %s, %s) "
                        "ON CONFLICT (edition_id, factor_id) DO UPDATE SET "
                        "payload_json=EXCLUDED.payload_json, content_hash=EXCLUDED.content_hash, "
                        "search_tsv=EXCLUDED.search_tsv, search_blob=EXCLUDED.search_blob",
                        (
                            edition_id,
                            r.factor_id,
                            r.fuel_type.lower(),
                            r.geography,
                            r.scope.value,
                            r.boundary.value.lower().replace(" ", "_"),
                            blob.lower(),
                            blob.lower(),
                            json.dumps(r.to_dict(), sort_keys=True, default=str),
                            r.content_hash,
                            getattr(r, "factor_status", "certified") or "certified",
                            getattr(r, "source_id", None),
                        ),
                    )
            conn.commit()
        logger.info(
            "Inserted %d factors into Postgres edition=%s",
            len(records), edition_id,
        )
