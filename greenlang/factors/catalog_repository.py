# -*- coding: utf-8 -*-
"""
Factor catalog repository: SQLite (dev/single-node) + in-memory adapter.

Postgres operators can mirror the SQL in deployment/database/migrations/sql/V426__factors_catalog.sql.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from greenlang.data.emission_factor_record import EmissionFactorRecord

logger = logging.getLogger(__name__)


def _record_from_stored_payload(payload: Dict[str, Any]) -> EmissionFactorRecord:
    """Strip dataclass-only keys (e.g. GHGVectors class constants) from stored JSON."""
    data = dict(payload)
    vec = dict(data.get("vectors") or {})
    allowed = (
        "CO2",
        "CH4",
        "N2O",
        "HFCs",
        "PFCs",
        "SF6",
        "NF3",
        "biogenic_CO2",
    )
    data["vectors"] = {k: vec[k] for k in allowed if k in vec}
    dq = data.get("dqs")
    if isinstance(dq, dict):
        data["dqs"] = {
            k: dq[k]
            for k in (
                "temporal",
                "geographical",
                "technological",
                "representativeness",
                "methodological",
            )
            if k in dq
        }
    for gwp_key in ("gwp_100yr", "gwp_20yr"):
        gwp = data.get(gwp_key)
        if isinstance(gwp, dict):
            gwp = dict(gwp)
            gwp.pop("co2e_total", None)
            data[gwp_key] = gwp
    prov = data.get("provenance")
    if isinstance(prov, dict):
        prov = dict(prov)
        prov.pop("citation", None)
        data["provenance"] = prov
    data.pop("content_hash", None)
    return EmissionFactorRecord.from_dict(data)


@dataclass
class EditionRow:
    edition_id: str
    status: str
    label: str
    manifest_hash: str
    changelog_json: str


class FactorCatalogRepository(ABC):
    """Abstract catalog access for Factors API."""

    @abstractmethod
    def list_editions(self, include_pending: bool = True) -> List[EditionRow]:
        raise NotImplementedError

    @abstractmethod
    def get_default_edition_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def resolve_edition(self, requested: Optional[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_changelog(self, edition_id: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_manifest_dict(self, edition_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
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
    ) -> Tuple[List[EmissionFactorRecord], int]:
        raise NotImplementedError

    @abstractmethod
    def get_factor(self, edition_id: str, factor_id: str) -> Optional[EmissionFactorRecord]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def search_facets(
        self,
        edition_id: str,
        include_preview: bool = False,
        include_connector: bool = False,
        max_values: int = 80,
    ) -> Dict[str, Any]:
        """Facet counts for list/search filters (M2); keys are value -> count."""
        raise NotImplementedError

    @abstractmethod
    def coverage_stats(self, edition_id: str) -> Dict[str, Any]:
        """Return coverage dict compatible with API CoverageStats model."""
        raise NotImplementedError

    @abstractmethod
    def list_factor_summaries(self, edition_id: str) -> List[Dict[str, str]]:
        """Per row: factor_id, content_hash, factor_status (for edition compare)."""
        raise NotImplementedError


def _init_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS editions (
            edition_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT '',
            manifest_hash TEXT NOT NULL DEFAULT '',
            manifest_json TEXT NOT NULL DEFAULT '{}',
            changelog_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS catalog_factors (
            edition_id TEXT NOT NULL,
            factor_id TEXT NOT NULL,
            fuel_type TEXT NOT NULL DEFAULT '',
            geography TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL DEFAULT '',
            boundary TEXT NOT NULL DEFAULT '',
            search_blob TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            PRIMARY KEY (edition_id, factor_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cf_edition_geo
            ON catalog_factors (edition_id, geography);
        CREATE INDEX IF NOT EXISTS idx_cf_edition_fuel
            ON catalog_factors (edition_id, fuel_type);
        CREATE INDEX IF NOT EXISTS idx_cf_edition_scope
            ON catalog_factors (edition_id, scope);
        """
    )


def _apply_sqlite_migrations(conn: sqlite3.Connection) -> None:
    """Add CTO v0.1 columns and auxiliary tables to existing SQLite files."""
    cur = conn.execute("PRAGMA table_info(catalog_factors)")
    colnames = {row[1] for row in cur.fetchall()}
    alters = []
    specs = [
        ("factor_status", "TEXT DEFAULT 'certified'"),
        ("source_id", "TEXT"),
        ("source_release", "TEXT"),
        ("source_record_id", "TEXT"),
        ("release_version", "TEXT"),
        ("validation_flags", "TEXT DEFAULT '{}'"),
        ("replacement_factor_id", "TEXT"),
        ("license_class", "TEXT"),
        ("activity_tags", "TEXT NOT NULL DEFAULT '[]'"),
        ("sector_tags", "TEXT NOT NULL DEFAULT '[]'"),
    ]
    for col, typ in specs:
        if col not in colnames:
            alters.append(f"ALTER TABLE catalog_factors ADD COLUMN {col} {typ}")
    for sql in alters:
        conn.execute(sql)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_artifacts (
            artifact_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            retrieved_at TEXT NOT NULL DEFAULT (datetime('now')),
            url TEXT,
            content_type TEXT,
            sha256 TEXT NOT NULL,
            bytes_size INTEGER,
            storage_uri TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ingest_runs (
            run_id TEXT PRIMARY KEY,
            artifact_id TEXT,
            edition_id TEXT,
            parser_id TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            finished_at TEXT,
            row_counts_json TEXT NOT NULL DEFAULT '{}',
            owner TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS factor_lineage (
            edition_id TEXT NOT NULL,
            factor_id TEXT NOT NULL,
            artifact_id TEXT,
            ingest_run_id TEXT,
            lineage_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (edition_id, factor_id)
        );
        CREATE TABLE IF NOT EXISTS qa_reviews (
            review_id TEXT PRIMARY KEY,
            edition_id TEXT NOT NULL,
            factor_id TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS api_usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            api_key_hash TEXT,
            tier TEXT,
            hit_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS policy_applicability (
            rule_id TEXT NOT NULL,
            version TEXT NOT NULL,
            regulation_tag TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (rule_id, version)
        );
        CREATE INDEX IF NOT EXISTS idx_qa_edition ON qa_reviews (edition_id, factor_id);
        """
    )


def _cap_facet_counts(counts: Dict[str, int], max_values: int) -> Dict[str, int]:
    if max_values < 8:
        max_values = 8
    if len(counts) <= max_values:
        return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    head = dict(ordered[: max_values - 1])
    tail = sum(c for _, c in ordered[max_values - 1 :])
    head["_other"] = tail
    return head


def _status_visibility_sql(include_preview: bool, include_connector: bool) -> str:
    """SQL fragment for non-deprecated rows visible under status flags."""
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


class SqliteFactorCatalogRepository(FactorCatalogRepository):
    """SQLite-backed catalog for local CI and air-gapped bundles."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            _init_sqlite_schema(conn)
            _apply_sqlite_migrations(conn)
            conn.commit()
        finally:
            conn.close()
        logger.debug("SQLite catalog repository initialized at %s", self.db_path)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path))
        c.row_factory = sqlite3.Row
        return c

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
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO editions (edition_id, status, label, manifest_hash, manifest_json, changelog_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(edition_id) DO UPDATE SET
                    status=excluded.status,
                    label=excluded.label,
                    manifest_hash=excluded.manifest_hash,
                    manifest_json=excluded.manifest_json,
                    changelog_json=excluded.changelog_json
                """,
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
        finally:
            conn.close()

    def insert_factors(
        self,
        edition_id: str,
        records: Sequence[EmissionFactorRecord],
    ) -> None:
        conn = self._conn()
        try:
            rows = []
            for r in records:
                blob = " ".join(
                    [
                        r.factor_id,
                        r.fuel_type,
                        r.geography,
                        r.scope.value,
                        r.boundary.value,
                        " ".join(r.tags),
                        " ".join(getattr(r, "activity_tags", []) or []),
                        " ".join(getattr(r, "sector_tags", []) or []),
                        getattr(r, "license_class", "") or "",
                        r.notes or "",
                    ]
                )
                vf = json.dumps(getattr(r, "validation_flags", {}) or {}, sort_keys=True)
                at = json.dumps(list(getattr(r, "activity_tags", []) or []), sort_keys=True)
                st = json.dumps(list(getattr(r, "sector_tags", []) or []), sort_keys=True)
                rows.append(
                    (
                        edition_id,
                        r.factor_id,
                        r.fuel_type.lower(),
                        r.geography,
                        r.scope.value,
                        r.boundary.value.lower().replace(" ", "_"),
                        blob.lower(),
                        json.dumps(r.to_dict(), sort_keys=True, default=str),
                        r.content_hash,
                        getattr(r, "factor_status", "certified") or "certified",
                        getattr(r, "source_id", None),
                        getattr(r, "source_release", None),
                        getattr(r, "source_record_id", None),
                        getattr(r, "release_version", None),
                        vf,
                        getattr(r, "replacement_factor_id", None),
                        getattr(r, "license_class", None),
                        at,
                        st,
                    )
                )
            conn.executemany(
                """
                INSERT INTO catalog_factors (
                    edition_id, factor_id, fuel_type, geography, scope, boundary,
                    search_blob, payload_json, content_hash,
                    factor_status, source_id, source_release, source_record_id,
                    release_version, validation_flags, replacement_factor_id,
                    license_class, activity_tags, sector_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edition_id, factor_id) DO UPDATE SET
                    fuel_type=excluded.fuel_type,
                    geography=excluded.geography,
                    scope=excluded.scope,
                    boundary=excluded.boundary,
                    search_blob=excluded.search_blob,
                    payload_json=excluded.payload_json,
                    content_hash=excluded.content_hash,
                    factor_status=excluded.factor_status,
                    source_id=excluded.source_id,
                    source_release=excluded.source_release,
                    source_record_id=excluded.source_record_id,
                    release_version=excluded.release_version,
                    validation_flags=excluded.validation_flags,
                    replacement_factor_id=excluded.replacement_factor_id,
                    license_class=excluded.license_class,
                    activity_tags=excluded.activity_tags,
                    sector_tags=excluded.sector_tags
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def list_editions(self, include_pending: bool = True) -> List[EditionRow]:
        conn = self._conn()
        try:
            q = "SELECT edition_id, status, label, manifest_hash, changelog_json FROM editions ORDER BY edition_id DESC"
            cur = conn.execute(q)
            rows = []
            for r in cur.fetchall():
                if not include_pending and r["status"] == "pending":
                    continue
                rows.append(
                    EditionRow(
                        edition_id=r["edition_id"],
                        status=r["status"],
                        label=r["label"] or "",
                        manifest_hash=r["manifest_hash"] or "",
                        changelog_json=r["changelog_json"] or "[]",
                    )
                )
            return rows
        finally:
            conn.close()

    def get_default_edition_id(self) -> str:
        conn = self._conn()
        try:
            r = conn.execute(
                "SELECT edition_id FROM editions WHERE status = 'stable' ORDER BY edition_id DESC LIMIT 1"
            ).fetchone()
            if r:
                return str(r["edition_id"])
            r2 = conn.execute(
                "SELECT edition_id FROM editions ORDER BY edition_id DESC LIMIT 1"
            ).fetchone()
            if r2:
                return str(r2["edition_id"])
        finally:
            conn.close()
        return ""

    def resolve_edition(self, requested: Optional[str]) -> str:
        if requested:
            conn = self._conn()
            try:
                ok = conn.execute(
                    "SELECT 1 FROM editions WHERE edition_id = ? LIMIT 1", (requested,)
                ).fetchone()
                if ok:
                    return requested
            finally:
                conn.close()
            raise ValueError(f"Unknown edition_id: {requested!r}")
        default = self.get_default_edition_id()
        if not default:
            raise ValueError("No editions in catalog database")
        return default

    def get_changelog(self, edition_id: str) -> List[str]:
        conn = self._conn()
        try:
            r = conn.execute(
                "SELECT changelog_json FROM editions WHERE edition_id = ?", (edition_id,)
            ).fetchone()
            if not r:
                return []
            return list(json.loads(r["changelog_json"] or "[]"))
        finally:
            conn.close()

    def get_manifest_dict(self, edition_id: str) -> Dict[str, Any]:
        conn = self._conn()
        try:
            r = conn.execute(
                "SELECT manifest_json FROM editions WHERE edition_id = ?", (edition_id,)
            ).fetchone()
            if not r:
                return {}
            return dict(json.loads(r["manifest_json"] or "{}"))
        finally:
            conn.close()

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
    ) -> Tuple[List[EmissionFactorRecord], int]:
        clauses = ["edition_id = ?"]
        params: List[Any] = [edition_id]
        clauses.append(_status_visibility_sql(include_preview, include_connector))
        if fuel_type:
            clauses.append("LOWER(fuel_type) = ?")
            params.append(fuel_type.lower())
        if geography:
            clauses.append("geography = ?")
            params.append(geography)
        if scope:
            clauses.append("scope = ?")
            params.append(scope)
        if boundary:
            clauses.append("boundary = ?")
            params.append(boundary.lower().replace(" ", "_"))
        where = " AND ".join(clauses)
        conn = self._conn()
        try:
            total = conn.execute(
                f"SELECT COUNT(*) AS c FROM catalog_factors WHERE {where}", params
            ).fetchone()["c"]
            offset = (page - 1) * limit
            params2 = list(params) + [limit, offset]
            cur = conn.execute(
                f"""
                SELECT payload_json FROM catalog_factors
                WHERE {where}
                ORDER BY factor_id
                LIMIT ? OFFSET ?
                """,
                params2,
            )
            recs = [_record_from_stored_payload(json.loads(row["payload_json"])) for row in cur]
            return recs, int(total)
        finally:
            conn.close()

    def get_factor(self, edition_id: str, factor_id: str) -> Optional[EmissionFactorRecord]:
        conn = self._conn()
        try:
            r = conn.execute(
                """
                SELECT payload_json FROM catalog_factors
                WHERE edition_id = ? AND factor_id = ?
                """,
                (edition_id, factor_id),
            ).fetchone()
            if not r:
                return None
            return _record_from_stored_payload(json.loads(r["payload_json"]))
        finally:
            conn.close()

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
        qn = f"%{query.lower()}%"
        vis = _status_visibility_sql(include_preview, include_connector)
        clauses = ["edition_id = ?", vis, "search_blob LIKE ?"]
        params: List[Any] = [edition_id, qn]
        if geography:
            clauses.append("geography = ?")
            params.append(geography)
        if factor_status:
            clauses.append("COALESCE(factor_status, 'certified') = ?")
            params.append(factor_status)
        if source_id:
            clauses.append("source_id = ?")
            params.append(source_id)
        where = " AND ".join(clauses)
        conn = self._conn()
        try:
            cur = conn.execute(
                f"""
                SELECT payload_json FROM catalog_factors
                WHERE {where}
                ORDER BY factor_id
                LIMIT ?
                """,
                params + [limit],
            )
            return [_record_from_stored_payload(json.loads(r["payload_json"])) for r in cur]
        finally:
            conn.close()

    def search_facets(
        self,
        edition_id: str,
        include_preview: bool = False,
        include_connector: bool = False,
        max_values: int = 80,
    ) -> Dict[str, Any]:
        vis = _status_visibility_sql(include_preview, include_connector)
        conn = self._conn()
        facets: Dict[str, Dict[str, int]] = {}
        try:

            def load(expr: str, key: str) -> None:
                cur = conn.execute(
                    f"""
                    SELECT {expr} AS k, COUNT(*) AS c
                    FROM catalog_factors
                    WHERE edition_id = ? AND {vis}
                    GROUP BY k
                    ORDER BY c DESC
                    """,
                    (edition_id,),
                )
                raw = {str(r["k"]): int(r["c"]) for r in cur.fetchall()}
                facets[key] = _cap_facet_counts(raw, max_values)

            load("COALESCE(factor_status, 'certified')", "factor_status")
            load(
                "COALESCE(NULLIF(TRIM(source_id), ''), '(unset)')",
                "source_id",
            )
            load("geography", "geography")
            load("scope", "scope")
            load("boundary", "boundary")
            load("fuel_type", "fuel_type")
        finally:
            conn.close()
        return {"edition_id": edition_id, "facets": facets}

    def list_factor_summaries(self, edition_id: str) -> List[Dict[str, str]]:
        conn = self._conn()
        try:
            cur = conn.execute(
                """
                SELECT factor_id, content_hash,
                       COALESCE(factor_status, 'certified') AS factor_status
                FROM catalog_factors
                WHERE edition_id = ?
                ORDER BY factor_id
                """,
                (edition_id,),
            )
            return [
                {
                    "factor_id": str(r["factor_id"]),
                    "content_hash": str(r["content_hash"]),
                    "factor_status": str(r["factor_status"]),
                }
                for r in cur.fetchall()
            ]
        finally:
            conn.close()

    def coverage_stats(self, edition_id: str) -> Dict[str, Any]:
        conn = self._conn()
        try:
            total = conn.execute(
                "SELECT COUNT(*) AS c FROM catalog_factors WHERE edition_id = ?",
                (edition_id,),
            ).fetchone()["c"]
            geo_rows = conn.execute(
                """
                SELECT geography, COUNT(*) AS c FROM catalog_factors
                WHERE edition_id = ? GROUP BY geography
                """,
                (edition_id,),
            ).fetchall()
            fuel_rows = conn.execute(
                """
                SELECT fuel_type, COUNT(*) AS c FROM catalog_factors
                WHERE edition_id = ? GROUP BY fuel_type
                """,
                (edition_id,),
            ).fetchall()
            scope_rows = conn.execute(
                """
                SELECT scope, COUNT(*) AS c FROM catalog_factors
                WHERE edition_id = ? GROUP BY scope
                """,
                (edition_id,),
            ).fetchall()
            boundary_rows = conn.execute(
                """
                SELECT boundary, COUNT(*) AS c FROM catalog_factors
                WHERE edition_id = ? GROUP BY boundary
                """,
                (edition_id,),
            ).fetchall()
            status_rows = conn.execute(
                """
                SELECT COALESCE(factor_status, 'certified') AS st, COUNT(*) AS c
                FROM catalog_factors
                WHERE edition_id = ? GROUP BY st
                """,
                (edition_id,),
            ).fetchall()
        finally:
            conn.close()
        scopes = {"1": 0, "2": 0, "3": 0}
        for r in scope_rows:
            k = str(r["scope"])
            if k in scopes:
                scopes[k] = int(r["c"])
        boundaries = {str(r["boundary"]): int(r["c"]) for r in boundary_rows}
        by_geography = {str(r["geography"]): int(r["c"]) for r in geo_rows}
        by_fuel_type = {str(r["fuel_type"]): int(r["c"]) for r in fuel_rows}
        by_status = {str(r["st"]): int(r["c"]) for r in status_rows}
        certified = by_status.get("certified", 0)
        preview = by_status.get("preview", 0)
        connector_visible = by_status.get("connector_only", 0)
        return {
            "total_factors": int(total),
            "total_catalog": int(total),
            "certified": certified,
            "preview": preview,
            "connector_visible": connector_visible,
            "geographies": len(by_geography),
            "fuel_types": len(by_fuel_type),
            "scopes": scopes,
            "boundaries": boundaries,
            "by_geography": by_geography,
            "by_fuel_type": by_fuel_type,
            "by_status": by_status,
        }


def _memory_status_visible(
    factor: EmissionFactorRecord,
    include_preview: bool,
    include_connector: bool,
) -> bool:
    st = getattr(factor, "factor_status", "certified") or "certified"
    if st == "deprecated":
        return False
    if st == "certified":
        return True
    if st == "preview":
        return include_preview
    if st == "connector_only":
        return include_connector
    return False


class MemoryFactorCatalogRepository(FactorCatalogRepository):
    """Wraps EmissionFactorDatabase as a single synthetic edition."""

    def __init__(self, edition_id: str, label: str, db: Any):
        from greenlang.data.emission_factor_database import EmissionFactorDatabase

        if not isinstance(db, EmissionFactorDatabase):
            raise TypeError("db must be EmissionFactorDatabase")
        self.edition_id = edition_id
        self.label = label
        self._db = db
        self._factors = list(db.factors.values())
        from greenlang.factors.edition_manifest import build_manifest_for_factors

        self._manifest = build_manifest_for_factors(
            edition_id, "stable", self._factors, changelog=[f"Built-in catalog {edition_id}"]
        )

    def list_editions(self, include_pending: bool = True) -> List[EditionRow]:
        return [
            EditionRow(
                edition_id=self.edition_id,
                status="stable",
                label=self.label,
                manifest_hash=self._manifest.manifest_fingerprint(),
                changelog_json=json.dumps(self._manifest.changelog),
            )
        ]

    def get_default_edition_id(self) -> str:
        return self.edition_id

    def resolve_edition(self, requested: Optional[str]) -> str:
        if not requested:
            return self.edition_id
        if requested != self.edition_id:
            raise ValueError(f"Unknown edition {requested!r}")
        return self.edition_id

    def get_changelog(self, edition_id: str) -> List[str]:
        if edition_id != self.edition_id:
            return []
        return list(self._manifest.changelog)

    def get_manifest_dict(self, edition_id: str) -> Dict[str, Any]:
        if edition_id != self.edition_id:
            return {}
        return self._manifest.to_dict()

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
    ) -> Tuple[List[EmissionFactorRecord], int]:
        if edition_id != self.edition_id:
            return [], 0
        rows = self._db.list_factors(
            fuel_type=fuel_type,
            geography=geography,
        )
        filtered: List[EmissionFactorRecord] = []
        for factor in rows:
            if not _memory_status_visible(factor, include_preview, include_connector):
                continue
            if scope and factor.scope.value != scope:
                continue
            if boundary and factor.boundary.value != boundary:
                continue
            filtered.append(factor)
        total = len(filtered)
        start = (page - 1) * limit
        return filtered[start : start + limit], total

    def get_factor(self, edition_id: str, factor_id: str) -> Optional[EmissionFactorRecord]:
        if edition_id != self.edition_id:
            return None
        for f in self._factors:
            if f.factor_id == factor_id:
                return f
        return None

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
        if edition_id != self.edition_id:
            return []
        q_lower = query.lower()
        results: List[EmissionFactorRecord] = []
        for factor in self._factors:
            if not _memory_status_visible(factor, include_preview, include_connector):
                continue
            if geography and factor.geography != geography:
                continue
            if factor_status and (getattr(factor, "factor_status", "certified") or "certified") != factor_status:
                continue
            if source_id and getattr(factor, "source_id", None) != source_id:
                continue
            searchable = " ".join(
                [
                    factor.fuel_type,
                    factor.geography,
                    factor.scope.value,
                    factor.boundary.value,
                    " ".join(factor.tags),
                    " ".join(getattr(factor, "activity_tags", []) or []),
                    " ".join(getattr(factor, "sector_tags", []) or []),
                    getattr(factor, "license_class", "") or "",
                    factor.notes or "",
                ]
            ).lower()
            if q_lower in searchable:
                results.append(factor)
            if len(results) >= limit:
                break
        return results

    def search_facets(
        self,
        edition_id: str,
        include_preview: bool = False,
        include_connector: bool = False,
        max_values: int = 80,
    ) -> Dict[str, Any]:
        if edition_id != self.edition_id:
            return {"edition_id": edition_id, "facets": {}}
        acc: Dict[str, Dict[str, int]] = {
            "factor_status": {},
            "source_id": {},
            "geography": {},
            "scope": {},
            "boundary": {},
            "fuel_type": {},
        }
        for factor in self._factors:
            if not _memory_status_visible(factor, include_preview, include_connector):
                continue
            st = getattr(factor, "factor_status", "certified") or "certified"
            sid = getattr(factor, "source_id", None) or "(unset)"
            keys = [
                ("factor_status", st),
                ("source_id", str(sid)),
                ("geography", factor.geography),
                ("scope", factor.scope.value),
                ("boundary", factor.boundary.value),
                ("fuel_type", factor.fuel_type),
            ]
            for fk, val in keys:
                acc[fk][val] = acc[fk].get(val, 0) + 1
        capped = {k: _cap_facet_counts(v, max_values) for k, v in acc.items()}
        return {"edition_id": edition_id, "facets": capped}

    def list_factor_summaries(self, edition_id: str) -> List[Dict[str, str]]:
        if edition_id != self.edition_id:
            return []
        return [
            {
                "factor_id": f.factor_id,
                "content_hash": f.content_hash,
                "factor_status": getattr(f, "factor_status", "certified") or "certified",
            }
            for f in self._factors
        ]

    def coverage_stats(self, edition_id: str) -> Dict[str, Any]:
        if edition_id != self.edition_id:
            return {
                "total_factors": 0,
                "total_catalog": 0,
                "certified": 0,
                "preview": 0,
                "connector_visible": 0,
                "geographies": 0,
                "fuel_types": 0,
                "scopes": {"1": 0, "2": 0, "3": 0},
                "boundaries": {},
                "by_geography": {},
                "by_fuel_type": {},
                "by_status": {},
            }
        geographies = set()
        fuel_types = set()
        scopes = {"1": 0, "2": 0, "3": 0}
        boundaries: Dict[str, int] = {}
        by_geography: Dict[str, int] = {}
        by_fuel_type: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        for factor in self._factors:
            st = getattr(factor, "factor_status", "certified") or "certified"
            by_status[st] = by_status.get(st, 0) + 1
            geographies.add(factor.geography)
            fuel_types.add(factor.fuel_type)
            scopes[factor.scope.value] = scopes.get(factor.scope.value, 0) + 1
            boundaries[factor.boundary.value] = boundaries.get(factor.boundary.value, 0) + 1
            by_geography[factor.geography] = by_geography.get(factor.geography, 0) + 1
            by_fuel_type[factor.fuel_type] = by_fuel_type.get(factor.fuel_type, 0) + 1
        return {
            "total_factors": len(self._factors),
            "total_catalog": len(self._factors),
            "certified": by_status.get("certified", 0),
            "preview": by_status.get("preview", 0),
            "connector_visible": by_status.get("connector_only", 0),
            "geographies": len(geographies),
            "fuel_types": len(fuel_types),
            "scopes": scopes,
            "boundaries": boundaries,
            "by_geography": by_geography,
            "by_fuel_type": by_fuel_type,
            "by_status": by_status,
        }
