# -*- coding: utf-8 -*-
"""Real v0.1 alpha factor repository (Wave D / TaskCreate #31 / WS9-T5).

Replaces the in-test shim path that the SDK E2E demo (Wave C #19) relied on.
Records are stored verbatim — the JSON blob written by :meth:`publish` is
the EXACT same dict returned by :meth:`get_by_urn`. No coercion, no field
loss; the ``factor_record_v0_1.schema.json`` contract round-trips bit-for-bit.

Storage backends
----------------
- SQLite (``sqlite:///path/to.db`` or ``sqlite:///:memory:``)
- Postgres (``postgresql://...`` or ``postgres://...``) — uses the
  ``factors_v0_1.factor`` table created by Alembic revision 0001.

The SQLite schema mirrors the Postgres DDL columns 1:1 so a record written
on SQLite can be moved to Postgres with a column-aligned dump.

Immutability
------------
Once a URN is published it is immutable at the repository surface. There
is no ``update`` / ``upsert`` API; the SQLite schema declares
``urn TEXT PRIMARY KEY`` so a duplicate INSERT is rejected by the engine,
and on top of that :meth:`publish` raises :class:`FactorURNAlreadyExistsError`
on conflict. SQLite still permits low-level ``UPDATE`` SQL — the
JSONB-stored ``record_jsonb`` blob is therefore a *trust boundary*: callers
must use this repository class (never raw DB cursors) for every write.
Corrections happen via the v0.1 ``supersedes_urn`` field on a NEW record,
not by mutating an existing row.

Phase 2 / WS8 — secure-by-default publish gates (CTO P0/P1 fix, 2026-04-27)
---------------------------------------------------------------------------
The repository runs the seven-gate :class:`PublishGateOrchestrator` BY
DEFAULT on every :meth:`publish` (constructor parameter
``publish_env='production'`` is the default, secure-by-default behaviour).
Operators that need the legacy single-gate provenance path must opt OUT
explicitly via ``publish_env='legacy'`` — the bypass is intentionally loud
(a one-time logger.warning records the URN audited under legacy mode).

CTO doc references: §6.1, §19.1 (FY27 Q1 alpha publish pipeline);
Phase 2 Plan §2.5 (publish gates) — fail-closed in production/staging.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from greenlang.factors.quality.alpha_provenance_gate import (
    AlphaProvenanceGate,
    AlphaProvenanceGateError,
)

logger = logging.getLogger(__name__)


__all__ = [
    "AlphaFactorRepository",
    "FactorURNAlreadyExistsError",
]


# ---------------------------------------------------------------------------
# Phase 2 constants — kept module-level so they survive subclass overrides
# of the table DDL but can still be referenced by tests for parity checks.
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

# Mirrors the CHECK constraint on factors_v0_1.provenance_edges.edge_type
# (V503). Keep in lockstep with the SQL.
_PROVENANCE_EDGE_TYPES = frozenset(
    {"extraction", "derivation", "correction", "supersedes"}
)

# Mirrors the CHECK constraints on factors_v0_1.changelog_events (V503).
_CHANGELOG_EVENT_TYPES = frozenset(
    {
        "schema_change",
        "factor_publish",
        "factor_supersede",
        "factor_deprecate",
        "source_add",
        "source_deprecate",
        "pack_release",
        "migration_apply",
    }
)
_CHANGELOG_CHANGE_CLASSES = frozenset(
    {"additive", "breaking", "deprecated", "removed"}
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FactorURNAlreadyExistsError(Exception):
    """Raised when :meth:`AlphaFactorRepository.publish` is called with a URN
    that already exists in the catalogue.

    The v0.1 contract is strictly immutable — a correction MUST be issued
    as a NEW record carrying ``supersedes_urn`` pointing at the original.
    """

    def __init__(self, urn: str) -> None:
        self.urn = urn
        super().__init__(
            f"Factor URN {urn!r} already exists; v0.1 alpha records are "
            "immutable. Issue a correction with supersedes_urn pointing at "
            "the original."
        )


# ---------------------------------------------------------------------------
# DSN parsing
# ---------------------------------------------------------------------------


def _resolve_sqlite_path(dsn: str) -> str:
    """Extract a sqlite filesystem path (or ``:memory:``) from ``dsn``.

    Accepts: ``sqlite:///abs/path.db``, ``sqlite:///:memory:``,
    ``sqlite://:memory:`` (SQLAlchemy-style), or a bare path.
    """
    raw = (dsn or "").strip()
    if not raw:
        return ":memory:"
    if raw.startswith("sqlite:"):
        # sqlite:///abs -> "/abs"; sqlite:///:memory: -> "/:memory:"
        body = raw[len("sqlite:"):]
        body = body.lstrip("/")
        if body in ("", ":memory:"):
            return ":memory:"
        # Re-add leading slash on POSIX absolute paths (sqlite:////root/x ->
        # /root/x) but keep Windows drive letters intact (sqlite:///C:/x).
        if not body.startswith(":") and not (
            len(body) >= 2 and body[1] == ":"
        ):
            body = "/" + body
        return unquote(body)
    return raw


def _is_postgres_dsn(dsn: str) -> bool:
    return dsn.startswith("postgres://") or dsn.startswith("postgresql://")


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class AlphaFactorRepository:
    """Stores and serves v0.1-alpha-shape factor records.

    Backed by SQLite (alpha) or Postgres (production via Alembic 0001).
    Records are stored as JSONB blobs that exactly match the
    ``factor_record_v0_1.schema.json`` contract — no lossy coercion.

    Thread-safe: each call opens a fresh connection in SQLite mode (so
    multi-thread access never crosses the SQLite ``check_same_thread``
    boundary). Postgres mode uses ``psycopg`` connections per call.

    Phase 2 publish enforcement
    ---------------------------
    The repository is **secure by default** — every :meth:`publish` runs
    the seven-gate :class:`PublishGateOrchestrator` BEFORE the legacy
    :class:`AlphaProvenanceGate`. The default environment is
    ``publish_env='production'``. Callers that want the pre-Phase-2
    behaviour (legacy provenance gate only) must opt out with
    ``publish_env='legacy'`` — a one-time WARNING log records the URN
    audited under legacy mode.

    Allowed values for ``publish_env``:

    * ``'production'`` — default; all 7 gates enforced; ontology + source
      registry tables MUST be present (fail-closed).
    * ``'staging'``   — all 7 gates enforced; ontology + source registry
      tables MUST be present (fail-closed); ``review_status`` may be
      ``approved`` or ``pending``.
    * ``'dev'``       — all 7 gates run, but missing ontology / source
      tables warn-and-skip (fail-open) so dev iteration on un-seeded
      checkouts still works.
    * ``'legacy'``    — orchestrator skipped; legacy
      :class:`AlphaProvenanceGate` only. INTENDED FOR EXPLICIT BYPASS.
    """

    #: Default publish environment surfaced to subclasses + reflection.
    REPOSITORY_DEFAULT_PUBLISH_ENV: str = "production"

    #: Set of all allowed ``publish_env`` values. Module-level constant
    #: so subclasses can extend without rebuilding the validator.
    _ALLOWED_PUBLISH_ENVS: Tuple[str, ...] = ("production", "staging", "dev", "legacy")

    # SQLite DDL — column types match the Postgres factors_v0_1.* tables
    # (TEXT for ISO-8601 timestamps; no engine-specific casting). Phase 2
    # extends the SQLite mirror with `factor_aliases`, `source_artifacts`,
    # `provenance_edges`, and `changelog_events` so SQLite-mode integration
    # tests can exercise the full repository surface without Postgres.
    _SQLITE_DDL = (
        # Core factor table (V500 + V506 mirror).
        # V506 (2026-04-27 additive amendment) adds five OPTIONAL columns:
        # activity_taxonomy_urn, confidence, created_at_pre_publish,
        # updated_at_pre_publish, superseded_by_urn. The SQLite column
        # names ``created_at_pre_publish`` / ``updated_at_pre_publish``
        # match the Postgres backing columns (V506 renames the public
        # contract names ``created_at`` / ``updated_at`` to disambiguate
        # from V500's existing row-creation ``created_at`` column).
        "CREATE TABLE IF NOT EXISTS alpha_factors_v0_1 ("
        " urn TEXT PRIMARY KEY,"
        " source_urn TEXT,"
        " factor_pack_urn TEXT,"
        " category TEXT,"
        " geography_urn TEXT,"
        " vintage_start DATE,"
        " vintage_end DATE,"
        " published_at TIMESTAMPTZ,"
        " record_jsonb TEXT NOT NULL,"
        " created_at TIMESTAMPTZ DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
        " activity_taxonomy_urn TEXT,"
        " confidence REAL,"
        " created_at_pre_publish TIMESTAMPTZ,"
        " updated_at_pre_publish TIMESTAMPTZ,"
        " superseded_by_urn TEXT"
        ")",
        # V501 mirror — factor_aliases (legacy EF id -> canonical URN).
        "CREATE TABLE IF NOT EXISTS alpha_factor_aliases_v0_1 ("
        " pk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " urn TEXT NOT NULL,"
        " legacy_id TEXT NOT NULL UNIQUE,"
        " kind TEXT NOT NULL DEFAULT 'EF' CHECK (kind IN ('EF','custom')),"
        " created_at TIMESTAMPTZ DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
        " retired_at TIMESTAMPTZ"
        ")",
        # V501 + V510 mirror — source_artifacts (raw immutable bytes).
        # The base 11 columns come from V505. Phase 3 audit gap C extended
        # the schema to the full 16+ Phase 3 contract row; the additive
        # columns are appended below so an existing table is migrated by
        # the post-DDL ALTER TABLE block in :meth:`_ensure_schema`.
        "CREATE TABLE IF NOT EXISTS alpha_source_artifacts_v0_1 ("
        " pk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " sha256 TEXT NOT NULL UNIQUE,"
        " source_urn TEXT NOT NULL,"
        " source_version TEXT NOT NULL,"
        " uri TEXT NOT NULL,"
        " content_type TEXT,"
        " size_bytes INTEGER,"
        " parser_id TEXT,"
        " parser_version TEXT,"
        " parser_commit TEXT,"
        " ingested_at TIMESTAMPTZ DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
        " metadata TEXT,"
        # V510 additive columns (Phase 3 audit gap C).
        " source_url TEXT,"
        " source_publication_date TEXT,"
        " parser_module TEXT,"
        " parser_function TEXT,"
        " operator TEXT,"
        " licence_class TEXT,"
        " redistribution_class TEXT,"
        " ingestion_run_id TEXT,"
        " status TEXT NOT NULL DEFAULT 'fetched'"
        ")",
        # V503 mirror — provenance_edges (factor URN -> source_artifact).
        "CREATE TABLE IF NOT EXISTS alpha_provenance_edges_v0_1 ("
        " pk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " factor_urn TEXT NOT NULL,"
        " source_artifact_pk INTEGER NOT NULL,"
        " row_ref TEXT NOT NULL,"
        " edge_type TEXT NOT NULL DEFAULT 'extraction'"
        "  CHECK (edge_type IN ('extraction','derivation','correction','supersedes')),"
        " created_at TIMESTAMPTZ DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
        " UNIQUE (factor_urn, source_artifact_pk, row_ref, edge_type)"
        ")",
        # V503 mirror — changelog_events (append-only schema/record audit).
        "CREATE TABLE IF NOT EXISTS alpha_changelog_events_v0_1 ("
        " pk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " event_type TEXT NOT NULL CHECK (event_type IN ("
        "  'schema_change','factor_publish','factor_supersede','factor_deprecate',"
        "  'source_add','source_deprecate','pack_release','migration_apply')),"
        " schema_version TEXT,"
        " subject_urn TEXT,"
        " change_class TEXT"
        "  CHECK (change_class IS NULL OR change_class IN ('additive','breaking','deprecated','removed')),"
        " migration_note_uri TEXT,"
        " actor TEXT NOT NULL,"
        " occurred_at TIMESTAMPTZ DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
        " metadata TEXT"
        ")",
    )
    _SQLITE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_source ON alpha_factors_v0_1(source_urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_pack ON alpha_factors_v0_1(factor_pack_urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_category ON alpha_factors_v0_1(category)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_geo ON alpha_factors_v0_1(geography_urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_published ON alpha_factors_v0_1(published_at DESC)",
        # Phase 2 indexes
        "CREATE INDEX IF NOT EXISTS ix_alpha_factor_aliases_v0_1_urn ON alpha_factor_aliases_v0_1(urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_source_artifacts_v0_1_source ON alpha_source_artifacts_v0_1(source_urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_source_artifacts_v0_1_version ON alpha_source_artifacts_v0_1(source_urn, source_version)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_provenance_edges_v0_1_factor ON alpha_provenance_edges_v0_1(factor_urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_provenance_edges_v0_1_artifact ON alpha_provenance_edges_v0_1(source_artifact_pk)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_changelog_events_v0_1_subject ON alpha_changelog_events_v0_1(subject_urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_changelog_events_v0_1_type ON alpha_changelog_events_v0_1(event_type, occurred_at DESC)",
        # V506 mirror — partial indexes for the additive contract fields.
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_activity ON alpha_factors_v0_1(activity_taxonomy_urn) WHERE activity_taxonomy_urn IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_superseded_by ON alpha_factors_v0_1(superseded_by_urn) WHERE superseded_by_urn IS NOT NULL",
    )

    # Whitelist of filter columns — guards SQL injection by NEVER
    # interpolating user-controlled column names into SQL. Values are
    # always passed as bind parameters.
    _FILTER_COLUMNS = (
        "source_urn",
        "factor_pack_urn",
        "category",
        "geography_urn",
        # V506 additive — Phase 2 contract field, available as a query
        # filter so SDK callers can list factors by activity taxonomy.
        "activity_taxonomy_urn",
    )

    def __init__(
        self,
        dsn: str,
        *,
        gate: Optional[AlphaProvenanceGate] = None,
        publish_orchestrator: Optional[Any] = None,
        publish_env: Optional[str] = None,
    ) -> None:
        """Open the repository.

        Args:
            dsn: ``sqlite:///path.db`` / ``sqlite:///:memory:`` /
                 ``postgresql://...``. A bare path is treated as SQLite.
            gate: An optional :class:`AlphaProvenanceGate` instance. Kept
                  as a defensive legacy-mode fallback (the seven-gate
                  orchestrator already covers gate 1 + gate 6 of the
                  legacy gate's surface).
            publish_orchestrator: BACKWARDS-COMPAT alias. ``None`` (the
                default) defers to ``publish_env``. ``True`` or a pre-built
                :class:`PublishGateOrchestrator` instance forces orchestrator
                use (existing Phase 2 callers). ``False`` is now a
                DEPRECATED alias for ``publish_env='legacy'`` — emits a
                ``DeprecationWarning`` once.
            publish_env: ``'production'`` (default — secure by default),
                ``'staging'``, ``'dev'``, or ``'legacy'``. Forwarded to
                the lazy-built orchestrator. Production / staging fail
                CLOSED on missing ontology / source registry tables;
                dev warns-and-skips (so un-seeded checkouts still iterate);
                legacy bypasses the orchestrator entirely (loud warning
                emitted on first publish).

        Raises:
            ValueError: when ``publish_env`` is not one of the four
                allowed values.
        """
        self._dsn = dsn or "sqlite:///:memory:"
        self._gate = gate or AlphaProvenanceGate()
        self._lock = threading.Lock()
        self._is_postgres = _is_postgres_dsn(self._dsn)

        # ----------------------------------------------------------------
        # Phase 2 / WS8 — secure-by-default publish gate selection.
        # ----------------------------------------------------------------
        # 1. Backwards-compat: ``publish_orchestrator=False`` -> legacy.
        legacy_via_compat = False
        if publish_orchestrator is False:
            import warnings  # noqa: PLC0415 — deferred to keep cold-import lean.
            warnings.warn(
                "AlphaFactorRepository(publish_orchestrator=False) is "
                "deprecated; pass publish_env='legacy' instead. The "
                "Phase 2 seven-gate orchestrator is now mandatory by "
                "default in production/staging/dev environments.",
                DeprecationWarning,
                stacklevel=2,
            )
            legacy_via_compat = True

        # 2. Resolve the final ``publish_env``.
        if legacy_via_compat:
            resolved_env = "legacy"
        else:
            resolved_env = (
                publish_env or self.REPOSITORY_DEFAULT_PUBLISH_ENV
            ).strip().lower()

        if resolved_env not in self._ALLOWED_PUBLISH_ENVS:
            raise ValueError(
                "AlphaFactorRepository: publish_env must be one of "
                f"{list(self._ALLOWED_PUBLISH_ENVS)!r}; got {resolved_env!r}"
            )
        self._publish_env: str = resolved_env

        # 3. Stash the orchestrator handle. ``None`` in production/staging/
        #    dev means "lazy-build one on first publish"; ``False`` (after
        #    the compat path above) is the legacy-mode marker; a pre-built
        #    instance is honored as-is (used by tests that inject a
        #    SourceRightsService).
        if resolved_env == "legacy":
            # In legacy mode the orchestrator is not used. We still keep
            # the slot so it is introspectable.
            self._publish_orchestrator: Any = False
            # One-time warning emission marker (see :meth:`publish`).
            self._legacy_warning_emitted: bool = False
        else:
            # Default-secure: ``None`` triggers lazy-build on first publish;
            # an explicit pre-built instance is honored.
            self._publish_orchestrator = publish_orchestrator
            self._legacy_warning_emitted = True  # not applicable
        # In-memory SQLite: hold a single shared connection for the lifetime
        # of the repo so the schema and rows persist across method calls.
        self._memory_conn: Optional[sqlite3.Connection] = None
        if not self._is_postgres:
            sqlite_path = _resolve_sqlite_path(self._dsn)
            self._sqlite_path = sqlite_path
            if sqlite_path == ":memory:":
                self._memory_conn = sqlite3.connect(
                    ":memory:", check_same_thread=False, isolation_level=None
                )
                self._memory_conn.row_factory = sqlite3.Row
            else:
                # Ensure parent dir exists.
                p = Path(sqlite_path)
                if p.parent and not p.parent.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._sqlite_path = None
        self._ensure_schema()

    # -- connection management --------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection (memory-shared or per-call file)."""
        if self._is_postgres:
            raise RuntimeError(
                "Postgres mode connections are managed via psycopg pool — "
                "use _connect_pg() instead."
            )
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(
            self._sqlite_path,  # type: ignore[arg-type]
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create table + indexes if they don't exist (SQLite only).

        Postgres mode relies on Alembic revision 0001 having already created
        the ``factors_v0_1.factor`` table.
        """
        if self._is_postgres:
            return
        conn = self._connect()
        try:
            # _SQLITE_DDL is a tuple of CREATE TABLE statements (Phase 2
            # added factor_aliases / source_artifacts / provenance_edges /
            # changelog_events alongside the original alpha_factors_v0_1).
            for ddl in self._SQLITE_DDL:
                conn.execute(ddl)
            for ddl in self._SQLITE_INDEXES:
                conn.execute(ddl)
            # Phase 3 audit gap C — additive ALTER TABLE for the
            # extended source_artifacts contract. ``CREATE TABLE IF NOT
            # EXISTS`` is a no-op on a pre-existing alpha DB so the
            # additive columns must be backfilled here. SQLite's ALTER
            # TABLE ADD COLUMN raises on duplicate; swallow that.
            for col_ddl in (
                "ALTER TABLE alpha_source_artifacts_v0_1 ADD COLUMN source_url TEXT",
                "ALTER TABLE alpha_source_artifacts_v0_1 ADD COLUMN source_publication_date TEXT",
                "ALTER TABLE alpha_source_artifacts_v0_1 ADD COLUMN parser_module TEXT",
                "ALTER TABLE alpha_source_artifacts_v0_1 ADD COLUMN parser_function TEXT",
                "ALTER TABLE alpha_source_artifacts_v0_1 ADD COLUMN operator TEXT",
                "ALTER TABLE alpha_source_artifacts_v0_1 ADD COLUMN licence_class TEXT",
                "ALTER TABLE alpha_source_artifacts_v0_1 ADD COLUMN redistribution_class TEXT",
                "ALTER TABLE alpha_source_artifacts_v0_1 ADD COLUMN ingestion_run_id TEXT",
                "ALTER TABLE alpha_source_artifacts_v0_1 ADD COLUMN status TEXT NOT NULL DEFAULT 'fetched'",
            ):
                try:
                    conn.execute(col_ddl)
                except Exception:  # noqa: BLE001 — column may already exist
                    pass
        finally:
            if self._memory_conn is None:
                conn.close()

    def _get_or_build_publish_orchestrator(self) -> Any:
        """Lazy-build the Phase 2 :class:`PublishGateOrchestrator`.

        Imports are deferred so that test paths (and legacy callers that
        never publish) don't pay the import cost. The instance is cached
        on the repo for subsequent publishes.

        Recognised ``self._publish_orchestrator`` states:

        * ``False`` — legacy mode; ``publish()`` MUST NOT call this helper.
        * ``None`` / ``True`` — opt-in marker; build a fresh orchestrator
          targeted at ``self._publish_env``.
        * Any other object — assumed to be a pre-built orchestrator
          instance and returned as-is.
        """
        existing = self._publish_orchestrator
        if existing is True or existing is None:
            # Lazy import to break the import cycle (publish_gates -> repo).
            from greenlang.factors.quality.publish_gates import PublishGateOrchestrator  # noqa: PLC0415

            orch = PublishGateOrchestrator(self, env=self._publish_env)
            self._publish_orchestrator = orch
            return orch
        if existing is False:
            # Defensive — legacy mode should never reach this helper.
            raise RuntimeError(
                "AlphaFactorRepository: publish_env='legacy' but "
                "_get_or_build_publish_orchestrator() was called; this "
                "is a programming error in the publish() router."
            )
        return existing

    def close(self) -> None:
        """Release the in-memory connection (if any)."""
        if self._memory_conn is not None:
            try:
                self._memory_conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._memory_conn = None

    # -- public API --------------------------------------------------------

    def publish(self, record: Dict[str, Any]) -> str:
        """Validate the record, then persist atomically.

        Args:
            record: A v0.1-shape factor dict that satisfies the
                ``factor_record_v0_1.schema.json`` contract AND the
                Phase 2 seven-gate orchestrator (production/staging/dev)
                OR the legacy :class:`AlphaProvenanceGate` (legacy mode).

        Returns:
            The canonical URN of the published record.

        Raises:
            PublishGateError: a Phase 2 / WS8 gate rejected the record
                (one of: SchemaValidationError, URNDuplicateError,
                OntologyReferenceError, SourceRegistryError,
                LicenceMismatchError, ProvenanceIncompleteError,
                LifecycleStatusError). Default behaviour in
                production/staging/dev.
            AlphaProvenanceGateError: legacy provenance-only validation
                failed. Fires only when ``publish_env='legacy'`` (or the
                deprecated ``publish_orchestrator=False`` alias).
            FactorURNAlreadyExistsError: a record with the same URN exists.
            ValueError: the record is missing the ``urn`` key entirely.
        """
        # ----------------------------------------------------------------
        # Gate selection (Phase 2 / WS8 — secure by default).
        # ----------------------------------------------------------------
        # ``publish_env`` is the canonical source of truth. ``legacy``
        # routes to the single-gate fallback; every other env runs the
        # seven-gate orchestrator BEFORE the legacy gate.
        used_orchestrator = False
        if self._publish_env == "legacy":
            # Loud, one-time audit log so any legacy bypass is
            # forensically traceable. The first record published under
            # legacy mode anchors the audit trail.
            if not self._legacy_warning_emitted:
                self._legacy_warning_emitted = True
                first_urn = (
                    record.get("urn") if isinstance(record, dict) else None
                )
                logger.warning(
                    "alpha_factor_repo: publish_env='legacy' — Phase 2 "
                    "seven-gate orchestrator BYPASSED. dsn=%s first_urn=%r. "
                    "All subsequent publishes on this repository instance "
                    "run the legacy AlphaProvenanceGate ONLY.",
                    self._dsn, first_urn,
                )
            self._gate.assert_valid(record)
        else:
            # Production / staging / dev — orchestrator first, then the
            # legacy gate as a defense-in-depth layer (gates 1+6 of the
            # legacy gate are a strict subset of orchestrator gates 1+6,
            # but running it twice is cheap and catches any drift).
            orch = self._get_or_build_publish_orchestrator()
            orch.assert_publishable(record)
            self._gate.assert_valid(record)
            used_orchestrator = True

        urn = record.get("urn")
        if not isinstance(urn, str) or not urn:
            # Defensive — schema gate already enforces this.
            raise ValueError("record['urn'] must be a non-empty string")

        # Snapshot the columns we mirror so SQL filters can hit indexes.
        source_urn = record.get("source_urn")
        pack_urn = record.get("factor_pack_urn")
        category = record.get("category")
        geo_urn = record.get("geography_urn")
        vintage_start = record.get("vintage_start")
        vintage_end = record.get("vintage_end")
        published_at = record.get("published_at")
        # V506 additive contract fields (2026-04-27). All five are OPTIONAL
        # — defaulting to NULL preserves backward compatibility for the
        # 691 alpha records that pre-date the amendment.
        activity_taxonomy_urn = record.get("activity_taxonomy_urn")
        confidence = record.get("confidence")
        created_at_pre_publish = record.get("created_at")
        updated_at_pre_publish = record.get("updated_at")
        superseded_by_urn = record.get("superseded_by_urn")
        record_blob = json.dumps(record, sort_keys=True, default=str, ensure_ascii=False)

        if self._is_postgres:
            return self._publish_pg(
                urn,
                source_urn,
                pack_urn,
                category,
                geo_urn,
                vintage_start,
                vintage_end,
                published_at,
                record_blob,
                activity_taxonomy_urn=activity_taxonomy_urn,
                confidence=confidence,
                created_at_pre_publish=created_at_pre_publish,
                updated_at_pre_publish=updated_at_pre_publish,
                superseded_by_urn=superseded_by_urn,
            )

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT 1 FROM alpha_factors_v0_1 WHERE urn = ?",
                    (urn,),
                )
                if cur.fetchone() is not None:
                    raise FactorURNAlreadyExistsError(urn)
                try:
                    conn.execute(
                        "INSERT INTO alpha_factors_v0_1 ("
                        " urn, source_urn, factor_pack_urn, category,"
                        " geography_urn, vintage_start, vintage_end,"
                        " published_at, record_jsonb,"
                        " activity_taxonomy_urn, confidence,"
                        " created_at_pre_publish, updated_at_pre_publish,"
                        " superseded_by_urn"
                        ") VALUES ("
                        " ?, ?, ?, ?, ?, ?, ?, ?, ?,"
                        " ?, ?, ?, ?, ?"
                        ")",
                        (
                            urn,
                            source_urn,
                            pack_urn,
                            category,
                            geo_urn,
                            vintage_start,
                            vintage_end,
                            published_at,
                            record_blob,
                            activity_taxonomy_urn,
                            confidence,
                            created_at_pre_publish,
                            updated_at_pre_publish,
                            superseded_by_urn,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    # Race: another writer beat us between SELECT and INSERT.
                    raise FactorURNAlreadyExistsError(urn) from exc
            finally:
                if self._memory_conn is None:
                    conn.close()

        logger.info("alpha_factor_repo: published urn=%s source=%s", urn, source_urn)

        # Phase 2 / WS8 — emit a structured ``factor_publish`` changelog
        # row whenever the orchestrator path was used, so the audit
        # surface (release_manifests + changelog_events) carries the
        # publish event end-to-end. Best-effort: failures here MUST NOT
        # roll back the publish itself.
        if used_orchestrator:
            try:
                operator = "system"
                ext = record.get("extraction") or {}
                if isinstance(ext, dict):
                    op = ext.get("operator")
                    if isinstance(op, str) and op:
                        operator = op
                self.record_changelog_event(
                    event_type="factor_publish",
                    subject_urn=urn,
                    change_class="additive",
                    actor=operator,
                    metadata={"env": self._publish_env, "source_urn": source_urn},
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "alpha_factor_repo: changelog emit failed for urn=%s: %s",
                    urn, exc,
                )

        return urn

    def _publish_pg(
        self,
        urn: str,
        source_urn: Optional[str],
        pack_urn: Optional[str],
        category: Optional[str],
        geo_urn: Optional[str],
        vintage_start: Optional[str],
        vintage_end: Optional[str],
        published_at: Optional[str],
        record_blob: str,
        *,
        activity_taxonomy_urn: Optional[str] = None,
        confidence: Optional[float] = None,
        created_at_pre_publish: Optional[str] = None,
        updated_at_pre_publish: Optional[str] = None,
        superseded_by_urn: Optional[str] = None,
    ) -> str:
        """Postgres publish path. Lazy import so SQLite users don't pay
        the ``psycopg`` dependency cost.

        The five ``*_pre_publish`` / ``*_urn`` / ``confidence`` /
        ``activity_taxonomy_urn`` keyword arguments mirror the V506
        additive contract fields. Each is OPTIONAL (default ``None``);
        callers that only know about the pre-amendment column set
        continue to work without modification.
        """
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Postgres DSN requires the 'psycopg' driver; install with "
                "`pip install greenlang[server]`."
            ) from exc

        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM factors_v0_1.factor WHERE urn = %s",
                    (urn,),
                )
                if cur.fetchone() is not None:
                    raise FactorURNAlreadyExistsError(urn)
                try:
                    cur.execute(
                        "INSERT INTO factors_v0_1.factor ("
                        " urn, source_urn, factor_pack_urn, category,"
                        " geography_urn, vintage_start, vintage_end,"
                        " published_at, record_jsonb"
                        ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                        (
                            urn,
                            source_urn,
                            pack_urn,
                            category,
                            geo_urn,
                            vintage_start,
                            vintage_end,
                            published_at,
                            record_blob,
                        ),
                    )
                except psycopg.errors.UniqueViolation as exc:  # type: ignore[attr-defined]
                    raise FactorURNAlreadyExistsError(urn) from exc
            conn.commit()
        return urn

    def get_by_urn(self, urn: str) -> Optional[Dict[str, Any]]:
        """Return the JSON-decoded ``record_jsonb`` directly (no coercion).

        Returns ``None`` if no record matches the URN.
        """
        if not isinstance(urn, str) or not urn:
            return None
        if self._is_postgres:
            return self._get_by_urn_pg(urn)
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT record_jsonb FROM alpha_factors_v0_1 WHERE urn = ?",
                (urn,),
            )
            row = cur.fetchone()
        finally:
            if self._memory_conn is None:
                conn.close()
        if row is None:
            return None
        return json.loads(row["record_jsonb"])

    def _get_by_urn_pg(self, urn: str) -> Optional[Dict[str, Any]]:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return None
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT record_jsonb FROM factors_v0_1.factor WHERE urn = %s",
                    (urn,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        # psycopg returns dict for JSONB columns directly; tolerate str too.
        blob = row[0]
        if isinstance(blob, dict):
            return blob
        if isinstance(blob, str):
            return json.loads(blob)
        return None

    def list_factors(
        self,
        *,
        geography_urn: Optional[str] = None,
        source_urn: Optional[str] = None,
        pack_urn: Optional[str] = None,
        category: Optional[str] = None,
        vintage_start_after: Optional[str] = None,
        vintage_end_before: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Return ``(records, next_cursor)`` filtered & cursor-paginated.

        Filters are AND-combined and bound as parameters (SQL-injection safe).
        Sort order is ``published_at DESC, urn ASC`` so cursors are stable
        even when two rows share the same publish timestamp.
        """
        # Clamp limit to reasonable bounds.
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500

        # Build the WHERE clause from the column whitelist only — never
        # interpolate user-controlled identifiers.
        where: List[str] = []
        params: List[Any] = []
        for col, val in (
            ("geography_urn", geography_urn),
            ("source_urn", source_urn),
            ("factor_pack_urn", pack_urn),
            ("category", category),
        ):
            if val is not None:
                where.append(f"{col} = ?")
                params.append(val)
        if vintage_start_after is not None:
            where.append("vintage_start > ?")
            params.append(vintage_start_after)
        if vintage_end_before is not None:
            where.append("vintage_end < ?")
            params.append(vintage_end_before)

        # Cursor encodes the last seen (published_at, urn) tuple so pages
        # never overlap or skip. Decoded form: "v1:<published_at>|<urn>".
        cursor_state = _decode_cursor(cursor)
        if cursor_state is not None:
            last_pub, last_urn = cursor_state
            where.append("(published_at < ? OR (published_at = ? AND urn > ?))")
            params.extend([last_pub, last_pub, last_urn])

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = (
            "SELECT urn, published_at, record_jsonb FROM alpha_factors_v0_1"
            + where_sql
            + " ORDER BY published_at DESC, urn ASC LIMIT ?"
        )
        params.append(limit + 1)  # Fetch one extra to detect more-pages.

        if self._is_postgres:
            return self._list_factors_pg(where, params, limit)

        conn = self._connect()
        try:
            cur = conn.execute(sql, tuple(params))
            rows = cur.fetchall()
        finally:
            if self._memory_conn is None:
                conn.close()

        records = [json.loads(r["record_jsonb"]) for r in rows[:limit]]
        next_cursor: Optional[str] = None
        if len(rows) > limit:
            last_keep = rows[limit - 1]
            next_cursor = _encode_cursor(last_keep["published_at"], last_keep["urn"])
        return records, next_cursor

    def _list_factors_pg(
        self, where: List[str], params: List[Any], limit: int
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return [], None
        # psycopg uses %s placeholders; rewrite the ? from above.
        where_pg = [w.replace("?", "%s") for w in where]
        where_sql = (" WHERE " + " AND ".join(where_pg)) if where_pg else ""
        sql = (
            "SELECT urn, published_at, record_jsonb FROM factors_v0_1.factor"
            + where_sql
            + " ORDER BY published_at DESC, urn ASC LIMIT %s"
        )
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows[:limit]:
            blob = r[2]
            out.append(blob if isinstance(blob, dict) else json.loads(blob))
        next_cursor: Optional[str] = None
        if len(rows) > limit:
            keep = rows[limit - 1]
            pub_str = (
                keep[1].isoformat()
                if hasattr(keep[1], "isoformat")
                else str(keep[1])
            )
            next_cursor = _encode_cursor(pub_str, keep[0])
        return out, next_cursor

    def list_sources(self) -> List[Dict[str, Any]]:
        """Return the alpha-flagged source registry rows."""
        try:
            from greenlang.factors.source_registry import alpha_v0_1_sources
            rows = alpha_v0_1_sources()
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_sources: registry load failed: %s", exc)
            return []
        out: List[Dict[str, Any]] = []
        for source_id, item in sorted((rows or {}).items()):
            row = dict(item) if isinstance(item, dict) else {}
            row.setdefault("source_id", source_id)
            out.append(row)
        return out

    def list_packs(
        self, source_urn: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return synthetic packs from the source registry (one per source).

        When the v0.1 alpha catalogue grows real pack metadata, this
        method will switch to a SELECT against an ``alpha_packs_v0_1``
        table; for now we mirror the synthetic shape used by the alpha
        router so the SDK contract is unchanged.
        """
        sources = self.list_sources()
        out: List[Dict[str, Any]] = []
        for item in sources:
            sid = str(item.get("source_id") or "unknown")
            s_urn = str(item.get("urn") or f"urn:gl:source:{sid}")
            if source_urn and s_urn != source_urn:
                continue
            version_str = str(item.get("source_version") or "0.1")
            out.append(
                {
                    "urn": f"urn:gl:pack:{sid}:default:v1",
                    "source_urn": s_urn,
                    "pack_id": "default",
                    "version": version_str,
                    "display_name": item.get("display_name"),
                    "factor_count": None,
                }
            )
        return out

    # -- Phase 2: methodology + alias filters ------------------------------

    def find_by_methodology(self, methodology_urn: str) -> List[Dict[str, Any]]:
        """Return every factor record whose ``methodology_urn`` matches.

        The ``methodology_urn`` is stored *inside* the ``record_jsonb`` blob
        (it is a record-level field, not a mirrored column), so this method
        decodes the blob and filters in-memory. The candidate set is bounded
        by the alpha catalog size (~1.5k rows in v0.1), so a sequential scan
        is acceptable; a future v1.0 migration may promote ``methodology_urn``
        to a mirrored column with a B-tree index.

        Args:
            methodology_urn: The full URN, e.g. ``urn:gl:methodology:ipcc-ar6-tier1``.

        Returns:
            A list of factor record dicts (JSONB-decoded). Empty list if no match.
        """
        if not isinstance(methodology_urn, str) or not methodology_urn:
            return []
        if self._is_postgres:
            return self._find_by_methodology_pg(methodology_urn)

        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT record_jsonb FROM alpha_factors_v0_1"
            )
            rows = cur.fetchall()
        finally:
            if self._memory_conn is None:
                conn.close()

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                rec = json.loads(r["record_jsonb"])
            except (TypeError, ValueError):
                continue
            if rec.get("methodology_urn") == methodology_urn:
                out.append(rec)
        return out

    def _find_by_methodology_pg(self, methodology_urn: str) -> List[Dict[str, Any]]:
        """Postgres path — uses the ``methodology_urn`` first-class column on
        ``factors_v0_1.factor`` (V500 schema)."""
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return []
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT record_jsonb FROM factors_v0_1.factor "
                    "WHERE methodology_urn = %s",
                    (methodology_urn,),
                )
                rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            blob = r[0]
            if isinstance(blob, dict):
                out.append(blob)
            elif isinstance(blob, str):
                try:
                    out.append(json.loads(blob))
                except ValueError:
                    continue
        return out

    def find_by_activity(
        self, activity_taxonomy_urn: str
    ) -> List[Dict[str, Any]]:
        """Return every factor record whose ``activity_taxonomy_urn`` matches.

        Phase 2 additive contract field (V506 / 2026-04-27 amendment).
        ``activity_taxonomy_urn`` IS mirrored as a first-class column on
        ``alpha_factors_v0_1`` (and on ``factors_v0_1.factor`` via V506),
        so this query is index-backed (partial index covers non-NULL
        rows). Records that pre-date the amendment carry ``NULL`` and
        are correctly excluded.

        Args:
            activity_taxonomy_urn: The full URN, e.g.
                ``urn:gl:activity:ipcc:1-a-1``.

        Returns:
            A list of factor record dicts (JSONB-decoded). Empty list
            if no match.
        """
        if not isinstance(activity_taxonomy_urn, str) or not activity_taxonomy_urn:
            return []
        if self._is_postgres:
            return self._find_by_activity_pg(activity_taxonomy_urn)

        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT record_jsonb FROM alpha_factors_v0_1 "
                "WHERE activity_taxonomy_urn = ?",
                (activity_taxonomy_urn,),
            )
            rows = cur.fetchall()
        finally:
            if self._memory_conn is None:
                conn.close()

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                rec = json.loads(r["record_jsonb"])
            except (TypeError, ValueError):
                continue
            out.append(rec)
        return out

    def _find_by_activity_pg(
        self, activity_taxonomy_urn: str
    ) -> List[Dict[str, Any]]:
        """Postgres path — uses the V506 ``activity_taxonomy_urn`` column."""
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return []
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT record_jsonb FROM factors_v0_1.factor "
                    "WHERE activity_taxonomy_urn = %s",
                    (activity_taxonomy_urn,),
                )
                rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            blob = r[0]
            if isinstance(blob, dict):
                out.append(blob)
            elif isinstance(blob, str):
                try:
                    out.append(json.loads(blob))
                except ValueError:
                    continue
        return out

    def find_by_alias(self, legacy_id: str) -> Optional[Dict[str, Any]]:
        """Resolve a legacy ``EF:...`` id to its canonical factor record.

        Joins ``factor_aliases`` against the factor table to return the
        full record. Retired aliases (``retired_at IS NOT NULL``) still
        resolve — callers that need to filter retired aliases should
        check ``record['urn']`` against the V500 immutability semantics
        (corrections are issued via ``supersedes_urn`` on a NEW record).

        Args:
            legacy_id: The legacy identifier (e.g. ``EF:eGRID:RFCE:CO2:2024``).

        Returns:
            The factor record dict, or ``None`` if no alias maps to it
            or the underlying factor row is missing.
        """
        if not isinstance(legacy_id, str) or not legacy_id:
            return None
        if self._is_postgres:
            return self._find_by_alias_pg(legacy_id)

        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT urn FROM alpha_factor_aliases_v0_1 WHERE legacy_id = ?",
                (legacy_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            urn = row["urn"]
            cur2 = conn.execute(
                "SELECT record_jsonb FROM alpha_factors_v0_1 WHERE urn = ?",
                (urn,),
            )
            rec_row = cur2.fetchone()
        finally:
            if self._memory_conn is None:
                conn.close()
        if rec_row is None:
            return None
        return json.loads(rec_row["record_jsonb"])

    def _find_by_alias_pg(self, legacy_id: str) -> Optional[Dict[str, Any]]:
        """Postgres path for :meth:`find_by_alias`."""
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return None
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT f.record_jsonb FROM factors_v0_1.factor_aliases a "
                    "JOIN factors_v0_1.factor f ON f.urn = a.urn "
                    "WHERE a.legacy_id = %s",
                    (legacy_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        blob = row[0]
        if isinstance(blob, dict):
            return blob
        if isinstance(blob, str):
            return json.loads(blob)
        return None

    def register_alias(
        self,
        urn: str,
        legacy_id: str,
        kind: str = "EF",
    ) -> int:
        """Register a legacy id -> canonical URN alias for backfill.

        Args:
            urn: The canonical factor URN; must already exist in the catalog.
            legacy_id: The legacy ``EF:...`` (or custom) identifier.
            kind: ``'EF'`` (default) or ``'custom'``.

        Returns:
            The ``pk_id`` of the newly inserted alias row.

        Raises:
            ValueError: when ``urn`` / ``legacy_id`` are empty, or ``kind``
                is not in {``EF``, ``custom``}.
            FactorURNAlreadyExistsError: re-used here as the conflict marker
                when an alias for the same ``legacy_id`` already exists.
        """
        if not isinstance(urn, str) or not urn:
            raise ValueError("register_alias: urn must be a non-empty string")
        if not isinstance(legacy_id, str) or not legacy_id:
            raise ValueError("register_alias: legacy_id must be a non-empty string")
        if kind not in ("EF", "custom"):
            raise ValueError(
                f"register_alias: kind must be 'EF' or 'custom'; got {kind!r}"
            )

        if self._is_postgres:
            return self._register_alias_pg(urn, legacy_id, kind)

        with self._lock:
            conn = self._connect()
            try:
                try:
                    cur = conn.execute(
                        "INSERT INTO alpha_factor_aliases_v0_1 "
                        " (urn, legacy_id, kind) VALUES (?, ?, ?)",
                        (urn, legacy_id, kind),
                    )
                    pk = cur.lastrowid
                except sqlite3.IntegrityError as exc:
                    raise FactorURNAlreadyExistsError(legacy_id) from exc
            finally:
                if self._memory_conn is None:
                    conn.close()
        logger.info(
            "alpha_factor_repo: registered alias legacy=%s urn=%s kind=%s",
            legacy_id, urn, kind,
        )
        return int(pk or 0)

    def _register_alias_pg(self, urn: str, legacy_id: str, kind: str) -> int:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Postgres DSN requires the 'psycopg' driver."
            ) from exc
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "INSERT INTO factors_v0_1.factor_aliases "
                        " (urn, legacy_id, kind) VALUES (%s, %s, %s) "
                        "RETURNING pk_id",
                        (urn, legacy_id, kind),
                    )
                    row = cur.fetchone()
                except psycopg.errors.UniqueViolation as exc:  # type: ignore[attr-defined]
                    raise FactorURNAlreadyExistsError(legacy_id) from exc
            conn.commit()
        return int(row[0]) if row else 0

    # -- Phase 2: source artifact + provenance edge registration -----------

    def register_artifact(
        self,
        sha256: str,
        source_urn: str,
        version: str,
        uri: str,
        parser_meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Register a raw source artifact and return its ``pk_id``.

        Args:
            sha256: Lowercase hex SHA-256 of the artifact bytes.
            source_urn: The owning source URN.
            version: The source version this artifact belongs to.
            uri: A pointer to the artifact bytes (``s3://...`` or local).
            parser_meta: Optional dict carrying any of
                ``parser_id``, ``parser_version``, ``parser_commit``,
                ``content_type``, ``size_bytes``, ``metadata``.

        Returns:
            The artifact ``pk_id`` (autoincrement).

        Raises:
            ValueError: when sha256 is not a 64-char lowercase hex string,
                or when any required argument is empty.
            FactorURNAlreadyExistsError: re-used as the conflict marker
                when an artifact with the same sha256 already exists.
        """
        if not isinstance(sha256, str) or not _SHA256_RE.match(sha256 or ""):
            raise ValueError(
                "register_artifact: sha256 must be a 64-char lowercase hex string"
            )
        if not isinstance(source_urn, str) or not source_urn:
            raise ValueError("register_artifact: source_urn must be non-empty")
        if not isinstance(version, str) or not version:
            raise ValueError("register_artifact: version must be non-empty")
        if not isinstance(uri, str) or not uri:
            raise ValueError("register_artifact: uri must be non-empty")

        meta = parser_meta or {}
        parser_id = meta.get("parser_id")
        parser_version = meta.get("parser_version")
        parser_commit = meta.get("parser_commit")
        content_type = meta.get("content_type")
        size_bytes = meta.get("size_bytes")
        extra_metadata = meta.get("metadata")
        meta_blob = (
            json.dumps(extra_metadata, sort_keys=True, default=str)
            if extra_metadata is not None
            else None
        )

        if self._is_postgres:
            return self._register_artifact_pg(
                sha256, source_urn, version, uri,
                content_type, size_bytes,
                parser_id, parser_version, parser_commit,
                meta_blob,
            )

        with self._lock:
            conn = self._connect()
            try:
                try:
                    cur = conn.execute(
                        "INSERT INTO alpha_source_artifacts_v0_1 "
                        " (sha256, source_urn, source_version, uri,"
                        "  content_type, size_bytes,"
                        "  parser_id, parser_version, parser_commit, metadata)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            sha256, source_urn, version, uri,
                            content_type, size_bytes,
                            parser_id, parser_version, parser_commit,
                            meta_blob,
                        ),
                    )
                    pk = cur.lastrowid
                except sqlite3.IntegrityError as exc:
                    raise FactorURNAlreadyExistsError(sha256) from exc
            finally:
                if self._memory_conn is None:
                    conn.close()
        logger.info(
            "alpha_factor_repo: registered artifact sha256=%s source=%s version=%s",
            sha256, source_urn, version,
        )
        return int(pk or 0)

    def _register_artifact_pg(
        self,
        sha256: str,
        source_urn: str,
        version: str,
        uri: str,
        content_type: Optional[str],
        size_bytes: Optional[int],
        parser_id: Optional[str],
        parser_version: Optional[str],
        parser_commit: Optional[str],
        meta_blob: Optional[str],
    ) -> int:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Postgres DSN requires the 'psycopg' driver."
            ) from exc
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "INSERT INTO factors_v0_1.source_artifacts "
                        " (sha256, source_urn, source_version, uri,"
                        "  content_type, size_bytes,"
                        "  parser_id, parser_version, parser_commit, metadata)"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)"
                        " RETURNING pk_id",
                        (
                            sha256, source_urn, version, uri,
                            content_type, size_bytes,
                            parser_id, parser_version, parser_commit,
                            meta_blob,
                        ),
                    )
                    row = cur.fetchone()
                except psycopg.errors.UniqueViolation as exc:  # type: ignore[attr-defined]
                    raise FactorURNAlreadyExistsError(sha256) from exc
            conn.commit()
        return int(row[0]) if row else 0

    def link_provenance(
        self,
        factor_urn: str,
        artifact_pk: int,
        row_ref: str,
        edge_type: str = "extraction",
    ) -> int:
        """Insert a provenance_edges row linking a factor URN to an artifact.

        Args:
            factor_urn: The canonical factor URN.
            artifact_pk: The ``pk_id`` returned by :meth:`register_artifact`.
            row_ref: A row-level pointer inside the artifact (e.g.
                ``"sheet=Annex2;row=42"``).
            edge_type: One of ``extraction`` (default), ``derivation``,
                ``correction``, ``supersedes``.

        Returns:
            The ``pk_id`` of the inserted edge row.

        Raises:
            ValueError: empty factor_urn / row_ref, invalid edge_type, or
                artifact_pk that is not a positive integer.
            FactorURNAlreadyExistsError: re-used as conflict marker when
                the same edge already exists (UNIQUE constraint).
        """
        if not isinstance(factor_urn, str) or not factor_urn:
            raise ValueError("link_provenance: factor_urn must be non-empty")
        if not isinstance(artifact_pk, int) or artifact_pk <= 0:
            raise ValueError("link_provenance: artifact_pk must be a positive int")
        if not isinstance(row_ref, str) or not row_ref:
            raise ValueError("link_provenance: row_ref must be non-empty")
        if edge_type not in _PROVENANCE_EDGE_TYPES:
            raise ValueError(
                f"link_provenance: edge_type must be one of {sorted(_PROVENANCE_EDGE_TYPES)}; "
                f"got {edge_type!r}"
            )

        if self._is_postgres:
            return self._link_provenance_pg(
                factor_urn, artifact_pk, row_ref, edge_type
            )

        with self._lock:
            conn = self._connect()
            try:
                try:
                    cur = conn.execute(
                        "INSERT INTO alpha_provenance_edges_v0_1 "
                        " (factor_urn, source_artifact_pk, row_ref, edge_type)"
                        " VALUES (?, ?, ?, ?)",
                        (factor_urn, artifact_pk, row_ref, edge_type),
                    )
                    pk = cur.lastrowid
                except sqlite3.IntegrityError as exc:
                    raise FactorURNAlreadyExistsError(
                        f"{factor_urn}|{artifact_pk}|{row_ref}|{edge_type}"
                    ) from exc
            finally:
                if self._memory_conn is None:
                    conn.close()
        return int(pk or 0)

    def _link_provenance_pg(
        self,
        factor_urn: str,
        artifact_pk: int,
        row_ref: str,
        edge_type: str,
    ) -> int:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Postgres DSN requires the 'psycopg' driver."
            ) from exc
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "INSERT INTO factors_v0_1.provenance_edges "
                        " (factor_urn, source_artifact_pk, row_ref, edge_type)"
                        " VALUES (%s, %s, %s, %s) RETURNING pk_id",
                        (factor_urn, artifact_pk, row_ref, edge_type),
                    )
                    row = cur.fetchone()
                except psycopg.errors.UniqueViolation as exc:  # type: ignore[attr-defined]
                    raise FactorURNAlreadyExistsError(
                        f"{factor_urn}|{artifact_pk}|{row_ref}|{edge_type}"
                    ) from exc
            conn.commit()
        return int(row[0]) if row else 0

    def record_changelog_event(
        self,
        event_type: str,
        subject_urn: Optional[str],
        change_class: Optional[str],
        actor: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        schema_version: Optional[str] = None,
        migration_note_uri: Optional[str] = None,
    ) -> int:
        """Append an entry to the changelog_events audit table.

        Args:
            event_type: One of the eight allowed event types
                (``schema_change``, ``factor_publish``, ``factor_supersede``,
                ``factor_deprecate``, ``source_add``, ``source_deprecate``,
                ``pack_release``, ``migration_apply``).
            subject_urn: The URN the event is about (factor / source / pack);
                may be ``None`` for global schema events.
            change_class: One of ``additive`` / ``breaking`` / ``deprecated``
                / ``removed``, or ``None``.
            actor: Who performed the change (operator id, service name).
            metadata: Optional JSON metadata.
            schema_version: Schema version for ``schema_change`` events.
            migration_note_uri: Pointer to the migration note for breaking
                / deprecated changes.

        Returns:
            The ``pk_id`` of the new row.
        """
        if event_type not in _CHANGELOG_EVENT_TYPES:
            raise ValueError(
                f"record_changelog_event: event_type must be one of "
                f"{sorted(_CHANGELOG_EVENT_TYPES)}; got {event_type!r}"
            )
        if change_class is not None and change_class not in _CHANGELOG_CHANGE_CLASSES:
            raise ValueError(
                f"record_changelog_event: change_class must be in "
                f"{sorted(_CHANGELOG_CHANGE_CLASSES)} or None; got {change_class!r}"
            )
        if not isinstance(actor, str) or not actor:
            raise ValueError("record_changelog_event: actor must be non-empty")

        meta_blob = (
            json.dumps(metadata, sort_keys=True, default=str)
            if metadata is not None
            else None
        )

        if self._is_postgres:
            return self._record_changelog_event_pg(
                event_type, subject_urn, change_class, actor,
                meta_blob, schema_version, migration_note_uri,
            )

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO alpha_changelog_events_v0_1 "
                    " (event_type, schema_version, subject_urn, change_class,"
                    "  migration_note_uri, actor, metadata)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        event_type, schema_version, subject_urn, change_class,
                        migration_note_uri, actor, meta_blob,
                    ),
                )
                pk = cur.lastrowid
            finally:
                if self._memory_conn is None:
                    conn.close()
        return int(pk or 0)

    def _record_changelog_event_pg(
        self,
        event_type: str,
        subject_urn: Optional[str],
        change_class: Optional[str],
        actor: str,
        meta_blob: Optional[str],
        schema_version: Optional[str],
        migration_note_uri: Optional[str],
    ) -> int:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Postgres DSN requires the 'psycopg' driver."
            ) from exc
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO factors_v0_1.changelog_events "
                    " (event_type, schema_version, subject_urn, change_class,"
                    "  migration_note_uri, actor, metadata)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)"
                    " RETURNING pk_id",
                    (
                        event_type, schema_version, subject_urn, change_class,
                        migration_note_uri, actor, meta_blob,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row[0]) if row else 0

    # -- diagnostics -------------------------------------------------------

    def count(self) -> int:
        """Total number of stored records (mostly used by tests)."""
        if self._is_postgres:
            try:
                import psycopg  # type: ignore  # noqa: PLC0415
            except ImportError:
                return 0
            with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM factors_v0_1.factor")
                    row = cur.fetchone()
            return int(row[0]) if row else 0
        conn = self._connect()
        try:
            cur = conn.execute("SELECT COUNT(*) FROM alpha_factors_v0_1")
            row = cur.fetchone()
        finally:
            if self._memory_conn is None:
                conn.close()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Cursor encoding — opaque to clients, stable round-trip
# ---------------------------------------------------------------------------


_CURSOR_PREFIX = "v1:"
_CURSOR_SEP = "|"


def _encode_cursor(published_at: Any, urn: str) -> str:
    """Encode the last-seen (published_at, urn) tuple as an opaque cursor."""
    pub = published_at if isinstance(published_at, str) else (
        published_at.isoformat()
        if hasattr(published_at, "isoformat")
        else _now_iso()
    )
    return f"{_CURSOR_PREFIX}{pub}{_CURSOR_SEP}{urn}"


def _decode_cursor(cursor: Optional[str]) -> Optional[Tuple[str, str]]:
    """Decode the opaque cursor; ``None`` means start-of-list."""
    if not cursor or not isinstance(cursor, str):
        return None
    if not cursor.startswith(_CURSOR_PREFIX):
        return None
    body = cursor[len(_CURSOR_PREFIX):]
    if _CURSOR_SEP not in body:
        return None
    pub, urn = body.split(_CURSOR_SEP, 1)
    if not pub or not urn:
        return None
    return pub, urn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
