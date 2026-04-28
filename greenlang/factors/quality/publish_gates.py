# -*- coding: utf-8 -*-
"""GreenLang Factors v0.1 — Publish-time validation gate orchestrator (Phase 2 / WS8).

Implements the seven CTO-mandated ordered gates that every record MUST pass
before it can land in the canonical ``factors_v0_1.factor`` table:

    1. Schema validation        (FactorRecordV0_1.model_validate)
    2. URN uniqueness           (factor.urn AND factor_aliases.legacy_id)
    3. Ontology FK references   (geography / unit / methodology / activity / source / pack)
    4. Source registry / alpha  (source flagged alpha_v0_1=true OR env=dev)
    5. Licence match            (record.licence == registry.licence + redistribution_class)
    6. Provenance completeness  (extraction.* fields + sha256 format)
    7. Lifecycle status         (review.review_status approved for production)

Each gate is idempotent + composable. The orchestrator runs gates in order
and aborts on the first failure with a typed exception. ``dry_run`` runs
every gate and returns a per-gate :class:`GateResult` so a CLI / dashboard
can show the full failure matrix.

The orchestrator is a strict superset of :class:`AlphaProvenanceGate`:
gate 1 (Pydantic) and gate 6 (provenance) together are at least as strict
as the legacy gate, plus four additional gates that the legacy gate does
not enforce.

CTO doc references: Phase 2 §2.5 (publish gates), §2.7 (acceptance tests),
§19.1 (provenance completeness for alpha sources).
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError as PydanticValidationError

from greenlang.factors.schemas.factor_record_v0_1 import FactorRecordV0_1

logger = logging.getLogger(__name__)


__all__ = [
    "GateOutcome",
    "GateResult",
    "PublishGateError",
    "SchemaValidationError",
    "URNDuplicateError",
    "OntologyReferenceError",
    "SourceRegistryError",
    "LicenceMismatchError",
    "ProvenanceIncompleteError",
    "LifecycleStatusError",
    "PublishGateOrchestrator",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mandatory extraction sub-fields (CTO doc §19.1).
_REQUIRED_EXTRACTION_FIELDS: Tuple[str, ...] = (
    "source_url",
    "source_record_id",
    "source_publication",
    "source_version",
    "raw_artifact_uri",
    "raw_artifact_sha256",
    "parser_id",
    "parser_version",
    "parser_commit",
    "row_ref",
    "ingested_at",
    "operator",
)

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

#: Ontology fields the orchestrator FK-checks. Tuple order is the order
#: gate_3 reports failures (predictable for tests).
_ONTOLOGY_FIELDS: Tuple[Tuple[str, str], ...] = (
    # (record-key, ontology-table-name-on-sqlite/Postgres)
    ("geography_urn", "geography"),
    ("unit_urn", "unit"),
    ("methodology_urn", "methodology"),
    ("source_urn", "source"),
    ("factor_pack_urn", "factor_pack"),
)

#: Environments where a record's source need not be alpha_v0_1.
_DEV_ENVS: frozenset = frozenset({"dev", "development", "test", "staging"})

#: Environments where ontology / source-registry tables MUST exist
#: (fail-CLOSED on missing table). Phase 2 CTO P0 fix (2026-04-27).
_FAIL_CLOSED_ENVS: frozenset = frozenset({"production", "staging"})

#: Map gate id -> human-readable rule.
_GATE_RULES: Dict[str, str] = {
    "gate_1_schema": "FactorRecordV0_1.model_validate(record)",
    "gate_2_urn_uniqueness": "factor.urn unique AND factor_aliases.legacy_id unique",
    "gate_3_ontology_fk": "geography/unit/methodology/source/pack URNs resolve",
    "gate_4_source_registry": "source registered AND alpha_v0_1=true (or env=dev)",
    "gate_5_licence_match": "record.licence == registry.licence + redistribution_class",
    "gate_6_provenance_completeness": "extraction.* present + sha256 hex regex",
    "gate_7_lifecycle_status": "review.review_status='approved' (production)",
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class GateOutcome:
    """Discriminator for :class:`GateResult.outcome`. String constants only."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class GateResult:
    """Outcome of a single gate. Used by :meth:`PublishGateOrchestrator.dry_run`.

    Attributes:
        gate_id: stable identifier (e.g. ``gate_3_ontology_fk``).
        rule: human-readable description of what the gate enforces.
        outcome: one of ``PASS`` / ``FAIL`` / ``NOT_RUN``.
        urn: the URN under test, if known.
        reason: failure reason. Empty string on PASS.
        details: optional dict for structured callers (e.g. CI dashboards).
    """

    gate_id: str
    rule: str
    outcome: str
    urn: Optional[str] = None
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PublishGateError(Exception):
    """Base class for every publish-gate failure.

    Carries the ``gate_id``, ``urn``, and ``reason`` so callers (CLI,
    repository, dashboard) can render a structured rejection notice.
    """

    gate_id: str = ""
    urn: Optional[str] = None
    reason: str = ""

    def __init__(
        self,
        reason: str,
        *,
        gate_id: Optional[str] = None,
        urn: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.gate_id = gate_id or self.gate_id
        self.urn = urn
        self.reason = reason
        self.details: Dict[str, Any] = dict(details or {})
        suffix = f" (urn={urn!r})" if urn else ""
        super().__init__(f"[{self.gate_id}] {reason}{suffix}")


class SchemaValidationError(PublishGateError):
    """Gate 1 — Pydantic mirror rejected the record."""
    gate_id = "gate_1_schema"


class URNDuplicateError(PublishGateError):
    """Gate 2 — duplicate factor URN OR colliding alias mapping."""
    gate_id = "gate_2_urn_uniqueness"


class OntologyReferenceError(PublishGateError):
    """Gate 3 — at least one ontology FK could not be resolved."""
    gate_id = "gate_3_ontology_fk"


class SourceRegistryError(PublishGateError):
    """Gate 4 — source missing from registry OR not flagged alpha_v0_1."""
    gate_id = "gate_4_source_registry"


class LicenceMismatchError(PublishGateError):
    """Gate 5 — record.licence does not match registry pin OR redistribution denied."""
    gate_id = "gate_5_licence_match"


class ProvenanceIncompleteError(PublishGateError):
    """Gate 6 — extraction provenance fields missing / malformed."""
    gate_id = "gate_6_provenance_completeness"


class LifecycleStatusError(PublishGateError):
    """Gate 7 — review.review_status not approved (production)."""
    gate_id = "gate_7_lifecycle_status"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class PublishGateOrchestrator:
    """Run the seven CTO-mandated publish-time validation gates.

    The orchestrator is intentionally **stateless** — it holds references
    to the repository and source-rights service but never mutates them.

    Two entry points:

    * :meth:`assert_publishable` — run gates 1..7 in order; first failure
      raises a typed :class:`PublishGateError` subclass.
    * :meth:`dry_run` — run every gate; never raises. Returns a list of
      :class:`GateResult` (one per gate).

    Each gate is exposed as a public method (e.g. :meth:`gate_3_ontology_fk`)
    so callers can run subsets in isolation (used by tests + CI smoke).

    Args:
        repo: an :class:`AlphaFactorRepository` (or any object exposing the
            same ``_connect``, ``_is_postgres``, ``get_by_urn``,
            ``find_by_alias`` surface).
        source_rights: optional :class:`SourceRightsService` for licence
            and redistribution_class enforcement. When ``None``, gate 5
            falls back to a registry-only licence comparison.
        env: ``'production'`` (default) / ``'staging'`` / ``'dev'``. Affects
            gate 4 (alpha_v0_1 strictness) and gate 7 (review status).
    """

    def __init__(
        self,
        repo: Any,
        *,
        source_rights: Optional[Any] = None,
        env: str = "production",
    ) -> None:
        if repo is None:
            raise ValueError("PublishGateOrchestrator: repo must not be None")
        self.repo = repo
        self.source_rights = source_rights
        self.env = (env or "production").strip().lower()
        # Lazy-loaded source registry index (URN -> dict). The registry is
        # read once per orchestrator; tests instantiate fresh orchestrators
        # so there is no staleness risk.
        self._registry_index: Optional[Dict[str, Dict[str, Any]]] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def assert_publishable(self, record: Dict[str, Any]) -> None:
        """Run gates 1..7. First failure raises a typed exception.

        Args:
            record: candidate v0.1 factor record (raw dict).

        Raises:
            SchemaValidationError, URNDuplicateError, OntologyReferenceError,
            SourceRegistryError, LicenceMismatchError,
            ProvenanceIncompleteError, LifecycleStatusError.
        """
        if not isinstance(record, dict):
            raise SchemaValidationError(
                f"record must be a dict; got {type(record).__name__}",
                urn=None,
            )

        # Gates run in order; each raises its own typed exception.
        self.gate_1_schema(record)
        self.gate_2_urn_uniqueness(record)
        self.gate_3_ontology_fk(record)
        self.gate_4_source_registry(record)
        self.gate_5_licence_match(record)
        self.gate_6_provenance_completeness(record)
        self.gate_7_lifecycle_status(record)

        logger.info(
            "publish_gates: all 7 gates PASS for urn=%s env=%s",
            record.get("urn"),
            self.env,
        )

    def dry_run(self, record: Dict[str, Any]) -> List[GateResult]:
        """Run every gate; never raise. Return ordered per-gate results.

        Args:
            record: candidate v0.1 factor record.

        Returns:
            List of seven :class:`GateResult` objects (one per gate),
            each PASS / FAIL / NOT_RUN. NOT_RUN is reserved for the case
            where a gate cannot execute (e.g. record is not a dict and
            we short-circuit gates 2..7).
        """
        results: List[GateResult] = []
        urn = record.get("urn") if isinstance(record, dict) else None

        # If the record isn't a dict we can't even attempt any other gate.
        if not isinstance(record, dict):
            results.append(
                GateResult(
                    gate_id="gate_1_schema",
                    rule=_GATE_RULES["gate_1_schema"],
                    outcome=GateOutcome.FAIL,
                    urn=None,
                    reason=f"record must be a dict; got {type(record).__name__}",
                )
            )
            for gate_id in (
                "gate_2_urn_uniqueness",
                "gate_3_ontology_fk",
                "gate_4_source_registry",
                "gate_5_licence_match",
                "gate_6_provenance_completeness",
                "gate_7_lifecycle_status",
            ):
                results.append(
                    GateResult(
                        gate_id=gate_id,
                        rule=_GATE_RULES[gate_id],
                        outcome=GateOutcome.NOT_RUN,
                        urn=None,
                        reason="upstream gate failed; gate skipped",
                    )
                )
            return results

        # Each individual gate may pass or raise; we capture both.
        gate_methods: Sequence[Tuple[str, Any]] = (
            ("gate_1_schema", self.gate_1_schema),
            ("gate_2_urn_uniqueness", self.gate_2_urn_uniqueness),
            ("gate_3_ontology_fk", self.gate_3_ontology_fk),
            ("gate_4_source_registry", self.gate_4_source_registry),
            ("gate_5_licence_match", self.gate_5_licence_match),
            ("gate_6_provenance_completeness", self.gate_6_provenance_completeness),
            ("gate_7_lifecycle_status", self.gate_7_lifecycle_status),
        )
        for gate_id, method in gate_methods:
            try:
                method(record)
                results.append(
                    GateResult(
                        gate_id=gate_id,
                        rule=_GATE_RULES[gate_id],
                        outcome=GateOutcome.PASS,
                        urn=urn,
                    )
                )
            except PublishGateError as exc:
                results.append(
                    GateResult(
                        gate_id=gate_id,
                        rule=_GATE_RULES[gate_id],
                        outcome=GateOutcome.FAIL,
                        urn=exc.urn or urn,
                        reason=exc.reason,
                        details=dict(exc.details),
                    )
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.exception(
                    "publish_gates: unexpected exception in %s for urn=%s",
                    gate_id, urn,
                )
                results.append(
                    GateResult(
                        gate_id=gate_id,
                        rule=_GATE_RULES[gate_id],
                        outcome=GateOutcome.FAIL,
                        urn=urn,
                        reason=f"unexpected error: {exc}",
                    )
                )
        return results

    # -----------------------------------------------------------------
    # Individual gates
    # -----------------------------------------------------------------

    def gate_1_schema(self, record: Dict[str, Any]) -> None:
        """Gate 1 — Pydantic schema validation against the frozen contract."""
        urn = record.get("urn") if isinstance(record, dict) else None
        try:
            FactorRecordV0_1.model_validate(record)
        except PydanticValidationError as exc:
            # Pydantic returns a structured error list. Compress to a
            # single human-readable string + keep the structured form
            # in ``details`` for programmatic callers.
            errors = exc.errors()
            head = errors[0] if errors else {}
            loc = ".".join(str(x) for x in head.get("loc", ())) or "<root>"
            msg = head.get("msg") or str(exc)
            reason = (
                f"schema validation failed at {loc!r}: {msg} "
                f"({len(errors)} error(s) total)"
            )
            raise SchemaValidationError(
                reason,
                urn=urn,
                details={"errors": errors},
            ) from exc

    def gate_2_urn_uniqueness(self, record: Dict[str, Any]) -> None:
        """Gate 2 — primary URN unique AND alias not bound to a different URN."""
        urn = record.get("urn")
        if not isinstance(urn, str) or not urn:
            # Schema gate would have caught this; defensive only.
            raise URNDuplicateError(
                "record.urn missing or non-string",
                urn=None,
            )

        # Primary URN check: query the factor table directly via the
        # repo's connect path so we don't have to clone its private SQL.
        existing = self.repo.get_by_urn(urn)
        if existing is not None:
            raise URNDuplicateError(
                f"factor URN already exists in catalog",
                urn=urn,
            )

        # Alias check: if the record carries factor_id_alias, ensure no
        # OTHER URN claims that alias.
        alias = record.get("factor_id_alias")
        if isinstance(alias, str) and alias:
            try:
                existing_alias_record = self.repo.find_by_alias(alias)
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "gate_2: find_by_alias raised for legacy_id=%r: %s; treating as no-match",
                    alias, exc,
                )
                existing_alias_record = None
            if existing_alias_record is not None:
                bound_urn = existing_alias_record.get("urn")
                if bound_urn != urn:
                    raise URNDuplicateError(
                        f"factor_id_alias {alias!r} is already bound to a "
                        f"different urn ({bound_urn!r})",
                        urn=urn,
                        details={"alias": alias, "bound_urn": bound_urn},
                    )

    def gate_3_ontology_fk(self, record: Dict[str, Any]) -> None:
        """Gate 3 — every ontology URN must resolve in its respective table.

        The orchestrator probes each ontology table via a simple
        ``SELECT 1 FROM <table> WHERE urn = ?`` query (bind parameter,
        never string-interpolated).

        Production / staging fail CLOSED (Phase 2 CTO P0 fix, 2026-04-27):
          * If the ontology table does NOT exist on this backend, gate 3
            raises :class:`OntologyReferenceError` immediately. Production
            publish requires a fully-seeded ontology.
          * Dev preserves the legacy warn-and-skip behaviour so engineers
            without a fully-seeded checkout can still iterate.

        Tables not referenced by the record (i.e. ``record[field]`` is
        empty) are not probed regardless of env.
        """
        urn = record.get("urn")
        is_fail_closed = self.env in _FAIL_CLOSED_ENVS

        misses: List[Tuple[str, str]] = []
        missing_tables: List[Tuple[str, str]] = []
        for field_key, table_name in _ONTOLOGY_FIELDS:
            value = record.get(field_key)
            if not isinstance(value, str) or not value:
                # gate_1 already enforces presence for required URN
                # fields. For optional fields (e.g. activity_taxonomy_urn,
                # currently not in v0.1 surface) we simply skip.
                continue

            # Production/staging: assert the table is reachable BEFORE
            # the row probe. Missing table -> fatal in fail-closed envs;
            # warn-and-skip in dev / legacy.
            if is_fail_closed and not self._assert_ontology_table_present(
                table_name
            ):
                missing_tables.append((field_key, table_name))
                continue

            ok = self._ontology_lookup(table_name, value)
            if ok is False:
                misses.append((field_key, value))
            # ok is None means the table is unreachable on this backend.
            # In dev that is the warn-and-skip path; in fail-closed envs
            # we already short-circuited above via missing_tables.

        if missing_tables:
            head_field, head_table = missing_tables[0]
            details = [
                {"field": f, "table": t} for (f, t) in missing_tables
            ]
            raise OntologyReferenceError(
                f"Ontology table {head_table!r} not present — "
                f"{self.env} publish requires seeded ontology "
                f"(field={head_field}, "
                f"{len(missing_tables)} missing table(s) total)",
                urn=urn,
                details={"missing_tables": details, "env": self.env},
            )

        if misses:
            details = [{"field": f, "value": v} for (f, v) in misses]
            head_field, head_value = misses[0]
            raise OntologyReferenceError(
                f"ontology FK miss: {head_field}={head_value!r} not in registry "
                f"({len(misses)} miss(es) total)",
                urn=urn,
                details={"misses": details},
            )

    def gate_4_source_registry(self, record: Dict[str, Any]) -> None:
        """Gate 4 — source registered AND (alpha_v0_1 OR env permits).

        Production / staging fail CLOSED (Phase 2 CTO P0 fix, 2026-04-27):
          * If the ``source`` ontology / registry table is not reachable,
            production/staging raise :class:`SourceRegistryError`. Dev
            preserves the legacy fail-open path so engineers without a
            seeded checkout can still iterate.
        """
        urn = record.get("urn")
        source_urn = record.get("source_urn")
        if not isinstance(source_urn, str) or not source_urn:
            # Schema-gate already enforces; defensive only.
            raise SourceRegistryError(
                "record.source_urn missing", urn=urn,
            )

        # Phase 2 CTO P0 — fail-CLOSED on missing source ontology/registry
        # table in production/staging. The registry index lookup below
        # falls back to the YAML file when the SourceRightsService is
        # not wired; we still want to assert the SQL-side ``source``
        # table exists so any production publish that races the seed
        # script fails LOUDLY rather than silently skipping the gate.
        if self.env in _FAIL_CLOSED_ENVS and not self._assert_ontology_table_present(
            "source"
        ):
            raise SourceRegistryError(
                f"Ontology table 'source' not present — {self.env} "
                f"publish requires seeded source registry",
                urn=urn,
                details={"missing_table": "source", "env": self.env},
            )

        registry_row = self._lookup_registry_row(source_urn)
        if registry_row is None:
            raise SourceRegistryError(
                f"source {source_urn!r} not present in source_registry.yaml",
                urn=urn,
                details={"source_urn": source_urn},
            )

        # alpha_v0_1 enforcement — strict in production, relaxed in dev/staging.
        is_alpha = bool(registry_row.get("alpha_v0_1"))
        if not is_alpha and self.env not in _DEV_ENVS:
            raise SourceRegistryError(
                f"source {source_urn!r} is not alpha_v0_1=true; "
                f"production publish denied",
                urn=urn,
                details={"source_urn": source_urn, "env": self.env},
            )

    def gate_5_licence_match(self, record: Dict[str, Any]) -> None:
        """Gate 5 — record.licence aligns with registry pin (and redistribution)."""
        urn = record.get("urn")
        source_urn = record.get("source_urn")
        record_licence = record.get("licence")

        registry_row = self._lookup_registry_row(source_urn) if isinstance(
            source_urn, str
        ) else None

        if registry_row is not None:
            registry_licence = registry_row.get("licence")
            if (
                isinstance(registry_licence, str)
                and registry_licence
                and isinstance(record_licence, str)
                and record_licence
                and record_licence != registry_licence
            ):
                raise LicenceMismatchError(
                    f"record.licence={record_licence!r} does not match "
                    f"registry pin {registry_licence!r} for source {source_urn!r}",
                    urn=urn,
                    details={
                        "record_licence": record_licence,
                        "registry_licence": registry_licence,
                        "source_urn": source_urn,
                    },
                )

        # Redistribution_class enforcement via SourceRightsService when wired.
        if self.source_rights is not None and isinstance(source_urn, str):
            try:
                decision = self.source_rights.check_record_licence_matches_registry(
                    source_urn, record_licence
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "gate_5: SourceRightsService raised; treating as ALLOW: %s",
                    exc,
                )
                decision = None
            if decision is not None and getattr(decision, "denied", False):
                reason = getattr(decision, "reason", "rights service denied")
                raise LicenceMismatchError(
                    f"SourceRightsService denied: {reason}",
                    urn=urn,
                    details={"reason": reason, "source_urn": source_urn},
                )
            # Ingestion gate (redistribution_class enforcement).
            try:
                ing_decision = self.source_rights.check_ingestion_allowed(
                    source_urn
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "gate_5: ingestion check raised; treating as ALLOW: %s",
                    exc,
                )
                ing_decision = None
            if ing_decision is not None and getattr(ing_decision, "denied", False):
                reason = getattr(ing_decision, "reason", "ingestion denied")
                raise LicenceMismatchError(
                    f"SourceRightsService ingestion denied: {reason}",
                    urn=urn,
                    details={"reason": reason, "source_urn": source_urn},
                )

    def gate_6_provenance_completeness(self, record: Dict[str, Any]) -> None:
        """Gate 6 — extraction.* fields complete + sha256 64-hex regex.

        The Pydantic gate (gate 1) catches missing-field cases when present.
        Gate 6 exists for two reasons:
          (a) callers may run individual gates out of order — gate 6 must
              still hard-stop a record missing extraction metadata even if
              gate 1 was skipped;
          (b) when a ``source_artifacts`` row is registered with the same
              sha256 + source_urn, log a positive correlation for the audit
              trail (the v0.1 contract does NOT require pre-registration).
        """
        urn = record.get("urn")
        extraction = record.get("extraction")
        if not isinstance(extraction, dict):
            raise ProvenanceIncompleteError(
                "extraction object missing or not a dict",
                urn=urn,
            )

        missing = [
            f for f in _REQUIRED_EXTRACTION_FIELDS
            if extraction.get(f) in (None, "")
        ]
        if missing:
            raise ProvenanceIncompleteError(
                f"extraction missing required fields: {sorted(missing)}",
                urn=urn,
                details={"missing": sorted(missing)},
            )

        sha = extraction.get("raw_artifact_sha256")
        if not isinstance(sha, str) or not _SHA256_RE.match(sha):
            raise ProvenanceIncompleteError(
                f"extraction.raw_artifact_sha256={sha!r} must be 64 lowercase hex chars",
                urn=urn,
                details={"raw_artifact_sha256": sha},
            )

        # Correlation log only — never fail the gate. v0.1 alpha lets
        # parsers register the artifact AFTER publish.
        source_urn = record.get("source_urn")
        if isinstance(source_urn, str) and source_urn:
            artifact_pk = self._lookup_artifact(sha, source_urn)
            if artifact_pk is None:
                logger.info(
                    "gate_6: artifact sha256=%s source=%s not pre-registered "
                    "(non-fatal in v0.1)",
                    sha, source_urn,
                )

    def gate_7_lifecycle_status(self, record: Dict[str, Any]) -> None:
        """Gate 7 — review.review_status='approved' for production publish."""
        urn = record.get("urn")
        review = record.get("review")
        if not isinstance(review, dict):
            raise LifecycleStatusError(
                "review object missing or not a dict",
                urn=urn,
            )

        status = review.get("review_status")

        # Rejected records never publish, regardless of env.
        if status == "rejected":
            raise LifecycleStatusError(
                "review.review_status='rejected'; rejected records do not publish",
                urn=urn,
                details={"review_status": status},
            )

        if self.env == "production":
            if status != "approved":
                raise LifecycleStatusError(
                    f"production publish requires review.review_status='approved'; "
                    f"got {status!r}",
                    urn=urn,
                    details={"review_status": status, "env": self.env},
                )
            for f in ("approved_by", "approved_at"):
                if not review.get(f):
                    raise LifecycleStatusError(
                        f"production publish requires review.{f}",
                        urn=urn,
                        details={"missing": f, "env": self.env},
                    )
        elif self.env == "staging":
            if status not in ("approved", "pending"):
                raise LifecycleStatusError(
                    f"staging publish requires review_status in "
                    f"{{'approved','pending'}}; got {status!r}",
                    urn=urn,
                    details={"review_status": status, "env": self.env},
                )
        # dev / test: any non-rejected status passes.

    # -----------------------------------------------------------------
    # Backend probes (private helpers)
    # -----------------------------------------------------------------

    def _ontology_lookup(self, table_name: str, urn_value: str) -> Optional[bool]:
        """Return True if URN exists, False if not, None if table unreachable.

        Uses bind parameters; never interpolates ``urn_value`` into SQL.
        ``table_name`` IS interpolated, but only from a closed whitelist
        (:data:`_ONTOLOGY_FIELDS`) so injection is not possible.
        """
        if table_name not in {t for (_f, t) in _ONTOLOGY_FIELDS}:
            # Hard-fail any caller that smuggles in an unknown table.
            logger.error("ontology_lookup: rejected unknown table %r", table_name)
            return None

        is_postgres = bool(getattr(self.repo, "_is_postgres", False))
        if is_postgres:
            return self._ontology_lookup_pg(table_name, urn_value)
        return self._ontology_lookup_sqlite(table_name, urn_value)

    def _assert_ontology_table_present(self, table_name: str) -> bool:
        """Phase 2 CTO P0 — return True iff the named ontology table exists.

        Probes the backend with ``SELECT 1 FROM <table> LIMIT 1`` (bind
        params not needed; ``table_name`` is whitelist-validated by the
        caller). The probe SUCCEEDS even when the table is empty —
        empty-but-present is a legitimate dev-seed condition. The probe
        FAILS only when the table itself is absent or the connection is
        broken; both conditions render the FK enforcement non-functional
        and so production/staging treat them as gate failures.

        Args:
            table_name: A whitelisted ontology table name. Currently:
                geography / unit / methodology / source / factor_pack.

        Returns:
            ``True`` if the table is reachable (regardless of row count),
            ``False`` if the SELECT 1 probe failed for ANY reason other
            than empty results.
        """
        # Defensive whitelist re-check (also performed by gate_3 callers).
        whitelist = {t for (_f, t) in _ONTOLOGY_FIELDS}
        if table_name not in whitelist:
            logger.error(
                "_assert_ontology_table_present: rejected unknown table %r",
                table_name,
            )
            return False

        is_postgres = bool(getattr(self.repo, "_is_postgres", False))
        if is_postgres:
            return self._assert_ontology_table_present_pg(table_name)
        return self._assert_ontology_table_present_sqlite(table_name)

    def _assert_ontology_table_present_sqlite(self, table_name: str) -> bool:
        """SQLite path for :meth:`_assert_ontology_table_present`."""
        try:
            conn = self.repo._connect()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_assert_ontology_table_present_sqlite: cannot get conn: %s",
                exc,
            )
            return False
        try:
            try:
                # SELECT 1 FROM <t> LIMIT 1 — succeeds on empty table,
                # raises sqlite3.OperationalError ("no such table: ...")
                # when the table is absent. Any other exception is also
                # treated as not-present (fail-CLOSED in production).
                conn.execute(
                    f"SELECT 1 FROM {table_name} LIMIT 1"
                ).fetchone()
                return True
            except sqlite3.OperationalError as exc:
                logger.debug(
                    "_assert_ontology_table_present_sqlite: %r missing (%s)",
                    table_name, exc,
                )
                return False
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "_assert_ontology_table_present_sqlite: %r probe failed: %s",
                    table_name, exc,
                )
                return False
        finally:
            mem = getattr(self.repo, "_memory_conn", None)
            if mem is None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

    def _assert_ontology_table_present_pg(self, table_name: str) -> bool:
        """Postgres path for :meth:`_assert_ontology_table_present`."""
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "_assert_ontology_table_present_pg: psycopg unavailable"
            )
            return False
        dsn = getattr(self.repo, "_dsn", None)
        if not dsn:
            logger.warning("_assert_ontology_table_present_pg: repo has no _dsn")
            return False
        try:
            with psycopg.connect(dsn) as conn:  # type: ignore[arg-type]
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT 1 FROM factors_v0_1.{table_name} LIMIT 1"
                    )
                    cur.fetchone()
            return True
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.debug(
                "_assert_ontology_table_present_pg: %r probe failed: %s",
                table_name, exc,
            )
            return False

    def _ontology_lookup_sqlite(
        self, table_name: str, urn_value: str
    ) -> Optional[bool]:
        try:
            conn = self.repo._connect()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("ontology_lookup: cannot get conn: %s", exc)
            return None
        try:
            try:
                cur = conn.execute(
                    f"SELECT 1 FROM {table_name} WHERE urn = ? LIMIT 1",
                    (urn_value,),
                )
                row = cur.fetchone()
            except sqlite3.OperationalError as exc:
                # Table doesn't exist on this backend (common in unit
                # tests where only some ontology tables are seeded).
                logger.debug(
                    "ontology_lookup: table %r missing on sqlite (%s); "
                    "skipping FK enforcement",
                    table_name, exc,
                )
                return None
        finally:
            mem = getattr(self.repo, "_memory_conn", None)
            if mem is None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        return row is not None

    def _ontology_lookup_pg(
        self, table_name: str, urn_value: str
    ) -> Optional[bool]:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            logger.warning("ontology_lookup_pg: psycopg unavailable")
            return None
        dsn = getattr(self.repo, "_dsn", None)
        if not dsn:
            logger.warning("ontology_lookup_pg: repo has no _dsn")
            return None
        try:
            with psycopg.connect(dsn) as conn:  # type: ignore[arg-type]
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT 1 FROM factors_v0_1.{table_name} "
                        "WHERE urn = %s LIMIT 1",
                        (urn_value,),
                    )
                    row = cur.fetchone()
        except Exception as exc:  # noqa: BLE001 — defensive
            # On Postgres, a missing-table error is a hard ops failure
            # that should block publish — log loudly but still return
            # None so we don't false-negative in transient outages.
            logger.error(
                "ontology_lookup_pg: query failed table=%s urn=%s: %s",
                table_name, urn_value, exc,
            )
            return None
        return row is not None

    def _lookup_artifact(
        self, sha256: str, source_urn: str
    ) -> Optional[int]:
        """Return source_artifacts.pk_id if pre-registered, else None.

        Used by gate 6 for correlation logging only. Never raises.
        """
        is_postgres = bool(getattr(self.repo, "_is_postgres", False))
        try:
            if is_postgres:
                return self._lookup_artifact_pg(sha256, source_urn)
            return self._lookup_artifact_sqlite(sha256, source_urn)
        except Exception as exc:  # noqa: BLE001
            logger.debug("lookup_artifact: probe failed: %s", exc)
            return None

    def _lookup_artifact_sqlite(
        self, sha256: str, source_urn: str
    ) -> Optional[int]:
        conn = self.repo._connect()  # type: ignore[attr-defined]
        try:
            try:
                cur = conn.execute(
                    "SELECT pk_id FROM alpha_source_artifacts_v0_1 "
                    "WHERE sha256 = ? AND source_urn = ? LIMIT 1",
                    (sha256, source_urn),
                )
                row = cur.fetchone()
            except sqlite3.OperationalError:
                return None
        finally:
            mem = getattr(self.repo, "_memory_conn", None)
            if mem is None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        if row is None:
            return None
        # sqlite3.Row supports both index + key.
        return int(row[0])

    def _lookup_artifact_pg(
        self, sha256: str, source_urn: str
    ) -> Optional[int]:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return None
        dsn = getattr(self.repo, "_dsn", None)
        if not dsn:
            return None
        with psycopg.connect(dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pk_id FROM factors_v0_1.source_artifacts "
                    "WHERE sha256 = %s AND source_urn = %s LIMIT 1",
                    (sha256, source_urn),
                )
                row = cur.fetchone()
        return int(row[0]) if row else None

    def _lookup_registry_row(self, source_urn: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the source_registry.yaml row for ``source_urn`` (URN-keyed).

        Loaded lazily and cached for the lifetime of the orchestrator.
        Tests that need an isolated registry pass a ``SourceRightsService``
        constructed with their own registry index — the licence-pin lookup
        in :meth:`gate_5_licence_match` then routes through that service.
        """
        if not isinstance(source_urn, str) or not source_urn:
            return None
        if self._registry_index is None:
            self._registry_index = self._load_registry_index()
        return self._registry_index.get(source_urn)

    def _load_registry_index(self) -> Dict[str, Dict[str, Any]]:
        """Load source_registry.yaml as a URN-keyed index.

        On any error returns an empty dict (gate 4 will then fail with a
        "source not in registry" diagnostic, which is the correct safe
        default). When ``source_rights`` is supplied AND it has a
        ``registry_index`` attribute (which :class:`SourceRightsService`
        does), use THAT instead — this lets tests inject a synthetic
        registry without touching the YAML file.
        """
        # Prefer the SourceRightsService's index when wired (test isolation).
        sr = self.source_rights
        if sr is not None:
            ri = getattr(sr, "registry_index", None)
            if isinstance(ri, dict):
                return dict(ri)

        try:
            from greenlang.factors.source_registry import _load_raw_sources  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.warning("publish_gates: source_registry import failed: %s", exc)
            return {}
        try:
            rows = _load_raw_sources()
        except Exception as exc:  # noqa: BLE001
            logger.warning("publish_gates: registry load failed: %s", exc)
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for item in rows or []:
            if not isinstance(item, dict):
                continue
            urn = item.get("urn")
            if isinstance(urn, str) and urn:
                out[urn] = item
        return out
