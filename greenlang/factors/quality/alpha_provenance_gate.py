# -*- coding: utf-8 -*-
"""
GreenLang Factors v0.1 Alpha — Provenance Gate (WS2-T1).

Hard gate that REJECTS any factor record missing the alpha-required
provenance/review metadata. Sits in front of every ingestion / normalisation
path that targets the v0.1 Alpha catalogue.

Two layers of enforcement (both run, both must pass):

1. JSON-Schema validation against
   ``config/schemas/factor_record_v0_1.schema.json`` (FROZEN 2026-04-25)
   using ``jsonschema.Draft202012Validator``. Every error is collected.

2. Additional alpha-only constraints that the schema cannot fully express:

   * ``extraction.raw_artifact_sha256`` — lower-case 64-char hex
   * ``extraction.parser_commit``        — 7-40 hex chars
   * ``extraction.parser_version``       — semver (``MAJOR.MINOR.PATCH``,
     optional ``-prerelease`` suffix)
   * ``extraction.operator``             — ``bot:<id>`` or ``human:<email>``
   * ``review.review_status == 'approved'`` requires ``approved_by`` AND
     ``approved_at``
   * ``review.review_status == 'rejected'`` requires ``rejection_reason``
   * ``gwp_basis`` must be ``"ar6"`` (alpha-only basis)

Notes
-----
- Pure-Python; no DB calls; safe to call from any process or thread.
- The compiled ``Draft202012Validator`` is memoised on the instance so
  repeated calls avoid re-parsing the schema.
- Soft check: :py:meth:`AlphaProvenanceGate.check_alpha_source` returns
  ``True`` iff the record's ``source_urn`` matches an entry in
  ``source_registry.yaml`` flagged ``alpha_v0_1: true``. This does NOT
  fail validation — it is a discoverability hint for callers.

CTO doc references: §6.1, §19.1 (FY27 Q1 alpha), Wave B / TaskCreate #5.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Iterable, List, Optional

from jsonschema import Draft202012Validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default schema location
# ---------------------------------------------------------------------------

# greenlang/factors/quality/alpha_provenance_gate.py -> repo_root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SCHEMA_PATH = (
    _REPO_ROOT / "config" / "schemas" / "factor_record_v0_1.schema.json"
)


# ---------------------------------------------------------------------------
# Regex constants — keep in sync with the schema
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_PARSER_COMMIT_RE = re.compile(r"^[a-f0-9]{7,40}$")
# Semver: MAJOR.MINOR.PATCH with optional ``-prerelease`` segment.
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[A-Za-z0-9.-]+)?$"
)
# bot:<id>  OR  human:<email>
_OPERATOR_RE = re.compile(
    r"^(bot:[a-z0-9_.-]+|human:[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})$"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AlphaProvenanceGateError(Exception):
    """Raised when a factor record fails the Alpha Provenance Gate.

    ``args[0]`` is a single newline-joined string containing every failure
    reason discovered during the run. ``self.failures`` exposes the same
    list for programmatic inspection.
    """

    def __init__(self, failures: List[str]):
        self.failures: List[str] = list(failures)
        message = (
            f"AlphaProvenanceGate rejected record ({len(failures)} failure"
            f"{'s' if len(failures) != 1 else ''}):\n  - "
            + "\n  - ".join(failures)
        )
        super().__init__(message)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class AlphaProvenanceGate:
    """v0.1 Alpha provenance enforcer.

    Validates a factor record against the ``factor_record_v0_1`` schema and
    the additional alpha gate constraints (see module docstring).

    Thread-safe: instance-level state is read-only after ``__init__``; the
    compiled validator is held under a small lock for first-time
    initialisation, but subsequent calls take no locks.
    """

    REQUIRED_EXTRACTION_FIELDS = (
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

    REQUIRED_REVIEW_FIELDS = ("review_status", "reviewer", "reviewed_at")

    def __init__(self, schema_path: Optional[Path] = None) -> None:
        self._schema_path: Path = Path(schema_path) if schema_path else _DEFAULT_SCHEMA_PATH
        self._lock = threading.Lock()
        self._validator: Optional[Draft202012Validator] = None
        self._schema: Optional[dict] = None
        self._alpha_source_urns: Optional[frozenset] = None

    # -- internals ---------------------------------------------------------

    def _ensure_validator(self) -> Draft202012Validator:
        """Lazy-load and memoise the compiled JSON-Schema validator."""
        if self._validator is not None:
            return self._validator
        with self._lock:
            if self._validator is None:
                schema = json.loads(
                    self._schema_path.read_text(encoding="utf-8")
                )
                Draft202012Validator.check_schema(schema)
                self._schema = schema
                self._validator = Draft202012Validator(schema)
        return self._validator  # type: ignore[return-value]

    def _alpha_sources(self) -> frozenset:
        """Lazy-load the set of alpha-flagged source URNs from the registry.

        Returns an empty frozenset if the registry is unavailable or yaml
        is not installed — soft-check failure must never block validation.
        """
        if self._alpha_source_urns is not None:
            return self._alpha_source_urns
        urns: set = set()
        try:
            import yaml  # type: ignore  # noqa: PLC0415

            registry_path = (
                _REPO_ROOT
                / "greenlang"
                / "factors"
                / "data"
                / "source_registry.yaml"
            )
            if registry_path.is_file():
                data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
                for src in (data or {}).get("sources", []) or []:
                    if not isinstance(src, dict):
                        continue
                    if not src.get("alpha_v0_1"):
                        continue
                    urn = src.get("urn")
                    if isinstance(urn, str) and urn:
                        urns.add(urn)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "AlphaProvenanceGate: source_registry load failed (%s); "
                "check_alpha_source() will return False.",
                exc,
            )
        self._alpha_source_urns = frozenset(urns)
        return self._alpha_source_urns

    # -- public API --------------------------------------------------------

    def validate(self, record: dict) -> List[str]:
        """Return a list of human-readable failure reasons.

        Empty list ``[]`` means the record passed every check.
        """
        failures: List[str] = []
        if not isinstance(record, dict):
            return [
                f"record must be a dict; got {type(record).__name__}"
            ]

        # Layer 1 — schema validation. Collect every error.
        validator = self._ensure_validator()
        had_schema_failure = False
        for err in sorted(
            validator.iter_errors(record), key=lambda e: list(e.absolute_path)
        ):
            path = ".".join(str(p) for p in err.absolute_path) or "<root>"
            failures.append(f"schema[{path}]: {err.message}")
            had_schema_failure = True

        # Layer 2 — alpha gate constraints.
        failures.extend(self._check_extraction(record))
        failures.extend(self._check_review(record))
        failures.extend(self._check_gwp_basis(record))

        # Emit the schema-failure counter for the v0.1 alpha Grafana dashboard
        # (panel 5). Failure to emit must NEVER break validation.
        if had_schema_failure:
            try:
                from greenlang.factors.observability.prometheus_exporter import (
                    get_factors_metrics,
                )

                source_label = _extract_source_label(record)
                get_factors_metrics().record_schema_validation_failure(
                    schema="factor_record_v0_1", source=source_label
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "AlphaProvenanceGate: metric emit failed (%s); ignoring.", exc
                )

        return failures

    def assert_valid(self, record: dict) -> None:
        """Raise :class:`AlphaProvenanceGateError` if the record fails any check."""
        failures = self.validate(record)
        if failures:
            # Emit the provenance-rejection counter for the v0.1 alpha Grafana
            # dashboard (panel 6). The reason label uses the first failure
            # bucket prefix (``schema``, ``alpha``, etc.) to keep cardinality
            # bounded. Failure to emit must NEVER break the raise.
            try:
                from greenlang.factors.observability.prometheus_exporter import (
                    get_factors_metrics,
                )

                source_label = _extract_source_label(record)
                reason_label = _bucket_failure_reason(failures[0])
                get_factors_metrics().record_alpha_provenance_rejection(
                    source=source_label, reason=reason_label
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "AlphaProvenanceGate: rejection metric emit failed (%s); ignoring.",
                    exc,
                )
            raise AlphaProvenanceGateError(failures)

    def check_alpha_source(self, record: dict) -> bool:
        """Soft check: is ``record["source_urn"]`` an alpha-listed source?

        Returns ``False`` for non-dict records, missing source_urn, unknown
        URNs, or when the registry cannot be read. Never raises.
        """
        if not isinstance(record, dict):
            return False
        source_urn = record.get("source_urn")
        if not isinstance(source_urn, str) or not source_urn:
            return False
        return source_urn in self._alpha_sources()

    # -- per-block alpha-gate checks --------------------------------------

    def _check_extraction(self, record: dict) -> List[str]:
        failures: List[str] = []
        extraction = record.get("extraction")
        if not isinstance(extraction, dict):
            # Schema layer already flagged this; do not double-report.
            return failures

        # Mandatory presence (covered by schema, but we re-check so
        # callers using the gate without pre-schema-validation get a
        # complete picture).
        for field_name in self.REQUIRED_EXTRACTION_FIELDS:
            if field_name not in extraction or extraction[field_name] in (None, ""):
                failures.append(
                    f"alpha[extraction.{field_name}]: required field missing or empty"
                )

        # Format-level checks (additional to the schema).
        sha = extraction.get("raw_artifact_sha256")
        if isinstance(sha, str) and not _SHA256_RE.match(sha):
            failures.append(
                "alpha[extraction.raw_artifact_sha256]: must be 64 lowercase "
                f"hex chars; got {sha!r}"
            )

        commit = extraction.get("parser_commit")
        if isinstance(commit, str) and not _PARSER_COMMIT_RE.match(commit):
            failures.append(
                "alpha[extraction.parser_commit]: must be 7-40 lowercase hex "
                f"chars; got {commit!r}"
            )

        version = extraction.get("parser_version")
        if isinstance(version, str) and not _SEMVER_RE.match(version):
            failures.append(
                "alpha[extraction.parser_version]: must be semver "
                f"MAJOR.MINOR.PATCH(-prerelease)?; got {version!r}"
            )

        operator = extraction.get("operator")
        if isinstance(operator, str) and not _OPERATOR_RE.match(operator):
            failures.append(
                "alpha[extraction.operator]: must match "
                "'bot:<id>' or 'human:<email>'; got "
                f"{operator!r}"
            )

        return failures

    def _check_review(self, record: dict) -> List[str]:
        failures: List[str] = []
        review = record.get("review")
        if not isinstance(review, dict):
            return failures

        for field_name in self.REQUIRED_REVIEW_FIELDS:
            if field_name not in review or review[field_name] in (None, ""):
                failures.append(
                    f"alpha[review.{field_name}]: required field missing or empty"
                )

        status = review.get("review_status")
        if status == "approved":
            for field_name in ("approved_by", "approved_at"):
                if field_name not in review or review[field_name] in (None, ""):
                    failures.append(
                        f"alpha[review.{field_name}]: required when "
                        "review_status == 'approved'"
                    )
        elif status == "rejected":
            reason = review.get("rejection_reason")
            if reason in (None, ""):
                failures.append(
                    "alpha[review.rejection_reason]: required when "
                    "review_status == 'rejected'"
                )

        return failures

    def _check_gwp_basis(self, record: dict) -> List[str]:
        gwp_basis = record.get("gwp_basis")
        if gwp_basis != "ar6":
            return [
                "alpha[gwp_basis]: must be 'ar6' in v0.1 alpha; "
                f"got {gwp_basis!r}"
            ]
        return []


# ---------------------------------------------------------------------------
# Bootstrap / normalize helpers
# ---------------------------------------------------------------------------


_GATE_ENV_VAR = "GL_FACTORS_ALPHA_PROVENANCE_GATE"


def alpha_gate_enabled(default_on: bool = False) -> bool:
    """Return True iff the alpha provenance gate should run.

    Resolution rules:

    * If ``GL_FACTORS_ALPHA_PROVENANCE_GATE`` is explicitly set, parse it as
      a truthy/falsy flag (``1/0``, ``true/false``, ``yes/no``, ``on/off``)
      — case-insensitive. Anything unrecognised is treated as the
      ``default_on`` value.
    * Otherwise, fall back to ``default_on`` (the caller's policy).
    """
    raw = os.getenv(_GATE_ENV_VAR)
    if raw is None or raw.strip() == "":
        return default_on
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default_on


def run_gate_on_records(
    records: Iterable[dict],
    *,
    schema_path: Optional[Path] = None,
) -> List[dict]:
    """Run :class:`AlphaProvenanceGate` over an iterable of records.

    Returns a list of ``{"record_index": int, "failures": [str, ...]}``
    entries — empty list means every record passed.
    """
    gate = AlphaProvenanceGate(schema_path=schema_path)
    report: List[dict] = []
    for idx, rec in enumerate(records):
        failures = gate.validate(rec)
        if failures:
            report.append({"record_index": idx, "failures": failures})
    return report


# ---------------------------------------------------------------------------
# Metric label helpers (low cardinality)
# ---------------------------------------------------------------------------


def _extract_source_label(record: dict) -> str:
    """Best-effort source label extraction for the alpha metrics.

    Tries (in order): ``record["source_id"]``, ``record["source_urn"]``
    (parsed for the publication slug), then falls back to ``"unknown"``.
    """
    if not isinstance(record, dict):
        return "unknown"
    sid = record.get("source_id")
    if isinstance(sid, str) and sid:
        return sid
    urn = record.get("source_urn")
    if isinstance(urn, str) and urn:
        # urn:greenlang:source:<publication>:<version>
        parts = urn.split(":")
        if len(parts) >= 4 and parts[3]:
            return parts[3]
    return "unknown"


def _bucket_failure_reason(failure_msg: str) -> str:
    """Bucket a verbose failure string into a low-cardinality reason label.

    The Grafana alert rule for panel 6 fires on rate-of-rejection per source
    + reason, so we MUST keep the label set tiny. Anything else collapses to
    ``other``.
    """
    if not isinstance(failure_msg, str) or not failure_msg:
        return "unknown"
    head = failure_msg.split(":", 1)[0].strip().lower()
    # Common buckets: ``schema[...]``, ``alpha[extraction.x]``, ``alpha[review.x]``,
    # ``alpha[gwp_basis]``.
    if head.startswith("schema["):
        return "schema_violation"
    if head.startswith("alpha[extraction"):
        return "extraction_metadata"
    if head.startswith("alpha[review"):
        return "review_metadata"
    if head.startswith("alpha[gwp_basis"):
        return "gwp_basis"
    return "other"


__all__ = [
    "AlphaProvenanceGate",
    "AlphaProvenanceGateError",
    "alpha_gate_enabled",
    "run_gate_on_records",
]
