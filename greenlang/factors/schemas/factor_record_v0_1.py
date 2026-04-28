# -*- coding: utf-8 -*-
"""GreenLang Factors v0.1 Alpha - Pydantic v2 mirror of the FROZEN schema.

This module is the typed Python mirror of the JSON Schema at
``config/schemas/factor_record_v0_1.schema.json`` (FROZEN 2026-04-25).
The JSON Schema remains the canonical source of truth; this mirror gives
Python callers IDE / mypy support, per-field Pydantic validators, and a
:func:`model_to_jsonschema_diff` helper used by the CI gate.

Design
------
The CTO brief (Phase 2 §2.1) requires "one Pydantic class per field
group" with FLAT field aggregation on the top-level record (the JSON
Schema has flat properties; nesting at the Pydantic layer would create
an artificial divergence). We therefore expose:

* Nine ``*Fields`` submodels — one per CTO field group. They subclass
  :class:`GreenLangBase` and are pure documentation aids: they live to
  let static-analysis tools group fields semantically. They are NOT
  composed via inheritance into :class:`FactorRecordV0_1` because
  Pydantic's multiple inheritance with ``extra="forbid"`` and the deep
  cross-field validators (e.g. ``review_status='approved'`` requires
  ``approved_by``) is brittle. Instead the top-level model copies the
  same field declarations and uses ``model_validator`` for cross-field
  rules.
* Two nested models — :class:`ExtractionMetadata` and
  :class:`ReviewMetadata` — which DO mirror the JSON Schema's nested
  objects directly.
* The top-level :class:`FactorRecordV0_1` (subclass of
  :class:`GreenLangRecord`) which is the public surface.

Field-level validators enforce the additional alpha-only constraints
the JSON Schema cannot fully express (URN canonical-parse via
``greenlang.factors.ontology.urn.parse``, ``vintage_end >=
vintage_start``, conditional review-state requirements, etc.).

Usage
-----
::

    from greenlang.factors.schemas.factor_record_v0_1 import FactorRecordV0_1

    record = FactorRecordV0_1(**raw_dict)
    record.model_dump(mode="json")  # round-trips to a JSON-Schema-valid dict

CTO doc references: Phase 2 §2.1, §6.1 (URN canonical form), §19.1
(provenance fields complete for alpha sources).
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import (
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from greenlang.factors.ontology.urn import InvalidUrnError, parse as parse_urn
from greenlang.schemas.base import GreenLangBase, GreenLangRecord

__all__ = [
    "Category",
    "Citation",
    "ClimateBasisFields",
    "ContextFields",
    "ExtractionMetadata",
    "FactorRecordV0_1",
    "FrozenSchemaPath",
    "GwpBasis",
    "IdentityFields",
    "LicenceConstraints",
    "LicenceFields",
    "LifecycleFields",
    "LineageFields",
    "QualityFields",
    "Resolution",
    "ReviewMetadata",
    "ReviewStatus",
    "TimeFields",
    "UncertaintyDistribution",
    "ValueUnitFields",
    "model_to_jsonschema_diff",
]


# ---------------------------------------------------------------------------
# Frozen JSON-Schema lookup
# ---------------------------------------------------------------------------

# greenlang/factors/schemas/factor_record_v0_1.py -> repo_root
_REPO_ROOT = Path(__file__).resolve().parents[3]
FrozenSchemaPath: Path = (
    _REPO_ROOT / "config" / "schemas" / "factor_record_v0_1.schema.json"
)


# ---------------------------------------------------------------------------
# Regex constants — IDENTICAL to the JSON Schema patterns. Keep in sync.
# ---------------------------------------------------------------------------

_URN_FACTOR_RE = re.compile(
    r"^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$"
)
_URN_FACTOR_SUPERSEDES_RE = re.compile(
    r"^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[A-Za-z0-9._-]+){2,4}:v[1-9][0-9]*$"
)
# Reverse pointer (`superseded_by_urn`) — strict lowercase canonical form,
# matching the v0.1 schema regex for ``urn`` itself. Note this is STRICTER
# than ``_URN_FACTOR_SUPERSEDES_RE`` (which retains its original
# uppercase-tolerant pattern for backward compatibility with legacy
# correction trails). New reverse-link writes use the canonical form.
_URN_FACTOR_SUPERSEDED_BY_RE = re.compile(
    r"^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$"
)
_URN_ACTIVITY_RE = re.compile(
    r"^urn:gl:activity:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*)?$"
)
_URN_SOURCE_RE = re.compile(r"^urn:gl:source:[a-z0-9][a-z0-9-]*$")
_URN_PACK_RE = re.compile(
    r"^urn:gl:pack:[a-z0-9][a-z0-9-]*:[a-z0-9][a-z0-9._-]*:v[1-9][0-9]*$"
)
_URN_METHODOLOGY_RE = re.compile(r"^urn:gl:methodology:[a-z0-9][a-z0-9-]*$")
_URN_GEO_RE = re.compile(
    r"^urn:gl:geo:(global|country|subregion|state_or_province|grid_zone|"
    r"bidding_zone|balancing_authority):[a-zA-Z0-9._-]+$"
)
_URN_UNIT_RE = re.compile(r"^urn:gl:unit:[a-zA-Z0-9._/-]+$")
_FACTOR_ID_ALIAS_RE = re.compile(r"^EF:[A-Za-z0-9_.:-]+$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_PARSER_COMMIT_RE = re.compile(r"^[a-f0-9]{7,40}$")
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[A-Za-z0-9.-]+)?$"
)
_OPERATOR_RE = re.compile(
    r"^(bot:[a-z0-9_.-]+|human:[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})$"
)
_HUMAN_EMAIL_RE = re.compile(
    r"^human:[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)


# ---------------------------------------------------------------------------
# Pattern lookup table (used by the diff helper).
#
# Pydantic v2 doesn't surface patterns enforced inside ``field_validator``
# in ``model_json_schema()``. We mirror the regex strings from the frozen
# JSON Schema verbatim here so the diff helper can do a pure-text
# comparison against the frozen schema's pattern keywords.
#
# Keep these strings IDENTICAL to the JSON Schema's pattern keywords —
# the test ``test_pydantic_patterns_match_frozen_schema`` will fail
# otherwise.
# ---------------------------------------------------------------------------

_PYDANTIC_FIELD_PATTERNS: Dict[str, str] = {
    "urn": (
        r"^urn:gl:factor:[a-z0-9][a-z0-9-]*"
        r"(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$"
    ),
    "factor_id_alias": r"^EF:[A-Za-z0-9_.:-]+$",
    "source_urn": r"^urn:gl:source:[a-z0-9][a-z0-9-]*$",
    "factor_pack_urn": (
        r"^urn:gl:pack:[a-z0-9][a-z0-9-]*:[a-z0-9][a-z0-9._-]*:v[1-9][0-9]*$"
    ),
    "unit_urn": r"^urn:gl:unit:[a-zA-Z0-9._/-]+$",
    "geography_urn": (
        r"^urn:gl:geo:(global|country|subregion|state_or_province|"
        r"grid_zone|bidding_zone|balancing_authority):[a-zA-Z0-9._-]+$"
    ),
    "methodology_urn": r"^urn:gl:methodology:[a-z0-9][a-z0-9-]*$",
    "supersedes_urn": (
        r"^urn:gl:factor:[a-z0-9][a-z0-9-]*"
        r"(:[A-Za-z0-9._-]+){2,4}:v[1-9][0-9]*$"
    ),
    # Phase 2 additive amendment (2026-04-27) — five OPTIONAL fields.
    # ``confidence``, ``created_at``, ``updated_at`` have no regex pattern
    # so they are NOT listed here; only the URN-pattern fields are.
    "superseded_by_urn": (
        r"^urn:gl:factor:[a-z0-9][a-z0-9-]*"
        r"(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$"
    ),
    "activity_taxonomy_urn": (
        r"^urn:gl:activity:[a-z0-9][a-z0-9-]*"
        r"(:[a-z0-9][a-z0-9._-]*)?$"
    ),
}


# ---------------------------------------------------------------------------
# Closed enums (mirrors the JSON Schema "enum" keywords)
# ---------------------------------------------------------------------------


class Category(str, Enum):
    """Narrowed alpha category enum (CTO Phase 2 §2.1)."""

    SCOPE1 = "scope1"
    SCOPE2_LOCATION_BASED = "scope2_location_based"
    SCOPE2_MARKET_BASED = "scope2_market_based"
    GRID_INTENSITY = "grid_intensity"
    FUEL = "fuel"
    REFRIGERANT = "refrigerant"
    FUGITIVE = "fugitive"
    PROCESS = "process"
    CBAM_DEFAULT = "cbam_default"


class GwpBasis(str, Enum):
    """Alpha is AR6-only."""

    AR6 = "ar6"


class Resolution(str, Enum):
    """Time resolution. Most alpha factors are annual."""

    ANNUAL = "annual"
    MONTHLY = "monthly"
    HOURLY = "hourly"
    POINT_IN_TIME = "point-in-time"


class ReviewStatus(str, Enum):
    """Lifecycle. Only ``approved`` is visible in production."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CitationType(str, Enum):
    """Citation discriminator."""

    DOI = "doi"
    URL = "url"
    PUBLICATION = "publication"
    SECTION = "section"
    TABLE = "table"


class UncertaintyDistribution(str, Enum):
    """Optional uncertainty distribution shape."""

    LOGNORMAL = "lognormal"
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"


# ---------------------------------------------------------------------------
# Common URN validator helper
# ---------------------------------------------------------------------------


def _check_urn_pattern_and_parse(
    value: str, pattern: re.Pattern, field_name: str
) -> str:
    """Validate ``value`` matches ``pattern`` AND parses cleanly via urn.parse.

    Raises:
        ValueError: if either check fails.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string, got {type(value).__name__}"
        )
    if not pattern.match(value):
        raise ValueError(
            f"{field_name}={value!r} does not match the v0.1 pattern"
        )
    try:
        parse_urn(value)
    except InvalidUrnError as exc:
        raise ValueError(
            f"{field_name}={value!r} failed canonical parse: {exc}"
        ) from exc
    return value


# ---------------------------------------------------------------------------
# Field-group submodels — one per CTO field group.
#
# These are documentation-grade typed mirrors. They are not composed into
# the top-level record via inheritance (multiple inheritance + extra=forbid
# + cross-field validators is brittle in Pydantic v2). They expose the
# same fields with the same validators so downstream callers can import
# and use them in isolation if they only need one group.
# ---------------------------------------------------------------------------


class IdentityFields(GreenLangBase):
    """Group 1 — identity (urn / aliases / pack / source).

    Mirrors the schema fields: ``urn``, ``factor_id_alias``,
    ``source_urn``, ``factor_pack_urn``.
    """

    urn: str = Field(
        ...,
        description=(
            "Canonical public id. Format: urn:gl:factor:<source>:<namespace>:"
            "<id>:v<version>. Globally unique. Never changes."
        ),
    )
    factor_id_alias: Optional[str] = Field(
        default=None,
        description=(
            "Optional non-canonical alias retained for backward compatibility "
            "with the v1 schema's EF: identifier. SDK and API responses MUST "
            "treat ``urn`` as primary."
        ),
    )
    source_urn: str = Field(
        ...,
        description=(
            "URN of the upstream source. MUST resolve to a registered entry "
            "in greenlang/factors/data/source_registry.yaml."
        ),
    )
    factor_pack_urn: str = Field(
        ...,
        description=(
            "URN of the owning Factor Pack. Format: urn:gl:pack:<source>:"
            "<pack-id>:v<version>."
        ),
    )

    @field_validator("urn")
    @classmethod
    def _validate_urn(cls, v: str) -> str:
        return _check_urn_pattern_and_parse(v, _URN_FACTOR_RE, "urn")

    @field_validator("factor_id_alias")
    @classmethod
    def _validate_factor_id_alias(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not _FACTOR_ID_ALIAS_RE.match(v):
            raise ValueError(
                f"factor_id_alias={v!r} must match '^EF:[A-Za-z0-9_.:-]+$'"
            )
        return v

    @field_validator("source_urn")
    @classmethod
    def _validate_source_urn(cls, v: str) -> str:
        return _check_urn_pattern_and_parse(v, _URN_SOURCE_RE, "source_urn")

    @field_validator("factor_pack_urn")
    @classmethod
    def _validate_factor_pack_urn(cls, v: str) -> str:
        return _check_urn_pattern_and_parse(
            v, _URN_PACK_RE, "factor_pack_urn"
        )


class ValueUnitFields(GreenLangBase):
    """Group 2 — numeric value plus unit URN."""

    value: float = Field(
        ...,
        description=(
            "The numeric emission factor value, expressed in the unit "
            "identified by unit_urn. MUST be > 0."
        ),
    )
    unit_urn: str = Field(
        ...,
        description=(
            "URN of the unit. Examples: urn:gl:unit:kgco2e/kwh, "
            "urn:gl:unit:kgco2e/kg."
        ),
    )

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: float) -> float:
        # Schema uses exclusiveMinimum: 0.
        if not isinstance(v, (int, float)):
            raise ValueError(f"value must be numeric, got {type(v).__name__}")
        if isinstance(v, bool):  # bool is an int subclass; reject
            raise ValueError("value must not be bool")
        if float(v) <= 0:
            raise ValueError(f"value={v!r} must be > 0 (exclusiveMinimum)")
        return float(v)

    @field_validator("unit_urn")
    @classmethod
    def _validate_unit_urn(cls, v: str) -> str:
        # Pattern-only check: the frozen schema's unit pattern allows
        # mixed case [a-zA-Z0-9._/-]+ for forward compatibility (e.g.,
        # 'kWh', 'tCO2e') while ``greenlang.factors.ontology.urn``'s
        # unit_re is lowercase-only. The schema is the source of truth
        # for v0.1 alpha; the URN parser will be aligned in Phase 2 §2.2.
        if not isinstance(v, str):
            raise ValueError(
                f"unit_urn must be a string, got {type(v).__name__}"
            )
        if not _URN_UNIT_RE.match(v):
            raise ValueError(
                f"unit_urn={v!r} does not match the v0.1 pattern"
            )
        return v


class ContextFields(GreenLangBase):
    """Group 3 — name / description / category / geography / methodology / boundary / resolution."""

    name: Annotated[str, StringConstraints(min_length=1, max_length=200)] = Field(
        ...,
        description=(
            "Human-readable display name (en-US). Example: 'Grid "
            "electricity, consumption mix, India'."
        ),
    )
    description: Annotated[str, StringConstraints(min_length=30, max_length=2000)] = Field(
        ...,
        description=(
            "2-3 sentence description that states the boundary and "
            "exclusions. Required for auditor readability."
        ),
    )
    category: Category = Field(
        ...,
        description=(
            "Narrowed alpha category enum. Scope 3, freight, agricultural, "
            "waste, finance proxies are deferred to v0.5+."
        ),
    )
    geography_urn: str = Field(
        ...,
        description="URN of the geography.",
    )
    methodology_urn: str = Field(
        ...,
        description="URN of the methodology.",
    )
    boundary: Annotated[str, StringConstraints(min_length=10, max_length=2000)] = Field(
        ...,
        description=(
            "Free-text statement of what is included and excluded. "
            "Auditor-facing."
        ),
    )
    resolution: Resolution = Field(
        ...,
        description=(
            "Time resolution. Most alpha factors are annual; hourly grid "
            "intensity is deferred to v2.5."
        ),
    )
    # Phase 2 additive amendment (2026-04-27).
    activity_taxonomy_urn: Optional[str] = Field(
        default=None,
        description=(
            "URN of the activity taxonomy entry. Optional in v0.1; "
            "required from v0.2. Resolves to factors_v0_1.activity.urn."
        ),
    )

    @field_validator("geography_urn")
    @classmethod
    def _validate_geography_urn(cls, v: str) -> str:
        # See top-level FactorRecordV0_1._validate_geography_urn note: we
        # skip urn.parse() because greenlang.factors.ontology.urn carries
        # a different ALLOWED_GEO_TYPES list than the frozen v0.1 schema.
        # The schema regex is the source of truth for alpha.
        if not isinstance(v, str):
            raise ValueError(
                f"geography_urn must be a string, got {type(v).__name__}"
            )
        if not _URN_GEO_RE.match(v):
            raise ValueError(
                f"geography_urn={v!r} does not match the v0.1 pattern"
            )
        return v

    @field_validator("methodology_urn")
    @classmethod
    def _validate_methodology_urn(cls, v: str) -> str:
        return _check_urn_pattern_and_parse(
            v, _URN_METHODOLOGY_RE, "methodology_urn"
        )

    @field_validator("activity_taxonomy_urn")
    @classmethod
    def _validate_activity_taxonomy_urn(
        cls, v: Optional[str]
    ) -> Optional[str]:
        # Optional in v0.1 — None is allowed. When supplied, must match the
        # canonical activity-URN pattern AND parse cleanly via urn.parse()
        # (rejects uppercase namespace segments per CTO doc §6.1.1).
        if v is None:
            return None
        if not _URN_ACTIVITY_RE.match(v):
            raise ValueError(
                f"activity_taxonomy_urn={v!r} does not match "
                "the v0.1 pattern"
            )
        try:
            parse_urn(v)
        except InvalidUrnError as exc:
            raise ValueError(
                f"activity_taxonomy_urn={v!r} failed canonical parse: "
                f"{exc}"
            ) from exc
        return v


class TimeFields(GreenLangBase):
    """Group 4 — vintage / publication / deprecation timestamps."""

    vintage_start: date = Field(
        ...,
        description="Earliest date covered by this factor (ISO 8601 YYYY-MM-DD).",
    )
    vintage_end: date = Field(
        ...,
        description=(
            "Latest date covered by this factor (ISO 8601 YYYY-MM-DD). "
            "MUST be >= vintage_start."
        ),
    )
    published_at: datetime = Field(
        ...,
        description=(
            "When this factor record was published to the production "
            "catalogue. Immutable after publish."
        ),
    )
    deprecated_at: Optional[datetime] = Field(
        default=None,
        description="If deprecated, when. Once set, never reset to null.",
    )

    @model_validator(mode="after")
    def _check_vintage_order(self) -> "TimeFields":
        if self.vintage_end < self.vintage_start:
            raise ValueError(
                f"vintage_end ({self.vintage_end}) must be >= "
                f"vintage_start ({self.vintage_start})"
            )
        return self


class ClimateBasisFields(GreenLangBase):
    """Group 5 — GWP basis + horizon."""

    gwp_basis: GwpBasis = Field(
        ...,
        description="Alpha is AR6-only.",
    )
    gwp_horizon: Literal[20, 100, 500] = Field(
        ...,
        description=(
            "GWP time horizon in years. 100 is the default for corporate "
            "inventories; 20 for short-lived climate forcers; 500 rare."
        ),
    )

    @field_validator("gwp_basis")
    @classmethod
    def _validate_gwp_basis(cls, v: Any) -> GwpBasis:
        if isinstance(v, GwpBasis):
            return v
        if v != "ar6":
            raise ValueError(f"gwp_basis must be 'ar6' in v0.1 alpha; got {v!r}")
        return GwpBasis.AR6


class _UncertaintyBlock(GreenLangBase):
    """Inner shape of the optional ``uncertainty`` field."""

    distribution: Optional[UncertaintyDistribution] = None
    mean: Optional[float] = None
    stddev: Optional[float] = Field(default=None, ge=0)
    p2_5: Optional[float] = None
    p97_5: Optional[float] = None
    pedigree: Optional[Dict[str, Any]] = None


class QualityFields(GreenLangBase):
    """Group 6 — optional uncertainty quantification.

    Optional in alpha (CTO doc §19.1); REQUIRED from v0.9+.
    """

    uncertainty: Optional[_UncertaintyBlock] = Field(
        default=None,
        description=(
            "Optional in alpha; required from v0.9+. When present: "
            "{distribution: 'lognormal'|'normal'|'uniform'|'triangular', ...}."
        ),
    )
    # Phase 2 additive amendment (2026-04-27). Distinct from
    # ``uncertainty`` (which describes the value distribution).
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional confidence score in [0,1]. Methodology lead's "
            "subjective confidence in the factor's accuracy. Distinct "
            "from uncertainty (which describes the value distribution)."
        ),
    )


class LicenceConstraints(GreenLangBase):
    """Optional licence-constraints sub-block."""

    redistribution: Optional[
        Literal["allowed", "attribution", "restricted", "forbidden"]
    ] = None
    attribution_required: Optional[bool] = None
    caching_seconds: Optional[int] = None
    notes: Optional[str] = None

    # Schema declares the constraints object permissively (no
    # additionalProperties). Allow extras here so we don't reject records
    # that carry forward-compatible licence keys. Caller must rely on the
    # JSON Schema layer for strict shape policing of this sub-object.
    model_config = ConfigDict(extra="allow")


class LicenceFields(GreenLangBase):
    """Group 7 — licence + optional constraints."""

    licence: Annotated[str, StringConstraints(min_length=1, max_length=128)] = Field(
        ...,
        description=(
            "SPDX identifier (e.g., 'CC-BY-4.0', 'OGL-UK-3.0', "
            "'public-domain-us-gov') or proprietary tag."
        ),
    )
    licence_constraints: Optional[LicenceConstraints] = Field(
        default=None,
        description=(
            "Optional. {redistribution: ..., attribution_required: bool, "
            "caching_seconds?: int, notes?: str}."
        ),
    )


class Citation(GreenLangBase):
    """A single bibliographic citation. ``additionalProperties: false``."""

    type: CitationType = Field(..., description="Citation discriminator.")
    value: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description="The DOI / URL / publication / section / table reference.",
    )
    title: Optional[str] = None
    page: Optional[Union[int, str]] = None


class ExtractionMetadata(GreenLangBase):
    """Inner ``extraction`` object — provenance of extraction.

    Every field is MANDATORY in alpha (CTO doc §19.1: "provenance fields
    complete for alpha sources").
    """

    source_url: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description="Canonical landing URL for the source.",
    )
    source_record_id: Annotated[str, StringConstraints(min_length=1, max_length=256)] = Field(
        ...,
        description=(
            "Stable identifier of the row/cell within the source artefact. "
            "Example: 'sheet=Stationary_Combustion;row=42'."
        ),
    )
    source_publication: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description="Title or DOI of the source publication.",
    )
    source_version: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description="Upstream version pin (e.g., '2025.1' for DEFRA).",
    )
    raw_artifact_uri: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description=(
            "S3-compatible URI to the immutable raw artefact "
            "(PDF/CSV/XLSX/JSON)."
        ),
    )
    raw_artifact_sha256: str = Field(
        ...,
        description="SHA-256 (lowercase hex) over the raw artefact bytes.",
    )
    parser_id: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description=(
            "Dotted module path of the parser (e.g., "
            "'greenlang.factors.ingestion.parsers.desnz_uk')."
        ),
    )
    parser_version: str = Field(
        ...,
        description="Semver of the parser at extraction time.",
    )
    parser_commit: str = Field(
        ...,
        description="Git commit SHA of the parser code (7-40 lowercase hex).",
    )
    row_ref: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description=(
            "Sheet/table/page/row reference within the raw artefact."
        ),
    )
    ingested_at: datetime = Field(
        ...,
        description="When the parser ingested this record.",
    )
    operator: str = Field(
        ...,
        description=(
            "Identity of who/what ran the extraction. 'bot:parser_<id>' for "
            "automated ingest; 'human:<email>' for manual hot-fix."
        ),
    )

    @field_validator("raw_artifact_sha256")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ValueError(
                f"raw_artifact_sha256={v!r} must be 64 lowercase hex chars"
            )
        return v

    @field_validator("parser_version")
    @classmethod
    def _validate_parser_version(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"parser_version={v!r} must be semver "
                "MAJOR.MINOR.PATCH(-prerelease)?"
            )
        return v

    @field_validator("parser_commit")
    @classmethod
    def _validate_parser_commit(cls, v: str) -> str:
        if not _PARSER_COMMIT_RE.match(v):
            raise ValueError(
                f"parser_commit={v!r} must be 7-40 lowercase hex chars"
            )
        return v

    @field_validator("operator")
    @classmethod
    def _validate_operator(cls, v: str) -> str:
        if not _OPERATOR_RE.match(v):
            raise ValueError(
                f"operator={v!r} must match 'bot:<id>' or 'human:<email>'"
            )
        return v


class ReviewMetadata(GreenLangBase):
    """Inner ``review`` object — methodology-lead approval state."""

    review_status: ReviewStatus = Field(
        ...,
        description=(
            "Lifecycle. Only 'approved' records are visible in production. "
            "'pending' lives in staging namespace; 'rejected' kept for audit."
        ),
    )
    reviewer: str = Field(
        ...,
        description="Email of the methodology lead or delegated reviewer.",
    )
    reviewed_at: datetime = Field(...)
    approved_by: Optional[str] = Field(
        default=None,
        description=(
            "Email of the approver. REQUIRED when review_status == "
            "'approved'."
        ),
    )
    approved_at: Optional[datetime] = Field(
        default=None,
        description=(
            "When approval was granted. REQUIRED when review_status == "
            "'approved'."
        ),
    )
    diff_from_source_uri: Optional[str] = Field(
        default=None,
        description=(
            "Optional URI to a diff report between the parsed record and "
            "the prior version."
        ),
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description=(
            "Required when review_status == 'rejected' (free text)."
        ),
    )

    @field_validator("reviewer")
    @classmethod
    def _validate_reviewer(cls, v: str) -> str:
        if not _HUMAN_EMAIL_RE.match(v):
            raise ValueError(
                f"reviewer={v!r} must match 'human:<email>'"
            )
        return v

    @field_validator("approved_by")
    @classmethod
    def _validate_approved_by(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not _HUMAN_EMAIL_RE.match(v):
            raise ValueError(
                f"approved_by={v!r} must match 'human:<email>'"
            )
        return v

    @model_validator(mode="after")
    def _check_status_invariants(self) -> "ReviewMetadata":
        if self.review_status == ReviewStatus.APPROVED:
            missing = []
            if not self.approved_by:
                missing.append("approved_by")
            if not self.approved_at:
                missing.append("approved_at")
            if missing:
                raise ValueError(
                    "review.review_status='approved' requires fields: "
                    f"{missing}"
                )
        elif self.review_status == ReviewStatus.REJECTED:
            if not self.rejection_reason:
                raise ValueError(
                    "review.review_status='rejected' requires "
                    "rejection_reason (non-empty string)"
                )
        return self


class LineageFields(GreenLangBase):
    """Group 8 — extraction provenance + citations + tags + supersedes."""

    extraction: ExtractionMetadata = Field(
        ...,
        description=(
            "Provenance of extraction. Every field MANDATORY in alpha "
            "(CTO doc §19.1)."
        ),
    )
    citations: Annotated[List[Citation], Field(min_length=1)] = Field(
        ...,
        description=(
            "DOIs, URLs, publication references, table/section refs. "
            "At least one citation required."
        ),
    )
    tags: Optional[List[Annotated[str, StringConstraints(min_length=1, max_length=64)]]] = Field(
        default=None,
        description=(
            "Optional indexed tags (e.g., ['egrid', 'us', '2024', "
            "'subregion'])."
        ),
    )
    supersedes_urn: Optional[str] = Field(
        default=None,
        description=(
            "URN of the prior factor this record replaces. Set on "
            "revisions."
        ),
    )

    @field_validator("supersedes_urn")
    @classmethod
    def _validate_supersedes_urn(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not _URN_FACTOR_SUPERSEDES_RE.match(v):
            raise ValueError(
                f"supersedes_urn={v!r} does not match v0.1 supersedes pattern"
            )
        try:
            parse_urn(v)
        except InvalidUrnError as exc:
            raise ValueError(
                f"supersedes_urn={v!r} failed canonical parse: {exc}"
            ) from exc
        return v


class LifecycleFields(GreenLangBase):
    """Group 9 — review/approval lifecycle."""

    review: ReviewMetadata = Field(
        ...,
        description=(
            "Review metadata. CTO doc §19.1: only methodology-lead-approved "
            "records flip to production."
        ),
    )
    # Phase 2 additive amendment (2026-04-27). Three OPTIONAL lifecycle
    # timestamps + reverse-pointer URN. None of them is required by the
    # frozen schema; absent values round-trip as ``null``.
    created_at: Optional[datetime] = Field(
        default=None,
        description=(
            "Wall-clock timestamp when the record was first staged for "
            "review. Distinct from published_at (when it became visible "
            "in production)."
        ),
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description=(
            "Wall-clock timestamp of the most recent metadata edit "
            "pre-publish. Immutable after published_at is set; tracking "
            "edits during the staging window only."
        ),
    )
    superseded_by_urn: Optional[str] = Field(
        default=None,
        description=(
            "Reverse pointer: when set, this factor has been superseded "
            "by the named URN. Inverse of supersedes_urn. Set on the "
            "prior factor when a correction is issued."
        ),
    )

    @field_validator("superseded_by_urn")
    @classmethod
    def _validate_superseded_by_urn(
        cls, v: Optional[str]
    ) -> Optional[str]:
        if v is None:
            return None
        if not _URN_FACTOR_SUPERSEDED_BY_RE.match(v):
            raise ValueError(
                f"superseded_by_urn={v!r} does not match v0.1 "
                "superseded_by pattern"
            )
        try:
            parse_urn(v)
        except InvalidUrnError as exc:
            raise ValueError(
                f"superseded_by_urn={v!r} failed canonical parse: {exc}"
            ) from exc
        return v


# ---------------------------------------------------------------------------
# Top-level record (FLAT — mirrors the JSON Schema's flat properties).
#
# Inherits from GreenLangRecord (per CTO Phase 2 §2.1 constraint). We
# strip the inherited timestamp/tenant/provenance fields out of the
# JSON-Schema-equivalent surface by overriding model_config + custom
# model_dump for callers who want a strict-v0.1 dict.
# ---------------------------------------------------------------------------


class FactorRecordV0_1(GreenLangRecord):
    """Pydantic v2 mirror of ``factor_record_v0_1.schema.json``.

    The JSON Schema is FROZEN; this Pydantic class adds typed access and
    cross-field validators. Use :func:`model_to_jsonschema_diff` to
    confirm the generated schema still aligns with the frozen file.
    """

    # GreenLangRecord brings in TimestampMixin (created_at, updated_at),
    # TenantMixin (tenant_id), ProvenanceMixin (provenance_hash). Those
    # are NOT in the v0.1 frozen schema; they are platform-side audit
    # fields. We keep them (per the Phase 2 §2.1 inheritance constraint)
    # but they are never serialised into the v0.1 wire shape — see the
    # ``to_v0_1_dict`` helper below.

    # GreenLangBase already sets ``extra="forbid"``. We restate it here
    # for clarity (matches the JSON Schema's ``additionalProperties:
    # false``).
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        from_attributes=True,
    )

    # ----- Identity (group 1) -------------------------------------------
    urn: str = Field(
        ...,
        description=(
            "Canonical public id. Format: urn:gl:factor:<source>:<namespace>:"
            "<id>:v<version>. Globally unique. Never changes."
        ),
    )
    factor_id_alias: Optional[str] = Field(
        default=None,
        description=(
            "Optional non-canonical alias retained for backward "
            "compatibility with the v1 schema's EF: identifier."
        ),
    )
    source_urn: str = Field(
        ...,
        description="URN of the upstream source.",
    )
    factor_pack_urn: str = Field(
        ...,
        description="URN of the owning Factor Pack.",
    )

    # ----- Context (group 3) - placed BEFORE value to mirror schema order
    name: Annotated[str, StringConstraints(min_length=1, max_length=200)] = Field(
        ...,
        description="Human-readable display name (en-US).",
    )
    description: Annotated[str, StringConstraints(min_length=30, max_length=2000)] = Field(
        ...,
        description=(
            "2-3 sentence description that states the boundary and "
            "exclusions. Required for auditor readability."
        ),
    )
    category: Category = Field(
        ...,
        description="Narrowed alpha category enum.",
    )

    # ----- Value+Unit (group 2) -----------------------------------------
    value: float = Field(
        ...,
        description=(
            "The numeric emission factor value, expressed in the unit "
            "identified by unit_urn. MUST be > 0."
        ),
    )
    unit_urn: str = Field(..., description="URN of the unit.")

    # ----- Climate basis (group 5) --------------------------------------
    gwp_basis: GwpBasis = Field(
        ...,
        description="Alpha is AR6-only.",
    )
    gwp_horizon: Literal[20, 100, 500] = Field(
        ...,
        description="GWP time horizon in years.",
    )

    # ----- Context cont'd (geography / vintage / methodology / boundary)
    geography_urn: str = Field(..., description="URN of the geography.")
    vintage_start: date = Field(
        ...,
        description="Earliest date covered by this factor (YYYY-MM-DD).",
    )
    vintage_end: date = Field(
        ...,
        description="Latest date covered (YYYY-MM-DD). MUST be >= vintage_start.",
    )
    resolution: Resolution = Field(
        ...,
        description="Time resolution.",
    )
    methodology_urn: str = Field(..., description="URN of the methodology.")
    boundary: Annotated[str, StringConstraints(min_length=10, max_length=2000)] = Field(
        ...,
        description="Free-text statement of what is included and excluded.",
    )
    # Phase 2 additive amendment (2026-04-27).
    activity_taxonomy_urn: Optional[str] = Field(
        default=None,
        description=(
            "URN of the activity taxonomy entry. Optional in v0.1; "
            "required from v0.2."
        ),
    )

    # ----- Quality (group 6) - OPTIONAL in alpha ------------------------
    uncertainty: Optional[_UncertaintyBlock] = Field(
        default=None,
        description="Optional in alpha; required from v0.9+.",
    )
    # Phase 2 additive amendment (2026-04-27).
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional confidence score in [0,1]. Distinct from "
            "uncertainty."
        ),
    )

    # ----- Licence (group 7) --------------------------------------------
    licence: Annotated[str, StringConstraints(min_length=1, max_length=128)] = Field(
        ...,
        description="SPDX identifier or proprietary tag.",
    )
    licence_constraints: Optional[LicenceConstraints] = Field(
        default=None,
        description="Optional licence-constraints sub-block.",
    )

    # ----- Lineage (group 8) --------------------------------------------
    citations: Annotated[List[Citation], Field(min_length=1)] = Field(
        ...,
        description=(
            "DOIs, URLs, publication references. At least one citation "
            "required."
        ),
    )
    tags: Optional[List[Annotated[str, StringConstraints(min_length=1, max_length=64)]]] = Field(
        default=None,
        description="Optional indexed tags.",
    )
    supersedes_urn: Optional[str] = Field(
        default=None,
        description="URN of the prior factor this record replaces.",
    )
    # Phase 2 additive amendment (2026-04-27) — reverse-pointer + staging
    # timestamps. All three are OPTIONAL.
    superseded_by_urn: Optional[str] = Field(
        default=None,
        description=(
            "Reverse pointer to the URN that supersedes this factor. "
            "Inverse of supersedes_urn."
        ),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description=(
            "Wall-clock timestamp when the record was first staged for "
            "review."
        ),
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description=(
            "Wall-clock timestamp of the most recent metadata edit "
            "pre-publish."
        ),
    )

    # ----- Time (group 4 cont'd) ----------------------------------------
    published_at: datetime = Field(
        ...,
        description="When this factor record was published.",
    )
    deprecated_at: Optional[datetime] = Field(
        default=None,
        description="If deprecated, when. Once set, never reset to null.",
    )

    # ----- Lineage extraction (group 8) ---------------------------------
    extraction: ExtractionMetadata = Field(
        ...,
        description="Provenance of extraction. All fields mandatory in alpha.",
    )

    # ----- Lifecycle (group 9) ------------------------------------------
    review: ReviewMetadata = Field(
        ...,
        description="Review metadata.",
    )

    # ---- Validators (re-applied at the top level) ----------------------

    @field_validator("urn")
    @classmethod
    def _validate_urn(cls, v: str) -> str:
        return _check_urn_pattern_and_parse(v, _URN_FACTOR_RE, "urn")

    @field_validator("factor_id_alias")
    @classmethod
    def _validate_factor_id_alias(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not _FACTOR_ID_ALIAS_RE.match(v):
            raise ValueError(
                f"factor_id_alias={v!r} must match '^EF:[A-Za-z0-9_.:-]+$'"
            )
        return v

    @field_validator("source_urn")
    @classmethod
    def _validate_source_urn(cls, v: str) -> str:
        return _check_urn_pattern_and_parse(v, _URN_SOURCE_RE, "source_urn")

    @field_validator("factor_pack_urn")
    @classmethod
    def _validate_factor_pack_urn(cls, v: str) -> str:
        return _check_urn_pattern_and_parse(v, _URN_PACK_RE, "factor_pack_urn")

    @field_validator("unit_urn")
    @classmethod
    def _validate_unit_urn(cls, v: str) -> str:
        # Pattern-only check: the frozen schema's unit pattern allows
        # mixed case [a-zA-Z0-9._/-]+ for forward compatibility (e.g.,
        # 'kWh', 'tCO2e') while ``greenlang.factors.ontology.urn``'s
        # unit_re is lowercase-only. The schema is the source of truth
        # for v0.1 alpha; the URN parser will be aligned in Phase 2 §2.2.
        if not isinstance(v, str):
            raise ValueError(
                f"unit_urn must be a string, got {type(v).__name__}"
            )
        if not _URN_UNIT_RE.match(v):
            raise ValueError(
                f"unit_urn={v!r} does not match the v0.1 pattern"
            )
        return v

    @field_validator("geography_urn")
    @classmethod
    def _validate_geography_urn(cls, v: str) -> str:
        # See ContextFields._validate_geography_urn for why this skips
        # urn.parse() and only pattern-checks.
        if not isinstance(v, str):
            raise ValueError(
                f"geography_urn must be a string, got {type(v).__name__}"
            )
        if not _URN_GEO_RE.match(v):
            raise ValueError(
                f"geography_urn={v!r} does not match the v0.1 pattern"
            )
        return v

    @field_validator("methodology_urn")
    @classmethod
    def _validate_methodology_urn(cls, v: str) -> str:
        return _check_urn_pattern_and_parse(
            v, _URN_METHODOLOGY_RE, "methodology_urn"
        )

    @field_validator("supersedes_urn")
    @classmethod
    def _validate_supersedes_urn(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not _URN_FACTOR_SUPERSEDES_RE.match(v):
            raise ValueError(
                f"supersedes_urn={v!r} does not match v0.1 supersedes pattern"
            )
        try:
            parse_urn(v)
        except InvalidUrnError as exc:
            raise ValueError(
                f"supersedes_urn={v!r} failed canonical parse: {exc}"
            ) from exc
        return v

    # Phase 2 additive amendment (2026-04-27) — validators for the five
    # OPTIONAL fields. Three of them are timestamps (no extra constraint
    # beyond Pydantic's datetime parsing) and one is numeric (handled by
    # Field(ge=0, le=1)). Only the two URN fields need pattern + canonical
    # parse checks.
    @field_validator("superseded_by_urn")
    @classmethod
    def _validate_superseded_by_urn(
        cls, v: Optional[str]
    ) -> Optional[str]:
        if v is None:
            return None
        if not _URN_FACTOR_SUPERSEDED_BY_RE.match(v):
            raise ValueError(
                f"superseded_by_urn={v!r} does not match v0.1 "
                "superseded_by pattern"
            )
        try:
            parse_urn(v)
        except InvalidUrnError as exc:
            raise ValueError(
                f"superseded_by_urn={v!r} failed canonical parse: {exc}"
            ) from exc
        return v

    @field_validator("activity_taxonomy_urn")
    @classmethod
    def _validate_activity_taxonomy_urn(
        cls, v: Optional[str]
    ) -> Optional[str]:
        if v is None:
            return None
        if not _URN_ACTIVITY_RE.match(v):
            raise ValueError(
                f"activity_taxonomy_urn={v!r} does not match the v0.1 pattern"
            )
        try:
            parse_urn(v)
        except InvalidUrnError as exc:
            raise ValueError(
                f"activity_taxonomy_urn={v!r} failed canonical parse: {exc}"
            ) from exc
        return v

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: float) -> float:
        if isinstance(v, bool):
            raise ValueError("value must not be bool")
        if not isinstance(v, (int, float)):
            raise ValueError(f"value must be numeric, got {type(v).__name__}")
        if float(v) <= 0:
            raise ValueError(f"value={v!r} must be > 0 (exclusiveMinimum)")
        return float(v)

    @field_validator("gwp_basis")
    @classmethod
    def _validate_gwp_basis(cls, v: Any) -> GwpBasis:
        if isinstance(v, GwpBasis):
            return v
        if v != "ar6":
            raise ValueError(
                f"gwp_basis must be 'ar6' in v0.1 alpha; got {v!r}"
            )
        return GwpBasis.AR6

    @model_validator(mode="after")
    def _check_vintage_order(self) -> "FactorRecordV0_1":
        if self.vintage_end < self.vintage_start:
            raise ValueError(
                f"vintage_end ({self.vintage_end}) must be >= "
                f"vintage_start ({self.vintage_start})"
            )
        return self

    # ---- Helpers --------------------------------------------------------

    #: Field names that belong to the v0.1 frozen-schema surface (i.e.
    #: every property declared in ``factor_record_v0_1.schema.json``). The
    #: helper :meth:`to_v0_1_dict` uses this to strip platform-side
    #: GreenLangRecord fields (created_at, updated_at, tenant_id,
    #: provenance_hash) so the result round-trips through the JSON
    #: Schema validator without ``additionalProperties`` failures.
    V0_1_SCHEMA_FIELDS: Tuple[str, ...] = (
        "urn",
        "factor_id_alias",
        "source_urn",
        "factor_pack_urn",
        "name",
        "description",
        "category",
        "value",
        "unit_urn",
        "gwp_basis",
        "gwp_horizon",
        "geography_urn",
        "vintage_start",
        "vintage_end",
        "resolution",
        "methodology_urn",
        "boundary",
        "uncertainty",
        "licence",
        "licence_constraints",
        "citations",
        "tags",
        "supersedes_urn",
        "published_at",
        "deprecated_at",
        "extraction",
        "review",
        # Phase 2 additive amendment (2026-04-27) — five OPTIONAL fields.
        "activity_taxonomy_urn",
        "confidence",
        "created_at",
        "updated_at",
        "superseded_by_urn",
    )

    def to_v0_1_dict(self) -> Dict[str, Any]:
        """Return a JSON-Schema-aligned dict (drops platform audit fields).

        The platform inherits ``created_at`` / ``updated_at`` /
        ``tenant_id`` / ``provenance_hash`` from :class:`GreenLangRecord`
        for cross-cutting audit purposes; those keys are NOT part of the
        v0.1 frozen wire contract. This helper drops them so the dict
        round-trips through the JSON-Schema validator without
        ``additionalProperties`` failures.
        """
        data = self.model_dump(mode="json")
        return {k: v for k, v in data.items() if k in self.V0_1_SCHEMA_FIELDS}


# ---------------------------------------------------------------------------
# JSON-Schema diff helper (used by the CI gate)
# ---------------------------------------------------------------------------


def _normalise_schema_for_diff(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalised view of a JSON Schema for set-based diffing.

    Pydantic's ``model_json_schema`` and the hand-authored frozen schema
    differ cosmetically (ordering, $defs presence, default values, OpenAPI
    metadata). The CI gate compares the SEMANTIC surface — required
    fields, declared properties, enum values, regex patterns — not the
    raw object shape. This helper extracts that semantic surface.
    """
    out: Dict[str, Any] = {
        "required": sorted(schema.get("required") or []),
        "properties": {},
        "extra": {
            "additionalProperties": schema.get("additionalProperties"),
            "type": schema.get("type"),
        },
    }
    for prop_name, prop_schema in (schema.get("properties") or {}).items():
        if not isinstance(prop_schema, dict):
            continue
        entry: Dict[str, Any] = {}
        if "enum" in prop_schema:
            entry["enum"] = sorted(prop_schema["enum"])
        if "pattern" in prop_schema:
            entry["pattern"] = prop_schema["pattern"]
        if "format" in prop_schema:
            entry["format"] = prop_schema["format"]
        # Capture nested object's required + properties at one extra level.
        if isinstance(prop_schema.get("properties"), dict):
            entry["nested_required"] = sorted(prop_schema.get("required") or [])
            entry["nested_property_names"] = sorted(
                prop_schema["properties"].keys()
            )
        out["properties"][prop_name] = entry
    return out


def _resolve_pydantic_property(
    prop_name: str,
    schema: Dict[str, Any],
    defs: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve a single Pydantic-generated property to comparable form.

    Pydantic emits ``$ref`` / ``anyOf`` for enums and nested models.
    We chase one level of ``$ref`` into ``$defs`` so the enum / pattern
    / nested-required surface lines up with the frozen schema's inline
    declarations.
    """
    prop_schema = schema.get("properties", {}).get(prop_name)
    if not isinstance(prop_schema, dict):
        return {}

    # Chase a single $ref into $defs.
    target = prop_schema
    if "$ref" in target:
        ref = target["$ref"].split("/")[-1]
        target = defs.get(ref, {}) or {}
    elif "anyOf" in target:
        # Optional fields appear as anyOf: [..., {"type": "null"}].
        for cand in target["anyOf"]:
            if isinstance(cand, dict) and "$ref" in cand:
                ref = cand["$ref"].split("/")[-1]
                target = defs.get(ref, {}) or {}
                break
            if isinstance(cand, dict) and "type" in cand and cand["type"] != "null":
                target = cand
                break

    entry: Dict[str, Any] = {}
    if "enum" in target:
        entry["enum"] = sorted(target["enum"])
    if "pattern" in target:
        entry["pattern"] = target["pattern"]
    if "format" in target:
        entry["format"] = target["format"]
    if isinstance(target.get("properties"), dict):
        entry["nested_required"] = sorted(target.get("required") or [])
        entry["nested_property_names"] = sorted(target["properties"].keys())
    return entry


def _normalise_pydantic_schema_for_diff(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Same as :func:`_normalise_schema_for_diff` but resolves $defs/$refs.

    Used on the Pydantic-generated schema; the frozen JSON Schema is
    already inline. Patterns enforced inside ``field_validator`` are
    looked up from :data:`_PYDANTIC_FIELD_PATTERNS` because Pydantic v2
    does not expose validator-internal regexes through
    ``model_json_schema()``.
    """
    defs = schema.get("$defs", {}) or {}
    out: Dict[str, Any] = {
        "required": sorted(schema.get("required") or []),
        "properties": {},
        "extra": {
            "additionalProperties": schema.get("additionalProperties"),
            "type": schema.get("type"),
        },
    }
    for prop_name in (schema.get("properties") or {}):
        entry = _resolve_pydantic_property(prop_name, schema, defs)
        # Patches: enrich with patterns enforced by field_validators that
        # Pydantic's auto-generated schema does NOT surface.
        if (
            "pattern" not in entry
            and prop_name in _PYDANTIC_FIELD_PATTERNS
        ):
            entry["pattern"] = _PYDANTIC_FIELD_PATTERNS[prop_name]
        out["properties"][prop_name] = entry
    return out


def model_to_jsonschema_diff(
    *,
    model: Optional[type] = None,
    frozen_schema_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Diff the Pydantic mirror's generated schema against the frozen file.

    Returns a dict with these keys::

        {
            "missing_required":   [...],   # required in frozen, not in pydantic
            "extra_required":     [...],   # required in pydantic, not in frozen
            "missing_properties": [...],   # property in frozen, not in pydantic
            "extra_properties":   [...],   # property in pydantic, not in frozen
            "enum_mismatches":    [{...}], # property -> {frozen, pydantic}
            "pattern_mismatches": [{...}], # property -> {frozen, pydantic}
            "format_mismatches":  [{...}], # property -> {frozen, pydantic}
            "nested_required_mismatches": [{...}],
            "nested_property_name_mismatches": [{...}],
            "additional_properties_match": bool,
            "type_match": bool,
        }

    Empty lists everywhere + both bools True == perfect alignment. The
    caller (CI gate) decides whether any mismatch is acceptable.
    """
    target_model = model or FactorRecordV0_1
    schema_path = frozen_schema_path or FrozenSchemaPath
    frozen = json.loads(schema_path.read_text(encoding="utf-8"))

    # ``GreenLangRecord`` injects platform fields. Remove them from the
    # generated schema before diffing — they are deliberately NOT part of
    # the v0.1 wire shape.
    #
    # NOTE (2026-04-27 amendment): ``created_at`` and ``updated_at`` were
    # originally stripped here because the canonical v0.1 record did not
    # carry them; ``GreenLangRecord``'s mixin shadowed those names with
    # platform-side audit fields. The Phase 2 additive amendment promoted
    # both timestamps into the public contract, so the Pydantic mirror's
    # top-level :class:`FactorRecordV0_1` REDECLARES both as
    # ``Optional[datetime] = None`` (which overrides the mixin's
    # ``default_factory=utcnow`` non-Optional fields). They are therefore
    # NO LONGER stripped here — they must align with the frozen schema's
    # newly added optional declarations.
    pyd_raw = target_model.model_json_schema()
    platform_fields = {
        "tenant_id",
        "provenance_hash",
        "V0_1_SCHEMA_FIELDS",
    }
    pyd_props = pyd_raw.get("properties") or {}
    pyd_required = pyd_raw.get("required") or []
    pyd_raw["properties"] = {
        k: v for k, v in pyd_props.items() if k not in platform_fields
    }
    pyd_raw["required"] = [r for r in pyd_required if r not in platform_fields]

    f_norm = _normalise_schema_for_diff(frozen)
    p_norm = _normalise_pydantic_schema_for_diff(pyd_raw)

    f_required = set(f_norm["required"])
    p_required = set(p_norm["required"])
    f_props = set(f_norm["properties"].keys())
    p_props = set(p_norm["properties"].keys())

    enum_mismatches: List[Dict[str, Any]] = []
    pattern_mismatches: List[Dict[str, Any]] = []
    format_mismatches: List[Dict[str, Any]] = []
    nested_req_mismatches: List[Dict[str, Any]] = []
    nested_prop_mismatches: List[Dict[str, Any]] = []

    for prop in sorted(f_props & p_props):
        f_entry = f_norm["properties"][prop]
        p_entry = p_norm["properties"][prop]
        if f_entry.get("enum") != p_entry.get("enum") and (
            f_entry.get("enum") or p_entry.get("enum")
        ):
            enum_mismatches.append(
                {
                    "property": prop,
                    "frozen": f_entry.get("enum"),
                    "pydantic": p_entry.get("enum"),
                }
            )
        if f_entry.get("pattern") != p_entry.get("pattern") and (
            f_entry.get("pattern") or p_entry.get("pattern")
        ):
            pattern_mismatches.append(
                {
                    "property": prop,
                    "frozen": f_entry.get("pattern"),
                    "pydantic": p_entry.get("pattern"),
                }
            )
        # Format: Pydantic may emit format=date or date-time even when the
        # frozen schema declares it; only count as mismatch when the
        # frozen file declares a format and the Pydantic schema either
        # omits it or declares a DIFFERENT one.
        f_fmt = f_entry.get("format")
        p_fmt = p_entry.get("format")
        if f_fmt and p_fmt and f_fmt != p_fmt:
            format_mismatches.append(
                {
                    "property": prop,
                    "frozen": f_fmt,
                    "pydantic": p_fmt,
                }
            )
        # Nested-object diffing: only flag when the FROZEN schema
        # declares inline nested info. If the frozen schema is silent
        # (e.g. ``licence_constraints`` is declared as type:object with
        # no ``properties``), the Pydantic mirror is FREE to provide
        # tighter typing without that being a divergence — the JSON
        # Schema layer remains the authoritative looser shape. We only
        # report a mismatch when the frozen side has declared something
        # AND the Pydantic side either disagrees or fails to declare.
        if "nested_required" in f_entry and f_entry.get(
            "nested_required"
        ) != p_entry.get("nested_required"):
            nested_req_mismatches.append(
                {
                    "property": prop,
                    "frozen": f_entry.get("nested_required"),
                    "pydantic": p_entry.get("nested_required"),
                }
            )
        if "nested_property_names" in f_entry and f_entry.get(
            "nested_property_names"
        ) != p_entry.get("nested_property_names"):
            nested_prop_mismatches.append(
                {
                    "property": prop,
                    "frozen": f_entry.get("nested_property_names"),
                    "pydantic": p_entry.get("nested_property_names"),
                }
            )

    return {
        "missing_required": sorted(f_required - p_required),
        "extra_required": sorted(p_required - f_required),
        "missing_properties": sorted(f_props - p_props),
        "extra_properties": sorted(p_props - f_props),
        "enum_mismatches": enum_mismatches,
        "pattern_mismatches": pattern_mismatches,
        "format_mismatches": format_mismatches,
        "nested_required_mismatches": nested_req_mismatches,
        "nested_property_name_mismatches": nested_prop_mismatches,
        "additional_properties_match": (
            f_norm["extra"]["additionalProperties"]
            == p_norm["extra"]["additionalProperties"]
        ),
        "type_match": f_norm["extra"]["type"] == p_norm["extra"]["type"],
    }
