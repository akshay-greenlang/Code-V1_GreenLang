# -*- coding: utf-8 -*-
"""Typed Pydantic v2 models for the Factors SDK.

Every model here inherits from :class:`greenlang.schemas.base.GreenLangBase`
per the house schema-base migration (see MEMORY.md, 2026-03-30).

The models are **permissive** on the inbound side — the live API evolves,
and pinning clients to exact server shapes would make them brittle.  We
relax ``extra="forbid"`` on response-shaped models and keep validation
strict only on request models where typos are bugs worth catching early.
"""
from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, model_validator

from greenlang.schemas.base import GreenLangBase


# ---------------------------------------------------------------------------
# Response-shaped base: permissive (forward compatible with server drift).
# ---------------------------------------------------------------------------


class _SDKResponseModel(GreenLangBase):
    """Base for SDK response models.

    Overrides the strict ``extra="forbid"`` policy from GreenLangBase so
    a client pinned to SDK v1.0 keeps working when the server starts
    returning new optional fields.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_default=True,
        from_attributes=True,
    )


# ---------------------------------------------------------------------------
# Primitive / shared shapes
# ---------------------------------------------------------------------------


class Jurisdiction(_SDKResponseModel):
    """Geographic jurisdiction tag (country/subnational)."""

    code: str = Field(..., description="ISO country or subnational code (e.g. 'US', 'US-CA')")
    name: Optional[str] = Field(None, description="Human-readable jurisdiction name")
    level: Optional[str] = Field(
        None,
        description="Geographic level: country | subnational | region | global",
    )


class ActivitySchema(_SDKResponseModel):
    """Schema describing an activity type (fuel, electricity, material …)."""

    activity_id: str = Field(..., description="Stable activity identifier")
    label: Optional[str] = Field(None, description="Display label")
    unit: Optional[str] = Field(None, description="Canonical activity unit")
    category: Optional[str] = Field(None, description="High-level category")
    description: Optional[str] = Field(None)


class QualityScore(_SDKResponseModel):
    """Pedigree / data-quality score breakdown."""

    overall_score: float = Field(..., description="Composite DQS score 0-100")
    rating: Optional[str] = Field(None, description="Letter grade (A-E) or tier")
    temporal: Optional[float] = Field(None)
    geographical: Optional[float] = Field(None)
    technological: Optional[float] = Field(None)
    representativeness: Optional[float] = Field(None)
    methodological: Optional[float] = Field(None)


class Uncertainty(_SDKResponseModel):
    """Uncertainty envelope for a factor or resolved result."""

    ci_95: Optional[float] = Field(
        None, description="95% confidence interval half-width (fraction)"
    )
    distribution: Optional[str] = Field(
        None, description="Distribution shape (normal, lognormal, triangular, ...)"
    )
    std_dev: Optional[float] = Field(None)
    sample_size: Optional[int] = Field(None)


class GasBreakdown(_SDKResponseModel):
    """Per-gas breakdown of CO2-equivalent emissions.

    The server keeps HFCs, PFCs, SF6, NF3, and biogenic CO2 as
    *separate* components — CTO non-negotiable, never rolled up.
    """

    CO2: Optional[float] = Field(None, description="CO2 (kg/unit)")
    CH4: Optional[float] = Field(None, description="CH4 (kg/unit)")
    N2O: Optional[float] = Field(None, description="N2O (kg/unit)")
    HFCs: Optional[float] = Field(None, description="HFC mix (kg/unit)")
    PFCs: Optional[float] = Field(None, description="PFC mix (kg/unit)")
    SF6: Optional[float] = Field(None, description="SF6 (kg/unit)")
    NF3: Optional[float] = Field(None, description="NF3 (kg/unit)")
    biogenic_CO2: Optional[float] = Field(
        None, description="Biogenic CO2 (kept separate — never rolled into CO2e)"
    )
    ch4_gwp: Optional[float] = Field(None, description="GWP multiplier applied to CH4")
    n2o_gwp: Optional[float] = Field(None, description="GWP multiplier applied to N2O")


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------


class Source(_SDKResponseModel):
    """Upstream publisher / source (EPA, DEFRA, IEA, …)."""

    source_id: str = Field(..., description="Stable source identifier")
    organization: Optional[str] = Field(None)
    publication: Optional[str] = Field(None)
    year: Optional[int] = Field(None)
    url: Optional[str] = Field(None)
    methodology: Optional[str] = Field(None)
    license: Optional[str] = Field(None)
    version: Optional[str] = Field(None)


class MethodPack(_SDKResponseModel):
    """Method pack descriptor (corporate_scope1, product_lca, …)."""

    method_pack_id: str = Field(..., description="Method pack identifier")
    name: Optional[str] = Field(None)
    version: Optional[str] = Field(None)
    scope: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    jurisdictions: List[str] = Field(default_factory=list)


class Edition(_SDKResponseModel):
    """Catalog edition descriptor."""

    edition_id: str = Field(..., description="Edition id (e.g. 'ef_2026_q1')")
    status: Optional[str] = Field(None, description="published | pending | deprecated")
    label: Optional[str] = Field(None)
    manifest_hash: Optional[str] = Field(None, description="SHA-256 of edition manifest")
    published_at: Optional[datetime] = Field(None)


class Factor(_SDKResponseModel):
    """Full emission factor record (response shape)."""

    factor_id: str = Field(..., description="Stable factor identifier")
    fuel_type: Optional[str] = Field(None)
    unit: Optional[str] = Field(None)
    geography: Optional[str] = Field(None)
    geography_level: Optional[str] = Field(None)
    scope: Optional[str] = Field(None)
    boundary: Optional[str] = Field(None)

    # Emission values
    co2_per_unit: Optional[float] = Field(None, description="CO2 kg/unit")
    ch4_per_unit: Optional[float] = Field(None, description="CH4 kg/unit")
    n2o_per_unit: Optional[float] = Field(None, description="N2O kg/unit")
    co2e_per_unit: Optional[float] = Field(None, description="CO2e kg/unit")

    # Quality / provenance
    data_quality: Optional[QualityScore] = Field(None)
    source: Optional[Source] = Field(None)
    uncertainty_95ci: Optional[float] = Field(None)

    # Validity
    valid_from: Optional[str] = Field(None)
    valid_to: Optional[str] = Field(None)

    # Status
    factor_status: Optional[str] = Field(
        None, description="certified | preview | connector_only | deprecated"
    )
    license: Optional[str] = Field(None)
    license_class: Optional[str] = Field(None)
    compliance_frameworks: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    activity_tags: List[str] = Field(default_factory=list)
    sector_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(None)

    # Versioning
    edition_id: Optional[str] = Field(None)
    source_id: Optional[str] = Field(None)
    source_release: Optional[str] = Field(None)
    release_version: Optional[str] = Field(None)
    replacement_factor_id: Optional[str] = Field(None)
    content_hash: Optional[str] = Field(None)


class FactorMatch(_SDKResponseModel):
    """Candidate returned by POST /factors/match."""

    factor_id: str = Field(...)
    score: float = Field(..., description="Match confidence 0-1")
    explanation: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search / list responses
# ---------------------------------------------------------------------------


class SearchResponse(_SDKResponseModel):
    """Unified search response (covers /search, /search/v2, /list)."""

    factors: List[Factor] = Field(default_factory=list)
    count: Optional[int] = Field(None, description="Results in this page")
    total_count: Optional[int] = Field(None, description="Total across all pages")
    page: Optional[int] = Field(None)
    page_size: Optional[int] = Field(None)
    total_pages: Optional[int] = Field(None)
    offset: Optional[int] = Field(None)
    limit: Optional[int] = Field(None)
    query: Optional[str] = Field(None)
    edition_id: Optional[str] = Field(None)
    search_time_ms: Optional[float] = Field(None)
    sort_by: Optional[str] = Field(None)
    sort_order: Optional[str] = Field(None)
    next_cursor: Optional[str] = Field(None, description="Cursor for next page, if any")


class CoverageReport(_SDKResponseModel):
    """Coverage statistics response."""

    total_factors: Optional[int] = Field(None)
    by_geography: Dict[str, int] = Field(default_factory=dict)
    by_scope: Dict[str, int] = Field(default_factory=dict)
    by_fuel_type: Dict[str, int] = Field(default_factory=dict)
    by_source: Dict[str, int] = Field(default_factory=dict)
    edition_id: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Resolution (7-step cascade)
# ---------------------------------------------------------------------------


class ResolutionRequest(GreenLangBase):
    """Input payload for POST /factors/resolve-explain.

    Strict (``extra="forbid"``) so typos in request fields fail loudly
    before the RPC round-trip.
    """

    activity: str = Field(..., description="Activity description or canonical id")
    method_profile: str = Field(
        ...,
        description="Method profile (e.g. 'corporate_scope1', 'corporate_scope2_location_based')",
    )
    jurisdiction: Optional[str] = Field(
        None, description="ISO country/subnational code (e.g. 'US', 'US-CA')"
    )
    reporting_date: Optional[str] = Field(
        None, description="ISO-8601 date the emission is reported against"
    )
    supplier_id: Optional[str] = Field(None)
    facility_id: Optional[str] = Field(None)
    utility_or_grid_region: Optional[str] = Field(None)
    preferred_sources: List[str] = Field(default_factory=list)
    extras: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-pack specific extras (free-form dict)",
    )


# ---------------------------------------------------------------------------
# Wave 2 envelope models — new typed surfaces the resolver now emits.
# ---------------------------------------------------------------------------


class ChosenFactor(_SDKResponseModel):
    """Wave 2 ``chosen_factor`` envelope.

    Describes the factor the resolver selected plus the pack-level
    metadata needed to reproduce the decision.
    """

    factor_id: str = Field(..., description="Stable factor identifier")
    factor_version: Optional[str] = Field(None, description="Semver of the factor record itself")
    release_version: Optional[str] = Field(
        None,
        description=(
            "Release version of the method pack that produced the factor — "
            "distinct from ``factor_version`` (individual record rev)."
        ),
    )
    method_profile: Optional[str] = Field(None)
    method_pack_id: Optional[str] = Field(None)
    method_pack_version: Optional[str] = Field(None)
    pack_id: Optional[str] = Field(None, description="Alias of method_pack_id used by some Wave 2 endpoints")
    co2e_per_unit: Optional[float] = Field(None)
    unit: Optional[str] = Field(None)
    geography: Optional[str] = Field(None)
    scope: Optional[str] = Field(None)


class SourceDescriptor(_SDKResponseModel):
    """Nested ``source`` block on the Wave 2 envelope."""

    source_id: str = Field(..., description="Stable source id (e.g. 'epa_ghg_2026')")
    organization: Optional[str] = Field(None)
    publication: Optional[str] = Field(None)
    year: Optional[int] = Field(None)
    url: Optional[str] = Field(None)
    methodology: Optional[str] = Field(None)
    license: Optional[str] = Field(None)
    license_class: Optional[str] = Field(None)
    version: Optional[str] = Field(None)
    release_version: Optional[str] = Field(None)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class QualityEnvelope(_SDKResponseModel):
    """Composite quality envelope — Wave 2 surfaces the 0-100 FQS scalar."""

    composite_fqs_0_100: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Composite Factor Quality Score (0 worst, 100 best)",
    )
    overall_score: Optional[float] = Field(None)
    rating: Optional[str] = Field(None)
    temporal: Optional[float] = Field(None)
    geographical: Optional[float] = Field(None)
    technological: Optional[float] = Field(None)
    representativeness: Optional[float] = Field(None)
    methodological: Optional[float] = Field(None)


class UncertaintyEnvelope(_SDKResponseModel):
    """Richer Wave 2 uncertainty envelope (superset of :class:`Uncertainty`)."""

    ci_95: Optional[float] = Field(None, description="95% CI half-width as a fraction")
    ci_lower: Optional[float] = Field(None)
    ci_upper: Optional[float] = Field(None)
    distribution: Optional[str] = Field(None)
    std_dev: Optional[float] = Field(None)
    sample_size: Optional[int] = Field(None)
    pedigree_matrix: Dict[str, Any] = Field(default_factory=dict)


class LicensingEnvelope(_SDKResponseModel):
    """Licensing envelope — Wave 2 surfaces the full upstream chain."""

    license: Optional[str] = Field(None, description="Human-readable license name")
    license_class: Optional[str] = Field(
        None, description="certified | preview | connector_only | redistributable"
    )
    redistribution_class: Optional[str] = Field(None)
    upstream_licenses: List[str] = Field(default_factory=list)
    attribution: Optional[str] = Field(None)
    restrictions: List[str] = Field(default_factory=list)


class DeprecationStatus(_SDKResponseModel):
    """Structured Wave 2 deprecation status (was a bare string pre-Wave 2)."""

    status: Optional[str] = Field(
        None, description="current | scheduled | deprecated | retired"
    )
    effective_from: Optional[str] = Field(None)
    effective_to: Optional[str] = Field(None)
    replacement_factor_id: Optional[str] = Field(None)
    reason: Optional[str] = Field(None)
    notice_url: Optional[str] = Field(None)


class SignedReceipt(_SDKResponseModel):
    """Wave 2a canonical signed receipt.

    Fields match the server emission in
    :mod:`greenlang.factors.middleware.signed_receipts`. Backward-compatible
    parsing of the pre-Wave-2a shape is handled via
    :meth:`from_response_dict`.

    New (canonical) fields:
        * ``receipt_id`` — UUIDv4 minted per response.
        * ``signature`` — base64 signature bytes.
        * ``verification_key_hint`` — 16-hex-char fingerprint for key lookup.
        * ``alg`` — ``"sha256-hmac"`` or ``"ed25519"`` (was ``algorithm``).
        * ``payload_hash`` — SHA-256 hex of the canonical signed payload
          (was ``signed_over`` pre-Wave-2a).
    """

    receipt_id: Optional[str] = Field(None)
    signature: str = Field(...)
    verification_key_hint: Optional[str] = Field(None)
    alg: str = Field(..., description="sha256-hmac | ed25519")
    payload_hash: Optional[str] = Field(None)
    signed_at: Optional[str] = Field(None)
    key_id: Optional[str] = Field(None, description="Retained for key rotation")
    edition_id: Optional[str] = Field(None)

    # --- Compatibility layer --------------------------------------------

    @classmethod
    def from_response_dict(
        cls, raw: Dict[str, Any]
    ) -> "SignedReceipt":
        """Build a :class:`SignedReceipt` from any supported wire shape.

        Wave 2a renamed several keys; we read the new names first and fall
        back to the deprecated aliases. A :class:`DeprecationWarning` is
        emitted exactly once per alias encountered so callers can migrate
        before v2.0.0 removes the fallbacks.
        """
        if not isinstance(raw, dict):
            raise TypeError(
                f"SignedReceipt.from_response_dict expected dict, got {type(raw).__name__}"
            )

        data: Dict[str, Any] = dict(raw)

        # alg <- algorithm (deprecated)
        if "alg" not in data and "algorithm" in data:
            warnings.warn(
                "Signed receipt key 'algorithm' is deprecated; server should emit 'alg'. "
                "SDK fallback will be removed in v2.0.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            data["alg"] = data.get("algorithm")

        # payload_hash <- signed_over.body_hash / signed_over (deprecated)
        if "payload_hash" not in data and "signed_over" in data:
            warnings.warn(
                "Signed receipt key 'signed_over' is deprecated; use 'payload_hash'. "
                "SDK fallback will be removed in v2.0.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            so = data.get("signed_over")
            if isinstance(so, dict):
                # pre-Wave-2a emitted a nested envelope; pull body_hash
                data["payload_hash"] = so.get("body_hash") or so.get("payload_hash")
            elif isinstance(so, str):
                data["payload_hash"] = so

        return cls.model_validate(data)


class ResolvedFactor(_SDKResponseModel):
    """Output payload from /explain and /resolve-explain.

    The server returns a rich structure; we surface the most useful
    fields as typed attributes and keep the raw payload under ``.raw``
    for forward compatibility.

    Wave 2/2.5 additions (all optional for forward compatibility):
        * ``chosen_factor`` — typed :class:`ChosenFactor` envelope.
        * ``source`` — nested :class:`SourceDescriptor`.
        * ``quality`` — :class:`QualityEnvelope` with composite FQS 0-100.
        * ``licensing`` — :class:`LicensingEnvelope`.
        * ``release_version`` — method-pack release (distinct from
          ``factor_version`` which tracks a single record rev).
        * ``audit_text`` / ``audit_text_draft`` — Wave 2.5 narrative.
        * ``signed_receipt`` — Wave 2a canonical receipt.
    """

    chosen_factor_id: Optional[str] = Field(None)
    chosen_factor: Optional[ChosenFactor] = Field(None)
    factor_id: Optional[str] = Field(None)
    factor_version: Optional[str] = Field(None)
    release_version: Optional[str] = Field(
        None,
        description="Method-pack release version (distinct from factor_version).",
    )
    method_profile: Optional[str] = Field(None)
    method_pack_version: Optional[str] = Field(None)

    fallback_rank: Optional[int] = Field(
        None, description="Which of the 7 cascade steps produced the winner"
    )
    step_label: Optional[str] = Field(None)
    why_chosen: Optional[str] = Field(None)

    source: Optional[SourceDescriptor] = Field(None)
    quality_score: Optional[QualityScore] = Field(None)
    quality: Optional[QualityEnvelope] = Field(None)
    uncertainty: Optional[UncertaintyEnvelope] = Field(None)
    gas_breakdown: Optional[GasBreakdown] = Field(None)
    co2e_basis: Optional[str] = Field(None, description="GWP set applied")

    assumptions: List[str] = Field(default_factory=list)
    alternates: List[Dict[str, Any]] = Field(default_factory=list)

    licensing: Optional[LicensingEnvelope] = Field(None)
    deprecation_status: Optional[Any] = Field(
        None,
        description=(
            "Deprecation status — may be a plain string (pre-Wave-2) or a "
            "structured DeprecationStatus object (Wave 2+)."
        ),
    )
    deprecation_replacement: Optional[str] = Field(None)

    # Wave 2.5: narrative audit text. ``audit_text_draft=True`` indicates
    # the template has not yet been reviewed/approved by a human auditor.
    audit_text: Optional[str] = Field(
        None,
        description="Human-readable narrative explaining the resolution decision.",
    )
    audit_text_draft: Optional[bool] = Field(
        None,
        description="True when ``audit_text`` is a draft from an unapproved template.",
    )

    # Wave 2a signed receipt (top-level).
    signed_receipt: Optional[SignedReceipt] = Field(None)

    explain: Dict[str, Any] = Field(default_factory=dict)
    edition_id: Optional[str] = Field(None)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_receipt_key(cls, data: Any) -> Any:
        """Accept either ``signed_receipt`` (Wave 2a) or ``_signed_receipt``
        (legacy) when parsing a resolved-factor payload.

        Emits a :class:`DeprecationWarning` on the legacy key so callers can
        migrate their integrations before v2.0.0.
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "signed_receipt" not in out and "_signed_receipt" in out:
            warnings.warn(
                "Top-level '_signed_receipt' is deprecated; server should emit 'signed_receipt'. "
                "SDK fallback will be removed in v2.0.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            out["signed_receipt"] = out.pop("_signed_receipt")
        # Normalise a dict receipt through the compat parser so that the
        # key renames (algorithm -> alg, signed_over -> payload_hash) are
        # handled before Pydantic validation rejects the unknown aliases.
        if isinstance(out.get("signed_receipt"), dict):
            out["signed_receipt"] = SignedReceipt.from_response_dict(
                out["signed_receipt"]
            )
        return out


# ---------------------------------------------------------------------------
# Diff / Audit
# ---------------------------------------------------------------------------


class FactorDiff(_SDKResponseModel):
    """Field-by-field diff returned by /factors/{id}/diff.

    Non-factor_id fields are optional so the SDK can parse partial /
    mocked responses where e.g. `left_edition` is carried on the
    response header instead of the body.
    """

    factor_id: str
    left_edition: Optional[str] = Field(None)
    right_edition: Optional[str] = Field(None)
    status: Optional[str] = Field(
        None, description="unchanged | changed | added | removed | not_found"
    )
    left: Dict[str, Any] = Field(default_factory=dict)
    right: Dict[str, Any] = Field(default_factory=dict)
    left_exists: Optional[bool] = Field(None)
    right_exists: Optional[bool] = Field(None)
    changes: List[Dict[str, Any]] = Field(default_factory=list)
    left_content_hash: Optional[str] = Field(None)
    right_content_hash: Optional[str] = Field(None)


class AuditBundle(_SDKResponseModel):
    """Full audit bundle from /factors/{id}/audit-bundle (Enterprise only).

    ``edition_id`` is usually delivered in the response body but may be
    carried in the ``X-GreenLang-Edition`` header for mock / partial
    responses — leave it optional so SDK parsing survives either shape.
    """

    factor_id: str
    edition_id: Optional[str] = Field(None)
    content_hash: Optional[str] = Field(None)
    payload_sha256: Optional[str] = Field(None)
    normalized_record: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    license_info: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)
    verification_chain: Dict[str, Any] = Field(default_factory=dict)
    raw_artifact_uri: Optional[str] = Field(None)
    parser_log: Optional[str] = Field(None)
    qa_errors: List[str] = Field(default_factory=list)
    reviewer_decision: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Tenant overrides (Consulting/Platform tier)
# ---------------------------------------------------------------------------


class Override(GreenLangBase):
    """Tenant-scoped factor override (POST /factors/overrides)."""

    model_config = ConfigDict(extra="allow", validate_default=True)

    factor_id: str = Field(..., description="Factor being overridden")
    tenant_id: Optional[str] = Field(
        None, description="Tenant scope (server fills from auth if omitted)"
    )
    co2e_per_unit: Optional[float] = Field(None)
    justification: Optional[str] = Field(
        None, description="Why the override is applied (audit trail)"
    )
    effective_from: Optional[str] = Field(None)
    effective_to: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Batch jobs
# ---------------------------------------------------------------------------


class BatchJobHandle(_SDKResponseModel):
    """Handle returned by POST /factors/resolve/batch.

    Batch resolution jobs run asynchronously server-side.  The SDK
    exposes a polling helper (``client.wait_for_batch()``) that loops
    on ``GET /factors/jobs/{job_id}`` until ``status`` reaches a
    terminal value.
    """

    job_id: str = Field(...)
    status: str = Field(
        ..., description="queued | running | completed | failed | cancelled"
    )
    total_items: Optional[int] = Field(None)
    processed_items: Optional[int] = Field(None)
    progress_percent: Optional[float] = Field(None)
    results_url: Optional[str] = Field(None)
    created_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)
    results: Optional[List[Dict[str, Any]]] = Field(None)

    @property
    def is_terminal(self) -> bool:
        """True when the job has reached a terminal state."""
        return self.status in {"completed", "failed", "cancelled"}


__all__ = [
    "Jurisdiction",
    "ActivitySchema",
    "QualityScore",
    "Uncertainty",
    "GasBreakdown",
    "Source",
    "MethodPack",
    "Edition",
    "Factor",
    "FactorMatch",
    "SearchResponse",
    "CoverageReport",
    "ResolutionRequest",
    "ResolvedFactor",
    "FactorDiff",
    "AuditBundle",
    "Override",
    "BatchJobHandle",
    # Wave 2 envelope models
    "ChosenFactor",
    "SourceDescriptor",
    "QualityEnvelope",
    "UncertaintyEnvelope",
    "LicensingEnvelope",
    "DeprecationStatus",
    "SignedReceipt",
]
