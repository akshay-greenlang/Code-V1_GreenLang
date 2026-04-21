# -*- coding: utf-8 -*-
"""
GraphQL Type Definitions for GreenLang Factors (GAP-12).

Strawberry types that mirror the SDK Pydantic models in
:mod:`greenlang.factors.sdk.python.models` so every REST endpoint
in :mod:`greenlang.integration.api.routes.factors` has a GraphQL
equivalent (query or mutation).

Design rules:
    * One-to-one mirror of the SDK shapes so generated client code
      round-trips through the REST <-> GraphQL surface cleanly.
    * All quantitative fields are nullable Float — factors in the catalog
      may not have CH4/N2O components and the GraphQL response surface
      should not force clients to special-case `null`.
    * Relay-style connection for paginated queries so clients can use
      cursor-based pagination (first/after) without changing the REST
      offset/limit contract.
    * Field-level `description=` strings so the generated
      introspection payload is self-documenting.
"""

from __future__ import annotations

import strawberry
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------
# Scalar + PageInfo.
#
# The existing ``greenlang.integration.api.graphql.types`` module has
# broken transitive imports (legacy ``greenlang.api.graphql.*`` paths
# that no longer exist).  To keep the Factors types independently
# importable we define a JSON scalar + PageInfo here.  When the base
# module is repaired, the Factors build can switch to importing from
# there; until then these locally-defined shapes are the source of
# truth for the Factors surface.
# ----------------------------------------------------------------------

from greenlang.schemas.enums import SortOrder

JSON = strawberry.scalar(
    Dict[str, Any],
    serialize=lambda v: v,
    parse_value=lambda v: v,
    description="Opaque JSON value.",
    name="JSON",
)


@strawberry.type
class PageInfo:
    """Relay-style pagination info."""

    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str]
    end_cursor: Optional[str]
    total_count: int
    total_pages: int
    current_page: int


# ==============================================================================
# Enum Types
# ==============================================================================


@strawberry.enum
class FactorStatus(Enum):
    """Lifecycle status for an emission factor."""

    CERTIFIED = "certified"
    PREVIEW = "preview"
    CONNECTOR_ONLY = "connector_only"
    DEPRECATED = "deprecated"


@strawberry.enum
class EditionStatus(Enum):
    """Catalog edition lifecycle."""

    PUBLISHED = "published"
    PENDING = "pending"
    DEPRECATED = "deprecated"


@strawberry.enum
class FactorScope(Enum):
    """GHG Protocol scope label."""

    SCOPE_1 = "1"
    SCOPE_2 = "2"
    SCOPE_3 = "3"


@strawberry.enum
class BatchJobStatus(Enum):
    """Batch resolution job state."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@strawberry.enum
class FactorSortField(Enum):
    """Sortable fields for factor list queries."""

    RELEVANCE = "relevance"
    FACTOR_ID = "factor_id"
    CO2E_PER_UNIT = "co2e_per_unit"
    DATA_QUALITY = "data_quality"
    VALID_FROM = "valid_from"
    SOURCE_YEAR = "source_year"


# ==============================================================================
# Value-object types (nested within Factor / ResolvedFactor)
# ==============================================================================


@strawberry.type
class Jurisdiction:
    """Geographic jurisdiction tag."""

    code: str
    name: Optional[str] = None
    level: Optional[str] = None


@strawberry.type
class ActivitySchema:
    """Schema describing an activity type."""

    activity_id: str
    label: Optional[str] = None
    unit: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None


@strawberry.type
class QualityScore:
    """Pedigree / data-quality score breakdown."""

    overall_score: float
    rating: Optional[str] = None
    temporal: Optional[float] = None
    geographical: Optional[float] = None
    technological: Optional[float] = None
    representativeness: Optional[float] = None
    methodological: Optional[float] = None


@strawberry.type
class Uncertainty:
    """Uncertainty envelope for a factor or resolved result."""

    ci_95: Optional[float] = None
    distribution: Optional[str] = None
    std_dev: Optional[float] = None
    sample_size: Optional[int] = None


@strawberry.type
class GasBreakdown:
    """Per-gas breakdown of CO2-equivalent emissions.

    HFCs, PFCs, SF6, NF3, and biogenic CO2 are kept separate per CTO
    non-negotiable — never rolled up.
    """

    co2: Optional[float] = None
    ch4: Optional[float] = None
    n2o: Optional[float] = None
    hfcs: Optional[float] = None
    pfcs: Optional[float] = None
    sf6: Optional[float] = None
    nf3: Optional[float] = None
    biogenic_co2: Optional[float] = None
    ch4_gwp: Optional[float] = None
    n2o_gwp: Optional[float] = None


# ==============================================================================
# Core entity types
# ==============================================================================


@strawberry.type
class Source:
    """Upstream publisher (EPA, DEFRA, IEA, ...)."""

    source_id: strawberry.ID
    organization: Optional[str] = None
    publication: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    methodology: Optional[str] = None
    license: Optional[str] = None
    version: Optional[str] = None


@strawberry.type
class MethodPack:
    """Method pack descriptor."""

    method_pack_id: strawberry.ID
    name: Optional[str] = None
    version: Optional[str] = None
    scope: Optional[str] = None
    description: Optional[str] = None
    jurisdictions: List[str] = strawberry.field(default_factory=list)


@strawberry.type
class Edition:
    """Catalog edition descriptor."""

    edition_id: strawberry.ID
    status: Optional[str] = None
    label: Optional[str] = None
    manifest_hash: Optional[str] = None
    published_at: Optional[datetime] = None


@strawberry.type
class Factor:
    """Full emission factor record."""

    factor_id: strawberry.ID
    fuel_type: Optional[str] = None
    unit: Optional[str] = None
    geography: Optional[str] = None
    geography_level: Optional[str] = None
    scope: Optional[str] = None
    boundary: Optional[str] = None

    co2_per_unit: Optional[float] = None
    ch4_per_unit: Optional[float] = None
    n2o_per_unit: Optional[float] = None
    co2e_per_unit: Optional[float] = None

    data_quality: Optional[QualityScore] = None
    source: Optional[Source] = None
    uncertainty_95ci: Optional[float] = None

    valid_from: Optional[str] = None
    valid_to: Optional[str] = None

    factor_status: Optional[str] = None
    license: Optional[str] = None
    license_class: Optional[str] = None
    compliance_frameworks: List[str] = strawberry.field(default_factory=list)
    tags: List[str] = strawberry.field(default_factory=list)
    activity_tags: List[str] = strawberry.field(default_factory=list)
    sector_tags: List[str] = strawberry.field(default_factory=list)
    notes: Optional[str] = None

    edition_id: Optional[str] = None
    source_id: Optional[str] = None
    source_release: Optional[str] = None
    release_version: Optional[str] = None
    replacement_factor_id: Optional[str] = None
    content_hash: Optional[str] = None


@strawberry.type
class FactorMatch:
    """Candidate returned by `match` query."""

    factor_id: strawberry.ID
    score: float
    explanation: Optional[JSON] = None


@strawberry.type
class ResolvedFactor:
    """Output of the 7-step resolution cascade."""

    chosen_factor_id: Optional[str] = None
    factor_id: Optional[str] = None
    factor_version: Optional[str] = None
    method_profile: Optional[str] = None
    method_pack_version: Optional[str] = None

    fallback_rank: Optional[int] = None
    step_label: Optional[str] = None
    why_chosen: Optional[str] = None

    quality_score: Optional[QualityScore] = None
    uncertainty: Optional[Uncertainty] = None
    gas_breakdown: Optional[GasBreakdown] = None
    co2e_basis: Optional[str] = None

    assumptions: List[str] = strawberry.field(default_factory=list)
    alternates: Optional[JSON] = None

    deprecation_status: Optional[str] = None
    deprecation_replacement: Optional[str] = None

    explain: Optional[JSON] = None
    edition_id: Optional[str] = None


@strawberry.type
class FactorDiff:
    """Field-by-field diff of a factor between two editions."""

    factor_id: strawberry.ID
    left_edition: str
    right_edition: str
    status: str
    left_exists: Optional[bool] = None
    right_exists: Optional[bool] = None
    changes: Optional[JSON] = None
    left_content_hash: Optional[str] = None
    right_content_hash: Optional[str] = None


@strawberry.type
class AuditBundle:
    """Full audit bundle (Enterprise tier only)."""

    factor_id: strawberry.ID
    edition_id: str
    content_hash: Optional[str] = None
    payload_sha256: Optional[str] = None
    normalized_record: Optional[JSON] = None
    provenance: Optional[JSON] = None
    license_info: Optional[JSON] = None
    quality: Optional[JSON] = None
    verification_chain: Optional[JSON] = None
    raw_artifact_uri: Optional[str] = None
    parser_log: Optional[str] = None
    qa_errors: List[str] = strawberry.field(default_factory=list)
    reviewer_decision: Optional[str] = None


@strawberry.type
class Override:
    """Tenant-scoped factor override."""

    overlay_id: strawberry.ID
    tenant_id: strawberry.ID
    factor_id: str
    override_value: float
    override_unit: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    active: bool = True


@strawberry.type
class CoverageReport:
    """Coverage statistics response."""

    total_factors: Optional[int] = None
    by_geography: Optional[JSON] = None
    by_scope: Optional[JSON] = None
    by_fuel_type: Optional[JSON] = None
    by_source: Optional[JSON] = None
    edition_id: Optional[str] = None


@strawberry.type
class BatchJobHandle:
    """Handle for an async batch resolution job."""

    job_id: strawberry.ID
    status: str
    total_items: Optional[int] = None
    processed_items: Optional[int] = None
    progress_percent: Optional[float] = None
    results_url: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# ==============================================================================
# Connection (Relay-style pagination)
# ==============================================================================


@strawberry.type
class FactorEdge:
    """Edge in a FactorConnection."""

    node: Factor
    cursor: str


@strawberry.type
class FactorConnection:
    """Paginated list of factors (Relay connection)."""

    edges: List[FactorEdge]
    nodes: List[Factor]
    page_info: PageInfo
    total_count: int
    edition_id: Optional[str] = None
    search_time_ms: Optional[float] = None


# ==============================================================================
# Input types
# ==============================================================================


@strawberry.input
class FactorFilterInput:
    """Filter criteria for factor list/search queries."""

    fuel_type: Optional[str] = None
    geography: Optional[str] = None
    scope: Optional[str] = None
    boundary: Optional[str] = None
    source_id: Optional[str] = None
    factor_status: Optional[str] = None
    license_class: Optional[str] = None
    edition: Optional[str] = None
    activity_tags: Optional[List[str]] = None
    sector_tags: Optional[List[str]] = None
    dqs_min: Optional[float] = None
    valid_on_date: Optional[str] = None
    include_preview: bool = False
    include_connector: bool = False


@strawberry.input
class FactorSortInput:
    """Sort spec for factor queries."""

    field: FactorSortField = FactorSortField.RELEVANCE
    order: SortOrder = SortOrder.DESC


@strawberry.input
class ResolutionRequestInput:
    """Input payload for the 7-step resolution cascade."""

    activity: str
    method_profile: str
    jurisdiction: Optional[str] = None
    reporting_date: Optional[str] = None
    supplier_id: Optional[str] = None
    facility_id: Optional[str] = None
    utility_or_grid_region: Optional[str] = None
    preferred_sources: Optional[List[str]] = None
    extras: Optional[JSON] = None
    edition: Optional[str] = None
    include_preview: bool = False
    include_connector: bool = False


@strawberry.input
class SearchFactorsInput:
    """Advanced search input for `searchFactors`."""

    query: str
    filter: Optional[FactorFilterInput] = None
    sort: Optional[FactorSortInput] = None
    first: Optional[int] = None
    after: Optional[str] = None


@strawberry.input
class MatchInput:
    """Input for activity-to-factor matching."""

    activity_description: str
    geography: Optional[str] = None
    fuel_type: Optional[str] = None
    scope: Optional[str] = None
    limit: Optional[int] = 10
    edition: Optional[str] = None


@strawberry.input
class OverrideInput:
    """Input for creating a tenant factor override."""

    factor_id: strawberry.ID
    tenant_id: Optional[strawberry.ID] = None
    co2e_per_unit: float
    override_unit: Optional[str] = "kg_co2e"
    justification: Optional[str] = None
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[JSON] = None


__all__ = [
    # Enums
    "FactorStatus",
    "EditionStatus",
    "FactorScope",
    "BatchJobStatus",
    "FactorSortField",
    # Value objects
    "Jurisdiction",
    "ActivitySchema",
    "QualityScore",
    "Uncertainty",
    "GasBreakdown",
    # Entities
    "Source",
    "MethodPack",
    "Edition",
    "Factor",
    "FactorMatch",
    "ResolvedFactor",
    "FactorDiff",
    "AuditBundle",
    "Override",
    "CoverageReport",
    "BatchJobHandle",
    # Connections
    "FactorEdge",
    "FactorConnection",
    # Inputs
    "FactorFilterInput",
    "FactorSortInput",
    "ResolutionRequestInput",
    "SearchFactorsInput",
    "MatchInput",
    "OverrideInput",
]
