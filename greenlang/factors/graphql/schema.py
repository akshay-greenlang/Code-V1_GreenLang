# -*- coding: utf-8 -*-
"""GraphQL schema for the Factors API.

The schema mirrors the 16-field REST contract one-to-one via the
envelope dataclasses in :mod:`greenlang.factors.resolution.result`.
If a field isn't exposed in REST, it isn't exposed here. The
conversion helpers in :mod:`resolvers` guarantee this invariant.

We prefer ``strawberry`` when it's installed (it's the FastAPI-idiomatic
GraphQL library, supports async, introspection, GraphiQL) and fall back
to a minimal typed shim so imports don't explode in environments
without the library. The shim doesn't execute queries — callers who
hit :func:`build_schema` without strawberry installed get a clean
``RuntimeError`` and the /v1/graphql route returns 503.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


try:  # pragma: no cover - exercised at import-time only
    import strawberry  # type: ignore

    _HAS_STRAWBERRY = True
except Exception:  # pragma: no cover
    strawberry = None  # type: ignore
    _HAS_STRAWBERRY = False


# ---------------------------------------------------------------------------
# GraphQL Types — mirror the 16-field REST contract exactly.
# ---------------------------------------------------------------------------


if _HAS_STRAWBERRY:

    @strawberry.type
    class GasBreakdownGQL:
        co2_kg: float = 0.0
        ch4_kg: float = 0.0
        n2o_kg: float = 0.0
        hfcs_kg: float = 0.0
        pfcs_kg: float = 0.0
        sf6_kg: float = 0.0
        nf3_kg: float = 0.0
        biogenic_co2_kg: float = 0.0
        co2e_total_kg: float = 0.0
        gwp_basis: str = "IPCC_AR6_100"

    @strawberry.type
    class ChosenFactorGQL:
        id: str
        name: str
        version: str
        factor_family: str

    @strawberry.type
    class SourceDescriptorGQL:
        id: str
        version: str
        name: str
        authority: str

    @strawberry.type
    class QualityEnvelopeGQL:
        composite_fqs_0_100: float
        temporal_score: float
        geographic_score: float
        technology_score: float
        verification_score: float
        completeness_score: float
        rating: Optional[str] = None
        formula_version: Optional[str] = None

    @strawberry.type
    class UncertaintyEnvelopeGQL:
        type: str
        low: Optional[float] = None
        high: Optional[float] = None
        distribution: Optional[str] = None
        note: Optional[str] = None

    @strawberry.type
    class LicensingEnvelopeGQL:
        redistribution_class: str
        customer_entitlement_required: bool = False
        watermark_required: bool = False

    @strawberry.type
    class DeprecationStatusGQL:
        status: str
        replacement_pointer_factor_id: Optional[str] = None
        deprecated_at: Optional[str] = None
        reason: Optional[str] = None

    @strawberry.type
    class AlternateCandidateGQL:
        factor_id: str
        tie_break_score: float
        why_not_chosen: str
        reason_lost: Optional[str] = None
        source_id: Optional[str] = None
        vintage: Optional[int] = None
        redistribution_class: Optional[str] = None

    @strawberry.type
    class ResolvedFactorGQL:
        # #1 chosen factor envelope
        chosen_factor: Optional[ChosenFactorGQL] = None
        # #4 source envelope
        source: Optional[SourceDescriptorGQL] = None
        # #10 quality envelope
        quality: Optional[QualityEnvelopeGQL] = None
        # #11 uncertainty envelope
        uncertainty: Optional[UncertaintyEnvelopeGQL] = None
        # #12 licensing envelope
        licensing: Optional[LicensingEnvelopeGQL] = None
        # #15 deprecation envelope
        deprecation_status: Optional[DeprecationStatusGQL] = None
        # #2 alternates
        alternates: List[AlternateCandidateGQL] = strawberry.field(default_factory=list)
        # #3 why
        why_this_won: Optional[str] = None
        # #5 release_version
        release_version: Optional[str] = None
        # #6 method_pack
        method_pack: Optional[str] = None
        # #7 validity window
        valid_from: Optional[str] = None
        valid_to: Optional[str] = None
        # #8 gas breakdown
        gas_breakdown: Optional[GasBreakdownGQL] = None
        # #9 co2e basis
        co2e_basis: Optional[str] = None
        # #13 assumptions
        assumptions: List[str] = strawberry.field(default_factory=list)
        # #14 fallback rank
        fallback_rank: Optional[int] = None
        step_label: Optional[str] = None
        # #16 audit text
        audit_text: Optional[str] = None
        audit_text_draft: Optional[bool] = None
        # Tracing
        resolved_at: Optional[str] = None
        engine_version: Optional[str] = None
        method_pack_version: Optional[str] = None
        # Legacy compat (deprecated in REST; surfaced identically here so the
        # two contracts are byte-compatible).
        chosen_factor_id: Optional[str] = None
        method_profile: Optional[str] = None

    @strawberry.type
    class MethodPackGQL:
        pack_id: str
        name: str
        version: str
        description: Optional[str] = None

    @strawberry.type
    class SourceGQL:
        id: str
        name: str
        authority: Optional[str] = None
        version: Optional[str] = None

    @strawberry.type
    class EditionGQL:
        edition_id: str
        label: Optional[str] = None
        created_at: Optional[str] = None

    @strawberry.type
    class ReleaseGQL:
        edition_id: str
        cut_at: Optional[str] = None
        approver: Optional[str] = None

    @strawberry.type
    class FactorRecordGQL:
        factor_id: str
        factor_family: Optional[str] = None
        fuel_type: Optional[str] = None
        status: Optional[str] = None
        redistribution_class: Optional[str] = None

    @strawberry.type
    class QualitySummaryGQL:
        factor_id: str
        composite_fqs_0_100: float
        rating: Optional[str] = None
        label: Optional[str] = None

    @strawberry.type
    class ExplainPayloadGQL:
        receipt_id: Optional[str] = None
        factor_id: Optional[str] = None
        resolved_json: Optional[str] = None  # canonical JSON of ResolvedFactor
        explain_json: Optional[str] = None   # canonical JSON of .explain()


def build_schema() -> Any:
    """Return a Strawberry Schema.

    Raises:
        RuntimeError: if Strawberry is not installed.
    """
    if not _HAS_STRAWBERRY:  # pragma: no cover
        raise RuntimeError(
            "GraphQL support requires 'strawberry-graphql'. "
            "Install with `pip install strawberry-graphql[fastapi]`."
        )
    from .resolvers import Query

    return strawberry.Schema(query=Query)


__all__ = [
    "_HAS_STRAWBERRY",
    "build_schema",
]
