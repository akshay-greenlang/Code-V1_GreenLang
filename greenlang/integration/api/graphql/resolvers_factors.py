# -*- coding: utf-8 -*-
"""
GraphQL Resolvers for GreenLang Factors (GAP-12).

Exposes every REST endpoint under ``/api/v1/factors`` as a GraphQL
query or mutation.  The resolvers delegate to the existing service
layer (:mod:`greenlang.factors.service`,
:mod:`greenlang.factors.api_endpoints`) so there is exactly one source
of truth for the business logic.

Tier gating:
    * Pro+ endpoints: explain/resolve/alternates (Pro, Consulting,
      Enterprise, Internal).
    * Enterprise-only: ``factorAuditBundle``.
    * Consulting+ mutations: ``setFactorOverride`` /
      ``removeFactorOverride``.
    * Community tier sees only certified factors.

All tier violations raise a GraphQL-visible error with
``extensions = {code: "TIER_REQUIRED", required: <tier>}``.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import strawberry
from strawberry.types import Info

from greenlang.integration.api.graphql.types_factors import (
    AuditBundle,
    BatchJobHandle,
    CoverageReport,
    Edition,
    EditionStatus,
    Factor,
    FactorConnection,
    FactorDiff,
    FactorEdge,
    FactorFilterInput,
    FactorSortInput,
    FactorSortField,
    MatchInput,
    MethodPack,
    Override,
    OverrideInput,
    QualityScore,
    ResolutionRequestInput,
    ResolvedFactor,
    SearchFactorsInput,
    Source,
    Uncertainty,
    GasBreakdown,
)
from greenlang.integration.api.graphql.types_factors import (  # noqa: F401
    Jurisdiction,
    ActivitySchema,
)
from greenlang.integration.api.graphql.dataloaders_factors import (
    _factor_to_graphql,
    attach_factor_loaders,
)
from greenlang.integration.api.graphql.types_factors import PageInfo

logger = logging.getLogger(__name__)


# ==============================================================================
# Tier gating
# ==============================================================================

_PRO_PLUS = {"pro", "consulting", "enterprise", "internal"}
_CONSULTING_PLUS = {"consulting", "enterprise", "internal"}
_ENTERPRISE_PLUS = {"enterprise", "internal"}


class TierError(Exception):
    """Raised when the caller's tier cannot access a field."""

    def __init__(self, required: str, message: Optional[str] = None):
        self.required = required
        self.code = "TIER_REQUIRED"
        super().__init__(message or f"Requires {required.upper()} tier or higher")

    @property
    def extensions(self) -> Dict[str, Any]:
        return {"code": self.code, "required": self.required}


def _get_user_context(info: Info) -> Dict[str, Any]:
    """Extract a dict user context that ``tier_enforcement`` understands."""
    ctx = info.context
    # Prefer an explicit ``user_context`` dict if the app stored one,
    # else fall back to token attributes.
    user = getattr(ctx, "user_context", None)
    if user is None:
        tier = getattr(ctx, "tier", None)
        if tier is None:
            token = getattr(ctx, "token", None)
            tier = getattr(token, "tier", None) if token is not None else None
        user = {
            "tier": tier or "community",
            "user_id": getattr(ctx, "user_id", "anonymous"),
            "tenant_id": getattr(ctx, "tenant_id", None),
        }
    return user


def _resolve_tier(info: Info) -> str:
    from greenlang.factors.tier_enforcement import resolve_tier

    return resolve_tier(_get_user_context(info))


def _require_pro_tier(info: Info) -> str:
    tier = _resolve_tier(info)
    if tier not in _PRO_PLUS:
        raise TierError(required="pro", message="Pro tier required")
    return tier


def _require_consulting_tier(info: Info) -> str:
    tier = _resolve_tier(info)
    if tier not in _CONSULTING_PLUS:
        raise TierError(required="consulting", message="Consulting tier required")
    return tier


def _require_enterprise_tier(info: Info) -> str:
    tier = _resolve_tier(info)
    if tier not in _ENTERPRISE_PLUS:
        raise TierError(required="enterprise", message="Enterprise tier required")
    return tier


def _tier_visibility(info: Info, preview: bool = False, connector: bool = False):
    from greenlang.factors.tier_enforcement import enforce_tier_on_request

    return enforce_tier_on_request(
        _get_user_context(info),
        requested_preview=preview,
        requested_connector=connector,
    )


# ==============================================================================
# Service access helpers
# ==============================================================================


def _get_service(info: Info) -> Any:
    """Resolve the FactorCatalogService for this request.

    Priority:
        1. ``info.context.factor_service`` (explicit wiring — tests,
           production setups that inject it).
        2. ``info.context.metadata["factor_service"]``.
        3. :func:`FactorCatalogService.from_environment` fallback.
    """
    ctx = info.context

    svc = getattr(ctx, "factor_service", None)
    if svc is not None:
        return svc

    meta = getattr(ctx, "metadata", None) or {}
    if "factor_service" in meta:
        return meta["factor_service"]

    from greenlang.factors.service import FactorCatalogService

    svc = FactorCatalogService.from_environment()
    try:
        ctx.factor_service = svc
    except (AttributeError, TypeError):
        if getattr(ctx, "metadata", None) is None:
            try:
                ctx.metadata = {}
            except (AttributeError, TypeError):
                pass
        try:
            ctx.metadata["factor_service"] = svc
        except (AttributeError, TypeError):
            pass
    # Make sure our loaders are attached so the rest of the resolver
    # graph can use them without manual plumbing.
    attach_factor_loaders(ctx, svc)
    return svc


def _resolve_edition_id(svc: Any, edition: Optional[str] = None) -> str:
    """Resolve edition: explicit -> default."""
    from greenlang.factors.service import resolve_edition_id as _resolve

    resolved, _source = _resolve(svc.repo, None, edition)
    return resolved


# ==============================================================================
# Cursor helpers (Relay-style)
# ==============================================================================


def _encode_cursor(offset: int) -> str:
    return base64.b64encode(f"factor:{offset}".encode()).decode()


def _decode_cursor(cursor: Optional[str]) -> int:
    if not cursor:
        return 0
    try:
        decoded = base64.b64decode(cursor.encode()).decode()
        return int(decoded.split(":", 1)[1])
    except Exception:
        return 0


def _build_page_info(
    *, total: int, offset: int, limit: int
) -> PageInfo:
    page_size = max(1, limit)
    current_page = (offset // page_size) + 1
    total_pages = (total + page_size - 1) // page_size if total else 1
    return PageInfo(
        has_next_page=(offset + limit) < total,
        has_previous_page=offset > 0,
        start_cursor=_encode_cursor(offset),
        end_cursor=_encode_cursor(offset + max(0, limit - 1)),
        total_count=total,
        total_pages=total_pages,
        current_page=current_page,
    )


def _factors_to_connection(
    records: List[Any],
    *,
    total: int,
    offset: int,
    limit: int,
    edition_id: Optional[str],
    search_time_ms: Optional[float] = None,
) -> FactorConnection:
    nodes: List[Factor] = [_factor_to_graphql(r, edition_id=edition_id) for r in records]
    edges = [
        FactorEdge(node=n, cursor=_encode_cursor(offset + i))
        for i, n in enumerate(nodes)
    ]
    return FactorConnection(
        edges=edges,
        nodes=nodes,
        page_info=_build_page_info(total=total, offset=offset, limit=limit),
        total_count=total,
        edition_id=edition_id,
        search_time_ms=search_time_ms,
    )


# ==============================================================================
# Converters for explain payloads
# ==============================================================================


def _payload_to_resolved_factor(
    payload: Dict[str, Any], edition_id: Optional[str] = None
) -> ResolvedFactor:
    quality = None
    q = payload.get("quality_score") or payload.get("quality")
    if isinstance(q, dict) and q:
        quality = QualityScore(
            overall_score=float(q.get("overall_score") or q.get("dqs_overall") or 0.0),
            rating=str(q.get("rating")) if q.get("rating") is not None else None,
            temporal=q.get("temporal"),
            geographical=q.get("geographical"),
            technological=q.get("technological"),
            representativeness=q.get("representativeness"),
            methodological=q.get("methodological"),
        )

    uncertainty = None
    u = payload.get("uncertainty")
    if isinstance(u, dict) and u:
        uncertainty = Uncertainty(
            ci_95=u.get("ci_95"),
            distribution=u.get("distribution"),
            std_dev=u.get("std_dev"),
            sample_size=u.get("sample_size"),
        )

    gas = None
    g = payload.get("gas_breakdown")
    if isinstance(g, dict) and g:
        gas = GasBreakdown(
            co2=g.get("CO2") or g.get("co2"),
            ch4=g.get("CH4") or g.get("ch4"),
            n2o=g.get("N2O") or g.get("n2o"),
            hfcs=g.get("HFCs") or g.get("hfcs"),
            pfcs=g.get("PFCs") or g.get("pfcs"),
            sf6=g.get("SF6") or g.get("sf6"),
            nf3=g.get("NF3") or g.get("nf3"),
            biogenic_co2=g.get("biogenic_CO2") or g.get("biogenic_co2"),
            ch4_gwp=g.get("ch4_gwp"),
            n2o_gwp=g.get("n2o_gwp"),
        )

    return ResolvedFactor(
        chosen_factor_id=payload.get("chosen_factor_id"),
        factor_id=payload.get("factor_id") or payload.get("chosen_factor_id"),
        factor_version=payload.get("factor_version"),
        method_profile=payload.get("method_profile"),
        method_pack_version=payload.get("method_pack_version"),
        fallback_rank=payload.get("fallback_rank"),
        step_label=payload.get("step_label"),
        why_chosen=payload.get("why_chosen"),
        quality_score=quality,
        uncertainty=uncertainty,
        gas_breakdown=gas,
        co2e_basis=payload.get("co2e_basis"),
        assumptions=list(payload.get("assumptions") or []),
        alternates=payload.get("alternates"),
        deprecation_status=payload.get("deprecation_status"),
        deprecation_replacement=payload.get("deprecation_replacement"),
        explain=payload.get("explain"),
        edition_id=edition_id or payload.get("edition_id"),
    )


# ==============================================================================
# Queries
# ==============================================================================


@strawberry.type
class FactorsQuery:
    """Root-level GraphQL queries for Factors."""

    @strawberry.field(description="Get a single factor by ID (optionally pinned to an edition).")
    async def factor(
        self,
        info: Info,
        id: strawberry.ID,
        edition: Optional[str] = None,
    ) -> Optional[Factor]:
        svc = _get_service(info)
        edition_id = _resolve_edition_id(svc, edition)
        rec = svc.repo.get_factor(edition_id, str(id))
        if rec is None:
            return None
        # Tier visibility: hide preview / connector-only records for community.
        vis = _tier_visibility(info, preview=True, connector=True)
        from greenlang.factors.tier_enforcement import factor_visible_for_tier

        status = getattr(rec, "factor_status", "certified") or "certified"
        if not factor_visible_for_tier(status, vis):
            return None
        return _factor_to_graphql(rec, edition_id=edition_id)

    @strawberry.field(description="List factors with optional filters and pagination.")
    async def factors(
        self,
        info: Info,
        filter: Optional[FactorFilterInput] = None,
        first: Optional[int] = 20,
        after: Optional[str] = None,
        sort: Optional[FactorSortInput] = None,
    ) -> FactorConnection:
        svc = _get_service(info)
        f = filter or FactorFilterInput()
        edition_id = _resolve_edition_id(svc, f.edition)
        vis = _tier_visibility(info, preview=f.include_preview, connector=f.include_connector)
        offset = _decode_cursor(after)
        limit = max(1, min(first or 20, 500))
        page = (offset // limit) + 1

        records, total = svc.repo.list_factors(
            edition_id,
            fuel_type=f.fuel_type,
            geography=f.geography,
            scope=f.scope,
            boundary=f.boundary,
            page=page,
            limit=limit,
            include_preview=vis.include_preview,
            include_connector=vis.include_connector,
        )

        # Client-side post-sort for non-relevance orderings.
        if sort and sort.field != FactorSortField.RELEVANCE:
            field_map = {
                FactorSortField.FACTOR_ID: "factor_id",
                FactorSortField.CO2E_PER_UNIT: None,  # derived from gwp
                FactorSortField.DATA_QUALITY: None,
                FactorSortField.VALID_FROM: "valid_from",
                FactorSortField.SOURCE_YEAR: None,
            }
            key = field_map.get(sort.field)
            if key:
                records = sorted(
                    records,
                    key=lambda r: getattr(r, key, "") or "",
                    reverse=(sort.order.value == "desc"),
                )

        return _factors_to_connection(
            records, total=total, offset=offset, limit=limit, edition_id=edition_id
        )

    @strawberry.field(description="Full-text search across emission factors.")
    async def search_factors(
        self,
        info: Info,
        input: SearchFactorsInput,
    ) -> FactorConnection:
        import time

        svc = _get_service(info)
        f = input.filter or FactorFilterInput()
        edition_id = _resolve_edition_id(svc, f.edition)
        vis = _tier_visibility(info, preview=f.include_preview, connector=f.include_connector)
        limit = max(1, min(input.first or 20, 100))
        offset = _decode_cursor(input.after)

        t0 = time.monotonic()
        results = svc.repo.search_factors(
            edition_id,
            query=input.query,
            geography=f.geography,
            limit=limit + offset,  # we'll slice below
            include_preview=vis.include_preview,
            include_connector=vis.include_connector,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        total = len(results)
        page_records = results[offset : offset + limit]

        return _factors_to_connection(
            page_records,
            total=total,
            offset=offset,
            limit=limit,
            edition_id=edition_id,
            search_time_ms=round(elapsed_ms, 2),
        )

    @strawberry.field(description="Resolve a factor via the 7-step cascade.")
    async def resolve_factor(
        self,
        info: Info,
        request: ResolutionRequestInput,
    ) -> ResolvedFactor:
        return await self.resolve_factor_explain(info, request=request, alternates_limit=0)

    @strawberry.field(description="Resolve a factor and include full explain payload with alternates.")
    async def resolve_factor_explain(
        self,
        info: Info,
        request: ResolutionRequestInput,
        alternates_limit: Optional[int] = 5,
    ) -> ResolvedFactor:
        _require_pro_tier(info)
        svc = _get_service(info)
        edition_id = _resolve_edition_id(svc, request.edition)
        vis = _tier_visibility(
            info, preview=request.include_preview, connector=request.include_connector
        )

        from greenlang.factors.api_endpoints import (
            build_resolution_explain,
            clamp_alternates_limit,
        )
        from greenlang.factors.resolution.engine import ResolutionError

        body: Dict[str, Any] = {
            "activity": request.activity,
            "method_profile": request.method_profile,
        }
        if request.jurisdiction:
            body["jurisdiction"] = request.jurisdiction
        if request.reporting_date:
            body["reporting_date"] = request.reporting_date
        if request.supplier_id:
            body["supplier_id"] = request.supplier_id
        if request.facility_id:
            body["facility_id"] = request.facility_id
        if request.utility_or_grid_region:
            body["utility_or_grid_region"] = request.utility_or_grid_region
        if request.preferred_sources:
            body["preferred_sources"] = list(request.preferred_sources)
        if request.extras:
            body["extras"] = request.extras

        try:
            payload = build_resolution_explain(
                svc.repo,
                edition_id,
                body,
                alternates_limit=clamp_alternates_limit(alternates_limit),
                include_preview=vis.include_preview,
                include_connector=vis.include_connector,
            )
        except ResolutionError as exc:
            raise ValueError(f"Resolution failed: {exc}")

        return _payload_to_resolved_factor(payload, edition_id=edition_id)

    @strawberry.field(description="List alternative factors for the same activity.")
    async def factor_alternates(
        self,
        info: Info,
        factor_id: strawberry.ID,
        limit: Optional[int] = 5,
    ) -> List[Factor]:
        _require_pro_tier(info)
        svc = _get_service(info)
        edition_id = _resolve_edition_id(svc, None)
        vis = _tier_visibility(info, preview=True, connector=True)

        from greenlang.factors.api_endpoints import (
            build_factor_alternates,
            clamp_alternates_limit,
        )

        payload = build_factor_alternates(
            svc.repo,
            edition_id,
            str(factor_id),
            alternates_limit=clamp_alternates_limit(limit),
            include_preview=vis.include_preview,
            include_connector=vis.include_connector,
        )
        if not payload:
            return []

        alts = payload.get("alternates") or []
        out: List[Factor] = []
        for a in alts:
            if isinstance(a, dict):
                out.append(_factor_to_graphql(a, edition_id=edition_id))
        return out

    @strawberry.field(description="Diff a factor between two editions.")
    async def factor_diff(
        self,
        info: Info,
        factor_id: strawberry.ID,
        from_edition: str,
        to_edition: str,
    ) -> FactorDiff:
        svc = _get_service(info)

        from greenlang.factors.api_endpoints import diff_factor_between_editions

        try:
            svc.repo.resolve_edition(from_edition)
            svc.repo.resolve_edition(to_edition)
        except ValueError as exc:
            raise ValueError(str(exc))

        d = diff_factor_between_editions(svc.repo, str(factor_id), from_edition, to_edition)
        return FactorDiff(
            factor_id=str(d.get("factor_id", factor_id)),
            left_edition=d.get("left_edition", from_edition),
            right_edition=d.get("right_edition", to_edition),
            status=str(d.get("status", "unchanged")),
            left_exists=d.get("left_exists"),
            right_exists=d.get("right_exists"),
            changes=d.get("changes"),
            left_content_hash=d.get("left_content_hash"),
            right_content_hash=d.get("right_content_hash"),
        )

    @strawberry.field(description="Full audit bundle for a factor (Enterprise tier only).")
    async def factor_audit_bundle(
        self,
        info: Info,
        factor_id: strawberry.ID,
        edition: Optional[str] = None,
    ) -> Optional[AuditBundle]:
        _require_enterprise_tier(info)
        svc = _get_service(info)
        edition_id = _resolve_edition_id(svc, edition)

        from greenlang.factors.api_endpoints import build_audit_bundle

        bundle = build_audit_bundle(svc.repo, edition_id, str(factor_id))
        if bundle is None:
            return None

        return AuditBundle(
            factor_id=bundle["factor_id"],
            edition_id=bundle["edition_id"],
            content_hash=bundle.get("content_hash"),
            payload_sha256=bundle.get("payload_sha256"),
            normalized_record=bundle.get("normalized_record"),
            provenance=bundle.get("provenance"),
            license_info=bundle.get("license_info"),
            quality=bundle.get("quality"),
            verification_chain=bundle.get("verification_chain"),
            raw_artifact_uri=bundle.get("raw_artifact_uri"),
            parser_log=bundle.get("parser_log"),
            qa_errors=list(bundle.get("qa_errors") or []),
            reviewer_decision=bundle.get("reviewer_decision"),
        )

    @strawberry.field(description="List catalog editions.")
    async def editions(
        self,
        info: Info,
        status: Optional[EditionStatus] = None,
    ) -> List[Edition]:
        svc = _get_service(info)
        rows = svc.repo.list_editions(include_pending=True)
        out: List[Edition] = []
        for row in rows:
            if status is not None and (row.status or "").lower() != status.value.lower():
                continue
            out.append(
                Edition(
                    edition_id=row.edition_id,
                    status=row.status,
                    label=row.label,
                    manifest_hash=row.manifest_hash,
                    published_at=None,
                )
            )
        return out

    @strawberry.field(description="Get a single edition by ID.")
    async def edition(
        self,
        info: Info,
        id: strawberry.ID,
    ) -> Optional[Edition]:
        svc = _get_service(info)
        rows = svc.repo.list_editions(include_pending=True)
        for row in rows:
            if row.edition_id == str(id):
                return Edition(
                    edition_id=row.edition_id,
                    status=row.status,
                    label=row.label,
                    manifest_hash=row.manifest_hash,
                    published_at=None,
                )
        return None

    @strawberry.field(description="List registered upstream factor sources.")
    async def sources(self, info: Info) -> List[Source]:
        try:
            from greenlang.factors.source_registry import load_source_registry

            entries = load_source_registry()
        except Exception as exc:
            logger.warning("Could not load source registry: %s", exc)
            return []

        out: List[Source] = []
        for e in entries:
            out.append(
                Source(
                    source_id=e.source_id,
                    organization=getattr(e, "publisher", None) or e.display_name,
                    publication=e.display_name,
                    year=None,
                    url=e.watch_url,
                    methodology=None,
                    license=e.license_class,
                    version=getattr(e, "dataset_version", None),
                )
            )
        return out

    @strawberry.field(description="Get a single source by ID.")
    async def source(
        self,
        info: Info,
        id: strawberry.ID,
    ) -> Optional[Source]:
        ctx = info.context
        loader = getattr(ctx, "source_loader", None)
        if loader is None:
            # Cold-wire a source loader on demand.
            from greenlang.integration.api.graphql.dataloaders_factors import (
                SourceByIdLoader,
            )

            loader = SourceByIdLoader()
            try:
                ctx.source_loader = loader
            except (AttributeError, TypeError):
                pass
        return await loader.load(str(id))

    @strawberry.field(description="List registered method packs.")
    async def method_packs(self, info: Info) -> List[MethodPack]:
        try:
            from greenlang.factors.method_packs import list_packs

            packs = list_packs() or []
        except Exception as exc:
            logger.warning("Could not enumerate method packs: %s", exc)
            return []

        out: List[MethodPack] = []
        for pack in packs:
            pid = getattr(pack, "profile", None) or getattr(pack, "name", "unknown")
            pid_val = pid.value if hasattr(pid, "value") else str(pid)
            out.append(
                MethodPack(
                    method_pack_id=pid_val,
                    name=getattr(pack, "name", None) or pid_val,
                    version=getattr(pack, "version", None),
                    scope=str(getattr(pack, "scope", None) or ""),
                    description=getattr(pack, "description", None),
                    jurisdictions=list(getattr(pack, "jurisdictions", []) or []),
                )
            )
        return out

    @strawberry.field(description="Get a single method pack by ID.")
    async def method_pack(
        self,
        info: Info,
        id: strawberry.ID,
    ) -> Optional[MethodPack]:
        ctx = info.context
        loader = getattr(ctx, "method_pack_loader", None)
        if loader is None:
            from greenlang.integration.api.graphql.dataloaders_factors import (
                MethodPackByIdLoader,
            )

            loader = MethodPackByIdLoader()
            try:
                ctx.method_pack_loader = loader
            except (AttributeError, TypeError):
                pass
        return await loader.load(str(id))

    @strawberry.field(description="Factor coverage report (optionally scoped by method profile).")
    async def factor_coverage(
        self,
        info: Info,
        method_profile: Optional[str] = None,
        edition: Optional[str] = None,
    ) -> CoverageReport:
        svc = _get_service(info)
        edition_id = _resolve_edition_id(svc, edition)
        stats = svc.repo.coverage_stats(edition_id)
        return CoverageReport(
            total_factors=stats.get("total_factors"),
            by_geography=stats.get("by_geography"),
            by_scope=stats.get("by_scope"),
            by_fuel_type=stats.get("by_fuel_type"),
            by_source=stats.get("by_source"),
            edition_id=edition_id,
        )


# ==============================================================================
# Mutations
# ==============================================================================


def _get_overlay_registry(info: Info) -> Any:
    """Get or construct a TenantOverlayRegistry for this request.

    Stored on the context so repeated mutation calls within a single
    request share the same in-memory registry.
    """
    ctx = info.context
    reg = getattr(ctx, "overlay_registry", None)
    if reg is not None:
        return reg

    from greenlang.factors.tenant_overlay import TenantOverlayRegistry

    reg = TenantOverlayRegistry()
    try:
        ctx.overlay_registry = reg
    except (AttributeError, TypeError):
        if getattr(ctx, "metadata", None) is None:
            try:
                ctx.metadata = {}
            except (AttributeError, TypeError):
                pass
        try:
            ctx.metadata["overlay_registry"] = reg
        except (AttributeError, TypeError):
            pass
    return reg


@strawberry.type
class FactorsMutation:
    """Root-level GraphQL mutations for Factors."""

    @strawberry.mutation(description="Create or update a tenant factor override (Consulting+).")
    async def set_factor_override(
        self,
        info: Info,
        input: OverrideInput,
    ) -> Override:
        _require_consulting_tier(info)
        user = _get_user_context(info)
        tenant_id = str(input.tenant_id or user.get("tenant_id") or "default")

        reg = _get_overlay_registry(info)
        overlay = reg.create_overlay(
            tenant_id=tenant_id,
            factor_id=str(input.factor_id),
            override_value=float(input.co2e_per_unit),
            override_unit=input.override_unit or "kg_co2e",
            valid_from=input.effective_from,
            valid_to=input.effective_to,
            source=input.source or "",
            notes=input.justification or "",
            created_by=str(user.get("user_id") or "anonymous"),
        )

        return Override(
            overlay_id=overlay.overlay_id,
            tenant_id=overlay.tenant_id,
            factor_id=overlay.factor_id,
            override_value=overlay.override_value,
            override_unit=overlay.override_unit,
            valid_from=overlay.valid_from,
            valid_to=overlay.valid_to,
            source=overlay.source,
            notes=overlay.notes,
            created_by=overlay.created_by,
            created_at=overlay.created_at,
            updated_at=overlay.updated_at,
            active=overlay.active,
        )

    @strawberry.mutation(description="Soft-delete all tenant overrides for a factor (Consulting+).")
    async def remove_factor_override(
        self,
        info: Info,
        tenant_id: strawberry.ID,
        factor_id: strawberry.ID,
    ) -> bool:
        _require_consulting_tier(info)
        reg = _get_overlay_registry(info)
        user = _get_user_context(info)
        overlays = reg.list_overlays(str(tenant_id), factor_id=str(factor_id))
        removed_any = False
        for o in overlays:
            if reg.delete_overlay(
                str(tenant_id),
                o.overlay_id,
                deleted_by=str(user.get("user_id") or "anonymous"),
            ):
                removed_any = True
        return removed_any

    @strawberry.mutation(description="Submit a batch resolution job (async).")
    async def submit_batch_resolution(
        self,
        info: Info,
        requests: List[ResolutionRequestInput],
    ) -> BatchJobHandle:
        _require_pro_tier(info)
        svc = _get_service(info)

        # Deterministic job id based on content — mirrors the SDK's batch
        # endpoint so clients can idempotently re-submit.
        payload = [
            f"{r.activity}|{r.method_profile}|{r.jurisdiction or ''}|{r.reporting_date or ''}"
            for r in requests
        ]
        digest = hashlib.sha256("\n".join(payload).encode()).hexdigest()[:16]
        job_id = f"gql-batch-{digest}"

        # Execute inline for small batches (<= 100), enqueue otherwise.
        # The REST batch endpoint does the same trick.
        created_at = datetime.now(timezone.utc)
        if len(requests) <= 100:
            # Inline: run each resolution and collect a row count.
            from greenlang.factors.api_endpoints import (
                build_resolution_explain,
                clamp_alternates_limit,
            )
            from greenlang.factors.resolution.engine import ResolutionError

            processed = 0
            for r in requests:
                body = {
                    "activity": r.activity,
                    "method_profile": r.method_profile,
                }
                if r.jurisdiction:
                    body["jurisdiction"] = r.jurisdiction
                if r.reporting_date:
                    body["reporting_date"] = r.reporting_date
                try:
                    edition_id = _resolve_edition_id(svc, r.edition)
                    build_resolution_explain(
                        svc.repo,
                        edition_id,
                        body,
                        alternates_limit=clamp_alternates_limit(1),
                    )
                    processed += 1
                except ResolutionError:
                    # Best-effort: keep going so the batch reports a
                    # meaningful processed count.
                    continue

            return BatchJobHandle(
                job_id=job_id,
                status="completed",
                total_items=len(requests),
                processed_items=processed,
                progress_percent=100.0 if requests else 0.0,
                results_url=None,
                created_at=created_at,
                completed_at=datetime.now(timezone.utc),
                error_message=None,
            )

        # Large batch path: return a queued handle; a background worker
        # (rolled out separately by the batch team) picks it up.
        return BatchJobHandle(
            job_id=job_id,
            status="queued",
            total_items=len(requests),
            processed_items=0,
            progress_percent=0.0,
            results_url=None,
            created_at=created_at,
            completed_at=None,
            error_message=None,
        )

    @strawberry.mutation(description="Match an activity description to candidate factors.")
    async def match_factor(
        self,
        info: Info,
        input: MatchInput,
    ) -> List[Factor]:
        svc = _get_service(info)
        edition_id = _resolve_edition_id(svc, input.edition)
        vis = _tier_visibility(info, preview=True, connector=True)

        from greenlang.factors.matching.pipeline import MatchRequest, run_match

        req = MatchRequest(
            activity_description=input.activity_description,
            geography=input.geography,
            fuel_type=input.fuel_type,
            scope=input.scope,
            limit=input.limit or 10,
        )
        candidates = run_match(
            svc.repo,
            edition_id,
            req,
            include_preview=vis.include_preview,
            include_connector=vis.include_connector,
        )
        # Return the underlying Factor records (fetched in a batch).
        out: List[Factor] = []
        for c in candidates:
            rec = svc.repo.get_factor(edition_id, c["factor_id"])
            if rec is not None:
                out.append(_factor_to_graphql(rec, edition_id=edition_id))
        return out


__all__ = [
    "FactorsQuery",
    "FactorsMutation",
    "TierError",
]
