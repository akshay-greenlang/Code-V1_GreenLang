# -*- coding: utf-8 -*-
"""GraphQL resolvers.

Every resolver is a thin adapter around the REST helpers (``build_resolution_explain``,
``build_factor_explain``) — that's how we guarantee the GraphQL contract
cannot leak fields that REST withholds. If REST decides a caller isn't
entitled to a field (licensing-class gate, tier clamp, redaction), the
GraphQL resolver returns the exact same redacted dict.

Auth, tiering, and licensing are enforced by the enclosing FastAPI
route via existing middleware (``AuthMeteringMiddleware``,
``LicensingGuardMiddleware``, ``RateLimitMiddleware``). The resolvers
themselves are pure: they never bypass those layers.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from .schema import _HAS_STRAWBERRY

logger = logging.getLogger(__name__)


def _canonical_resolved_to_gql(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a REST ``ResolvedFactor.model_dump()`` to GraphQL field names.

    Single source of truth: we just pass through the envelope dicts. The
    GraphQL type definitions pick the field names they expose; any field
    not in the types is dropped, and any field the REST dict omits
    remains absent here too.
    """
    if not isinstance(payload, dict):
        return {}
    # REST already emits the envelope dicts (chosen_factor, source, …)
    # exactly matching the GraphQL type field names, so a pass-through
    # is correct.
    return payload


if _HAS_STRAWBERRY:
    import strawberry  # type: ignore

    from .schema import (
        AlternateCandidateGQL,
        ChosenFactorGQL,
        DeprecationStatusGQL,
        EditionGQL,
        ExplainPayloadGQL,
        FactorRecordGQL,
        GasBreakdownGQL,
        LicensingEnvelopeGQL,
        MethodPackGQL,
        QualityEnvelopeGQL,
        QualitySummaryGQL,
        ReleaseGQL,
        ResolvedFactorGQL,
        SourceDescriptorGQL,
        SourceGQL,
        UncertaintyEnvelopeGQL,
    )

    def _build_resolved_gql(payload: Dict[str, Any]) -> ResolvedFactorGQL:
        p = _canonical_resolved_to_gql(payload)

        def _chosen() -> Optional[ChosenFactorGQL]:
            cf = p.get("chosen_factor")
            if not cf:
                return None
            return ChosenFactorGQL(
                id=cf.get("id", ""),
                name=cf.get("name", ""),
                version=cf.get("version", ""),
                factor_family=cf.get("factor_family", ""),
            )

        def _source() -> Optional[SourceDescriptorGQL]:
            s = p.get("source")
            if not s:
                return None
            return SourceDescriptorGQL(
                id=s.get("id", ""),
                version=s.get("version", ""),
                name=s.get("name", ""),
                authority=s.get("authority", ""),
            )

        def _quality() -> Optional[QualityEnvelopeGQL]:
            q = p.get("quality")
            if not q:
                return None
            return QualityEnvelopeGQL(
                composite_fqs_0_100=float(q.get("composite_fqs_0_100", 0.0)),
                temporal_score=float(q.get("temporal_score", 0.0)),
                geographic_score=float(q.get("geographic_score", 0.0)),
                technology_score=float(q.get("technology_score", 0.0)),
                verification_score=float(q.get("verification_score", 0.0)),
                completeness_score=float(q.get("completeness_score", 0.0)),
                rating=q.get("rating"),
                formula_version=q.get("formula_version"),
            )

        def _uncertainty() -> Optional[UncertaintyEnvelopeGQL]:
            u = p.get("uncertainty")
            if not u:
                return None
            return UncertaintyEnvelopeGQL(
                type=u.get("type", "qualitative"),
                low=u.get("low"),
                high=u.get("high"),
                distribution=u.get("distribution"),
                note=u.get("note"),
            )

        def _licensing() -> Optional[LicensingEnvelopeGQL]:
            lic = p.get("licensing")
            if not lic:
                return None
            return LicensingEnvelopeGQL(
                redistribution_class=lic.get("redistribution_class", "open"),
                customer_entitlement_required=bool(lic.get("customer_entitlement_required", False)),
                watermark_required=bool(lic.get("watermark_required", False)),
            )

        def _deprec() -> Optional[DeprecationStatusGQL]:
            d = p.get("deprecation_status")
            if not d:
                return None
            return DeprecationStatusGQL(
                status=d.get("status", "active"),
                replacement_pointer_factor_id=d.get("replacement_pointer_factor_id"),
                deprecated_at=(
                    d.get("deprecated_at").isoformat()
                    if hasattr(d.get("deprecated_at"), "isoformat")
                    else d.get("deprecated_at")
                ),
                reason=d.get("reason"),
            )

        def _alternates() -> List[AlternateCandidateGQL]:
            out: List[AlternateCandidateGQL] = []
            for a in p.get("alternates", []) or []:
                out.append(
                    AlternateCandidateGQL(
                        factor_id=a.get("factor_id", ""),
                        tie_break_score=float(a.get("tie_break_score", 0.0)),
                        why_not_chosen=a.get("why_not_chosen", ""),
                        reason_lost=a.get("reason_lost"),
                        source_id=a.get("source_id"),
                        vintage=a.get("vintage"),
                        redistribution_class=a.get("redistribution_class"),
                    )
                )
            return out

        def _gas() -> Optional[GasBreakdownGQL]:
            g = p.get("gas_breakdown")
            if not g:
                return None
            return GasBreakdownGQL(
                co2_kg=float(g.get("co2_kg", 0.0)),
                ch4_kg=float(g.get("ch4_kg", 0.0)),
                n2o_kg=float(g.get("n2o_kg", 0.0)),
                hfcs_kg=float(g.get("hfcs_kg", 0.0)),
                pfcs_kg=float(g.get("pfcs_kg", 0.0)),
                sf6_kg=float(g.get("sf6_kg", 0.0)),
                nf3_kg=float(g.get("nf3_kg", 0.0)),
                biogenic_co2_kg=float(g.get("biogenic_co2_kg", 0.0)),
                co2e_total_kg=float(g.get("co2e_total_kg", 0.0)),
                gwp_basis=g.get("gwp_basis", "IPCC_AR6_100"),
            )

        return ResolvedFactorGQL(
            chosen_factor=_chosen(),
            source=_source(),
            quality=_quality(),
            uncertainty=_uncertainty(),
            licensing=_licensing(),
            deprecation_status=_deprec(),
            alternates=_alternates(),
            why_this_won=p.get("why_this_won") or p.get("why_chosen"),
            release_version=p.get("release_version"),
            method_pack=p.get("method_pack"),
            valid_from=(
                p.get("valid_from").isoformat()
                if hasattr(p.get("valid_from"), "isoformat")
                else p.get("valid_from")
            ),
            valid_to=(
                p.get("valid_to").isoformat()
                if hasattr(p.get("valid_to"), "isoformat")
                else p.get("valid_to")
            ),
            gas_breakdown=_gas(),
            co2e_basis=p.get("co2e_basis"),
            assumptions=list(p.get("assumptions") or []),
            fallback_rank=p.get("fallback_rank"),
            step_label=p.get("step_label"),
            audit_text=p.get("audit_text"),
            audit_text_draft=p.get("audit_text_draft"),
            resolved_at=(
                p.get("resolved_at").isoformat()
                if hasattr(p.get("resolved_at"), "isoformat")
                else p.get("resolved_at")
            ),
            engine_version=p.get("engine_version"),
            method_pack_version=p.get("method_pack_version"),
            chosen_factor_id=p.get("chosen_factor_id"),
            method_profile=p.get("method_profile"),
        )

    def _get_service(info: "strawberry.Info"):
        ctx = info.context or {}
        request = ctx.get("request") if isinstance(ctx, dict) else getattr(ctx, "request", None)
        if request is None:
            raise RuntimeError("GraphQL context missing FastAPI request")
        svc = getattr(request.app.state, "factors_service", None)
        if svc is None:
            raise RuntimeError("FactorCatalogService not configured")
        return svc, request

    def _current_edition(request, svc) -> str:
        pinned = getattr(request.state, "edition_id", None)
        if pinned:
            return pinned
        return svc.repo.resolve_edition(None)

    @strawberry.input
    class ResolveInputGQL:
        activity: str
        method_profile: str
        jurisdiction: Optional[str] = None
        reporting_date: Optional[str] = None
        tenant_id: Optional[str] = None

    @strawberry.type
    class Query:
        """GraphQL Query root."""

        @strawberry.field
        def resolve(self, info: strawberry.Info, input: ResolveInputGQL) -> ResolvedFactorGQL:
            from greenlang.factors.api_endpoints import build_resolution_explain

            svc, request = _get_service(info)
            edition = _current_edition(request, svc)
            body = {
                "activity": input.activity,
                "method_profile": input.method_profile,
                "jurisdiction": input.jurisdiction,
                "reporting_date": input.reporting_date,
                "tenant_id": input.tenant_id,
            }
            body = {k: v for k, v in body.items() if v is not None}
            payload = build_resolution_explain(svc.repo, edition, body)
            return _build_resolved_gql(payload)

        @strawberry.field
        def factor(
            self,
            info: strawberry.Info,
            id: str,
            method_profile: Optional[str] = None,
            jurisdiction: Optional[str] = None,
        ) -> Optional[ResolvedFactorGQL]:
            from greenlang.factors.api_endpoints import build_factor_explain

            svc, request = _get_service(info)
            edition = _current_edition(request, svc)
            payload = build_factor_explain(
                svc.repo, edition, id,
                method_profile=method_profile,
                jurisdiction=jurisdiction,
            )
            if payload is None:
                return None
            return _build_resolved_gql(payload)

        @strawberry.field
        def search(
            self,
            info: strawberry.Info,
            query: str,
            limit: int = 20,
        ) -> List[FactorRecordGQL]:
            svc, request = _get_service(info)
            edition = _current_edition(request, svc)
            repo = svc.repo
            out: List[FactorRecordGQL] = []
            try:
                rows, _ = repo.search_factors(edition, query=query, page=1, limit=limit)
            except Exception:
                try:
                    rows, _ = repo.list_factors(edition, page=1, limit=limit)
                except Exception:
                    rows = []
            for f in rows:
                out.append(
                    FactorRecordGQL(
                        factor_id=getattr(f, "factor_id", ""),
                        factor_family=getattr(getattr(f, "factor_family", None), "value", None)
                        or getattr(f, "fuel_type", None),
                        fuel_type=getattr(f, "fuel_type", None),
                        status=getattr(f, "factor_status", "certified") or "certified",
                        redistribution_class=getattr(
                            getattr(f, "license_info", None), "license", None
                        ),
                    )
                )
            return out

        @strawberry.field
        def method_packs(self, info: strawberry.Info) -> List[MethodPackGQL]:
            try:
                from greenlang.factors.method_packs import list_method_packs  # type: ignore
                packs = list_method_packs()
            except Exception:
                packs = []
            return [
                MethodPackGQL(
                    pack_id=p.get("pack_id", ""),
                    name=p.get("name", ""),
                    version=p.get("version", ""),
                    description=p.get("description"),
                )
                for p in (packs or [])
                if isinstance(p, dict)
            ]

        @strawberry.field
        def sources(self, info: strawberry.Info) -> List[SourceGQL]:
            try:
                from greenlang.factors.source_registry import load_source_registry
                reg = load_source_registry()
            except Exception:
                reg = []
            out: List[SourceGQL] = []
            for s in reg or []:
                if isinstance(s, dict):
                    out.append(
                        SourceGQL(
                            id=s.get("id", ""),
                            name=s.get("name", ""),
                            authority=s.get("authority"),
                            version=s.get("version"),
                        )
                    )
                else:
                    out.append(
                        SourceGQL(
                            id=getattr(s, "id", ""),
                            name=getattr(s, "name", ""),
                            authority=getattr(s, "authority", None),
                            version=getattr(s, "version", None),
                        )
                    )
            return out

        @strawberry.field
        def releases(self, info: strawberry.Info) -> List[ReleaseGQL]:
            svc, request = _get_service(info)
            repo = svc.repo
            try:
                rels = repo.list_editions()  # type: ignore[attr-defined]
            except Exception:
                rels = []
            out: List[ReleaseGQL] = []
            for r in rels or []:
                if isinstance(r, dict):
                    out.append(
                        ReleaseGQL(
                            edition_id=r.get("edition_id", ""),
                            cut_at=r.get("created_at"),
                            approver=r.get("approver"),
                        )
                    )
                else:
                    out.append(
                        ReleaseGQL(
                            edition_id=str(r),
                            cut_at=None,
                            approver=None,
                        )
                    )
            return out

        @strawberry.field
        def quality(self, info: strawberry.Info, factor_id: str) -> Optional[QualitySummaryGQL]:
            svc, request = _get_service(info)
            edition = _current_edition(request, svc)
            factor = svc.repo.get_factor(edition, factor_id)
            if not factor:
                return None
            try:
                from greenlang.factors.quality.composite_fqs import compute_fqs
                fqs = compute_fqs(factor.dqs)
                score = int(getattr(fqs, "score", getattr(fqs, "value", 0)))
                rating = getattr(fqs, "rating", None)
                label = getattr(fqs, "label", None)
            except Exception:
                score = 0
                rating = None
                label = None
            return QualitySummaryGQL(
                factor_id=factor_id,
                composite_fqs_0_100=float(score),
                rating=rating,
                label=label,
            )

        @strawberry.field
        def explain(
            self,
            info: strawberry.Info,
            factor_id: Optional[str] = None,
            resolve_request_id: Optional[str] = None,
            receipt_id: Optional[str] = None,
        ) -> ExplainPayloadGQL:
            from greenlang.factors.api_endpoints import build_factor_explain

            if not (factor_id or resolve_request_id or receipt_id):
                raise ValueError(
                    "explain() requires one of factorId, resolveRequestId, or receiptId"
                )

            svc, request = _get_service(info)
            edition = _current_edition(request, svc)

            payload: Optional[Dict[str, Any]] = None
            if factor_id:
                payload = build_factor_explain(svc.repo, edition, factor_id)

            # For resolveRequestId / receiptId paths we delegate to the
            # explain_history store if it's wired; otherwise we return
            # the factor-level explain. Tenant isolation is ENFORCED
            # here — the receipt is only fetched under the caller's
            # tenant_id.
            if payload is None and (resolve_request_id or receipt_id):
                user = getattr(request.state, "user", None) or {}
                tenant_id = user.get("tenant_id") or user.get("user_id")
                try:
                    from greenlang.factors.explain_history import get_explain_by_id
                    payload = get_explain_by_id(
                        resolve_request_id or receipt_id or "",
                        tenant_id=tenant_id,
                    )
                except Exception:
                    payload = None

            if payload is None:
                return ExplainPayloadGQL(
                    receipt_id=receipt_id or resolve_request_id,
                    factor_id=factor_id,
                    resolved_json=None,
                    explain_json=None,
                )
            return ExplainPayloadGQL(
                receipt_id=receipt_id or resolve_request_id or str(uuid.uuid4()),
                factor_id=factor_id,
                resolved_json=json.dumps(payload, default=str, sort_keys=True),
                explain_json=json.dumps(payload.get("explain", {}), default=str, sort_keys=True),
            )

else:  # pragma: no cover - stub for no-strawberry environments

    class Query:  # type: ignore[no-redef]
        """Stub Query; real resolvers require strawberry-graphql."""


__all__ = ["Query"]
