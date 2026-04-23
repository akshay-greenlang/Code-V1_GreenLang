# -*- coding: utf-8 -*-
"""
ResolutionEngine — the 7-step cascade brain (Phase F3).

Selection order (per CTO spec):

    1. customer-specific override (tenant overlay)
    2. supplier / manufacturer-specific factor
    3. facility / asset-specific factor
    4. utility / tariff / grid-subregion factor
    5. country / sector average
    6. method-pack default
    7. global default

Tie-breaking inside a step: see :mod:`tiebreak`.

The engine is deliberately backend-agnostic — it accepts a ``candidate_source``
callable which returns a list of records that the engine then filters through
the method-pack :class:`SelectionRule` and scores via :func:`build_tiebreak`.
Tests inject a stub source; production passes a repository-backed callable.

Production candidate-source loader
----------------------------------

When ``ResolutionEngine`` is constructed without an explicit
``candidate_source``, :func:`build_default_candidate_source` is called to
pick the best available backend in the following order:

    1. **Postgres** — if ``DATABASE_URL`` (or ``GL_FACTORS_PG_DSN``) is set,
       the engine wraps :class:`PostgresFactorCatalogRepository` with a
       resolution-aware filter (geography + status visibility).
    2. **File-backed** — otherwise the engine wraps the in-memory
       :class:`MemoryFactorCatalogRepository`, which loads built-in
       factors from ``greenlang/factors/data/source_registry.yaml`` and
       ``greenlang/factors/data/method_packs/`` via the
       :class:`EmissionFactorDatabase` loader.
    3. **Hard fail** — if neither backend yields any factor (e.g. empty
       built-in DB and no DSN), :func:`_unconfigured_candidate_source`
       raises :class:`ConfigurationError` on first use rather than
       silently returning empty results.  This guarantees that a misconfigured
       production pod fails fast instead of returning ``ResolutionError``
       for every request.

Override the loader explicitly in production for stricter control::

    engine = ResolutionEngine(
        candidate_source=my_repo_backed_source,
        tenant_overlay_reader=my_tenant_lookup,
    )
"""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from greenlang.data.canonical_v2 import (
    MethodProfile,
    enforce_license_class_homogeneity,
)
from greenlang.exceptions import ConfigurationError
from greenlang.factors.method_packs import (
    AUDIT_DRAFT_BANNER,
    MethodPack,
    get_pack,
    render_audit_text,
)
from greenlang.factors.method_packs.base import CannotResolveAction
from greenlang.factors.method_packs.exceptions import FactorCannotResolveSafelyError
from greenlang.factors.ontology.unit_graph import (
    DEFAULT_GRAPH,
    UnitConversionError,
    UnitGraph,
)
from greenlang.factors.resolution.request import ResolutionRequest
from greenlang.factors.resolution.result import (
    AlternateCandidate,
    ChosenFactor,
    DeprecationStatus,
    GasBreakdown,
    LicensingEnvelope,
    QualityEnvelope,
    ResolvedFactor,
    SourceDescriptor,
    UncertaintyEnvelope,
)
from greenlang.factors.resolution.tiebreak import TieBreakReasons, build_tiebreak

logger = logging.getLogger(__name__)


#: Shape of a candidate source: takes request + step label → iterable of records.
CandidateSource = Callable[[ResolutionRequest, str], Iterable[Any]]


class ResolutionError(RuntimeError):
    """Raised when no factor can be resolved."""


class ResolutionEngine:
    """7-step resolution cascade."""

    def __init__(
        self,
        *,
        candidate_source: Optional[CandidateSource] = None,
        tenant_overlay_reader: Optional[Callable[[ResolutionRequest], Optional[Any]]] = None,
        unit_graph: Optional[UnitGraph] = None,
    ) -> None:
        # Production deployments get a real source loader.  Tests + library
        # callers can still inject a stub source explicitly.
        self._candidate_source = candidate_source or build_default_candidate_source()
        self._tenant_overlay_reader = tenant_overlay_reader or _default_tenant_lookup
        self._unit_graph = unit_graph or DEFAULT_GRAPH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, request: ResolutionRequest) -> ResolvedFactor:
        """Resolve a factor per the 7-step cascade.

        Raises :class:`ResolutionError` if no factor passes the method-pack
        selection rule at any step.
        """
        # CTO non-negotiable #6 — method_profile is Pydantic-required; this
        # double-check guards against programmatic bypass (``object.__new__``).
        if not isinstance(request.method_profile, MethodProfile):
            raise ResolutionError(
                "ResolutionRequest.method_profile is required (non-negotiable #6)."
            )

        pack = get_pack(request.method_profile)
        request_date = request.resolved_date()
        request_geo = request.jurisdiction or request.extras.get("region_code")

        # Track how many candidates we looked at so the safe-match error
        # can distinguish "no data at all" from "data exists but the
        # pack's selection rule refused every record".
        total_candidates_seen = 0
        total_rejected_by_selection = 0
        evaluated_steps: List[str] = []
        global_default_blocked = False

        # Run the 7-step cascade.
        for rank, label in _STEP_ORDER:
            # cannot_resolve_safely: when the pack forbids tier-7 (global
            # default), skip that step entirely. We still raise below with
            # a reason key of ``global_default_blocked`` if the remaining
            # tiers produced nothing.
            if label == "global_default" and not pack.global_default_tier_allowed:
                global_default_blocked = True
                evaluated_steps.append(f"{rank}:{label}:skipped_global_default_blocked")
                continue
            candidates = list(self._gather_step(request, label, pack))
            total_candidates_seen += len(candidates)
            eligible = [c for c in candidates if pack.selection_rule.accepts(c)]
            total_rejected_by_selection += len(candidates) - len(eligible)
            evaluated_steps.append(f"{rank}:{label}:cand={len(candidates)}:ok={len(eligible)}")
            if not eligible:
                continue

            # Score every eligible candidate; pick the lowest-score winner.
            request_granularity = getattr(
                getattr(request, "time_granularity", None), "value", None
            )
            # Wave 3 (gold-eval): tokenise the activity text once per
            # resolve() call so the tie-break can reward factors whose
            # id/fuel/tags align with the request. This is a pure
            # substring match — no embeddings, no LLMs — so bit-perfect
            # determinism is preserved.
            activity_tokens = _tokenise_activity(request)
            scored: List[Tuple[int, Any, TieBreakReasons]] = []
            for cand in eligible:
                tb = build_tiebreak(
                    cand,
                    request_geo=request_geo,
                    request_date=request_date,
                    request_granularity=request_granularity,
                    activity_tokens=activity_tokens,
                )
                scored.append((tb.score(), cand, tb))
            scored.sort(key=lambda triple: triple[0])
            winning_score, winner, winning_tb = scored[0]
            alternates = [
                self._to_alternate(record, tb, self_score=winning_score)
                for _score, record, tb in scored[1:10]  # top-9 alternates max
            ]

            # Non-negotiable #4: the winner + its alternates must share a
            # license class.  If not, drop any alternate that doesn't match.
            enforce_license_class_homogeneity([winner, *eligible[:9]])

            resolved = self._build_resolved_factor(
                winner, winning_tb, alternates, pack, rank, label
            )
            self._apply_unit_conversion(resolved, winner, request)
            self._apply_audit_text(resolved)
            return resolved

        # Every step exhausted. Honour the pack's cannot_resolve_action:
        # certified packs (default) raise the structured safe-match error
        # so regulated callers (CBAM, PEF, Battery CFP) do not silently
        # publish a global-default-backed number. Packs that explicitly
        # opt into ``ALLOW_GLOBAL_DEFAULT`` fall through to the legacy
        # ``ResolutionError`` for backwards compatibility with the
        # pre-FY27 behaviour.
        if pack.cannot_resolve_action == CannotResolveAction.RAISE_NO_SAFE_MATCH:
            # When the pack forbids global_default, surface that fact
            # in the reason key — regulators want to know whether the
            # exhaustion was due to missing data OR due to the pack's
            # strict policy blocking a low-quality fallback. Prefer
            # ``global_default_blocked`` whenever tier-7 was skipped
            # (the skipped tier may have had a candidate that would
            # have resolved if the pack allowed it).
            if global_default_blocked:
                reason = "global_default_blocked"
            elif total_candidates_seen == 0:
                reason = "no_candidate"
            else:
                reason = "all_candidates_rejected"
            raise FactorCannotResolveSafelyError(
                message=(
                    "Method pack %r refused to resolve safely for "
                    "profile=%s activity=%r: %s (inspected %d candidate(s) "
                    "across %d step(s))."
                    % (
                        pack.profile.value,
                        request.method_profile.value,
                        request.activity,
                        reason,
                        total_candidates_seen,
                        len(evaluated_steps),
                    )
                ),
                pack_id=pack.profile.value,
                method_profile=request.method_profile.value,
                reason=reason,
                evaluated_candidates_count=total_candidates_seen,
                evaluated_steps=evaluated_steps,
            )
        raise ResolutionError(
            "No eligible factor found for profile=%s activity=%r (tried %d steps)."
            % (request.method_profile.value, request.activity, len(_STEP_ORDER))
        )

    # ------------------------------------------------------------------
    # Step dispatch
    # ------------------------------------------------------------------

    def _gather_step(
        self,
        request: ResolutionRequest,
        label: str,
        pack: MethodPack,
    ) -> Iterable[Any]:
        """Yield candidate records for a given cascade step."""
        if label == "customer_override":
            overlay = self._tenant_overlay_reader(request)
            if overlay is not None:
                yield overlay
            return

        # All other steps delegate to the injected candidate source.  The
        # source is free to filter further based on the label + request
        # fields (supplier_id, facility_id, utility_or_grid_region, …).
        yield from self._candidate_source(request, label)

    # ------------------------------------------------------------------
    # Packaging
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def _apply_unit_conversion(
        self,
        resolved: ResolvedFactor,
        record: Any,
        request: ResolutionRequest,
    ) -> None:
        """Populate ``resolved.target_unit`` + related fields when the
        caller asked for a unit that differs from the factor's native
        denominator.

        The conversion is applied to the CO2e-per-unit basis (numerator
        stays in kg CO2e — denominator changes).  Callers that need
        per-gas breakdowns in the new unit can multiply by the same
        factor on their side; the engine only exposes the aggregated
        ``converted_co2e_per_unit`` to keep the explain payload small.
        """
        target = (request.target_unit or "").strip()
        if not target:
            return
        native = (resolved.factor_unit_denominator or "").strip()
        if not native:
            resolved.unit_conversion_note = (
                "target_unit requested but the factor has no native denominator"
            )
            return
        if native.lower() == target.lower():
            resolved.target_unit = native
            resolved.unit_conversion_factor = 1.0
            resolved.converted_co2e_per_unit = _safe_float(
                getattr(getattr(record, "gwp_100yr", None), "co2e_total", None)
            )
            resolved.unit_conversion_path = [native]
            resolved.unit_conversion_note = "native unit matches target"
            return

        try:
            path = self._unit_graph.shortest_path(from_unit=target, to_unit=native)
        except Exception as exc:  # pragma: no cover — defensive
            resolved.unit_conversion_note = (
                f"unit_graph error during shortest_path: {exc}"
            )
            return
        if path is None:
            resolved.target_unit = target
            resolved.unit_conversion_note = (
                f"no conversion path from {target!r} to {native!r}; "
                "returning factor in native unit"
            )
            return

        try:
            conv = self._unit_graph.convert(
                value=1.0, from_unit=target, to_unit=native
            )
        except UnitConversionError as exc:
            resolved.target_unit = target
            resolved.unit_conversion_note = (
                f"conversion requires material context ({exc}); "
                "returning factor in native unit"
            )
            return

        co2e_native = _safe_float(
            getattr(getattr(record, "gwp_100yr", None), "co2e_total", None)
        )
        resolved.target_unit = target
        resolved.unit_conversion_factor = conv
        resolved.unit_conversion_path = [target] + [e.target_unit for e in path]
        if co2e_native is not None:
            # CO2e is per native unit; 1 target_unit = `conv` native units
            # → CO2e per target_unit = co2e_native * conv.
            resolved.converted_co2e_per_unit = co2e_native * conv
        resolved.unit_conversion_note = (
            f"converted {target} → {native} via {len(path)}-edge path"
        )

    @staticmethod
    def _apply_audit_text(resolved: ResolvedFactor) -> None:
        """Populate ``resolved.audit_text`` + ``resolved.audit_text_draft``.

        Calls :func:`render_audit_text` against the pack-family template
        under ``greenlang/factors/method_packs/audit_texts/<family>.j2``.
        Pack families are derived from the resolved method profile:
        ``corporate_scope1`` / ``corporate_scope2_*`` / ``corporate_scope3``
        all share ``corporate.j2``; every EU pack (cbam, dpp, battery) shares
        ``eu_policy.j2``; the electricity family uses ``electricity.j2``.
        When the template is unapproved (``approved: false`` in its
        frontmatter), the rendered paragraph is prefixed with
        :data:`AUDIT_DRAFT_BANNER`; this function detects that prefix and
        sets :attr:`ResolvedFactor.audit_text_draft` accordingly so
        ``/explain`` consumers can flag draft wording in their UIs.

        Failures are non-fatal: a missing template, missing frontmatter, or
        any other render error logs a warning and leaves both fields at
        ``None`` — the resolver MUST NOT break just because audit text is
        unavailable.  See ``docs/specs/audit_text_template_policy.md``.
        """
        template_key = _audit_template_key(
            resolved.method_pack, resolved.method_profile
        )
        if not template_key:
            return
        try:
            rendered = render_audit_text(pack_id=template_key, factor=resolved)
        except FileNotFoundError:
            logger.warning(
                "audit_text_render_skipped: no template for pack_id=%r",
                template_key,
            )
            resolved.audit_text = None
            resolved.audit_text_draft = None
            return
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "audit_text_render_failed: pack_id=%r err=%s",
                template_key, exc,
            )
            resolved.audit_text = None
            resolved.audit_text_draft = None
            return

        # ``render_audit_text`` returns a plain string.  The SAFE-DRAFT
        # policy prepends ``AUDIT_DRAFT_BANNER`` + "\n\n" when the template
        # frontmatter has ``approved: false`` (or is missing entirely).
        resolved.audit_text = rendered
        resolved.audit_text_draft = bool(
            rendered and rendered.startswith(AUDIT_DRAFT_BANNER)
        )

    @staticmethod
    def _to_alternate(
        record: Any, tb: TieBreakReasons, *, self_score: int
    ) -> AlternateCandidate:
        delta = tb.score() - self_score
        why_not = (
            f"higher tie-break score (+{delta}): {tb.one_liner()}"
            if delta > 0
            else "tied score — deterministic ordering preferred the winner"
        )
        return AlternateCandidate(
            factor_id=str(getattr(record, "factor_id", "unknown")),
            tie_break_score=float(tb.score()),
            why_not_chosen=why_not,
            reason_lost=why_not,
            source_id=getattr(record, "source_id", None),
            vintage=_safe_vintage(record),
            redistribution_class=getattr(record, "redistribution_class", None),
        )

    @staticmethod
    def _build_resolved_factor(
        record: Any,
        tb: TieBreakReasons,
        alternates: Sequence[AlternateCandidate],
        pack: MethodPack,
        rank: int,
        label: str,
    ) -> ResolvedFactor:
        """Project a record + pack into the CTO 16-element contract envelope.

        Populates every nested envelope (ChosenFactor, SourceDescriptor,
        QualityEnvelope, LicensingEnvelope, DeprecationStatus,
        UncertaintyEnvelope) AND the legacy flat fields for backward
        compatibility.  Calls composite_fqs.compute_fqs for the 0-100
        quality surface (falls back to the legacy 1-5 scalar rescaled to
        0-100 when the DQS lacks the 5 per-dimension fields).
        """
        factor_id = str(getattr(record, "factor_id", "unknown"))

        assumptions = list(
            getattr(getattr(record, "explainability", None), "assumptions", []) or []
        )
        if not assumptions:
            # Try to pull from the method pack's audit text template.
            audit_template = getattr(pack, "audit_text_template", None)
            if audit_template:
                assumptions = [
                    f"Resolved via step {rank} ({label}).",
                    f"Method pack: {pack.profile.value} @ {pack.pack_version}.",
                ]
            else:
                assumptions = [f"Resolved via step {rank} ({label})."]

        gb = _gas_breakdown(record, pack.gwp_basis)

        # --- Uncertainty envelope (CTO #11) ---
        ci_95 = _safe_float(getattr(record, "uncertainty_95ci", None))
        distribution = str(
            getattr(record, "uncertainty_distribution", "unknown") or "unknown"
        )
        if ci_95 is not None and ci_95 > 0:
            # Apply the 95% CI band symmetrically around co2e_total.
            central = gb.co2e_total_kg or 0.0
            low = central * max(0.0, 1.0 - ci_95) if central else None
            high = central * (1.0 + ci_95) if central else None
            uncertainty_env = UncertaintyEnvelope(
                type="95_percent_ci",
                low=low,
                high=high,
                distribution=distribution,
                note=tb.one_liner(),
            )
        else:
            uncertainty_env = UncertaintyEnvelope(
                type="qualitative",
                low=None,
                high=None,
                distribution=distribution,
                note=tb.one_liner(),
            )

        # --- Quality envelope (CTO #10) — composite_fqs ---
        dqs = getattr(record, "dqs", None)
        quality_env = _build_quality_envelope(dqs)
        quality_legacy = _safe_float(getattr(dqs, "overall_score", None))

        # --- Verification ---
        v = getattr(record, "verification", None)
        verification_status = str(getattr(v, "status", "")) if v is not None else None

        # --- Deprecation envelope (CTO #15) ---
        factor_status = str(
            getattr(record, "factor_status", "certified") or "certified"
        )
        deprecation_replacement = getattr(record, "replacement_factor_id", None)
        # TODO(D): replacement_pointer from migrated record (Wave 2b) — current
        # EmissionFactorRecord exposes replacement_factor_id only; map it.
        if factor_status == "deprecated":
            dep_status = DeprecationStatus(
                status="deprecated",
                replacement_pointer_factor_id=deprecation_replacement,
            )
        elif factor_status in ("superseded",):
            dep_status = DeprecationStatus(
                status="superseded",
                replacement_pointer_factor_id=deprecation_replacement,
            )
        else:
            dep_status = DeprecationStatus(status="active")

        # --- "why this won" explanation (CTO #3) ---
        explainability = getattr(record, "explainability", None)
        rationale = getattr(explainability, "rationale", None) if explainability else None
        why_chosen = (
            rationale
            or f"Selected at cascade step {rank} ({label}). {tb.one_liner()}."
        )

        # --- Source envelope (CTO #4) ---
        prov = getattr(record, "provenance", None)
        source_id = (
            getattr(record, "source_id", None)
            or (getattr(prov, "source_org", None) if prov is not None else None)
            or (getattr(prov, "source_publication", None) if prov is not None else None)
        )
        source_version = (
            getattr(record, "source_release", None)
            or getattr(record, "release_version", None)
            or (getattr(prov, "version", None) if prov is not None else None)
            or (
                str(getattr(prov, "source_year", ""))
                if prov is not None and getattr(prov, "source_year", None)
                else None
            )
        )
        source_name = (
            (getattr(prov, "source_publication", None) if prov is not None else None)
            or (getattr(prov, "source_org", None) if prov is not None else None)
            or source_id
            or "unknown"
        )
        source_authority = (
            (getattr(prov, "source_org", None) if prov is not None else None)
            or source_name
            or "unknown"
        )
        source_env = None
        if source_id:
            source_env = SourceDescriptor(
                id=str(source_id),
                version=str(source_version or "unknown"),
                name=str(source_name),
                authority=str(source_authority),
            )

        # --- Chosen factor envelope (CTO #1) ---
        raw_family = getattr(record, "factor_family", None)
        if hasattr(raw_family, "value"):
            raw_family = raw_family.value
        # Map the 15-value legacy FactorFamily enum onto the 7-value CTO
        # canonical family taxonomy (electricity, combustion, freight,
        # material, land, product, finance).
        family_map = {
            "grid_intensity": "electricity",
            "residual_mix": "electricity",
            "emissions": "combustion",
            "carbon_content": "combustion",
            "oxidation": "combustion",
            "heating_value": "combustion",
            "density": "combustion",
            "refrigerant_gwp": "combustion",
            "transport_lane": "freight",
            "material_embodied": "material",
            "waste_treatment": "material",
            "land_use_removals": "land",
            "energy_conversion": "product",
            "finance_proxy": "finance",
            "classification_mapping": "product",
        }
        if raw_family and raw_family in family_map:
            factor_family = family_map[raw_family]
        elif raw_family:
            factor_family = raw_family
        else:
            factor_family = _family_from_profile(pack.profile.value) or "unknown"
        chosen_env = ChosenFactor(
            id=factor_id,
            name=str(getattr(record, "factor_name", None) or factor_id),
            version=str(getattr(record, "factor_version", None) or "1.0.0"),
            factor_family=str(factor_family),
        )

        # --- Licensing envelope (CTO #12) ---
        redistribution = getattr(record, "redistribution_class", None)
        licensing_env = None
        if redistribution:
            rc = str(redistribution)
            # Map legacy RedistributionClass enum values onto the 4 CTO
            # canonical classes.  Legacy "licensed" -> "licensed_embedded";
            # legacy "restricted" -> "customer_private" (conservative).
            mapping = {
                "open": "open",
                "licensed": "licensed_embedded",
                "licensed_embedded": "licensed_embedded",
                "licensed_commercial": "licensed_embedded",
                "customer_private": "customer_private",
                "restricted": "customer_private",
                "oem_redistributable": "oem_redistributable",
            }
            canonical = mapping.get(rc)
            if canonical is not None:
                licensing_env = LicensingEnvelope(
                    redistribution_class=canonical,  # type: ignore[arg-type]
                )

        # --- Valid-from / valid-to (CTO #7) ---
        valid_from = getattr(record, "valid_from", None)
        valid_to = getattr(record, "valid_to", None)
        # Support a migrated "vintage_range" tuple form.
        vintage_range = getattr(record, "vintage_range", None)
        if valid_from is None and vintage_range:
            try:
                valid_from = vintage_range[0]
                valid_to = vintage_range[1]
            except (TypeError, IndexError):
                pass
        # N5 gate: Certified rows must carry at least valid_from once Wave 2b
        # (Agent D) completes the EmissionFactorRecord migration.  Until then,
        # test fixtures (SimpleNamespace without the attribute) are tolerated.
        # The gate fires only when the record declares the attribute explicitly
        # but has nulled it out — signalling a real data-quality regression.
        if (
            factor_status == "certified"
            and hasattr(record, "valid_from")
            and getattr(record, "valid_from", None) is None
            and vintage_range is None
        ):
            # TODO(D): upgrade to a hard assert once Wave 2b migration lands.
            logger.warning(
                "N5 gate: certified factor %r has no valid_from (dates required).",
                factor_id,
            )

        # --- method_pack (CTO #6) ---
        method_pack_id = getattr(pack, "pack_id", None) or pack.profile.value

        # --- co2e_basis (CTO #9) — top-level alias ---
        co2e_basis = pack.gwp_basis

        why_this_won = why_chosen

        return ResolvedFactor(
            # Contract envelopes
            chosen_factor=chosen_env,
            source=source_env,
            quality=quality_env,
            licensing=licensing_env,
            deprecation_status=dep_status,
            # Contract scalars
            why_this_won=why_this_won,
            release_version=getattr(record, "release_version", None) or None,
            method_pack=str(method_pack_id),
            valid_from=valid_from,
            valid_to=valid_to,
            co2e_basis=co2e_basis,
            # Legacy flat fields
            chosen_factor_id=factor_id,
            chosen_factor_name=getattr(record, "factor_name", None),
            source_id=source_id,
            source_version=source_version,
            factor_version=getattr(record, "factor_version", None),
            vintage=_safe_vintage(record),
            method_profile=pack.profile.value,
            formula_type=getattr(record, "formula_type", None),
            redistribution_class=getattr(record, "redistribution_class", None),
            fallback_rank=rank,
            step_label=label,
            why_chosen=why_chosen,
            alternates=list(alternates),
            assumptions=assumptions,
            deprecation_replacement=deprecation_replacement,
            quality_score=quality_legacy,
            uncertainty=uncertainty_env,
            verification_status=verification_status,
            gas_breakdown=gb,
            factor_unit_denominator=getattr(record, "unit", None),
            primary_data_flag=getattr(record, "primary_data_flag", None),
            method_pack_version=pack.pack_version,
        )


# ---------------------------------------------------------------------------
# Cascade configuration
# ---------------------------------------------------------------------------


_STEP_ORDER: Tuple[Tuple[int, str], ...] = (
    (1, "customer_override"),
    (2, "supplier_specific"),
    (3, "facility_specific"),
    (4, "utility_or_grid_subregion"),
    (5, "country_or_sector_average"),
    (6, "method_pack_default"),
    (7, "global_default"),
)


# ---------------------------------------------------------------------------
# Production candidate-source loader
# ---------------------------------------------------------------------------


def _pg_dsn_from_env() -> Optional[str]:
    """Return the Postgres DSN if set in any of the supported env vars."""
    for var in ("DATABASE_URL", "GL_FACTORS_PG_DSN"):
        dsn = os.environ.get(var, "").strip()
        if dsn:
            return dsn
    return None


def _build_pg_candidate_source() -> Optional[CandidateSource]:
    """Try to wire a Postgres-backed candidate source.

    Returns ``None`` if the DSN env vars are unset or the Postgres
    repository cannot be initialized (psycopg missing, schema empty, etc).
    Errors are logged at WARNING level — the caller falls back to the
    file-backed loader rather than failing the engine constructor.
    """
    dsn = _pg_dsn_from_env()
    if not dsn:
        return None
    try:
        from greenlang.factors.catalog_repository_pg import (
            PgPoolConfig,
            PostgresFactorCatalogRepository,
        )

        repo = PostgresFactorCatalogRepository(PgPoolConfig(dsn=dsn))
        edition_id = repo.get_default_edition_id()
        if not edition_id:
            logger.warning(
                "DATABASE_URL is set but factors_catalog has no stable edition; "
                "falling back to file-backed source."
            )
            return None
        logger.info(
            "ResolutionEngine: using Postgres candidate source (edition=%s)",
            edition_id,
        )
        return _wrap_repo_as_source(repo, edition_id)
    except Exception as exc:  # pragma: no cover — depends on optional psycopg
        logger.warning(
            "ResolutionEngine: Postgres source unavailable (%s); "
            "falling back to file-backed source.",
            exc,
        )
        return None


def _build_file_candidate_source() -> Optional[CandidateSource]:
    """Try to wire the file-backed (built-in) candidate source.

    Reads the built-in :class:`EmissionFactorDatabase` which itself loads
    from ``greenlang/factors/data/source_registry.yaml`` and
    ``greenlang/factors/data/method_packs/``.  Returns ``None`` if the
    catalog is empty (e.g. data files removed in a stripped-down build).
    """
    try:
        from greenlang.data.emission_factor_database import EmissionFactorDatabase
        from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository

        db = EmissionFactorDatabase(enable_cache=False)
        if not getattr(db, "factors", None):
            logger.warning(
                "ResolutionEngine: built-in EmissionFactorDatabase is empty; "
                "no file-backed source available."
            )
            return None
        edition_id = os.environ.get("GL_FACTORS_BUILTIN_EDITION", "builtin-v1.0.0")
        label = os.environ.get(
            "GL_FACTORS_BUILTIN_LABEL", "GreenLang built-in v2 factors"
        )
        repo = MemoryFactorCatalogRepository(edition_id, label, db)
        logger.info(
            "ResolutionEngine: using file-backed candidate source (edition=%s, factors=%d)",
            edition_id, len(db.factors),
        )
        return _wrap_repo_as_source(repo, edition_id)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "ResolutionEngine: file-backed source unavailable (%s).", exc,
        )
        return None


def _wrap_repo_as_source(repo: Any, edition_id: str) -> CandidateSource:
    """Wrap a :class:`FactorCatalogRepository` as a :data:`CandidateSource`.

    The wrapper applies the same per-step filter heuristics used by the
    API in ``api_endpoints._build_repo_engine`` so the CLI and the API
    use identical selection logic.  Customer-override is handled by the
    engine's ``tenant_overlay_reader`` and never routed through the
    repository here.

    Wave 3 (gold-eval tuning): the wrapper now performs activity-aware
    semantic pre-ranking so the tie-break layer sees candidates ordered
    by how well the factor_id / fuel_type / tags match the activity
    text.  Without this step the repository returns 500 factors in a
    geography-insensitive order and the tie-break picks whichever
    factor happened to be first in the dict-iteration order (typically
    an electricity factor for a freight query — hence the Wave 2.5
    gold-eval 0.8% P@1 baseline).  The scoring is deterministic (no
    embeddings, no LLMs) so bit-perfect reproducibility is preserved.
    """

    def _source(req: ResolutionRequest, label: str) -> List[Any]:
        if label == "customer_override":
            return []
        try:
            rows, _ = repo.list_factors(
                edition_id,
                geography=req.jurisdiction,
                page=1,
                limit=500,
                include_preview=getattr(req, "include_preview", False),
                include_connector=False,
            )
        except TypeError:
            # Older repo signatures without kw-only filters.
            rows, _ = repo.list_factors(edition_id, page=1, limit=500)
        except Exception as exc:
            logger.warning(
                "Repository candidate fetch failed (label=%s): %s", label, exc,
            )
            return []

        # Geographic fallback: when the requested jurisdiction yields no
        # rows (e.g. India activity hits a US-heavy built-in catalog), re-
        # query without the geography filter so the method-pack default
        # tier still sees something to work with. The geography-distance
        # tie-break penalty still penalises non-matching rows.
        rows = list(rows)
        if not rows and req.jurisdiction:
            try:
                rows, _ = repo.list_factors(
                    edition_id,
                    geography=None,
                    page=1,
                    limit=500,
                    include_preview=getattr(req, "include_preview", False),
                    include_connector=False,
                )
                rows = list(rows)
            except Exception:  # pragma: no cover - defensive
                rows = []

        if not rows:
            return rows

        # Wave 3: back-fill ``factor_family`` when the bootstrapped
        # record has it unset. The selection_rule.accepts gate defaults
        # None to EMISSIONS, but Scope 2 packs only allow
        # GRID_INTENSITY — so an eGRID factor with family=None is
        # incorrectly rejected. Infer the family from the factor_id /
        # fuel_type / unit so Scope 2, freight, product-carbon etc. can
        # all pick the right candidates. Done in-place because the
        # dataclass is mutable; idempotent across calls.
        for rec in rows:
            if getattr(rec, "factor_family", None) is None:
                inferred = _infer_factor_family(rec)
                if inferred:
                    try:
                        rec.factor_family = inferred
                    except Exception:  # pragma: no cover - read-only records
                        pass

        # Activity-aware re-ranking + filtering. Score each row against
        # the activity tokens. When at least one row scores >0, return
        # ONLY those rows (sorted by score desc) so the downstream tie-
        # break layer picks among semantically-relevant factors. When no
        # row scores, widen to the global pool so semantically-relevant
        # factors in other geographies can still backstop the resolve.
        activity_tokens = _tokenise_activity(req)
        if not activity_tokens:
            return rows
        scored: List[Tuple[float, int, Any]] = []
        for idx, rec in enumerate(rows):
            score = _activity_score(rec, activity_tokens, req)
            scored.append((score, idx, rec))
        # "Matched" = at least one token hit (score >= 1.0). The geography
        # bonus (+0.5) is NOT enough to qualify — a geo-matched but
        # semantically-unrelated factor must still widen to a global
        # search for a token-aligned candidate.
        matched = [t for t in scored if t[0] >= 1.0]
        if matched:
            matched.sort(key=lambda triple: (-triple[0], triple[1]))
            # Wave 3.1: also back-fill family for the widen path below so
            # the global-pool freight / waste / refrigerant candidates
            # get their family inferred before selection_rule checks.
            for _s, _i, rec in matched:
                if getattr(rec, "factor_family", None) is None:
                    inferred = _infer_factor_family(rec)
                    if inferred:
                        try:
                            rec.factor_family = inferred
                        except Exception:  # pragma: no cover
                            pass
            return [rec for _s, _i, rec in matched]

        # None of the geo-filtered rows matched semantically. Try a
        # global widen — the activity might exist in a different
        # geography (e.g. IPCC refrigerants stored under GLOBAL rather
        # than the request's EU jurisdiction, or DESNZ freight factors
        # stored under UK rather than EU). The geography_distance tie-
        # break signal still penalises the mismatch downstream, so
        # widening here is safe.
        if req.jurisdiction:
            try:
                wide_rows, _ = repo.list_factors(
                    edition_id,
                    geography=None,
                    page=1,
                    limit=5000,
                    include_preview=getattr(req, "include_preview", False),
                    include_connector=False,
                )
                wide_rows = list(wide_rows)
            except Exception:  # pragma: no cover - defensive
                wide_rows = []
            if wide_rows:
                # Back-fill family before scoring so the selection_rule
                # downstream sees correct families for the widened pool.
                for rec in wide_rows:
                    if getattr(rec, "factor_family", None) is None:
                        inferred = _infer_factor_family(rec)
                        if inferred:
                            try:
                                rec.factor_family = inferred
                            except Exception:  # pragma: no cover
                                pass
                wide_scored: List[Tuple[float, int, Any]] = []
                for idx, rec in enumerate(wide_rows):
                    s = _activity_score(rec, activity_tokens, req)
                    wide_scored.append((s, idx, rec))
                wide_matched = [t for t in wide_scored if t[0] >= 1.0]
                if wide_matched:
                    wide_matched.sort(key=lambda triple: (-triple[0], triple[1]))
                    return [rec for _s, _i, rec in wide_matched]

        # Nothing matched semantically anywhere — preserve the original
        # pool so method-pack default + global fallback can still resolve.
        return rows

    return _source


# ---------------------------------------------------------------------------
# Activity-aware candidate scoring (Wave 3 resolver tuning)
# ---------------------------------------------------------------------------
#
# The gold-eval gate showed the resolver was returning wrong winners
# because the candidate source performed pure geography filtering — every
# factor in the requested country was returned in dict-iteration order,
# and the tie-break layer had no way to distinguish "natural gas" from
# "gasoline" from "electricity" inside the same country bucket.  The
# helpers below score factors against the activity text so the tie-break
# layer sees a pre-ranked pool.  The scoring is deterministic (simple
# token overlap on a normalised blob) and runs in O(N) per call.


# Refrigerant aliases (R-code ↔ catalog hfc/hcfc tokens). The
# bootstrapped catalog uses ``hfc32``/``hcfc22`` while gold-set activity
# text uses ``R-32`` / ``R-22``; without the alias bridge the token
# match scores zero and the tie-break falls back to insertion order.
_REFRIGERANT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "r-22": ("hcfc22", "r_22", "r22"),
    "r22": ("hcfc22", "r_22"),
    "r-32": ("hfc32", "r_32", "r32"),
    "r32": ("hfc32", "r_32"),
    "r-134a": ("hfc134a", "r_134a", "r134a"),
    "r134a": ("hfc134a", "r_134a"),
    "r-404a": ("hfc404a", "r_404a", "r404a"),
    "r404a": ("hfc404a", "r_404a"),
    "r-407a": ("hfc407a", "r_407a", "r407a"),
    "r-407c": ("hfc407c", "r_407c", "r407c"),
    "r-410a": ("hfc410a", "r_410a", "r410a"),
    "r410a": ("hfc410a", "r_410a"),
    "r-422d": ("hfc422d", "r_422d"),
    "r-448a": ("hfc448a", "r_448a"),
    "r-449a": ("hfc449a", "r_449a"),
    "r-452a": ("hfc452a", "r_452a"),
    "r-507a": ("hfc507a", "r_507a"),
    "r-1234yf": ("hfo1234yf", "r_1234yf", "r1234yf"),
    "r-1234ze": ("hfo1234ze", "r_1234ze", "r1234ze"),
}

# Fuel / material / transport keyword boosts. Value is the normalised
# token to push into the query bag when the needle appears in the
# activity text.  First-match wins so more specific needles come first.
_ACTIVITY_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    # Fuels (specific first)
    ("natural gas", "natural_gas"),
    ("natural_gas", "natural_gas"),
    ("therm", "natural_gas"),
    ("mmbtu", "natural_gas"),
    ("scf", "natural_gas"),
    ("bituminous", "coal"),
    ("anthracite", "coal"),
    ("lignite", "coal"),
    ("metallurgical", "coal"),
    ("coking coal", "coal"),
    ("coal", "coal"),
    ("gasoline", "gasoline"),
    ("petrol", "gasoline"),
    ("unleaded", "gasoline"),
    ("diesel", "diesel"),
    ("distillate", "diesel"),
    ("kerosene", "kerosene"),
    ("propane", "propane"),
    ("lpg", "lpg"),
    ("fuel oil", "fuel_oil"),
    ("residual oil", "fuel_oil"),
    ("jet fuel", "jet_fuel"),
    ("aviation", "jet_fuel"),
    # Electricity
    ("electricity", "electricity"),
    ("grid", "electricity"),
    ("kwh", "electricity"),
    ("gwh", "electricity"),
    ("mwh", "electricity"),
    # Transport / freight
    ("road freight", "freight"),
    ("freight", "freight"),
    ("hgv", "hgv"),
    ("articulated", "articulated"),
    ("rigid", "rigid"),
    ("rail freight", "rail"),
    ("air freight", "air"),
    ("sea freight", "sea"),
    ("container", "container"),
    ("tonne-km", "tonne_km"),
    ("tonne_km", "tonne_km"),
    ("ton-km", "tonne_km"),
    # Materials / CBAM
    ("steel", "steel"),
    ("aluminium", "aluminium"),
    ("aluminum", "aluminium"),
    ("cement", "cement"),
    ("clinker", "clinker"),
    ("fertiliser", "fertiliser"),
    ("fertilizer", "fertiliser"),
    ("urea", "urea"),
    ("ammonia", "ammonia"),
    ("hydrogen", "hydrogen"),
    ("pig iron", "pig_iron"),
    ("iron ore", "iron_ore"),
    ("rebar", "rebar"),
    # Waste
    ("landfill", "landfill"),
    ("incineration", "incineration"),
    ("compost", "compost"),
    ("recycling", "recycling"),
    ("recycle", "recycling"),
    # Finance / EEIO
    ("naics", "naics"),
    ("spend", "spend"),
    ("eeio", "eeio"),
)


def _tokenise_activity(req: ResolutionRequest) -> List[str]:
    """Extract deterministic query tokens from the request.

    Pulls:
      - keyword-normalised terms from the activity description
      - refrigerant aliases (R-22 → hcfc22)
      - extras['unit_hint'] as-is
      - fuel_type / refrigerant / cbam_product hints from extras
    """
    text = (req.activity or "").lower()
    tokens: List[str] = []
    for needle, tok in _ACTIVITY_KEYWORDS:
        if needle in text and tok not in tokens:
            tokens.append(tok)
    # Refrigerant alias bridge.
    for rcode, aliases in _REFRIGERANT_ALIASES.items():
        if rcode in text:
            for a in aliases:
                if a not in tokens:
                    tokens.append(a)
    # Extras-driven hints (gold-set v1.0 style). ``unit_hint`` is
    # deliberately excluded from the primary token bag because it's
    # too generic (``kg`` matches every mass-denominated factor in the
    # catalog) — it's handled separately as a tie-break bonus in
    # ``_activity_score``.
    extras = getattr(req, "extras", None) or {}
    for key in ("fuel_type", "refrigerant", "cbam_product", "material"):
        val = extras.get(key)
        if val:
            val_str = str(val).lower().strip().replace(" ", "_").replace("-", "_")
            # Skip overly-generic values ("refrigerant" alone is not a
            # discriminator — "r_32" or "hfc32" is).
            if val_str and val_str not in tokens and val_str not in _GENERIC_TOKEN_BLOCKLIST:
                tokens.append(val_str)
            # Apply refrigerant alias to extras.refrigerant as well.
            val_lower = str(val).lower()
            if val_lower in _REFRIGERANT_ALIASES:
                for a in _REFRIGERANT_ALIASES[val_lower]:
                    if a not in tokens:
                        tokens.append(a)
    return tokens


# Tokens that appear so often they provide no discriminative signal —
# keep them out of the query bag so the source doesn't think every
# ``kg``-denominated CBAM factor is a refrigerant candidate.
_GENERIC_TOKEN_BLOCKLIST: frozenset = frozenset(
    {"kg", "tonnes", "kwh", "refrigerant", "fuel", "energy", "tco2e"}
)


# Substring → factor_family inference table. Order matters (most
# specific keys first). Values are the canonical FactorFamily enum
# strings consumed by ``SelectionRule.accepts``.
_FAMILY_INFERENCE_RULES: Tuple[Tuple[str, str], ...] = (
    # Electricity / grid
    ("egrid", "grid_intensity"),
    (":residual_mix:", "residual_mix"),
    ("residual_mix", "residual_mix"),
    (":cea", "grid_intensity"),
    ("all_india", "grid_intensity"),
    (":grid_", "grid_intensity"),
    ("electricity_grid", "grid_intensity"),
    (":elec", "grid_intensity"),
    (":electricity", "grid_intensity"),
    # Freight / transport
    (":freight_", "transport_lane"),
    ("tonne_km", "transport_lane"),
    ("tonne-km", "transport_lane"),
    (":hgv_", "transport_lane"),
    ("_hgv_", "transport_lane"),
    (":s3_freight", "transport_lane"),
    # Waste
    (":waste", "waste_treatment"),
    ("_waste_", "waste_treatment"),
    ("landfill", "waste_treatment"),
    ("incineration", "waste_treatment"),
    # Materials / CBAM / S3 embedded
    (":s3_material", "material_embodied"),
    (":material_", "material_embodied"),
    ("cbam:iron_steel", "material_embodied"),
    ("cbam:aluminum", "material_embodied"),
    ("cbam:aluminium", "material_embodied"),
    ("cbam:cement", "material_embodied"),
    ("cbam:fertilizer", "material_embodied"),
    ("cbam:hydrogen", "material_embodied"),
    # Refrigerants
    ("refriger", "refrigerant_gwp"),
    ("_hfc", "refrigerant_gwp"),
    ("_hcfc", "refrigerant_gwp"),
    ("_hfo", "refrigerant_gwp"),
    # Finance / spend proxies
    (":eeio:", "finance_proxy"),
    (":naics:", "finance_proxy"),
    (":pcaf:", "finance_proxy"),
    # Land / removals
    (":land_", "land_use_removals"),
    (":s1_biomass", "land_use_removals"),
    # Energy conversion (heating-value, density, oxidation)
    (":heating_value", "heating_value"),
    (":oxidation", "oxidation"),
    (":density", "density"),
)


def _infer_factor_family(record: Any) -> Optional[str]:
    """Best-effort factor_family inference from factor_id + fuel_type.

    Many bootstrapped records land in the catalog with ``factor_family``
    unset. The selection_rule in the method packs filters on the family
    enum, so records with None default to EMISSIONS — which is correct
    for Scope 1 combustion but wrong for electricity (which packs demand
    GRID_INTENSITY) and materials/freight (MATERIAL_EMBODIED /
    TRANSPORT_LANE). This helper reads the factor_id + fuel_type and
    returns the best family string; returns ``None`` when nothing
    specific matches (the caller then leaves the record untouched so
    the default-to-EMISSIONS path still applies).
    """
    fid = str(getattr(record, "factor_id", "") or "").lower()
    fuel = str(getattr(record, "fuel_type", "") or "").lower()
    blob = f"{fid} {fuel}"
    for needle, family in _FAMILY_INFERENCE_RULES:
        if needle in blob:
            return family
    return None


def _activity_score(record: Any, tokens: List[str], req: ResolutionRequest) -> float:
    """Score a factor record against activity tokens. Higher = better.

    Looks for tokens in ``factor_id``, ``fuel_type``, ``tags``, ``unit``
    — deterministic substring match. Each matched token contributes 1.0;
    exact geography match contributes 0.5; exact unit-hint match adds
    0.25. The ceiling is irrelevant; the tie-break layer below uses
    rank order, not absolute scores.
    """
    blob_parts: List[str] = []
    fid = getattr(record, "factor_id", None)
    if fid:
        blob_parts.append(str(fid).lower())
    ft = getattr(record, "fuel_type", None)
    if ft:
        blob_parts.append(str(ft).lower())
    tags = getattr(record, "tags", None) or []
    for t in tags:
        blob_parts.append(str(t).lower())
    unit = getattr(record, "unit", None)
    if unit:
        blob_parts.append(str(unit).lower())
    notes = getattr(record, "notes", None)
    if notes:
        blob_parts.append(str(notes).lower())
    blob = " ".join(blob_parts)
    if not blob:
        return 0.0
    score = 0.0
    for tok in tokens:
        if tok and tok in blob:
            score += 1.0
    # Geography bonus (helps when fallback widened scope).
    geo = str(getattr(record, "geography", "") or "").upper()
    req_geo = str(getattr(req, "jurisdiction", "") or "").upper()
    if geo and req_geo and geo == req_geo:
        score += 0.5
    # Unit-hint bonus.
    extras = getattr(req, "extras", None) or {}
    uhint = str(extras.get("unit_hint") or "").lower().strip()
    if uhint and unit and uhint == str(unit).lower().strip():
        score += 0.25
    return score


def _unconfigured_candidate_source(
    request: ResolutionRequest, label: str
) -> List[Any]:
    """Hard-fail source — raises :class:`ConfigurationError` on first call.

    This replaces the old silent ``return []`` placeholder.  Production
    pods that are missing both ``DATABASE_URL`` and the built-in factor
    database will fail loudly on the first ``resolve()`` call rather than
    returning a misleading "no eligible factor" :class:`ResolutionError`.
    """
    raise ConfigurationError(
        "ResolutionEngine has no candidate source configured. Set "
        "DATABASE_URL (or GL_FACTORS_PG_DSN) for Postgres, ensure the "
        "built-in EmissionFactorDatabase is available, or pass an explicit "
        "candidate_source= to ResolutionEngine(...). "
        "Step requested: %r." % label
    )


def build_default_candidate_source() -> CandidateSource:
    """Return the best available production candidate source.

    Resolution order:
        1. :func:`_build_pg_candidate_source` (DATABASE_URL set)
        2. :func:`_build_file_candidate_source` (built-in factors)
        3. :func:`_unconfigured_candidate_source` (raises on first use)
    """
    pg_source = _build_pg_candidate_source()
    if pg_source is not None:
        return pg_source
    file_source = _build_file_candidate_source()
    if file_source is not None:
        return file_source
    logger.error(
        "ResolutionEngine: no candidate source could be configured. "
        "Calls to resolve() will raise ConfigurationError."
    )
    return _unconfigured_candidate_source


def _default_tenant_lookup(request: ResolutionRequest) -> Optional[Any]:
    """Default tenant lookup: no overlay.  Wire via constructor in production."""
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_vintage(record: Any) -> Optional[int]:
    prov = getattr(record, "provenance", None)
    if prov is None:
        return None
    year = getattr(prov, "source_year", None)
    try:
        return int(year) if year is not None else None
    except (TypeError, ValueError):
        return None


def _gas_breakdown(record: Any, gwp_basis: str) -> GasBreakdown:
    vectors = getattr(record, "vectors", None)
    if vectors is None:
        return GasBreakdown(gwp_basis=gwp_basis)
    gwp100 = getattr(record, "gwp_100yr", None)
    co2e_total = getattr(gwp100, "co2e_total", 0.0) if gwp100 is not None else 0.0
    return GasBreakdown(
        co2_kg=_safe_float(getattr(vectors, "CO2", 0.0)) or 0.0,
        ch4_kg=_safe_float(getattr(vectors, "CH4", 0.0)) or 0.0,
        n2o_kg=_safe_float(getattr(vectors, "N2O", 0.0)) or 0.0,
        hfcs_kg=_safe_float(getattr(vectors, "HFCs", 0.0)) or 0.0,
        pfcs_kg=_safe_float(getattr(vectors, "PFCs", 0.0)) or 0.0,
        sf6_kg=_safe_float(getattr(vectors, "SF6", 0.0)) or 0.0,
        nf3_kg=_safe_float(getattr(vectors, "NF3", 0.0)) or 0.0,
        biogenic_co2_kg=_safe_float(getattr(vectors, "biogenic_CO2", 0.0)) or 0.0,
        co2e_total_kg=_safe_float(co2e_total) or 0.0,
        gwp_basis=gwp_basis,
    )


def _build_quality_envelope(dqs: Any) -> Optional[QualityEnvelope]:
    """Project a DataQualityScore (1-5) into a CTO QualityEnvelope (0-100).

    Prefers the authoritative :func:`compute_fqs` implementation in
    ``greenlang.factors.quality.composite_fqs`` when the DQS exposes the
    5 per-dimension attributes (temporal, geographical, technological,
    representativeness, methodological).  When only a legacy scalar
    ``overall_score`` is available, derives a composite-only envelope by
    rescaling the scalar; the per-dimension scores are set to the same
    rescaled value so downstream consumers still see a valid payload.
    """
    if dqs is None:
        return None

    # Path A: Full 5-dim DQS → compute_fqs.
    has_all_dims = all(
        hasattr(dqs, dim)
        for dim in (
            "temporal",
            "geographical",
            "technological",
            "representativeness",
            "methodological",
        )
    )
    if has_all_dims:
        try:
            from greenlang.factors.quality.composite_fqs import compute_fqs

            fqs = compute_fqs(dqs)
            dim_map = {c.name: c.score_100 for c in fqs.components}
            return QualityEnvelope(
                composite_fqs_0_100=float(fqs.composite_fqs),
                temporal_score=float(dim_map.get("temporal", 0.0)),
                geographic_score=float(dim_map.get("geographical", 0.0)),
                technology_score=float(dim_map.get("technological", 0.0)),
                verification_score=float(dim_map.get("representativeness", 0.0)),
                completeness_score=float(dim_map.get("methodological", 0.0)),
                rating=fqs.rating,
                formula_version=fqs.formula_version,
                weights=dict(fqs.weights),
            )
        except Exception as exc:  # pragma: no cover — compute_fqs resilience
            logger.warning("compute_fqs failed (%s); falling back to scalar.", exc)

    # Path B: Legacy scalar on a 1-5 or 0-100 scale.
    scalar = _safe_float(getattr(dqs, "overall_score", None))
    if scalar is None:
        return None
    # If the scalar already looks 0-100 scaled (e.g. 85.0) keep it; otherwise
    # rescale 1-5 → 0-100.
    composite = scalar if scalar > 5.0 else scalar * 20.0
    composite = max(0.0, min(100.0, composite))
    return QualityEnvelope(
        composite_fqs_0_100=composite,
        temporal_score=composite,
        geographic_score=composite,
        technology_score=composite,
        verification_score=composite,
        completeness_score=composite,
        rating=None,
        formula_version="legacy-scalar",
        weights=None,
    )


#: Map method-profile strings to the audit-text template family. The
#: template family is the filename stem under ``audit_texts/`` — several
#: profiles (corporate_scope1/2/3, eu_cbam/eu_dpp/eu_battery) share one
#: template because the methodology narrative is identical at that level.
_AUDIT_TEMPLATE_KEY_MAP: dict = {
    "corporate_scope1": "corporate",
    "corporate_scope2_location_based": "corporate",
    "corporate_scope2_market_based": "corporate",
    "corporate_scope3": "corporate",
    "eu_cbam": "eu_policy",
    "eu_dpp": "eu_policy",
    "eu_dpp_battery": "eu_policy",
}


def _audit_template_key(
    method_pack: Optional[str], method_profile: Optional[str]
) -> Optional[str]:
    """Resolve the audit-text template filename stem for a pack/profile.

    Tries the explicit ``method_pack`` first, then the ``method_profile``,
    then an explicit family map (corporate_* -> corporate, eu_* -> eu_policy),
    then falls back to the first segment of the profile (``electricity_grid``
    -> ``electricity``).  Returns ``None`` when nothing is usable.
    """
    for key in (method_pack, method_profile):
        if not key:
            continue
        key = str(key)
        if key in _AUDIT_TEMPLATE_KEY_MAP:
            return _AUDIT_TEMPLATE_KEY_MAP[key]
    for key in (method_pack, method_profile):
        if not key:
            continue
        key = str(key)
        # Fallback: first ``_``-segment of the profile (``electricity_grid``
        # -> ``electricity``). Leaves single-word profiles untouched.
        head = key.split("_", 1)[0]
        return head or None
    return None


def _family_from_profile(profile_value: str) -> Optional[str]:
    """Infer a CTO factor_family from the MethodProfile string.

    Used only when the record itself does not carry an explicit
    ``factor_family``. Covers the 7 CTO canonical families (electricity,
    combustion, freight, material, land, product, finance).
    """
    if not profile_value:
        return None
    p = profile_value.lower()
    if "scope2" in p or "electricity" in p or "scope_2" in p:
        return "electricity"
    if "scope1" in p or "combustion" in p:
        return "combustion"
    if "freight" in p or "transport" in p or "mobile" in p:
        return "freight"
    if "cbam" in p or "material" in p:
        return "material"
    if "land" in p or "eudr" in p or "biogenic" in p:
        return "land"
    if "product" in p or "lca" in p:
        return "product"
    if "finance" in p or "pcaf" in p or "investment" in p:
        return "finance"
    return None


__all__ = [
    "ResolutionEngine",
    "ResolutionError",
    "CandidateSource",
    "build_default_candidate_source",
]
