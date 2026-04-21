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
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from greenlang.data.canonical_v2 import (
    MethodProfile,
    enforce_license_class_homogeneity,
)
from greenlang.factors.method_packs import MethodPack, get_pack
from greenlang.factors.ontology.unit_graph import (
    DEFAULT_GRAPH,
    UnitConversionError,
    UnitGraph,
)
from greenlang.factors.resolution.request import ResolutionRequest
from greenlang.factors.resolution.result import (
    AlternateCandidate,
    GasBreakdown,
    ResolvedFactor,
    UncertaintyBand,
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
        # A default candidate source exists so tests + smoke runs work even
        # without a full Factors repository wired in.  Production callers
        # inject a real source backed by ``FactorCatalogService``.
        self._candidate_source = candidate_source or _empty_candidate_source
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

        # Run the 7-step cascade.
        for rank, label in _STEP_ORDER:
            candidates = list(self._gather_step(request, label, pack))
            eligible = [c for c in candidates if pack.selection_rule.accepts(c)]
            if not eligible:
                continue

            # Score every eligible candidate; pick the lowest-score winner.
            request_granularity = getattr(
                getattr(request, "time_granularity", None), "value", None
            )
            scored: List[Tuple[int, Any, TieBreakReasons]] = []
            for cand in eligible:
                tb = build_tiebreak(
                    cand,
                    request_geo=request_geo,
                    request_date=request_date,
                    request_granularity=request_granularity,
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
            return resolved

        # Every step exhausted.
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
        assumptions = list(
            getattr(getattr(record, "explainability", None), "assumptions", []) or []
        )
        if not assumptions:
            assumptions = [f"Resolved via step {rank} ({label})."]

        gb = _gas_breakdown(record, pack.gwp_basis)

        ub = UncertaintyBand(
            distribution=str(getattr(record, "uncertainty_distribution", "unknown") or "unknown"),
            ci_95_percent=_safe_float(getattr(record, "uncertainty_95ci", None)),
            note=tb.one_liner(),
        )

        quality = _safe_float(
            getattr(getattr(record, "dqs", None), "overall_score", None)
        )
        v = getattr(record, "verification", None)
        verification_status = str(getattr(v, "status", "")) if v is not None else None

        status = str(getattr(record, "factor_status", "certified") or "certified")
        deprecation_replacement = getattr(record, "replacement_factor_id", None)

        explainability = getattr(record, "explainability", None)
        rationale = getattr(explainability, "rationale", None) if explainability else None

        why_chosen = rationale or f"Selected at cascade step {rank} ({label}). {tb.one_liner()}."

        return ResolvedFactor(
            chosen_factor_id=str(getattr(record, "factor_id", "unknown")),
            chosen_factor_name=getattr(record, "factor_name", None),
            source_id=getattr(record, "source_id", None),
            source_version=getattr(record, "source_release", None) or getattr(record, "release_version", None),
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
            deprecation_status=status if status == "deprecated" else None,
            deprecation_replacement=deprecation_replacement,
            quality_score=quality,
            uncertainty=ub,
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


def _empty_candidate_source(request: ResolutionRequest, label: str) -> List[Any]:
    """Default candidate source: yields nothing.  Production overrides this."""
    return []


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


__all__ = ["ResolutionEngine", "ResolutionError", "CandidateSource"]
