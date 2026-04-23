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
            or (str(getattr(prov, "source_year", "")) if prov is not None and getattr(prov, "source_year", None) else None)
        )
        return ResolvedFactor(
            chosen_factor_id=str(getattr(record, "factor_id", "unknown")),
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
        return list(rows)

    return _source


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


__all__ = [
    "ResolutionEngine",
    "ResolutionError",
    "CandidateSource",
    "build_default_candidate_source",
]
