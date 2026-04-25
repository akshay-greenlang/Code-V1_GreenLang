# -*- coding: utf-8 -*-
"""End-to-end alpha demo helpers (Wave C / TaskCreate #19 / WS6-T3).

Why this exists
---------------
The alpha v0.1 catalog is back-fed from the legacy ``EmissionFactorRecord``
shape via :func:`greenlang.factors.api_v0_1_alpha_routes._coerce_v0_1`.
A real publish pipeline that takes a v0.1-shape dict end-to-end is not
yet wired (it is Wave D scope), so a strict round-trip
"publish a v0.1 factor -> SDK fetch the SAME v0.1 factor" demo cannot
go through ``ingest_builtin_database`` without a lossy coercion step.

The shim below installs:

  * A minimal fake ``factors_service`` on ``app.state`` so the alpha
    router's ``_service(request)`` and ``_edition_id(request, svc)``
    helpers find a backing store.
  * A monkey-patched ``_coerce_v0_1`` (in
    :mod:`greenlang.factors.api_v0_1_alpha_routes`) that returns the
    seeded v0.1 factor dict verbatim — bypassing the legacy-record
    extraction path that would otherwise drop most of the v0.1
    provenance and review fields.

This is the only path that lets the canonical alpha demo described in
CTO doc §19.1 succeed bit-for-bit today (one record flows: schema gate
-> AlphaProvenanceGate -> alpha API -> SDK fetch -> field-by-field
verify). When the real Wave D publish pipeline lands, the shim is
deleted and the SDK e2e test points at the real publisher.

Production code is NOT modified by this helper.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class _FakeRepo:
    """Minimal repository surface the alpha router actually calls.

    The router uses three methods:

      * ``resolve_edition(requested) -> str``
      * ``get_factor(edition, factor_id_or_urn) -> Optional[Any]``
      * ``list_factors(edition, page=..., limit=..., include_preview=...,
        include_connector=...) -> Tuple[List[Any], int]``
    """

    def __init__(
        self,
        edition_id: str,
        factors_by_urn: Dict[str, Dict[str, Any]],
    ) -> None:
        self._edition_id = edition_id
        # Each value is a ready-made v0.1 factor dict; the router wraps
        # whichever one matched into a sentinel object that is then
        # passed to (the monkey-patched) ``_coerce_v0_1``.
        self._factors_by_urn = dict(factors_by_urn)

    # -- Repository surface ------------------------------------------------

    def resolve_edition(self, requested: Optional[str]) -> str:
        return self._edition_id

    def get_default_edition_id(self) -> str:
        return self._edition_id

    def get_factor(self, edition: str, factor_id: str, **_: Any) -> Optional[Any]:
        # Return a sentinel that carries the matched v0.1 factor on it.
        # ``_coerce_v0_1`` is monkey-patched and pulls the dict back out.
        record = self._factors_by_urn.get(factor_id)
        if record is None:
            return None
        return _SentinelRecord(record)

    def list_factors(
        self,
        edition: str,
        *_args: Any,
        page: int = 1,
        limit: int = 50,
        **_kwargs: Any,
    ) -> Tuple[List[Any], int]:
        rows = [_SentinelRecord(rec) for rec in self._factors_by_urn.values()]
        if page > 1:
            return [], len(rows)
        return rows[:limit], len(rows)


class _SentinelRecord:
    """Opaque carrier for the v0.1 factor dict seeded by the test.

    The real ``EmissionFactorRecord`` is dataclass-shaped; the alpha
    router accesses every legacy attribute via ``getattr(record, attr,
    default)``. We rely on the monkey-patched ``_coerce_v0_1`` reading
    the ``_v0_1_factor`` attribute first and never touching the legacy
    fields, so the sentinel intentionally has no other state.
    """

    def __init__(self, factor: Dict[str, Any]) -> None:
        self._v0_1_factor = dict(factor)


class _FakeService:
    """Minimal service shim — the router only ever touches ``.repo``."""

    def __init__(self, repo: _FakeRepo) -> None:
        self.repo = repo


def install_alpha_e2e_shim(
    monkeypatch: Any,
    app: Any,
    *,
    edition_id: str,
    factors: List[Dict[str, Any]],
    mode: str = "shim",
) -> None:
    """Wire the v0.1 factors into the running alpha app.

    Args:
        monkeypatch: pytest's ``monkeypatch`` fixture.
        app: the FastAPI app returned by ``create_factors_app()``.
        edition_id: synthetic edition id surfaced by ``/v1/healthz``.
        factors: list of v0.1-shape factor dicts (each must satisfy the
            ``factor_record_v0_1.schema.json`` contract AND the
            ``AlphaProvenanceGate`` — assert in the test before calling
            this helper).
        mode: ``"shim"`` (default — Wave C path) installs the in-memory
            ``_FakeRepo`` + ``_coerce_v0_1`` monkey-patch; ``"real"`` (Wave D
            path) instead publishes via :class:`AlphaFactorRepository` and
            attaches it to ``app.state.alpha_factor_repo``. Both modes also
            seed a shim ``factors_service`` so ``/v1/healthz`` reports the
            requested edition id deterministically.

    Side effects:
        * ``app.state.factors_service`` -> :class:`_FakeService` keyed on
          each factor's ``urn``.
        * In shim mode:
          ``greenlang.factors.api_v0_1_alpha_routes._coerce_v0_1`` is
          patched to return the seeded v0.1 dict verbatim. The patch is
          undone automatically when the test's monkeypatch fixture
          finalises.
        * In real mode: ``app.state.alpha_factor_repo`` is set to a fresh
          in-memory :class:`AlphaFactorRepository` with each factor
          published; the legacy shim path is *also* installed so
          ``/v1/healthz`` reports the requested edition id and existing
          tests that rely on the shim continue to pass.
    """
    by_urn: Dict[str, Dict[str, Any]] = {f["urn"]: f for f in factors}
    repo = _FakeRepo(edition_id, by_urn)
    app.state.factors_service = _FakeService(repo)

    from greenlang.factors import api_v0_1_alpha_routes as alpha_routes

    def _coerce_v0_1_passthrough(record: Any, source_version: str = "0.1") -> Dict[str, Any]:
        # Deliberately ignore the legacy-record coercion path. The
        # sentinel record carries the seeded v0.1 dict verbatim.
        if isinstance(record, _SentinelRecord):
            return dict(record._v0_1_factor)
        # Fallback: empty dict (the router will then 404 cleanly).
        return {}

    monkeypatch.setattr(
        alpha_routes, "_coerce_v0_1", _coerce_v0_1_passthrough
    )

    if mode == "real":
        from greenlang.factors.repositories import AlphaFactorRepository

        real_repo = AlphaFactorRepository(dsn="sqlite:///:memory:")
        for record in factors:
            real_repo.publish(record)
        app.state.alpha_factor_repo = real_repo
    elif mode == "shim":
        # Explicitly clear any previously-bound real repo so the router
        # falls through to the shim path.
        app.state.alpha_factor_repo = None
    else:  # pragma: no cover — defensive
        raise ValueError(f"install_alpha_e2e_shim(mode={mode!r}) is invalid")


def good_ipcc_ar6_factor() -> Dict[str, Any]:
    """Canonical v0.1-shape IPCC AR6 stationary-combustion factor.

    Mirrors the ``good_factor`` fixture used by
    ``test_factor_record_v0_1_schema_loads.py`` and
    ``test_alpha_provenance_gate.py``. Kept in sync intentionally —
    if those fixtures drift, this helper drifts with them, so the
    canonical demo never tests against a record shape that the schema
    gate rejects.
    """
    return {
        "urn": (
            "urn:gl:factor:ipcc-ar6:stationary-combustion:"
            "natural-gas-residential:v1"
        ),
        "factor_id_alias": (
            "EF:IPCC:stationary-combustion:natural-gas-residential:v1"
        ),
        "source_urn": "urn:gl:source:ipcc-ar6",
        "factor_pack_urn": "urn:gl:pack:ipcc-ar6:tier-1-defaults:2021.0",
        "name": "Stationary combustion of natural gas (residential), CO2e",
        "description": (
            "Default Tier 1 emission factor for residential stationary "
            "combustion of natural gas, expressed in kgCO2e/TJ on a net "
            "calorific value (NCV) basis. Boundary excludes upstream "
            "extraction and distribution losses."
        ),
        "category": "fuel",
        "value": 56100.0,
        "unit_urn": "urn:gl:unit:kgco2e/tj",
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": "urn:gl:geo:global:world",
        "vintage_start": "2021-01-01",
        "vintage_end": "2099-12-31",
        "resolution": "annual",
        "methodology_urn": (
            "urn:gl:methodology:ipcc-tier-1-stationary-combustion"
        ),
        "boundary": (
            "Net calorific value basis. Excludes upstream extraction and "
            "distribution losses."
        ),
        "licence": "IPCC-PUBLIC",
        "citations": [
            {
                "type": "publication",
                "value": "IPCC AR6 WG3 Annex III, Table 1.4",
                "title": (
                    "IPCC Sixth Assessment Report — Working Group III, "
                    "Annex III"
                ),
            },
            {
                "type": "url",
                "value": "https://www.ipcc.ch/report/ar6/wg3/",
            },
        ],
        "published_at": "2026-04-25T12:00:00Z",
        "extraction": {
            "source_url": "https://www.ipcc.ch/report/ar6/wg3/",
            "source_record_id": (
                "annex-iii;table-1.4;row=natural-gas-residential"
            ),
            "source_publication": (
                "IPCC Sixth Assessment Report — Working Group III, "
                "Annex III"
            ),
            "source_version": "AR6-WG3-Annex-III",
            "raw_artifact_uri": (
                "s3://greenlang-factors-raw/ipcc/ar6/wg3-annex-iii.pdf"
            ),
            "raw_artifact_sha256": (
                "0123456789abcdef0123456789abcdef"
                "0123456789abcdef0123456789abcd00"
            ),
            "parser_id": "greenlang.factors.ingestion.parsers.ipcc_defaults",
            "parser_version": "0.1.0",
            "parser_commit": "deadbeefcafe",
            "row_ref": (
                "Sheet=N/A; Table=1.4; Row=Natural Gas (Residential); "
                "Column=Default EF (kgCO2e/TJ)"
            ),
            "ingested_at": "2026-04-25T11:55:00Z",
            "operator": "bot:parser_ipcc_defaults",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:methodology-lead@greenlang.io",
            "reviewed_at": "2026-04-25T11:58:00Z",
            "approved_by": "human:methodology-lead@greenlang.io",
            "approved_at": "2026-04-25T11:59:00Z",
        },
    }


__all__ = [
    "install_alpha_e2e_shim",
    "good_ipcc_ar6_factor",
]
