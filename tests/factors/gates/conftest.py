# -*- coding: utf-8 -*-
"""Shared fixtures for the 7-gate release suite.

The fixtures here intentionally reuse the top-level ``tests/factors/conftest.py``
where possible (``seed_factors``, ``api_client``, ``signing_key_pair``, ...).
Only gate-specific helpers live in this file:

    * ``certified_factor_records`` — dict of SimpleNamespace / dataclass
      factor records keyed by family (combustion / electricity / refrigerant /
      material / freight / land / finance / product) for N1 + N3.
    * ``co2e_only_record`` — a Certified-status record with gas vectors
      scrubbed to zero so only ``co2e_total`` carries a number. N1 must fail
      this record.
    * ``caller_contexts`` — one dict per {tier, entitlements, tenant_id,
      oem_id} combo used by N4 + N7.
    * ``make_resolve_request`` / ``make_alternate`` — factory helpers
      used by N3 to construct minimal ResolvedFactor payloads.

All production code is imported directly; the only thing we mock is the
``candidate_source`` callable on :class:`ResolutionEngine` so the tests
do not depend on a seeded catalog.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Pre-existing repo hazard: ``greenlang/agents/__init__.py`` declares a
# lazy ``BoilerAgent`` importer that points at a non-existent
# ``greenlang.agents.boiler_agent`` module.  It trips any code path that
# transitively imports ``greenlang.sdk`` (which re-exports
# ``greenlang.integration.sdk``, which in turn pulls
# ``enhanced_client`` -> ``from greenlang.agents import BoilerAgent``).
#
# The N6 gate test imports the CBAM / CSRD calculator agents, both of
# which use ``from greenlang.sdk.base import Agent``.  We register a
# minimal stub module here so the cascade resolves during test
# collection without us having to touch the broken production code
# (out-of-scope for the N6 guard-wiring task).
# ---------------------------------------------------------------------------


def _install_boiler_agent_stub() -> None:
    name = "greenlang.agents.boiler_agent"
    if name in sys.modules:
        return
    stub = ModuleType(name)

    class BoilerAgent:  # pragma: no cover — test environment stub
        """Placeholder used solely so ``greenlang.agents.__getattr__``
        can resolve ``BoilerAgent`` without raising ImportError during
        collection of factor tests.
        """

    stub.BoilerAgent = BoilerAgent  # type: ignore[attr-defined]
    sys.modules[name] = stub


def _install_csrd_legacy_stubs() -> None:
    """Stub the legacy ``greenlang.sdk.emission_factor_client`` /
    ``greenlang.models.emission_factor`` module paths that the CSRD
    agent still references.  Upstream refactors moved the real code to
    ``greenlang.integration.sdk.emission_factor_client`` and
    ``greenlang.data.models.emission_factor`` but the CSRD agent kept
    the old paths.  Fixing those is out of scope here — we install
    minimal stubs so the N6 gate test can import the class.
    """
    sdk_name = "greenlang.sdk.emission_factor_client"
    if sdk_name not in sys.modules:
        sdk_stub = ModuleType(sdk_name)

        class EmissionFactorClient:  # pragma: no cover — stub
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "EmissionFactorClient stub - do not instantiate in tests"
                )

        class EmissionFactorNotFoundError(Exception):
            pass

        class UnitNotAvailableError(Exception):
            pass

        class DatabaseConnectionError(Exception):
            pass

        sdk_stub.EmissionFactorClient = EmissionFactorClient
        sdk_stub.EmissionFactorNotFoundError = EmissionFactorNotFoundError
        sdk_stub.UnitNotAvailableError = UnitNotAvailableError
        sdk_stub.DatabaseConnectionError = DatabaseConnectionError
        sys.modules[sdk_name] = sdk_stub

    models_name = "greenlang.models.emission_factor"
    if models_name not in sys.modules:
        models_pkg = "greenlang.models"
        if models_pkg not in sys.modules:
            pkg_stub = ModuleType(models_pkg)
            pkg_stub.__path__ = []  # type: ignore[attr-defined]
            sys.modules[models_pkg] = pkg_stub
        models_stub = ModuleType(models_name)

        class EmissionFactor:  # pragma: no cover — stub
            pass

        class EmissionResult:  # pragma: no cover — stub
            pass

        models_stub.EmissionFactor = EmissionFactor
        models_stub.EmissionResult = EmissionResult
        sys.modules[models_name] = models_stub


_install_boiler_agent_stub()
_install_csrd_legacy_stubs()


# ---------------------------------------------------------------------------
# Import shim: make ``applications.GL_CBAM_APP.CBAM_Importer_Copilot.<...>``
# resolvable.
#
# The ``applications/`` tree uses dashes in directory names (``GL-CBAM-APP``,
# ``CBAM-Importer-Copilot``) because those are the shipped folder names;
# but Python package identifiers must be underscored.  The N6 gate test
# imports with the underscored form, so we register synthetic package
# modules that point ``sys.modules`` at the dashed directories.
#
# Kept inside the gates conftest (rather than a broader shared conftest)
# because this is the only suite that reaches directly into the
# applications tree — the rest of the factor tests stay within
# ``greenlang.factors.*``.
# ---------------------------------------------------------------------------


def _ensure_namespace_pkg(dotted_name: str, search_path: Path) -> None:
    """Register a lightweight namespace package in ``sys.modules``.

    The module is a bare ``ModuleType`` with a ``__path__`` set so the
    default import system can resolve submodules on demand.  This is
    strictly lazier than eagerly exec'ing every ``__init__.py`` — we
    must not boot the full application package (and its heavy
    transitive imports) just to allow the N6 gate tests to attach.
    """
    if dotted_name in sys.modules:
        return
    if not search_path.exists():
        return
    ns = ModuleType(dotted_name)
    ns.__path__ = [str(search_path)]  # type: ignore[attr-defined]
    sys.modules[dotted_name] = ns


def _register_applications_tree() -> None:
    """Wire up the underscored package names used by the N6 gate test.

    We stop short of exec'ing the agent modules — the actual
    ``from applications.GL_CBAM_APP.* import ...`` statement inside a
    test triggers a normal Python import, at which point the agent's
    own imports resolve in the proper test context.
    """
    repo_root = Path(__file__).resolve().parents[3]
    apps_dir = repo_root / "applications"
    _ensure_namespace_pkg("applications", apps_dir)

    # ----- CBAM -----
    cbam_root = apps_dir / "GL-CBAM-APP" / "CBAM-Importer-Copilot"
    if cbam_root.exists():
        _ensure_namespace_pkg(
            "applications.GL_CBAM_APP", apps_dir / "GL-CBAM-APP"
        )
        _ensure_namespace_pkg(
            "applications.GL_CBAM_APP.CBAM_Importer_Copilot", cbam_root
        )
        _ensure_namespace_pkg(
            "applications.GL_CBAM_APP.CBAM_Importer_Copilot.agents",
            cbam_root / "agents",
        )

    # ----- CSRD -----
    csrd_root = apps_dir / "GL-CSRD-APP" / "CSRD-Reporting-Platform"
    if csrd_root.exists():
        _ensure_namespace_pkg(
            "applications.GL_CSRD_APP", apps_dir / "GL-CSRD-APP"
        )
        _ensure_namespace_pkg(
            "applications.GL_CSRD_APP.CSRD_Reporting_Platform", csrd_root
        )
        _ensure_namespace_pkg(
            "applications.GL_CSRD_APP.CSRD_Reporting_Platform.agents",
            csrd_root / "agents",
        )


_register_applications_tree()


# ---------------------------------------------------------------------------
# Factor record factories — enough fields for SelectionRule.accepts()
# ---------------------------------------------------------------------------


def _vectors(**gases: float) -> SimpleNamespace:
    """Build a GHGVectors-like namespace. Missing gases default to zero."""
    defaults = dict(
        CO2=0.0, CH4=0.0, N2O=0.0, HFCs=0.0, PFCs=0.0, SF6=0.0, NF3=0.0,
        biogenic_CO2=0.0,
    )
    defaults.update(gases)
    return SimpleNamespace(**defaults)


def _gwp100(co2e_total: float) -> SimpleNamespace:
    return SimpleNamespace(
        co2e_total=co2e_total,
        gwp_set=SimpleNamespace(value="IPCC_AR6_100"),
        CH4_gwp=28,
        N2O_gwp=273,
    )


def _base_record(
    *,
    factor_id: str,
    family: str,
    vectors: SimpleNamespace,
    co2e_total: float,
    redistribution_class: str = "open",
    factor_status: str = "certified",
    source_id: str = "epa_hub",
    source_release: str = "2024.1",
    factor_version: str = "1.0.0",
    valid_from: date = date(2024, 1, 1),
    valid_to: Optional[date] = date(2099, 12, 31),
    jurisdiction_country: Optional[str] = "US",
    denominator_unit: str = "kWh",
) -> SimpleNamespace:
    """Construct a duck-typed factor record.

    Fields cover every accessor used by resolution.engine, tier_enforcement,
    and the N1-N7 checks so tests never need real YAML ingest.
    """
    jurisdiction = SimpleNamespace(
        country=jurisdiction_country, region=None, grid_region=None,
    )
    denominator = SimpleNamespace(unit=denominator_unit)
    return SimpleNamespace(
        factor_id=factor_id,
        factor_name=f"Test {family} factor {factor_id}",
        factor_family=family,
        formula_type="direct_factor",
        geography=jurisdiction_country,
        jurisdiction=jurisdiction,
        denominator=denominator,
        unit=denominator_unit,
        vectors=vectors,
        gwp_100yr=_gwp100(co2e_total),
        scope=SimpleNamespace(value="1"),
        boundary=SimpleNamespace(value="combustion"),
        provenance=SimpleNamespace(
            source_org="EPA", source_publication="Test", source_year=2024,
            version=source_release,
        ),
        source_id=source_id,
        source_release=source_release,
        source_version=source_release,
        release_version=source_release,
        factor_version=factor_version,
        status=factor_status,
        factor_status=factor_status,
        redistribution_class=redistribution_class,
        license_class=redistribution_class,
        valid_from=valid_from,
        valid_to=valid_to,
        verification=SimpleNamespace(status="external_verified"),
        explainability=SimpleNamespace(assumptions=[], rationale=None),
        dqs=SimpleNamespace(overall_score=85.0),
        uncertainty_95ci=0.05,
        uncertainty_distribution="normal",
        replacement_factor_id=None,
        primary_data_flag="secondary",
    )


@pytest.fixture()
def certified_factor_records() -> Dict[str, SimpleNamespace]:
    """One Certified-status record per family, populated with real gas vectors."""
    return {
        # Combustion must carry CO2, CH4, N2O.
        "combustion": _base_record(
            factor_id="EF:US:diesel:2024:v1",
            family="emissions",
            vectors=_vectors(CO2=10.18, CH4=0.00082, N2O=0.000164),
            co2e_total=10.2097,
            denominator_unit="gallon",
        ),
        # Electricity — CO2 + CH4 + N2O required.
        "electricity": _base_record(
            factor_id="EF:US:egrid:2023:v1",
            family="grid_intensity",
            vectors=_vectors(CO2=0.382, CH4=0.00004, N2O=0.000005),
            co2e_total=0.3853,
            denominator_unit="kWh",
        ),
        # Refrigerants — HFC / PFC / SF6 / NF3 fluorinated gases.
        "refrigerant": _base_record(
            factor_id="EF:GLOBAL:r410a:2023:v1",
            family="refrigerant_gwp",
            vectors=_vectors(HFCs=1.0),
            co2e_total=2088.0,
            denominator_unit="kg",
        ),
        # Material / CBAM
        "material": _base_record(
            factor_id="EF:EU:steel:2024:v1",
            family="material_embodied",
            vectors=_vectors(CO2=1.80, CH4=0.005, N2O=0.0001),
            co2e_total=1.85,
            jurisdiction_country="EU",
            denominator_unit="kg",
        ),
        # Freight
        "freight": _base_record(
            factor_id="EF:US:freight_truck:2023:v1",
            family="transport_lane",
            vectors=_vectors(CO2=0.103, CH4=0.0001, N2O=0.00005),
            co2e_total=0.105,
            denominator_unit="tonne-km",
        ),
        # Land use removals
        "land": _base_record(
            factor_id="EF:ID:palm:2023:v1",
            family="land_use_removals",
            vectors=_vectors(CO2=7.1, CH4=0.01, N2O=0.001),
            co2e_total=7.3,
            jurisdiction_country="ID",
            denominator_unit="kg",
        ),
        # Finance proxy
        "finance": _base_record(
            factor_id="EF:US:pcaf:2023:v1",
            family="finance_proxy",
            vectors=_vectors(CO2=0.000120),
            co2e_total=0.000123,
            denominator_unit="USD",
        ),
        # Product carbon
        "product": _base_record(
            factor_id="EF:GLOBAL:cement:2023:v1",
            family="material_embodied",
            vectors=_vectors(CO2=0.92, CH4=0.001, N2O=0.0001),
            co2e_total=0.93,
            jurisdiction_country=None,
            denominator_unit="kg",
        ),
    }


@pytest.fixture()
def co2e_only_record() -> SimpleNamespace:
    """A Certified-status factor with ONLY co2e_total populated (no gases).

    This is the exact shape that N1 must reject. All per-gas vectors are
    zero but the aggregate CO2e is non-zero — i.e. the source gave us
    a single co2e number with no decomposition.
    """
    return _base_record(
        factor_id="EF:US:co2e_only_bad:2024:v1",
        family="emissions",
        vectors=_vectors(),  # all zeros
        co2e_total=2.5,
        denominator_unit="L",
    )


# ---------------------------------------------------------------------------
# Caller contexts — used by N4 + N7
# ---------------------------------------------------------------------------


@dataclass
class CallerContext:
    """Duck-typed caller principal used by licensing / tier checks."""

    tier: str
    entitlements: List[str] = field(default_factory=list)
    tenant_id: Optional[str] = None
    oem_id: Optional[str] = None
    user_id: str = "test-caller"

    def as_request_state_user(self) -> Dict[str, Any]:
        """Match the shape read by LicensingGuardMiddleware._caller_grants."""
        return {
            "tier": self.tier,
            "entitlements": list(self.entitlements),
            "packs": list(self.entitlements),
            "tenant_id": self.tenant_id,
            "oem_id": self.oem_id,
        }


@pytest.fixture()
def caller_contexts() -> Dict[str, CallerContext]:
    return {
        "community_anon": CallerContext(tier="community", entitlements=[]),
        "developer_pro": CallerContext(tier="pro", entitlements=[]),
        "consulting": CallerContext(tier="consulting", entitlements=[]),
        "enterprise": CallerContext(
            tier="enterprise", entitlements=["licensed", "commercial"],
        ),
        "enterprise_tenant_a": CallerContext(
            tier="enterprise", entitlements=["licensed"], tenant_id="tenant-a",
        ),
        "enterprise_tenant_b": CallerContext(
            tier="enterprise", entitlements=["licensed"], tenant_id="tenant-b",
        ),
        "oem_partner": CallerContext(
            tier="enterprise",
            entitlements=["licensed", "oem_redistributable"],
            oem_id="oem-partner-1",
        ),
        # Community hitting premium packs
        "community_plus_freight": CallerContext(
            tier="community", entitlements=["freight_premium"],
        ),
    }


# ---------------------------------------------------------------------------
# Stubbable candidate-source for ResolutionEngine (N3)
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_candidate_source() -> Callable[[Dict[str, List[Any]]], Callable]:
    """Factory returning a ``candidate_source`` callable.

    Given a {step_label -> [records]} dict, returns a callable suitable for
    ``ResolutionEngine(candidate_source=...)``.
    """

    def _make(records_by_step: Dict[str, List[Any]]):
        def _source(_req, label: str) -> Iterable[Any]:
            return list(records_by_step.get(label, []))

        return _source

    return _make


# ---------------------------------------------------------------------------
# Expose the _base_record / _vectors helpers to each test module so they
# don't have to redefine them.
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_record() -> Callable[..., SimpleNamespace]:
    return _base_record


@pytest.fixture()
def make_vectors() -> Callable[..., SimpleNamespace]:
    return _vectors
