# -*- coding: utf-8 -*-
"""Unit tests for the ``method_profile_guard`` middleware (CTO N6).

These tests exercise the guard in isolation — no ResolutionEngine, no
catalog repository, no application pipelines.  They cover:

    * decorator semantics on both classes and functions
    * ``require_method_profile`` raising inside a policy-workflow scope
    * ``require_method_profile`` no-op for library / CLI callers
    * the CBAM and CSRD calculator agents carry the policy-workflow marker

The N6 gate test (``tests/factors/gates/test_n6_method_profile_gate.py``)
exercises the end-to-end path with the real agents; keeping the
middleware-level coverage here lets us pin the contract for the guard
itself without booting the CBAM / CSRD agents on every assertion.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


# ---------------------------------------------------------------------------
# Work around pre-existing repo hazard: ``greenlang/agents/__init__.py``
# lazy-imports a non-existent ``greenlang.agents.boiler_agent`` whenever
# anything pulls ``greenlang.sdk`` transitively (which CBAM and CSRD
# agents do via ``greenlang.sdk.base``).  Installing a tiny stub lets
# the CBAM / CSRD agent modules import successfully during these
# middleware tests without touching the broken production code.
# ---------------------------------------------------------------------------


def _install_boiler_agent_stub() -> None:
    name = "greenlang.agents.boiler_agent"
    if name in sys.modules:
        return
    stub = ModuleType(name)

    class BoilerAgent:  # pragma: no cover — test environment stub
        pass

    stub.BoilerAgent = BoilerAgent  # type: ignore[attr-defined]
    sys.modules[name] = stub


def _install_csrd_legacy_stubs() -> None:
    """Stub the legacy ``greenlang.sdk.emission_factor_client`` /
    ``greenlang.models.emission_factor`` imports still referenced by
    the CSRD agent.  See the matching block in
    ``tests/factors/gates/conftest.py``.
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

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.middleware.method_profile_guard import (
    MethodProfileMissingError,
    is_policy_workflow_caller,
    policy_workflow,
    require_method_profile,
)


# ---------------------------------------------------------------------------
# App-tree import shim — mirror the gates conftest so this middleware
# test module can run standalone (e.g. ``pytest tests/factors/middleware``)
# without depending on the gates fixtures.
# ---------------------------------------------------------------------------


def _register_applications_tree() -> None:
    """Register underscored namespace package aliases for ``applications/GL-*-APP/...``.

    We register only the package-level stubs; importing the agent modules
    themselves is left to ``importlib.import_module`` calls inside the
    test functions (lazy — so heavy agent dependencies only load if the
    test actually runs).
    """
    repo_root = Path(__file__).resolve().parents[3]
    apps_dir = repo_root / "applications"

    def _ns(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        if not path.exists():
            return
        mod = ModuleType(name)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = mod

    _ns("applications", apps_dir)
    cbam_root = apps_dir / "GL-CBAM-APP" / "CBAM-Importer-Copilot"
    _ns("applications.GL_CBAM_APP", apps_dir / "GL-CBAM-APP")
    _ns("applications.GL_CBAM_APP.CBAM_Importer_Copilot", cbam_root)
    _ns(
        "applications.GL_CBAM_APP.CBAM_Importer_Copilot.agents",
        cbam_root / "agents",
    )
    csrd_root = apps_dir / "GL-CSRD-APP" / "CSRD-Reporting-Platform"
    _ns("applications.GL_CSRD_APP", apps_dir / "GL-CSRD-APP")
    _ns("applications.GL_CSRD_APP.CSRD_Reporting_Platform", csrd_root)
    _ns(
        "applications.GL_CSRD_APP.CSRD_Reporting_Platform.agents",
        csrd_root / "agents",
    )


_register_applications_tree()


# ---------------------------------------------------------------------------
# Decorator semantics
# ---------------------------------------------------------------------------


class TestPolicyWorkflowDecorator:
    def test_policy_workflow_decorator_marks_class(self):
        """Applying the decorator sets the marker attributes on the class."""

        @policy_workflow
        class MyAgent:
            def run(self):
                return "ok"

        assert getattr(MyAgent, "_gl_policy_workflow", False) is True
        assert getattr(MyAgent, "__policy_workflow__", False) is True

    def test_policy_workflow_decorator_marks_function(self):
        """Functions carry the same marker so free-function entrypoints work."""

        @policy_workflow
        def cbam_batch_job(x):
            return x

        assert cbam_batch_job._gl_policy_workflow is True
        assert cbam_batch_job.__policy_workflow__ is True
        # Behavioural preservation.
        assert cbam_batch_job(42) == 42

    def test_is_policy_workflow_caller_class(self):
        @policy_workflow
        class Agent:
            pass

        assert is_policy_workflow_caller(Agent) is True
        assert is_policy_workflow_caller(Agent()) is True

        class Plain:
            pass

        assert is_policy_workflow_caller(Plain) is False
        assert is_policy_workflow_caller(Plain()) is False

    def test_is_policy_workflow_caller_none_defaults_to_contextvar(self):
        """When caller=None the helper inspects the active context var."""
        assert is_policy_workflow_caller(None) is False

        @policy_workflow
        def scope():
            return is_policy_workflow_caller(None)

        assert scope() is True


# ---------------------------------------------------------------------------
# require_method_profile enforcement
# ---------------------------------------------------------------------------


class TestRequireMethodProfile:
    def test_require_method_profile_raises_on_missing(self):
        """policy_workflow caller + no method_profile -> raises."""

        @policy_workflow
        def cbam_like():
            require_method_profile(None)

        with pytest.raises(MethodProfileMissingError) as exc_info:
            cbam_like()
        assert "method_profile" in str(exc_info.value).lower()
        # The error carries the list of canonical profiles for operator hint.
        assert exc_info.value.expected_profiles  # non-empty tuple

    def test_require_method_profile_raises_on_blank_string(self):
        @policy_workflow
        def scope():
            require_method_profile({"method_profile": "   "})

        with pytest.raises(MethodProfileMissingError):
            scope()

    def test_require_method_profile_ok_with_profile(self):
        """policy_workflow caller + method_profile set -> no raise."""

        @policy_workflow
        def csrd_like():
            require_method_profile(
                {"method_profile": MethodProfile.CORPORATE_SCOPE2_LOCATION}
            )

        # Should not raise.
        csrd_like()

    def test_require_method_profile_ok_with_string_profile(self):
        @policy_workflow
        def scope():
            require_method_profile({"method_profile": "corporate_scope2_location_based"})

        scope()

    def test_require_method_profile_ok_with_direct_enum(self):
        @policy_workflow
        def scope():
            # Caller passes the profile directly, not a dict.
            require_method_profile(MethodProfile.EU_CBAM)

        scope()

    def test_non_policy_caller_no_enforcement(self):
        """Developer SDK / CLI calls with no method_profile never raise."""
        # Outside any @policy_workflow context the guard is a no-op, even
        # if the payload would otherwise be considered missing.
        require_method_profile(None)
        require_method_profile({"method_profile": None})
        require_method_profile({"method_profile": ""})

    def test_require_method_profile_explicit_caller_object(self):
        """Explicit caller= overrides the ContextVar check."""

        @policy_workflow
        class Agent:
            pass

        # Even outside the @policy_workflow scope, passing the decorated
        # class / instance as caller= forces enforcement.
        with pytest.raises(MethodProfileMissingError):
            require_method_profile(None, caller=Agent)

        with pytest.raises(MethodProfileMissingError):
            require_method_profile(None, caller=Agent())

    def test_error_carries_caller_identity(self):
        """Error messages surface the closest caller for operator triage."""

        @policy_workflow
        def cbam_run_step():
            require_method_profile(None)

        with pytest.raises(MethodProfileMissingError) as exc_info:
            cbam_run_step()
        assert exc_info.value.caller is not None
        assert "cbam_run_step" in str(exc_info.value.caller).lower() or \
               "policy" in str(exc_info.value.caller).lower()


# ---------------------------------------------------------------------------
# Downstream agents carry the marker
# ---------------------------------------------------------------------------


class TestAgentsAreDecorated:
    def test_cbam_agent_is_decorated(self):
        """CBAM EmissionsCalculatorAgent_v2 must expose ``_gl_policy_workflow``."""
        mod = importlib.import_module(
            "applications.GL_CBAM_APP.CBAM_Importer_Copilot.agents."
            "emissions_calculator_agent_v2"
        )
        # Canonical class + PEP8 alias both live on the module.
        assert getattr(mod.EmissionsCalculatorAgent_v2, "_gl_policy_workflow", False) is True
        assert getattr(mod.EmissionsCalculatorAgent_v2, "__policy_workflow__", False) is True
        assert getattr(mod.EmissionsCalculatorAgentV2, "_gl_policy_workflow", False) is True

    def test_csrd_agent_is_decorated(self):
        """CSRD CalculatorAgentV2 must expose ``_gl_policy_workflow``."""
        mod = importlib.import_module(
            "applications.GL_CSRD_APP.CSRD_Reporting_Platform.agents."
            "calculator_agent_v2"
        )
        assert getattr(mod.CalculatorAgentV2, "_gl_policy_workflow", False) is True
        assert getattr(mod.CalculatorAgentV2, "__policy_workflow__", False) is True
