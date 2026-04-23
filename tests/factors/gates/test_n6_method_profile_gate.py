# -*- coding: utf-8 -*-
"""
N6 — Policy workflows must pass a method_profile at every factor lookup.

Non-negotiable #6 has two layers:

    a) :class:`greenlang.factors.resolution.request.ResolutionRequest`
       already Pydantic-requires ``method_profile``. The resolution
       engine also double-checks with an ``isinstance(..., MethodProfile)``
       guard at the top of :meth:`resolve`. These layers are GREEN today.

    b) Policy workflows (CBAM, CSRD, EUDR, India CCTS) must never bypass
       the resolution engine by reaching into the catalog repository
       directly. This is the gap. The guard lives in
       :mod:`greenlang.factors.middleware.method_profile_guard`
       (STUB today — see TODO list in that module).

This test validates (a) as PASS and marks (b) as xfail until the guard
is wired into the policy workflow entrypoints.

Run standalone::

    pytest tests/factors/gates/test_n6_method_profile_gate.py -v
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.resolution import ResolutionEngine, ResolutionRequest


# ---------------------------------------------------------------------------
# Layer (a) — Pydantic / engine rejection.
# ---------------------------------------------------------------------------


class TestN6ResolutionRequestRequiresMethodProfile:
    """ResolutionRequest construction fails without method_profile (Pydantic)."""

    def test_missing_method_profile_raises_on_construction(self):
        with pytest.raises(Exception) as exc_info:
            ResolutionRequest(activity="diesel combustion")  # type: ignore[call-arg]
        msg = str(exc_info.value).lower()
        assert "method_profile" in msg or "field required" in msg, (
            "N6 violation: ResolutionRequest accepted a request without "
            "method_profile. Pydantic must mark it required at the top of "
            f"the class. Error was: {exc_info.value!r}"
        )

    def test_engine_rejects_programmatic_none_method_profile(self):
        """Even if a caller bypasses Pydantic (object.__new__), the engine
        rejects a None method_profile at the top of resolve()."""
        from greenlang.factors.resolution.engine import ResolutionError

        # Build a request that skips validation: model_construct bypasses
        # Pydantic validators, mirroring an attacker using __new__.
        req = ResolutionRequest.model_construct(
            activity="diesel",
            method_profile=None,  # type: ignore[arg-type]
        )
        engine = ResolutionEngine(candidate_source=lambda *_: [])
        with pytest.raises(ResolutionError) as exc_info:
            engine.resolve(req)
        assert "method_profile" in str(exc_info.value).lower(), (
            "N6 violation: engine.resolve() didn't surface method_profile in "
            f"its error message. Got: {exc_info.value!r}"
        )


# ---------------------------------------------------------------------------
# Layer (b) — Policy-workflow guard over raw factor lookups.
# ---------------------------------------------------------------------------


class TestN6PolicyWorkflowGuard:
    """Raw factor lookup from inside CBAM / CSRD must require method_profile."""

    def test_guard_raises_when_inside_policy_workflow(self):
        """Unit-level: the guard itself fires when in-workflow + missing profile."""
        from greenlang.factors.middleware.method_profile_guard import (
            MethodProfileMissingError,
            policy_workflow,
            require_method_profile,
        )

        @policy_workflow
        def cbam_like_path():
            # Simulating a raw lookup without method_profile.
            require_method_profile(None)

        with pytest.raises(MethodProfileMissingError) as exc_info:
            cbam_like_path()
        assert "method_profile" in str(exc_info.value).lower()

    def test_guard_is_noop_outside_policy_workflow(self):
        """Library and CLI callers are unaffected (no false positives)."""
        from greenlang.factors.middleware.method_profile_guard import (
            require_method_profile,
        )

        # Outside the @policy_workflow decorator → must NOT raise.
        require_method_profile(None)

    def test_guard_accepts_method_profile(self):
        from greenlang.factors.middleware.method_profile_guard import (
            policy_workflow,
            require_method_profile,
        )

        @policy_workflow
        def csrd_like_path():
            require_method_profile(MethodProfile.CORPORATE_SCOPE1)

        csrd_like_path()

    # ---- previously xfail — policy_workflow now wired into CBAM + CSRD ----

    def test_cbam_workflow_entrypoint_is_marked_policy_workflow(self):
        """The CBAM calculator must be decorated as a policy workflow."""
        from applications.GL_CBAM_APP.CBAM_Importer_Copilot.agents.emissions_calculator_agent_v2 import (  # noqa: E501
            EmissionsCalculatorAgentV2,  # type: ignore[attr-defined]
        )

        # The class (or its run method) should declare itself a policy_workflow.
        marker = getattr(EmissionsCalculatorAgentV2, "_gl_policy_workflow", False)
        assert marker is True, (
            "N6 violation: CBAM EmissionsCalculatorAgentV2 is not marked as a "
            "policy_workflow. Add @policy_workflow to its entrypoint so the "
            "guard fires on raw factor lookups."
        )

    def test_csrd_workflow_entrypoint_is_marked_policy_workflow(self):
        from applications.GL_CSRD_APP.CSRD_Reporting_Platform.agents.calculator_agent_v2 import (  # noqa: E501
            CalculatorAgentV2,  # type: ignore[attr-defined]
        )

        marker = getattr(CalculatorAgentV2, "_gl_policy_workflow", False)
        assert marker is True, (
            "N6 violation: CSRD CalculatorAgentV2 is not marked as a "
            "policy_workflow."
        )
