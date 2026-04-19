# -*- coding: utf-8 -*-
"""
PolicyEngine - v3 Policy Graph enforcement facade
==================================================

Wraps :class:`greenlang.governance.policy.enforcer.PolicyEnforcer` under a
clean product API that returns structured dict responses instead of raising
:class:`RuntimeError` on policy denial.

Usage::

    from greenlang.policy_graph import PolicyEngine

    engine = PolicyEngine()
    result = engine.check_install(manifest, "/packs/my-pack", stage="publish")
    if not result["allowed"]:
        print(result["reason"])

    result = engine.evaluate("bundles/install.rego", {"pack": {...}})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PolicyEngine:
    """
    Unified policy enforcement engine for the v3 Policy Graph.

    This class provides a structured-response interface over the existing
    :class:`~greenlang.governance.policy.enforcer.PolicyEnforcer`.  All methods
    return ``{"allowed": bool, "reason": str | None}`` dicts rather than
    raising exceptions on denial, making them safe for pipeline orchestration.

    Args:
        policy_dir: Directory containing OPA ``.rego`` policy files.
            Defaults to ``~/.greenlang/policies``.

    Example::

        engine = PolicyEngine()
        result = engine.check_install(manifest, "/packs/scope1")
        assert result == {"allowed": True, "reason": None}
    """

    def __init__(self, policy_dir: Path | None = None) -> None:
        from greenlang.governance.policy.enforcer import PolicyEnforcer

        self._enforcer = PolicyEnforcer(policy_dir=policy_dir)
        logger.info(
            "PolicyEngine initialised with policy_dir=%s",
            self._enforcer.policy_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_install(
        self,
        manifest: Any,
        path: str,
        stage: str = "publish",
    ) -> Dict[str, Any]:
        """
        Check whether a pack may be installed or published.

        Wraps the standalone
        :func:`greenlang.governance.policy.enforcer.check_install` function,
        converting any ``RuntimeError`` into a structured denial response.

        Args:
            manifest: Pack manifest object (Pydantic model or plain dict).
            path: Filesystem path to the pack directory.
            stage: ``"publish"`` or ``"add"``.

        Returns:
            ``{"allowed": True, "reason": None}`` on success, or
            ``{"allowed": False, "reason": "<denial message>"}`` on denial.
        """
        try:
            from greenlang.governance.policy.enforcer import (
                check_install as _check_install,
            )

            _check_install(manifest, path, stage)
            return {"allowed": True, "reason": None}
        except RuntimeError as exc:
            logger.warning("Policy denied install: %s", exc)
            return {"allowed": False, "reason": str(exc)}
        except Exception as exc:
            logger.error("Policy check_install error: %s", exc)
            return {"allowed": False, "reason": f"policy error: {exc}"}

    def check_run(
        self,
        pipeline: Any,
        ctx: Any,
    ) -> Dict[str, Any]:
        """
        Check whether a pipeline may be executed.

        Wraps the standalone
        :func:`greenlang.governance.policy.enforcer.check_run` function,
        converting any ``RuntimeError`` into a structured denial response.

        Args:
            pipeline: Pipeline object (must expose ``to_policy_doc()`` or
                ``__dict__``).
            ctx: Execution context (must expose ``egress_targets`` and
                ``region`` attributes).

        Returns:
            ``{"allowed": True, "reason": None}`` on success, or
            ``{"allowed": False, "reason": "<denial message>"}`` on denial.
        """
        try:
            from greenlang.governance.policy.enforcer import (
                check_run as _check_run,
            )

            _check_run(pipeline, ctx)
            return {"allowed": True, "reason": None}
        except RuntimeError as exc:
            logger.warning("Policy denied run: %s", exc)
            return {"allowed": False, "reason": str(exc)}
        except Exception as exc:
            logger.error("Policy check_run error: %s", exc)
            return {"allowed": False, "reason": f"policy error: {exc}"}

    def evaluate(
        self,
        bundle: str,
        input_data: Dict[str, Any],
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Raw OPA evaluation against a Rego bundle or policy file.

        Delegates to :func:`greenlang.governance.policy.opa.evaluate`.

        Args:
            bundle: Path to the ``.rego`` policy file or bundle
                (resolved by the underlying OPA integration).
            input_data: Input document for OPA evaluation.
            data: Optional supplementary data document.

        Returns:
            OPA decision dict (at minimum ``{"allow": bool, "reason": str}``).
        """
        from greenlang.governance.policy.opa import evaluate as _opa_evaluate

        result = _opa_evaluate(bundle, input_data, data=data)
        logger.info(
            "OPA evaluate bundle=%s allowed=%s",
            bundle,
            result.get("allow"),
        )
        return result

    # ------------------------------------------------------------------
    # Delegated convenience methods
    # ------------------------------------------------------------------

    def list_policies(self) -> list[str]:
        """Return names of all loaded policies."""
        return self._enforcer.list_policies()

    def get_policy(self, name: str) -> Optional[str]:
        """Return raw Rego content for a named policy."""
        return self._enforcer.get_policy(name)

    def add_policy(self, policy_file: Path) -> None:
        """
        Add a policy file to the engine's policy directory.

        Args:
            policy_file: Path to the ``.rego`` file to add.
        """
        self._enforcer.add_policy(policy_file)

    def remove_policy(self, name: str) -> None:
        """
        Remove a named policy.

        Args:
            name: Policy stem name (without ``.rego`` extension).
        """
        self._enforcer.remove_policy(name)
