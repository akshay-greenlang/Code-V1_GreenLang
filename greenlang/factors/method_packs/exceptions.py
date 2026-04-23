# -*- coding: utf-8 -*-
"""Method-pack-specific exception classes.

These exceptions implement the ``cannot_resolve_safely`` contract defined in
``docs/specs/method_pack_template.md`` §9.  The primary raison d'etre is
``FactorCannotResolveSafelyError``: without it, CBAM, PEF, and Battery CFP
declarations silently fall through to a low-quality global default, which
is a regulatory violation under EU 2023/956 Art. 4(2) (CBAM), the EU PEF
methodology, and EU 2023/1542 Art. 7 (Battery).

The exception inherits from :class:`FactorsException` so it plays nicely
with the rest of the GreenLang exception hierarchy (structured ``context``
payload, consistent prefix, etc.) without forcing callers to import from
two places.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from greenlang.utilities.exceptions.factors import FactorsException


class FactorCannotResolveSafelyError(FactorsException):
    """Raised when a method pack cannot resolve safely to any candidate.

    Triggered by :class:`~greenlang.factors.method_packs.base.MethodPack`
    instances whose ``cannot_resolve_action`` is
    :class:`CannotResolveAction.RAISE_NO_SAFE_MATCH` (the default for every
    certified pack) when the 7-step cascade produces zero candidates that
    satisfy the selection + boundary rules AND no fallback to a global
    default is permitted.

    Structured payload attached to ``context``:

    * ``pack_id`` — the method-pack identifier (profile value).
    * ``method_profile`` — duplicated for grep-ability.
    * ``reason`` — short machine-readable key: ``"no_candidate"``,
      ``"global_default_blocked"``, or a pack-specific reason.
    * ``evaluated_candidates_count`` — how many candidates were inspected
      and then filtered out by the selection rule before the exception
      fired. Useful for distinguishing "no data at all" (0) from "data
      exists but is of the wrong shape" (> 0).

    Example::

        raise FactorCannotResolveSafelyError(
            message="CBAM pack refused to fall back to global default",
            pack_id="eu_cbam",
            method_profile="eu_cbam",
            reason="global_default_blocked",
            evaluated_candidates_count=3,
        )
    """

    def __init__(
        self,
        message: str,
        *,
        pack_id: Optional[str] = None,
        method_profile: Optional[str] = None,
        reason: Optional[str] = None,
        evaluated_candidates_count: Optional[int] = None,
        evaluated_steps: Optional[Iterable[str]] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        context = dict(context or {})
        if pack_id is not None:
            context["pack_id"] = pack_id
        if method_profile is not None:
            context["method_profile"] = method_profile
        if reason is not None:
            context["reason"] = reason
        if evaluated_candidates_count is not None:
            context["evaluated_candidates_count"] = int(evaluated_candidates_count)
        if evaluated_steps is not None:
            context["evaluated_steps"] = list(evaluated_steps)
        super().__init__(message, agent_name=agent_name, context=context)

        # Mirror the structured fields as direct attributes so callers can
        # do ``err.pack_id`` without digging through ``err.context``.
        self.pack_id = pack_id
        self.method_profile = method_profile
        self.reason = reason
        self.evaluated_candidates_count = evaluated_candidates_count


__all__ = ["FactorCannotResolveSafelyError"]
