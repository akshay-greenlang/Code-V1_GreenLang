# -*- coding: utf-8 -*-
"""
method_profile_guard — CTO non-negotiable #6 enforcement.

CTO non-negotiable #6: a policy-workflow code path (CBAM agent, CSRD
agent, EUDR workflow, India CCTS, ...) must NEVER call a raw factor
lookup without a ``method_profile``. ``ResolutionRequest`` already
enforces this at the Pydantic level, but nothing prevents a caller
from reaching into
:class:`greenlang.factors.service.FactorCatalogService` or the catalog
repositories (``FactorCatalogRepository.list_factors`` /
``get_factor``) directly.

This module is the last-mile guard. It exposes:

    * :func:`policy_workflow` — decorator that marks a class OR a
      function as a policy-workflow-scoped entrypoint. Sets
      ``_gl_policy_workflow = True`` (and ``__policy_workflow__ = True``
      for general-purpose detection) on the target, and also toggles
      a :class:`~contextvars.ContextVar` while inside the callable
      so nested raw-lookup callsites can detect the context.
    * :func:`require_method_profile` — free-function guard that raises
      :class:`MethodProfileMissingError` when invoked from inside a
      policy workflow without a ``method_profile``.
    * :func:`is_policy_workflow_caller` — boolean helper that returns
      whether a given object (class, function, or instance) carries
      the policy-workflow marker.

Design rules:

    * Library / CLI / developer-SDK callers are NEVER affected — the
      guard short-circuits unless the ContextVar is set.
    * Explorer-style search calls (``search_factors``, facet lookups,
      catalog browsing) are intentionally NOT guarded; they surface
      metadata, not regulated numeric values.
    * The guard accepts three input shapes on ``require_method_profile``:
      a raw value (``MethodProfile`` / string / ``None``), a dict of
      kwargs, or any object exposing a ``method_profile`` attribute.
    * Error messages name the offending caller and list the expected
      method profiles so operators have something actionable.

Test coverage lives in
``tests/factors/middleware/test_method_profile_guard.py`` and the
7-gate suite ``tests/factors/gates/test_n6_method_profile_gate.py``.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import logging
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


__all__ = [
    "MethodProfileMissingError",
    "require_method_profile",
    "policy_workflow",
    "is_policy_workflow_caller",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MethodProfileMissingError(RuntimeError):
    """Raised when a policy-workflow callsite reaches a raw factor lookup
    without supplying a ``method_profile``.

    Carries the caller name and the list of canonical method profiles
    so the operator / reviewer can fix the call site quickly.
    """

    def __init__(
        self,
        caller: Optional[str] = None,
        expected_profiles: Optional[Iterable[str]] = None,
    ) -> None:
        self.caller = caller or "<unknown policy workflow>"
        self.expected_profiles = tuple(expected_profiles or ())
        profiles_hint = (
            f" Expected one of: {', '.join(self.expected_profiles)}."
            if self.expected_profiles
            else ""
        )
        super().__init__(
            "Non-negotiable #6: policy workflow %r attempted a raw factor "
            "lookup without a method_profile. Every CBAM / CSRD / EUDR / "
            "India-CCTS workflow must bind a MethodProfile before reaching "
            "the catalog.%s"
            % (self.caller, profiles_hint)
        )


# ---------------------------------------------------------------------------
# Canonical method-profile names used in the error hint
# ---------------------------------------------------------------------------


_KNOWN_METHOD_PROFILES: tuple[str, ...] = (
    "eu_cbam",
    "corporate_scope1",
    "corporate_scope2_location_based",
    "corporate_scope2_market_based",
    "corporate_scope3",
    "eudr_due_diligence",
    "india_ccts",
    "product_pcf",
)


# ---------------------------------------------------------------------------
# Context-var tracking "are we inside a policy workflow"
# ---------------------------------------------------------------------------


_IN_POLICY_WORKFLOW: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gl_factors_in_policy_workflow", default=False
)

# Stack of caller names so nested guards surface the closest policy
# workflow in error messages (e.g. "CBAM.EmissionsCalculatorAgent_v2.run").
_CALLER_STACK: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "gl_factors_caller_stack", default=tuple()
)


F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qualname(target: Any) -> str:
    """Best-effort fully-qualified name for telemetry / error messages."""
    if inspect.isclass(target) or inspect.isfunction(target) or inspect.ismethod(target):
        mod = getattr(target, "__module__", "")
        qn = getattr(target, "__qualname__", getattr(target, "__name__", repr(target)))
        return f"{mod}.{qn}" if mod else qn
    return type(target).__name__


def is_policy_workflow_caller(caller: Any) -> bool:
    """Return ``True`` when ``caller`` carries the policy-workflow marker.

    Accepts a class, a function, a bound method, or an instance. Falls
    back to checking the ``_IN_POLICY_WORKFLOW`` context variable when
    ``caller`` is ``None`` — this mirrors "are we inside a decorated
    entrypoint right now?".
    """
    if caller is None:
        return bool(_IN_POLICY_WORKFLOW.get())
    if getattr(caller, "_gl_policy_workflow", False):
        return True
    if getattr(caller, "__policy_workflow__", False):
        return True
    # For bound methods and instances, check the class too.
    cls = getattr(caller, "__class__", None)
    if cls is not None and cls is not type(caller):
        if getattr(cls, "_gl_policy_workflow", False):
            return True
        if getattr(cls, "__policy_workflow__", False):
            return True
    return False


def _extract_method_profile(request_or_kwargs: Any) -> Any:
    """Pull ``method_profile`` out of any of the shapes this guard accepts."""
    if request_or_kwargs is None:
        return None
    # 1. Raw value — caller passed the profile directly.
    if hasattr(request_or_kwargs, "value") and hasattr(request_or_kwargs, "name"):
        # Looks like an Enum (MethodProfile.*) — treat as already resolved.
        return request_or_kwargs
    if isinstance(request_or_kwargs, str):
        return request_or_kwargs
    # 2. Mapping (kwargs / extras dict).
    if isinstance(request_or_kwargs, dict):
        return request_or_kwargs.get("method_profile")
    # 3. Object with attribute.
    return getattr(request_or_kwargs, "method_profile", None)


def _is_present(method_profile: Any) -> bool:
    """Non-blank check covering strings, enums, and None."""
    if method_profile is None:
        return False
    if isinstance(method_profile, str):
        return bool(method_profile.strip())
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def require_method_profile(
    request_or_kwargs: Any = None,
    *,
    caller: Optional[Union[str, Any]] = None,
) -> None:
    """Raise :class:`MethodProfileMissingError` if we are inside a policy
    workflow and no ``method_profile`` has been supplied.

    Library / CLI callers — i.e. code NOT executing inside an
    ``@policy_workflow`` scope — are untouched. This is deliberate:
    Explorer search, developer SDK "give me metadata for this factor",
    and catalog-bootstrap scripts must continue to work without a
    method profile.

    Args:
        request_or_kwargs: The request / kwargs / value being inspected.
            Accepts a raw :class:`MethodProfile` / string, a ``dict`` of
            keyword arguments, or any object with a ``method_profile``
            attribute. ``None`` is treated as "absent".
        caller: Optional caller identifier — a string or an object from
            which we derive a qualified name. If omitted we fall back
            to the caller stack maintained by ``@policy_workflow``.
    """
    # If ``caller`` is a class / function / instance, treat it as the
    # authoritative marker even when the ContextVar isn't set. This
    # lets a plain function call ``require_method_profile(req, caller=self)``
    # outside the decorator.
    inside_policy_scope = bool(_IN_POLICY_WORKFLOW.get())
    caller_is_policy = is_policy_workflow_caller(caller) if caller is not None else False

    if not (inside_policy_scope or caller_is_policy):
        return

    method_profile = _extract_method_profile(request_or_kwargs)
    if _is_present(method_profile):
        return

    # Build a helpful caller label: explicit > stack > repr.
    if isinstance(caller, str):
        caller_name: Optional[str] = caller
    elif caller is not None:
        caller_name = _qualname(caller)
    else:
        stack = _CALLER_STACK.get()
        caller_name = stack[-1] if stack else None

    logger.warning(
        "N6 guard fired: method_profile missing for caller=%s", caller_name
    )
    raise MethodProfileMissingError(
        caller=caller_name, expected_profiles=_KNOWN_METHOD_PROFILES
    )


def _wrap_function(func: F, *, caller_name: str) -> F:
    """Return a wrapper that sets the policy-workflow context vars."""

    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        token_flag = _IN_POLICY_WORKFLOW.set(True)
        token_stack = _CALLER_STACK.set(_CALLER_STACK.get() + (caller_name,))
        try:
            return func(*args, **kwargs)
        finally:
            _CALLER_STACK.reset(token_stack)
            _IN_POLICY_WORKFLOW.reset(token_flag)

    # Preserve / set the marker on the wrapper itself.
    _wrapper._gl_policy_workflow = True  # type: ignore[attr-defined]
    _wrapper.__policy_workflow__ = True  # type: ignore[attr-defined]
    return _wrapper  # type: ignore[return-value]


def _wrap_class(cls: type) -> type:
    """Attach the marker to ``cls`` and auto-wrap any recognised
    entrypoint methods (``run``, ``process``, ``calculate``,
    ``calculate_batch``, ``__call__``) so calls through them flip the
    context var.  Methods that aren't present are silently skipped.
    """
    cls._gl_policy_workflow = True  # type: ignore[attr-defined]
    cls.__policy_workflow__ = True  # type: ignore[attr-defined]

    entrypoint_names = (
        "run",
        "process",
        "calculate",
        "calculate_batch",
        "calculate_metric",
        "__call__",
    )
    class_label = _qualname(cls)
    for name in entrypoint_names:
        method = cls.__dict__.get(name)
        if method is None or not callable(method):
            continue
        if getattr(method, "_gl_policy_workflow_wrapped", False):
            continue
        wrapped = _wrap_function(method, caller_name=f"{class_label}.{name}")
        wrapped._gl_policy_workflow_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, name, wrapped)
    return cls


def policy_workflow(target: Union[type, Callable[..., Any]]) -> Any:
    """Mark ``target`` as a policy-workflow entrypoint.

    Can decorate either a class or a function. In both cases the target
    gets ``_gl_policy_workflow = True`` and ``__policy_workflow__ = True``.

    When applied to a class, every recognised entrypoint method (``run``,
    ``process``, ``calculate``, ``calculate_batch``, ``calculate_metric``,
    ``__call__``) is wrapped so that nested calls to
    :func:`require_method_profile` inside those methods raise when the
    profile is missing.

    Example::

        @policy_workflow
        class EmissionsCalculatorAgent_v2(Agent[CalculatorInput, CalculatorOutput]):
            def run(self, input_data): ...

        @policy_workflow
        def cbam_batch_job(...):
            ...
    """
    if inspect.isclass(target):
        return _wrap_class(target)
    if callable(target):
        return _wrap_function(target, caller_name=_qualname(target))
    raise TypeError(
        f"@policy_workflow expected a class or callable, got {type(target).__name__}"
    )
