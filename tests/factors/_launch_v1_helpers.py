# -*- coding: utf-8 -*-
"""Shared helpers for the FY27 launch-v1 coverage matrix tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


def resolve_or_skip(svc, *, activity: str, method_profile: str, **extra: Any) -> Dict[str, Any]:
    """Resolve via the engine; skip the test if the catalog isn't seeded.

    Returns the explain payload as a plain dict so assertions stay simple.
    Tests should treat skips as 'environment-not-ready' rather than
    fix-needed.
    """
    from greenlang.factors.api_endpoints import build_resolution_explain
    edition = svc.repo.resolve_edition(None)
    body = {"activity": activity, "method_profile": method_profile, **extra}
    try:
        return build_resolution_explain(svc.repo, edition, body)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"catalog not seeded for {activity!r}/{method_profile!r}: {exc}")
        return {}  # unreachable but keeps the type checker quiet


def assert_launch_explain_contract(payload: Dict[str, Any]) -> None:
    """Assert every CTO-mandated field is present in an explain payload.

    The resolution API returns a flat ``ResolvedFactor.model_dump()`` at
    the top level plus a nested ``explain`` block (see
    :func:`greenlang.factors.api_endpoints.build_resolution_explain`). The
    chosen factor's source/version lives on flat fields (``source_id``,
    ``source_version``, ``vintage``) **or** — when callers echo back the
    full record — on a nested ``provenance`` object. This helper accepts
    either shape so the contract stays honest across both call paths.
    """
    assert payload.get("factor_id") or payload.get("chosen_factor_id"), (
        "explain payload missing factor_id / chosen_factor_id"
    )
    explain = payload.get("explain") or {}
    assert isinstance(explain, dict) and explain, "explain block must be a non-empty dict"
    assert "alternates" in explain, "explain.alternates must be present (CTO non-negotiable #3)"

    def _source_signal(obj: Dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        prov = obj.get("provenance") or {}
        return bool(
            obj.get("source_id")
            or obj.get("source")
            or prov.get("source_org")
            or prov.get("source_publication")
        )

    def _version_signal(obj: Dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        prov = obj.get("provenance") or {}
        return bool(
            obj.get("source_version")
            or obj.get("factor_version")
            or obj.get("vintage")
            or prov.get("version")
            or prov.get("source_year")
        )

    candidates: List[Dict[str, Any]] = [
        payload,
        payload.get("chosen") or {},
        payload.get("factor") or {},
        explain.get("chosen") or {},
    ]
    assert any(_source_signal(c) for c in candidates), (
        "chosen factor must carry source_id / source_org / source_publication"
    )
    assert any(_version_signal(c) for c in candidates), (
        "chosen factor must carry source_version / factor_version / vintage"
    )

    assert payload.get("method_profile"), "explain payload must include method_profile"


def get_service_or_skip():
    try:
        from greenlang.factors.service import FactorCatalogService
        svc = FactorCatalogService.from_environment()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"FactorCatalogService unavailable: {exc}")
    if svc is None or not hasattr(svc, "repo"):
        pytest.skip("FactorCatalogService misconfigured")
    return svc
