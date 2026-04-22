# -*- coding: utf-8 -*-
"""Tests for the JWT_SECRET → GL_JWT_SECRET compatibility helper."""
from __future__ import annotations

import importlib
import logging

import pytest


def _reload_deps(monkeypatch: pytest.MonkeyPatch):
    """Reload dependencies so the module-level globals pick up new env."""
    import greenlang.integration.api.dependencies as deps
    # Reset the one-shot warning flag so successive tests each see it.
    deps._jwt_legacy_warned = False
    importlib.reload(deps)
    return deps


def test_canonical_name_preferred_when_both_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GL_JWT_SECRET", "x" * 48)
    monkeypatch.setenv("JWT_SECRET", "legacy-value-48-chars-............")
    deps = _reload_deps(monkeypatch)
    assert deps.get_jwt_secret() == "x" * 48


def test_legacy_name_works_alone_and_warns_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("GL_JWT_SECRET", raising=False)
    monkeypatch.setenv("JWT_SECRET", "legacy-48-chars-long-....................")

    # Bypass the module-reload dance (caplog doesn't attach cleanly while
    # Python re-imports the module).  Exercise the helper directly with
    # the warning flag reset to its pristine process-start state.
    import greenlang.integration.api.dependencies as deps

    # Attach caplog to the real module logger so we capture all output.
    caplog.set_level(logging.WARNING, logger=deps.logger.name)

    deps._jwt_legacy_warned = False
    first = deps.get_jwt_secret()
    second = deps.get_jwt_secret()
    third = deps.get_jwt_secret()

    assert first == second == third == "legacy-48-chars-long-...................."

    deprecation_lines = [
        r for r in caplog.records
        if "JWT_SECRET is deprecated" in r.getMessage()
    ]
    assert len(deprecation_lines) == 1, (
        "deprecation warning must fire exactly once per process; "
        f"saw {len(deprecation_lines)}"
    )


def test_empty_when_neither_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GL_JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    deps = _reload_deps(monkeypatch)
    assert deps.get_jwt_secret() == ""


def test_canonical_takes_precedence_even_if_legacy_longer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Canonical short, legacy very long — still prefer canonical.
    monkeypatch.setenv("GL_JWT_SECRET", "canonical-32-chars-long..........")
    monkeypatch.setenv("JWT_SECRET", "x" * 256)
    deps = _reload_deps(monkeypatch)
    assert deps.get_jwt_secret().startswith("canonical-")
