# -*- coding: utf-8 -*-
"""Framework adapter base protocol + registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from greenlang.scope_engine.models import Framework, FrameworkView, ScopeComputation


@runtime_checkable
class FrameworkAdapter(Protocol):
    framework: Framework

    def project(self, computation: ScopeComputation) -> FrameworkView: ...


_REGISTRY: dict[Framework, FrameworkAdapter] = {}


def register(adapter: FrameworkAdapter) -> None:
    _REGISTRY[adapter.framework] = adapter


def get(framework: Framework) -> FrameworkAdapter:
    try:
        return _REGISTRY[framework]
    except KeyError as e:
        raise ValueError(f"No framework adapter registered for {framework}") from e


def available() -> list[Framework]:
    return sorted(_REGISTRY.keys(), key=lambda f: f.value)
