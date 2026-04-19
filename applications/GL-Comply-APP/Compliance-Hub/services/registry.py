# -*- coding: utf-8 -*-
"""Framework adapter registry + FrameworkAdapter protocol.

Stub implementation. Real adapters will import each app's pipeline
(e.g. applications.GL_CSRD_APP.CSRD_Reporting_Platform.csrd_pipeline).
Populated in COMPLY-APP 2 (task #16).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from schemas.models import (
    ComplianceRequest,
    FrameworkEnum,
    FrameworkResult,
)


@runtime_checkable
class FrameworkAdapter(Protocol):
    framework: FrameworkEnum

    async def run(self, request: ComplianceRequest) -> FrameworkResult: ...


_REGISTRY: dict[FrameworkEnum, FrameworkAdapter] = {}


def register(adapter: FrameworkAdapter) -> None:
    _REGISTRY[adapter.framework] = adapter


def get(framework: FrameworkEnum) -> FrameworkAdapter:
    try:
        return _REGISTRY[framework]
    except KeyError as e:
        raise ValueError(f"No adapter registered for framework {framework}") from e


def available() -> list[FrameworkEnum]:
    return sorted(_REGISTRY.keys(), key=lambda f: f.value)
