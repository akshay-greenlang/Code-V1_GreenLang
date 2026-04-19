# -*- coding: utf-8 -*-
"""GL-Comply-APP Python SDK clients (sync + async)."""

from __future__ import annotations

from typing import Any, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from schemas.models import (
    ApplicabilityRequest,
    ApplicabilityResult,
    ComplianceRequest,
    UnifiedComplianceReport,
)


DEFAULT_BASE_URL = "http://localhost:8080/api/v1"


def _require_httpx():
    if httpx is None:  # pragma: no cover
        raise RuntimeError("httpx is required for SDK; pip install httpx")


class ComplyClient:
    """Synchronous client for GL-Comply-APP."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        _require_httpx()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    def __enter__(self) -> "ComplyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def intake(self, request: ComplianceRequest) -> UnifiedComplianceReport:
        r = self._client.post("/compliance/intake", json=request.model_dump(mode="json"))
        r.raise_for_status()
        return UnifiedComplianceReport.model_validate(r.json())

    def get_job(self, job_id: str) -> UnifiedComplianceReport:
        r = self._client.get(f"/compliance/jobs/{job_id}")
        r.raise_for_status()
        return UnifiedComplianceReport.model_validate(r.json())

    def applicability(self, request: ApplicabilityRequest) -> ApplicabilityResult:
        r = self._client.post(
            "/compliance/applicability", json=request.model_dump(mode="json")
        )
        r.raise_for_status()
        return ApplicabilityResult.model_validate(r.json())

    def frameworks(self) -> list[str]:
        r = self._client.get("/compliance/frameworks")
        r.raise_for_status()
        return r.json().get("frameworks", [])


class AsyncComplyClient:
    """Asynchronous client for GL-Comply-APP."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        _require_httpx()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout
        )

    async def __aenter__(self) -> "AsyncComplyClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def intake(self, request: ComplianceRequest) -> UnifiedComplianceReport:
        r = await self._client.post(
            "/compliance/intake", json=request.model_dump(mode="json")
        )
        r.raise_for_status()
        return UnifiedComplianceReport.model_validate(r.json())

    async def get_job(self, job_id: str) -> UnifiedComplianceReport:
        r = await self._client.get(f"/compliance/jobs/{job_id}")
        r.raise_for_status()
        return UnifiedComplianceReport.model_validate(r.json())

    async def applicability(
        self, request: ApplicabilityRequest
    ) -> ApplicabilityResult:
        r = await self._client.post(
            "/compliance/applicability", json=request.model_dump(mode="json")
        )
        r.raise_for_status()
        return ApplicabilityResult.model_validate(r.json())

    async def frameworks(self) -> list[str]:
        r = await self._client.get("/compliance/frameworks")
        r.raise_for_status()
        return r.json().get("frameworks", [])
