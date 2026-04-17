# -*- coding: utf-8 -*-
"""
IEA Statistics connector (F061).

Fetches country-level electricity grid emission factors and fuel-specific
factors from the IEA Data Services API.  Coverage: 150+ countries,
5+ fuel types, 10+ years => ~7,500-10,000 factors.

Factors are ``connector_only`` (Enterprise tier) and may not be
redistributed per IEA licensing terms.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.factors.connectors.base import (
    BaseConnector,
    ConnectorCapabilities,
    ConnectorHealthResult,
    ConnectorStatus,
)
from greenlang.factors.connectors.config import ConnectorConfig
from greenlang.factors.connectors.metrics import (
    record_factors_fetched,
    set_quota_remaining,
    track_connector_call,
)

logger = logging.getLogger(__name__)

# IEA fuel code -> GreenLang fuel_type mapping
_IEA_FUEL_MAP: Dict[str, str] = {
    "COAL": "coal",
    "NATGAS": "natural_gas",
    "OIL": "oil",
    "NUCLEAR": "nuclear",
    "HYDRO": "hydro",
    "WIND": "wind",
    "SOLAR": "solar",
    "BIOMASS": "biomass",
    "GEOTHERM": "geothermal",
    "OTHRENEW": "other_renewable",
    "TOTAL": "grid_average",
    "ELEC": "electricity",
}

# Default IEA API endpoint
_DEFAULT_ENDPOINT = "https://api.iea.org/v1"

# Scope: IEA grid factors are Scope 2 (electricity purchased)
_DEFAULT_SCOPE = "2"

# GWP set: IEA uses IPCC AR5
_DEFAULT_GWP = "AR5"


class IEAConnector(BaseConnector):
    """
    Connector to IEA Data Services API for emission factors.

    Provides grid-average and fuel-specific CO2 emission factors for
    electricity generation across 150+ countries.
    """

    def __init__(
        self,
        *,
        license_key: Optional[str] = None,
        config: Optional[ConnectorConfig] = None,
    ) -> None:
        super().__init__(license_key=license_key)
        self._config = config or ConnectorConfig.from_env("iea")
        if not self._license_key and self._config.license_key:
            self._license_key = self._config.license_key
        self._endpoint = self._config.api_endpoint or _DEFAULT_ENDPOINT
        self._quota_remaining: Optional[int] = None

    @property
    def source_id(self) -> str:
        return "iea_statistics"

    @property
    def capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities(
            supports_metadata_only=True,
            supports_batch_fetch=True,
            supports_incremental=True,
            supports_real_time=False,
            max_batch_size=self._config.batch_size,
            requires_license=True,
            license_class="commercial_connector",
            typical_factor_count=10000,
        )

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: Optional[bytes] = None,
        license_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to the IEA API."""
        key = self.get_license_key(license_key)
        url = f"{self._endpoint.rstrip('/')}/{path.lstrip('/')}"

        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "GreenLang-Factors/1.0",
        }
        if body:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        start = time.monotonic()

        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_sec) as resp:  # nosec B310
                elapsed = time.monotonic() - start
                self._track_request(success=True)

                # Track quota from response headers
                quota_header = resp.headers.get("X-RateLimit-Remaining")
                if quota_header:
                    self._quota_remaining = int(quota_header)
                    set_quota_remaining(self.source_id, self._quota_remaining)

                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as exc:
            self._track_request(success=False)
            if exc.code == 401:
                from greenlang.exceptions.connector import ConnectorAuthError
                raise ConnectorAuthError(
                    "IEA API authentication failed (invalid license key)",
                    connector_name=self.source_id,
                    status_code=401,
                )
            if exc.code == 429:
                from greenlang.exceptions.connector import ConnectorRateLimitError
                retry_after = int(exc.headers.get("Retry-After", "60") if exc.headers else "60")
                raise ConnectorRateLimitError(
                    "IEA API rate limit exceeded",
                    connector_name=self.source_id,
                    retry_after_seconds=retry_after,
                )
            from greenlang.exceptions.connector import ConnectorServerError
            raise ConnectorServerError(
                f"IEA API returned HTTP {exc.code}: {exc.reason}",
                connector_name=self.source_id,
                status_code=exc.code,
            )
        except OSError as exc:
            self._track_request(success=False)
            from greenlang.exceptions.connector import ConnectorNetworkError
            raise ConnectorNetworkError(
                f"IEA API network error: {exc}",
                connector_name=self.source_id,
                original_error=exc,
            )

    def fetch_metadata(self) -> List[Dict[str, Any]]:
        """
        Fetch IEA factor metadata (countries, fuels, years).

        Returns factor stubs without actual values — safe to call
        without checking license quota.
        """
        with track_connector_call(self.source_id, "fetch_metadata"):
            data = self._request("emissions/electricity/metadata")

        records = data.get("records") or data.get("data") or []
        result: List[Dict[str, Any]] = []
        for rec in records:
            country = rec.get("country_code", rec.get("iso3", ""))
            fuel = rec.get("fuel_code", rec.get("product", "TOTAL"))
            year = rec.get("year", rec.get("time_period", ""))
            gl_fuel = _IEA_FUEL_MAP.get(fuel, fuel.lower())
            factor_id = f"EF:IEA:{gl_fuel}:{country}:{year}:v1"
            result.append({
                "factor_id": factor_id,
                "source_id": self.source_id,
                "fuel_type": gl_fuel,
                "geography": country,
                "year": str(year),
                "scope": _DEFAULT_SCOPE,
                "factor_status": "connector_only",
                "license_class": "commercial_connector",
            })

        logger.info("IEA metadata: %d factor stubs fetched", len(result))
        return result

    def fetch_factors(
        self,
        factor_ids: List[str],
        *,
        license_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch full IEA emission factors with values.

        Requires a valid IEA license key.
        """
        if not factor_ids:
            return []

        with track_connector_call(self.source_id, "fetch_factors"):
            payload = json.dumps({"factor_ids": factor_ids}).encode("utf-8")
            data = self._request(
                "emissions/electricity/factors",
                method="POST",
                body=payload,
                license_key=license_key,
            )

        records = data.get("records") or data.get("data") or []
        result: List[Dict[str, Any]] = []
        for rec in records:
            country = rec.get("country_code", "")
            fuel = rec.get("fuel_code", "TOTAL")
            year = rec.get("year", "")
            value = rec.get("emission_factor") or rec.get("value")
            unit = rec.get("unit", "kg_co2e_kwh")
            gl_fuel = _IEA_FUEL_MAP.get(fuel, fuel.lower())

            factor = {
                "factor_id": f"EF:IEA:{gl_fuel}:{country}:{year}:v1",
                "source_id": self.source_id,
                "fuel_type": gl_fuel,
                "geography": country,
                "year": str(year),
                "scope": _DEFAULT_SCOPE,
                "co2e_total": float(value) if value is not None else None,
                "unit": unit,
                "gwp_set": _DEFAULT_GWP,
                "factor_status": "connector_only",
                "license_class": "commercial_connector",
                "redistribution_allowed": False,
                "dqs": {"temporal": 5, "geographical": 4, "technological": 4, "representativeness": 4, "methodological": 5},
            }
            result.append(factor)

        record_factors_fetched(self.source_id, len(result))
        logger.info("IEA factors: %d fetched for %d requested", len(result), len(factor_ids))
        return result

    def health_check(self) -> ConnectorHealthResult:
        """Check IEA API connectivity."""
        start = time.monotonic()
        try:
            self._request("health", license_key=self._license_key or "health-check")
            latency = int((time.monotonic() - start) * 1000)
            result = ConnectorHealthResult(
                status=ConnectorStatus.HEALTHY,
                latency_ms=latency,
                message="IEA API responding",
                checked_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            latency = int((time.monotonic() - start) * 1000)
            result = ConnectorHealthResult(
                status=ConnectorStatus.UNAVAILABLE,
                latency_ms=latency,
                message=str(exc),
                checked_at=datetime.now(timezone.utc).isoformat(),
            )
        self._last_health = result
        return result

    @property
    def quota_remaining(self) -> Optional[int]:
        return self._quota_remaining
