# -*- coding: utf-8 -*-
"""
Electricity Maps connector (F063).

Fetches real-time and historical grid carbon intensity data from
Electricity Maps API.  Coverage: 200+ electricity zones globally.

Unique: factors change hourly/daily (unlike static annual factors).
Provides both average and marginal carbon intensity.
"""

from __future__ import annotations

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

_DEFAULT_ENDPOINT = "https://api.electricitymaps.com/v3"

# Zone code -> GreenLang geography mapping (subset)
_ZONE_GEO_MAP: Dict[str, str] = {
    "US-CAL-CISO": "US-CA",
    "US-NY-NYIS": "US-NY",
    "US-TEX-ERCO": "US-TX",
    "US-MIDA-PJM": "US-PJM",
    "US-NW-PACW": "US-NW",
    "DE": "DE",
    "FR": "FR",
    "GB": "GB",
    "ES": "ES",
    "IT": "IT",
    "NL": "NL",
    "SE": "SE",
    "NO": "NO",
    "DK-DK1": "DK",
    "JP-TK": "JP",
    "AU-NSW": "AU-NSW",
    "AU-VIC": "AU-VIC",
    "IN-WE": "IN-WE",
    "IN-NO": "IN-NO",
    "CN-NE": "CN-NE",
    "BR-S": "BR-S",
}


class ElectricityMapsConnector(BaseConnector):
    """
    Connector to Electricity Maps API for real-time grid carbon intensity.

    Provides:
    - Current carbon intensity by electricity zone
    - Historical hourly/daily averages
    - Average vs marginal intensity
    """

    def __init__(
        self,
        *,
        license_key: Optional[str] = None,
        config: Optional[ConnectorConfig] = None,
    ) -> None:
        super().__init__(license_key=license_key)
        self._config = config or ConnectorConfig.from_env("electricity_maps")
        if not self._license_key and self._config.license_key:
            self._license_key = self._config.license_key
        self._endpoint = self._config.api_endpoint or _DEFAULT_ENDPOINT
        self._quota_remaining: Optional[int] = None

    @property
    def source_id(self) -> str:
        return "electricity_maps"

    @property
    def capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities(
            supports_metadata_only=True,
            supports_batch_fetch=True,
            supports_incremental=True,
            supports_real_time=True,
            max_batch_size=50,  # Zone-level queries
            requires_license=True,
            license_class="commercial_connector",
            typical_factor_count=5000,
        )

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        license_key: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to Electricity Maps API."""
        key = self.get_license_key(license_key)
        url = f"{self._endpoint.rstrip('/')}/{path.lstrip('/')}"

        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"

        headers = {
            "auth-token": key,
            "Accept": "application/json",
            "User-Agent": "GreenLang-Factors/1.0",
        }

        req = urllib.request.Request(url, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_sec) as resp:  # nosec B310
                self._track_request(success=True)

                quota = resp.headers.get("X-RateLimit-Remaining")
                if quota:
                    self._quota_remaining = int(quota)
                    set_quota_remaining(self.source_id, self._quota_remaining)

                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self._track_request(success=False)
            if exc.code == 401:
                from greenlang.exceptions.connector import ConnectorAuthError
                raise ConnectorAuthError(
                    "Electricity Maps authentication failed",
                    connector_name=self.source_id,
                    status_code=401,
                )
            if exc.code == 429:
                from greenlang.exceptions.connector import ConnectorRateLimitError
                raise ConnectorRateLimitError(
                    "Electricity Maps rate limit exceeded",
                    connector_name=self.source_id,
                )
            from greenlang.exceptions.connector import ConnectorServerError
            raise ConnectorServerError(
                f"Electricity Maps API returned HTTP {exc.code}",
                connector_name=self.source_id,
                status_code=exc.code,
            )
        except OSError as exc:
            self._track_request(success=False)
            from greenlang.exceptions.connector import ConnectorNetworkError
            raise ConnectorNetworkError(
                f"Electricity Maps network error: {exc}",
                connector_name=self.source_id,
                original_error=exc,
            )

    def fetch_metadata(self) -> List[Dict[str, Any]]:
        """
        Fetch available electricity zones.

        Returns metadata stubs for all known zones without actual values.
        """
        with track_connector_call(self.source_id, "fetch_metadata"):
            data = self._request("zones")

        zones = data if isinstance(data, dict) else {}
        result: List[Dict[str, Any]] = []

        for zone_key, zone_info in zones.items():
            if isinstance(zone_info, dict):
                display_name = zone_info.get("zoneName", zone_key)
                country_code = zone_info.get("countryCode", zone_key[:2])
            else:
                display_name = zone_key
                country_code = zone_key[:2]

            geo = _ZONE_GEO_MAP.get(zone_key, country_code)
            factor_id = f"EF:EMAP:grid_average:{zone_key}:latest:v1"

            result.append({
                "factor_id": factor_id,
                "source_id": self.source_id,
                "zone_key": zone_key,
                "display_name": display_name,
                "geography": geo,
                "fuel_type": "electricity",
                "scope": "2",
                "factor_status": "connector_only",
                "license_class": "commercial_connector",
                "real_time": True,
            })

        logger.info("Electricity Maps metadata: %d zones", len(result))
        return result

    def fetch_factors(
        self,
        factor_ids: List[str],
        *,
        license_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch current carbon intensity for the requested zones.

        Extracts zone_key from factor_ids and queries the latest
        carbon intensity for each zone.
        """
        if not factor_ids:
            return []

        # Extract zone_keys from factor_ids
        zone_keys = []
        for fid in factor_ids:
            parts = fid.split(":")
            if len(parts) >= 5:
                zone_keys.append(parts[3])  # EF:EMAP:grid_average:{zone_key}:...
            else:
                zone_keys.append(fid)

        results: List[Dict[str, Any]] = []
        for zone_key in zone_keys:
            try:
                with track_connector_call(self.source_id, "fetch_factors"):
                    data = self._request(
                        "carbon-intensity/latest",
                        params={"zone": zone_key},
                        license_key=license_key,
                    )

                ci = data.get("carbonIntensity") or data.get("data", {}).get("carbonIntensity")
                if ci is None:
                    logger.warning("No carbon intensity data for zone=%s", zone_key)
                    continue

                geo = _ZONE_GEO_MAP.get(zone_key, zone_key[:2])
                ts = data.get("datetime") or data.get("updatedAt", "")
                date_part = ts[:10] if ts else datetime.now(timezone.utc).strftime("%Y-%m-%d")

                factor = {
                    "factor_id": f"EF:EMAP:grid_average:{zone_key}:{date_part}:v1",
                    "source_id": self.source_id,
                    "zone_key": zone_key,
                    "geography": geo,
                    "fuel_type": "electricity",
                    "scope": "2",
                    "co2e_total": float(ci) / 1000.0,  # gCO2/kWh -> kgCO2/kWh
                    "unit": "kg_co2e_kwh",
                    "original_value": float(ci),
                    "original_unit": "g_co2e_kwh",
                    "gwp_set": "AR5",
                    "factor_status": "connector_only",
                    "license_class": "commercial_connector",
                    "redistribution_allowed": False,
                    "timestamp": ts,
                    "real_time": True,
                    "dqs": {"temporal": 5, "geographical": 5, "technological": 4, "representativeness": 4, "methodological": 4},
                }
                results.append(factor)
            except Exception:
                logger.exception("Failed to fetch carbon intensity for zone=%s", zone_key)

        record_factors_fetched(self.source_id, len(results))
        logger.info(
            "Electricity Maps: %d zone factors fetched for %d requested",
            len(results), len(zone_keys),
        )
        return results

    def fetch_historical(
        self,
        zone_key: str,
        date: str,
        *,
        license_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical hourly carbon intensity for a zone on a specific date.

        Returns up to 24 hourly factor records for the given date.
        """
        with track_connector_call(self.source_id, "fetch_historical"):
            data = self._request(
                "carbon-intensity/history",
                params={"zone": zone_key, "datetime": f"{date}T00:00:00Z"},
                license_key=license_key,
            )

        history = data.get("history") or data.get("data") or []
        geo = _ZONE_GEO_MAP.get(zone_key, zone_key[:2])
        results: List[Dict[str, Any]] = []

        for entry in history:
            ci = entry.get("carbonIntensity")
            ts = entry.get("datetime", "")
            if ci is None:
                continue

            hour = ts[11:13] if len(ts) > 13 else "00"
            results.append({
                "factor_id": f"EF:EMAP:grid_average:{zone_key}:{date}:h{hour}:v1",
                "source_id": self.source_id,
                "zone_key": zone_key,
                "geography": geo,
                "fuel_type": "electricity",
                "scope": "2",
                "co2e_total": float(ci) / 1000.0,
                "unit": "kg_co2e_kwh",
                "original_value": float(ci),
                "original_unit": "g_co2e_kwh",
                "timestamp": ts,
                "hour": int(hour),
            })

        record_factors_fetched(self.source_id, len(results))
        return results

    def health_check(self) -> ConnectorHealthResult:
        """Check Electricity Maps API connectivity."""
        start = time.monotonic()
        try:
            self._request("zones", license_key=self._license_key or "health")
            latency = int((time.monotonic() - start) * 1000)
            result = ConnectorHealthResult(
                status=ConnectorStatus.HEALTHY,
                latency_ms=latency,
                message="Electricity Maps API responding",
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
