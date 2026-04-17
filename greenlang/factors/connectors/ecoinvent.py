# -*- coding: utf-8 -*-
"""
ecoinvent LCA database connector (F062).

Fetches life-cycle-based emission factors from ecoinvent v3.x API.
Extracts GWP (Global Warming Potential) indicators from full LCA results
and normalizes them to GreenLang EmissionFactorRecord format.

Coverage: ~10,000-20,000 processes relevant to GHG reporting.
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
    track_connector_call,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.ecoinvent.org/v3"
_DEFAULT_DB_VERSION = "3.10"

# Allocation methods supported by ecoinvent
ALLOCATION_CUTOFF = "cut-off"
ALLOCATION_APOS = "apos"
ALLOCATION_CONSEQUENTIAL = "consequential"

# ecoinvent activity categories relevant to GHG reporting
_GHG_RELEVANT_CATEGORIES = frozenset({
    "electricity production",
    "heat production",
    "transport",
    "fuel production",
    "cement production",
    "steel production",
    "aluminium production",
    "chemical production",
    "waste treatment",
    "recycling",
    "agriculture",
    "mining",
})

# Geography normalization
_ECOINVENT_GEO_MAP: Dict[str, str] = {
    "GLO": "GLOBAL",
    "RoW": "ROW",
    "RER": "EU",
    "RNA": "NA",
    "RAS": "ASIA",
    "RAF": "AFRICA",
    "RLA": "LATAM",
}


class EcoinventConnector(BaseConnector):
    """
    Connector to ecoinvent LCA database API.

    Extracts GWP-based emission factors from full LCA calculation results.
    Supports cut-off, APOS, and consequential allocation methods.
    """

    def __init__(
        self,
        *,
        license_key: Optional[str] = None,
        config: Optional[ConnectorConfig] = None,
        allocation_method: str = ALLOCATION_CUTOFF,
        db_version: str = _DEFAULT_DB_VERSION,
    ) -> None:
        super().__init__(license_key=license_key)
        self._config = config or ConnectorConfig.from_env("ecoinvent")
        if not self._license_key and self._config.license_key:
            self._license_key = self._config.license_key
        self._endpoint = self._config.api_endpoint or _DEFAULT_ENDPOINT
        self._allocation = allocation_method
        self._db_version = db_version

    @property
    def source_id(self) -> str:
        return "ecoinvent"

    @property
    def capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities(
            supports_metadata_only=True,
            supports_batch_fetch=True,
            supports_incremental=False,
            supports_real_time=False,
            max_batch_size=100,  # ecoinvent LCA calcs are heavier
            requires_license=True,
            license_class="commercial_connector",
            typical_factor_count=15000,
        )

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: Optional[bytes] = None,
        license_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to ecoinvent API."""
        key = self.get_license_key(license_key)
        url = f"{self._endpoint.rstrip('/')}/{path.lstrip('/')}"

        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "GreenLang-Factors/1.0",
            "X-ecoinvent-Version": self._db_version,
        }
        if body:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_sec) as resp:  # nosec B310
                self._track_request(success=True)
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self._track_request(success=False)
            if exc.code == 401:
                from greenlang.exceptions.connector import ConnectorAuthError
                raise ConnectorAuthError(
                    "ecoinvent API authentication failed",
                    connector_name=self.source_id,
                    status_code=401,
                )
            if exc.code == 429:
                from greenlang.exceptions.connector import ConnectorRateLimitError
                raise ConnectorRateLimitError(
                    "ecoinvent API rate limit exceeded",
                    connector_name=self.source_id,
                )
            from greenlang.exceptions.connector import ConnectorServerError
            raise ConnectorServerError(
                f"ecoinvent API returned HTTP {exc.code}",
                connector_name=self.source_id,
                status_code=exc.code,
            )
        except OSError as exc:
            self._track_request(success=False)
            from greenlang.exceptions.connector import ConnectorNetworkError
            raise ConnectorNetworkError(
                f"ecoinvent API network error: {exc}",
                connector_name=self.source_id,
                original_error=exc,
            )

    def fetch_metadata(self) -> List[Dict[str, Any]]:
        """
        Fetch ecoinvent process metadata (activities, geographies).

        Returns factor stubs filtered to GHG-relevant categories.
        """
        with track_connector_call(self.source_id, "fetch_metadata"):
            data = self._request(
                f"activities?version={self._db_version}"
                f"&allocation={self._allocation}"
            )

        activities = data.get("activities") or data.get("data") or []
        result: List[Dict[str, Any]] = []

        for act in activities:
            activity_name = act.get("activity_name", "").lower()
            # Filter to GHG-relevant categories
            if not any(cat in activity_name for cat in _GHG_RELEVANT_CATEGORIES):
                continue

            activity_id = act.get("activity_id", act.get("id", ""))
            geo = act.get("geography", "GLO")
            geo_norm = _ECOINVENT_GEO_MAP.get(geo, geo)
            ref_product = act.get("reference_product", "")
            unit = act.get("unit", "kg")

            factor_id = f"EF:ECI:{activity_id}:{geo}:v{self._db_version.replace('.', '')}"
            result.append({
                "factor_id": factor_id,
                "source_id": self.source_id,
                "activity_name": act.get("activity_name", ""),
                "reference_product": ref_product,
                "geography": geo_norm,
                "functional_unit": unit,
                "allocation_method": self._allocation,
                "db_version": self._db_version,
                "factor_status": "connector_only",
                "license_class": "commercial_connector",
            })

        logger.info("ecoinvent metadata: %d GHG-relevant activity stubs", len(result))
        return result

    def fetch_factors(
        self,
        factor_ids: List[str],
        *,
        license_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch LCA results and extract GWP emission factors.

        Performs batch LCIA calculation for the requested activities
        and extracts the GWP indicator (kg CO2e per functional unit).
        """
        if not factor_ids:
            return []

        # Extract activity_ids from factor_ids
        activity_ids = []
        for fid in factor_ids:
            parts = fid.split(":")
            if len(parts) >= 4:
                activity_ids.append(parts[2])  # EF:ECI:{activity_id}:...
            else:
                activity_ids.append(fid)

        with track_connector_call(self.source_id, "fetch_factors"):
            payload = json.dumps({
                "activity_ids": activity_ids,
                "allocation": self._allocation,
                "version": self._db_version,
                "lcia_methods": ["IPCC 2021 GWP100"],
            }).encode("utf-8")
            data = self._request(
                "calculate/batch",
                method="POST",
                body=payload,
                license_key=license_key,
            )

        results_list = data.get("results") or data.get("data") or []
        factors: List[Dict[str, Any]] = []

        for res in results_list:
            activity_id = res.get("activity_id", "")
            geo = res.get("geography", "GLO")
            geo_norm = _ECOINVENT_GEO_MAP.get(geo, geo)

            # Extract GWP from LCIA results
            lcia = res.get("lcia_results") or {}
            gwp_result = lcia.get("IPCC 2021 GWP100") or lcia.get("climate change") or {}
            gwp_value = gwp_result.get("value") or gwp_result.get("amount")
            gwp_unit = gwp_result.get("unit", "kg CO2-eq")

            # Uncertainty from ecoinvent pedigree matrix
            uncertainty = res.get("uncertainty") or {}
            ci_95 = uncertainty.get("ci_95") or uncertainty.get("variance")

            factor_id = f"EF:ECI:{activity_id}:{geo}:v{self._db_version.replace('.', '')}"
            factor = {
                "factor_id": factor_id,
                "source_id": self.source_id,
                "activity_name": res.get("activity_name", ""),
                "reference_product": res.get("reference_product", ""),
                "geography": geo_norm,
                "scope": "3",  # LCA-based factors are typically Scope 3
                "co2e_total": float(gwp_value) if gwp_value is not None else None,
                "unit": "kg_co2e",
                "original_unit": gwp_unit,
                "functional_unit": res.get("unit", "kg"),
                "gwp_set": "AR6",  # IPCC 2021 = AR6
                "allocation_method": self._allocation,
                "db_version": self._db_version,
                "factor_status": "connector_only",
                "license_class": "commercial_connector",
                "redistribution_allowed": False,
                "uncertainty_95ci": float(ci_95) if ci_95 is not None else None,
                "dqs": {
                    "temporal": 4,
                    "geographical": 4 if geo != "GLO" else 2,
                    "technological": 4,
                    "representativeness": 3,
                    "methodological": 5,
                },
            }
            factors.append(factor)

        record_factors_fetched(self.source_id, len(factors))
        logger.info(
            "ecoinvent factors: %d calculated for %d activities",
            len(factors), len(activity_ids),
        )
        return factors

    def health_check(self) -> ConnectorHealthResult:
        """Check ecoinvent API connectivity."""
        start = time.monotonic()
        try:
            self._request("status", license_key=self._license_key or "health")
            latency = int((time.monotonic() - start) * 1000)
            result = ConnectorHealthResult(
                status=ConnectorStatus.HEALTHY,
                latency_ms=latency,
                message="ecoinvent API responding",
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
