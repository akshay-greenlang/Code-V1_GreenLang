# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.connectors.electricity_maps (F063)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from greenlang.factors.connectors.config import ConnectorConfig
from greenlang.factors.connectors.electricity_maps import (
    ElectricityMapsConnector,
    _ZONE_GEO_MAP,
)


def _mock_config() -> ConnectorConfig:
    return ConnectorConfig(
        source_id="electricity_maps",
        api_endpoint="https://mock-emap.test/v3",
        license_key="test-token",
        timeout_sec=10,
    )


def _api_response(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")


class TestElectricityMapsConnector:
    def test_source_id(self):
        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        assert c.source_id == "electricity_maps"

    def test_capabilities(self):
        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        cap = c.capabilities
        assert cap.supports_real_time is True
        assert cap.requires_license is True
        assert cap.typical_factor_count == 5000

    @patch("greenlang.factors.connectors.electricity_maps.urllib.request.urlopen")
    def test_fetch_metadata_zones(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = _api_response({
            "DE": {"zoneName": "Germany", "countryCode": "DE"},
            "FR": {"zoneName": "France", "countryCode": "FR"},
            "US-CAL-CISO": {"zoneName": "California", "countryCode": "US"},
        })
        resp.headers = MagicMock()
        resp.headers.get.return_value = None
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        meta = c.fetch_metadata()
        assert len(meta) == 3
        assert all(m["fuel_type"] == "electricity" for m in meta)
        assert all(m["scope"] == "2" for m in meta)
        assert all(m["real_time"] is True for m in meta)

    @patch("greenlang.factors.connectors.electricity_maps.urllib.request.urlopen")
    def test_fetch_factors(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = _api_response({
            "carbonIntensity": 350,
            "datetime": "2026-04-17T12:00:00Z",
        })
        resp.headers = MagicMock()
        resp.headers.get.return_value = "800"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        factors = c.fetch_factors(["EF:EMAP:grid_average:DE:latest:v1"])
        assert len(factors) == 1
        f = factors[0]
        assert f["co2e_total"] == 0.35  # 350 gCO2/kWh -> 0.35 kgCO2/kWh
        assert f["unit"] == "kg_co2e_kwh"
        assert f["original_value"] == 350.0
        assert f["original_unit"] == "g_co2e_kwh"
        assert f["real_time"] is True
        assert c.quota_remaining == 800

    @patch("greenlang.factors.connectors.electricity_maps.urllib.request.urlopen")
    def test_fetch_factors_empty(self, mock_urlopen):
        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        assert c.fetch_factors([]) == []

    @patch("greenlang.factors.connectors.electricity_maps.urllib.request.urlopen")
    def test_fetch_historical(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = _api_response({"history": [
            {"carbonIntensity": 300, "datetime": "2026-04-16T00:00:00Z"},
            {"carbonIntensity": 280, "datetime": "2026-04-16T01:00:00Z"},
            {"carbonIntensity": 260, "datetime": "2026-04-16T02:00:00Z"},
        ]})
        resp.headers = MagicMock()
        resp.headers.get.return_value = None
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        historical = c.fetch_historical("DE", "2026-04-16")
        assert len(historical) == 3
        assert historical[0]["hour"] == 0
        assert historical[1]["hour"] == 1
        assert historical[2]["co2e_total"] == 0.26

    @patch("greenlang.factors.connectors.electricity_maps.urllib.request.urlopen")
    def test_auth_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://mock", 401, "Unauthorized", {}, None,
        )
        c = ElectricityMapsConnector(license_key="bad", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorAuthError
        with pytest.raises(ConnectorAuthError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.electricity_maps.urllib.request.urlopen")
    def test_rate_limit(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://mock", 429, "Rate Limited", {}, None,
        )
        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorRateLimitError
        with pytest.raises(ConnectorRateLimitError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.electricity_maps.urllib.request.urlopen")
    def test_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("DNS failed")
        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorNetworkError
        with pytest.raises(ConnectorNetworkError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.electricity_maps.urllib.request.urlopen")
    def test_health_check_healthy(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = b'{"DE": {"zoneName": "Germany"}}'
        resp.headers = MagicMock()
        resp.headers.get.return_value = None
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        health = c.health_check()
        assert health.status.value == "healthy"

    @patch("greenlang.factors.connectors.electricity_maps.urllib.request.urlopen")
    def test_health_check_unavailable(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("timeout")
        c = ElectricityMapsConnector(license_key="k", config=_mock_config())
        health = c.health_check()
        assert health.status.value == "unavailable"

    def test_zone_geo_map(self):
        assert _ZONE_GEO_MAP["US-CAL-CISO"] == "US-CA"
        assert _ZONE_GEO_MAP["DE"] == "DE"
        assert _ZONE_GEO_MAP["GB"] == "GB"
