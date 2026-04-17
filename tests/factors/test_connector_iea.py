# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.connectors.iea (F061)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from greenlang.factors.connectors.config import ConnectorConfig
from greenlang.factors.connectors.iea import IEAConnector, _IEA_FUEL_MAP


def _mock_config() -> ConnectorConfig:
    return ConnectorConfig(
        source_id="iea",
        api_endpoint="https://mock-iea.test/v1",
        license_key="test-key-123",
        timeout_sec=10,
    )


def _make_api_response(records: list) -> bytes:
    return json.dumps({"records": records}).encode("utf-8")


class TestIEAConnector:
    def test_source_id(self):
        c = IEAConnector(license_key="k", config=_mock_config())
        assert c.source_id == "iea_statistics"

    def test_capabilities(self):
        c = IEAConnector(license_key="k", config=_mock_config())
        cap = c.capabilities
        assert cap.requires_license is True
        assert cap.supports_batch_fetch is True
        assert cap.typical_factor_count == 10000

    @patch("greenlang.factors.connectors.iea.urllib.request.urlopen")
    def test_fetch_metadata(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = _make_api_response([
            {"country_code": "US", "fuel_code": "COAL", "year": 2024},
            {"country_code": "DE", "fuel_code": "NATGAS", "year": 2024},
            {"country_code": "FR", "fuel_code": "TOTAL", "year": 2023},
        ])
        resp.headers = MagicMock()
        resp.headers.get.return_value = None
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = IEAConnector(license_key="test-key", config=_mock_config())
        meta = c.fetch_metadata()
        assert len(meta) == 3
        assert meta[0]["factor_id"] == "EF:IEA:coal:US:2024:v1"
        assert meta[0]["fuel_type"] == "coal"
        assert meta[1]["fuel_type"] == "natural_gas"
        assert meta[2]["fuel_type"] == "grid_average"
        assert all(m["factor_status"] == "connector_only" for m in meta)

    @patch("greenlang.factors.connectors.iea.urllib.request.urlopen")
    def test_fetch_factors(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = _make_api_response([
            {"country_code": "US", "fuel_code": "COAL", "year": 2024,
             "emission_factor": 0.95, "unit": "kg_co2e_kwh"},
        ])
        resp.headers = MagicMock()
        resp.headers.get.return_value = "950"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = IEAConnector(license_key="test-key", config=_mock_config())
        factors = c.fetch_factors(["EF:IEA:coal:US:2024:v1"])
        assert len(factors) == 1
        assert factors[0]["co2e_total"] == 0.95
        assert factors[0]["unit"] == "kg_co2e_kwh"
        assert factors[0]["redistribution_allowed"] is False
        assert c.quota_remaining == 950

    @patch("greenlang.factors.connectors.iea.urllib.request.urlopen")
    def test_fetch_factors_empty(self, mock_urlopen):
        c = IEAConnector(license_key="k", config=_mock_config())
        assert c.fetch_factors([]) == []

    @patch("greenlang.factors.connectors.iea.urllib.request.urlopen")
    def test_auth_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://mock", 401, "Unauthorized", {}, None,
        )
        c = IEAConnector(license_key="bad-key", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorAuthError
        with pytest.raises(ConnectorAuthError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.iea.urllib.request.urlopen")
    def test_rate_limit_error(self, mock_urlopen):
        import urllib.error
        headers = MagicMock()
        headers.get.return_value = "30"
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://mock", 429, "Too Many Requests", headers, None,
        )
        c = IEAConnector(license_key="k", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorRateLimitError
        with pytest.raises(ConnectorRateLimitError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.iea.urllib.request.urlopen")
    def test_server_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://mock", 500, "Internal Server Error", {}, None,
        )
        c = IEAConnector(license_key="k", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorServerError
        with pytest.raises(ConnectorServerError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.iea.urllib.request.urlopen")
    def test_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("Connection refused")
        c = IEAConnector(license_key="k", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorNetworkError
        with pytest.raises(ConnectorNetworkError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.iea.urllib.request.urlopen")
    def test_health_check_healthy(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = b'{"status": "ok"}'
        resp.headers = MagicMock()
        resp.headers.get.return_value = None
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = IEAConnector(license_key="k", config=_mock_config())
        health = c.health_check()
        assert health.status.value == "healthy"

    @patch("greenlang.factors.connectors.iea.urllib.request.urlopen")
    def test_health_check_unavailable(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("timeout")
        c = IEAConnector(license_key="k", config=_mock_config())
        health = c.health_check()
        assert health.status.value == "unavailable"

    def test_fuel_map_coverage(self):
        assert "COAL" in _IEA_FUEL_MAP
        assert "NATGAS" in _IEA_FUEL_MAP
        assert "ELEC" in _IEA_FUEL_MAP
        assert _IEA_FUEL_MAP["TOTAL"] == "grid_average"
