# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.connectors.ecoinvent (F062)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from greenlang.factors.connectors.config import ConnectorConfig
from greenlang.factors.connectors.ecoinvent import (
    ALLOCATION_APOS,
    ALLOCATION_CUTOFF,
    EcoinventConnector,
    _ECOINVENT_GEO_MAP,
    _GHG_RELEVANT_CATEGORIES,
)


def _mock_config() -> ConnectorConfig:
    return ConnectorConfig(
        source_id="ecoinvent",
        api_endpoint="https://mock-ecoinvent.test/v3",
        license_key="test-key",
        timeout_sec=60,
        batch_size=100,
    )


def _api_response(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")


class TestEcoinventConnector:
    def test_source_id(self):
        c = EcoinventConnector(license_key="k", config=_mock_config())
        assert c.source_id == "ecoinvent"

    def test_capabilities(self):
        c = EcoinventConnector(license_key="k", config=_mock_config())
        cap = c.capabilities
        assert cap.requires_license is True
        assert cap.max_batch_size == 100
        assert cap.typical_factor_count == 15000

    @patch("greenlang.factors.connectors.ecoinvent.urllib.request.urlopen")
    def test_fetch_metadata(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = _api_response({"activities": [
            {"activity_id": "act1", "activity_name": "Electricity production, coal",
             "geography": "US", "reference_product": "electricity", "unit": "kWh"},
            {"activity_id": "act2", "activity_name": "Steel production, blast furnace",
             "geography": "DE", "reference_product": "steel", "unit": "kg"},
            {"activity_id": "act3", "activity_name": "Knitting, polyester",
             "geography": "CN", "reference_product": "fabric", "unit": "kg"},
        ]})
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = EcoinventConnector(license_key="k", config=_mock_config())
        meta = c.fetch_metadata()
        # Should include electricity + steel but NOT knitting (not GHG-relevant)
        assert len(meta) == 2
        assert any("act1" in m["factor_id"] for m in meta)
        assert any("act2" in m["factor_id"] for m in meta)
        assert all(m["factor_status"] == "connector_only" for m in meta)

    @patch("greenlang.factors.connectors.ecoinvent.urllib.request.urlopen")
    def test_fetch_factors(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = _api_response({"results": [
            {
                "activity_id": "act1",
                "activity_name": "Electricity production",
                "geography": "US",
                "reference_product": "electricity",
                "unit": "kWh",
                "lcia_results": {
                    "IPCC 2021 GWP100": {"value": 0.85, "unit": "kg CO2-eq"},
                },
                "uncertainty": {"ci_95": 0.12},
            },
        ]})
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = EcoinventConnector(license_key="k", config=_mock_config())
        factors = c.fetch_factors(["EF:ECI:act1:US:v310"])
        assert len(factors) == 1
        assert factors[0]["co2e_total"] == 0.85
        assert factors[0]["gwp_set"] == "AR6"
        assert factors[0]["scope"] == "3"
        assert factors[0]["uncertainty_95ci"] == 0.12
        assert factors[0]["allocation_method"] == ALLOCATION_CUTOFF

    @patch("greenlang.factors.connectors.ecoinvent.urllib.request.urlopen")
    def test_fetch_factors_empty(self, mock_urlopen):
        c = EcoinventConnector(license_key="k", config=_mock_config())
        assert c.fetch_factors([]) == []

    def test_allocation_method(self):
        c = EcoinventConnector(
            license_key="k",
            config=_mock_config(),
            allocation_method=ALLOCATION_APOS,
        )
        assert c._allocation == ALLOCATION_APOS

    def test_db_version(self):
        c = EcoinventConnector(
            license_key="k",
            config=_mock_config(),
            db_version="3.9",
        )
        assert c._db_version == "3.9"

    @patch("greenlang.factors.connectors.ecoinvent.urllib.request.urlopen")
    def test_auth_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://mock", 401, "Unauthorized", {}, None,
        )
        c = EcoinventConnector(license_key="bad", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorAuthError
        with pytest.raises(ConnectorAuthError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.ecoinvent.urllib.request.urlopen")
    def test_rate_limit(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://mock", 429, "Rate Limited", {}, None,
        )
        c = EcoinventConnector(license_key="k", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorRateLimitError
        with pytest.raises(ConnectorRateLimitError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.ecoinvent.urllib.request.urlopen")
    def test_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("Connection refused")
        c = EcoinventConnector(license_key="k", config=_mock_config())
        from greenlang.exceptions.connector import ConnectorNetworkError
        with pytest.raises(ConnectorNetworkError):
            c.fetch_metadata()

    @patch("greenlang.factors.connectors.ecoinvent.urllib.request.urlopen")
    def test_health_check(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = b'{"status": "ok"}'
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = EcoinventConnector(license_key="k", config=_mock_config())
        health = c.health_check()
        assert health.status.value == "healthy"

    def test_geo_map(self):
        assert _ECOINVENT_GEO_MAP["GLO"] == "GLOBAL"
        assert _ECOINVENT_GEO_MAP["RER"] == "EU"

    def test_ghg_categories(self):
        assert "electricity production" in _GHG_RELEVANT_CATEGORIES
        assert "transport" in _GHG_RELEVANT_CATEGORIES
        assert "cement production" in _GHG_RELEVANT_CATEGORIES
