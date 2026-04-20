# -*- coding: utf-8 -*-
"""Tests for Phase 2.5 Connect hardening."""
from __future__ import annotations

import asyncio

import pytest

from greenlang.connect import (
    AWSCostExplorerConnector,
    BaseConnector,
    ConnectorAuthError,
    ConnectorDependencyError,
    ConnectorError,
    ConnectorRegistry,
    DatabricksConnector,
    SAPS4HanaConnector,
    SnowflakeConnector,
    SourceSpec,
    WorkdayConnector,
    default_registry,
)


# --------------------------------------------------------------------------
# Registry + describe
# --------------------------------------------------------------------------


class TestRegistry:
    def test_all_expected_connectors_registered(self):
        registry = default_registry()
        available = set(registry.available())
        for cid in (
            "sap-s4hana",
            "workday",
            "snowflake",
            "databricks",
            "aws-cost-explorer",
        ):
            assert cid in available

    def test_unknown_connector_raises_value_error(self):
        registry = default_registry()
        with pytest.raises(ValueError):
            registry.get("nope")

    def test_describe_reports_credentials(self):
        registry = default_registry()
        rows = {r["connector_id"]: r for r in registry.describe()}
        assert rows["sap-s4hana"]["required_credentials"] == [
            "base_url", "client_id", "client_secret",
        ]
        assert rows["snowflake"]["required_credentials"] == [
            "account", "user", "warehouse",
        ]
        assert rows["aws-cost-explorer"]["required_credentials"] == [
            "aws_access_key_id", "aws_secret_access_key",
        ]
        assert rows["workday"]["required_python_package"] == "httpx"


# --------------------------------------------------------------------------
# Healthcheck structured verdict
# --------------------------------------------------------------------------


class TestHealthcheck:
    def test_missing_credentials_reported(self):
        connector = SAPS4HanaConnector()
        result = asyncio.run(connector.healthcheck({}))
        assert result.ok is False
        assert "base_url" in result.missing_credentials
        assert "client_id" in result.missing_credentials

    def test_healthcheck_ok_when_creds_and_dep_present(self):
        """httpx is a base dependency of greenlang-cli, so dep is present."""
        connector = SAPS4HanaConnector()
        result = asyncio.run(
            connector.healthcheck(
                {
                    "base_url": "https://erp.example.com",
                    "client_id": "x",
                    "client_secret": "y",
                }
            )
        )
        # httpx is already a core dep (see pyproject.toml)
        assert result.ok is True
        assert result.missing_credentials == []


# --------------------------------------------------------------------------
# Dry-run extraction (no external calls)
# --------------------------------------------------------------------------


class TestDryRun:
    def _spec(self, cid: str, credentials: dict[str, str], filters=None) -> SourceSpec:
        return SourceSpec(
            tenant_id="t1",
            connector_id=cid,
            credentials=credentials,
            filters=filters or {},
            dry_run=True,
        )

    def test_sap_dry_run(self):
        c = SAPS4HanaConnector()
        r = asyncio.run(c.extract(self._spec(
            "sap-s4hana",
            {"base_url": "x", "client_id": "y", "client_secret": "z"},
        )))
        assert r.connector_id == "sap-s4hana"
        assert r.row_count == 0
        assert r.metadata["dry_run"] is True

    def test_snowflake_dry_run(self):
        c = SnowflakeConnector()
        r = asyncio.run(c.extract(self._spec(
            "snowflake",
            {"account": "a", "user": "u", "warehouse": "w"},
            {"query": "SELECT 1"},
        )))
        assert r.row_count == 0
        assert r.metadata["dry_run"] is True

    def test_aws_cost_dry_run(self):
        c = AWSCostExplorerConnector()
        r = asyncio.run(c.extract(self._spec(
            "aws-cost-explorer",
            {"aws_access_key_id": "k", "aws_secret_access_key": "s"},
        )))
        assert r.row_count == 0
        assert r.metadata["dry_run"] is True

    def test_workday_dry_run(self):
        c = WorkdayConnector()
        r = asyncio.run(c.extract(self._spec(
            "workday",
            {"report_url": "https://workday", "username": "u", "password": "p"},
        )))
        assert r.row_count == 0

    def test_databricks_dry_run(self):
        c = DatabricksConnector()
        r = asyncio.run(c.extract(self._spec(
            "databricks",
            {"server_hostname": "h", "http_path": "/sql/1"},
            {"query": "SELECT 1"},
        )))
        assert r.row_count == 0


# --------------------------------------------------------------------------
# Auth + dependency errors on live extraction
# --------------------------------------------------------------------------


class TestErrors:
    def test_missing_credentials_raises(self):
        c = SAPS4HanaConnector()
        spec = SourceSpec(
            tenant_id="t1",
            connector_id="sap-s4hana",
            credentials={},  # missing
            filters={},
            dry_run=False,
        )
        with pytest.raises(ConnectorAuthError):
            asyncio.run(c.extract(spec))

    def test_aws_missing_creds_raises(self):
        c = AWSCostExplorerConnector()
        spec = SourceSpec(
            tenant_id="t1",
            connector_id="aws-cost-explorer",
            credentials={"aws_access_key_id": "k"},  # missing secret
            filters={},
            dry_run=False,
        )
        with pytest.raises(ConnectorAuthError):
            asyncio.run(c.extract(spec))

    def test_error_hierarchy(self):
        assert issubclass(ConnectorAuthError, ConnectorError)
        assert issubclass(ConnectorDependencyError, ConnectorError)


# --------------------------------------------------------------------------
# Custom registry
# --------------------------------------------------------------------------


class _DummyConnector(BaseConnector):
    connector_id = "dummy"
    required_credentials = ("token",)
    required_python_package = None

    async def _extract_records(self, spec):
        return [{"foo": "bar"}]


class TestCustomConnector:
    def test_register_and_extract(self):
        registry = ConnectorRegistry()
        registry.register("dummy", _DummyConnector)
        connector = registry.get("dummy")
        spec = SourceSpec(
            tenant_id="t1",
            connector_id="dummy",
            credentials={"token": "t"},
            filters={},
            dry_run=False,
        )
        result = asyncio.run(connector.extract(spec))
        assert result.row_count == 1
        assert result.records[0] == {"foo": "bar"}
        assert len(result.checksum) == 64

    def test_extract_missing_token_fails(self):
        registry = ConnectorRegistry()
        registry.register("dummy", _DummyConnector)
        connector = registry.get("dummy")
        spec = SourceSpec(
            tenant_id="t1",
            connector_id="dummy",
            credentials={},
            filters={},
            dry_run=False,
        )
        with pytest.raises(ConnectorAuthError):
            asyncio.run(connector.extract(spec))
