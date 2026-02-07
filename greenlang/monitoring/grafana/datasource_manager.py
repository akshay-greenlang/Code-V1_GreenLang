# -*- coding: utf-8 -*-
"""
Grafana Data Source Manager
=============================

Manages the GreenLang data source configuration in Grafana, including
provisioning, health checking, and synchronisation of the standard
data source set defined in the PRD.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-002 Grafana Dashboards - Python SDK
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from greenlang.monitoring.grafana.client import (
    GrafanaClient,
    GrafanaConflictError,
    GrafanaNotFoundError,
)
from greenlang.monitoring.grafana.models import DataSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GreenLang standard data sources (from PRD Section 3.2)
# ---------------------------------------------------------------------------

GREENLANG_DATASOURCES: list[dict[str, Any]] = [
    {
        "name": "Thanos",
        "type": "prometheus",
        "uid": "thanos",
        "url": "http://thanos-query.monitoring.svc:9090",
        "isDefault": True,
        "jsonData": {
            "timeInterval": "15s",
            "queryTimeout": "60s",
            "httpMethod": "POST",
            "manageAlerts": True,
            "prometheusType": "Thanos",
            "cacheLevel": "High",
            "incrementalQuerying": True,
            "incrementalQueryOverlapWindow": "10m",
            "exemplarTraceIdDestinations": [
                {"name": "traceID", "datasourceUid": "jaeger"},
            ],
        },
    },
    {
        "name": "Prometheus",
        "type": "prometheus",
        "uid": "prometheus",
        "url": "http://gl-prometheus-server.monitoring.svc:9090",
        "jsonData": {
            "timeInterval": "15s",
            "queryTimeout": "30s",
            "httpMethod": "POST",
        },
    },
    {
        "name": "Loki",
        "type": "loki",
        "uid": "loki",
        "url": "http://loki-read.monitoring.svc:3100",
        "jsonData": {
            "maxLines": 5000,
            "derivedFields": [
                {
                    "datasourceUid": "jaeger",
                    "matcherRegex": '"trace_id":"(\\w+)"',
                    "name": "TraceID",
                    "url": "${__value.raw}",
                },
            ],
        },
    },
    {
        "name": "Jaeger",
        "type": "jaeger",
        "uid": "jaeger",
        "url": "http://jaeger-query.monitoring.svc:16686",
        "jsonData": {
            "tracesToLogsV2": {
                "datasourceUid": "loki",
                "spanStartTimeShift": "-1h",
                "spanEndTimeShift": "1h",
                "filterByTraceID": True,
                "filterBySpanID": True,
            },
        },
    },
    {
        "name": "Alertmanager",
        "type": "alertmanager",
        "uid": "alertmanager",
        "url": "http://gl-prometheus-alertmanager.monitoring.svc:9093",
        "jsonData": {
            "implementation": "prometheus",
            "handleGrafanaManagedAlerts": True,
        },
    },
    {
        "name": "PostgreSQL",
        "type": "postgres",
        "uid": "postgresql",
        "url": "${POSTGRES_READ_HOST}:5432",
        "jsonData": {
            "database": "greenlang",
            "user": "grafana_reader",
            "sslmode": "verify-full",
            "maxOpenConns": 10,
            "maxIdleConns": 5,
            "connMaxLifetime": 14400,
            "postgresVersion": 1500,
            "timescaledb": True,
        },
    },
    {
        "name": "CloudWatch",
        "type": "cloudwatch",
        "uid": "cloudwatch",
        "url": "",
        "jsonData": {
            "authType": "default",
            "defaultRegion": "eu-west-1",
        },
    },
]


class DataSourceManager:
    """Manage Grafana data sources for the GreenLang platform.

    Provides methods to provision, test, and synchronise the standard
    data source configuration.

    Usage::

        async with GrafanaClient(base_url, api_key) as client:
            manager = DataSourceManager(client)
            results = await manager.provision_datasources()
            health = await manager.test_all_datasources()
    """

    def __init__(self, client: GrafanaClient) -> None:
        """Initialise the data source manager.

        Args:
            client: Connected GrafanaClient instance.
        """
        self._client = client

    async def provision_datasources(
        self,
        datasources: Optional[list[dict[str, Any]]] = None,
        overwrite: bool = False,
    ) -> dict[str, DataSource]:
        """Provision data sources in Grafana.

        Creates data sources that do not exist. If overwrite is True,
        existing data sources are updated. Otherwise, they are skipped.

        Args:
            datasources: List of data source config dicts. Defaults to
                         GREENLANG_DATASOURCES.
            overwrite: Whether to overwrite existing data sources.

        Returns:
            Dict mapping data source UID to DataSource model.
        """
        if datasources is None:
            datasources = GREENLANG_DATASOURCES

        provisioned: dict[str, DataSource] = {}
        for ds_config in datasources:
            uid = ds_config.get("uid", "")
            name = ds_config.get("name", "")
            ds_model = DataSource(
                uid=uid,
                name=name,
                type=ds_config["type"],
                url=ds_config.get("url", ""),
                isDefault=ds_config.get("isDefault", False),
                jsonData=ds_config.get("jsonData", {}),
                secureJsonData=ds_config.get("secureJsonData", {}),
                access="proxy",
                editable=False,
            )

            try:
                existing = await self._client.get_datasource(uid)
                if overwrite:
                    ds_model.id = existing.id
                    updated = await self._client.update_datasource(ds_model)
                    provisioned[uid] = updated
                    logger.info("DataSource updated: uid=%s name=%s", uid, name)
                else:
                    provisioned[uid] = existing
                    logger.info("DataSource already exists (skipped): uid=%s name=%s", uid, name)
            except GrafanaNotFoundError:
                created = await self._client.create_datasource(ds_model)
                provisioned[uid] = created
                logger.info("DataSource provisioned: uid=%s name=%s type=%s", uid, name, ds_config["type"])

        logger.info(
            "DataSource provisioning complete: %d data sources", len(provisioned)
        )
        return provisioned

    async def test_all_datasources(
        self,
        datasources: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, dict[str, Any]]:
        """Health-check all provisioned data sources.

        Args:
            datasources: List of data source config dicts. Defaults to
                         GREENLANG_DATASOURCES.

        Returns:
            Dict mapping data source UID to health check result dict.
            Each result contains 'status' ('ok' or 'error') and 'message'.
        """
        if datasources is None:
            datasources = GREENLANG_DATASOURCES

        results: dict[str, dict[str, Any]] = {}
        for ds_config in datasources:
            uid = ds_config.get("uid", "")
            name = ds_config.get("name", "")
            try:
                health = await self._client.test_datasource(uid)
                results[uid] = {"status": "ok", "message": str(health), "name": name}
                logger.info("DataSource health OK: uid=%s name=%s", uid, name)
            except Exception as exc:
                results[uid] = {"status": "error", "message": str(exc), "name": name}
                logger.warning("DataSource health FAIL: uid=%s name=%s error=%s", uid, name, exc)

        healthy = sum(1 for r in results.values() if r["status"] == "ok")
        logger.info(
            "DataSource health check: %d/%d healthy", healthy, len(results)
        )
        return results

    async def get_datasource_health(self, uid: str) -> dict[str, Any]:
        """Health-check a single data source.

        Args:
            uid: Data source UID.

        Returns:
            Health check result dict with 'status' and 'message'.
        """
        try:
            result = await self._client.test_datasource(uid)
            return {"status": "ok", "message": str(result)}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    async def list_datasources(self) -> list[DataSource]:
        """List all configured data sources.

        Returns:
            List of DataSource models.
        """
        return await self._client.get_datasources()

    async def delete_datasource(self, uid: str) -> None:
        """Delete a data source by UID.

        Args:
            uid: Data source UID.
        """
        await self._client.delete_datasource(uid)
