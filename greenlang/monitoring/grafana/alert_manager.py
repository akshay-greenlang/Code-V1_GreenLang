# -*- coding: utf-8 -*-
"""
Grafana Alert Manager
======================

Manages Grafana Unified Alerting resources: alert rules, contact points,
notification policies, and mute timings for the GreenLang platform.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-002 Grafana Dashboards - Python SDK
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from greenlang.monitoring.grafana.client import GrafanaClient, GrafanaNotFoundError
from greenlang.monitoring.grafana.models import (
    AlertRule,
    ContactPoint,
    NotificationPolicy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GreenLang default contact points (from PRD Section 3.3)
# ---------------------------------------------------------------------------

GREENLANG_CONTACT_POINTS: list[dict[str, Any]] = [
    {
        "name": "platform-critical",
        "type": "slack",
        "settings": {
            "recipient": "#platform-alerts-critical",
            "username": "GreenLang Grafana",
            "mentionChannel": "here",
        },
    },
    {
        "name": "platform-warning",
        "type": "slack",
        "settings": {
            "recipient": "#platform-alerts",
            "username": "GreenLang Grafana",
        },
    },
    {
        "name": "platform-info",
        "type": "email",
        "settings": {
            "addresses": "platform-team@greenlang.io",
            "singleEmail": True,
        },
    },
]


GREENLANG_MUTE_TIMINGS: list[dict[str, Any]] = [
    {
        "name": "maintenance-window",
        "time_intervals": [
            {
                "times": [{"start_time": "02:00", "end_time": "04:00"}],
                "weekdays": ["sunday"],
                "location": "UTC",
            },
        ],
    },
]


class GrafanaAlertManager:
    """Manage Grafana Unified Alerting configuration.

    Provides methods to provision contact points, notification policies,
    mute timings, and alert rules for the GreenLang platform.

    Usage::

        async with GrafanaClient(base_url, api_key) as client:
            manager = GrafanaAlertManager(client)
            await manager.provision_contact_points()
            await manager.provision_notification_policies()
            await manager.provision_mute_timings()
    """

    def __init__(self, client: GrafanaClient) -> None:
        """Initialise the alert manager.

        Args:
            client: Connected GrafanaClient instance.
        """
        self._client = client

    # -- Contact points ------------------------------------------------------

    async def provision_contact_points(
        self,
        contact_points: Optional[list[dict[str, Any]]] = None,
    ) -> list[ContactPoint]:
        """Provision contact points in Grafana.

        Creates contact points that do not already exist. Existing contact
        points with the same name are skipped.

        Args:
            contact_points: List of contact point config dicts. Defaults to
                            GREENLANG_CONTACT_POINTS.

        Returns:
            List of created/existing ContactPoint models.
        """
        if contact_points is None:
            contact_points = GREENLANG_CONTACT_POINTS

        existing_cps = await self._client.get_contact_points()
        existing_names = {cp.name for cp in existing_cps}

        provisioned: list[ContactPoint] = []
        for cp_config in contact_points:
            name = cp_config["name"]
            if name in existing_names:
                logger.info("Contact point already exists (skipped): name=%s", name)
                matching = next((cp for cp in existing_cps if cp.name == name), None)
                if matching:
                    provisioned.append(matching)
                continue

            cp = ContactPoint(
                name=name,
                type=cp_config["type"],
                settings=cp_config.get("settings", {}),
            )
            created = await self._client.create_contact_point(cp)
            provisioned.append(created)
            logger.info("Contact point provisioned: name=%s type=%s", name, cp_config["type"])

        logger.info("Contact point provisioning complete: %d total", len(provisioned))
        return provisioned

    # -- Notification policies -----------------------------------------------

    async def provision_notification_policies(
        self,
        default_receiver: str = "platform-warning",
        critical_receiver: str = "platform-critical",
        warning_receiver: str = "platform-warning",
        info_receiver: str = "platform-info",
    ) -> NotificationPolicy:
        """Provision the GreenLang notification policy tree.

        Creates a three-tier routing policy:
          - Critical: severity=critical -> critical_receiver (1h repeat)
          - Warning: severity=warning -> warning_receiver (4h repeat)
          - Info: severity=info -> info_receiver (12h repeat)

        Args:
            default_receiver: Default contact point for unmatched alerts.
            critical_receiver: Contact point for critical alerts.
            warning_receiver: Contact point for warning alerts.
            info_receiver: Contact point for info alerts.

        Returns:
            Root NotificationPolicy that was applied.
        """
        policy = NotificationPolicy(
            receiver=default_receiver,
            group_by=["grafana_folder", "alertname"],
            group_wait="30s",
            group_interval="5m",
            repeat_interval="4h",
            routes=[
                NotificationPolicy(
                    receiver=critical_receiver,
                    object_matchers=[["severity", "=", "critical"]],
                    group_wait="10s",
                    group_interval="1m",
                    repeat_interval="1h",
                    mute_time_intervals=["maintenance-window"],
                ),
                NotificationPolicy(
                    receiver=warning_receiver,
                    object_matchers=[["severity", "=", "warning"]],
                    group_wait="30s",
                    group_interval="5m",
                    repeat_interval="4h",
                ),
                NotificationPolicy(
                    receiver=info_receiver,
                    object_matchers=[["severity", "=", "info"]],
                    group_wait="1m",
                    group_interval="10m",
                    repeat_interval="12h",
                ),
            ],
        )

        await self._client.set_notification_policy(policy)
        logger.info("Notification policy tree provisioned with %d routes", len(policy.routes))
        return policy

    # -- Mute timings --------------------------------------------------------

    async def provision_mute_timings(
        self,
        mute_timings: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Provision mute timings in Grafana.

        Creates mute timings that do not already exist. Existing mute
        timings with the same name are skipped.

        Args:
            mute_timings: List of mute timing config dicts. Defaults to
                          GREENLANG_MUTE_TIMINGS.

        Returns:
            List of created/existing mute timing dicts.
        """
        if mute_timings is None:
            mute_timings = GREENLANG_MUTE_TIMINGS

        existing = await self._client.get_mute_timings()
        existing_names = {mt.get("name", "") for mt in existing}

        provisioned: list[dict[str, Any]] = []
        for mt_config in mute_timings:
            name = mt_config["name"]
            if name in existing_names:
                logger.info("Mute timing already exists (skipped): name=%s", name)
                matching = next((mt for mt in existing if mt.get("name") == name), None)
                if matching:
                    provisioned.append(matching)
                continue

            created = await self._client.create_mute_timing(
                name=name,
                time_intervals=mt_config["time_intervals"],
            )
            provisioned.append(created)
            logger.info("Mute timing provisioned: name=%s", name)

        return provisioned

    # -- Alert rules ---------------------------------------------------------

    async def create_alert_rule_from_promql(
        self,
        title: str,
        expr: str,
        folder_uid: str,
        rule_group: str = "default",
        for_duration: str = "5m",
        severity: str = "warning",
        summary: str = "",
        description: str = "",
        runbook_url: str = "",
        datasource_uid: str = "thanos",
        condition_ref: str = "C",
        no_data_state: str = "NoData",
        exec_err_state: str = "Error",
    ) -> AlertRule:
        """Create an alert rule from a PromQL expression.

        Builds the full alert query structure required by Grafana Unified
        Alerting, including the data query, reduce step, and threshold
        condition.

        Args:
            title: Alert rule title.
            expr: PromQL expression.
            folder_uid: Folder UID for the alert rule.
            rule_group: Alert rule group name.
            for_duration: Duration before firing (e.g. '5m').
            severity: Alert severity label.
            summary: Alert summary annotation.
            description: Alert description annotation.
            runbook_url: Runbook URL annotation.
            datasource_uid: Data source UID for the query.
            condition_ref: Condition reference ID.
            no_data_state: State when no data.
            exec_err_state: State on execution error.

        Returns:
            Created AlertRule model.
        """
        # Build the Grafana alert query structure
        data: list[dict[str, Any]] = [
            {
                "refId": "A",
                "queryType": "",
                "relativeTimeRange": {"from": 600, "to": 0},
                "datasourceUid": datasource_uid,
                "model": {
                    "expr": expr,
                    "instant": True,
                    "refId": "A",
                    "datasource": {"type": "prometheus", "uid": datasource_uid},
                },
            },
            {
                "refId": "B",
                "queryType": "",
                "relativeTimeRange": {"from": 600, "to": 0},
                "datasourceUid": "__expr__",
                "model": {
                    "conditions": [
                        {
                            "evaluator": {"params": [], "type": "gt"},
                            "operator": {"type": "and"},
                            "query": {"params": ["B"]},
                            "reducer": {"params": [], "type": "last"},
                            "type": "query",
                        },
                    ],
                    "datasource": {"type": "__expr__", "uid": "__expr__"},
                    "expression": "A",
                    "reducer": "last",
                    "refId": "B",
                    "type": "reduce",
                },
            },
            {
                "refId": condition_ref,
                "queryType": "",
                "relativeTimeRange": {"from": 600, "to": 0},
                "datasourceUid": "__expr__",
                "model": {
                    "conditions": [
                        {
                            "evaluator": {"params": [0], "type": "gt"},
                            "operator": {"type": "and"},
                            "query": {"params": [condition_ref]},
                            "reducer": {"params": [], "type": "last"},
                            "type": "query",
                        },
                    ],
                    "datasource": {"type": "__expr__", "uid": "__expr__"},
                    "expression": "B",
                    "refId": condition_ref,
                    "type": "threshold",
                },
            },
        ]

        # Build annotations
        annotations: dict[str, str] = {}
        if summary:
            annotations["summary"] = summary
        if description:
            annotations["description"] = description
        if runbook_url:
            annotations["runbook_url"] = runbook_url

        rule = AlertRule(
            title=title,
            folderUID=folder_uid,
            ruleGroup=rule_group,
            condition=condition_ref,
            data=data,
            noDataState=no_data_state,
            execErrState=exec_err_state,
            for_=for_duration,
            labels={"severity": severity, "service": "greenlang"},
            annotations=annotations,
        )

        created = await self._client.create_alert_rule(rule)
        logger.info(
            "Alert rule created: uid=%s title=%s severity=%s",
            created.uid, title, severity,
        )
        return created

    async def list_firing_alerts(self) -> list[AlertRule]:
        """List all alert rules (the caller should filter by state).

        Note: Grafana API returns alert rules, not alert instances.
        To get firing instances, use the Alertmanager API.

        Returns:
            List of all AlertRule models.
        """
        return await self._client.get_alert_rules()

    async def silence_alert(
        self,
        alert_name: str,
        duration_seconds: int = 3600,
        comment: str = "Silenced via GreenLang SDK",
        created_by: str = "greenlang-sdk",
    ) -> dict[str, Any]:
        """Create a silence for an alert by name.

        Uses the Grafana annotation API to create a silence annotation.
        For Alertmanager-native silences, use the Alertmanager API directly.

        Args:
            alert_name: Alert name to match.
            duration_seconds: Silence duration in seconds.
            comment: Silence comment.
            created_by: Creator identity.

        Returns:
            Created annotation dict.
        """
        import time

        now_ms = int(time.time() * 1000)
        end_ms = now_ms + (duration_seconds * 1000)

        result = await self._client.create_annotation(
            text=f"Silence: {alert_name} - {comment}",
            tags=["silence", alert_name],
            time_from=now_ms,
            time_to=end_ms,
        )
        logger.info(
            "Alert silenced: name=%s duration=%ds",
            alert_name, duration_seconds,
        )
        return result

    async def delete_alert_rule(self, uid: str) -> None:
        """Delete an alert rule by UID.

        Args:
            uid: Alert rule UID.
        """
        await self._client.delete_alert_rule(uid)

    async def get_alert_rule(self, uid: str) -> AlertRule:
        """Get an alert rule by UID.

        Args:
            uid: Alert rule UID.

        Returns:
            AlertRule model.
        """
        return await self._client.get_alert_rule(uid)
