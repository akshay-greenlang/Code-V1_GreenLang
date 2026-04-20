# -*- coding: utf-8 -*-
"""AWS Cost Explorer connector — spend feed for Scope 3 Cat 1."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Iterable

from greenlang.connect.base import (
    BaseConnector,
    ConnectorExtractionError,
    SourceSpec,
)

logger = logging.getLogger(__name__)


class AWSCostExplorerConnector(BaseConnector):
    """boto3 CostExplorer ``GetCostAndUsage`` extractor."""

    connector_id = "aws-cost-explorer"
    required_credentials = ("aws_access_key_id", "aws_secret_access_key")
    required_python_package = "boto3"

    async def _extract_records(self, spec: SourceSpec) -> Iterable[dict[str, Any]]:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConnectorExtractionError(
                "boto3 is required for AWS Cost Explorer extraction"
            ) from exc

        region = spec.credentials.get("aws_region", "us-east-1")
        client = boto3.client(
            "ce",
            aws_access_key_id=spec.credentials["aws_access_key_id"],
            aws_secret_access_key=spec.credentials["aws_secret_access_key"],
            region_name=region,
        )

        end = date.fromisoformat(
            spec.filters.get("end_date") or date.today().isoformat()
        )
        start = date.fromisoformat(
            spec.filters.get("start_date")
            or (end - timedelta(days=30)).isoformat()
        )
        granularity = spec.filters.get("granularity", "MONTHLY")
        metrics = spec.filters.get("metrics") or ["UnblendedCost"]
        group_by = spec.filters.get("group_by") or [
            {"Type": "DIMENSION", "Key": "SERVICE"}
        ]

        try:
            resp = client.get_cost_and_usage(
                TimePeriod={
                    "Start": start.isoformat(),
                    "End": end.isoformat(),
                },
                Granularity=granularity,
                Metrics=metrics,
                GroupBy=group_by,
            )
        except Exception as exc:  # noqa: BLE001
            raise ConnectorExtractionError(
                f"AWS Cost Explorer request failed: {exc}"
            ) from exc

        records: list[dict[str, Any]] = []
        for period in resp.get("ResultsByTime", []):
            for group in period.get("Groups", []):
                records.append(
                    {
                        "source_system": "aws-cost-explorer",
                        "tenant_id": spec.tenant_id,
                        "time_period_start": period.get("TimePeriod", {}).get("Start"),
                        "time_period_end": period.get("TimePeriod", {}).get("End"),
                        "group_keys": group.get("Keys"),
                        "metrics": group.get("Metrics"),
                    }
                )
        return records
