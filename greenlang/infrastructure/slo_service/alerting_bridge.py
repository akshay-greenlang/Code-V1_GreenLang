# -*- coding: utf-8 -*-
"""
Alerting Bridge - OBS-005: SLO/SLI Definitions & Error Budget Management

Bridges SLO burn rate and budget alerts to the OBS-004 Unified Alerting
Service.  Translates SLO-specific alert data into the Alert model
understood by the alerting pipeline.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from greenlang.infrastructure.slo_service.models import (
    BurnRateAlert,
    ErrorBudget,
    SLO,
)


# ---------------------------------------------------------------------------
# AlertingBridge
# ---------------------------------------------------------------------------


class AlertingBridge:
    """Bridge between SLO alerts and the OBS-004 alerting service.

    Translates burn rate alerts and budget alerts into the unified
    alert format and dispatches them to the alerting service.

    Attributes:
        _enabled: Whether the bridge is active.
        _alerting_service: Reference to the alerting service (optional).
    """

    def __init__(
        self,
        enabled: bool = True,
        alerting_service: Any = None,
    ) -> None:
        """Initialize the alerting bridge.

        Args:
            enabled: Whether to dispatch alerts.
            alerting_service: OBS-004 alerting service instance.
        """
        self._enabled = enabled
        self._alerting_service = alerting_service

    @property
    def enabled(self) -> bool:
        """Whether the bridge is enabled."""
        return self._enabled

    def fire_burn_rate_alert(
        self,
        alert: BurnRateAlert,
        slo: SLO,
    ) -> Dict[str, Any]:
        """Dispatch a burn rate alert to the alerting service.

        Args:
            alert: BurnRateAlert that triggered.
            slo: SLO definition.

        Returns:
            Dictionary with dispatch result.
        """
        if not self._enabled:
            return {"dispatched": False, "reason": "bridge_disabled"}

        alert_data = {
            "source": "slo-service",
            "name": f"SLOBurnRate_{alert.burn_window}_{slo.safe_name}",
            "severity": alert.severity,
            "title": alert.message,
            "description": (
                f"Burn rate alert for SLO '{slo.name}' ({alert.burn_window}): "
                f"long={alert.burn_rate_long:.2f}x, "
                f"short={alert.burn_rate_short:.2f}x, "
                f"threshold={alert.threshold}x"
            ),
            "labels": {
                "slo_id": slo.slo_id,
                "service": slo.service,
                "burn_window": alert.burn_window,
            },
            "annotations": {
                "slo_target": str(slo.target),
                "burn_rate_long": str(round(alert.burn_rate_long, 4)),
                "burn_rate_short": str(round(alert.burn_rate_short, 4)),
            },
            "team": slo.team,
            "service": slo.service,
        }

        try:
            if self._alerting_service is not None:
                self._alerting_service.fire_alert(alert_data)

            logger.info(
                "SLO burn rate alert dispatched: %s (%s)",
                slo.slo_id,
                alert.burn_window,
            )
            return {"dispatched": True, "alert_data": alert_data}

        except Exception as exc:
            logger.error("Failed to dispatch burn rate alert: %s", exc)
            return {"dispatched": False, "error": str(exc)}

    def fire_budget_alert(
        self,
        budget: ErrorBudget,
        slo: SLO,
    ) -> Dict[str, Any]:
        """Dispatch a budget status alert to the alerting service.

        Args:
            budget: ErrorBudget state.
            slo: SLO definition.

        Returns:
            Dictionary with dispatch result.
        """
        if not self._enabled:
            return {"dispatched": False, "reason": "bridge_disabled"}

        severity_map = {
            "exhausted": "critical",
            "critical": "warning",
            "warning": "info",
            "healthy": "info",
        }

        alert_data = {
            "source": "slo-service",
            "name": f"SLOBudget_{budget.status.value}_{slo.safe_name}",
            "severity": severity_map.get(budget.status.value, "info"),
            "title": (
                f"SLO '{slo.name}' error budget {budget.status.value}: "
                f"{budget.remaining_percent:.1f}% remaining"
            ),
            "description": (
                f"Error budget for SLO '{slo.name}' (target {slo.target}%) "
                f"is {budget.status.value}. "
                f"Budget remaining: {budget.remaining_percent:.1f}%, "
                f"SLI: {budget.sli_value:.3f}%"
            ),
            "labels": {
                "slo_id": slo.slo_id,
                "service": slo.service,
                "budget_status": budget.status.value,
            },
            "annotations": {
                "slo_target": str(slo.target),
                "budget_remaining_percent": str(round(budget.remaining_percent, 4)),
                "consumed_percent": str(round(budget.consumed_percent, 4)),
                "sli_value": str(round(budget.sli_value, 6)),
            },
            "team": slo.team,
            "service": slo.service,
        }

        try:
            if self._alerting_service is not None:
                self._alerting_service.fire_alert(alert_data)

            logger.info(
                "SLO budget alert dispatched: %s (%s)",
                slo.slo_id,
                budget.status.value,
            )
            return {"dispatched": True, "alert_data": alert_data}

        except Exception as exc:
            logger.error("Failed to dispatch budget alert: %s", exc)
            return {"dispatched": False, "error": str(exc)}

    def resolve_alert(
        self,
        slo: SLO,
        alert_type: str = "burn_rate",
    ) -> Dict[str, Any]:
        """Resolve a previously fired SLO alert.

        Args:
            slo: SLO definition.
            alert_type: Type of alert to resolve.

        Returns:
            Dictionary with resolution result.
        """
        if not self._enabled:
            return {"resolved": False, "reason": "bridge_disabled"}

        try:
            logger.info(
                "SLO alert resolved: %s (%s)", slo.slo_id, alert_type
            )
            return {"resolved": True, "slo_id": slo.slo_id, "type": alert_type}
        except Exception as exc:
            logger.error("Failed to resolve SLO alert: %s", exc)
            return {"resolved": False, "error": str(exc)}
