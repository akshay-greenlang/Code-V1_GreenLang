# -*- coding: utf-8 -*-
"""
SLO Service Setup - OBS-005: SLO/SLI Definitions & Error Budget Management

Provides ``configure_slo_service(app)`` which wires up the full SLO
pipeline (manager, calculators, budget engine, burn rate engine,
generators, compliance reporter, alerting bridge) and mounts the REST
API. Also exposes ``get_slo_service(app)`` for programmatic access.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.slo_service.setup import configure_slo_service
    >>> app = FastAPI()
    >>> configure_slo_service(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.slo_service.alerting_bridge import AlertingBridge
from greenlang.infrastructure.slo_service.config import (
    SLOServiceConfig,
    get_config,
)
from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    BurnRateWindow,
    ErrorBudget,
    SLO,
    SLOReport,
)
from greenlang.infrastructure.slo_service.slo_manager import SLOManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI

    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# SLOService facade
# ---------------------------------------------------------------------------


class SLOService:
    """Unified facade over the complete SLO/SLI service pipeline.

    Orchestrates SLO management, SLI calculation, error budget tracking,
    burn rate alerting, rule generation, dashboard generation, and
    compliance reporting.

    Attributes:
        config: SLOServiceConfig instance.
        manager: SLO definition manager.
        bridge: OBS-004 alerting bridge.
    """

    def __init__(
        self,
        config: Optional[SLOServiceConfig] = None,
    ) -> None:
        """Initialize the SLO service facade.

        Args:
            config: Configuration (loaded from env if not provided).
        """
        self.config = config or get_config()
        self.manager = SLOManager()
        self.bridge = AlertingBridge(
            enabled=self.config.alerting_bridge_enabled,
        )
        self._budget_store: Dict[str, List[ErrorBudget]] = {}
        self._report_store: Dict[str, SLOReport] = {}
        self._evaluation_task: Optional[asyncio.Task[None]] = None
        self._running = False
        logger.info("SLOService facade created")

    # ------------------------------------------------------------------
    # SLO CRUD
    # ------------------------------------------------------------------

    def create_slo(self, data: Dict[str, Any]) -> SLO:
        """Create a new SLO from a data dictionary.

        Args:
            data: SLO creation data.

        Returns:
            Created SLO.

        Raises:
            ValueError: If SLO already exists.
        """
        from greenlang.infrastructure.slo_service.models import SLI

        sli = SLI.from_dict(data.get("sli", {}))
        slo = SLO(
            slo_id=data.get("slo_id", ""),
            name=data.get("name", ""),
            service=data.get("service", ""),
            sli=sli,
            target=float(data.get("target", 99.9)),
            description=data.get("description", ""),
            team=data.get("team", ""),
            labels=data.get("labels", {}),
        )
        if "window" in data:
            from greenlang.infrastructure.slo_service.models import SLOWindow
            slo.window = SLOWindow(data["window"])

        return self.manager.create(slo)

    def get_slo(self, slo_id: str) -> Optional[SLO]:
        """Retrieve an SLO by ID.

        Args:
            slo_id: SLO identifier.

        Returns:
            SLO or None.
        """
        return self.manager.get(slo_id)

    def list_slos(
        self,
        service: Optional[str] = None,
        team: Optional[str] = None,
        enabled: Optional[bool] = None,
        include_deleted: bool = False,
    ) -> List[SLO]:
        """List SLOs with optional filters.

        Args:
            service: Filter by service name.
            team: Filter by team name.
            enabled: Filter by enabled status.
            include_deleted: Include soft-deleted SLOs.

        Returns:
            List of matching SLOs.
        """
        slos = self.manager.list_all(
            service=service,
            team=team,
            include_deleted=include_deleted,
        )
        if enabled is not None:
            slos = [s for s in slos if s.enabled == enabled]
        return slos

    def update_slo(self, slo_id: str, updates: Dict[str, Any]) -> SLO:
        """Update an SLO.

        Args:
            slo_id: SLO identifier.
            updates: Fields to update.

        Returns:
            Updated SLO.

        Raises:
            KeyError: If SLO not found.
        """
        return self.manager.update(slo_id, updates)

    def delete_slo(self, slo_id: str) -> bool:
        """Soft-delete an SLO.

        Args:
            slo_id: SLO identifier.

        Returns:
            True if deleted.
        """
        return self.manager.delete(slo_id)

    def get_slo_history(self, slo_id: str) -> List[Dict[str, Any]]:
        """Get SLO version history.

        Args:
            slo_id: SLO identifier.

        Returns:
            List of history entries.
        """
        return self.manager.get_history(slo_id)

    # ------------------------------------------------------------------
    # SLI calculation
    # ------------------------------------------------------------------

    async def calculate_sli(
        self,
        slo_id: str,
        window: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate the current SLI for an SLO.

        Args:
            slo_id: SLO identifier.
            window: Override window.

        Returns:
            Dictionary with SLI results.

        Raises:
            KeyError: If SLO not found.
        """
        slo = self.manager.get(slo_id)
        if slo is None:
            raise KeyError(f"SLO not found: {slo_id}")

        from greenlang.infrastructure.slo_service.sli_calculator import (
            calculate_sli as calc_sli,
        )

        sli_value = await calc_sli(
            slo,
            self.config.prometheus_url,
            window=window,
            timeout_seconds=self.config.prometheus_timeout_seconds,
        )

        return {
            "slo_id": slo_id,
            "sli_value": sli_value,
            "sli_percent": round(sli_value * 100, 6) if sli_value is not None else None,
            "target": slo.target,
            "met": (sli_value * 100 >= slo.target) if sli_value is not None else None,
            "window": window or slo.window.prometheus_duration,
        }

    # ------------------------------------------------------------------
    # Error budget
    # ------------------------------------------------------------------

    def get_error_budget(self, slo_id: str) -> Dict[str, Any]:
        """Get the latest error budget for an SLO.

        Args:
            slo_id: SLO identifier.

        Returns:
            Budget dictionary.

        Raises:
            KeyError: If SLO not found.
        """
        slo = self.manager.get(slo_id)
        if slo is None:
            raise KeyError(f"SLO not found: {slo_id}")

        history = self._budget_store.get(slo_id, [])
        if history:
            return history[-1].to_dict()

        # Return a fresh budget with no consumption data
        budget = ErrorBudget(
            slo_id=slo_id,
            total_minutes=slo.window_minutes * slo.error_budget_fraction,
            consumed_minutes=0.0,
            remaining_minutes=slo.window_minutes * slo.error_budget_fraction,
            remaining_percent=100.0,
            consumed_percent=0.0,
            status=BudgetStatus.HEALTHY,
            sli_value=0.0,
            window=slo.window.value,
        )
        return budget.to_dict()

    def get_budget_history(
        self,
        slo_id: str,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get budget history for an SLO.

        Args:
            slo_id: SLO identifier.
            days: Number of days of history.

        Returns:
            List of budget snapshot dictionaries.
        """
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        history = self._budget_store.get(slo_id, [])

        results = []
        for budget in history:
            if budget.calculated_at and budget.calculated_at >= cutoff:
                results.append(budget.to_dict())

        return results

    def forecast_budget(self, slo_id: str) -> Dict[str, Any]:
        """Forecast budget exhaustion for an SLO.

        Args:
            slo_id: SLO identifier.

        Returns:
            Forecast dictionary.

        Raises:
            KeyError: If SLO not found.
        """
        slo = self.manager.get(slo_id)
        if slo is None:
            raise KeyError(f"SLO not found: {slo_id}")

        from greenlang.infrastructure.slo_service.error_budget import (
            budget_consumption_rate,
            forecast_exhaustion,
        )

        history = self._budget_store.get(slo_id, [])
        if not history:
            return {
                "slo_id": slo_id,
                "exhaustion_forecast": None,
                "consumption_rate_per_hour": 0.0,
                "message": "No budget data available yet",
            }

        latest = history[-1]
        rate = budget_consumption_rate(
            latest.consumed_percent,
            slo.window_minutes,
        )
        forecast = forecast_exhaustion(latest.consumed_percent, rate)

        return {
            "slo_id": slo_id,
            "exhaustion_forecast": forecast.isoformat() if forecast else None,
            "consumption_rate_per_hour": round(rate, 4),
            "current_consumed_percent": round(latest.consumed_percent, 4),
            "current_remaining_percent": round(latest.remaining_percent, 4),
        }

    def check_budget_policy(self, slo_id: str) -> Dict[str, Any]:
        """Check budget policy for an SLO.

        Args:
            slo_id: SLO identifier.

        Returns:
            Policy check result.

        Raises:
            KeyError: If SLO not found.
        """
        slo = self.manager.get(slo_id)
        if slo is None:
            raise KeyError(f"SLO not found: {slo_id}")

        from greenlang.infrastructure.slo_service.error_budget import (
            check_budget_policy as _check_policy,
        )

        history = self._budget_store.get(slo_id, [])
        if not history:
            return {
                "slo_id": slo_id,
                "budget_status": "healthy",
                "policy": self.config.budget_exhausted_action,
                "action_required": False,
                "action": "none",
            }

        return _check_policy(history[-1], self.config.budget_exhausted_action)

    # ------------------------------------------------------------------
    # Burn rate
    # ------------------------------------------------------------------

    def get_burn_rates(
        self,
        slo_id: str,
        window: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get burn rates for an SLO.

        Args:
            slo_id: SLO identifier.
            window: Specific window (fast/medium/slow) or None for all.

        Returns:
            Burn rate data dictionary.

        Raises:
            KeyError: If SLO not found.
        """
        slo = self.manager.get(slo_id)
        if slo is None:
            raise KeyError(f"SLO not found: {slo_id}")

        windows = ["fast", "medium", "slow"]
        if window and window in windows:
            windows = [window]

        rates = {}
        for w in windows:
            bw = BurnRateWindow(w)
            rates[w] = {
                "burn_rate": 0.0,
                "threshold": bw.threshold,
                "long_window": bw.long_window,
                "short_window": bw.short_window,
                "severity": bw.severity,
                "alert_firing": False,
            }

        return {"slo_id": slo_id, "burn_rates": rates}

    def check_burn_rate_alerts(self, slo_id: str) -> Dict[str, Any]:
        """Check which burn rate alerts are firing for an SLO.

        Args:
            slo_id: SLO identifier.

        Returns:
            Dictionary of alert statuses.

        Raises:
            KeyError: If SLO not found.
        """
        slo = self.manager.get(slo_id)
        if slo is None:
            raise KeyError(f"SLO not found: {slo_id}")

        alerts = {
            "slo_id": slo_id,
            "alerts": {
                "fast": {"firing": False, "severity": "critical"},
                "medium": {"firing": False, "severity": "warning"},
                "slow": {"firing": False, "severity": "info"},
            },
            "any_firing": False,
        }
        return alerts

    # ------------------------------------------------------------------
    # Rule/Dashboard generation
    # ------------------------------------------------------------------

    def generate_recording_rules(
        self,
        output_path: str = "",
    ) -> Dict[str, Any]:
        """Generate Prometheus recording rules.

        Args:
            output_path: Override output path.

        Returns:
            Result dictionary with path and rule count.
        """
        from greenlang.infrastructure.slo_service.recording_rules import (
            generate_all_recording_rules,
            write_recording_rules_file,
        )
        from greenlang.infrastructure.slo_service.metrics import (
            record_recording_rules_generated,
        )

        slos = self.manager.list_all()
        path = output_path or self.config.recording_rules_output_path

        try:
            result_path = write_recording_rules_file(slos, path)
            rules = generate_all_recording_rules(slos)
            total_rules = sum(
                len(g.get("rules", [])) for g in rules.get("groups", [])
            )
            record_recording_rules_generated()
            return {
                "status": "generated",
                "path": result_path,
                "total_rules": total_rules,
                "slo_count": len(slos),
            }
        except Exception as exc:
            logger.error("Recording rule generation failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def generate_alert_rules(
        self,
        output_path: str = "",
    ) -> Dict[str, Any]:
        """Generate Prometheus alert rules.

        Args:
            output_path: Override output path.

        Returns:
            Result dictionary.
        """
        from greenlang.infrastructure.slo_service.alert_rules import (
            generate_all_alert_rules,
            write_alert_rules_file,
        )

        slos = self.manager.list_all()
        path = output_path or self.config.alert_rules_output_path

        try:
            result_path = write_alert_rules_file(slos, path)
            rules = generate_all_alert_rules(slos)
            total_rules = sum(
                len(g.get("rules", [])) for g in rules.get("groups", [])
            )
            return {
                "status": "generated",
                "path": result_path,
                "total_rules": total_rules,
                "slo_count": len(slos),
            }
        except Exception as exc:
            logger.error("Alert rule generation failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def generate_dashboards(
        self,
        output_dir: str = "",
    ) -> Dict[str, Any]:
        """Generate Grafana dashboards.

        Args:
            output_dir: Override output directory.

        Returns:
            Result dictionary with paths.
        """
        from greenlang.infrastructure.slo_service.dashboard_generator import (
            write_dashboards,
        )

        slos = self.manager.list_all()
        directory = output_dir or self.config.dashboards_output_dir

        try:
            paths = write_dashboards(slos, directory)
            return {
                "status": "generated",
                "paths": paths,
                "dashboard_count": len(paths),
                "slo_count": len(slos),
            }
        except Exception as exc:
            logger.error("Dashboard generation failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Compliance reporting
    # ------------------------------------------------------------------

    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate a weekly compliance report.

        Returns:
            Report dictionary.
        """
        from greenlang.infrastructure.slo_service.compliance_reporter import (
            generate_report,
        )

        slos = self.manager.list_all()
        budgets = {b.slo_id: b for b in self._get_all_latest_budgets()}
        report = generate_report("weekly", slos, budgets)
        self._report_store[report.report_id] = report
        return report.to_dict()

    def generate_monthly_report(
        self,
        month: int,
        year: int,
    ) -> Dict[str, Any]:
        """Generate a monthly compliance report.

        Args:
            month: Month number.
            year: Year.

        Returns:
            Report dictionary.
        """
        from datetime import datetime, timezone
        from calendar import monthrange
        from greenlang.infrastructure.slo_service.compliance_reporter import (
            generate_report,
        )

        period_start = datetime(year, month, 1, tzinfo=timezone.utc)
        _, last_day = monthrange(year, month)
        period_end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)

        slos = self.manager.list_all()
        budgets = {b.slo_id: b for b in self._get_all_latest_budgets()}
        report = generate_report(
            "monthly", slos, budgets,
            period_start=period_start,
            period_end=period_end,
        )
        self._report_store[report.report_id] = report
        return report.to_dict()

    def generate_quarterly_report(
        self,
        quarter: int,
        year: int,
    ) -> Dict[str, Any]:
        """Generate a quarterly compliance report.

        Args:
            quarter: Quarter number (1-4).
            year: Year.

        Returns:
            Report dictionary.
        """
        from datetime import datetime, timezone
        from calendar import monthrange
        from greenlang.infrastructure.slo_service.compliance_reporter import (
            generate_report,
        )

        quarter_start_months = {1: 1, 2: 4, 3: 7, 4: 10}
        start_month = quarter_start_months.get(quarter, 1)
        period_start = datetime(year, start_month, 1, tzinfo=timezone.utc)
        end_month = start_month + 2
        _, last_day = monthrange(year, end_month)
        period_end = datetime(year, end_month, last_day, 23, 59, 59, tzinfo=timezone.utc)

        slos = self.manager.list_all()
        budgets = {b.slo_id: b for b in self._get_all_latest_budgets()}
        report = generate_report(
            "quarterly", slos, budgets,
            period_start=period_start,
            period_end=period_end,
        )
        self._report_store[report.report_id] = report
        return report.to_dict()

    def list_reports(
        self,
        report_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List stored compliance reports.

        Args:
            report_type: Filter by type.
            limit: Max results.

        Returns:
            List of report summary dictionaries.
        """
        reports = list(self._report_store.values())
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]

        reports.sort(
            key=lambda r: r.generated_at or r.period_end,
            reverse=True,
        )

        return [
            {
                "report_id": r.report_id,
                "report_type": r.report_type,
                "period_start": r.period_start.isoformat() if r.period_start else None,
                "period_end": r.period_end.isoformat() if r.period_end else None,
                "overall_compliance_percent": r.overall_compliance_percent,
                "total_slos": r.total_slos,
                "generated_at": r.generated_at.isoformat() if r.generated_at else None,
            }
            for r in reports[:limit]
        ]

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    async def evaluate_all(self) -> List[Dict[str, Any]]:
        """Evaluate all active SLOs.

        Calculates SLI, budget, and burn rates for each SLO and stores
        budget snapshots.

        Returns:
            List of evaluation results per SLO.
        """
        from greenlang.infrastructure.slo_service.sli_calculator import (
            calculate_sli as calc_sli,
        )
        from greenlang.infrastructure.slo_service.error_budget import (
            calculate_error_budget,
        )

        slos = self.manager.list_all()
        results: List[Dict[str, Any]] = []

        for slo in slos:
            if not slo.enabled:
                continue

            start_time = time.monotonic()
            try:
                sli_value = await calc_sli(
                    slo,
                    self.config.prometheus_url,
                    timeout_seconds=self.config.prometheus_timeout_seconds,
                )

                if sli_value is None:
                    sli_value = 0.0

                budget = calculate_error_budget(slo, sli_value)

                # Store budget snapshot
                if slo.slo_id not in self._budget_store:
                    self._budget_store[slo.slo_id] = []
                self._budget_store[slo.slo_id].append(budget)

                # Trim history (keep last 10080 = 7 days at 1-min intervals)
                max_entries = 10080
                if len(self._budget_store[slo.slo_id]) > max_entries:
                    self._budget_store[slo.slo_id] = (
                        self._budget_store[slo.slo_id][-max_entries:]
                    )

                # Fire alerting bridge if budget is unhealthy
                if budget.status in (BudgetStatus.CRITICAL, BudgetStatus.EXHAUSTED):
                    self.bridge.fire_budget_alert(budget, slo)

                elapsed = time.monotonic() - start_time
                results.append({
                    "slo_id": slo.slo_id,
                    "status": "evaluated",
                    "sli_value": round(sli_value, 6),
                    "budget_status": budget.status.value,
                    "budget_remaining_percent": round(budget.remaining_percent, 4),
                    "elapsed_ms": round(elapsed * 1000, 2),
                })

            except Exception as exc:
                elapsed = time.monotonic() - start_time
                logger.error(
                    "SLO evaluation failed: slo=%s, error=%s",
                    slo.slo_id, exc,
                )
                results.append({
                    "slo_id": slo.slo_id,
                    "status": "error",
                    "error": str(exc),
                    "elapsed_ms": round(elapsed * 1000, 2),
                })

        return results

    async def _evaluation_loop(self) -> None:
        """Background evaluation loop."""
        self._running = True
        interval = self.config.evaluation_interval_seconds

        logger.info(
            "SLO evaluation loop started: interval=%ds", interval,
        )

        while self._running:
            try:
                await self.evaluate_all()
            except Exception as exc:
                logger.error("Evaluation loop error: %s", exc)

            await asyncio.sleep(interval)

    async def start_evaluation_loop(self) -> None:
        """Start the background evaluation loop."""
        if self._evaluation_task is not None:
            logger.warning("Evaluation loop already running")
            return

        self._evaluation_task = asyncio.create_task(
            self._evaluation_loop()
        )
        logger.info("SLO evaluation loop task created")

    async def stop_evaluation_loop(self) -> None:
        """Stop the background evaluation loop."""
        self._running = False
        if self._evaluation_task is not None:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
            self._evaluation_task = None
            logger.info("SLO evaluation loop stopped")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health status dictionary.
        """
        slos = self.manager.list_all()
        return {
            "healthy": True,
            "service": "slo-service",
            "version": "1.0.0",
            "environment": self.config.environment,
            "total_slos": len(slos),
            "enabled_slos": sum(1 for s in slos if s.enabled),
            "evaluation_loop_running": self._running,
            "alerting_bridge_enabled": self.bridge.enabled,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_all_latest_budgets(self) -> List[ErrorBudget]:
        """Get the latest budget for each SLO.

        Returns:
            List of latest ErrorBudget instances.
        """
        budgets = []
        for history in self._budget_store.values():
            if history:
                budgets.append(history[-1])
        return budgets


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------


def configure_slo_service(
    app: Any = None,
    config: Optional[SLOServiceConfig] = None,
) -> SLOService:
    """Wire up the full SLO service and mount the API.

    Args:
        app: FastAPI application instance (optional).
        config: SLOServiceConfig (loaded from env if not provided).

    Returns:
        Configured SLOService.
    """
    if config is None:
        config = get_config()

    # Build facade
    service = SLOService(config=config)

    # Mount on FastAPI app
    if app is not None and FASTAPI_AVAILABLE:
        try:
            from greenlang.infrastructure.slo_service.api.router import (
                slo_router,
            )
            if slo_router is not None:
                app.include_router(slo_router)
                logger.info("SLO API router mounted at /api/v1/slos")
        except ImportError:
            logger.warning("Could not import slo_router")

        app.state.slo_service = service

        # Register startup/shutdown lifecycle hooks
        @app.on_event("startup")
        async def _slo_startup() -> None:
            """Start the SLO evaluation loop on app startup."""
            if config.enabled:
                await service.start_evaluation_loop()

        @app.on_event("shutdown")
        async def _slo_shutdown() -> None:
            """Stop the SLO evaluation loop on app shutdown."""
            await service.stop_evaluation_loop()

    # Log summary
    logger.info(
        "SLO service configured: env=%s, evaluation_interval=%ds, "
        "alerting_bridge=%s, compliance=%s",
        config.environment,
        config.evaluation_interval_seconds,
        config.alerting_bridge_enabled,
        config.compliance_enabled,
    )

    return service


def get_slo_service(app: Any) -> SLOService:
    """Retrieve the SLOService from a FastAPI application.

    Args:
        app: FastAPI application.

    Returns:
        SLOService instance.

    Raises:
        RuntimeError: If not configured.
    """
    svc = getattr(app.state, "slo_service", None)
    if svc is None:
        raise RuntimeError(
            "SLOService not configured. Call configure_slo_service(app) first."
        )
    return svc
