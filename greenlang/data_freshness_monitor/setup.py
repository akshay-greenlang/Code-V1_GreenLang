# -*- coding: utf-8 -*-
"""
Data Freshness Monitor Service Setup - AGENT-DATA-016

Provides ``configure_freshness_monitor(app)`` which wires up the Data
Freshness Monitor SDK (dataset registry, SLA definition, freshness checker,
staleness detector, refresh predictor, alert manager, monitoring pipeline,
provenance tracker) and mounts the REST API.

Also exposes ``get_freshness_monitor(app)`` for programmatic access,
``get_router()`` for obtaining the FastAPI APIRouter, and the
``DataFreshnessMonitorService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.data_freshness_monitor.setup import configure_freshness_monitor
    >>> app = FastAPI()
    >>> configure_freshness_monitor(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.data_freshness_monitor.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

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
# Optional engine imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.dataset_registry import (
        DatasetRegistryEngine,
    )
except ImportError:
    DatasetRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.sla_definition import (
        SLADefinitionEngine,
    )
except ImportError:
    SLADefinitionEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.freshness_checker import (
        FreshnessCheckerEngine,
    )
except ImportError:
    FreshnessCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.staleness_detector import (
        StalenessDetectorEngine,
    )
except ImportError:
    StalenessDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.refresh_predictor import (
        RefreshPredictorEngine,
    )
except ImportError:
    RefreshPredictorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.alert_manager import (
        AlertManagerEngine,
    )
except ImportError:
    AlertManagerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.freshness_pipeline import (
        FreshnessMonitorPipelineEngine,
    )
except ImportError:
    FreshnessMonitorPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.metrics import PROMETHEUS_AVAILABLE
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Build a SHA-256 hash for arbitrary data."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ===================================================================
# DataFreshnessMonitorService facade
# ===================================================================


class DataFreshnessMonitorService:
    """Facade service for the Data Freshness Monitor SDK.

    Wires together the 7 engines (DatasetRegistry, SLADefinition,
    FreshnessChecker, StalenessDetector, RefreshPredictor,
    AlertManager, FreshnessMonitorPipeline) behind a simple API
    suitable for REST endpoint delegation.

    Attributes:
        config: Service configuration dictionary.
        _provenance: ProvenanceTracker instance.
        _dataset_registry: DatasetRegistryEngine instance.
        _sla_definition: SLADefinitionEngine instance.
        _freshness_checker: FreshnessCheckerEngine instance.
        _staleness_detector: StalenessDetectorEngine instance.
        _refresh_predictor: RefreshPredictorEngine instance.
        _alert_manager: AlertManagerEngine instance.
        _pipeline: FreshnessMonitorPipelineEngine instance.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize DataFreshnessMonitorService.

        Args:
            config: Optional configuration dictionary. Defaults to
                sensible production defaults when not provided.
        """
        self.config = config or {}
        self._provenance = get_provenance_tracker()

        # Engine stubs -- created lazily or via startup()
        self._dataset_registry: Any = None
        self._sla_definition: Any = None
        self._freshness_checker: Any = None
        self._staleness_detector: Any = None
        self._refresh_predictor: Any = None
        self._alert_manager: Any = None
        self._pipeline: Any = None

        # In-memory stores
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._sla_definitions: Dict[str, Dict[str, Any]] = {}
        self._checks: Dict[str, Dict[str, Any]] = {}
        self._breaches: Dict[str, Dict[str, Any]] = {}
        self._alerts: Dict[str, Dict[str, Any]] = {}
        self._predictions: Dict[str, Dict[str, Any]] = {}
        self._pipeline_results: Dict[str, Dict[str, Any]] = {}

        # Aggregate counters
        self._stats = {
            "total_datasets": 0,
            "total_sla_definitions": 0,
            "total_checks": 0,
            "total_breaches": 0,
            "total_alerts": 0,
            "total_predictions": 0,
            "total_pipelines": 0,
        }

        self._started = False
        logger.info("DataFreshnessMonitorService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Initialize all engines."""
        if DatasetRegistryEngine is not None:
            self._dataset_registry = DatasetRegistryEngine()
        if SLADefinitionEngine is not None:
            self._sla_definition = SLADefinitionEngine()
        if FreshnessCheckerEngine is not None:
            self._freshness_checker = FreshnessCheckerEngine(
                config=self.config,
            )
        if StalenessDetectorEngine is not None:
            self._staleness_detector = StalenessDetectorEngine()
        if RefreshPredictorEngine is not None:
            self._refresh_predictor = RefreshPredictorEngine()
        if AlertManagerEngine is not None:
            self._alert_manager = AlertManagerEngine()
        if FreshnessMonitorPipelineEngine is not None:
            self._pipeline = FreshnessMonitorPipelineEngine()

        self._started = True
        logger.info("DataFreshnessMonitorService started")

    def shutdown(self) -> None:
        """Shutdown the service."""
        self._started = False
        logger.info("DataFreshnessMonitorService shutdown")

    # ------------------------------------------------------------------
    # Health & Statistics
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return service health status.

        Returns:
            Dictionary with service status, engine availability, and
            store sizes.
        """
        return {
            "status": "healthy" if self._started else "starting",
            "service": "data_freshness_monitor",
            "engines": {
                "dataset_registry": self._dataset_registry is not None,
                "sla_definition": self._sla_definition is not None,
                "freshness_checker": self._freshness_checker is not None,
                "staleness_detector": self._staleness_detector is not None,
                "refresh_predictor": self._refresh_predictor is not None,
                "alert_manager": self._alert_manager is not None,
                "pipeline": self._pipeline is not None,
            },
            "stores": {
                "datasets": len(self._datasets),
                "sla_definitions": len(self._sla_definitions),
                "checks": len(self._checks),
                "breaches": len(self._breaches),
                "alerts": len(self._alerts),
                "predictions": len(self._predictions),
            },
            "timestamp": _utcnow().isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return service statistics.

        Returns:
            Dictionary with aggregate counts and store sizes.
        """
        return {
            **self._stats,
            "datasets_stored": len(self._datasets),
            "sla_definitions_stored": len(self._sla_definitions),
            "checks_stored": len(self._checks),
            "breaches_stored": len(self._breaches),
            "alerts_stored": len(self._alerts),
            "predictions_stored": len(self._predictions),
            "pipeline_results_stored": len(self._pipeline_results),
            "provenance_entries": self._provenance.entry_count,
            "timestamp": _utcnow().isoformat(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Alias for get_stats() used by the router."""
        return self.get_stats()

    def get_health(self) -> Dict[str, Any]:
        """Alias for health_check() used by the router.

        Returns:
            Service health dictionary.
        """
        return self.health_check()

    # ------------------------------------------------------------------
    # Dataset Management
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        name: str,
        source: str = "",
        owner: str = "",
        refresh_cadence: str = "daily",
        priority: int = 5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a dataset for freshness monitoring.

        Args:
            name: Human-readable dataset name.
            source: Source system identifier (e.g. ERP, API, manual).
            owner: Dataset owner (team or individual).
            refresh_cadence: Expected refresh frequency (realtime, hourly,
                daily, weekly, monthly, quarterly, annual).
            priority: Dataset priority (1=highest, 10=lowest).
            tags: Optional tags for grouping and filtering.
            metadata: Additional dataset metadata.

        Returns:
            Dictionary with registered dataset details.
        """
        dataset_id = str(uuid.uuid4())
        dataset = {
            "dataset_id": dataset_id,
            "name": name,
            "source": source,
            "owner": owner,
            "refresh_cadence": refresh_cadence,
            "priority": priority,
            "tags": tags or [],
            "metadata": metadata or {},
            "status": "active",
            "last_refreshed_at": None,
            "last_checked_at": None,
            "freshness_score": None,
            "freshness_level": None,
            "sla_status": "unknown",
            "created_at": _utcnow().isoformat(),
            "updated_at": _utcnow().isoformat(),
            "provenance_hash": _compute_hash({
                "dataset_id": dataset_id,
                "name": name,
                "source": source,
            }),
        }

        if self._dataset_registry is not None:
            try:
                result = self._dataset_registry.register_dataset(
                    name=name,
                    source_name=source,
                    source_type="",
                    owner=owner,
                    refresh_cadence=refresh_cadence,
                    priority="medium",
                    tags=tags,
                    metadata=metadata,
                )
                if hasattr(result, "id"):
                    dataset["id"] = result.id
            except Exception as exc:
                logger.warning(
                    "Dataset registry engine error, using fallback: %s", exc,
                )

        self._datasets[dataset_id] = dataset
        self._stats["total_datasets"] += 1
        self._provenance.record(
            "dataset", dataset_id, "register", dataset["provenance_hash"],
        )
        logger.info("Registered dataset %s (%s)", name, dataset_id)
        return dataset

    def list_datasets(
        self,
        status: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List registered datasets with optional filters.

        Args:
            status: Optional status filter (active, inactive, archived).
            source: Optional source filter.
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with datasets list and pagination metadata.
        """
        items = list(self._datasets.values())
        if status is not None:
            items = [d for d in items if d.get("status") == status]
        if source is not None:
            items = [d for d in items if d.get("source") == source]
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "datasets": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get a registered dataset by ID.

        Args:
            dataset_id: Unique dataset identifier.

        Returns:
            Dataset dictionary or None if not found.
        """
        return self._datasets.get(dataset_id)

    def update_dataset(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        source: Optional[str] = None,
        owner: Optional[str] = None,
        refresh_cadence: Optional[str] = None,
        priority: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a registered dataset.

        Args:
            dataset_id: Unique dataset identifier.
            name: Optional new name.
            source: Optional new source.
            owner: Optional new owner.
            refresh_cadence: Optional new refresh cadence.
            priority: Optional new priority.
            tags: Optional replacement tags.
            metadata: Optional additional metadata to merge.
            status: Optional new status.

        Returns:
            Updated dataset dictionary.

        Raises:
            ValueError: If dataset not found.
        """
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        if name is not None:
            dataset["name"] = name
        if source is not None:
            dataset["source"] = source
        if owner is not None:
            dataset["owner"] = owner
        if refresh_cadence is not None:
            dataset["refresh_cadence"] = refresh_cadence
        if priority is not None:
            dataset["priority"] = priority
        if tags is not None:
            dataset["tags"] = tags
        if metadata is not None:
            dataset["metadata"].update(metadata)
        if status is not None:
            dataset["status"] = status

        dataset["updated_at"] = _utcnow().isoformat()
        dataset["provenance_hash"] = _compute_hash({
            "dataset_id": dataset_id,
            "updated_at": dataset["updated_at"],
        })

        self._provenance.record(
            "dataset", dataset_id, "update", dataset["provenance_hash"],
        )
        logger.info("Updated dataset %s", dataset_id)
        return dataset

    def delete_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Remove a dataset from monitoring.

        Args:
            dataset_id: Unique dataset identifier.

        Returns:
            Dictionary with deletion status.

        Raises:
            ValueError: If dataset not found.
        """
        if dataset_id not in self._datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        self._datasets.pop(dataset_id)
        self._provenance.record(
            "dataset", dataset_id, "delete",
            _compute_hash({"dataset_id": dataset_id, "action": "delete"}),
        )
        logger.info("Deleted dataset %s", dataset_id)
        return {"dataset_id": dataset_id, "status": "deleted"}

    # ------------------------------------------------------------------
    # SLA Definition Management
    # ------------------------------------------------------------------

    def create_sla(
        self,
        dataset_id: str = "",
        name: str = "",
        warning_hours: float = 24.0,
        critical_hours: float = 72.0,
        severity: str = "high",
        escalation_policy: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an SLA definition for a dataset.

        Args:
            dataset_id: Dataset this SLA applies to (empty for default).
            name: Human-readable SLA name.
            warning_hours: Hours before warning-level alert fires.
            critical_hours: Hours before critical-level alert fires.
            severity: Default breach severity classification.
            escalation_policy: Escalation chain configuration.
            metadata: Additional SLA metadata.

        Returns:
            Dictionary with SLA definition details.
        """
        sla_id = str(uuid.uuid4())
        sla = {
            "sla_id": sla_id,
            "dataset_id": dataset_id,
            "name": name or f"sla-{sla_id[:8]}",
            "warning_hours": warning_hours,
            "critical_hours": critical_hours,
            "severity": severity,
            "escalation_policy": escalation_policy or {},
            "metadata": metadata or {},
            "status": "active",
            "created_at": _utcnow().isoformat(),
            "updated_at": _utcnow().isoformat(),
            "provenance_hash": _compute_hash({
                "sla_id": sla_id,
                "dataset_id": dataset_id,
                "warning_hours": warning_hours,
                "critical_hours": critical_hours,
            }),
        }

        if self._sla_definition is not None:
            try:
                result = self._sla_definition.create(
                    dataset_id=dataset_id,
                    name=name,
                    warning_hours=warning_hours,
                    critical_hours=critical_hours,
                    severity=severity,
                    escalation_policy=escalation_policy,
                )
                sla.update(result)
            except Exception as exc:
                logger.warning(
                    "SLA definition engine error, using fallback: %s", exc,
                )

        self._sla_definitions[sla_id] = sla
        self._stats["total_sla_definitions"] += 1
        self._provenance.record(
            "sla_definition", sla_id, "create", sla["provenance_hash"],
        )
        logger.info("Created SLA definition %s", sla_id)
        return sla

    def list_slas(
        self,
        dataset_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List SLA definitions with optional dataset filter.

        Args:
            dataset_id: Optional dataset ID filter.
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with SLA definitions list and pagination metadata.
        """
        items = list(self._sla_definitions.values())
        if dataset_id is not None:
            items = [s for s in items if s.get("dataset_id") == dataset_id]
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "sla_definitions": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_sla(self, sla_id: str) -> Optional[Dict[str, Any]]:
        """Get an SLA definition by ID.

        Args:
            sla_id: Unique SLA identifier.

        Returns:
            SLA definition dictionary or None if not found.
        """
        return self._sla_definitions.get(sla_id)

    def update_sla(
        self,
        sla_id: str,
        name: Optional[str] = None,
        warning_hours: Optional[float] = None,
        critical_hours: Optional[float] = None,
        severity: Optional[str] = None,
        escalation_policy: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an SLA definition.

        Args:
            sla_id: Unique SLA identifier.
            name: Optional new name.
            warning_hours: Optional new warning threshold.
            critical_hours: Optional new critical threshold.
            severity: Optional new severity classification.
            escalation_policy: Optional new escalation policy.
            metadata: Optional additional metadata to merge.
            status: Optional new status.

        Returns:
            Updated SLA definition dictionary.

        Raises:
            ValueError: If SLA not found.
        """
        sla = self._sla_definitions.get(sla_id)
        if sla is None:
            raise ValueError(f"SLA {sla_id} not found")

        if name is not None:
            sla["name"] = name
        if warning_hours is not None:
            sla["warning_hours"] = warning_hours
        if critical_hours is not None:
            sla["critical_hours"] = critical_hours
        if severity is not None:
            sla["severity"] = severity
        if escalation_policy is not None:
            sla["escalation_policy"] = escalation_policy
        if metadata is not None:
            sla["metadata"].update(metadata)
        if status is not None:
            sla["status"] = status

        sla["updated_at"] = _utcnow().isoformat()
        sla["provenance_hash"] = _compute_hash({
            "sla_id": sla_id,
            "updated_at": sla["updated_at"],
        })

        self._provenance.record(
            "sla_definition", sla_id, "update", sla["provenance_hash"],
        )
        logger.info("Updated SLA definition %s", sla_id)
        return sla

    # ------------------------------------------------------------------
    # Freshness Checking
    # ------------------------------------------------------------------

    def run_check(
        self,
        dataset_id: str,
        last_refreshed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a freshness check on a single dataset.

        Computes age since last update, applies freshness scoring,
        evaluates SLA compliance, and optionally generates breach records.

        Args:
            dataset_id: Dataset to check.
            last_refreshed_at: Optional ISO timestamp of last refresh.
                If not provided, uses stored dataset metadata.

        Returns:
            Dictionary with freshness check results.
        """
        start_t = time.time()
        check_id = str(uuid.uuid4())

        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Determine last refresh time
        refresh_ts = last_refreshed_at or dataset.get("last_refreshed_at")
        now = _utcnow()

        # Compute age in hours
        age_hours = 0.0
        if refresh_ts is not None:
            try:
                if isinstance(refresh_ts, str):
                    refresh_dt = datetime.fromisoformat(
                        refresh_ts.replace("Z", "+00:00"),
                    )
                else:
                    refresh_dt = refresh_ts
                if refresh_dt.tzinfo is None:
                    refresh_dt = refresh_dt.replace(tzinfo=timezone.utc)
                delta = now - refresh_dt
                age_hours = delta.total_seconds() / 3600.0
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Cannot parse refresh timestamp %s: %s", refresh_ts, exc,
                )

        # Compute freshness level and score
        freshness = self._compute_freshness(age_hours)

        # Evaluate SLA
        sla_result = self._evaluate_sla(dataset_id, age_hours)

        elapsed = (time.time() - start_t) * 1000.0

        check = {
            "check_id": check_id,
            "dataset_id": dataset_id,
            "checked_at": now.isoformat(),
            "age_hours": round(age_hours, 4),
            "freshness_score": freshness["score"],
            "freshness_level": freshness["level"],
            "sla_status": sla_result["sla_status"],
            "sla_breach": sla_result.get("breach"),
            "processing_time_ms": elapsed,
            "provenance_hash": _compute_hash({
                "check_id": check_id,
                "dataset_id": dataset_id,
                "age_hours": round(age_hours, 4),
            }),
        }

        # Update dataset state
        dataset["last_checked_at"] = now.isoformat()
        dataset["freshness_score"] = freshness["score"]
        dataset["freshness_level"] = freshness["level"]
        dataset["sla_status"] = sla_result["sla_status"]

        # Store breach if detected
        if sla_result.get("breach") is not None:
            breach = sla_result["breach"]
            breach_id = breach.get("breach_id", str(uuid.uuid4()))
            breach["breach_id"] = breach_id
            self._breaches[breach_id] = breach
            self._stats["total_breaches"] += 1

        self._checks[check_id] = check
        self._stats["total_checks"] += 1
        self._provenance.record(
            "freshness_check", check_id, "check", check["provenance_hash"],
        )
        return check

    def run_batch_check(
        self,
        dataset_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run freshness checks on multiple datasets.

        Args:
            dataset_ids: List of dataset IDs to check. If None,
                checks all registered datasets.

        Returns:
            Dictionary with batch check results.
        """
        start_t = time.time()
        batch_id = str(uuid.uuid4())
        ids = dataset_ids or list(self._datasets.keys())

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for ds_id in ids:
            try:
                result = self.run_check(dataset_id=ds_id)
                results.append(result)
            except Exception as exc:
                errors.append({
                    "dataset_id": ds_id,
                    "error": str(exc),
                })

        elapsed = (time.time() - start_t) * 1000.0

        output = {
            "batch_id": batch_id,
            "total_checked": len(results),
            "total_errors": len(errors),
            "results": results,
            "errors": errors,
            "processing_time_ms": elapsed,
            "provenance_hash": _compute_hash({
                "batch_id": batch_id,
                "total_checked": len(results),
            }),
        }

        self._provenance.record(
            "batch_check", batch_id, "batch_check",
            output["provenance_hash"],
        )
        logger.info(
            "Batch check %s completed: %d checked, %d errors in %.1fms",
            batch_id, len(results), len(errors), elapsed,
        )
        return output

    def list_checks(
        self,
        dataset_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List freshness check results with optional dataset filter.

        Args:
            dataset_id: Optional dataset ID filter.
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with checks list and pagination metadata.
        """
        items = list(self._checks.values())
        if dataset_id is not None:
            items = [c for c in items if c.get("dataset_id") == dataset_id]
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "checks": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # SLA Breach Management
    # ------------------------------------------------------------------

    def list_breaches(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List SLA breaches with optional filters.

        Args:
            severity: Optional severity filter (info, low, medium,
                high, critical).
            status: Optional status filter (detected, acknowledged,
                investigating, resolved).
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with breaches list and pagination metadata.
        """
        items = list(self._breaches.values())
        if severity is not None:
            items = [b for b in items if b.get("severity") == severity]
        if status is not None:
            items = [b for b in items if b.get("status") == status]
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "breaches": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_breach(self, breach_id: str) -> Optional[Dict[str, Any]]:
        """Get a breach by ID.

        Args:
            breach_id: Unique breach identifier.

        Returns:
            Breach dictionary or None if not found.
        """
        return self._breaches.get(breach_id)

    def update_breach(
        self,
        breach_id: str,
        status: Optional[str] = None,
        resolution_notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a breach status.

        Args:
            breach_id: Unique breach identifier.
            status: New status (detected, acknowledged, investigating,
                resolved).
            resolution_notes: Optional resolution notes.
            metadata: Optional additional metadata to merge.

        Returns:
            Updated breach dictionary.

        Raises:
            ValueError: If breach not found.
        """
        breach = self._breaches.get(breach_id)
        if breach is None:
            raise ValueError(f"Breach {breach_id} not found")

        if status is not None:
            breach["status"] = status
            if status == "resolved":
                breach["resolved_at"] = _utcnow().isoformat()
        if resolution_notes is not None:
            breach["resolution_notes"] = resolution_notes
        if metadata is not None:
            breach.setdefault("metadata", {}).update(metadata)

        breach["updated_at"] = _utcnow().isoformat()
        breach["provenance_hash"] = _compute_hash({
            "breach_id": breach_id,
            "status": breach["status"],
            "updated_at": breach["updated_at"],
        })

        self._provenance.record(
            "breach", breach_id, "update", breach["provenance_hash"],
        )
        logger.info("Updated breach %s to status %s", breach_id, status)
        return breach

    # ------------------------------------------------------------------
    # Alert Management
    # ------------------------------------------------------------------

    def list_alerts(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List generated alerts with optional filters.

        Args:
            severity: Optional severity filter.
            status: Optional status filter (open, acknowledged, resolved).
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with alerts list and pagination metadata.
        """
        items = list(self._alerts.values())
        if severity is not None:
            items = [a for a in items if a.get("severity") == severity]
        if status is not None:
            items = [a for a in items if a.get("status") == status]
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "alerts": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def get_predictions(
        self,
        dataset_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get refresh predictions with optional dataset filter.

        Args:
            dataset_id: Optional dataset ID filter.
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dictionary with predictions list and pagination metadata.
        """
        items = list(self._predictions.values())
        if dataset_id is not None:
            items = [p for p in items if p.get("dataset_id") == dataset_id]
        total = len(items)
        page = items[offset:offset + limit]
        return {
            "predictions": page,
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        dataset_ids: Optional[List[str]] = None,
        run_predictions: bool = True,
        generate_alerts: bool = True,
    ) -> Dict[str, Any]:
        """Run the full freshness monitoring pipeline end-to-end.

        Executes: check freshness -> detect staleness -> predict refreshes
        -> evaluate SLAs -> generate alerts -> produce provenance trail.

        Args:
            dataset_ids: Dataset IDs to monitor. If None, monitors all
                registered datasets.
            run_predictions: Whether to run refresh predictions.
            generate_alerts: Whether to generate alerts for breaches.

        Returns:
            Dictionary with full pipeline results.
        """
        start_t = time.time()
        pipeline_id = str(uuid.uuid4())

        ids = dataset_ids or list(self._datasets.keys())

        # Step 1: Batch freshness check
        batch_result = self.run_batch_check(dataset_ids=ids)

        # Step 2: Detect staleness patterns (fallback)
        staleness_patterns: List[Dict[str, Any]] = []
        if self._staleness_detector is not None:
            try:
                patterns = self._staleness_detector.detect(
                    checks=batch_result.get("results", []),
                )
                staleness_patterns = patterns.get("patterns", [])
            except Exception as exc:
                logger.warning(
                    "Staleness detector error: %s", exc,
                )
        else:
            # Fallback: flag datasets with critical SLA status
            for check in batch_result.get("results", []):
                if check.get("sla_status") == "critical":
                    staleness_patterns.append({
                        "pattern_id": str(uuid.uuid4()),
                        "dataset_id": check.get("dataset_id", ""),
                        "pattern_type": "sla_critical",
                        "severity": "high",
                        "detected_at": _utcnow().isoformat(),
                    })

        # Step 3: Refresh predictions (optional)
        predictions: List[Dict[str, Any]] = []
        if run_predictions and self._refresh_predictor is not None:
            try:
                pred_result = self._refresh_predictor.predict(
                    dataset_ids=ids,
                )
                predictions = pred_result.get("predictions", [])
                for pred in predictions:
                    pred_id = pred.get("prediction_id", str(uuid.uuid4()))
                    pred["prediction_id"] = pred_id
                    self._predictions[pred_id] = pred
                    self._stats["total_predictions"] += 1
            except Exception as exc:
                logger.warning(
                    "Refresh predictor error: %s", exc,
                )

        # Step 4: Generate alerts for breaches (optional)
        alerts: List[Dict[str, Any]] = []
        if generate_alerts:
            for check in batch_result.get("results", []):
                breach = check.get("sla_breach")
                if breach is not None:
                    alert = self._generate_alert(
                        breach_id=breach.get("breach_id", ""),
                        dataset_id=check.get("dataset_id", ""),
                        severity=breach.get("severity", "warning"),
                        message=(
                            f"SLA breach for dataset "
                            f"{check.get('dataset_id', 'unknown')}: "
                            f"age={check.get('age_hours', 0):.1f}h"
                        ),
                    )
                    alerts.append(alert)

        elapsed = (time.time() - start_t) * 1000.0

        result = {
            "pipeline_id": pipeline_id,
            "dataset_ids": ids,
            "batch_result": {
                "batch_id": batch_result.get("batch_id", ""),
                "total_checked": batch_result.get("total_checked", 0),
                "total_errors": batch_result.get("total_errors", 0),
            },
            "staleness_patterns": staleness_patterns,
            "predictions": predictions,
            "alerts": alerts,
            "status": "completed",
            "total_processing_time_ms": elapsed,
            "provenance_hash": _compute_hash({
                "pipeline_id": pipeline_id,
                "total_checked": batch_result.get("total_checked", 0),
                "total_patterns": len(staleness_patterns),
                "total_predictions": len(predictions),
                "total_alerts": len(alerts),
            }),
        }

        self._pipeline_results[pipeline_id] = result
        self._stats["total_pipelines"] += 1
        self._provenance.record(
            "pipeline", pipeline_id, "monitor",
            result["provenance_hash"],
        )
        logger.info(
            "Pipeline %s completed: %d checked, %d patterns, "
            "%d predictions, %d alerts in %.1fms",
            pipeline_id,
            batch_result.get("total_checked", 0),
            len(staleness_patterns),
            len(predictions),
            len(alerts),
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_freshness(self, age_hours: float) -> Dict[str, Any]:
        """Compute freshness level and score from age in hours.

        Uses a 5-tier piecewise scoring model:
        - excellent: age < 1h  -> score 1.0
        - good:      age < 6h  -> score 0.8
        - fair:      age < 24h -> score 0.6
        - poor:      age < 72h -> score 0.4
        - stale:     age >= 72h -> score 0.2

        Args:
            age_hours: Dataset age in hours since last refresh.

        Returns:
            Dictionary with freshness level and score.
        """
        if age_hours < 1.0:
            return {"level": "excellent", "score": 1.0}
        elif age_hours < 6.0:
            return {"level": "good", "score": 0.8}
        elif age_hours < 24.0:
            return {"level": "fair", "score": 0.6}
        elif age_hours < 72.0:
            return {"level": "poor", "score": 0.4}
        else:
            return {"level": "stale", "score": 0.2}

    def _evaluate_sla(
        self,
        dataset_id: str,
        age_hours: float,
    ) -> Dict[str, Any]:
        """Evaluate SLA compliance for a dataset.

        Checks the dataset's age against applicable SLA thresholds
        and creates breach records for violations.

        Args:
            dataset_id: Dataset identifier.
            age_hours: Current age in hours.

        Returns:
            Dictionary with SLA evaluation results.
        """
        # Find applicable SLA
        applicable_slas = [
            s for s in self._sla_definitions.values()
            if s.get("dataset_id") == dataset_id
            and s.get("status") == "active"
        ]

        # Use default thresholds if no SLA defined
        warning_hours = 24.0
        critical_hours = 72.0
        sla_id = ""

        if applicable_slas:
            sla = applicable_slas[0]
            warning_hours = sla.get("warning_hours", 24.0)
            critical_hours = sla.get("critical_hours", 72.0)
            sla_id = sla.get("sla_id", "")

        if age_hours >= critical_hours:
            breach = {
                "breach_id": str(uuid.uuid4()),
                "dataset_id": dataset_id,
                "sla_id": sla_id,
                "severity": "critical",
                "age_hours": round(age_hours, 4),
                "threshold_hours": critical_hours,
                "status": "detected",
                "detected_at": _utcnow().isoformat(),
                "resolved_at": None,
                "provenance_hash": _compute_hash({
                    "dataset_id": dataset_id,
                    "severity": "critical",
                    "age_hours": round(age_hours, 4),
                }),
            }
            return {"sla_status": "critical", "breach": breach}
        elif age_hours >= warning_hours:
            breach = {
                "breach_id": str(uuid.uuid4()),
                "dataset_id": dataset_id,
                "sla_id": sla_id,
                "severity": "warning",
                "age_hours": round(age_hours, 4),
                "threshold_hours": warning_hours,
                "status": "detected",
                "detected_at": _utcnow().isoformat(),
                "resolved_at": None,
                "provenance_hash": _compute_hash({
                    "dataset_id": dataset_id,
                    "severity": "warning",
                    "age_hours": round(age_hours, 4),
                }),
            }
            return {"sla_status": "warning", "breach": breach}
        else:
            return {"sla_status": "compliant", "breach": None}

    def _generate_alert(
        self,
        breach_id: str,
        dataset_id: str,
        severity: str,
        message: str,
    ) -> Dict[str, Any]:
        """Generate an alert for a breach.

        Args:
            breach_id: Breach that triggered this alert.
            dataset_id: Dataset involved.
            severity: Alert severity level.
            message: Alert message.

        Returns:
            Dictionary with alert details.
        """
        alert_id = str(uuid.uuid4())
        alert = {
            "alert_id": alert_id,
            "breach_id": breach_id,
            "dataset_id": dataset_id,
            "severity": severity,
            "message": message,
            "status": "open",
            "created_at": _utcnow().isoformat(),
            "acknowledged_at": None,
            "resolved_at": None,
            "provenance_hash": _compute_hash({
                "alert_id": alert_id,
                "breach_id": breach_id,
                "severity": severity,
            }),
        }

        if self._alert_manager is not None:
            try:
                result = self._alert_manager.send(
                    alert_id=alert_id,
                    breach_id=breach_id,
                    dataset_id=dataset_id,
                    severity=severity,
                    message=message,
                )
                alert.update(result)
            except Exception as exc:
                logger.warning(
                    "Alert manager error, using fallback: %s", exc,
                )

        self._alerts[alert_id] = alert
        self._stats["total_alerts"] += 1
        self._provenance.record(
            "alert", alert_id, "generate", alert["provenance_hash"],
        )
        return alert


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_instance: Optional[DataFreshnessMonitorService] = None
_service_lock = threading.Lock()


def get_service() -> DataFreshnessMonitorService:
    """Return the singleton DataFreshnessMonitorService.

    Thread-safe lazy initialization. Returns the same instance
    on every call within the process.

    Returns:
        The global DataFreshnessMonitorService singleton.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = DataFreshnessMonitorService()
                _service_instance.startup()
    return _service_instance


def set_service(
    service: DataFreshnessMonitorService,
) -> None:
    """Replace the singleton DataFreshnessMonitorService.

    Primarily used for testing to inject mock services.

    Args:
        service: New service instance to install.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("DataFreshnessMonitorService replaced programmatically")


def reset_service() -> DataFreshnessMonitorService:
    """Reset and return a new singleton instance.

    Returns:
        A fresh DataFreshnessMonitorService singleton.
    """
    global _service_instance
    with _service_lock:
        _service_instance = DataFreshnessMonitorService()
        _service_instance.startup()
    return _service_instance


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------


def configure_freshness_monitor(
    app: Any,
) -> DataFreshnessMonitorService:
    """Configure the freshness monitor service on a FastAPI app.

    Attaches the service to ``app.state.data_freshness_monitor_service``
    and optionally includes the router.

    Args:
        app: FastAPI application instance.

    Returns:
        The configured DataFreshnessMonitorService.
    """
    service = get_service()
    app.state.data_freshness_monitor_service = service

    # Attempt to include the router
    try:
        from greenlang.data_freshness_monitor.api.router import router
        if router is not None:
            app.include_router(router)
    except ImportError:
        logger.warning(
            "Freshness monitor router not available; skipping route "
            "registration"
        )

    logger.info("Data freshness monitor service configured on app")
    return service


def get_freshness_monitor(
    app: Any,
) -> Optional[DataFreshnessMonitorService]:
    """Retrieve the freshness monitor service from a FastAPI app.

    Args:
        app: FastAPI application instance.

    Returns:
        DataFreshnessMonitorService or None if not configured.
    """
    return getattr(
        app.state, "data_freshness_monitor_service", None,
    )


def get_router() -> Any:
    """Return the FastAPI APIRouter for the freshness monitor service.

    Returns:
        FastAPI APIRouter instance or None if FastAPI is not available.
    """
    try:
        from greenlang.data_freshness_monitor.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "DataFreshnessMonitorService",
    "configure_freshness_monitor",
    "get_freshness_monitor",
    "get_router",
    "get_service",
    "set_service",
    "reset_service",
    "_compute_hash",
]
