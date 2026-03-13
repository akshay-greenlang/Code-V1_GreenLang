# -*- coding: utf-8 -*-
"""
Continuous Monitoring Service Facade - AGENT-EUDR-033

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point for supply chain monitoring,
deforestation alert correlation, compliance auditing, change detection,
risk score tracking, data freshness validation, and regulatory tracking.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 (GL-EUDR-CM-033)
Status: Production Ready
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-CM-033"

try:
    from psycopg_pool import AsyncConnectionPool
    PSYCOPG_POOL_AVAILABLE = True
except ImportError:
    AsyncConnectionPool = None  # type: ignore[assignment,misc]
    PSYCOPG_POOL_AVAILABLE = False

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False

from greenlang.agents.eudr.continuous_monitoring.config import (
    ContinuousMonitoringConfig, get_config,
)

try:
    from greenlang.agents.eudr.continuous_monitoring.provenance import (
        ProvenanceTracker, GENESIS_HASH,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    GENESIS_HASH = "0" * 64

# Engine imports (conditional)
try:
    from greenlang.agents.eudr.continuous_monitoring.supply_chain_monitor import SupplyChainMonitor
except ImportError:
    SupplyChainMonitor = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.continuous_monitoring.deforestation_monitor import DeforestationMonitor
except ImportError:
    DeforestationMonitor = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.continuous_monitoring.compliance_checker import ComplianceChecker
except ImportError:
    ComplianceChecker = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.continuous_monitoring.change_detector import ChangeDetector
except ImportError:
    ChangeDetector = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.continuous_monitoring.risk_score_monitor import RiskScoreMonitor
except ImportError:
    RiskScoreMonitor = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.continuous_monitoring.data_freshness_validator import DataFreshnessValidator
except ImportError:
    DataFreshnessValidator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.continuous_monitoring.regulatory_tracker import RegulatoryTracker
except ImportError:
    RegulatoryTracker = None  # type: ignore[misc,assignment]


class ContinuousMonitoringService:
    """Unified service facade for AGENT-EUDR-033."""

    def __init__(self, config: Optional[ContinuousMonitoringConfig] = None) -> None:
        self.config = config or get_config()
        self._provenance = ProvenanceTracker() if ProvenanceTracker else None
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None
        self._engines: Dict[str, Any] = {}
        self._initialized = False
        logger.info("ContinuousMonitoringService created")

    async def startup(self) -> None:
        """Initialize database connections and engines."""
        start = time.monotonic()
        if PSYCOPG_POOL_AVAILABLE:
            try:
                db_url = (
                    f"host={self.config.db_host} port={self.config.db_port} "
                    f"dbname={self.config.db_name} user={self.config.db_user} "
                    f"password={self.config.db_password}"
                )
                self._db_pool = AsyncConnectionPool(
                    conninfo=db_url, min_size=self.config.db_pool_min,
                    max_size=self.config.db_pool_max, open=False,
                )
                await self._db_pool.open()
            except Exception as e:
                logger.warning(f"PostgreSQL pool init failed: {e}")
                self._db_pool = None

        if REDIS_AVAILABLE:
            try:
                redis_url = f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}"
                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception as e:
                logger.warning(f"Redis init failed: {e}")
                self._redis = None

        self._init_engines()
        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000
        logger.info(f"ContinuousMonitoringService startup: {len(self._engines)}/7 engines in {elapsed:.1f}ms")

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines."""
        specs: List[Tuple[str, Any]] = [
            ("supply_chain_monitor", SupplyChainMonitor),
            ("deforestation_monitor", DeforestationMonitor),
            ("compliance_checker", ComplianceChecker),
            ("change_detector", ChangeDetector),
            ("risk_score_monitor", RiskScoreMonitor),
            ("data_freshness_validator", DataFreshnessValidator),
            ("regulatory_tracker", RegulatoryTracker),
        ]
        for name, cls in specs:
            if cls is not None:
                try:
                    self._engines[name] = cls(config=self.config)
                except Exception as e:
                    logger.warning(f"Engine '{name}' init failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown all engines and connections."""
        for name, engine in self._engines.items():
            if hasattr(engine, "shutdown"):
                try:
                    result = engine.shutdown()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass
        if self._db_pool:
            try:
                await self._db_pool.close()
            except Exception:
                pass
        self._initialized = False

    # -- Supply Chain Monitor --
    async def scan_supply_chain(self, operator_id: str, suppliers: List[Dict[str, Any]]) -> Any:
        engine = self._engines.get("supply_chain_monitor")
        if engine:
            return await engine.scan_supply_chain(operator_id, suppliers)
        raise RuntimeError("SupplyChainMonitor not available")

    async def get_scan(self, scan_id: str) -> Any:
        engine = self._engines.get("supply_chain_monitor")
        return await engine.get_scan(scan_id) if engine else None

    async def list_scans(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("supply_chain_monitor")
        return await engine.list_scans(**kwargs) if engine else []

    async def list_alerts(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("supply_chain_monitor")
        return await engine.list_alerts(**kwargs) if engine else []

    # -- Deforestation Monitor --
    async def check_deforestation(self, operator_id: str, alerts: List[Dict[str, Any]], entities: Optional[List[Dict[str, Any]]] = None) -> Any:
        engine = self._engines.get("deforestation_monitor")
        if engine:
            return await engine.check_deforestation_alerts(operator_id, alerts, entities)
        raise RuntimeError("DeforestationMonitor not available")

    async def get_deforestation_record(self, monitor_id: str) -> Any:
        engine = self._engines.get("deforestation_monitor")
        return await engine.get_record(monitor_id) if engine else None

    async def list_deforestation_records(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("deforestation_monitor")
        return await engine.list_records(**kwargs) if engine else []

    async def get_investigation(self, investigation_id: str) -> Any:
        engine = self._engines.get("deforestation_monitor")
        return await engine.get_investigation(investigation_id) if engine else None

    async def list_investigations(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("deforestation_monitor")
        return await engine.list_investigations(**kwargs) if engine else []

    # -- Compliance Checker --
    async def run_compliance_audit(self, operator_id: str, operator_data: Dict[str, Any]) -> Any:
        engine = self._engines.get("compliance_checker")
        if engine:
            return await engine.run_compliance_audit(operator_id, operator_data)
        raise RuntimeError("ComplianceChecker not available")

    async def get_audit(self, audit_id: str) -> Any:
        engine = self._engines.get("compliance_checker")
        return await engine.get_audit(audit_id) if engine else None

    async def list_audits(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("compliance_checker")
        return await engine.list_audits(**kwargs) if engine else []

    # -- Change Detector --
    async def detect_changes(self, operator_id: str, entity_snapshots: List[Dict[str, Any]]) -> Any:
        engine = self._engines.get("change_detector")
        if engine:
            return await engine.detect_changes(operator_id, entity_snapshots)
        raise RuntimeError("ChangeDetector not available")

    async def get_change(self, detection_id: str) -> Any:
        engine = self._engines.get("change_detector")
        return await engine.get_detection(detection_id) if engine else None

    async def list_changes(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("change_detector")
        return await engine.list_detections(**kwargs) if engine else []

    # -- Risk Score Monitor --
    async def monitor_risk_scores(self, operator_id: str, entity_id: str, score_history: List[Dict[str, Any]], entity_type: str = "supplier", incidents: Optional[List[Dict[str, Any]]] = None) -> Any:
        engine = self._engines.get("risk_score_monitor")
        if engine:
            return await engine.monitor_risk_scores(operator_id, entity_id, score_history, entity_type, incidents)
        raise RuntimeError("RiskScoreMonitor not available")

    async def get_risk_monitor(self, monitor_id: str) -> Any:
        engine = self._engines.get("risk_score_monitor")
        return await engine.get_record(monitor_id) if engine else None

    async def list_risk_monitors(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("risk_score_monitor")
        return await engine.list_records(**kwargs) if engine else []

    # -- Data Freshness Validator --
    async def validate_freshness(self, operator_id: str, entities: List[Dict[str, Any]]) -> Any:
        engine = self._engines.get("data_freshness_validator")
        if engine:
            return await engine.validate_data_age(operator_id, entities)
        raise RuntimeError("DataFreshnessValidator not available")

    async def get_freshness_record(self, freshness_id: str) -> Any:
        engine = self._engines.get("data_freshness_validator")
        return await engine.get_record(freshness_id) if engine else None

    async def list_freshness_records(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("data_freshness_validator")
        return await engine.list_records(**kwargs) if engine else []

    async def generate_freshness_report(self, operator_id: str) -> Dict[str, Any]:
        engine = self._engines.get("data_freshness_validator")
        if engine:
            return await engine.generate_freshness_reports(operator_id)
        return {"error": "DataFreshnessValidator not available"}

    # -- Regulatory Tracker --
    async def check_regulatory(self, operator_id: str, updates: Optional[List[Dict[str, Any]]] = None) -> Any:
        engine = self._engines.get("regulatory_tracker")
        if engine:
            return await engine.fetch_regulatory_updates(operator_id, updates)
        raise RuntimeError("RegulatoryTracker not available")

    async def get_regulatory_record(self, tracking_id: str) -> Any:
        engine = self._engines.get("regulatory_tracker")
        return await engine.get_record(tracking_id) if engine else None

    async def list_regulatory_records(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("regulatory_tracker")
        return await engine.list_records(**kwargs) if engine else []

    # -- Dashboard --
    async def get_dashboard(self, operator_id: str) -> Dict[str, Any]:
        """Get aggregated dashboard data for an operator."""
        dashboard: Dict[str, Any] = {
            "agent_id": _AGENT_ID,
            "operator_id": operator_id,
            "engines": {},
        }

        for name, engine in self._engines.items():
            try:
                health = await engine.health_check()
                dashboard["engines"][name] = health
            except Exception as e:
                dashboard["engines"][name] = {"status": "error", "error": str(e)[:100]}

        return dashboard

    # -- Health --
    async def health_check(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "agent_id": _AGENT_ID, "version": _VERSION,
            "status": "healthy", "initialized": self._initialized,
            "engines": {}, "connections": {},
        }
        from datetime import datetime, timezone
        result["timestamp"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        for name, engine in self._engines.items():
            if hasattr(engine, "health_check"):
                try:
                    result["engines"][name] = await engine.health_check()
                except Exception as e:
                    result["engines"][name] = {"status": "error", "error": str(e)}
            else:
                result["engines"][name] = {"status": "available"}

        unhealthy = sum(
            1 for v in result["engines"].values()
            if isinstance(v, dict) and v.get("status") in ("error", "not_loaded")
        )
        if unhealthy > 3:
            result["status"] = "unhealthy"
        elif unhealthy > 0:
            result["status"] = "degraded"
        return result

    @property
    def engine_count(self) -> int:
        return len(self._engines)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


_service_lock = threading.Lock()
_service_instance: Optional[ContinuousMonitoringService] = None


def get_service(config: Optional[ContinuousMonitoringConfig] = None) -> ContinuousMonitoringService:
    """Return the thread-safe singleton service instance."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ContinuousMonitoringService(config)
    return _service_instance


def reset_service() -> None:
    """Reset the singleton (for testing only)."""
    global _service_instance
    with _service_lock:
        _service_instance = None


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager."""
    service = get_service()
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()
