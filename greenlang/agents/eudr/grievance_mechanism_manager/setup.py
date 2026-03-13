# -*- coding: utf-8 -*-
"""
Grievance Mechanism Manager Service Facade - AGENT-EUDR-032

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point for grievance analytics, root cause
analysis, mediation workflows, remediation tracking, risk scoring,
collective grievance handling, and regulatory reporting.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-032 (GL-EUDR-GMM-032)
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
_AGENT_ID = "GL-EUDR-GMM-032"

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

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig, get_config,
)

try:
    from greenlang.agents.eudr.grievance_mechanism_manager.provenance import (
        ProvenanceTracker, GENESIS_HASH,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    GENESIS_HASH = "0" * 64

# Engine imports (conditional)
try:
    from greenlang.agents.eudr.grievance_mechanism_manager.grievance_analytics_engine import GrievanceAnalyticsEngine
except ImportError:
    GrievanceAnalyticsEngine = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.grievance_mechanism_manager.root_cause_analyzer import RootCauseAnalyzer
except ImportError:
    RootCauseAnalyzer = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.grievance_mechanism_manager.mediation_workflow_manager import MediationWorkflowManager
except ImportError:
    MediationWorkflowManager = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.grievance_mechanism_manager.remediation_tracker import RemediationTracker
except ImportError:
    RemediationTracker = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.grievance_mechanism_manager.risk_scoring_engine import RiskScoringEngine
except ImportError:
    RiskScoringEngine = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.grievance_mechanism_manager.collective_grievance_handler import CollectiveGrievanceHandler
except ImportError:
    CollectiveGrievanceHandler = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.grievance_mechanism_manager.regulatory_reporter import RegulatoryReporter
except ImportError:
    RegulatoryReporter = None  # type: ignore[misc,assignment]


class GrievanceMechanismManagerService:
    """Unified service facade for AGENT-EUDR-032."""

    def __init__(self, config: Optional[GrievanceMechanismManagerConfig] = None) -> None:
        self.config = config or get_config()
        self._provenance = ProvenanceTracker() if ProvenanceTracker else None
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None
        self._engines: Dict[str, Any] = {}
        self._initialized = False
        logger.info("GrievanceMechanismManagerService created")

    async def startup(self) -> None:
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
        logger.info(f"GrievanceMechanismManagerService startup: {len(self._engines)}/7 engines in {elapsed:.1f}ms")

    def _init_engines(self) -> None:
        specs: List[Tuple[str, Any]] = [
            ("grievance_analytics", GrievanceAnalyticsEngine),
            ("root_cause_analyzer", RootCauseAnalyzer),
            ("mediation_workflow", MediationWorkflowManager),
            ("remediation_tracker", RemediationTracker),
            ("risk_scoring", RiskScoringEngine),
            ("collective_handler", CollectiveGrievanceHandler),
            ("regulatory_reporter", RegulatoryReporter),
        ]
        for name, cls in specs:
            if cls is not None:
                try:
                    self._engines[name] = cls(config=self.config)
                except Exception as e:
                    logger.warning(f"Engine '{name}' init failed: {e}")

    async def shutdown(self) -> None:
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

    # -- Analytics --
    async def analyze_patterns(self, operator_id: str, grievances: List[Dict[str, Any]]) -> Any:
        engine = self._engines.get("grievance_analytics")
        if engine:
            return await engine.analyze_patterns(operator_id, grievances)
        raise RuntimeError("GrievanceAnalyticsEngine not available")

    async def get_analytics(self, analytics_id: str) -> Any:
        engine = self._engines.get("grievance_analytics")
        return await engine.get_analytics(analytics_id) if engine else None

    async def list_analytics(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("grievance_analytics")
        return await engine.list_analytics(**kwargs) if engine else []

    # -- Root Cause --
    async def analyze_root_cause(self, grievance_id: str, operator_id: str, data: Dict[str, Any], method: Optional[str] = None) -> Any:
        engine = self._engines.get("root_cause_analyzer")
        if engine:
            return await engine.analyze(grievance_id, operator_id, data, method)
        raise RuntimeError("RootCauseAnalyzer not available")

    async def get_root_cause(self, root_cause_id: str) -> Any:
        engine = self._engines.get("root_cause_analyzer")
        return await engine.get_root_cause(root_cause_id) if engine else None

    async def list_root_causes(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("root_cause_analyzer")
        return await engine.list_root_causes(**kwargs) if engine else []

    # -- Mediation --
    async def initiate_mediation(self, grievance_id: str, operator_id: str, parties: List[Dict[str, Any]], mediator_type: str = "internal", mediator_id: Optional[str] = None) -> Any:
        engine = self._engines.get("mediation_workflow")
        if engine:
            return await engine.initiate_mediation(grievance_id, operator_id, parties, mediator_type, mediator_id)
        raise RuntimeError("MediationWorkflowManager not available")

    async def advance_mediation(self, mediation_id: str, target_stage: Optional[str] = None) -> Any:
        engine = self._engines.get("mediation_workflow")
        if engine:
            return await engine.advance_stage(mediation_id, target_stage)
        raise RuntimeError("MediationWorkflowManager not available")

    async def record_mediation_session(self, mediation_id: str, session_data: Dict[str, Any]) -> Any:
        engine = self._engines.get("mediation_workflow")
        if engine:
            return await engine.record_session(mediation_id, session_data)
        raise RuntimeError("MediationWorkflowManager not available")

    async def record_mediation_agreement(self, mediation_id: str, agreement_data: Dict[str, Any]) -> Any:
        engine = self._engines.get("mediation_workflow")
        if engine:
            return await engine.record_agreement(mediation_id, agreement_data)
        raise RuntimeError("MediationWorkflowManager not available")

    async def set_mediation_settlement(self, mediation_id: str, terms: Dict[str, Any], status: str = "accepted") -> Any:
        engine = self._engines.get("mediation_workflow")
        if engine:
            return await engine.set_settlement(mediation_id, terms, status)
        raise RuntimeError("MediationWorkflowManager not available")

    async def get_mediation(self, mediation_id: str) -> Any:
        engine = self._engines.get("mediation_workflow")
        return await engine.get_mediation(mediation_id) if engine else None

    async def list_mediations(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("mediation_workflow")
        return await engine.list_mediations(**kwargs) if engine else []

    # -- Remediation --
    async def create_remediation(self, grievance_id: str, operator_id: str, rem_type: str, actions: Optional[List[Dict[str, Any]]] = None) -> Any:
        engine = self._engines.get("remediation_tracker")
        if engine:
            return await engine.create_remediation(grievance_id, operator_id, rem_type, actions)
        raise RuntimeError("RemediationTracker not available")

    async def update_remediation_progress(self, remediation_id: str, completion: float, status: Optional[str] = None) -> Any:
        engine = self._engines.get("remediation_tracker")
        if engine:
            return await engine.update_progress(remediation_id, completion, status)
        raise RuntimeError("RemediationTracker not available")

    async def verify_remediation(self, remediation_id: str, evidence: List[Dict[str, Any]], indicators: Optional[Dict[str, Any]] = None) -> Any:
        engine = self._engines.get("remediation_tracker")
        if engine:
            return await engine.verify_remediation(remediation_id, evidence, indicators)
        raise RuntimeError("RemediationTracker not available")

    async def record_remediation_satisfaction(self, remediation_id: str, score: float) -> Any:
        engine = self._engines.get("remediation_tracker")
        if engine:
            return await engine.record_satisfaction(remediation_id, score)
        raise RuntimeError("RemediationTracker not available")

    async def get_remediation(self, remediation_id: str) -> Any:
        engine = self._engines.get("remediation_tracker")
        return await engine.get_remediation(remediation_id) if engine else None

    async def list_remediations(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("remediation_tracker")
        return await engine.list_remediations(**kwargs) if engine else []

    # -- Risk Scoring --
    async def compute_risk_score(self, operator_id: str, scope: str, scope_id: str, grievances: List[Dict[str, Any]]) -> Any:
        engine = self._engines.get("risk_scoring")
        if engine:
            return await engine.compute_risk_score(operator_id, scope, scope_id, grievances)
        raise RuntimeError("RiskScoringEngine not available")

    async def get_risk_score(self, risk_score_id: str) -> Any:
        engine = self._engines.get("risk_scoring")
        return await engine.get_risk_score(risk_score_id) if engine else None

    async def list_risk_scores(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("risk_scoring")
        return await engine.list_risk_scores(**kwargs) if engine else []

    # -- Collective Grievances --
    async def create_collective(self, operator_id: str, title: str, individual_ids: Optional[List[str]] = None, description: str = "", category: str = "process", lead_id: Optional[str] = None, affected: int = 1) -> Any:
        engine = self._engines.get("collective_handler")
        if engine:
            return await engine.create_collective(operator_id, title, individual_ids, description, category, lead_id, affected)
        raise RuntimeError("CollectiveGrievanceHandler not available")

    async def add_collective_demands(self, collective_id: str, demands: List[Dict[str, Any]]) -> Any:
        engine = self._engines.get("collective_handler")
        if engine:
            return await engine.add_demands(collective_id, demands)
        raise RuntimeError("CollectiveGrievanceHandler not available")

    async def update_collective_status(self, collective_id: str, status: str) -> Any:
        engine = self._engines.get("collective_handler")
        if engine:
            return await engine.update_status(collective_id, status)
        raise RuntimeError("CollectiveGrievanceHandler not available")

    async def get_collective(self, collective_id: str) -> Any:
        engine = self._engines.get("collective_handler")
        return await engine.get_collective(collective_id) if engine else None

    async def list_collectives(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("collective_handler")
        return await engine.list_collectives(**kwargs) if engine else []

    # -- Regulatory Reports --
    async def generate_regulatory_report(self, operator_id: str, report_type: str, grievances: Optional[List[Dict[str, Any]]] = None, remediations: Optional[List[Dict[str, Any]]] = None) -> Any:
        engine = self._engines.get("regulatory_reporter")
        if engine:
            return await engine.generate_report(operator_id, report_type, grievances, remediations=remediations)
        raise RuntimeError("RegulatoryReporter not available")

    async def get_report(self, report_id: str) -> Any:
        engine = self._engines.get("regulatory_reporter")
        return await engine.get_report(report_id) if engine else None

    async def list_reports(self, **kwargs: Any) -> List[Any]:
        engine = self._engines.get("regulatory_reporter")
        return await engine.list_reports(**kwargs) if engine else []

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
_service_instance: Optional[GrievanceMechanismManagerService] = None


def get_service(config: Optional[GrievanceMechanismManagerConfig] = None) -> GrievanceMechanismManagerService:
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = GrievanceMechanismManagerService(config)
    return _service_instance


def reset_service() -> None:
    global _service_instance
    with _service_lock:
        _service_instance = None


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    service = get_service()
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()
