# -*- coding: utf-8 -*-
"""
Stakeholder Engagement Service Facade - AGENT-EUDR-031

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point. Provides the primary API used by
the FastAPI router layer to manage stakeholder mapping, FPIC workflows,
grievance mechanism operations, consultation records, multi-channel
communications, engagement quality assessment, and compliance reporting
per EUDR stakeholder engagement requirements.

Engines (7):
    1. StakeholderMapper              - Centralized stakeholder registry
    2. FPICWorkflowEngine             - FPIC workflow management
    3. GrievanceMechanism             - Complaint management system
    4. ConsultationRecordManager      - Consultation record keeping
    5. CommunicationHub               - Multi-channel communications
    6. IndigenousRightsEngagementVerifier - Engagement quality verification
    7. ComplianceReporter             - Compliance report generation

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-031 Stakeholder Engagement Tool (GL-EUDR-SET-031)
Regulation: EU 2023/1115 (EUDR) Articles 2, 4, 8, 9, 10, 11, 12, 29, 31
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
_AGENT_ID = "GL-EUDR-SET-031"

# ---------------------------------------------------------------------------
# Optional dependency imports with graceful fallback
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
    get_config,
)

try:
    from greenlang.agents.eudr.stakeholder_engagement.provenance import (
        ProvenanceTracker,
        GENESIS_HASH,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

# Engine imports (conditional)
try:
    from greenlang.agents.eudr.stakeholder_engagement.stakeholder_mapper import StakeholderMapper
except ImportError:
    StakeholderMapper = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.stakeholder_engagement.fpic_workflow_engine import FPICWorkflowEngine
except ImportError:
    FPICWorkflowEngine = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.stakeholder_engagement.grievance_mechanism import GrievanceMechanism
except ImportError:
    GrievanceMechanism = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.stakeholder_engagement.consultation_record_manager import ConsultationRecordManager
except ImportError:
    ConsultationRecordManager = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.stakeholder_engagement.communication_hub import CommunicationHub
except ImportError:
    CommunicationHub = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.stakeholder_engagement.engagement_verifier import IndigenousRightsEngagementVerifier
except ImportError:
    IndigenousRightsEngagementVerifier = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.stakeholder_engagement.compliance_reporter import ComplianceReporter
except ImportError:
    ComplianceReporter = None  # type: ignore[misc,assignment]

# Metrics imports (conditional)
try:
    from greenlang.agents.eudr.stakeholder_engagement.metrics import (
        record_stakeholder_registered,
        record_fpic_workflow_started,
        record_fpic_consent_granted,
        record_grievance_submitted,
        record_grievance_resolved,
        record_consultation_recorded,
        record_communication_sent,
        record_assessment_completed,
        record_report_generated,
        observe_fpic_workflow_duration,
        observe_grievance_resolution_duration,
        observe_consultation_duration,
        observe_engagement_assessment_duration,
        observe_report_generation_duration,
        set_active_stakeholders,
        set_active_fpic_workflows,
        set_open_grievances,
        set_pending_communications,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


def _safe_metric(fn: Any, *args: Any) -> None:
    """Safely call a metrics function if available."""
    if METRICS_AVAILABLE and fn is not None:
        try:
            fn(*args)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Service Facade
# ---------------------------------------------------------------------------


class StakeholderEngagementService:
    """Unified service facade for AGENT-EUDR-031.

    Aggregates all 7 processing engines and provides a clean API for
    stakeholder engagement lifecycle management.

    Attributes:
        config: Agent configuration.
        _initialized: Whether startup has completed.
    """

    def __init__(
        self,
        config: Optional[StakeholderEngagementConfig] = None,
    ) -> None:
        """Initialize the service facade.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()

        # Provenance tracker
        if ProvenanceTracker is not None:
            self._provenance = ProvenanceTracker()
        else:
            self._provenance = None

        # Database / cache handles
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine references
        self._stakeholder_mapper: Optional[Any] = None
        self._fpic_engine: Optional[Any] = None
        self._grievance_mechanism: Optional[Any] = None
        self._consultation_manager: Optional[Any] = None
        self._communication_hub: Optional[Any] = None
        self._engagement_verifier: Optional[Any] = None
        self._compliance_reporter: Optional[Any] = None
        self._engines: Dict[str, Any] = {}

        self._initialized = False
        logger.info("StakeholderEngagementService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all engines and external connections."""
        start = time.monotonic()
        logger.info("StakeholderEngagementService startup initiated")

        # Initialize database pool
        if PSYCOPG_POOL_AVAILABLE:
            try:
                db_url = (
                    f"host={self.config.db_host} port={self.config.db_port} "
                    f"dbname={self.config.db_name} user={self.config.db_user} "
                    f"password={self.config.db_password}"
                )
                self._db_pool = AsyncConnectionPool(
                    conninfo=db_url,
                    min_size=self.config.db_pool_min,
                    max_size=self.config.db_pool_max,
                    open=False,
                )
                await self._db_pool.open()
                logger.info("PostgreSQL connection pool opened")
            except Exception as e:
                logger.warning(f"PostgreSQL pool init failed: {e}")
                self._db_pool = None

        # Initialize Redis
        if REDIS_AVAILABLE:
            try:
                redis_url = (
                    f"redis://{self.config.redis_host}:"
                    f"{self.config.redis_port}/{self.config.redis_db}"
                )
                self._redis = aioredis.from_url(
                    redis_url, decode_responses=True,
                )
                await self._redis.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis init failed: {e}")
                self._redis = None

        # Initialize engines
        self._init_engines()

        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000
        engine_count = len(self._engines)

        if self._provenance is not None:
            self._provenance.record(
                entity_type="service",
                action="startup",
                entity_id=_AGENT_ID,
                actor="system",
                metadata={
                    "engines_loaded": engine_count,
                    "startup_time_ms": round(elapsed, 2),
                    "db_available": self._db_pool is not None,
                    "redis_available": self._redis is not None,
                },
            )

        logger.info(
            f"StakeholderEngagementService startup complete: "
            f"{engine_count}/7 engines in {elapsed:.1f}ms"
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines with graceful degradation."""
        engine_specs: List[Tuple[str, Any]] = [
            ("stakeholder_mapper", StakeholderMapper),
            ("fpic_workflow_engine", FPICWorkflowEngine),
            ("grievance_mechanism", GrievanceMechanism),
            ("consultation_record_manager", ConsultationRecordManager),
            ("communication_hub", CommunicationHub),
            ("engagement_verifier", IndigenousRightsEngagementVerifier),
            ("compliance_reporter", ComplianceReporter),
        ]

        for name, engine_cls in engine_specs:
            if engine_cls is not None:
                try:
                    engine = engine_cls(config=self.config)
                    self._engines[name] = engine
                    logger.info(f"Engine '{name}' initialized")
                except Exception as e:
                    logger.warning(f"Engine '{name}' init failed: {e}")
            else:
                logger.debug(f"Engine '{name}' class not available")

        self._stakeholder_mapper = self._engines.get("stakeholder_mapper")
        self._fpic_engine = self._engines.get("fpic_workflow_engine")
        self._grievance_mechanism = self._engines.get("grievance_mechanism")
        self._consultation_manager = self._engines.get("consultation_record_manager")
        self._communication_hub = self._engines.get("communication_hub")
        self._engagement_verifier = self._engines.get("engagement_verifier")
        self._compliance_reporter = self._engines.get("compliance_reporter")

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections and engines."""
        logger.info("StakeholderEngagementService shutdown initiated")

        for name, engine in self._engines.items():
            if hasattr(engine, "shutdown"):
                try:
                    result = engine.shutdown()
                    if asyncio.iscoroutine(result):
                        await result
                    logger.info(f"Engine '{name}' shut down")
                except Exception as e:
                    logger.warning(f"Engine '{name}' shutdown error: {e}")

        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception as e:
                logger.warning(f"Redis close error: {e}")

        if self._db_pool is not None:
            try:
                await self._db_pool.close()
            except Exception as e:
                logger.warning(f"PostgreSQL pool close error: {e}")

        self._initialized = False
        logger.info("StakeholderEngagementService shutdown complete")

    # ------------------------------------------------------------------
    # Stakeholder Mapping
    # ------------------------------------------------------------------

    async def map_stakeholder(
        self, operator_id: str, stakeholder_data: Dict[str, Any],
    ) -> Any:
        """Register a new stakeholder."""
        if self._stakeholder_mapper is not None:
            result = await self._stakeholder_mapper.map_stakeholder(
                operator_id=operator_id, stakeholder_data=stakeholder_data,
            )
            _safe_metric(record_stakeholder_registered, stakeholder_data.get("type", "unknown"))
            if self._stakeholder_mapper is not None:
                count = len(self._stakeholder_mapper._stakeholders)
                _safe_metric(set_active_stakeholders, count)
            return result
        raise RuntimeError("StakeholderMapper engine not available")

    async def get_stakeholder(self, stakeholder_id: str) -> Any:
        """Get stakeholder by ID."""
        if self._stakeholder_mapper is not None:
            return await self._stakeholder_mapper.get_stakeholder(stakeholder_id)
        return None

    async def list_stakeholders(
        self, operator_id: Optional[str] = None,
        stakeholder_type: Optional[str] = None,
        country_code: Optional[str] = None,
    ) -> List[Any]:
        """List stakeholders with filters."""
        if self._stakeholder_mapper is not None:
            return await self._stakeholder_mapper.list_stakeholders(
                operator_id=operator_id, stakeholder_type=stakeholder_type,
                country_code=country_code,
            )
        return []

    # ------------------------------------------------------------------
    # FPIC Workflows
    # ------------------------------------------------------------------

    async def initiate_fpic(
        self, operator_id: str, stakeholder_id: str,
        supply_chain_node: str = "",
    ) -> Any:
        """Initiate FPIC workflow."""
        if self._fpic_engine is not None:
            result = await self._fpic_engine.initiate_fpic(
                operator_id=operator_id, stakeholder_id=stakeholder_id,
                supply_chain_node=supply_chain_node,
            )
            _safe_metric(record_fpic_workflow_started, "pending")
            return result
        raise RuntimeError("FPICWorkflowEngine not available")

    async def advance_fpic_stage(
        self, fpic_id: str, next_stage: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Advance FPIC workflow stage."""
        if self._fpic_engine is not None:
            return await self._fpic_engine.advance_stage(
                fpic_id=fpic_id, next_stage=next_stage, evidence=evidence,
            )
        raise RuntimeError("FPICWorkflowEngine not available")

    async def record_fpic_consent(
        self, fpic_id: str, consent_status: str,
        agreement_terms: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Record FPIC consent."""
        if self._fpic_engine is not None:
            result = await self._fpic_engine.record_consent(
                fpic_id=fpic_id, consent_status=consent_status,
                agreement_terms=agreement_terms,
            )
            _safe_metric(record_fpic_consent_granted, consent_status)
            return result
        raise RuntimeError("FPICWorkflowEngine not available")

    async def get_fpic_workflow(self, fpic_id: str) -> Any:
        """Get FPIC workflow by ID."""
        if self._fpic_engine is not None:
            return await self._fpic_engine.get_workflow(fpic_id)
        return None

    async def list_fpic_workflows(
        self, operator_id: Optional[str] = None,
        stakeholder_id: Optional[str] = None,
        current_stage: Optional[str] = None,
    ) -> List[Any]:
        """List FPIC workflows with filters."""
        if self._fpic_engine is not None:
            return await self._fpic_engine.list_workflows(
                operator_id=operator_id, stakeholder_id=stakeholder_id,
                current_stage=current_stage,
            )
        return []

    # ------------------------------------------------------------------
    # Grievance Mechanism
    # ------------------------------------------------------------------

    async def submit_grievance(
        self, operator_id: str, complaint_data: Dict[str, Any],
    ) -> Any:
        """Submit a grievance."""
        if self._grievance_mechanism is not None:
            result = await self._grievance_mechanism.submit_grievance(
                operator_id=operator_id, complaint_data=complaint_data,
            )
            _safe_metric(
                record_grievance_submitted,
                result.severity.value if hasattr(result, "severity") else "medium",
                result.category.value if hasattr(result, "category") else "process",
            )
            return result
        raise RuntimeError("GrievanceMechanism engine not available")

    async def triage_grievance(self, grievance_id: str) -> Any:
        """Triage a grievance."""
        if self._grievance_mechanism is not None:
            return await self._grievance_mechanism.triage_grievance(grievance_id)
        raise RuntimeError("GrievanceMechanism engine not available")

    async def investigate_grievance(
        self, grievance_id: str, notes: Dict[str, Any],
    ) -> None:
        """Record investigation notes."""
        if self._grievance_mechanism is not None:
            await self._grievance_mechanism.investigate(grievance_id, notes)
        else:
            raise RuntimeError("GrievanceMechanism engine not available")

    async def resolve_grievance(
        self, grievance_id: str, resolution: Dict[str, Any],
    ) -> Any:
        """Resolve a grievance."""
        if self._grievance_mechanism is not None:
            result = await self._grievance_mechanism.resolve(
                grievance_id, resolution,
            )
            _safe_metric(
                record_grievance_resolved,
                result.severity.value if hasattr(result, "severity") else "medium",
            )
            return result
        raise RuntimeError("GrievanceMechanism engine not available")

    async def appeal_grievance(
        self, grievance_id: str, appeal_reason: str,
    ) -> Any:
        """Appeal a grievance."""
        if self._grievance_mechanism is not None:
            return await self._grievance_mechanism.handle_appeal(
                grievance_id, appeal_reason,
            )
        raise RuntimeError("GrievanceMechanism engine not available")

    async def get_grievance(self, grievance_id: str) -> Any:
        """Get grievance by ID."""
        if self._grievance_mechanism is not None:
            return await self._grievance_mechanism.get_grievance(grievance_id)
        return None

    async def list_grievances(
        self, operator_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Any]:
        """List grievances with filters."""
        if self._grievance_mechanism is not None:
            return await self._grievance_mechanism.list_grievances(
                operator_id=operator_id, severity=severity,
                status=status, category=category,
            )
        return []

    # ------------------------------------------------------------------
    # Consultation Records
    # ------------------------------------------------------------------

    async def create_consultation(
        self, operator_id: str, consultation_data: Dict[str, Any],
    ) -> Any:
        """Create a consultation record."""
        if self._consultation_manager is not None:
            result = await self._consultation_manager.create_consultation(
                operator_id=operator_id,
                consultation_data=consultation_data,
            )
            _safe_metric(
                record_consultation_recorded,
                consultation_data.get("type", "community_meeting"),
            )
            return result
        raise RuntimeError("ConsultationRecordManager not available")

    async def add_consultation_participants(
        self, consultation_id: str, participants: List[Dict[str, Any]],
    ) -> None:
        """Add participants to a consultation."""
        if self._consultation_manager is not None:
            await self._consultation_manager.add_participants(
                consultation_id, participants,
            )
        else:
            raise RuntimeError("ConsultationRecordManager not available")

    async def record_consultation_outcomes(
        self, consultation_id: str, outcomes: List[str],
        commitments: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Record consultation outcomes."""
        if self._consultation_manager is not None:
            await self._consultation_manager.record_outcomes(
                consultation_id, outcomes, commitments,
            )
        else:
            raise RuntimeError("ConsultationRecordManager not available")

    async def attach_consultation_evidence(
        self, consultation_id: str, evidence_files: List[str],
    ) -> None:
        """Attach evidence to a consultation."""
        if self._consultation_manager is not None:
            await self._consultation_manager.attach_evidence(
                consultation_id, evidence_files,
            )
        else:
            raise RuntimeError("ConsultationRecordManager not available")

    async def finalize_consultation(self, consultation_id: str) -> Any:
        """Finalize a consultation record."""
        if self._consultation_manager is not None:
            return await self._consultation_manager.finalize_consultation(
                consultation_id,
            )
        raise RuntimeError("ConsultationRecordManager not available")

    async def get_consultation(self, consultation_id: str) -> Any:
        """Get consultation by ID."""
        if self._consultation_manager is not None:
            return await self._consultation_manager.get_consultation(
                consultation_id,
            )
        return None

    async def list_consultations(
        self, operator_id: Optional[str] = None,
        consultation_type: Optional[str] = None,
        is_finalized: Optional[bool] = None,
    ) -> List[Any]:
        """List consultations with filters."""
        if self._consultation_manager is not None:
            return await self._consultation_manager.list_consultations(
                operator_id=operator_id,
                consultation_type=consultation_type,
                is_finalized=is_finalized,
            )
        return []

    # ------------------------------------------------------------------
    # Communications
    # ------------------------------------------------------------------

    async def send_communication(
        self, operator_id: str, stakeholder_ids: List[str],
        message: str, channel: str = "email",
        subject: str = "", language: str = "en",
    ) -> Any:
        """Send a communication."""
        if self._communication_hub is not None:
            result = await self._communication_hub.send_communication(
                operator_id=operator_id, stakeholder_ids=stakeholder_ids,
                message=message, channel=channel, subject=subject,
                language=language,
            )
            _safe_metric(record_communication_sent, channel)
            return result
        raise RuntimeError("CommunicationHub engine not available")

    async def schedule_communication(
        self, operator_id: str, stakeholder_ids: List[str],
        message: str, scheduled_at: str, channel: str = "email",
        subject: str = "", language: str = "en",
    ) -> str:
        """Schedule a communication."""
        if self._communication_hub is not None:
            return await self._communication_hub.schedule_communication(
                operator_id=operator_id, stakeholder_ids=stakeholder_ids,
                message=message, scheduled_at=scheduled_at, channel=channel,
                subject=subject, language=language,
            )
        raise RuntimeError("CommunicationHub engine not available")

    async def send_campaign(
        self, campaign_data: Dict[str, Any],
    ) -> List[Any]:
        """Send a campaign."""
        if self._communication_hub is not None:
            return await self._communication_hub.send_campaign(campaign_data)
        raise RuntimeError("CommunicationHub engine not available")

    async def get_communication(self, communication_id: str) -> Any:
        """Get communication by ID."""
        if self._communication_hub is not None:
            return await self._communication_hub.get_communication(
                communication_id,
            )
        return None

    async def list_communications(
        self, operator_id: Optional[str] = None,
        channel: Optional[str] = None,
        delivery_status: Optional[str] = None,
    ) -> List[Any]:
        """List communications with filters."""
        if self._communication_hub is not None:
            return await self._communication_hub.list_communications(
                operator_id=operator_id, channel=channel,
                delivery_status=delivery_status,
            )
        return []

    # ------------------------------------------------------------------
    # Engagement Assessment
    # ------------------------------------------------------------------

    async def assess_engagement(
        self, operator_id: str, stakeholder_id: str,
        period: Optional[Dict[str, str]] = None,
        engagement_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Assess engagement quality."""
        if self._engagement_verifier is not None:
            result = await self._engagement_verifier.assess_engagement(
                operator_id=operator_id, stakeholder_id=stakeholder_id,
                period=period, engagement_data=engagement_data,
            )
            _safe_metric(record_assessment_completed)
            return result
        raise RuntimeError("IndigenousRightsEngagementVerifier not available")

    async def get_assessment(self, assessment_id: str) -> Any:
        """Get assessment by ID."""
        if self._engagement_verifier is not None:
            return await self._engagement_verifier.get_assessment(
                assessment_id,
            )
        return None

    # ------------------------------------------------------------------
    # Compliance Reporting
    # ------------------------------------------------------------------

    async def generate_report(
        self, operator_id: str, report_type: str,
        year: Optional[int] = None,
        period: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Generate a compliance report."""
        if self._compliance_reporter is None:
            raise RuntimeError("ComplianceReporter engine not available")

        from greenlang.agents.eudr.stakeholder_engagement.models import ReportType

        try:
            rtype = ReportType(report_type)
        except ValueError:
            valid = [t.value for t in ReportType]
            raise ValueError(
                f"Invalid report type '{report_type}'. Must be one of: {valid}"
            )

        if rtype == ReportType.DDS_SUMMARY:
            result = await self._compliance_reporter.generate_dds_summary(operator_id)
        elif rtype == ReportType.FPIC_COMPLIANCE:
            result = await self._compliance_reporter.generate_fpic_compliance_report(operator_id)
        elif rtype == ReportType.GRIEVANCE_ANNUAL:
            result = await self._compliance_reporter.generate_grievance_annual_report(
                operator_id, year or 2026,
            )
        elif rtype == ReportType.CONSULTATION_REGISTER:
            result = await self._compliance_reporter.generate_consultation_register(operator_id)
        elif rtype == ReportType.ENGAGEMENT_ASSESSMENT:
            result = await self._compliance_reporter.generate_engagement_assessment_report(operator_id)
        elif rtype == ReportType.COMMUNICATION_LOG:
            result = await self._compliance_reporter.generate_communication_log(operator_id, period)
        elif rtype == ReportType.EFFECTIVENESS:
            result = await self._compliance_reporter.generate_effectiveness_report(operator_id)
        else:
            raise ValueError(f"Report type '{report_type}' not implemented")

        _safe_metric(record_report_generated, report_type)
        return result

    async def get_report(self, report_id: str) -> Any:
        """Get report by ID."""
        if self._compliance_reporter is not None:
            return await self._compliance_reporter.get_report(report_id)
        return None

    async def export_report(
        self, report_id: str, format: str = "json", language: str = "en",
    ) -> bytes:
        """Export a report."""
        if self._compliance_reporter is not None:
            return await self._compliance_reporter.export_report(
                report_id, format=format, language=language,
            )
        raise RuntimeError("ComplianceReporter engine not available")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        result: Dict[str, Any] = {
            "agent_id": _AGENT_ID,
            "version": _VERSION,
            "status": "healthy",
            "initialized": self._initialized,
            "engines": {},
            "connections": {},
            "timestamp": None,
        }

        from datetime import datetime, timezone
        result["timestamp"] = datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat()

        # Check database
        db_status = "unavailable"
        if self._db_pool is not None:
            try:
                async with self._db_pool.connection() as conn:
                    await conn.execute("SELECT 1")
                db_status = "connected"
            except Exception:
                db_status = "error"
                result["status"] = "degraded"
        result["connections"]["postgresql"] = db_status

        # Check Redis
        redis_status = "unavailable"
        if self._redis is not None:
            try:
                await self._redis.ping()
                redis_status = "connected"
            except Exception:
                redis_status = "error"
        result["connections"]["redis"] = redis_status

        # Check engines
        expected_engines = [
            "stakeholder_mapper",
            "fpic_workflow_engine",
            "grievance_mechanism",
            "consultation_record_manager",
            "communication_hub",
            "engagement_verifier",
            "compliance_reporter",
        ]

        for engine_name in expected_engines:
            if engine_name in self._engines:
                engine = self._engines[engine_name]
                if hasattr(engine, "health_check"):
                    try:
                        eng_health = await engine.health_check()
                        result["engines"][engine_name] = eng_health
                    except Exception as e:
                        result["engines"][engine_name] = {
                            "status": "error", "error": str(e),
                        }
                else:
                    result["engines"][engine_name] = {"status": "available"}
            else:
                result["engines"][engine_name] = {"status": "not_loaded"}

        # Determine overall status
        unhealthy_engines = sum(
            1 for v in result["engines"].values()
            if isinstance(v, dict) and v.get("status") in ("error", "not_loaded")
        )
        if unhealthy_engines > 3:
            result["status"] = "unhealthy"
        elif unhealthy_engines > 0:
            result["status"] = "degraded"

        return result

    @property
    def engine_count(self) -> int:
        """Return the number of loaded engines."""
        return len(self._engines)

    @property
    def is_initialized(self) -> bool:
        """Return whether the service has been initialized."""
        return self._initialized


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_lock = threading.Lock()
_service_instance: Optional[StakeholderEngagementService] = None


def get_service(
    config: Optional[StakeholderEngagementConfig] = None,
) -> StakeholderEngagementService:
    """Get the global StakeholderEngagementService singleton instance.

    Thread-safe lazy initialization.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        StakeholderEngagementService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = StakeholderEngagementService(config)
    return _service_instance


def reset_service() -> None:
    """Reset the global service singleton to None (testing only)."""
    global _service_instance
    with _service_lock:
        _service_instance = None


# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for startup/shutdown.

    Usage:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.stakeholder_engagement.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None -- application runs between startup and shutdown.
    """
    service = get_service()
    await service.startup()
    logger.info("Stakeholder Engagement lifespan: startup complete")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info("Stakeholder Engagement lifespan: shutdown complete")
