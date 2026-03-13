# -*- coding: utf-8 -*-
"""
Improvement Plan Creator Service Facade - AGENT-EUDR-035

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point for finding aggregation, gap analysis,
SMART action generation, root cause mapping, prioritization, progress
tracking, and stakeholder coordination.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (GL-EUDR-IPC-035)
Status: Production Ready
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-IPC-035"

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

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig, get_config,
)

try:
    from greenlang.agents.eudr.improvement_plan_creator.provenance import (
        ProvenanceTracker, GENESIS_HASH,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    GENESIS_HASH = "0" * 64

from greenlang.agents.eudr.improvement_plan_creator import metrics as m

# Engine imports (conditional)
try:
    from greenlang.agents.eudr.improvement_plan_creator.finding_aggregator import FindingAggregator
except ImportError:
    FindingAggregator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.improvement_plan_creator.gap_analyzer import GapAnalyzer
except ImportError:
    GapAnalyzer = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.improvement_plan_creator.action_generator import ActionGenerator
except ImportError:
    ActionGenerator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.improvement_plan_creator.root_cause_mapper import RootCauseMapper
except ImportError:
    RootCauseMapper = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.improvement_plan_creator.prioritization_engine import PrioritizationEngine
except ImportError:
    PrioritizationEngine = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.improvement_plan_creator.progress_tracker import ProgressTracker
except ImportError:
    ProgressTracker = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.improvement_plan_creator.stakeholder_coordinator import StakeholderCoordinator
except ImportError:
    StakeholderCoordinator = None  # type: ignore[misc,assignment]

from greenlang.agents.eudr.improvement_plan_creator.models import (
    AGENT_ID,
    AGENT_VERSION,
    ActionStatus,
    AggregatedFindings,
    ComplianceGap,
    EUDRCommodity,
    Finding,
    ImprovementAction,
    ImprovementPlan,
    PlanReport,
    PlanStatus,
    PlanSummary,
    ProgressSnapshot,
    RiskLevel,
)


class ImprovementPlanCreatorService:
    """Unified service facade for AGENT-EUDR-035.

    Orchestrates the full improvement plan creation pipeline:
    1. Finding Aggregation - collect/dedup upstream findings
    2. Gap Analysis - identify compliance gaps
    3. Root Cause Mapping - 5-Whys and fishbone analysis
    4. Action Generation - create SMART improvement actions
    5. Prioritization - Eisenhower + risk-based ranking
    6. Progress Tracking - milestones, snapshots, overdue detection
    7. Stakeholder Coordination - RACI, notifications, acknowledgments
    """

    def __init__(self, config: Optional[ImprovementPlanCreatorConfig] = None) -> None:
        """Initialize ImprovementPlanCreatorService."""
        self.config = config or get_config()
        self._provenance = ProvenanceTracker() if ProvenanceTracker else None
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None
        self._engines: Dict[str, Any] = {}
        self._plan_store: Dict[str, ImprovementPlan] = {}
        self._initialized = False
        self._start_time = time.monotonic()
        logger.info("ImprovementPlanCreatorService created")

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
                logger.warning("PostgreSQL pool init failed: %s", e)
                self._db_pool = None

        if REDIS_AVAILABLE:
            try:
                redis_url = (
                    f"redis://{self.config.redis_host}:{self.config.redis_port}"
                    f"/{self.config.redis_db}"
                )
                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception as e:
                logger.warning("Redis init failed: %s", e)
                self._redis = None

        self._init_engines()
        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "ImprovementPlanCreatorService startup: %d/7 engines in %.1fms",
            len(self._engines), elapsed,
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines."""
        specs: List[Tuple[str, Any]] = [
            ("finding_aggregator", FindingAggregator),
            ("gap_analyzer", GapAnalyzer),
            ("action_generator", ActionGenerator),
            ("root_cause_mapper", RootCauseMapper),
            ("prioritization_engine", PrioritizationEngine),
            ("progress_tracker", ProgressTracker),
            ("stakeholder_coordinator", StakeholderCoordinator),
        ]
        for name, cls in specs:
            if cls is not None:
                try:
                    self._engines[name] = cls(config=self.config)
                except Exception as e:
                    logger.warning("Engine '%s' init failed: %s", name, e)

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

    # -----------------------------------------------------------------
    # Full Pipeline
    # -----------------------------------------------------------------

    async def create_improvement_plan(
        self,
        operator_id: str,
        findings: List[Finding],
        commodity: Optional[str] = None,
        title: str = "",
        description: str = "",
    ) -> ImprovementPlan:
        """Execute the full improvement plan creation pipeline.

        Args:
            operator_id: Operator under improvement.
            findings: Raw findings from upstream agents.
            commodity: Optional EUDR commodity.
            title: Optional plan title.
            description: Optional plan description.

        Returns:
            Complete ImprovementPlan with all components.
        """
        start = time.monotonic()
        plan_id = f"PLAN-{uuid.uuid4().hex[:12]}"

        commodity_enum = None
        if commodity:
            try:
                commodity_enum = EUDRCommodity(commodity)
            except ValueError:
                pass

        # Step 1: Aggregate findings
        aggregation = await self._aggregate(operator_id, findings, plan_id)

        # Step 2: Analyze gaps
        gaps = await self._analyze_gaps(aggregation, plan_id)

        # Step 3: Root cause analysis
        root_causes = await self._map_root_causes(gaps, plan_id)

        # Step 4: Generate actions
        actions = await self._generate_actions(gaps, plan_id)

        # Step 5: Prioritize actions
        actions = await self._prioritize(actions, gaps, root_causes)

        # Link root causes to actions
        rc_by_gap: Dict[str, str] = {}
        for rc in root_causes:
            if rc.gap_id not in rc_by_gap and rc.depth == max(
                r.depth for r in root_causes if r.gap_id == rc.gap_id
            ):
                rc_by_gap[rc.gap_id] = rc.root_cause_id
        for action in actions:
            if action.gap_id in rc_by_gap:
                action.root_cause_id = rc_by_gap[action.gap_id]

        # Calculate summary metrics
        total_cost = sum(a.estimated_cost for a in actions)
        max_deadline_days = 0
        for a in actions:
            if a.time_bound_deadline:
                days = (a.time_bound_deadline - datetime.now(timezone.utc)).days
                max_deadline_days = max(max_deadline_days, days)

        # Determine risk level from gaps
        risk_level = self._assess_risk_level(gaps)

        # Compute provenance
        provenance_hash = ""
        if self._provenance:
            provenance_data = {
                "plan_id": plan_id,
                "operator_id": operator_id,
                "gaps": len(gaps),
                "actions": len(actions),
                "root_causes": len(root_causes),
            }
            provenance_hash = self._provenance.compute_hash(provenance_data)

        plan = ImprovementPlan(
            plan_id=plan_id,
            operator_id=operator_id,
            title=title or f"EUDR Improvement Plan - {operator_id}",
            description=description or (
                f"Improvement plan addressing {len(gaps)} compliance gaps "
                f"with {len(actions)} prioritized actions"
            ),
            commodity=commodity_enum,
            status=PlanStatus.DRAFT,
            risk_level=risk_level,
            aggregation_id=aggregation.aggregation_id,
            gaps=gaps,
            root_causes=root_causes,
            actions=actions,
            total_gaps=len(gaps),
            total_actions=len(actions),
            estimated_total_cost=total_cost,
            estimated_completion_days=max(max_deadline_days, 1),
            target_completion=datetime.now(timezone.utc) + timedelta(
                days=max(max_deadline_days, 30)
            ),
            provenance_hash=provenance_hash,
        )

        self._plan_store[plan_id] = plan
        m.record_plan_created("draft")
        m.set_active_plans(
            sum(1 for p in self._plan_store.values() if p.status in (
                PlanStatus.DRAFT, PlanStatus.UNDER_REVIEW,
                PlanStatus.APPROVED, PlanStatus.ACTIVE,
            ))
        )

        # Update gap gauges
        m.set_critical_gaps_open(
            sum(1 for g in gaps if g.severity.value == "critical")
        )
        m.set_high_gaps_open(
            sum(1 for g in gaps if g.severity.value == "high")
        )

        elapsed = time.monotonic() - start
        m.observe_plan_creation_duration(elapsed)

        if self._provenance:
            self._provenance.record(
                "plan", "create", plan_id, AGENT_ID,
                metadata={"operator_id": operator_id, "gaps": len(gaps), "actions": len(actions)},
            )

        logger.info(
            "Created improvement plan %s: %d gaps, %d actions, "
            "risk=%s in %.1fms",
            plan_id, len(gaps), len(actions), risk_level.value,
            elapsed * 1000,
        )
        return plan

    async def _aggregate(
        self, operator_id: str, findings: List[Finding], plan_id: str
    ) -> AggregatedFindings:
        """Delegate to FindingAggregator engine."""
        engine = self._engines.get("finding_aggregator")
        if engine:
            return await engine.aggregate_findings(operator_id, findings, plan_id)
        # Fallback: pass through
        return AggregatedFindings(
            aggregation_id=f"AGG-{uuid.uuid4().hex[:12]}",
            operator_id=operator_id,
            findings=findings,
            total_findings=len(findings),
        )

    async def _analyze_gaps(
        self, aggregation: AggregatedFindings, plan_id: str
    ) -> List[ComplianceGap]:
        """Delegate to GapAnalyzer engine."""
        engine = self._engines.get("gap_analyzer")
        if engine:
            return await engine.analyze_gaps(aggregation, plan_id)
        return []

    async def _map_root_causes(
        self, gaps: List[ComplianceGap], plan_id: str
    ) -> list:
        """Delegate to RootCauseMapper engine."""
        engine = self._engines.get("root_cause_mapper")
        if engine:
            return await engine.analyze_root_causes(gaps, plan_id)
        return []

    async def _generate_actions(
        self, gaps: List[ComplianceGap], plan_id: str
    ) -> List[ImprovementAction]:
        """Delegate to ActionGenerator engine."""
        engine = self._engines.get("action_generator")
        if engine:
            return await engine.generate_actions(gaps, plan_id)
        return []

    async def _prioritize(
        self,
        actions: List[ImprovementAction],
        gaps: List[ComplianceGap],
        root_causes: list,
    ) -> List[ImprovementAction]:
        """Delegate to PrioritizationEngine."""
        engine = self._engines.get("prioritization_engine")
        if engine:
            return await engine.prioritize_actions(actions, gaps, root_causes)
        return actions

    def _assess_risk_level(self, gaps: List[ComplianceGap]) -> RiskLevel:
        """Assess overall risk level from gaps.

        Args:
            gaps: Compliance gaps.

        Returns:
            RiskLevel classification.
        """
        if not gaps:
            return RiskLevel.LOW

        critical = sum(1 for g in gaps if g.severity.value == "critical")
        high = sum(1 for g in gaps if g.severity.value == "high")

        if critical >= 3:
            return RiskLevel.CRITICAL
        elif critical >= 1 or high >= 3:
            return RiskLevel.HIGH
        elif high >= 1:
            return RiskLevel.STANDARD
        return RiskLevel.LOW

    # -----------------------------------------------------------------
    # Plan Operations
    # -----------------------------------------------------------------

    async def get_plan(self, plan_id: str) -> Optional[ImprovementPlan]:
        """Retrieve a plan by ID."""
        return self._plan_store.get(plan_id)

    async def list_plans(
        self,
        operator_id: Optional[str] = None,
        status: Optional[str] = None,
        commodity: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[PlanSummary]:
        """List plans with optional filtering.

        Args:
            operator_id: Filter by operator.
            status: Filter by status.
            commodity: Filter by commodity.
            limit: Max results.
            offset: Skip count.

        Returns:
            List of PlanSummary.
        """
        plans = list(self._plan_store.values())
        if operator_id:
            plans = [p for p in plans if p.operator_id == operator_id]
        if status:
            plans = [p for p in plans if p.status.value == status]
        if commodity:
            plans = [
                p for p in plans
                if p.commodity and p.commodity.value == commodity
            ]
        plans.sort(key=lambda p: p.created_at, reverse=True)
        sliced = plans[offset:offset + limit]

        summaries: List[PlanSummary] = []
        for plan in sliced:
            completed = sum(
                1 for a in plan.actions
                if a.status in (ActionStatus.COMPLETED, ActionStatus.VERIFIED, ActionStatus.CLOSED)
            )
            progress = Decimal("0")
            if plan.total_actions > 0:
                progress = (
                    Decimal(str(completed)) / Decimal(str(plan.total_actions)) * Decimal("100")
                )
            summaries.append(PlanSummary(
                plan_id=plan.plan_id,
                operator_id=plan.operator_id,
                title=plan.title,
                status=plan.status,
                commodity=plan.commodity,
                risk_level=plan.risk_level,
                total_gaps=plan.total_gaps,
                total_actions=plan.total_actions,
                actions_completed=completed,
                overall_progress=progress,
                created_at=plan.created_at,
            ))
        return summaries

    async def update_plan_status(
        self, plan_id: str, new_status: PlanStatus
    ) -> Optional[ImprovementPlan]:
        """Update a plan's status.

        Args:
            plan_id: Plan identifier.
            new_status: Target status.

        Returns:
            Updated plan or None if not found.
        """
        plan = self._plan_store.get(plan_id)
        if not plan:
            return None

        plan.status = new_status
        if new_status == PlanStatus.APPROVED:
            plan.approved_at = datetime.now(timezone.utc)
            m.record_plan_approved()
        elif new_status == PlanStatus.COMPLETED:
            plan.completed_at = datetime.now(timezone.utc)

        if self._provenance:
            self._provenance.record(
                "plan", "status_update", plan_id, AGENT_ID,
                metadata={"new_status": new_status.value},
            )
        return plan

    async def update_action_status(
        self,
        plan_id: str,
        action_id: str,
        new_status: str,
    ) -> Optional[ImprovementAction]:
        """Update an action's status.

        Args:
            plan_id: Plan identifier.
            action_id: Action identifier.
            new_status: New status value.

        Returns:
            Updated action or None.
        """
        engine = self._engines.get("action_generator")
        if engine:
            status = ActionStatus(new_status)
            return await engine.update_action_status(plan_id, action_id, status)
        return None

    # -----------------------------------------------------------------
    # Progress Operations
    # -----------------------------------------------------------------

    async def capture_progress(self, plan_id: str) -> Optional[ProgressSnapshot]:
        """Capture a progress snapshot for a plan."""
        plan = self._plan_store.get(plan_id)
        if not plan:
            return None
        engine = self._engines.get("progress_tracker")
        if engine:
            return await engine.capture_snapshot(plan)
        return None

    async def check_overdue(self, plan_id: str) -> List[ImprovementAction]:
        """Check for overdue actions in a plan."""
        plan = self._plan_store.get(plan_id)
        if not plan:
            return []
        engine = self._engines.get("progress_tracker")
        if engine:
            return await engine.check_overdue(plan)
        return []

    async def get_progress_snapshots(self, plan_id: str) -> List[ProgressSnapshot]:
        """Get progress snapshots for a plan."""
        engine = self._engines.get("progress_tracker")
        if engine:
            return await engine.get_snapshots(plan_id)
        return []

    async def review_effectiveness(self, plan_id: str) -> Dict[str, Any]:
        """Review effectiveness of completed actions."""
        plan = self._plan_store.get(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        engine = self._engines.get("progress_tracker")
        if engine:
            return await engine.review_effectiveness(plan)
        return {"error": "ProgressTracker not available"}

    # -----------------------------------------------------------------
    # Stakeholder Operations
    # -----------------------------------------------------------------

    async def assign_stakeholders(
        self,
        plan_id: str,
        action_id: str,
        stakeholders: List[Dict[str, Any]],
    ) -> List[Any]:
        """Assign stakeholders to an action in a plan."""
        plan = self._plan_store.get(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")

        action = None
        for a in plan.actions:
            if a.action_id == action_id:
                action = a
                break
        if not action:
            raise ValueError(f"Action {action_id} not found in plan {plan_id}")

        engine = self._engines.get("stakeholder_coordinator")
        if engine:
            return await engine.assign_stakeholders(action, stakeholders)
        return []

    async def send_notification(
        self,
        action_id: str,
        stakeholder_id: str,
        subject: str,
        body: str,
    ) -> Any:
        """Send a notification to a stakeholder."""
        engine = self._engines.get("stakeholder_coordinator")
        if engine:
            return await engine.send_notification(
                action_id, stakeholder_id, subject, body
            )
        raise RuntimeError("StakeholderCoordinator not available")

    # -----------------------------------------------------------------
    # Report Generation
    # -----------------------------------------------------------------

    async def generate_report(self, plan_id: str) -> Optional[PlanReport]:
        """Generate a comprehensive plan report.

        Args:
            plan_id: Plan identifier.

        Returns:
            PlanReport or None if plan not found.
        """
        start = time.monotonic()
        plan = self._plan_store.get(plan_id)
        if not plan:
            return None

        # Gaps summary
        gaps_summary: Dict[str, int] = {}
        for gap in plan.gaps:
            key = gap.severity.value
            gaps_summary[key] = gaps_summary.get(key, 0) + 1

        # Actions summary
        actions_summary: Dict[str, int] = {}
        for action in plan.actions:
            key = action.status.value
            actions_summary[key] = actions_summary.get(key, 0) + 1

        # Progress
        completed = sum(
            1 for a in plan.actions
            if a.status in (ActionStatus.COMPLETED, ActionStatus.VERIFIED, ActionStatus.CLOSED)
        )
        progress = Decimal("0")
        if plan.total_actions > 0:
            progress = (
                Decimal(str(completed)) / Decimal(str(plan.total_actions)) * Decimal("100")
            )

        # Stakeholder count
        stakeholder_engine = self._engines.get("stakeholder_coordinator")
        stakeholder_count = 0
        if stakeholder_engine:
            for action in plan.actions:
                assignments = await stakeholder_engine.get_assignments(action.action_id)
                stakeholder_count += len(assignments)

        provenance_hash = ""
        if self._provenance:
            provenance_data = {
                "report_plan_id": plan_id,
                "gaps_summary": gaps_summary,
                "actions_summary": actions_summary,
            }
            provenance_hash = self._provenance.compute_hash(provenance_data)

        report = PlanReport(
            report_id=f"RPT-{uuid.uuid4().hex[:12]}",
            plan_id=plan_id,
            operator_id=plan.operator_id,
            commodity=plan.commodity,
            plan_status=plan.status,
            risk_level=plan.risk_level,
            gaps_summary=gaps_summary,
            actions_summary=actions_summary,
            overall_progress=progress,
            on_track=completed >= (plan.total_actions * 0.5) if plan.total_actions > 0 else True,
            stakeholder_count=stakeholder_count,
            estimated_total_cost=plan.estimated_total_cost,
            provenance_hash=provenance_hash,
        )

        elapsed = time.monotonic() - start
        m.observe_report_generation_duration(elapsed)
        m.record_report_generated(self.config.report_format)

        return report

    # -----------------------------------------------------------------
    # Root Cause / Fishbone
    # -----------------------------------------------------------------

    async def build_fishbone(
        self, plan_id: str, gap_id: str
    ) -> Optional[Any]:
        """Build fishbone analysis for a specific gap in a plan."""
        plan = self._plan_store.get(plan_id)
        if not plan:
            return None

        gap = None
        for g in plan.gaps:
            if g.gap_id == gap_id:
                gap = g
                break
        if not gap:
            return None

        engine = self._engines.get("root_cause_mapper")
        if engine:
            root_causes = [rc for rc in plan.root_causes if rc.gap_id == gap_id]
            return await engine.build_fishbone(gap, root_causes or None)
        return None

    # -----------------------------------------------------------------
    # Dashboard & Health
    # -----------------------------------------------------------------

    async def get_dashboard(self, operator_id: str) -> Dict[str, Any]:
        """Get aggregated dashboard data for an operator."""
        plans = [
            p for p in self._plan_store.values()
            if p.operator_id == operator_id
        ]
        active = [
            p for p in plans
            if p.status in (PlanStatus.ACTIVE, PlanStatus.APPROVED, PlanStatus.DRAFT)
        ]

        total_actions = sum(p.total_actions for p in active)
        completed_actions = sum(
            sum(1 for a in p.actions if a.status in (
                ActionStatus.COMPLETED, ActionStatus.VERIFIED, ActionStatus.CLOSED
            ))
            for p in active
        )
        overdue_actions = 0
        now = datetime.now(timezone.utc)
        for p in active:
            for a in p.actions:
                if (
                    a.time_bound_deadline and a.time_bound_deadline < now
                    and a.status not in (
                        ActionStatus.COMPLETED, ActionStatus.VERIFIED,
                        ActionStatus.CLOSED, ActionStatus.CANCELLED
                    )
                ):
                    overdue_actions += 1

        return {
            "agent_id": _AGENT_ID,
            "operator_id": operator_id,
            "total_plans": len(plans),
            "active_plans": len(active),
            "total_actions": total_actions,
            "completed_actions": completed_actions,
            "overdue_actions": overdue_actions,
            "overall_progress": (
                round(completed_actions / total_actions * 100, 1)
                if total_actions > 0 else 0.0
            ),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Return agent health status."""
        result: Dict[str, Any] = {
            "agent_id": _AGENT_ID,
            "version": _VERSION,
            "status": "healthy",
            "initialized": self._initialized,
            "engines": {},
            "connections": {},
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
        }

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
        """Return number of loaded engines."""
        return len(self._engines)

    @property
    def is_initialized(self) -> bool:
        """Return initialization status."""
        return self._initialized


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_service_lock = threading.Lock()
_service_instance: Optional[ImprovementPlanCreatorService] = None


def get_service(
    config: Optional[ImprovementPlanCreatorConfig] = None,
) -> ImprovementPlanCreatorService:
    """Return the thread-safe singleton service instance."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ImprovementPlanCreatorService(config)
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
