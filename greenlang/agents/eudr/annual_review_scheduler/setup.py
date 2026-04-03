# -*- coding: utf-8 -*-
"""
Annual Review Scheduler Service Facade - AGENT-EUDR-034

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point for review cycle management,
deadline tracking, checklist generation, entity coordination,
year-over-year comparison, calendar management, and notification dispatch.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (GL-EUDR-ARS-034)
Status: Production Ready
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-ARS-034"

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

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig, get_config,
)

try:
    from greenlang.agents.eudr.annual_review_scheduler.provenance import (
        ProvenanceTracker, GENESIS_HASH,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    GENESIS_HASH = "0" * 64

# Engine imports (conditional)
try:
    from greenlang.agents.eudr.annual_review_scheduler.review_cycle_manager import ReviewCycleManager
except ImportError:
    ReviewCycleManager = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.annual_review_scheduler.deadline_tracker import DeadlineTracker
except ImportError:
    DeadlineTracker = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.annual_review_scheduler.checklist_generator import ChecklistGenerator
except ImportError:
    ChecklistGenerator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.annual_review_scheduler.entity_coordinator import EntityCoordinator
except ImportError:
    EntityCoordinator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.annual_review_scheduler.year_comparator import YearComparator
except ImportError:
    YearComparator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.annual_review_scheduler.calendar_manager import CalendarManager
except ImportError:
    CalendarManager = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.annual_review_scheduler.notification_engine import NotificationEngine
except ImportError:
    NotificationEngine = None  # type: ignore[misc,assignment]


class AnnualReviewSchedulerService:
    """Unified service facade for AGENT-EUDR-034."""

    def __init__(self, config: Optional[AnnualReviewSchedulerConfig] = None) -> None:
        self.config = config or get_config()
        self._provenance = ProvenanceTracker() if ProvenanceTracker else None
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None
        self._engines: Dict[str, Any] = {}
        self._initialized = False
        logger.info("AnnualReviewSchedulerService created")

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
                redis_url = f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}"
                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception as e:
                logger.warning("Redis init failed: %s", e)
                self._redis = None

        self._init_engines()
        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000
        logger.info("AnnualReviewSchedulerService startup: %s/7 engines in %.1fms", len(self._engines), elapsed)

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines."""
        specs: List[Tuple[str, Any]] = [
            ("review_cycle_manager", ReviewCycleManager),
            ("deadline_tracker", DeadlineTracker),
            ("checklist_generator", ChecklistGenerator),
            ("entity_coordinator", EntityCoordinator),
            ("year_comparator", YearComparator),
            ("calendar_manager", CalendarManager),
            ("notification_engine", NotificationEngine),
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

    # -- Review Cycle Manager --
    async def create_review_cycle(self, operator_id: str, review_year: int, commodities: Optional[List[str]] = None, start_date: Optional[datetime] = None) -> Any:
        engine = self._engines.get("review_cycle_manager")
        if engine:
            return await engine.create_review_cycle(operator_id, review_year, commodities, start_date)
        raise RuntimeError("ReviewCycleManager not available")

    async def schedule_tasks(self, cycle_id: str, additional_tasks: Optional[List[Dict[str, Any]]] = None) -> Any:
        engine = self._engines.get("review_cycle_manager")
        if engine:
            return await engine.schedule_tasks(cycle_id, additional_tasks)
        raise RuntimeError("ReviewCycleManager not available")

    async def update_cycle_status(self, cycle_id: str, new_status: Any) -> Any:
        engine = self._engines.get("review_cycle_manager")
        if engine:
            return await engine.update_cycle_status(cycle_id, new_status)
        raise RuntimeError("ReviewCycleManager not available")

    async def get_active_cycles(self, operator_id: Optional[str] = None) -> List[Any]:
        engine = self._engines.get("review_cycle_manager")
        return await engine.get_active_cycles(operator_id) if engine else []

    async def get_cycle(self, cycle_id: str) -> Any:
        engine = self._engines.get("review_cycle_manager")
        return await engine.get_cycle(cycle_id) if engine else None

    async def list_cycles(self, operator_id: Optional[str] = None, review_year: Optional[int] = None, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Any]:
        engine = self._engines.get("review_cycle_manager")
        return await engine.list_cycles(operator_id, review_year, status, limit, offset) if engine else []

    async def update_task_status(self, cycle_id: str, task_id: str, new_status: Any) -> Any:
        engine = self._engines.get("review_cycle_manager")
        if engine:
            return await engine.update_task_status(cycle_id, task_id, new_status)
        raise RuntimeError("ReviewCycleManager not available")

    # -- Deadline Tracker --
    async def register_deadline(self, operator_id: str, deadline_type: Any, title: str, due_date: datetime, article_reference: str = "", responsible_entity: Optional[str] = None, review_year: int = 0) -> Any:
        engine = self._engines.get("deadline_tracker")
        if engine:
            return await engine.register_deadline(operator_id, deadline_type, title, due_date, article_reference, responsible_entity, review_year)
        raise RuntimeError("DeadlineTracker not available")

    async def check_approaching_deadlines(self, operator_id: str, review_year: Optional[int] = None) -> Any:
        engine = self._engines.get("deadline_tracker")
        if engine:
            return await engine.check_approaching_deadlines(operator_id, review_year)
        raise RuntimeError("DeadlineTracker not available")

    async def submit_to_authority(self, operator_id: str, deadline_id: str, submission_data: Dict[str, Any]) -> Any:
        engine = self._engines.get("deadline_tracker")
        if engine:
            return await engine.submit_to_authority(operator_id, deadline_id, submission_data)
        raise RuntimeError("DeadlineTracker not available")

    async def track_submission_status(self, submission_id: str) -> Any:
        engine = self._engines.get("deadline_tracker")
        return await engine.track_submission_status(submission_id) if engine else None

    async def list_deadline_records(self, operator_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Any]:
        engine = self._engines.get("deadline_tracker")
        return await engine.list_records(operator_id, limit, offset) if engine else []

    # -- Checklist Generator --
    async def generate_checklist(self, operator_id: str, commodity: str = "general", cycle_id: str = "", custom_items: Optional[List[Dict[str, Any]]] = None) -> Any:
        engine = self._engines.get("checklist_generator")
        if engine:
            return await engine.generate_checklist(operator_id, commodity, cycle_id, custom_items)
        raise RuntimeError("ChecklistGenerator not available")

    async def customize_for_commodity(self, checklist_id: str, commodity: str) -> Any:
        engine = self._engines.get("checklist_generator")
        if engine:
            return await engine.customize_for_commodity(checklist_id, commodity)
        raise RuntimeError("ChecklistGenerator not available")

    async def track_checklist_completion(self, checklist_id: str, item_id: str, new_status: Any, notes: str = "") -> Any:
        engine = self._engines.get("checklist_generator")
        if engine:
            return await engine.track_completion(checklist_id, item_id, new_status, notes)
        raise RuntimeError("ChecklistGenerator not available")

    async def get_checklist(self, checklist_id: str) -> Any:
        engine = self._engines.get("checklist_generator")
        return await engine.get_checklist(checklist_id) if engine else None

    async def list_checklists(self, operator_id: Optional[str] = None, commodity: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Any]:
        engine = self._engines.get("checklist_generator")
        return await engine.list_checklists(operator_id, commodity, limit, offset) if engine else []

    # -- Entity Coordinator --
    async def identify_review_entities(self, operator_id: str, entities: List[Dict[str, Any]], cycle_id: str = "") -> Any:
        engine = self._engines.get("entity_coordinator")
        if engine:
            return await engine.identify_review_entities(operator_id, entities, cycle_id)
        raise RuntimeError("EntityCoordinator not available")

    async def cascade_reviews(self, coordination_id: str, child_entities: Optional[List[Dict[str, Any]]] = None) -> Any:
        engine = self._engines.get("entity_coordinator")
        if engine:
            return await engine.cascade_reviews(coordination_id, child_entities)
        raise RuntimeError("EntityCoordinator not available")

    async def track_dependencies(self, coordination_id: str) -> Any:
        engine = self._engines.get("entity_coordinator")
        if engine:
            return await engine.track_dependencies(coordination_id)
        raise RuntimeError("EntityCoordinator not available")

    async def aggregate_completion(self, coordination_id: str) -> Any:
        engine = self._engines.get("entity_coordinator")
        if engine:
            return await engine.aggregate_completion(coordination_id)
        raise RuntimeError("EntityCoordinator not available")

    async def list_coordination_records(self, operator_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Any]:
        engine = self._engines.get("entity_coordinator")
        return await engine.list_records(operator_id, limit, offset) if engine else []

    # -- Year Comparator --
    async def compare_years(self, operator_id: str, data_points: List[Dict[str, Any]]) -> Any:
        engine = self._engines.get("year_comparator")
        if engine:
            return await engine.compare_years(operator_id, data_points)
        raise RuntimeError("YearComparator not available")

    async def generate_comparison_report(self, comparison_id: str) -> Dict[str, Any]:
        engine = self._engines.get("year_comparator")
        if engine:
            return await engine.generate_comparison_report(comparison_id)
        return {"error": "YearComparator not available"}

    async def list_comparison_records(self, operator_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Any]:
        engine = self._engines.get("year_comparator")
        return await engine.list_records(operator_id, limit, offset) if engine else []

    # -- Calendar Manager --
    async def add_calendar_event(self, operator_id: str, event_type: Any, title: str, start_date: datetime, end_date: Optional[datetime] = None, description: str = "", all_day: bool = True, review_year: int = 0) -> Any:
        engine = self._engines.get("calendar_manager")
        if engine:
            return await engine.add_event(operator_id, event_type, title, start_date, end_date, description, all_day, review_year=review_year)
        raise RuntimeError("CalendarManager not available")

    async def get_upcoming_events(self, operator_id: Optional[str] = None, days_ahead: int = 30, limit: int = 50) -> List[Any]:
        engine = self._engines.get("calendar_manager")
        return await engine.get_upcoming_events(operator_id, days_ahead, limit=limit) if engine else []

    async def generate_ical(self, operator_id: str, review_year: Optional[int] = None) -> str:
        engine = self._engines.get("calendar_manager")
        if engine:
            return await engine.generate_ical(operator_id, review_year)
        return ""

    async def list_calendar_records(self, operator_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Any]:
        engine = self._engines.get("calendar_manager")
        return await engine.list_records(operator_id, limit, offset) if engine else []

    # -- Notification Engine --
    async def send_notification(self, operator_id: str, channel: Any, recipient: str, subject: str, body: str, cycle_id: str = "") -> Any:
        engine = self._engines.get("notification_engine")
        if engine:
            return await engine.send_notification(operator_id, channel, recipient, subject, body, cycle_id=cycle_id)
        raise RuntimeError("NotificationEngine not available")

    async def send_notification_batch(self, operator_id: str, notifications: List[Dict[str, Any]], cycle_id: str = "") -> Any:
        engine = self._engines.get("notification_engine")
        if engine:
            return await engine.send_batch(operator_id, notifications, cycle_id)
        raise RuntimeError("NotificationEngine not available")

    async def acknowledge_notification(self, notification_id: str) -> Any:
        engine = self._engines.get("notification_engine")
        return await engine.track_acknowledgments(notification_id) if engine else None

    async def escalate_overdue(self, operator_id: str, cycle_id: str = "", hours_threshold: Optional[int] = None) -> Any:
        engine = self._engines.get("notification_engine")
        if engine:
            return await engine.escalate_overdue(operator_id, cycle_id, hours_threshold)
        raise RuntimeError("NotificationEngine not available")

    async def list_notification_records(self, operator_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Any]:
        engine = self._engines.get("notification_engine")
        return await engine.list_records(operator_id, limit, offset) if engine else []

    # -- Summary --
    async def generate_summary(self, operator_id: str, review_year: int = 0) -> Dict[str, Any]:
        """Generate an overall review summary."""
        summary: Dict[str, Any] = {
            "agent_id": _AGENT_ID,
            "operator_id": operator_id,
            "review_year": review_year,
            "engines": {},
        }
        for name, engine in self._engines.items():
            try:
                health = await engine.health_check()
                summary["engines"][name] = health
            except Exception as e:
                summary["engines"][name] = {"status": "error", "error": str(e)[:100]}
        return summary

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
_service_instance: Optional[AnnualReviewSchedulerService] = None


def get_service(config: Optional[AnnualReviewSchedulerConfig] = None) -> AnnualReviewSchedulerService:
    """Return the thread-safe singleton service instance."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = AnnualReviewSchedulerService(config)
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
