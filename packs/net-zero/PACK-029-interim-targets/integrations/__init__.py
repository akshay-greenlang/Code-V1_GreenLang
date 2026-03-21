# -*- coding: utf-8 -*-
"""
PACK-029 Interim Targets Pack - Integration Layer
=====================================================

Phase 4 integration layer for the Interim Targets Pack providing
10 enterprise bridges for baseline import (PACK-021), sector pathway
integration (PACK-028), 30-agent MRV inventory, SBTi interim target
validation (21 criteria), CDP Climate Change questionnaire export,
TCFD Metrics & Targets disclosure, initiative deployment tracking,
internal carbon budget system, off-track alerting, and third-party
assurance portal integration.

Components:
    - PACK021Bridge: PACK-021 baseline, long-term target, SBTi pathway,
      and organizational boundary import for interim target decomposition
    - PACK028Bridge: PACK-028 sector milestones, technology roadmap,
      MACC curve levers, and sector benchmarks for validation
    - MRVBridge: 30-agent MRV routing with retry/circuit-breaker,
      annual inventory, scope aggregation, and variance analysis
    - SBTiBridge: 21-criteria SBTi validation (ambition, linearity,
      scope coverage, FLAG, consistency), submission package generation
    - CDPBridge: CDP C4.1/C4.2/C5.1/C6.1/C7.1 export with cross-
      reference validation
    - TCFDBridge: TCFD Table 1-4 (emissions, targets, risks, projections)
      with scenario analysis and cross-pillar consistency checks
    - InitiativeTrackerBridge: Initiative portfolio, RAG status, variance
      attribution, forecast impact, and budget tracking
    - BudgetSystemBridge: Carbon budget allocation, shadow pricing,
      carbon fee/levy, financial impact (NPV), rebalance triggers
    - AlertingBridge: Configurable alert rules, email/Slack/Teams/
      dashboard channels, escalation workflows, alert history
    - AssurancePortalBridge: ISO 14064-3 evidence package, document
      management, completeness checks, Big 4 provider API, version control

Architecture:
    PACK-021 Baseline --> PACK-029 Interim Decomposition
                               |
    PACK-028 Sector   --> Sector Pathway Validation
                               |
    MRV Agents (30)   --> Annual Inventory & Variance
                               |
    SBTi Validation   --> 21-Criteria Compliance
                               |
    CDP / TCFD        --> External Reporting Export
                               |
    Initiatives       --> Variance Attribution & Forecast
                               |
    Carbon Budget     --> Financial Impact & Rebalancing
                               |
    Alerting          --> Off-Track Notifications
                               |
    Assurance Portal  --> Third-Party Verification

Platform Integrations:
    - greenlang/agents/mrv/* (all 30 MRV agents)
    - packs/net-zero/PACK-021-net-zero-starter/* (baseline/target)
    - packs/net-zero/PACK-028-sector-pathway/* (sector pathways)
    - SBTi Corporate Standard V5.3 (target validation)
    - CDP Climate Change Questionnaire (C4/C5/C6/C7)
    - TCFD Recommendations (Metrics & Targets pillar)
    - ISO 14064-3 (verification/validation of GHG assertions)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-029"
__pack_name__ = "Interim Targets Pack"

# ---------------------------------------------------------------------------
# PACK-021 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack021_bridge import (
        BaselineStatus,
        SBTiPathwayType,
        BoundaryApproach,
        DataQualityTier as Pack021DataQualityTier,
        ImportStatus,
        PACK021BridgeConfig,
        BaselineImport,
        LongTermTargetImport,
        SBTiPathwayImport,
        BoundaryImport,
        PACK021IntegrationResult,
        PACK021Bridge,
        PACK021_COMPONENTS,
        SBTI_MINIMUM_AMBITION as PACK021_SBTI_AMBITION,
        SCOPE3_THRESHOLD_PCT,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("PACK-021 bridge import failed: %s", _e)

# ---------------------------------------------------------------------------
# PACK-028 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack028_bridge import (
        SectorType,
        MilestoneType,
        BenchmarkTier,
        LeverPriority,
        MilestoneStatus as SectorMilestoneStatus,
        PACK028BridgeConfig,
        SectorMilestoneImport,
        TechnologyRoadmapImport,
        AbatementLeverImport,
        SectorBenchmarkImport,
        PACK028IntegrationResult,
        PACK028Bridge,
        PACK028_COMPONENTS,
        SECTOR_INTERIM_MILESTONES,
        SECTOR_INTENSITY_METRICS,
        IEA_TECHNOLOGY_MILESTONES,
        SECTOR_BENCHMARKS,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("PACK-028 bridge import failed: %s", _e)

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        MRVScope,
        DataQualityTier as MRVDataQualityTier,
        AgentHealthStatus,
        VarianceDirection,
        MRVBridgeConfig,
        AgentResult,
        ScopeAggregate,
        VarianceResult,
        AnnualInventoryResult,
        MRVBridge,
        MRV_AGENT_REGISTRY,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("MRV bridge import failed: %s", _e)

# ---------------------------------------------------------------------------
# SBTi Bridge
# ---------------------------------------------------------------------------
try:
    from .sbti_bridge import (
        SBTiPathway,
        TargetType,
        CriteriaStatus,
        SubmissionStatus,
        TemperatureRating,
        LinearityStatus,
        SBTiBridgeConfig,
        InterimTargetDefinition,
        CriteriaValidation,
        LinearityAssessment,
        SBTiValidationResult,
        SBTiSubmissionPackage,
        SBTiBridge,
        SBTI_MINIMUM_AMBITION,
        INTERIM_TARGET_CRITERIA,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("SBTi bridge import failed: %s", _e)

# ---------------------------------------------------------------------------
# CDP Bridge
# ---------------------------------------------------------------------------
try:
    from .cdp_bridge import (
        CDPSection,
        CDPTargetType,
        CDPScope,
        CDPMethodology,
        ValidationSeverity,
        CDPBridgeConfig,
        CDPC41Export,
        CDPC42Row,
        CDPC42Export,
        CDPC51Export,
        CDPC61Export,
        CDPC71Export,
        CrossValidationResult,
        CDPExportResult,
        CDPBridge,
        SCOPE3_CATEGORY_NAMES,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("CDP bridge import failed: %s", _e)

# ---------------------------------------------------------------------------
# TCFD Bridge
# ---------------------------------------------------------------------------
try:
    from .tcfd_bridge import (
        TCFDPillar,
        TCFDScenario,
        TransitionRiskType,
        RiskLikelihood,
        RiskImpact,
        ConsistencyStatus,
        TCFDBridgeConfig,
        TCFDTable1Row,
        TCFDTable1Export,
        TCFDTable2Row,
        TCFDTable2Export,
        TCFDTable3Row,
        TCFDTable3Export,
        TCFDTable4Row,
        TCFDTable4Export,
        TCFDConsistencyCheck,
        TCFDExportResult,
        TCFDBridge,
        SCENARIO_CARBON_PRICES,
        DEFAULT_TRANSITION_RISKS,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("TCFD bridge import failed: %s", _e)

# ---------------------------------------------------------------------------
# Initiative Tracker Bridge
# ---------------------------------------------------------------------------
try:
    from .initiative_tracker_bridge import (
        InitiativeStatus,
        InitiativeCategory,
        RAGStatus,
        BudgetType,
        VarianceType,
        InitiativeTrackerConfig,
        Initiative,
        InitiativePortfolio,
        VarianceAttribution,
        ForecastResult,
        InitiativeTrackerResult,
        InitiativeTrackerBridge,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("Initiative tracker bridge import failed: %s", _e)

# ---------------------------------------------------------------------------
# Budget System Bridge
# ---------------------------------------------------------------------------
try:
    from .budget_system_bridge import (
        BudgetScope,
        CarbonPriceType,
        BudgetStatus,
        RebalanceTrigger,
        BudgetSystemConfig,
        AnnualCarbonBudget,
        CarbonPriceResult,
        FinancialImpact,
        RebalanceRecommendation,
        BudgetSystemResult,
        BudgetSystemBridge,
        DEFAULT_SHADOW_PRICE_SCHEDULE,
        EU_ETS_PRICE_ASSUMPTIONS,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("Budget system bridge import failed: %s", _e)

# ---------------------------------------------------------------------------
# Alerting Bridge
# ---------------------------------------------------------------------------
try:
    from .alerting_bridge import (
        AlertSeverity,
        AlertChannel,
        AlertStatus,
        AlertRuleType,
        EscalationLevel,
        AlertingBridgeConfig,
        AlertRule,
        Alert,
        AlertEvaluation,
        AlertingBridge,
        DEFAULT_ALERT_RULES,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("Alerting bridge import failed: %s", _e)

# ---------------------------------------------------------------------------
# Assurance Portal Bridge
# ---------------------------------------------------------------------------
try:
    from .assurance_portal_bridge import (
        AssuranceLevel,
        AssuranceStandard,
        AssuranceProvider,
        DocumentType,
        WorkflowStatus,
        EvidenceStatus,
        AssurancePortalConfig,
        EvidenceDocument,
        WorkpaperRequirement,
        CompletenessCheck,
        AssuranceRequest,
        EvidencePackage,
        AssurancePortalBridge,
        ISO_14064_3_REQUIREMENTS,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("Assurance portal bridge import failed: %s", _e)


# ---------------------------------------------------------------------------
# Connection Pooling Utilities
# ---------------------------------------------------------------------------

async def create_shared_db_pool(
    connection_string: str,
    min_size: int = 1,
    max_size: int = 10,
) -> "Any":
    """Create a shared async database connection pool for all bridges.

    Uses psycopg_pool.AsyncConnectionPool for PostgreSQL/TimescaleDB.
    """
    try:
        import psycopg_pool
        pool = psycopg_pool.AsyncConnectionPool(
            connection_string,
            min_size=min_size,
            max_size=max_size,
        )
        await pool.open()
        return pool
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "psycopg_pool not available; database pooling disabled"
        )
        return None


# ---------------------------------------------------------------------------
# Retry / Timeout Decorators
# ---------------------------------------------------------------------------

import asyncio
import functools
from typing import Callable, TypeVar

_T = TypeVar("_T")


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Retry decorator for async functions with exponential backoff.

    Example:
        @retry_async(max_attempts=3, base_delay=1.0)
        async def fetch_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: "Any", **kwargs: "Any") -> "Any":
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        delay = base_delay * (backoff_factor ** attempt)
                        await asyncio.sleep(delay)
            raise last_exc  # type: ignore
        return wrapper
    return decorator


def timeout_async(seconds: float = 30.0) -> Callable:
    """Timeout decorator for async functions.

    Example:
        @timeout_async(seconds=10.0)
        async def slow_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: "Any", **kwargs: "Any") -> "Any":
            return await asyncio.wait_for(
                func(*args, **kwargs), timeout=seconds,
            )
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Circuit Breaker Pattern
# ---------------------------------------------------------------------------

import time as _time


class CircuitBreaker:
    """Circuit breaker pattern for external API calls.

    Prevents cascading failures by opening the circuit after a
    threshold of consecutive failures, then half-opening after
    a reset timeout to test recovery.

    Example:
        breaker = CircuitBreaker(failure_threshold=5, reset_timeout=300)
        if breaker.is_closed():
            try:
                result = await api_call()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 300.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._state: str = "closed"  # closed, open, half_open
        self._half_open_calls: int = 0

    def is_closed(self) -> bool:
        """Check if circuit is closed (requests allowed)."""
        if self._state == "closed":
            return True
        if self._state == "open":
            if _time.monotonic() - self._last_failure_time > self.reset_timeout:
                self._state = "half_open"
                self._half_open_calls = 0
                return True
            return False
        if self._state == "half_open":
            return self._half_open_calls < self.half_open_max_calls
        return False

    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        self._state = "closed"
        self._half_open_calls = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = _time.monotonic()
        if self._state == "half_open":
            self._state = "open"
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"

    @property
    def state(self) -> str:
        """Current circuit state."""
        if self._state == "open":
            if _time.monotonic() - self._last_failure_time > self.reset_timeout:
                return "half_open"
        return self._state

    def get_status(self) -> "Dict[str, Any]":
        """Get circuit breaker status."""
        return {
            "state": self.state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "reset_timeout_seconds": self.reset_timeout,
        }


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class AsyncRateLimiter:
    """Token bucket rate limiter for API calls.

    Example:
        limiter = AsyncRateLimiter(rate=20, per_seconds=60)
        async with limiter:
            await api_call()
    """

    def __init__(self, rate: int = 20, per_seconds: float = 60.0) -> None:
        self.rate = rate
        self.per_seconds = per_seconds
        self._tokens: float = float(rate)
        self._last_refill: float = _time.monotonic()
        self._lock: "Optional[asyncio.Lock]" = None

    def _get_lock(self) -> "asyncio.Lock":
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        lock = self._get_lock()
        while True:
            async with lock:
                now = _time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    float(self.rate),
                    self._tokens + elapsed * (self.rate / self.per_seconds),
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            await asyncio.sleep(self.per_seconds / max(self.rate, 1))

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *args: "Any") -> None:
        pass


# ---------------------------------------------------------------------------
# Response Caching (Redis-backed)
# ---------------------------------------------------------------------------


class AsyncResponseCache:
    """Redis-backed response cache for integration results.

    Falls back to in-memory dict if Redis is not available.

    Example:
        cache = AsyncResponseCache(redis_url="redis://localhost:6379")
        await cache.connect()
        await cache.set("key", data, ttl=3600)
        result = await cache.get("key")
    """

    def __init__(
        self,
        redis_url: str = "",
        default_ttl: int = 3600,
        prefix: str = "pack029:",
    ) -> None:
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._redis: "Optional[Any]" = None
        self._memory: "Dict[str, Any]" = {}
        self._expiry: "Dict[str, float]" = {}

    async def connect(self) -> bool:
        """Connect to Redis (or use in-memory fallback)."""
        if not self.redis_url:
            return False
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self.redis_url)
            await self._redis.ping()
            return True
        except Exception:
            self._redis = None
            return False

    async def get(self, key: str) -> "Optional[Any]":
        """Get cached value."""
        full_key = f"{self.prefix}{key}"
        if self._redis:
            try:
                val = await self._redis.get(full_key)
                return json.loads(val) if val else None
            except Exception:
                pass
        # In-memory fallback
        if full_key in self._memory:
            if self._expiry.get(full_key, float("inf")) > _time.monotonic():
                return self._memory[full_key]
            else:
                del self._memory[full_key]
                del self._expiry[full_key]
        return None

    async def set(
        self, key: str, value: "Any", ttl: "Optional[int]" = None,
    ) -> None:
        """Set cached value."""
        full_key = f"{self.prefix}{key}"
        ttl = ttl or self.default_ttl
        if self._redis:
            try:
                await self._redis.set(
                    full_key, json.dumps(value, default=str), ex=ttl,
                )
                return
            except Exception:
                pass
        # In-memory fallback
        self._memory[full_key] = value
        self._expiry[full_key] = _time.monotonic() + ttl

    async def delete(self, key: str) -> None:
        """Delete cached value."""
        full_key = f"{self.prefix}{key}"
        if self._redis:
            try:
                await self._redis.delete(full_key)
            except Exception:
                pass
        self._memory.pop(full_key, None)
        self._expiry.pop(full_key, None)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# ---------------------------------------------------------------------------
# API Key Rotation
# ---------------------------------------------------------------------------


class APIKeyRotator:
    """API key rotation for external service integrations.

    Cycles through multiple API keys to distribute rate limits.

    Example:
        rotator = APIKeyRotator(keys=["key1", "key2", "key3"])
        current_key = rotator.get_current_key()
        rotator.rotate()
    """

    def __init__(self, keys: "Optional[List[str]]" = None) -> None:
        self._keys = keys or []
        self._current_index = 0
        self._usage_counts: "Dict[int, int]" = {}

    def get_current_key(self) -> str:
        """Get the current API key."""
        if not self._keys:
            return ""
        key = self._keys[self._current_index]
        self._usage_counts[self._current_index] = (
            self._usage_counts.get(self._current_index, 0) + 1
        )
        return key

    def rotate(self) -> str:
        """Rotate to the next API key."""
        if not self._keys:
            return ""
        self._current_index = (self._current_index + 1) % len(self._keys)
        return self._keys[self._current_index]

    def mark_exhausted(self, key_index: int) -> None:
        """Mark a key as rate-limited; skip to next."""
        if key_index == self._current_index:
            self.rotate()

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def get_status(self) -> "Dict[str, Any]":
        return {
            "total_keys": len(self._keys),
            "current_index": self._current_index,
            "usage_counts": dict(self._usage_counts),
        }


# ---------------------------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------------------------


async def integration_health_check() -> "Dict[str, Any]":
    """Health check for all PACK-029 integration bridges.

    Returns availability status for each bridge module.
    """
    bridges = {
        "pack021_bridge": "PACK021Bridge",
        "pack028_bridge": "PACK028Bridge",
        "mrv_bridge": "MRVBridge",
        "sbti_bridge": "SBTiBridge",
        "cdp_bridge": "CDPBridge",
        "tcfd_bridge": "TCFDBridge",
        "initiative_tracker_bridge": "InitiativeTrackerBridge",
        "budget_system_bridge": "BudgetSystemBridge",
        "alerting_bridge": "AlertingBridge",
        "assurance_portal_bridge": "AssurancePortalBridge",
    }

    results: "Dict[str, Any]" = {}
    available = 0
    for module_name, class_name in bridges.items():
        try:
            cls = globals().get(class_name)
            if cls is not None:
                results[module_name] = {"status": "available", "class": class_name}
                available += 1
            else:
                results[module_name] = {"status": "not_loaded", "class": class_name}
        except Exception as exc:
            results[module_name] = {"status": "error", "error": str(exc)}

    return {
        "pack_id": __pack_id__,
        "pack_name": __pack_name__,
        "version": __version__,
        "total_bridges": len(bridges),
        "available": available,
        "unavailable": len(bridges) - available,
        "bridges": results,
    }


# ---------------------------------------------------------------------------
# __all__ Export List
# ---------------------------------------------------------------------------

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- PACK-021 Bridge ---
    "PACK021Bridge",
    "PACK021BridgeConfig",
    "BaselineStatus",
    "SBTiPathwayType",
    "BoundaryApproach",
    "Pack021DataQualityTier",
    "ImportStatus",
    "BaselineImport",
    "LongTermTargetImport",
    "SBTiPathwayImport",
    "BoundaryImport",
    "PACK021IntegrationResult",
    "PACK021_COMPONENTS",
    "PACK021_SBTI_AMBITION",
    "SCOPE3_THRESHOLD_PCT",
    # --- PACK-028 Bridge ---
    "PACK028Bridge",
    "PACK028BridgeConfig",
    "SectorType",
    "MilestoneType",
    "BenchmarkTier",
    "LeverPriority",
    "SectorMilestoneStatus",
    "SectorMilestoneImport",
    "TechnologyRoadmapImport",
    "AbatementLeverImport",
    "SectorBenchmarkImport",
    "PACK028IntegrationResult",
    "PACK028_COMPONENTS",
    "SECTOR_INTERIM_MILESTONES",
    "SECTOR_INTENSITY_METRICS",
    "IEA_TECHNOLOGY_MILESTONES",
    "SECTOR_BENCHMARKS",
    # --- MRV Bridge ---
    "MRVBridge",
    "MRVBridgeConfig",
    "MRVScope",
    "MRVDataQualityTier",
    "AgentHealthStatus",
    "VarianceDirection",
    "AgentResult",
    "ScopeAggregate",
    "VarianceResult",
    "AnnualInventoryResult",
    "MRV_AGENT_REGISTRY",
    # --- SBTi Bridge ---
    "SBTiBridge",
    "SBTiBridgeConfig",
    "SBTiPathway",
    "TargetType",
    "CriteriaStatus",
    "SubmissionStatus",
    "TemperatureRating",
    "LinearityStatus",
    "InterimTargetDefinition",
    "CriteriaValidation",
    "LinearityAssessment",
    "SBTiValidationResult",
    "SBTiSubmissionPackage",
    "SBTI_MINIMUM_AMBITION",
    "INTERIM_TARGET_CRITERIA",
    # --- CDP Bridge ---
    "CDPBridge",
    "CDPBridgeConfig",
    "CDPSection",
    "CDPTargetType",
    "CDPScope",
    "CDPMethodology",
    "ValidationSeverity",
    "CDPC41Export",
    "CDPC42Row",
    "CDPC42Export",
    "CDPC51Export",
    "CDPC61Export",
    "CDPC71Export",
    "CrossValidationResult",
    "CDPExportResult",
    "SCOPE3_CATEGORY_NAMES",
    # --- TCFD Bridge ---
    "TCFDBridge",
    "TCFDBridgeConfig",
    "TCFDPillar",
    "TCFDScenario",
    "TransitionRiskType",
    "RiskLikelihood",
    "RiskImpact",
    "ConsistencyStatus",
    "TCFDTable1Row",
    "TCFDTable1Export",
    "TCFDTable2Row",
    "TCFDTable2Export",
    "TCFDTable3Row",
    "TCFDTable3Export",
    "TCFDTable4Row",
    "TCFDTable4Export",
    "TCFDConsistencyCheck",
    "TCFDExportResult",
    "SCENARIO_CARBON_PRICES",
    "DEFAULT_TRANSITION_RISKS",
    # --- Initiative Tracker Bridge ---
    "InitiativeTrackerBridge",
    "InitiativeTrackerConfig",
    "InitiativeStatus",
    "InitiativeCategory",
    "RAGStatus",
    "BudgetType",
    "VarianceType",
    "Initiative",
    "InitiativePortfolio",
    "VarianceAttribution",
    "ForecastResult",
    "InitiativeTrackerResult",
    # --- Budget System Bridge ---
    "BudgetSystemBridge",
    "BudgetSystemConfig",
    "BudgetScope",
    "CarbonPriceType",
    "BudgetStatus",
    "RebalanceTrigger",
    "AnnualCarbonBudget",
    "CarbonPriceResult",
    "FinancialImpact",
    "RebalanceRecommendation",
    "BudgetSystemResult",
    "DEFAULT_SHADOW_PRICE_SCHEDULE",
    "EU_ETS_PRICE_ASSUMPTIONS",
    # --- Alerting Bridge ---
    "AlertingBridge",
    "AlertingBridgeConfig",
    "AlertSeverity",
    "AlertChannel",
    "AlertStatus",
    "AlertRuleType",
    "EscalationLevel",
    "AlertRule",
    "Alert",
    "AlertEvaluation",
    "DEFAULT_ALERT_RULES",
    # --- Assurance Portal Bridge ---
    "AssurancePortalBridge",
    "AssurancePortalConfig",
    "AssuranceLevel",
    "AssuranceStandard",
    "AssuranceProvider",
    "DocumentType",
    "WorkflowStatus",
    "EvidenceStatus",
    "EvidenceDocument",
    "WorkpaperRequirement",
    "CompletenessCheck",
    "AssuranceRequest",
    "EvidencePackage",
    "ISO_14064_3_REQUIREMENTS",
    # --- Utilities ---
    "create_shared_db_pool",
    "retry_async",
    "timeout_async",
    "CircuitBreaker",
    "AsyncRateLimiter",
    "AsyncResponseCache",
    "APIKeyRotator",
    "integration_health_check",
]
