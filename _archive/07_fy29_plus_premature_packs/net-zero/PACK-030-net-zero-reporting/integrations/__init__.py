# -*- coding: utf-8 -*-
"""
PACK-030 Net Zero Reporting Pack - Integration Layer
=========================================================

Phase 4 integration layer for the Net Zero Reporting Pack providing
12 enterprise integrations for multi-framework report generation across
SBTi, CDP, TCFD, GRI, ISSB, SEC, and CSRD disclosures. Connects to
prerequisite packs (PACK-021/022/028/029), GL applications (SBTi/CDP/
TCFD/GHG), XBRL taxonomy registries, translation services, the GreenLang
Orchestrator, and a health monitoring subsystem.

Components:
    - PACK021Integration: PACK-021 baseline emissions, GHG inventory, and
      activity data import for all framework base year disclosures
    - PACK022Integration: PACK-022 reduction initiatives, MACC curves, and
      abatement action plans for SBTi/CDP/TCFD/GRI/CSRD action reporting
    - PACK028Integration: PACK-028 sector pathways, convergence data, and
      benchmarks for SDA validation and scenario analysis
    - PACK029Integration: PACK-029 interim targets, progress monitoring, and
      variance analysis for SBTi/CDP/TCFD target progress reporting
    - GLSBTiAppIntegration: GL-SBTi-APP target data, 21-criteria validation,
      and submission history for SBTi compliance reporting
    - GLCDPAppIntegration: GL-CDP-APP historical responses, scores, and
      peer benchmarks for CDP questionnaire export
    - GLTCFDAppIntegration: GL-TCFD-APP scenario analysis, climate risks,
      and opportunities for 4-pillar TCFD disclosure
    - GLGHGAppIntegration: GL-GHG-APP GHG inventory, emission factors, and
      activity data from 30 MRV agents for methodology documentation
    - XBRLTaxonomyIntegration: SEC and CSRD XBRL/iXBRL taxonomy fetch,
      cache, and tag validation for digital reporting
    - TranslationIntegration: Multi-language (EN/DE/FR/ES) narrative
      translation with climate glossary and citation preservation
    - OrchestratorIntegration: GreenLang Orchestrator registration, health
      reporting, and orchestrated workflow execution
    - HealthCheckIntegration: Unified health monitoring for all PACK-030
      integrations, prerequisite packs, apps, and infrastructure

Architecture:
    PACK-021 Baseline     --> Data Aggregation Engine
                               |
    PACK-022 Initiatives  --> Narrative Generation Engine
                               |
    PACK-028 Sectors      --> Framework Mapping Engine
                               |
    PACK-029 Targets      --> Validation Engine
                               |
    GL-SBTi-APP           --> SBTi Progress Workflow
    GL-CDP-APP            --> CDP Questionnaire Workflow
    GL-TCFD-APP           --> TCFD Disclosure Workflow
    GL-GHG-APP            --> Multi-Framework Reports
                               |
    XBRL Taxonomy         --> XBRL Tagging Engine
    Translation Service   --> Translation Engine
                               |
    Orchestrator          --> Pack Registration / Workflow Dispatch
    Health Check          --> Dashboard / Alerts

Platform Integrations:
    - packs/net-zero/PACK-021-net-zero-starter/* (baseline/inventory)
    - packs/net-zero/PACK-022-net-zero-acceleration/* (initiatives/MACC)
    - packs/net-zero/PACK-028-sector-pathway/* (sector pathways)
    - packs/net-zero/PACK-029-interim-targets/* (interim targets)
    - greenlang/apps/GL-SBTi-APP/* (SBTi validation)
    - greenlang/apps/GL-CDP-APP/* (CDP questionnaire)
    - greenlang/apps/GL-TCFD-APP/* (TCFD disclosure)
    - greenlang/apps/GL-GHG-APP/* (GHG inventory)
    - greenlang/agents/mrv/* (30 MRV agents)
    - AGENT-FOUND-001 Orchestrator (pack registration)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-030 Net Zero Reporting Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-030"
__pack_name__ = "Net Zero Reporting Pack"

# ---------------------------------------------------------------------------
# PACK-021 Integration (baseline, inventory, activity data)
# ---------------------------------------------------------------------------
try:
    from .pack021_integration import (
        BaselineStatus,
        InventoryScope,
        DataQualityTier as Pack021DataQualityTier,
        BoundaryApproach,
        ImportStatus as Pack021ImportStatus,
        ActivityDataCategory,
        PACK021_COMPONENTS,
        SCOPE3_CATEGORY_NAMES,
        PACK021IntegrationConfig,
        BaselineData,
        InventoryData,
        ActivityDataRecord,
        ActivityDataBundle,
        PACK021IntegrationResult,
        PACK021Integration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("PACK-021 integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# PACK-022 Integration (initiatives, MACC, abatement)
# ---------------------------------------------------------------------------
try:
    from .pack022_integration import (
        InitiativeStatus,
        InitiativeCategory,
        RAGStatus as Pack022RAGStatus,
        AbatementScope,
        MACCPriority,
        ImportStatus as Pack022ImportStatus,
        PACK022_COMPONENTS,
        PACK022IntegrationConfig,
        Initiative,
        InitiativePortfolio,
        MACCLever,
        MACCCurve,
        AbatementAction,
        PACK022IntegrationResult,
        PACK022Integration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("PACK-022 integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# PACK-028 Integration (sector pathways, convergence, benchmarks)
# ---------------------------------------------------------------------------
try:
    from .pack028_integration import (
        SectorType,
        PathwayScenario,
        BenchmarkTier,
        ConvergenceStatus,
        ImportStatus as Pack028ImportStatus,
        PACK028_COMPONENTS,
        SECTOR_INTENSITY_METRICS,
        IEA_NZE_SECTOR_TARGETS,
        SECTOR_BENCHMARKS,
        PACK028IntegrationConfig,
        SectorPathway,
        ConvergenceData,
        SectorBenchmark,
        TechnologyMilestone,
        PACK028IntegrationResult,
        PACK028Integration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("PACK-028 integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# PACK-029 Integration (interim targets, progress, variance)
# ---------------------------------------------------------------------------
try:
    from .pack029_integration import (
        TargetScope,
        TargetType,
        TargetStatus,
        VarianceDirection,
        RAGStatus as Pack029RAGStatus,
        ImportStatus as Pack029ImportStatus,
        PACK029_COMPONENTS,
        PACK029IntegrationConfig,
        InterimTarget,
        TargetPortfolio,
        ProgressRecord,
        ProgressSummary,
        VarianceDetail,
        VarianceReport,
        PACK029IntegrationResult,
        PACK029Integration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("PACK-029 integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# GL-SBTi-APP Integration (targets, validation, submissions)
# ---------------------------------------------------------------------------
try:
    from .gl_sbti_app_integration import (
        SBTiPathway,
        SBTiTargetType,
        SBTiTargetScope,
        ValidationStatus,
        SubmissionStatus,
        TemperatureRating,
        ImportStatus as SBTiImportStatus,
        SBTI_VALIDATION_CRITERIA,
        GLSBTiAppConfig,
        SBTiTarget,
        SBTiTargetPortfolio,
        CriteriaResult,
        SBTiValidationResult,
        SubmissionRecord,
        SubmissionHistory,
        GLSBTiAppResult,
        GLSBTiAppIntegration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("GL-SBTi-APP integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# GL-CDP-APP Integration (history, scores, benchmarks)
# ---------------------------------------------------------------------------
try:
    from .gl_cdp_app_integration import (
        CDPScore,
        CDPModule,
        CDPScoringCategory,
        ImportStatus as CDPImportStatus,
        CDP_MODULE_INFO,
        GLCDPAppConfig,
        CDPModuleResponse,
        CDPHistoryYear,
        CDPHistory,
        CDPScoreDetail,
        CDPPeerBenchmark,
        GLCDPAppResult,
        GLCDPAppIntegration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("GL-CDP-APP integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# GL-TCFD-APP Integration (scenarios, risks, opportunities)
# ---------------------------------------------------------------------------
try:
    from .gl_tcfd_app_integration import (
        TCFDPillar,
        ScenarioType,
        RiskType,
        RiskLikelihood,
        RiskImpact,
        RiskTimeHorizon,
        OpportunityType,
        ImportStatus as TCFDImportStatus,
        GLTCFDAppConfig,
        ScenarioResult,
        ScenarioAnalysis,
        ClimateRisk,
        RiskAssessment,
        ClimateOpportunity,
        OpportunityAssessment,
        GLTCFDAppResult,
        GLTCFDAppIntegration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("GL-TCFD-APP integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# GL-GHG-APP Integration (inventory, EFs, activity data)
# ---------------------------------------------------------------------------
try:
    from .gl_ghg_app_integration import (
        GHGScope,
        EFSource,
        VerificationLevel,
        InventoryStatus,
        ImportStatus as GHGImportStatus,
        MRV_AGENT_REGISTRY,
        GLGHGAppConfig,
        GHGInventory,
        EmissionFactorRecord,
        EmissionFactorBundle,
        ActivityDataSummary,
        GLGHGAppResult,
        GLGHGAppIntegration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("GL-GHG-APP integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# XBRL Taxonomy Integration (SEC, CSRD, tag validation)
# ---------------------------------------------------------------------------
try:
    from .xbrl_taxonomy_integration import (
        TaxonomyFramework,
        TagDataType,
        ValidationSeverity as XBRLValidationSeverity,
        ImportStatus as XBRLImportStatus,
        SEC_TAXONOMY_ELEMENTS,
        CSRD_TAXONOMY_ELEMENTS,
        XBRLIntegrationConfig,
        TaxonomyElement,
        Taxonomy,
        TagValidationResult,
        XBRLIntegrationResult,
        XBRLTaxonomyIntegration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("XBRL taxonomy integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# Translation Integration (multi-language, glossary)
# ---------------------------------------------------------------------------
try:
    from .translation_integration import (
        SupportedLanguage,
        TranslationProvider,
        TranslationQuality,
        CLIMATE_GLOSSARY,
        TranslationConfig,
        TranslationResult,
        LanguageDetectionResult,
        QualityValidationResult,
        TranslationIntegration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("Translation integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# Orchestrator Integration (registration, health, workflow)
# ---------------------------------------------------------------------------
try:
    from .orchestrator_integration import (
        PackStatus,
        WorkflowStatus,
        HealthStatus as OrchestratorHealthStatus,
        PACK030_CAPABILITIES,
        PACK030_DEPENDENCIES,
        OrchestratorConfig,
        PackRegistration,
        HealthReport,
        OrchestrationRequest,
        OrchestrationResponse,
        OrchestratorIntegration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("Orchestrator integration import failed: %s", _e)

# ---------------------------------------------------------------------------
# Health Check Integration (unified health monitoring)
# ---------------------------------------------------------------------------
try:
    from .health_check_integration import (
        HealthStatus,
        ComponentType,
        MONITORED_COMPONENTS,
        HealthCheckConfig,
        ComponentHealth,
        PackHealthResult,
        AppHealthResult,
        ExternalServiceResult,
        OverallHealthReport,
        HealthCheckIntegration,
    )
except ImportError as _e:
    import logging as _log
    _log.getLogger(__name__).debug("Health check integration import failed: %s", _e)


# ---------------------------------------------------------------------------
# Connection Pooling Utilities
# ---------------------------------------------------------------------------

async def create_shared_db_pool(
    connection_string: str,
    min_size: int = 1,
    max_size: int = 10,
) -> "Any":
    """Create a shared async database connection pool for all integrations.

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
from typing import Any, Callable, Dict, List, Optional, TypeVar

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

import json
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
        prefix: str = "pack030:",
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
    """Health check for all PACK-030 integration modules.

    Returns availability status for each integration module.
    """
    integrations = {
        "pack021_integration": "PACK021Integration",
        "pack022_integration": "PACK022Integration",
        "pack028_integration": "PACK028Integration",
        "pack029_integration": "PACK029Integration",
        "gl_sbti_app_integration": "GLSBTiAppIntegration",
        "gl_cdp_app_integration": "GLCDPAppIntegration",
        "gl_tcfd_app_integration": "GLTCFDAppIntegration",
        "gl_ghg_app_integration": "GLGHGAppIntegration",
        "xbrl_taxonomy_integration": "XBRLTaxonomyIntegration",
        "translation_integration": "TranslationIntegration",
        "orchestrator_integration": "OrchestratorIntegration",
        "health_check_integration": "HealthCheckIntegration",
    }

    results: "Dict[str, Any]" = {}
    available = 0
    for module_name, class_name in integrations.items():
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
        "total_integrations": len(integrations),
        "available": available,
        "unavailable": len(integrations) - available,
        "integrations": results,
    }


# ---------------------------------------------------------------------------
# __all__ Export List
# ---------------------------------------------------------------------------

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- PACK-021 Integration ---
    "PACK021Integration",
    "PACK021IntegrationConfig",
    "BaselineStatus",
    "InventoryScope",
    "Pack021DataQualityTier",
    "BoundaryApproach",
    "Pack021ImportStatus",
    "ActivityDataCategory",
    "PACK021_COMPONENTS",
    "SCOPE3_CATEGORY_NAMES",
    "BaselineData",
    "InventoryData",
    "ActivityDataRecord",
    "ActivityDataBundle",
    "PACK021IntegrationResult",
    # --- PACK-022 Integration ---
    "PACK022Integration",
    "PACK022IntegrationConfig",
    "InitiativeStatus",
    "InitiativeCategory",
    "Pack022RAGStatus",
    "AbatementScope",
    "MACCPriority",
    "Pack022ImportStatus",
    "PACK022_COMPONENTS",
    "Initiative",
    "InitiativePortfolio",
    "MACCLever",
    "MACCCurve",
    "AbatementAction",
    "PACK022IntegrationResult",
    # --- PACK-028 Integration ---
    "PACK028Integration",
    "PACK028IntegrationConfig",
    "SectorType",
    "PathwayScenario",
    "BenchmarkTier",
    "ConvergenceStatus",
    "Pack028ImportStatus",
    "PACK028_COMPONENTS",
    "SECTOR_INTENSITY_METRICS",
    "IEA_NZE_SECTOR_TARGETS",
    "SECTOR_BENCHMARKS",
    "SectorPathway",
    "ConvergenceData",
    "SectorBenchmark",
    "TechnologyMilestone",
    "PACK028IntegrationResult",
    # --- PACK-029 Integration ---
    "PACK029Integration",
    "PACK029IntegrationConfig",
    "TargetScope",
    "TargetType",
    "TargetStatus",
    "VarianceDirection",
    "Pack029RAGStatus",
    "Pack029ImportStatus",
    "PACK029_COMPONENTS",
    "InterimTarget",
    "TargetPortfolio",
    "ProgressRecord",
    "ProgressSummary",
    "VarianceDetail",
    "VarianceReport",
    "PACK029IntegrationResult",
    # --- GL-SBTi-APP Integration ---
    "GLSBTiAppIntegration",
    "GLSBTiAppConfig",
    "SBTiPathway",
    "SBTiTargetType",
    "SBTiTargetScope",
    "ValidationStatus",
    "SubmissionStatus",
    "TemperatureRating",
    "SBTiImportStatus",
    "SBTI_VALIDATION_CRITERIA",
    "SBTiTarget",
    "SBTiTargetPortfolio",
    "CriteriaResult",
    "SBTiValidationResult",
    "SubmissionRecord",
    "SubmissionHistory",
    "GLSBTiAppResult",
    # --- GL-CDP-APP Integration ---
    "GLCDPAppIntegration",
    "GLCDPAppConfig",
    "CDPScore",
    "CDPModule",
    "CDPScoringCategory",
    "CDPImportStatus",
    "CDP_MODULE_INFO",
    "CDPModuleResponse",
    "CDPHistoryYear",
    "CDPHistory",
    "CDPScoreDetail",
    "CDPPeerBenchmark",
    "GLCDPAppResult",
    # --- GL-TCFD-APP Integration ---
    "GLTCFDAppIntegration",
    "GLTCFDAppConfig",
    "TCFDPillar",
    "ScenarioType",
    "RiskType",
    "RiskLikelihood",
    "RiskImpact",
    "RiskTimeHorizon",
    "OpportunityType",
    "TCFDImportStatus",
    "ScenarioResult",
    "ScenarioAnalysis",
    "ClimateRisk",
    "RiskAssessment",
    "ClimateOpportunity",
    "OpportunityAssessment",
    "GLTCFDAppResult",
    # --- GL-GHG-APP Integration ---
    "GLGHGAppIntegration",
    "GLGHGAppConfig",
    "GHGScope",
    "EFSource",
    "VerificationLevel",
    "InventoryStatus",
    "GHGImportStatus",
    "MRV_AGENT_REGISTRY",
    "GHGInventory",
    "EmissionFactorRecord",
    "EmissionFactorBundle",
    "ActivityDataSummary",
    "GLGHGAppResult",
    # --- XBRL Taxonomy Integration ---
    "XBRLTaxonomyIntegration",
    "XBRLIntegrationConfig",
    "TaxonomyFramework",
    "TagDataType",
    "XBRLValidationSeverity",
    "XBRLImportStatus",
    "SEC_TAXONOMY_ELEMENTS",
    "CSRD_TAXONOMY_ELEMENTS",
    "TaxonomyElement",
    "Taxonomy",
    "TagValidationResult",
    "XBRLIntegrationResult",
    # --- Translation Integration ---
    "TranslationIntegration",
    "TranslationConfig",
    "SupportedLanguage",
    "TranslationProvider",
    "TranslationQuality",
    "CLIMATE_GLOSSARY",
    "TranslationResult",
    "LanguageDetectionResult",
    "QualityValidationResult",
    # --- Orchestrator Integration ---
    "OrchestratorIntegration",
    "OrchestratorConfig",
    "PackStatus",
    "WorkflowStatus",
    "OrchestratorHealthStatus",
    "PACK030_CAPABILITIES",
    "PACK030_DEPENDENCIES",
    "PackRegistration",
    "HealthReport",
    "OrchestrationRequest",
    "OrchestrationResponse",
    # --- Health Check Integration ---
    "HealthCheckIntegration",
    "HealthCheckConfig",
    "HealthStatus",
    "ComponentType",
    "MONITORED_COMPONENTS",
    "ComponentHealth",
    "PackHealthResult",
    "AppHealthResult",
    "ExternalServiceResult",
    "OverallHealthReport",
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
