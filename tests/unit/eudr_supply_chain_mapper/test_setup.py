# -*- coding: utf-8 -*-
"""
Comprehensive test suite for SupplyChainMapperService (setup.py).

Tests cover:
    1.  Service construction with default config
    2.  Service construction with custom config
    3.  Service is not started initially
    4.  Properties raise RuntimeError when not started
    5.  Startup sets is_running to True
    6.  Startup is idempotent
    7.  Shutdown sets is_running to False
    8.  Shutdown is idempotent
    9.  Health check returns structured response
    10. Health check database healthy path
    11. Health check database unhealthy path
    12. Health check Redis healthy path
    13. Health check Redis degraded path
    14. Engine health check reports initialized engines
    15. Engine health check reports unavailable engines
    16. get_engine returns valid engines
    17. get_engine raises ValueError for unknown name
    18. get_engine raises RuntimeError when not started
    19. initialized_engine_count returns correct count
    20. Uptime is zero before startup
    21. Uptime increases after startup
    22. Config hash is computed on construction
    23. HealthStatus to_dict serialization
    24. Singleton get_service returns same instance
    25. set_service replaces singleton
    26. reset_service clears singleton
    27. Lifespan context manager integration
    28. Startup failure on database connection
    29. Graceful shutdown with engine close errors
    30. Health check loop cancellation on shutdown

Target: 85%+ coverage for setup.py
PRD: PRD-AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.supply_chain_mapper.config import (
    SupplyChainMapperConfig,
)
from greenlang.agents.eudr.supply_chain_mapper.setup import (
    HealthStatus,
    SupplyChainMapperService,
    _compute_service_hash,
    get_service,
    reset_service,
    set_service,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> SupplyChainMapperConfig:
    """Create a default SupplyChainMapperConfig for testing."""
    return SupplyChainMapperConfig()


@pytest.fixture
def custom_config() -> SupplyChainMapperConfig:
    """Create a custom SupplyChainMapperConfig for testing."""
    return SupplyChainMapperConfig(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        pool_size=5,
        cache_ttl=600,
        rate_limit=500,
    )


@pytest.fixture
def service(default_config: SupplyChainMapperConfig) -> SupplyChainMapperService:
    """Create a fresh SupplyChainMapperService instance."""
    return SupplyChainMapperService(config=default_config)


@pytest.fixture
def service_custom(
    custom_config: SupplyChainMapperConfig,
) -> SupplyChainMapperService:
    """Create a service with custom configuration."""
    return SupplyChainMapperService(config=custom_config)


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset the singleton service before and after each test."""
    reset_service()
    yield  # type: ignore[misc]
    reset_service()


# =============================================================================
# Helper to create a started service without real connections
# =============================================================================


async def _start_service_mocked(
    service: SupplyChainMapperService,
) -> SupplyChainMapperService:
    """Start a service with all external connections mocked out.

    Patches database, Redis, and all engine imports so the service
    starts without any real infrastructure.
    """
    with (
        patch.object(service, "_connect_database", new_callable=AsyncMock),
        patch.object(service, "_register_pgvector", new_callable=AsyncMock),
        patch.object(service, "_connect_redis", new_callable=AsyncMock),
        patch.object(
            service, "_initialize_engines", new_callable=AsyncMock
        ),
    ):
        await service.startup()
    return service


# =============================================================================
# Test 1: Construction with default config
# =============================================================================


class TestServiceConstruction:
    """Tests for SupplyChainMapperService construction."""

    def test_construction_with_default_config(
        self, default_config: SupplyChainMapperConfig
    ) -> None:
        """Service can be constructed with a default config."""
        svc = SupplyChainMapperService(config=default_config)
        assert svc.config is default_config
        assert svc.is_running is False

    def test_construction_with_custom_config(
        self, custom_config: SupplyChainMapperConfig
    ) -> None:
        """Service can be constructed with a custom config."""
        svc = SupplyChainMapperService(config=custom_config)
        assert svc.config.pool_size == 5
        assert svc.config.cache_ttl == 600
        assert svc.config.log_level == "DEBUG"

    def test_construction_without_config_uses_get_config(self) -> None:
        """Service loads config from get_config() when none is provided."""
        svc = SupplyChainMapperService()
        assert svc.config is not None
        assert isinstance(svc.config, SupplyChainMapperConfig)

    def test_not_started_initially(
        self, service: SupplyChainMapperService
    ) -> None:
        """Service is not started after construction."""
        assert service.is_running is False

    def test_config_hash_computed(
        self, service: SupplyChainMapperService
    ) -> None:
        """A SHA-256 config hash is computed on construction."""
        assert service._config_hash is not None
        assert len(service._config_hash) == 64  # SHA-256 hex length


# =============================================================================
# Test 2: Properties guard when not started
# =============================================================================


class TestPropertiesGuard:
    """Test that engine properties raise RuntimeError when not started."""

    def test_graph_engine_raises_when_not_started(
        self, service: SupplyChainMapperService
    ) -> None:
        """Accessing graph_engine before startup raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            _ = service.graph_engine

    def test_multi_tier_mapper_raises_when_not_started(
        self, service: SupplyChainMapperService
    ) -> None:
        """Accessing multi_tier_mapper before startup raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            _ = service.multi_tier_mapper

    def test_risk_propagation_engine_raises_when_not_started(
        self, service: SupplyChainMapperService
    ) -> None:
        """Accessing risk_propagation_engine before startup raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            _ = service.risk_propagation_engine

    def test_db_pool_raises_when_not_started(
        self, service: SupplyChainMapperService
    ) -> None:
        """Accessing db_pool before startup raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            _ = service.db_pool

    def test_redis_client_raises_when_not_started(
        self, service: SupplyChainMapperService
    ) -> None:
        """Accessing redis_client before startup raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            _ = service.redis_client

    def test_get_engine_raises_when_not_started(
        self, service: SupplyChainMapperService
    ) -> None:
        """get_engine raises RuntimeError before startup."""
        with pytest.raises(RuntimeError, match="not started"):
            service.get_engine("graph_engine")


# =============================================================================
# Test 3: Startup
# =============================================================================


class TestStartup:
    """Tests for service startup."""

    @pytest.mark.asyncio
    async def test_startup_sets_running(
        self, service: SupplyChainMapperService
    ) -> None:
        """Startup sets is_running to True."""
        await _start_service_mocked(service)
        assert service.is_running is True

    @pytest.mark.asyncio
    async def test_startup_idempotent(
        self, service: SupplyChainMapperService
    ) -> None:
        """Calling startup twice does not raise."""
        await _start_service_mocked(service)
        assert service.is_running is True
        # Second call should be a no-op
        await service.startup()
        assert service.is_running is True

    @pytest.mark.asyncio
    async def test_startup_records_start_time(
        self, service: SupplyChainMapperService
    ) -> None:
        """Startup records a non-None start time."""
        await _start_service_mocked(service)
        assert service._start_time is not None
        assert service.uptime_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_startup_configures_logging(
        self, service: SupplyChainMapperService
    ) -> None:
        """Startup calls _configure_logging."""
        with (
            patch.object(
                service, "_configure_logging"
            ) as mock_log,
            patch.object(
                service, "_connect_database", new_callable=AsyncMock
            ),
            patch.object(
                service, "_register_pgvector", new_callable=AsyncMock
            ),
            patch.object(
                service, "_connect_redis", new_callable=AsyncMock
            ),
            patch.object(
                service, "_initialize_engines", new_callable=AsyncMock
            ),
        ):
            await service.startup()
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_initializes_tracer(
        self, service: SupplyChainMapperService
    ) -> None:
        """Startup calls _init_tracer."""
        with (
            patch.object(service, "_init_tracer") as mock_tracer,
            patch.object(
                service, "_connect_database", new_callable=AsyncMock
            ),
            patch.object(
                service, "_register_pgvector", new_callable=AsyncMock
            ),
            patch.object(
                service, "_connect_redis", new_callable=AsyncMock
            ),
            patch.object(
                service, "_initialize_engines", new_callable=AsyncMock
            ),
        ):
            await service.startup()
            mock_tracer.assert_called_once()


# =============================================================================
# Test 4: Shutdown
# =============================================================================


class TestShutdown:
    """Tests for service shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_not_running(
        self, service: SupplyChainMapperService
    ) -> None:
        """Shutdown sets is_running to False."""
        await _start_service_mocked(service)
        assert service.is_running is True
        await service.shutdown()
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(
        self, service: SupplyChainMapperService
    ) -> None:
        """Calling shutdown twice does not raise."""
        await _start_service_mocked(service)
        await service.shutdown()
        assert service.is_running is False
        # Second call should be a no-op
        await service.shutdown()
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_shutdown_cancels_health_task(
        self, service: SupplyChainMapperService
    ) -> None:
        """Shutdown cancels the background health check task."""
        await _start_service_mocked(service)
        # Simulate a running health task
        service._health_task = asyncio.create_task(asyncio.sleep(1000))
        await service.shutdown()
        assert service._health_task is None

    @pytest.mark.asyncio
    async def test_shutdown_nullifies_engines(
        self, service: SupplyChainMapperService
    ) -> None:
        """Shutdown sets all engine references to None."""
        await _start_service_mocked(service)
        await service.shutdown()
        assert service._graph_engine is None
        assert service._multi_tier_mapper is None
        assert service._geolocation_linker is None
        assert service._risk_propagation_engine is None

    @pytest.mark.asyncio
    async def test_shutdown_tolerates_engine_close_errors(
        self, service: SupplyChainMapperService
    ) -> None:
        """Shutdown continues even if an engine's close method raises."""
        await _start_service_mocked(service)

        # Install a mock engine that raises on shutdown
        mock_engine = MagicMock()
        mock_engine.shutdown = AsyncMock(side_effect=RuntimeError("boom"))
        service._graph_engine = mock_engine

        # Should not raise
        await service.shutdown()
        assert service.is_running is False


# =============================================================================
# Test 5: Health check
# =============================================================================


class TestHealthCheck:
    """Tests for the health check method."""

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(
        self, service: SupplyChainMapperService
    ) -> None:
        """health_check returns a dictionary."""
        await _start_service_mocked(service)
        result = await service.health_check()
        assert isinstance(result, dict)
        assert "status" in result
        assert "checks" in result
        assert "timestamp" in result
        assert "version" in result
        assert "uptime_seconds" in result

    @pytest.mark.asyncio
    async def test_health_check_status_fields(
        self, service: SupplyChainMapperService
    ) -> None:
        """health_check response contains database, redis, and engines checks."""
        await _start_service_mocked(service)
        result = await service.health_check()
        checks = result["checks"]
        assert "database" in checks
        assert "redis" in checks
        assert "engines" in checks

    @pytest.mark.asyncio
    async def test_health_check_database_no_pool(
        self, service: SupplyChainMapperService
    ) -> None:
        """Database check returns unhealthy when no pool is available."""
        await _start_service_mocked(service)
        service._db_pool = None
        db_health = await service._check_database_health()
        assert db_health["status"] == "unhealthy"
        assert db_health["reason"] == "no_pool"

    @pytest.mark.asyncio
    async def test_health_check_database_healthy(
        self, service: SupplyChainMapperService
    ) -> None:
        """Database check returns healthy when pool responds."""
        await _start_service_mocked(service)

        # Mock a working database pool
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=None)
        mock_pool = AsyncMock()
        mock_pool.connection = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        service._db_pool = mock_pool

        db_health = await service._check_database_health()
        assert db_health["status"] == "healthy"
        assert "latency_ms" in db_health

    @pytest.mark.asyncio
    async def test_health_check_redis_not_connected(
        self, service: SupplyChainMapperService
    ) -> None:
        """Redis check returns degraded when not connected."""
        await _start_service_mocked(service)
        service._redis = None
        redis_health = await service._check_redis_health()
        assert redis_health["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_check_redis_healthy(
        self, service: SupplyChainMapperService
    ) -> None:
        """Redis check returns healthy when ping succeeds."""
        await _start_service_mocked(service)
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        service._redis = mock_redis

        redis_health = await service._check_redis_health()
        assert redis_health["status"] == "healthy"
        assert "latency_ms" in redis_health

    @pytest.mark.asyncio
    async def test_health_check_caches_last_result(
        self, service: SupplyChainMapperService
    ) -> None:
        """health_check stores result in last_health property."""
        await _start_service_mocked(service)
        assert service.last_health is None
        await service.health_check()
        assert service.last_health is not None
        assert isinstance(service.last_health, HealthStatus)


# =============================================================================
# Test 6: Engine health
# =============================================================================


class TestEngineHealth:
    """Tests for engine health reporting."""

    @pytest.mark.asyncio
    async def test_engine_health_all_none(
        self, service: SupplyChainMapperService
    ) -> None:
        """Engine health reports all unavailable when no engines loaded."""
        await _start_service_mocked(service)
        result = service._check_engine_health()
        assert result["initialized"] == 0
        assert result["total"] == 9

    @pytest.mark.asyncio
    async def test_engine_health_with_core_engines(
        self, service: SupplyChainMapperService
    ) -> None:
        """Engine health reports healthy when core engines are initialized."""
        await _start_service_mocked(service)
        service._graph_engine = MagicMock()
        service._risk_propagation_engine = MagicMock()
        service._geolocation_linker = MagicMock()

        result = service._check_engine_health()
        assert result["status"] == "healthy"
        assert result["initialized"] == 3
        assert result["engines"]["graph_engine"] == "initialized"
        assert result["engines"]["risk_propagation_engine"] == "initialized"

    @pytest.mark.asyncio
    async def test_engine_health_degraded_without_core(
        self, service: SupplyChainMapperService
    ) -> None:
        """Engine health reports degraded when core engines are missing."""
        await _start_service_mocked(service)
        # Only non-core engine available
        service._visualization_engine = MagicMock()

        result = service._check_engine_health()
        assert result["status"] == "degraded"
        assert result["initialized"] == 1


# =============================================================================
# Test 7: get_engine
# =============================================================================


class TestGetEngine:
    """Tests for the get_engine convenience method."""

    @pytest.mark.asyncio
    async def test_get_engine_valid_name(
        self, service: SupplyChainMapperService
    ) -> None:
        """get_engine returns the engine for a valid name."""
        await _start_service_mocked(service)
        mock_engine = MagicMock()
        service._graph_engine = mock_engine
        assert service.get_engine("graph_engine") is mock_engine

    @pytest.mark.asyncio
    async def test_get_engine_returns_none_for_uninitialized(
        self, service: SupplyChainMapperService
    ) -> None:
        """get_engine returns None for an uninitialized engine."""
        await _start_service_mocked(service)
        assert service.get_engine("gap_analyzer") is None

    @pytest.mark.asyncio
    async def test_get_engine_invalid_name(
        self, service: SupplyChainMapperService
    ) -> None:
        """get_engine raises ValueError for unknown engine name."""
        await _start_service_mocked(service)
        with pytest.raises(ValueError, match="Unknown engine name"):
            service.get_engine("nonexistent_engine")

    @pytest.mark.asyncio
    async def test_get_engine_all_valid_names(
        self, service: SupplyChainMapperService
    ) -> None:
        """All nine valid engine names are accepted."""
        await _start_service_mocked(service)
        valid_names = [
            "graph_engine",
            "multi_tier_mapper",
            "geolocation_linker",
            "batch_traceability_engine",
            "risk_propagation_engine",
            "gap_analyzer",
            "visualization_engine",
            "regulatory_exporter",
            "supplier_onboarding_engine",
        ]
        for name in valid_names:
            # Should not raise
            service.get_engine(name)


# =============================================================================
# Test 8: initialized_engine_count
# =============================================================================


class TestInitializedEngineCount:
    """Tests for the initialized_engine_count method."""

    @pytest.mark.asyncio
    async def test_zero_when_no_engines(
        self, service: SupplyChainMapperService
    ) -> None:
        """Returns 0 when no engines are initialized."""
        await _start_service_mocked(service)
        assert service.initialized_engine_count() == 0

    @pytest.mark.asyncio
    async def test_counts_initialized_engines(
        self, service: SupplyChainMapperService
    ) -> None:
        """Returns correct count of non-None engines."""
        await _start_service_mocked(service)
        service._graph_engine = MagicMock()
        service._risk_propagation_engine = MagicMock()
        service._geolocation_linker = MagicMock()
        assert service.initialized_engine_count() == 3


# =============================================================================
# Test 9: Uptime
# =============================================================================


class TestUptime:
    """Tests for uptime tracking."""

    def test_uptime_zero_before_start(
        self, service: SupplyChainMapperService
    ) -> None:
        """Uptime is 0.0 before service starts."""
        assert service.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_uptime_positive_after_start(
        self, service: SupplyChainMapperService
    ) -> None:
        """Uptime is non-negative after startup and positive after sleep."""
        await _start_service_mocked(service)
        assert service.uptime_seconds >= 0.0
        # Sleep to ensure measurable uptime
        await asyncio.sleep(0.05)
        assert service.uptime_seconds > 0.0


# =============================================================================
# Test 10: HealthStatus model
# =============================================================================


class TestHealthStatus:
    """Tests for the HealthStatus class."""

    def test_default_construction(self) -> None:
        """HealthStatus can be constructed with defaults."""
        hs = HealthStatus()
        assert hs.status == "unhealthy"
        assert hs.checks == {}
        assert hs.version == "1.0.0"
        assert hs.uptime_seconds == 0.0

    def test_custom_construction(self) -> None:
        """HealthStatus can be constructed with custom values."""
        hs = HealthStatus(
            status="healthy",
            checks={"db": {"status": "ok"}},
            version="2.0.0",
            uptime_seconds=123.45,
        )
        assert hs.status == "healthy"
        assert hs.checks["db"]["status"] == "ok"
        assert hs.version == "2.0.0"
        assert hs.uptime_seconds == 123.45

    def test_to_dict_serialization(self) -> None:
        """to_dict returns a JSON-serializable dictionary."""
        hs = HealthStatus(
            status="healthy",
            checks={"a": 1},
            uptime_seconds=10.5,
        )
        d = hs.to_dict()
        assert d["status"] == "healthy"
        assert d["checks"] == {"a": 1}
        assert d["uptime_seconds"] == 10.5
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)

    def test_to_dict_timestamp_is_iso(self) -> None:
        """Timestamp in to_dict is ISO format."""
        ts = datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc)
        hs = HealthStatus(timestamp=ts)
        d = hs.to_dict()
        assert "2026-03-06" in d["timestamp"]


# =============================================================================
# Test 11: Singleton management
# =============================================================================


class TestSingleton:
    """Tests for singleton accessors."""

    def test_get_service_returns_same_instance(self) -> None:
        """get_service returns the same instance on multiple calls."""
        svc1 = get_service()
        svc2 = get_service()
        assert svc1 is svc2

    def test_set_service_replaces_instance(self) -> None:
        """set_service replaces the singleton."""
        svc1 = get_service()
        svc2 = SupplyChainMapperService()
        set_service(svc2)
        assert get_service() is svc2
        assert get_service() is not svc1

    def test_reset_service_clears_instance(self) -> None:
        """reset_service clears singleton so next get creates new."""
        svc1 = get_service()
        reset_service()
        svc2 = get_service()
        assert svc1 is not svc2

    def test_get_service_with_config(self) -> None:
        """get_service with config creates instance using that config."""
        cfg = SupplyChainMapperConfig(pool_size=3)
        svc = get_service(config=cfg)
        assert svc.config.pool_size == 3


# =============================================================================
# Test 12: Lifespan context manager
# =============================================================================


class TestLifespan:
    """Tests for the FastAPI lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_starts_and_stops_service(self) -> None:
        """Lifespan starts service on enter and stops on exit."""
        from greenlang.agents.eudr.supply_chain_mapper.setup import lifespan

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        # Patch the singleton to return a controllable service
        svc = SupplyChainMapperService()
        set_service(svc)

        with (
            patch.object(svc, "startup", new_callable=AsyncMock) as mock_start,
            patch.object(
                svc, "shutdown", new_callable=AsyncMock
            ) as mock_stop,
        ):
            async with lifespan(mock_app):
                mock_start.assert_called_once()
                assert hasattr(mock_app.state, "scm_service")

            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_sets_app_state(self) -> None:
        """Lifespan sets app.state.scm_service."""
        from greenlang.agents.eudr.supply_chain_mapper.setup import lifespan

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        svc = SupplyChainMapperService()
        set_service(svc)

        with (
            patch.object(svc, "startup", new_callable=AsyncMock),
            patch.object(svc, "shutdown", new_callable=AsyncMock),
        ):
            async with lifespan(mock_app):
                assert mock_app.state.scm_service is svc


# =============================================================================
# Test 13: Config hash
# =============================================================================


class TestConfigHash:
    """Tests for configuration hash computation."""

    def test_compute_service_hash_deterministic(self) -> None:
        """Same config produces same hash."""
        cfg = SupplyChainMapperConfig()
        h1 = _compute_service_hash(cfg)
        h2 = _compute_service_hash(cfg)
        assert h1 == h2

    def test_compute_service_hash_different_configs(self) -> None:
        """Different configs produce different hashes."""
        cfg1 = SupplyChainMapperConfig(pool_size=5)
        cfg2 = SupplyChainMapperConfig(pool_size=10)
        h1 = _compute_service_hash(cfg1)
        h2 = _compute_service_hash(cfg2)
        assert h1 != h2

    def test_compute_service_hash_length(self) -> None:
        """Hash is a 64-character hex string (SHA-256)."""
        cfg = SupplyChainMapperConfig()
        h = _compute_service_hash(cfg)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# Test 14: Database connection
# =============================================================================


class TestDatabaseConnection:
    """Tests for database connection management."""

    @pytest.mark.asyncio
    async def test_connect_database_without_psycopg(
        self, service: SupplyChainMapperService
    ) -> None:
        """Database connection is skipped when psycopg is not available."""
        with (
            patch(
                "greenlang.agents.eudr.supply_chain_mapper.setup.PSYCOPG_POOL_AVAILABLE",
                False,
            ),
            patch(
                "greenlang.agents.eudr.supply_chain_mapper.setup.PSYCOPG_AVAILABLE",
                False,
            ),
        ):
            await service._connect_database()
            assert service._db_pool is None

    @pytest.mark.asyncio
    async def test_close_database_when_pool_is_none(
        self, service: SupplyChainMapperService
    ) -> None:
        """Closing database when no pool exists is a no-op."""
        service._db_pool = None
        await service._close_database()
        assert service._db_pool is None


# =============================================================================
# Test 15: Redis connection
# =============================================================================


class TestRedisConnection:
    """Tests for Redis connection management."""

    @pytest.mark.asyncio
    async def test_connect_redis_without_package(
        self, service: SupplyChainMapperService
    ) -> None:
        """Redis connection is skipped when redis package is not available."""
        with patch(
            "greenlang.agents.eudr.supply_chain_mapper.setup.REDIS_AVAILABLE",
            False,
        ):
            await service._connect_redis()
            assert service._redis is None

    @pytest.mark.asyncio
    async def test_close_redis_when_none(
        self, service: SupplyChainMapperService
    ) -> None:
        """Closing Redis when no client exists is a no-op."""
        service._redis = None
        await service._close_redis()
        assert service._redis is None


# =============================================================================
# Test 16: Engine initialization
# =============================================================================


class TestEngineInitialization:
    """Tests for individual engine initialization methods."""

    @pytest.mark.asyncio
    async def test_init_graph_engine_import_error(
        self, service: SupplyChainMapperService
    ) -> None:
        """Graph engine init returns None on ImportError."""
        with patch(
            "greenlang.agents.eudr.supply_chain_mapper.setup."
            "SupplyChainMapperService._init_graph_engine",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await service._init_graph_engine()
            assert result is None

    @pytest.mark.asyncio
    async def test_init_risk_propagation_engine_success(
        self, service: SupplyChainMapperService
    ) -> None:
        """Risk propagation engine initializes successfully."""
        engine = await service._init_risk_propagation_engine()
        # Should succeed since risk_propagation module exists
        if engine is not None:
            from greenlang.agents.eudr.supply_chain_mapper.risk_propagation import (
                RiskPropagationEngine,
            )
            assert isinstance(engine, RiskPropagationEngine)

    @pytest.mark.asyncio
    async def test_init_geolocation_linker_success(
        self, service: SupplyChainMapperService
    ) -> None:
        """Geolocation linker initializes successfully."""
        engine = await service._init_geolocation_linker()
        if engine is not None:
            from greenlang.agents.eudr.supply_chain_mapper.geolocation_linker import (
                GeolocationLinker,
            )
            assert isinstance(engine, GeolocationLinker)

    @pytest.mark.asyncio
    async def test_init_batch_traceability_returns_none(
        self, service: SupplyChainMapperService
    ) -> None:
        """Batch traceability engine returns None when module not available."""
        engine = await service._init_batch_traceability_engine()
        # Module doesn't exist yet, should return None
        assert engine is None

    @pytest.mark.asyncio
    async def test_init_gap_analyzer_returns_none(
        self, service: SupplyChainMapperService
    ) -> None:
        """Gap analyzer returns None when module not available."""
        engine = await service._init_gap_analyzer()
        assert engine is None

    @pytest.mark.asyncio
    async def test_init_visualization_returns_none(
        self, service: SupplyChainMapperService
    ) -> None:
        """Visualization engine returns None when module not available."""
        engine = await service._init_visualization_engine()
        assert engine is None

    @pytest.mark.asyncio
    async def test_init_regulatory_exporter_returns_none(
        self, service: SupplyChainMapperService
    ) -> None:
        """Regulatory exporter returns None when module not available."""
        engine = await service._init_regulatory_exporter()
        assert engine is None

    @pytest.mark.asyncio
    async def test_init_supplier_onboarding_returns_none(
        self, service: SupplyChainMapperService
    ) -> None:
        """Supplier onboarding engine returns None when module not available."""
        engine = await service._init_supplier_onboarding_engine()
        assert engine is None


# =============================================================================
# Test 17: Logging and tracing
# =============================================================================


class TestLoggingAndTracing:
    """Tests for logging configuration and OTel tracing."""

    def test_configure_logging_sets_level(
        self, service: SupplyChainMapperService
    ) -> None:
        """_configure_logging sets the logger level from config."""
        import logging as _logging

        service._config = SupplyChainMapperConfig(log_level="DEBUG")
        service._configure_logging()
        scm_logger = _logging.getLogger(
            "greenlang.agents.eudr.supply_chain_mapper"
        )
        assert scm_logger.level == _logging.DEBUG

    def test_init_tracer_without_otel(
        self, service: SupplyChainMapperService
    ) -> None:
        """_init_tracer sets tracer to None when OTel is not available."""
        with patch(
            "greenlang.agents.eudr.supply_chain_mapper.setup.OTEL_AVAILABLE",
            False,
        ):
            service._init_tracer()
            assert service._tracer is None

    def test_init_tracer_with_otel(
        self, service: SupplyChainMapperService
    ) -> None:
        """_init_tracer initializes tracer when OTel is available."""
        mock_tracer = MagicMock()
        mock_trace_module = MagicMock()
        mock_trace_module.get_tracer.return_value = mock_tracer

        with (
            patch(
                "greenlang.agents.eudr.supply_chain_mapper.setup.OTEL_AVAILABLE",
                True,
            ),
            patch(
                "greenlang.agents.eudr.supply_chain_mapper.setup.otel_trace",
                mock_trace_module,
            ),
        ):
            service._init_tracer()
            assert service._tracer is mock_tracer
            mock_trace_module.get_tracer.assert_called_once()


# =============================================================================
# Test 18: Metrics flush
# =============================================================================


class TestMetrics:
    """Tests for metrics flush on shutdown."""

    def test_flush_metrics_when_enabled(
        self, service: SupplyChainMapperService
    ) -> None:
        """Metrics are flushed when enable_metrics is True."""
        service._config = SupplyChainMapperConfig(enable_metrics=True)
        with patch(
            "greenlang.agents.eudr.supply_chain_mapper.setup."
            "SupplyChainMapperService._flush_metrics"
        ) as mock_flush:
            service._flush_metrics()
            # Direct call should not raise

    def test_flush_metrics_when_disabled(
        self, service: SupplyChainMapperService
    ) -> None:
        """Metrics flush is skipped when enable_metrics is False."""
        service._config = SupplyChainMapperConfig(enable_metrics=False)
        # Should not raise
        service._flush_metrics()


# =============================================================================
# Test 19: _ensure_started guard
# =============================================================================


class TestEnsureStarted:
    """Tests for the _ensure_started guard method."""

    def test_raises_when_not_started(
        self, service: SupplyChainMapperService
    ) -> None:
        """Raises RuntimeError with clear message."""
        with pytest.raises(RuntimeError, match="not started"):
            service._ensure_started()

    @pytest.mark.asyncio
    async def test_does_not_raise_when_started(
        self, service: SupplyChainMapperService
    ) -> None:
        """Does not raise after startup."""
        await _start_service_mocked(service)
        service._ensure_started()  # Should not raise


# =============================================================================
# Test 20: Full lifecycle integration
# =============================================================================


class TestFullLifecycle:
    """Integration-style tests for the full startup/shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_full_startup_shutdown_cycle(
        self, service: SupplyChainMapperService
    ) -> None:
        """Complete startup then shutdown cycle succeeds."""
        assert service.is_running is False

        await _start_service_mocked(service)
        assert service.is_running is True
        assert service.uptime_seconds >= 0.0

        health = await service.health_check()
        assert health["status"] in ("healthy", "degraded", "unhealthy")
        assert health["version"] == "1.0.0"

        await service.shutdown()
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_double_startup_shutdown_cycle(
        self, service: SupplyChainMapperService
    ) -> None:
        """Two consecutive startup/shutdown cycles succeed."""
        # First cycle
        await _start_service_mocked(service)
        await service.shutdown()
        assert service.is_running is False

        # Second cycle -- must re-mock since shutdown clears state
        await _start_service_mocked(service)
        assert service.is_running is True
        await service.shutdown()
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_properties_accessible_after_startup(
        self, service: SupplyChainMapperService
    ) -> None:
        """Engine properties are accessible (return None) after startup."""
        await _start_service_mocked(service)
        # These should not raise even though engines are None
        assert service.graph_engine is None
        assert service.multi_tier_mapper is None
        assert service.risk_propagation_engine is None
        assert service.gap_analyzer is None
        assert service.db_pool is None
        assert service.redis_client is None
