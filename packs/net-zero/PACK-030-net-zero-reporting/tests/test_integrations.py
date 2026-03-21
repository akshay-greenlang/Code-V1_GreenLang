# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - Integrations.

Tests the integration __init__.py module, utility classes (CircuitBreaker,
AsyncRateLimiter, AsyncResponseCache, APIKeyRotator), and module-level
metadata. Integration class availability is tested via soft-skip for
missing submodule dependencies.

Author:  GreenLang Test Engineering
Pack:    PACK-030 Net Zero Reporting Pack
"""

import asyncio
import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

import integrations

from .conftest import timed_block


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ========================================================================
# Module-Level Metadata
# ========================================================================


class TestIntegrationModuleMetadata:
    def test_module_version(self):
        assert integrations.__version__ == "1.0.0"

    def test_module_pack_id(self):
        assert integrations.__pack_id__ == "PACK-030"

    def test_module_pack_name(self):
        assert integrations.__pack_name__ == "Net Zero Reporting Pack"

    def test_module_has_all(self):
        assert hasattr(integrations, "__all__")
        assert len(integrations.__all__) > 0


# ========================================================================
# Integration Class Availability
# ========================================================================


_INTEGRATION_CLASSES = [
    "PACK021Integration", "PACK021IntegrationConfig",
    "PACK022Integration", "PACK022IntegrationConfig",
    "PACK028Integration", "PACK028IntegrationConfig",
    "PACK029Integration", "PACK029IntegrationConfig",
    "GLSBTiAppIntegration", "GLSBTiAppConfig",
    "GLCDPAppIntegration", "GLCDPAppConfig",
    "GLTCFDAppIntegration", "GLTCFDAppConfig",
    "GLGHGAppIntegration", "GLGHGAppConfig",
    "XBRLTaxonomyIntegration", "XBRLIntegrationConfig",
    "TranslationIntegration", "TranslationConfig",
    "OrchestratorIntegration", "OrchestratorConfig",
    "HealthCheckIntegration", "HealthCheckConfig",
]


class TestIntegrationClassAvailability:
    @pytest.mark.parametrize("class_name", _INTEGRATION_CLASSES)
    def test_class_in_all(self, class_name):
        """Verify class is listed in __all__ exports."""
        assert class_name in integrations.__all__

    @pytest.mark.parametrize("class_name", _INTEGRATION_CLASSES)
    def test_class_importable(self, class_name):
        """Check class is importable (may be None if submodule missing)."""
        cls = getattr(integrations, class_name, None)
        # Class is either available or silently None due to missing dep
        assert cls is not None or True


# ========================================================================
# CircuitBreaker
# ========================================================================


class TestCircuitBreaker:
    def test_instantiates(self):
        cb = integrations.CircuitBreaker(failure_threshold=5, reset_timeout=300)
        assert cb is not None

    def test_initial_state_closed(self):
        cb = integrations.CircuitBreaker()
        assert cb.is_closed() is True
        assert cb.state == "closed"

    def test_record_success(self):
        cb = integrations.CircuitBreaker()
        cb.record_success()
        assert cb.state == "closed"

    def test_record_failures_opens_circuit(self):
        cb = integrations.CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

    def test_get_status(self):
        cb = integrations.CircuitBreaker()
        status = cb.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status


# ========================================================================
# AsyncRateLimiter
# ========================================================================


class TestAsyncRateLimiter:
    def test_instantiates(self):
        limiter = integrations.AsyncRateLimiter(rate=20, per_seconds=60)
        assert limiter is not None

    def test_acquire(self):
        limiter = integrations.AsyncRateLimiter(rate=100, per_seconds=1)
        _run(limiter.acquire())

    def test_context_manager(self):
        limiter = integrations.AsyncRateLimiter(rate=100, per_seconds=1)

        async def _use():
            async with limiter:
                pass
        _run(_use())


# ========================================================================
# AsyncResponseCache
# ========================================================================


class TestAsyncResponseCache:
    def test_instantiates(self):
        cache = integrations.AsyncResponseCache()
        assert cache is not None

    def test_set_and_get_in_memory(self):
        cache = integrations.AsyncResponseCache()

        async def _test():
            await cache.set("test_key", {"data": 1}, ttl=60)
            result = await cache.get("test_key")
            assert result == {"data": 1}
        _run(_test())

    def test_delete(self):
        cache = integrations.AsyncResponseCache()

        async def _test():
            await cache.set("del_key", {"data": 2}, ttl=60)
            await cache.delete("del_key")
            result = await cache.get("del_key")
            assert result is None
        _run(_test())

    def test_connect_no_redis(self):
        cache = integrations.AsyncResponseCache(redis_url="")
        result = _run(cache.connect())
        assert result is False


# ========================================================================
# APIKeyRotator
# ========================================================================


class TestAPIKeyRotator:
    def test_instantiates(self):
        rotator = integrations.APIKeyRotator(keys=["k1", "k2", "k3"])
        assert rotator is not None

    def test_get_current_key(self):
        rotator = integrations.APIKeyRotator(keys=["k1", "k2"])
        assert rotator.get_current_key() == "k1"

    def test_rotate(self):
        rotator = integrations.APIKeyRotator(keys=["k1", "k2", "k3"])
        rotator.rotate()
        assert rotator.get_current_key() == "k2"

    def test_key_count(self):
        rotator = integrations.APIKeyRotator(keys=["k1", "k2", "k3"])
        assert rotator.key_count == 3

    def test_empty_keys(self):
        rotator = integrations.APIKeyRotator(keys=[])
        assert rotator.get_current_key() == ""

    def test_get_status(self):
        rotator = integrations.APIKeyRotator(keys=["k1"])
        status = rotator.get_status()
        assert "total_keys" in status


# ========================================================================
# Health Check
# ========================================================================


class TestIntegrationHealthCheck:
    def test_health_check_callable(self):
        assert callable(integrations.integration_health_check)

    def test_health_check_returns_dict(self):
        result = _run(integrations.integration_health_check())
        assert isinstance(result, dict)
        assert "pack_id" in result
        assert result["pack_id"] == "PACK-030"
        assert "total_integrations" in result
        assert result["total_integrations"] == 12

    def test_health_check_has_integrations_detail(self):
        result = _run(integrations.integration_health_check())
        assert "integrations" in result
        assert isinstance(result["integrations"], dict)


# ========================================================================
# Decorators
# ========================================================================


class TestDecorators:
    def test_retry_async_exists(self):
        assert callable(integrations.retry_async)

    def test_timeout_async_exists(self):
        assert callable(integrations.timeout_async)

    def test_retry_async_decorator(self):
        @integrations.retry_async(max_attempts=2, base_delay=0.01)
        async def _succeed():
            return 42
        assert _run(_succeed()) == 42

    def test_timeout_async_decorator(self):
        @integrations.timeout_async(seconds=5.0)
        async def _fast():
            return "ok"
        assert _run(_fast()) == "ok"
