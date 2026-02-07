# -*- coding: utf-8 -*-
"""
Unit tests for configure_tracing setup module (OBS-003)

Tests the one-liner tracing setup function: initialization, idempotency,
disabled mode, shutdown lifecycle, and FastAPI attachment.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from greenlang.infrastructure.tracing_service.config import TracingConfig


# ============================================================================
# Helper: reset module state between tests
# ============================================================================


@pytest.fixture(autouse=True)
def reset_setup_state():
    """Reset the setup module's global state between tests."""
    yield
    try:
        from greenlang.infrastructure.tracing_service import setup

        setup._initialized = False
        setup._active_config = None
    except Exception:
        pass


# ============================================================================
# configure_tracing tests
# ============================================================================


class TestConfigureTracing:
    """Tests for the configure_tracing one-liner setup function."""

    def test_returns_tracing_config(self):
        """configure_tracing returns a TracingConfig instance."""
        from greenlang.infrastructure.tracing_service.setup import configure_tracing

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ):
            config = configure_tracing(service_name="test-svc")

        assert isinstance(config, TracingConfig)
        assert config.service_name == "test-svc"

    def test_idempotent_second_call(self):
        """Second call returns the existing config without re-initialising."""
        from greenlang.infrastructure.tracing_service.setup import configure_tracing

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ) as mock_provider, patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ):
            c1 = configure_tracing(service_name="svc")
            c2 = configure_tracing(service_name="other")

        assert c1 is c2
        # setup_provider should only be called once
        mock_provider.assert_called_once()

    def test_disabled_config_skips_setup(self):
        """When enabled=False, provider and instrumentors are not called."""
        from greenlang.infrastructure.tracing_service.setup import configure_tracing

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ) as mock_provider, patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors"
        ) as mock_instr:
            config = configure_tracing(
                config=TracingConfig(enabled=False, service_name="disabled")
            )

        assert config.enabled is False
        mock_provider.assert_not_called()
        mock_instr.assert_not_called()

    def test_custom_config_passed_through(self):
        """A pre-built TracingConfig is used when provided."""
        from greenlang.infrastructure.tracing_service.setup import configure_tracing

        custom = TracingConfig(
            service_name="custom-agent",
            sampling_rate=0.5,
            environment="staging",
        )

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ):
            result = configure_tracing(config=custom)

        assert result.service_name == "custom-agent"
        assert result.sampling_rate == pytest.approx(0.5)
        assert result.environment == "staging"

    def test_service_name_override_on_default_config(self):
        """service_name param overrides default config service_name."""
        from greenlang.infrastructure.tracing_service.setup import configure_tracing

        custom = TracingConfig()  # service_name defaults to "greenlang"

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ):
            result = configure_tracing(
                config=custom, service_name="override-svc"
            )

        assert result.service_name == "override-svc"

    def test_app_middleware_attached(self):
        """TracingMiddleware is attached when a FastAPI app is provided."""
        from greenlang.infrastructure.tracing_service.setup import configure_tracing

        mock_app = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ):
            configure_tracing(mock_app, service_name="api-svc")

        mock_app.add_middleware.assert_called_once()


# ============================================================================
# shutdown_tracing tests
# ============================================================================


class TestShutdownTracing:
    """Tests for the shutdown_tracing function."""

    def test_shutdown_resets_state(self):
        """shutdown_tracing resets _initialized and _active_config."""
        from greenlang.infrastructure.tracing_service.setup import (
            configure_tracing,
            shutdown_tracing,
            is_tracing_enabled,
            get_active_config,
        )

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ):
            configure_tracing(service_name="svc")

        with patch(
            "greenlang.infrastructure.tracing_service.setup.shutdown_provider"
        ):
            shutdown_tracing()

        assert get_active_config() is None

    def test_shutdown_calls_provider_shutdown(self):
        """shutdown_tracing delegates to provider.shutdown."""
        from greenlang.infrastructure.tracing_service.setup import shutdown_tracing

        with patch(
            "greenlang.infrastructure.tracing_service.setup.shutdown_provider"
        ) as mock_shutdown:
            shutdown_tracing()

        mock_shutdown.assert_called_once()


# ============================================================================
# is_tracing_enabled tests
# ============================================================================


class TestIsTracingEnabled:
    """Tests for the is_tracing_enabled function."""

    def test_false_before_init(self):
        """Returns False before configure_tracing is called."""
        from greenlang.infrastructure.tracing_service.setup import is_tracing_enabled

        assert is_tracing_enabled() is False

    def test_true_after_init_with_otel(self):
        """Returns True after configure_tracing when OTel is available."""
        from greenlang.infrastructure.tracing_service.setup import (
            configure_tracing,
            is_tracing_enabled,
        )

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.OTEL_AVAILABLE",
            True,
        ):
            configure_tracing(service_name="svc")
            assert is_tracing_enabled() is True


# ============================================================================
# get_active_config tests
# ============================================================================


class TestGetActiveConfig:
    """Tests for the get_active_config function."""

    def test_none_before_init(self):
        """Returns None before configure_tracing is called."""
        from greenlang.infrastructure.tracing_service.setup import get_active_config

        assert get_active_config() is None

    def test_returns_config_after_init(self):
        """Returns the active config after configure_tracing."""
        from greenlang.infrastructure.tracing_service.setup import (
            configure_tracing,
            get_active_config,
        )

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ):
            configure_tracing(service_name="svc")

        config = get_active_config()
        assert config is not None
        assert config.service_name == "svc"
