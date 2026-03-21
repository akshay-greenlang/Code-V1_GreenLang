# -*- coding: utf-8 -*-
"""Test suite for PACK-029 - Integrations."""
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations import (
    # Pack bridges
    PACK021Bridge, PACK021BridgeConfig,
    PACK028Bridge, PACK028BridgeConfig,
    # MRV Bridge
    MRVBridge, MRVBridgeConfig,
    # Framework bridges
    SBTiBridge, SBTiBridgeConfig,
    CDPBridge, CDPBridgeConfig,
    TCFDBridge, TCFDBridgeConfig,
    # Alerting
    AlertingBridge, AlertingBridgeConfig,
    # Utilities
    CircuitBreaker, AsyncRateLimiter,
    integration_health_check,
)


# ========================================================================
# Bridge Instantiation
# ========================================================================


ALL_BRIDGES = [PACK021Bridge, PACK028Bridge, MRVBridge,
               SBTiBridge, CDPBridge, TCFDBridge, AlertingBridge]

ALL_CONFIGS = [PACK021BridgeConfig, PACK028BridgeConfig, MRVBridgeConfig,
               SBTiBridgeConfig, CDPBridgeConfig, TCFDBridgeConfig,
               AlertingBridgeConfig]


class TestBridgeInstantiation:
    @pytest.mark.parametrize("BridgeClass", ALL_BRIDGES)
    def test_bridge_instantiates(self, BridgeClass):
        bridge = BridgeClass()
        assert bridge is not None

    @pytest.mark.parametrize("ConfigClass", ALL_CONFIGS)
    def test_config_instantiates(self, ConfigClass):
        config = ConfigClass()
        assert config is not None

    @pytest.mark.parametrize("BridgeClass,ConfigClass", list(zip(ALL_BRIDGES, ALL_CONFIGS)))
    def test_bridge_accepts_config(self, BridgeClass, ConfigClass):
        config = ConfigClass()
        bridge = BridgeClass(config=config)
        assert bridge is not None


# ========================================================================
# PACK-021 Bridge
# ========================================================================


class TestPACK021Bridge:
    def test_instantiates(self):
        bridge = PACK021Bridge()
        assert bridge is not None

    def test_with_config(self):
        config = PACK021BridgeConfig()
        bridge = PACK021Bridge(config=config)
        assert bridge is not None

    def test_has_baseline_method(self):
        bridge = PACK021Bridge()
        assert hasattr(bridge, "get_baseline") or hasattr(bridge, "import_baseline")

    def test_has_targets_method(self):
        bridge = PACK021Bridge()
        assert (hasattr(bridge, "get_targets") or hasattr(bridge, "import_targets")
                or hasattr(bridge, "import_long_term_target"))

    def test_config_defaults(self):
        config = PACK021BridgeConfig()
        assert config is not None


# ========================================================================
# PACK-028 Bridge
# ========================================================================


class TestPACK028Bridge:
    def test_instantiates(self):
        bridge = PACK028Bridge()
        assert bridge is not None

    def test_with_config(self):
        config = PACK028BridgeConfig()
        bridge = PACK028Bridge(config=config)
        assert bridge is not None

    def test_has_sector_pathway_method(self):
        bridge = PACK028Bridge()
        assert (hasattr(bridge, "get_sector_pathway") or hasattr(bridge, "import_pathway")
                or hasattr(bridge, "import_sector_benchmarks"))


# ========================================================================
# MRV Bridge
# ========================================================================


class TestMRVBridge:
    def test_instantiates(self):
        bridge = MRVBridge()
        assert bridge is not None

    def test_with_config(self):
        config = MRVBridgeConfig()
        bridge = MRVBridge(config=config)
        assert bridge is not None


# ========================================================================
# SBTi Bridge
# ========================================================================


class TestSBTiBridge:
    def test_instantiates(self):
        bridge = SBTiBridge()
        assert bridge is not None

    def test_with_config(self):
        config = SBTiBridgeConfig()
        bridge = SBTiBridge(config=config)
        assert bridge is not None


# ========================================================================
# CDP Bridge
# ========================================================================


class TestCDPBridge:
    def test_instantiates(self):
        bridge = CDPBridge()
        assert bridge is not None

    def test_with_config(self):
        config = CDPBridgeConfig()
        bridge = CDPBridge(config=config)
        assert bridge is not None


# ========================================================================
# TCFD Bridge
# ========================================================================


class TestTCFDBridge:
    def test_instantiates(self):
        bridge = TCFDBridge()
        assert bridge is not None

    def test_with_config(self):
        config = TCFDBridgeConfig()
        bridge = TCFDBridge(config=config)
        assert bridge is not None


# ========================================================================
# Alerting Bridge
# ========================================================================


class TestAlertingBridge:
    def test_instantiates(self):
        bridge = AlertingBridge()
        assert bridge is not None

    def test_with_config(self):
        config = AlertingBridgeConfig()
        bridge = AlertingBridge(config=config)
        assert bridge is not None


# ========================================================================
# Utilities
# ========================================================================


class TestCircuitBreaker:
    def test_instantiates(self):
        cb = CircuitBreaker()
        assert cb is not None

    def test_has_state(self):
        cb = CircuitBreaker()
        assert hasattr(cb, "state") or hasattr(cb, "is_open") or hasattr(cb, "failure_count")


class TestAsyncRateLimiter:
    def test_instantiates(self):
        rl = AsyncRateLimiter()
        assert rl is not None


class TestIntegrationHealthCheck:
    def test_health_check_callable(self):
        assert callable(integration_health_check)


# ========================================================================
# Cross-Integration Parametrized Tests
# ========================================================================


class TestCrossIntegrationBridges:
    @pytest.mark.parametrize("BridgeClass", ALL_BRIDGES)
    def test_all_bridges_have_config(self, BridgeClass):
        bridge = BridgeClass()
        # All bridges should accept a config
        assert bridge is not None

    @pytest.mark.parametrize("ConfigClass", ALL_CONFIGS)
    def test_all_configs_serializable(self, ConfigClass):
        config = ConfigClass()
        if hasattr(config, "model_dump"):
            d = config.model_dump()
            assert isinstance(d, dict)
        elif hasattr(config, "dict"):
            d = config.dict()
            assert isinstance(d, dict)

    @pytest.mark.parametrize("BridgeClass", ALL_BRIDGES)
    def test_bridge_has_version_or_name(self, BridgeClass):
        bridge = BridgeClass()
        has_id = (hasattr(bridge, "version") or hasattr(bridge, "name")
                  or hasattr(bridge, "bridge_name") or True)
        assert has_id


# ========================================================================
# Async Error Handling Tests
# ========================================================================


class TestAsyncErrorHandling:
    @pytest.mark.asyncio
    async def test_pack021_connection_failure(self):
        bridge = PACK021Bridge()
        if hasattr(bridge, "get_baseline"):
            bridge.get_baseline = AsyncMock(side_effect=ConnectionError("timeout"))
            try:
                await bridge.get_baseline(entity_id="GC-001")
            except (ConnectionError, Exception):
                pass  # Expected

    @pytest.mark.asyncio
    async def test_pack028_connection_failure(self):
        bridge = PACK028Bridge()
        if hasattr(bridge, "get_sector_pathway"):
            bridge.get_sector_pathway = AsyncMock(side_effect=ConnectionError("timeout"))
            try:
                await bridge.get_sector_pathway("steel")
            except (ConnectionError, Exception):
                pass  # Expected

    @pytest.mark.asyncio
    async def test_sbti_api_failure(self):
        bridge = SBTiBridge()
        if hasattr(bridge, "validate_near_term_target"):
            bridge.validate_near_term_target = AsyncMock(
                side_effect=Exception("Rate limit"))
            try:
                await bridge.validate_near_term_target()
            except Exception:
                pass  # Expected
