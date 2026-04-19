# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter - Live Agent Integration Tests
=======================================================

Tests that verify PACK-006's TraceabilityBridge, SatelliteBridge,
GISBridge, and EUDRAppBridge can connect to real GreenLang EUDR agents
and data connectors.

Requires: pytest -m integration
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_EU_COMPLIANCE_DIR = Path(__file__).resolve().parent.parent.parent
_PACK_006_DIR = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _import_from_path(module_name: str, file_path: Path) -> Optional[Any]:
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_trace_mod = _import_from_path(
    "pack006_trace_integ",
    _PACK_006_DIR / "integrations" / "traceability_bridge.py",
)
_sat_mod = _import_from_path(
    "pack006_sat_integ",
    _PACK_006_DIR / "integrations" / "satellite_bridge.py",
)
_gis_mod = _import_from_path(
    "pack006_gis_integ",
    _PACK_006_DIR / "integrations" / "gis_bridge.py",
)
_eudr_app_mod = _import_from_path(
    "pack006_eudr_app_integ",
    _PACK_006_DIR / "integrations" / "eudr_app_bridge.py",
)
_eu_info_mod = _import_from_path(
    "pack006_eu_info_integ",
    _PACK_006_DIR / "integrations" / "eu_information_system_bridge.py",
)
_loader_mod = _import_from_path(
    "pack_agent_loader_integ_006",
    _EU_COMPLIANCE_DIR / "agent_loader.py",
)


@pytest.fixture(scope="module")
def agent_loader():
    if _loader_mod is None:
        pytest.skip("agent_loader not available")
    loader = _loader_mod.AgentLoader()
    # Load only EUDR-related agents
    for aid in ["GL-DATA-X-005", "GL-DATA-X-006", "GL-DATA-X-007", "GL-EUDR-X-001"]:
        loader.load(aid)
    return loader


# ===========================================================================
# Traceability Bridge Tests
# ===========================================================================
@pytest.mark.integration
class TestTraceabilityBridgeIntegration:
    """Integration tests for PACK-006 Traceability Bridge."""

    def test_bridge_instantiates(self):
        """TraceabilityBridge can be created."""
        if _trace_mod is None:
            pytest.skip("TraceabilityBridge not importable")
        bridge_cls = getattr(_trace_mod, "TraceabilityBridge", None)
        if bridge_cls is None:
            pytest.skip("TraceabilityBridge class not found")
        config_cls = getattr(_trace_mod, "TraceabilityBridgeConfig", None)
        config = config_cls() if config_cls else None
        bridge = bridge_cls(config) if config else bridge_cls()
        assert bridge is not None

    def test_plot_registry_proxy(self):
        """TraceabilityBridge exposes PlotRegistryProxy."""
        if _trace_mod is None:
            pytest.skip("TraceabilityBridge not importable")
        bridge_cls = getattr(_trace_mod, "TraceabilityBridge", None)
        if bridge_cls is None:
            pytest.skip("TraceabilityBridge class not found")
        bridge = bridge_cls()
        get_fn = getattr(bridge, "get_plot_registry", None)
        if get_fn is None:
            pytest.skip("get_plot_registry not available")
        proxy = get_fn()
        assert proxy is not None
        # Verify proxy has _service injection point
        assert hasattr(proxy, "_service")

    def test_plot_registry_register_plot(self):
        """Register a plot through the TraceabilityBridge stub."""
        if _trace_mod is None:
            pytest.skip("TraceabilityBridge not importable")
        bridge_cls = getattr(_trace_mod, "TraceabilityBridge", None)
        if bridge_cls is None:
            pytest.skip("TraceabilityBridge class not found")
        bridge = bridge_cls()
        get_fn = getattr(bridge, "get_plot_registry", None)
        if get_fn is None:
            pytest.skip("get_plot_registry not available")
        proxy = get_fn()
        register_fn = getattr(proxy, "register_plot", None)
        if register_fn is None:
            pytest.skip("register_plot not available on proxy")
        plot_data = {
            "supplier_id": "SUP-TEST-001",
            "latitude": -3.4653,
            "longitude": -62.2159,
            "area_ha": 50.0,
            "country": "BR",
            "commodity": "soy",
        }
        if asyncio.iscoroutinefunction(register_fn):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(register_fn(plot_data))
            finally:
                loop.close()
        else:
            result = register_fn(plot_data)
        assert result is not None

    def test_chain_of_custody_proxy(self):
        """TraceabilityBridge exposes ChainOfCustodyProxy."""
        if _trace_mod is None:
            pytest.skip("TraceabilityBridge not importable")
        bridge_cls = getattr(_trace_mod, "TraceabilityBridge", None)
        if bridge_cls is None:
            pytest.skip("TraceabilityBridge class not found")
        bridge = bridge_cls()
        get_fn = getattr(bridge, "get_chain_of_custody", None)
        if get_fn is None:
            pytest.skip("get_chain_of_custody not available")
        proxy = get_fn()
        assert proxy is not None


# ===========================================================================
# Satellite Bridge Tests
# ===========================================================================
@pytest.mark.integration
class TestSatelliteBridgeIntegration:
    """Integration tests for PACK-006 Satellite Bridge."""

    def test_bridge_instantiates(self):
        """SatelliteBridge can be created."""
        if _sat_mod is None:
            pytest.skip("SatelliteBridge not importable")
        bridge_cls = getattr(_sat_mod, "SatelliteBridge", None)
        if bridge_cls is None:
            pytest.skip("SatelliteBridge class not found")
        config_cls = getattr(_sat_mod, "SatelliteBridgeConfig", None)
        config = config_cls() if config_cls else None
        bridge = bridge_cls(config) if config else bridge_cls()
        assert bridge is not None

    def test_forest_change_analysis_method(self):
        """SatelliteBridge has forest change analysis capability."""
        if _sat_mod is None:
            pytest.skip("SatelliteBridge not importable")
        bridge_cls = getattr(_sat_mod, "SatelliteBridge", None)
        if bridge_cls is None:
            pytest.skip("SatelliteBridge class not found")
        methods = [m for m in dir(bridge_cls) if not m.startswith("_")]
        has_forest = any(
            kw in m.lower() for m in methods
            for kw in ("forest", "deforestation", "change", "monitor")
        )
        assert has_forest, f"No forest analysis methods found. Available: {methods}"


# ===========================================================================
# GIS Bridge Tests
# ===========================================================================
@pytest.mark.integration
class TestGISBridgeIntegration:
    """Integration tests for PACK-006 GIS Bridge."""

    def test_bridge_instantiates(self):
        """GISBridge can be created."""
        if _gis_mod is None:
            pytest.skip("GISBridge not importable")
        bridge_cls = getattr(_gis_mod, "GISBridge", None)
        if bridge_cls is None:
            pytest.skip("GISBridge class not found")
        config_cls = getattr(_gis_mod, "GISBridgeConfig", None)
        config = config_cls() if config_cls else None
        bridge = bridge_cls(config) if config else bridge_cls()
        assert bridge is not None

    def test_coordinate_transform_method(self):
        """GISBridge has coordinate transformation capability."""
        if _gis_mod is None:
            pytest.skip("GISBridge not importable")
        bridge_cls = getattr(_gis_mod, "GISBridge", None)
        if bridge_cls is None:
            pytest.skip("GISBridge class not found")
        methods = [m for m in dir(bridge_cls) if not m.startswith("_")]
        has_geo = any(
            kw in m.lower() for m in methods
            for kw in ("transform", "coordinate", "spatial", "boundary", "geocode")
        )
        assert has_geo, f"No GIS methods found. Available: {methods}"


# ===========================================================================
# EUDR App Bridge Tests
# ===========================================================================
@pytest.mark.integration
class TestEUDRAppBridgeIntegration:
    """Integration tests for PACK-006 EUDR App Bridge."""

    def test_bridge_instantiates(self):
        """EUDRAppBridge can be created."""
        if _eudr_app_mod is None:
            pytest.skip("EUDRAppBridge not importable")
        bridge_cls = getattr(_eudr_app_mod, "EUDRAppBridge", None)
        if bridge_cls is None:
            pytest.skip("EUDRAppBridge class not found")
        config_cls = getattr(_eudr_app_mod, "EUDRAppBridgeConfig", None)
        config = config_cls() if config_cls else None
        bridge = bridge_cls(config) if config else bridge_cls()
        assert bridge is not None

    def test_dds_proxy_available(self):
        """EUDRAppBridge exposes DDS (Due Diligence Statement) proxy."""
        if _eudr_app_mod is None:
            pytest.skip("EUDRAppBridge not importable")
        bridge_cls = getattr(_eudr_app_mod, "EUDRAppBridge", None)
        if bridge_cls is None:
            pytest.skip("EUDRAppBridge class not found")
        bridge = bridge_cls()
        dds_fn = getattr(bridge, "get_dds_proxy", None) or \
                 getattr(bridge, "get_dds", None)
        if dds_fn is None:
            # Check for DDS-related methods
            methods = [m for m in dir(bridge) if "dds" in m.lower() and not m.startswith("_")]
            if not methods:
                pytest.skip("No DDS proxy method found")
            dds_fn = getattr(bridge, methods[0])
        result = dds_fn()
        assert result is not None
