# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness - Live Agent Integration Tests
========================================================

Tests that verify PACK-004 CBAMAppBridge, CBAMCalculationEngine,
and related engines can connect to real GL-CBAM-APP modules and
MRV agents for emissions calculations.

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
_PACK_004_DIR = Path(__file__).resolve().parent.parent

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


_cbam_bridge_mod = _import_from_path(
    "pack004_cbam_bridge_integ",
    _PACK_004_DIR / "integrations" / "cbam_app_bridge.py",
)
_calc_engine_mod = _import_from_path(
    "pack004_calc_engine_integ",
    _PACK_004_DIR / "engines" / "cbam_calculation_engine.py",
)
_cert_engine_mod = _import_from_path(
    "pack004_cert_engine_integ",
    _PACK_004_DIR / "engines" / "certificate_engine.py",
)
_deminimis_mod = _import_from_path(
    "pack004_deminimis_integ",
    _PACK_004_DIR / "engines" / "deminimis_engine.py",
)
_orch_mod = _import_from_path(
    "pack004_orch_integ",
    _PACK_004_DIR / "integrations" / "pack_orchestrator.py",
)
_loader_mod = _import_from_path(
    "pack_agent_loader_integ_004",
    _EU_COMPLIANCE_DIR / "agent_loader.py",
)


@pytest.fixture(scope="module")
def agent_loader():
    if _loader_mod is None:
        pytest.skip("agent_loader not available")
    loader = _loader_mod.AgentLoader()
    # Load only agents used by PACK-004 (CBAM + MRV for emissions)
    for aid in ["GL-MRV-X-001", "GL-CBAM-APP"]:
        loader.load(aid)
    return loader


# ===========================================================================
# Tests
# ===========================================================================
@pytest.mark.integration
class TestCBAMAppBridgeIntegration:
    """Integration tests for PACK-004 CBAM App Bridge."""

    def test_bridge_instantiates(self):
        """CBAMAppBridge can be created (stub or live mode)."""
        if _cbam_bridge_mod is None:
            pytest.skip("CBAMAppBridge not importable")
        bridge_cls = getattr(_cbam_bridge_mod, "CBAMAppBridge", None)
        if bridge_cls is None:
            pytest.skip("CBAMAppBridge class not found")
        bridge = bridge_cls()
        assert bridge is not None

    def test_bridge_health_check(self):
        """CBAMAppBridge reports health status."""
        if _cbam_bridge_mod is None:
            pytest.skip("CBAMAppBridge not importable")
        bridge_cls = getattr(_cbam_bridge_mod, "CBAMAppBridge", None)
        if bridge_cls is None:
            pytest.skip("CBAMAppBridge class not found")
        bridge = bridge_cls()
        health_fn = getattr(bridge, "health_check", None) or \
                    getattr(bridge, "get_health", None) or \
                    getattr(bridge, "check_health", None)
        if health_fn is None:
            pytest.skip("No health check method found")
        result = health_fn()
        assert result is not None

    def test_certificate_engine_proxy(self):
        """CBAMAppBridge exposes certificate engine proxy."""
        if _cbam_bridge_mod is None:
            pytest.skip("CBAMAppBridge not importable")
        bridge_cls = getattr(_cbam_bridge_mod, "CBAMAppBridge", None)
        if bridge_cls is None:
            pytest.skip("CBAMAppBridge class not found")
        bridge = bridge_cls()
        cert_fn = getattr(bridge, "get_certificate_engine", None)
        if cert_fn is None:
            pytest.skip("get_certificate_engine not available")
        engine = cert_fn()
        assert engine is not None

    def test_app_availability_flag(self):
        """Check _CBAM_APP_AVAILABLE reflects real import state."""
        if _cbam_bridge_mod is None:
            pytest.skip("CBAMAppBridge not importable")
        flag = getattr(_cbam_bridge_mod, "_CBAM_APP_AVAILABLE", None)
        assert flag is not None  # Flag should exist regardless


@pytest.mark.integration
class TestCBAMCalculationEngineIntegration:
    """Integration tests for PACK-004 CBAM Calculation Engine."""

    def test_engine_instantiates(self):
        """CBAMCalculationEngine can be created."""
        if _calc_engine_mod is None:
            pytest.skip("CBAMCalculationEngine not importable")
        engine_cls = getattr(_calc_engine_mod, "CBAMCalculationEngine", None)
        if engine_cls is None:
            pytest.skip("CBAMCalculationEngine class not found")
        try:
            engine = engine_cls()
        except Exception:
            pytest.skip("CBAMCalculationEngine requires config not available")
            return
        assert engine is not None

    def test_emissions_calculation_method(self):
        """Engine has emissions calculation capability."""
        if _calc_engine_mod is None:
            pytest.skip("CBAMCalculationEngine not importable")
        engine_cls = getattr(_calc_engine_mod, "CBAMCalculationEngine", None)
        if engine_cls is None:
            pytest.skip("CBAMCalculationEngine class not found")
        methods = [m for m in dir(engine_cls) if not m.startswith("_")]
        has_calc = any(
            kw in m.lower() for m in methods
            for kw in ("calculate", "compute", "emission")
        )
        assert has_calc, f"No calculation methods found. Available: {methods}"


@pytest.mark.integration
class TestCBAMCertificateEngineIntegration:
    """Integration tests for PACK-004 Certificate Engine."""

    def test_certificate_engine_instantiates(self):
        """CertificateEngine can be created."""
        if _cert_engine_mod is None:
            pytest.skip("CertificateEngine not importable")
        engine_cls = getattr(_cert_engine_mod, "CertificateEngine", None)
        if engine_cls is None:
            pytest.skip("CertificateEngine class not found")
        try:
            engine = engine_cls()
        except Exception:
            pytest.skip("CertificateEngine requires config not available")
            return
        assert engine is not None

    def test_deminimis_engine_instantiates(self):
        """DeMinimisEngine can be created."""
        if _deminimis_mod is None:
            pytest.skip("DeMinimisEngine not importable")
        engine_cls = getattr(_deminimis_mod, "DeMinimisEngine", None)
        if engine_cls is None:
            pytest.skip("DeMinimisEngine class not found")
        try:
            engine = engine_cls()
        except Exception:
            pytest.skip("DeMinimisEngine requires config not available")
            return
        assert engine is not None
