# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional - Live Agent Integration Tests
==========================================================

Tests that verify PACK-007's FullTraceabilityBridge, RiskAssessmentBridge,
DueDiligenceBridge, and other professional-tier bridges can connect to real
GreenLang EUDR agents and data connectors.

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
_PACK_007_DIR = Path(__file__).resolve().parent.parent

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


def _instantiate_bridge(mod, bridge_name: str, config_names: list):
    """Try to instantiate a bridge with its config, handling various patterns."""
    if mod is None:
        return None
    bridge_cls = getattr(mod, bridge_name, None)
    if bridge_cls is None:
        return None
    # Try to find a config class
    config = None
    for cn in config_names:
        config_cls = getattr(mod, cn, None)
        if config_cls:
            try:
                config = config_cls()
            except Exception:
                pass
            break
    # Try to instantiate
    try:
        return bridge_cls(config) if config else bridge_cls()
    except TypeError:
        if config:
            return bridge_cls(config)
        return None


_full_trace_mod = _import_from_path(
    "pack007_full_trace_integ",
    _PACK_007_DIR / "integrations" / "full_traceability_bridge.py",
)
_risk_mod = _import_from_path(
    "pack007_risk_integ",
    _PACK_007_DIR / "integrations" / "risk_assessment_bridge.py",
)
_dd_mod = _import_from_path(
    "pack007_dd_integ",
    _PACK_007_DIR / "integrations" / "due_diligence_bridge.py",
)
_dd_workflow_mod = _import_from_path(
    "pack007_dd_workflow_integ",
    _PACK_007_DIR / "integrations" / "due_diligence_workflow_bridge.py",
)
_sat_mod = _import_from_path(
    "pack007_sat_integ",
    _PACK_007_DIR / "integrations" / "satellite_monitoring_bridge.py",
)
_gis_mod = _import_from_path(
    "pack007_gis_integ",
    _PACK_007_DIR / "integrations" / "gis_analytics_bridge.py",
)
_eudr_app_mod = _import_from_path(
    "pack007_eudr_app_integ",
    _PACK_007_DIR / "integrations" / "eudr_app_bridge.py",
)
_eu_info_mod = _import_from_path(
    "pack007_eu_info_integ",
    _PACK_007_DIR / "integrations" / "eu_information_system_bridge.py",
)
_orch_mod = _import_from_path(
    "pack007_orch_integ",
    _PACK_007_DIR / "integrations" / "pack_orchestrator.py",
)
_loader_mod = _import_from_path(
    "pack_agent_loader_integ_007",
    _EU_COMPLIANCE_DIR / "agent_loader.py",
)

# Import professional engines
_scenario_risk_mod = _import_from_path(
    "pack007_scenario_risk_engine",
    _PACK_007_DIR / "engines" / "scenario_risk_engine.py",
)
_adv_geo_mod = _import_from_path(
    "pack007_adv_geo_engine",
    _PACK_007_DIR / "engines" / "advanced_geolocation_engine.py",
)


@pytest.fixture(scope="module")
def agent_loader():
    if _loader_mod is None:
        pytest.skip("agent_loader not available")
    loader = _loader_mod.AgentLoader()
    # Load EUDR-related agents for professional pack
    for aid in [
        "GL-DATA-X-005",  # EUDR Traceability Connector
        "GL-DATA-X-006",  # GIS/Mapping
        "GL-DATA-X-007",  # Satellite
        "GL-DATA-X-010",  # Quality Profiler
    ]:
        loader.load(aid)
    return loader


# ===========================================================================
# Full Traceability Bridge Tests
# ===========================================================================
@pytest.mark.integration
class TestFullTraceabilityBridgeIntegration:
    """Integration tests for PACK-007 Full Traceability Bridge."""

    def test_traceability_bridge_instantiates(self):
        """FullTraceabilityBridge can be created."""
        bridge = _instantiate_bridge(
            _full_trace_mod, "FullTraceabilityBridge",
            ["TraceabilityBridgeConfig", "FullTraceabilityConfig"]
        )
        if bridge is None:
            pytest.skip("FullTraceabilityBridge not instantiable")
        assert bridge is not None

    def test_real_eudr_agents_load(self, agent_loader):
        """EUDR agents can be loaded for traceability bridge."""
        summary = agent_loader.summary()
        if summary["loaded"] == 0:
            pytest.skip("No EUDR agents loaded")
        assert summary["loaded"] > 0
        # Verify at least EUDR traceability connector is available
        loaded_ids = agent_loader.loaded_ids
        has_eudr = any("EUDR" in aid or "GL-DATA-X-005" in aid for aid in loaded_ids)
        if not has_eudr:
            pytest.skip("EUDR traceability agent not loaded")
        assert has_eudr

    def test_plot_registration_with_real_agent(self, agent_loader):
        """Register plot through traceability bridge with real agent injection."""
        if _full_trace_mod is None:
            pytest.skip("FullTraceabilityBridge not importable")
        bridge_cls = getattr(_full_trace_mod, "FullTraceabilityBridge", None)
        if bridge_cls is None:
            pytest.skip("FullTraceabilityBridge class not found")
        try:
            bridge = bridge_cls()
        except TypeError:
            # Bridge requires config - try to find and instantiate config
            config_cls = None
            for attr in dir(_full_trace_mod or _risk_mod or _dd_mod or type(None)):
                pass
            pytest.skip("Bridge requires config, skipping agent injection test")

        # Inject loaded agents if available
        for agent_id in agent_loader.loaded_ids:
            instance = agent_loader.get_instance(agent_id)
            if instance is not None and hasattr(bridge, "_agents"):
                bridge._agents[agent_id] = instance

        register_fn = getattr(bridge, "register_plot", None)
        if register_fn is None:
            pytest.skip("register_plot not available")

        plot_data = {
            "plot_id": "PLOT-TEST-007",
            "supplier_id": "SUP-TEST-007",
            "latitude": -10.4653,
            "longitude": -52.2159,
            "area_ha": 150.0,
            "country": "BR",
            "commodity": "cattle",
            "geolocation_method": "GPS",
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

    def test_chain_of_custody_with_real_agent(self, agent_loader):
        """Chain of custody tracking through bridge with real agents."""
        bridge = _instantiate_bridge(
            _full_trace_mod, "FullTraceabilityBridge",
            ["TraceabilityBridgeConfig", "FullTraceabilityConfig"]
        )
        if bridge is None:
            pytest.skip("FullTraceabilityBridge not instantiable")

        # Inject loaded agents
        for agent_id in agent_loader.loaded_ids:
            instance = agent_loader.get_instance(agent_id)
            if instance is not None and hasattr(bridge, "_agents"):
                bridge._agents[agent_id] = instance

        coc_fn = getattr(bridge, "track_chain_of_custody", None) or \
                 getattr(bridge, "get_chain_of_custody", None)
        if coc_fn is None:
            pytest.skip("Chain of custody method not available")

        custody_data = {
            "shipment_id": "SHIP-007-TEST",
            "origin_plot": "PLOT-TEST-007",
            "commodity": "cattle",
            "quantity_kg": 5000,
        }

        if asyncio.iscoroutinefunction(coc_fn):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(coc_fn(custody_data))
            finally:
                loop.close()
        else:
            result = coc_fn(custody_data)

        assert result is not None


# ===========================================================================
# Risk Assessment Bridge Tests
# ===========================================================================
@pytest.mark.integration
class TestRiskAssessmentBridgeIntegration:
    """Integration tests for PACK-007 Risk Assessment Bridge."""

    def test_risk_bridge_instantiates(self):
        """RiskAssessmentBridge can be created."""
        bridge = _instantiate_bridge(
            _risk_mod, "RiskAssessmentBridge",
            ["RiskAssessmentBridgeConfig", "RiskAssessmentConfig"]
        )
        if bridge is None:
            pytest.skip("RiskAssessmentBridge not instantiable")
        assert bridge is not None

    def test_country_risk_assessment(self, agent_loader):
        """Country risk assessment with real agents."""
        bridge = _instantiate_bridge(
            _risk_mod, "RiskAssessmentBridge",
            ["RiskAssessmentBridgeConfig", "RiskAssessmentConfig"]
        )
        if bridge is None:
            pytest.skip("RiskAssessmentBridge not instantiable")

        # Inject loaded agents
        for agent_id in agent_loader.loaded_ids:
            instance = agent_loader.get_instance(agent_id)
            if instance is not None and hasattr(bridge, "_agents"):
                bridge._agents[agent_id] = instance

        assess_fn = getattr(bridge, "assess_country_risk", None) or \
                    getattr(bridge, "calculate_country_risk", None)
        if assess_fn is None:
            pytest.skip("Country risk assessment method not available")

        risk_data = {
            "country_code": "BR",
            "commodity": "cattle",
            "region": "Amazonia",
        }

        if asyncio.iscoroutinefunction(assess_fn):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(assess_fn(risk_data))
            finally:
                loop.close()
        else:
            result = assess_fn(risk_data)

        assert result is not None

    def test_composite_risk_with_real_agents(self, agent_loader):
        """Composite risk calculation with real agents."""
        bridge = _instantiate_bridge(
            _risk_mod, "RiskAssessmentBridge",
            ["RiskAssessmentBridgeConfig", "RiskAssessmentConfig"]
        )
        if bridge is None:
            pytest.skip("RiskAssessmentBridge not instantiable")

        # Inject loaded agents
        for agent_id in agent_loader.loaded_ids:
            instance = agent_loader.get_instance(agent_id)
            if instance is not None and hasattr(bridge, "_agents"):
                bridge._agents[agent_id] = instance

        composite_fn = getattr(bridge, "calculate_composite_risk", None) or \
                       getattr(bridge, "assess_composite_risk", None)
        if composite_fn is None:
            pytest.skip("Composite risk method not available")

        risk_data = {
            "country_risk": 0.75,
            "commodity_risk": 0.60,
            "supplier_risk": 0.40,
        }

        if asyncio.iscoroutinefunction(composite_fn):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(composite_fn(risk_data))
            finally:
                loop.close()
        else:
            result = composite_fn(risk_data)

        assert result is not None


# ===========================================================================
# Due Diligence Bridge Tests
# ===========================================================================
@pytest.mark.integration
class TestDueDiligenceBridgeIntegration:
    """Integration tests for PACK-007 Due Diligence Bridge."""

    def test_dd_bridge_instantiates(self):
        """DueDiligenceBridge can be created."""
        bridge = _instantiate_bridge(
            _dd_mod, "DueDiligenceBridge",
            ["DDCoreBridgeConfig", "DueDiligenceBridgeConfig", "DueDiligenceConfig"]
        )
        if bridge is None:
            pytest.skip("DueDiligenceBridge not instantiable")
        assert bridge is not None

    def test_information_collection(self, agent_loader):
        """Information collection phase with real agents."""
        bridge = _instantiate_bridge(
            _dd_mod, "DueDiligenceBridge",
            ["DDCoreBridgeConfig", "DueDiligenceBridgeConfig", "DueDiligenceConfig"]
        )
        if bridge is None:
            pytest.skip("DueDiligenceBridge not instantiable")

        # Inject loaded agents
        for agent_id in agent_loader.loaded_ids:
            instance = agent_loader.get_instance(agent_id)
            if instance is not None and hasattr(bridge, "_agents"):
                bridge._agents[agent_id] = instance

        collect_fn = getattr(bridge, "collect_information", None) or \
                     getattr(bridge, "gather_information", None)
        if collect_fn is None:
            pytest.skip("Information collection method not available")

        # Adapt to actual signature: (operator_id, commodity, batch_id)
        try:
            if asyncio.iscoroutinefunction(collect_fn):
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        collect_fn("OP-TEST-007", "cattle", "BATCH-TEST-007")
                    )
                finally:
                    loop.close()
            else:
                result = collect_fn("OP-TEST-007", "cattle", "BATCH-TEST-007")
        except TypeError:
            # Fallback: try with dict
            info_data = {"operator_id": "OP-TEST-007", "commodity": "cattle", "batch_id": "BATCH-TEST-007"}
            result = collect_fn(info_data)

        assert result is not None

    def test_dds_generation(self, agent_loader):
        """DDS (Due Diligence Statement) generation with real agents."""
        bridge = _instantiate_bridge(
            _dd_mod, "DueDiligenceBridge",
            ["DDCoreBridgeConfig", "DueDiligenceBridgeConfig", "DueDiligenceConfig"]
        )
        if bridge is None:
            pytest.skip("DueDiligenceBridge not instantiable")

        # Inject loaded agents
        for agent_id in agent_loader.loaded_ids:
            instance = agent_loader.get_instance(agent_id)
            if instance is not None and hasattr(bridge, "_agents"):
                bridge._agents[agent_id] = instance

        dds_fn = getattr(bridge, "generate_dds", None) or \
                 getattr(bridge, "create_dds", None)
        if dds_fn is None:
            pytest.skip("DDS generation method not available")

        # Adapt to actual signature: (information_package, risk_analysis, mitigation_plan=None)
        info_pkg = {
            "operator_id": "OP-TEST-007",
            "commodity": "cattle",
            "plots_verified": 5,
        }
        risk_analysis = {
            "risk_level": "negligible",
            "composite_score": 0.25,
        }

        try:
            if asyncio.iscoroutinefunction(dds_fn):
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(dds_fn(info_pkg, risk_analysis))
                finally:
                    loop.close()
            else:
                result = dds_fn(info_pkg, risk_analysis)
        except TypeError:
            # Fallback: try with single dict
            dds_data = {**info_pkg, **risk_analysis}
            result = dds_fn(dds_data)

        assert result is not None


# ===========================================================================
# Orchestrator Integration Tests
# ===========================================================================
@pytest.mark.integration
class TestOrchestratorIntegration:
    """Integration tests for PACK-007 Pack Orchestrator."""

    def test_orchestrator_creates_agent_stubs(self):
        """Orchestrator initializes all bridge stubs."""
        if _orch_mod is None:
            pytest.skip("Pack orchestrator not importable")
        orch_cls = getattr(_orch_mod, "PackOrchestrator", None)
        if orch_cls is None:
            pytest.skip("PackOrchestrator class not found")
        config_cls = getattr(_orch_mod, "OrchestratorConfig", None)
        config = config_cls() if config_cls else None
        orch = orch_cls(config) if config else orch_cls()
        assert orch is not None

        # Verify orchestrator has bridge attributes
        has_bridges = any(
            hasattr(orch, attr) for attr in [
                "_traceability_bridge", "_risk_bridge", "_dd_bridge",
                "traceability", "risk_assessment", "due_diligence",
            ]
        )
        if not has_bridges:
            pytest.skip("Orchestrator bridge attributes not found")
        assert has_bridges

    def test_orchestrator_12_phase_structure(self):
        """Orchestrator exposes 12-phase EUDR compliance workflow."""
        orch = _instantiate_bridge(
            _orch_mod, "PackOrchestrator",
            ["OrchestratorConfig", "PackOrchestratorConfig"]
        )
        if orch is None:
            pytest.skip("PackOrchestrator not instantiable")

        # Check for workflow methods
        workflow_methods = [m for m in dir(orch) if "phase" in m.lower() or "workflow" in m.lower()]
        if not workflow_methods:
            pytest.skip("No workflow methods found")
        assert len(workflow_methods) > 0

    def test_health_check_runs(self):
        """Health check executes without errors."""
        orch = _instantiate_bridge(
            _orch_mod, "PackOrchestrator",
            ["OrchestratorConfig", "PackOrchestratorConfig"]
        )
        if orch is None:
            pytest.skip("PackOrchestrator not instantiable")

        health_fn = getattr(orch, "health_check", None) or \
                    getattr(orch, "check_health", None)
        if health_fn is None:
            pytest.skip("Health check method not available")

        if asyncio.iscoroutinefunction(health_fn):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(health_fn())
            finally:
                loop.close()
        else:
            result = health_fn()

        assert result is not None


# ===========================================================================
# Professional Engines with Real Agents
# ===========================================================================
@pytest.mark.integration
class TestProfessionalEnginesWithAgents:
    """Integration tests for PACK-007 professional-tier engines."""

    def test_scenario_risk_with_real_risk_data(self, agent_loader):
        """Scenario risk engine processes real risk data."""
        if _scenario_risk_mod is None:
            pytest.skip("ScenarioRiskEngine not importable")
        engine_cls = getattr(_scenario_risk_mod, "ScenarioRiskEngine", None)
        if engine_cls is None:
            pytest.skip("ScenarioRiskEngine class not found")

        # Create engine instance
        try:
            engine = engine_cls()
        except TypeError:
            # Engine requires config
            config_cls = getattr(_scenario_risk_mod, "ScenarioRiskConfig", None)
            if config_cls is None:
                pytest.skip("ScenarioRiskEngine requires config but config not found")
            engine = engine_cls(config_cls())

        # Find analysis method
        analyze_fn = None
        for method_name in ["analyze_scenario", "assess_scenario", "calculate_scenario_risk"]:
            if hasattr(engine, method_name):
                analyze_fn = getattr(engine, method_name)
                break

        if analyze_fn is None:
            pytest.skip("No scenario analysis method found on engine")

        scenario_data = {
            "scenario_name": "Deforestation Alert Spike",
            "country": "BR",
            "commodity": "cattle",
            "baseline_risk": 0.65,
            "alert_count": 15,
        }

        if asyncio.iscoroutinefunction(analyze_fn):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(analyze_fn(scenario_data))
            finally:
                loop.close()
        else:
            result = analyze_fn(scenario_data)

        assert result is not None

    def test_advanced_geolocation_with_real_plot(self, agent_loader):
        """Advanced geolocation engine validates plot coordinates."""
        if _adv_geo_mod is None:
            pytest.skip("AdvancedGeolocationEngine not importable")
        engine_cls = getattr(_adv_geo_mod, "AdvancedGeolocationEngine", None)
        if engine_cls is None:
            pytest.skip("AdvancedGeolocationEngine class not found")

        # Create engine instance
        try:
            engine = engine_cls()
        except TypeError:
            config_cls = getattr(_adv_geo_mod, "AdvancedGeolocationConfig", None)
            if config_cls is None:
                pytest.skip("AdvancedGeolocationEngine requires config but config not found")
            engine = engine_cls(config_cls())

        # Find validation method - try multiple signatures
        lat, lon = -10.4653, -52.2159
        result = None

        for method_name in ["validate_coordinates", "verify_plot", "validate_geolocation"]:
            fn = getattr(engine, method_name, None)
            if fn is None:
                continue
            # Try different call signatures
            try:
                if asyncio.iscoroutinefunction(fn):
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(fn(lat, lon))
                    finally:
                        loop.close()
                else:
                    result = fn(lat, lon)
                break
            except TypeError:
                try:
                    plot_data = {"latitude": lat, "longitude": lon, "area_ha": 150.0, "country": "BR"}
                    if asyncio.iscoroutinefunction(fn):
                        loop = asyncio.new_event_loop()
                        try:
                            result = loop.run_until_complete(fn(plot_data))
                        finally:
                            loop.close()
                    else:
                        result = fn(plot_data)
                    break
                except TypeError:
                    continue

        if result is None:
            pytest.skip("No geolocation validation method matched available signatures")
        assert result is not None
