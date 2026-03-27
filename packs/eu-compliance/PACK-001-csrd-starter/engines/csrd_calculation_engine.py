# -*- coding: utf-8 -*-
"""
CSRDCalculationEngine - ESRS E1 Calculation Engine Wrapping MRV Bridge
=======================================================================

This engine wraps the PACK-001 MRVBridge to provide a high-level calculation
interface for ESRS E1 climate metrics. It loads real GreenLang MRV agents
when available and falls back to deterministic stub calculations otherwise.

The engine is the primary integration point between PACK-001 and the
GreenLang MRV agent ecosystem (GL-MRV-X-001 through GL-MRV-X-030).

Example:
    >>> engine = CSRDCalculationEngine()
    >>> engine.load_agents()  # Loads real MRV agents if available
    >>> result = await engine.calculate_scope1({
    ...     "stationary_combustion": {"fuel_type": "natural_gas", "quantity": 1000}
    ... })
    >>> print(result.aggregated.total_emissions)

Author: GreenLang Team
Version: 1.0.0
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Attempt to import MRVBridge from the same pack
try:
    from packs.eu_compliance.PACK_001_csrd_starter.integrations.mrv_bridge import (
        MRVBridge,
        MRVBridgeConfig,
    )
except ImportError:
    try:
        import importlib
        import sys
        from pathlib import Path

        _pack_root = Path(__file__).resolve().parent.parent
        if str(_pack_root) not in sys.path:
            sys.path.insert(0, str(_pack_root))
        _mrv = importlib.import_module("integrations.mrv_bridge")
        MRVBridge = _mrv.MRVBridge
        MRVBridgeConfig = _mrv.MRVBridgeConfig
    except (ImportError, AttributeError):
        MRVBridge = None  # type: ignore[assignment, misc]
        MRVBridgeConfig = None  # type: ignore[assignment, misc]


# Agent ID to GreenLang module mapping
_AGENT_MODULE_MAP: Dict[str, str] = {
    "GL-MRV-X-001": "greenlang.agents.mrv.stationary_combustion",
    "GL-MRV-X-002": "greenlang.agents.mrv.refrigerants_fgas",
    "GL-MRV-X-003": "greenlang.agents.mrv.mobile_combustion",
    "GL-MRV-X-004": "greenlang.agents.mrv.process_emissions",
    "GL-MRV-X-005": "greenlang.agents.mrv.fugitive_emissions",
    "GL-MRV-X-006": "greenlang.agents.mrv.land_use_emissions",
    "GL-MRV-X-007": "greenlang.waste_treatment",
    "GL-MRV-X-008": "greenlang.agents.mrv.agricultural_emissions",
    "GL-MRV-X-009": "greenlang.agents.mrv.scope2_location",
    "GL-MRV-X-010": "greenlang.agents.mrv.scope2_market",
    "GL-MRV-X-011": "greenlang.steam_heat",
    "GL-MRV-X-012": "greenlang.agents.mrv.cooling_purchase",
    "GL-MRV-X-013": "greenlang.dual_reporting",
    "GL-MRV-X-014": "greenlang.purchased_goods",
    "GL-MRV-X-015": "greenlang.agents.mrv.capital_goods",
    "GL-MRV-X-016": "greenlang.agents.mrv.fuel_energy_activities",
    "GL-MRV-X-017": "greenlang.agents.mrv.upstream_transportation",
    "GL-MRV-X-018": "greenlang.agents.mrv.waste_generated",
    "GL-MRV-X-019": "greenlang.agents.mrv.business_travel",
    "GL-MRV-X-020": "greenlang.agents.mrv.employee_commuting",
    "GL-MRV-X-021": "greenlang.upstream_leased",
    "GL-MRV-X-022": "greenlang.agents.mrv.downstream_transportation",
    "GL-MRV-X-023": "greenlang.agents.mrv.processing_sold_products",
    "GL-MRV-X-024": "greenlang.use_sold_products",
    "GL-MRV-X-025": "greenlang.end_of_life",
    "GL-MRV-X-026": "greenlang.downstream_leased",
    "GL-MRV-X-027": "greenlang.agents.mrv.franchises",
    "GL-MRV-X-028": "greenlang.agents.mrv.investments",
    "GL-MRV-X-029": "greenlang.agents.mrv.scope3_category_mapper",
    "GL-MRV-X-030": "greenlang.agents.mrv.audit_trail_lineage",
}


class CSRDCalculationEngine:
    """High-level ESRS E1 calculation engine for PACK-001.

    Wraps MRVBridge and provides methods to load real GreenLang MRV agents
    into the bridge's agent registry, replacing deterministic stubs with
    live agent instances.

    Attributes:
        bridge: The underlying MRVBridge instance
        loaded_agents: Set of agent IDs that were successfully loaded
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        if MRVBridge is None:
            raise ImportError(
                "MRVBridge not available. Ensure PACK-001 integrations are importable."
            )
        bridge_config = config if config is not None else MRVBridgeConfig()
        self.bridge = MRVBridge(bridge_config)
        self.loaded_agents: List[str] = []
        logger.info("CSRDCalculationEngine initialized")

    def load_agents(self, agent_ids: Optional[List[str]] = None) -> Dict[str, bool]:
        """Attempt to load real GreenLang MRV agent modules.

        For each agent ID, tries to import the corresponding greenlang module
        and find a service or engine class to instantiate. Successfully loaded
        agents are injected into the MRVBridge._agents dict.

        Args:
            agent_ids: Specific agent IDs to load. If None, attempts all 30.

        Returns:
            Dict mapping agent_id to load success boolean.
        """
        import importlib

        ids_to_load = agent_ids or list(_AGENT_MODULE_MAP.keys())
        results: Dict[str, bool] = {}

        for agent_id in ids_to_load:
            module_path = _AGENT_MODULE_MAP.get(agent_id)
            if module_path is None:
                results[agent_id] = False
                continue

            try:
                mod = importlib.import_module(module_path)
                agent_instance = _resolve_agent_instance(mod)
                if agent_instance is not None:
                    self.bridge._agents[agent_id] = agent_instance
                    self.loaded_agents.append(agent_id)
                    results[agent_id] = True
                    logger.info("Loaded agent %s from %s", agent_id, module_path)
                else:
                    results[agent_id] = False
                    logger.debug(
                        "Module %s imported but no agent class resolved", module_path
                    )
            except ImportError as exc:
                results[agent_id] = False
                logger.debug("Failed to import %s for %s: %s", module_path, agent_id, exc)

        return results

    async def calculate_scope1(self, data: Dict[str, Any]) -> Any:
        """Calculate Scope 1 emissions via bridge."""
        return await self.bridge.calculate_scope1(data)

    async def calculate_scope2(self, data: Dict[str, Any]) -> Any:
        """Calculate Scope 2 emissions via bridge."""
        return await self.bridge.calculate_scope2(data)

    async def calculate_scope3(
        self, data: Dict[str, Any], categories: Optional[List[int]] = None
    ) -> Any:
        """Calculate Scope 3 emissions via bridge."""
        return await self.bridge.calculate_scope3(data, categories)

    async def calculate_metric(self, metric_code: str, data: Dict[str, Any]) -> Any:
        """Calculate a single ESRS E1 metric."""
        return await self.bridge.route_calculation(metric_code, data)


def _resolve_agent_instance(module: Any) -> Any:
    """Try to find and instantiate an agent/service/engine from a module.

    Searches for common class name patterns used across MRV agents.
    """
    # Try known service facade patterns
    for attr_name in [
        "get_service",
        "StationaryCombustionService",
        "MobileCombustionService",
        "RefrigerantsFGasService",
        "ProcessEmissionsService",
        "FugitiveEmissionsService",
        "LandUseEmissionsService",
        "WasteTreatmentService",
        "AgriculturalEmissionsService",
        "Scope2LocationService",
        "Scope2MarketService",
        "SteamHeatService",
        "CoolingPurchaseService",
        "DualReportingService",
    ]:
        obj = getattr(module, attr_name, None)
        if obj is not None:
            if callable(obj) and not isinstance(obj, type):
                try:
                    return obj()
                except Exception:
                    pass
            elif isinstance(obj, type):
                try:
                    return obj()
                except Exception:
                    pass

    # Try generic patterns: *Service, *Engine, *Pipeline
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        if attr_name.endswith(("Service", "PipelineEngine")):
            cls = getattr(module, attr_name, None)
            if isinstance(cls, type):
                try:
                    return cls()
                except Exception:
                    continue

    return None
