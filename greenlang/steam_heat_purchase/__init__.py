# -*- coding: utf-8 -*-
"""
Steam/Heat Purchase Agent - AGENT-MRV-011

Scope 2 GHG emissions from purchased steam, district heating, and
district cooling per GHG Protocol Scope 2 Guidance (2015).

Package: greenlang.steam_heat_purchase
PRD: GL-MRV-X-022
DB Migration: V062
API: /api/v1/steam-heat-purchase
Metrics Prefix: gl_shp_

Engines:
    1. SteamHeatDatabaseEngine     - Emission factor database
    2. SteamEmissionsCalculatorEngine - Steam emission calculations
    3. HeatCoolingCalculatorEngine  - District heating & cooling
    4. CHPAllocationEngine          - CHP allocation methods
    5. UncertaintyQuantifierEngine  - Uncertainty quantification
    6. ComplianceCheckerEngine      - Regulatory compliance
    7. SteamHeatPipelineEngine      - Pipeline orchestrator

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-X-022"
__agent_label__ = "AGENT-MRV-011"


def __getattr__(name: str):
    """Lazy imports for heavy modules."""
    _lazy = {
        "SteamHeatDatabaseEngine": "greenlang.steam_heat_purchase.steam_heat_database",
        "SteamEmissionsCalculatorEngine": "greenlang.steam_heat_purchase.steam_emissions_calculator",
        "HeatCoolingCalculatorEngine": "greenlang.steam_heat_purchase.heat_cooling_calculator",
        "CHPAllocationEngine": "greenlang.steam_heat_purchase.chp_allocation",
        "UncertaintyQuantifierEngine": "greenlang.steam_heat_purchase.uncertainty_quantifier",
        "ComplianceCheckerEngine": "greenlang.steam_heat_purchase.compliance_checker",
        "SteamHeatPipelineEngine": "greenlang.steam_heat_purchase.steam_heat_pipeline",
        "SteamHeatPurchaseService": "greenlang.steam_heat_purchase.setup",
        "SteamHeatPurchaseConfig": "greenlang.steam_heat_purchase.config",
        "SteamHeatPurchaseMetrics": "greenlang.steam_heat_purchase.metrics",
        "SteamHeatPurchaseProvenance": "greenlang.steam_heat_purchase.provenance",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__agent_id__",
    "__agent_label__",
    "SteamHeatDatabaseEngine",
    "SteamEmissionsCalculatorEngine",
    "HeatCoolingCalculatorEngine",
    "CHPAllocationEngine",
    "UncertaintyQuantifierEngine",
    "ComplianceCheckerEngine",
    "SteamHeatPipelineEngine",
    "SteamHeatPurchaseService",
    "SteamHeatPurchaseConfig",
    "SteamHeatPurchaseMetrics",
    "SteamHeatPurchaseProvenance",
]
