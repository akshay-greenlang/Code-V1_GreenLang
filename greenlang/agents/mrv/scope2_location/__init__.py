# -*- coding: utf-8 -*-
"""
AGENT-MRV-009: Scope 2 Location-Based Emissions Agent.

Calculates Scope 2 location-based GHG emissions from purchased electricity,
steam, heating, and cooling using grid-average emission factors per the
GHG Protocol Scope 2 Guidance (2015).

Package: greenlang.agents.mrv.scope2_location
DB Migration: V060
API Prefix: /api/v1/scope2-location
Metrics Prefix: gl_s2l_
Table Prefix: s2l_

Engines:
    1. GridEmissionFactorDatabaseEngine — Grid EF database (eGRID, IEA, EU EEA, DEFRA)
    2. ElectricityEmissionsEngine — Electricity emission calculations
    3. SteamHeatCoolingEngine — Steam/heat/cooling calculations
    4. TransmissionLossEngine — T&D loss adjustments
    5. UncertaintyQuantifierEngine — Monte Carlo + analytical uncertainty
    6. ComplianceCheckerEngine — Multi-framework compliance checking
    7. Scope2LocationPipelineEngine — 8-stage orchestrated pipeline

Author: GreenLang Platform Team
Date: February 2026
"""

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-SCOPE2-009"

# Lazy exports
def __getattr__(name):
    _exports = {
        "GridEmissionFactorDatabaseEngine": "greenlang.agents.mrv.scope2_location.grid_factor_database",
        "ElectricityEmissionsEngine": "greenlang.agents.mrv.scope2_location.electricity_emissions",
        "SteamHeatCoolingEngine": "greenlang.agents.mrv.scope2_location.steam_heat_cooling",
        "TransmissionLossEngine": "greenlang.agents.mrv.scope2_location.transmission_loss",
        "UncertaintyQuantifierEngine": "greenlang.agents.mrv.scope2_location.uncertainty_quantifier",
        "ComplianceCheckerEngine": "greenlang.agents.mrv.scope2_location.compliance_checker",
        "Scope2LocationPipelineEngine": "greenlang.agents.mrv.scope2_location.scope2_location_pipeline",
        "Scope2LocationConfig": "greenlang.agents.mrv.scope2_location.config",
        "Scope2LocationMetrics": "greenlang.agents.mrv.scope2_location.metrics",
        "Scope2LocationProvenance": "greenlang.agents.mrv.scope2_location.provenance",
        "Scope2LocationService": "greenlang.agents.mrv.scope2_location.setup",
        "get_service": "greenlang.agents.mrv.scope2_location.setup",
    }
    if name in _exports:
        import importlib
        module = importlib.import_module(_exports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GridEmissionFactorDatabaseEngine",
    "ElectricityEmissionsEngine",
    "SteamHeatCoolingEngine",
    "TransmissionLossEngine",
    "UncertaintyQuantifierEngine",
    "ComplianceCheckerEngine",
    "Scope2LocationPipelineEngine",
    "Scope2LocationConfig",
    "Scope2LocationMetrics",
    "Scope2LocationProvenance",
    "Scope2LocationService",
    "get_service",
]
