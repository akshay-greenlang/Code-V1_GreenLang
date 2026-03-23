# -*- coding: utf-8 -*-
"""
AGENT-MRV-010: Scope 2 Market-Based Emissions Agent.

Calculates Scope 2 market-based GHG emissions from purchased electricity,
steam, heating, and cooling using contractual instruments, supplier-specific
factors, and residual mix factors per the GHG Protocol Scope 2 Guidance (2015).

Package: greenlang.agents.mrv.scope2_market
DB Migration: V061
API Prefix: /api/v1/scope2-market
Metrics Prefix: gl_s2m_
Table Prefix: s2m_

Engines:
    1. ContractualInstrumentDatabaseEngine — Instrument & factor database
    2. InstrumentAllocationEngine — Priority-based instrument allocation
    3. MarketEmissionsCalculatorEngine — Market-based emission calculations
    4. DualReportingEngine — Location vs. market dual reporting
    5. UncertaintyQuantifierEngine — Monte Carlo + analytical uncertainty
    6. ComplianceCheckerEngine — Multi-framework compliance checking
    7. Scope2MarketPipelineEngine — 8-stage orchestrated pipeline

Author: GreenLang Platform Team
Date: February 2026
"""

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-SCOPE2-010"

# Lazy exports
def __getattr__(name):
    _exports = {
        "ContractualInstrumentDatabaseEngine": "greenlang.agents.mrv.scope2_market.contractual_instrument_database",
        "InstrumentAllocationEngine": "greenlang.agents.mrv.scope2_market.instrument_allocation",
        "MarketEmissionsCalculatorEngine": "greenlang.agents.mrv.scope2_market.market_emissions_calculator",
        "DualReportingEngine": "greenlang.agents.mrv.scope2_market.dual_reporting",
        "UncertaintyQuantifierEngine": "greenlang.agents.mrv.scope2_market.uncertainty_quantifier",
        "ComplianceCheckerEngine": "greenlang.agents.mrv.scope2_market.compliance_checker",
        "Scope2MarketPipelineEngine": "greenlang.agents.mrv.scope2_market.scope2_market_pipeline",
        "Scope2MarketConfig": "greenlang.agents.mrv.scope2_market.config",
        "Scope2MarketMetrics": "greenlang.agents.mrv.scope2_market.metrics",
        "Scope2MarketProvenance": "greenlang.agents.mrv.scope2_market.provenance",
        "Scope2MarketService": "greenlang.agents.mrv.scope2_market.setup",
        "get_service": "greenlang.agents.mrv.scope2_market.setup",
    }
    if name in _exports:
        import importlib
        module = importlib.import_module(_exports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ContractualInstrumentDatabaseEngine",
    "InstrumentAllocationEngine",
    "MarketEmissionsCalculatorEngine",
    "DualReportingEngine",
    "UncertaintyQuantifierEngine",
    "ComplianceCheckerEngine",
    "Scope2MarketPipelineEngine",
    "Scope2MarketConfig",
    "Scope2MarketMetrics",
    "Scope2MarketProvenance",
    "Scope2MarketService",
    "get_service",
]
