"""
Registry Update: GL-014 to GL-019
================================

This file contains the AgentInfo definitions for the 6 missing agents
that need to be added to registry.py after their implementation is complete.

INSTRUCTIONS:
1. After all agents are implemented, add these entries to AGENT_DEFINITIONS in registry.py
2. Insert them after GL-013 and before GL-020
3. Run the registry health check to verify: python -m agents.registry

Generated: December 2025
"""

from dataclasses import dataclass, field
from typing import List

# These are the 6 missing agent definitions to add to registry.py

MISSING_AGENTS = [
    # ========================================
    # Process Heat Foundation (GL-014 to GL-019)
    # These were missing from the original registry
    # ========================================

    # AgentInfo("GL-014", "EXCHANGERPRO", "gl_014_heat_exchanger", "HeatExchangerOptimizerAgent",
    #           "Heat Exchangers", "Optimizer", "Medium", "P1", "$6B",
    #           "TEMA-compliant heat exchanger optimization with epsilon-NTU method, SHAP/LIME explainability, "
    #           "fouling prediction ML, cleaning schedule optimization, LMTD analysis, and zero-hallucination "
    #           "deterministic heat transfer calculations.",
    #           ["TEMA", "ASME"]),

    # AgentInfo("GL-015", "INSULSCAN", "gl_015_insulation", "InsulationAnalysisAgent",
    #           "Energy Conservation", "Monitor", "Low", "P2", "$3B",
    #           "Comprehensive insulation analysis with 50+ material database, SHAP/LIME explainability, "
    #           "economic thickness calculations, thermal imaging integration, zero-hallucination heat loss "
    #           "calculations, and ROI optimization.",
    #           ["ASTM C680"]),

    # AgentInfo("GL-016", "WATERGUARD", "gl_016_boiler_water", "BoilerWaterTreatmentAgent",
    #           "Boiler Systems", "Controller", "Medium", "P1", "$5B",
    #           "ASME/ABMA compliant water treatment with cycles of concentration optimization, "
    #           "SHAP/LIME explainability, blowdown control, chemical dosing ML optimization, "
    #           "and zero-hallucination chemistry calculations.",
    #           ["ASME", "ABMA"]),

    # AgentInfo("GL-017", "CONDENSYNC", "gl_017_condenser", "CondenserOptimizationAgent",
    #           "Steam Systems", "Optimizer", "Medium", "P2", "$4B",
    #           "HEI Standards compliant condenser optimization with cleanliness factor tracking, "
    #           "SHAP/LIME explainability, vacuum optimization, ML fouling prediction, "
    #           "and zero-hallucination heat transfer calculations.",
    #           ["HEI"]),

    # AgentInfo("GL-018", "UNIFIEDCOMBUSTION", "gl_018_unified_combustion", "UnifiedCombustionOptimizerAgent",
    #           "Combustion", "Optimizer", "High", "P0", "$24B",
    #           "Unified combustion optimizer with NFPA 85/86 compliance, SHAP/LIME explainability, "
    #           "causal inference, attention visualization, O2 trim, CO optimization, excess air control, "
    #           "comprehensive safety interlocks. Consolidates GL-002 and GL-004 functions.",
    #           ["NFPA 85", "NFPA 86"]),

    # AgentInfo("GL-019", "HEATSCHEDULER", "gl_019_heat_scheduler", "ProcessHeatingSchedulerAgent",
    #           "Planning", "Coordinator", "Medium", "P1", "$7B",
    #           "ML-based demand forecasting with SHAP/LIME explainability, thermal storage optimization, "
    #           "TOU tariff arbitrage, uncertainty quantification, production schedule integration, "
    #           "and SSE streaming schedule updates.",
    #           []),
]

# Python code to add to registry.py (after GL-013, before GL-020)
REGISTRY_PATCH = '''
    # ========================================
    # Process Heat Foundation (GL-014 to GL-019)
    # Added: December 2025
    # ========================================
    AgentInfo("GL-014", "EXCHANGERPRO", "gl_014_heat_exchanger", "HeatExchangerOptimizerAgent", "Heat Exchangers", "Optimizer", "Medium", "P1", "$6B"),
    AgentInfo("GL-015", "INSULSCAN", "gl_015_insulation", "InsulationAnalysisAgent", "Energy Conservation", "Monitor", "Low", "P2", "$3B"),
    AgentInfo("GL-016", "WATERGUARD", "gl_016_boiler_water", "BoilerWaterTreatmentAgent", "Boiler Systems", "Controller", "Medium", "P1", "$5B"),
    AgentInfo("GL-017", "CONDENSYNC", "gl_017_condenser", "CondenserOptimizationAgent", "Steam Systems", "Optimizer", "Medium", "P2", "$4B"),
    AgentInfo("GL-018", "UNIFIEDCOMBUSTION", "gl_018_unified_combustion", "UnifiedCombustionOptimizerAgent", "Combustion", "Optimizer", "High", "P0", "$24B"),
    AgentInfo("GL-019", "HEATSCHEDULER", "gl_019_heat_scheduler", "ProcessHeatingSchedulerAgent", "Planning", "Coordinator", "Medium", "P1", "$7B"),
'''

if __name__ == "__main__":
    print("Registry Update Instructions")
    print("=" * 50)
    print()
    print("After implementing GL-014 to GL-019 agents, add the following")
    print("to registry.py AGENT_DEFINITIONS list (after GL-013, before GL-020):")
    print()
    print(REGISTRY_PATCH)
    print()
    print("Total market size of new agents: $49B")
