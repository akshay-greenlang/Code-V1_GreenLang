# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT Determinism Tests Package.

Zero-hallucination validation tests ensuring bit-perfect reproducibility
of all fuel management calculations across runs, platforms, and environments.

Determinism Requirements:
    - SHA-256 hash consistency for provenance tracking
    - Bit-perfect reproducibility of calorific value calculations
    - Heating value (HHV/LHV) calculation determinism
    - Combustion stoichiometry reproducibility
    - Emission factor calculation consistency
    - Fuel blending ratio determinism
    - Cost optimization result stability

Standards Compliance:
    - ISO 6976:2016 - Natural gas calorific value
    - ISO 17225 - Solid biofuels specifications
    - ASTM D4809 - Heat of combustion
    - IPCC 2006 Guidelines - Emission factors
    - GHG Protocol - Emissions calculations

Author: GreenLang Industrial Optimization Team
Agent ID: GL-011
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-011"

DETERMINISM_SEED = 42
REPRODUCIBILITY_RUNS = 1000
GOLDEN_FILE_RUNS = 100
LLM_TEMPERATURE = 0.0
HASH_ALGORITHM = "sha256"
