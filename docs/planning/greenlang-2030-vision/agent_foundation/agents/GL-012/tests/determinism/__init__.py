# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Determinism Tests Package.

Zero-hallucination validation tests ensuring bit-perfect reproducibility
of all calculations across runs, platforms, and environments.

Determinism Requirements:
    - SHA-256 hash consistency
    - Bit-perfect reproducibility
    - Platform independence
    - Seed propagation verification

Author: GreenLang Industrial Optimization Team
Agent ID: GL-012
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-012"

DETERMINISM_SEED = 42
REPRODUCIBILITY_RUNS = 1000
