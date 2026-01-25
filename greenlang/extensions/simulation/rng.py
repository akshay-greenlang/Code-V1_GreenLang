# -*- coding: utf-8 -*-
"""
GreenLang RNG (GLRNG) - SIM-401 Compliant Location

This module provides the deterministic random number generator implementation
as specified in SIM-401. The actual implementation is in greenlang.intelligence.glrng
for reuse across simulation and AI agent contexts, but this wrapper provides
the spec-compliant import path.

Usage:
    from greenlang.simulation.rng import GLRNG

    # Create root RNG
    rng = GLRNG(seed=42)

    # Sample values
    x = rng.uniform(0, 1)
    y = rng.normal(mean=100, std=15)

    # Create substream for Monte Carlo trial
    trial_rng = rng.spawn("trial:0")
    sample = trial_rng.triangular(low=0.08, mode=0.12, high=0.22)

Author: GreenLang Framework Team
Date: October 2025
Spec: SIM-401 (Scenario Spec & Seeded RNG)
"""

# Import all public APIs from the actual implementation
from greenlang.intelligence.glrng import (
    # Core classes
    GLRNG,
    SplitMix64,

    # Functions
    derive_substream_seed,
    create_rng_from_config,
)

__all__ = [
    # Core classes
    "GLRNG",
    "SplitMix64",

    # Functions
    "derive_substream_seed",
    "create_rng_from_config",
]
