# -*- coding: utf-8 -*-
"""
Entity Resolution System

Multi-strategy entity resolution with confidence scoring.

Components:
- EntityResolver: Main resolution orchestrator
- Deterministic & Fuzzy Matchers
- MDM Integration (LEI, DUNS, OpenCorporates stubs)

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

from .resolver import EntityResolver
from .matchers import DeterministicMatcher, FuzzyMatcher
from .mdm_integration import MDMIntegrator

__all__ = [
    "EntityResolver",
    "DeterministicMatcher",
    "FuzzyMatcher",
    "MDMIntegrator",
]
