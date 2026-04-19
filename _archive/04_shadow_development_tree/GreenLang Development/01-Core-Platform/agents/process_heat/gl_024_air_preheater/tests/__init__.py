# -*- coding: utf-8 -*-
"""
GL-024 Air Preheater Agent - Test Suite

Comprehensive test coverage for all GL-024 modules:
    - schemas: Pydantic models and data validation
    - config: Configuration parameters and thresholds
    - calculations: Zero-hallucination heat transfer calculations
    - agent: Main orchestration logic
    - explainability: LIME-based explanations
    - provenance: Audit trail and hash verification

Run tests with:
    pytest greenlang/agents/process_heat/gl_024_air_preheater/tests/

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

__all__ = [
    "test_schemas",
    "test_calculations",
    "test_agent",
    "test_explainability",
    "test_provenance",
]
