# -*- coding: utf-8 -*-
"""GL-015 INSULSCAN - Integration tests.

Integration test package for end-to-end testing of the
Insulation Scanning and Thermal Assessment Agent.

Tests complete workflows including:
- Orchestrator analysis pipeline
- API endpoint integration
- Data flow validation
- Provenance chain validation
- Multi-component interaction

Test Categories:
    - test_orchestrator: Orchestrator integration tests
    - test_analysis_pipeline: Full analysis pipeline tests
    - test_api_integration: REST API integration tests

Author: GL-TestEngineer
Version: 1.0.0
"""

__all__ = [
    "test_orchestrator",
    "test_analysis_pipeline",
]
