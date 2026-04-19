# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Test Suite

Comprehensive test suite for the Predictive Maintenance Agent achieving 85%+ coverage.

Test modules:
- test_config.py: Configuration validation tests
- test_schemas.py: Schema and data model tests
- test_weibull.py: Weibull distribution analysis tests
- test_oil_analysis.py: Oil analysis trending tests
- test_vibration.py: Vibration signature analysis tests
- test_thermography.py: IR thermography fault detection tests
- test_mcsa.py: Motor current signature analysis tests
- test_failure_prediction.py: ML failure prediction tests
- test_predictor.py: Main predictor integration tests

Run tests:
    pytest greenlang/agents/process_heat/gl_013_predictive_maintenance/tests/ -v --cov

Coverage target: 85%+
"""

__version__ = "1.0.0"
