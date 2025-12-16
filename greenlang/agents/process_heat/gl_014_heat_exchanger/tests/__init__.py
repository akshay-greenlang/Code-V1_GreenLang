# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Test Suite

Comprehensive test coverage for the Heat Exchanger Optimization agent.
Target coverage: 85%+

Test Categories:
    - Unit tests for all calculation methods
    - Integration tests for component interactions
    - Performance tests for throughput validation
    - Compliance tests for TEMA/ASME standards

Test Files:
    - test_config.py: Configuration validation tests
    - test_schemas.py: Schema validation tests
    - test_effectiveness.py: e-NTU and LMTD calculations
    - test_fouling.py: Fouling analysis per TEMA RGP-T2.4
    - test_cleaning.py: Cleaning schedule optimization
    - test_tube_analysis.py: Tube integrity and Weibull analysis
    - test_hydraulics.py: Pressure drop calculations
    - test_economics.py: Economic analysis (NPV, ROI)
    - test_optimizer.py: Main optimizer integration

Usage:
    pytest greenlang/agents/process_heat/gl_014_heat_exchanger/tests/ -v --cov
"""

__version__ = "1.0.0"
