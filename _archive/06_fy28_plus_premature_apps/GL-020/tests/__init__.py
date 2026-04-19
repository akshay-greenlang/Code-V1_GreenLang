"""
GL-020 ECONOPULSE - Economizer Performance Agent Test Suite

Comprehensive test suite for economizer performance monitoring and fouling detection.

Agent ID: GL-020
Codename: ECONOPULSE
Name: EconomizerPerformanceAgent
Description: Monitors economizer performance and fouling

Test Coverage Target: 90%+

Test Modules:
- test_heat_transfer_calculator.py: LMTD, U-value, heat duty calculations
- test_fouling_calculator.py: Fouling factor, cleaning prediction
- test_economizer_efficiency.py: Effectiveness, heat recovery calculations
- test_thermal_properties.py: Water/gas Cp, IAPWS-IF97 validation
- test_alert_manager.py: Threshold, rate-of-change, deduplication

Author: GL-TestEngineer
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-020"
__codename__ = "ECONOPULSE"
