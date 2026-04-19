"""
GL-013 PredictiveMaintenance Test Suite

Comprehensive test suite for the GL-013 PREDICTMAINT (Predictive Maintenance Agent).
Targets 85%+ code coverage across all modules.

Test Categories:
- Unit Tests: Individual component testing
  - test_weibull.py: Weibull distribution analysis
  - test_rul_estimator.py: Remaining Useful Life estimation
  - test_signal_processing.py: Vibration, thermal, and electrical analysis
  - test_explainability.py: SHAP/LIME explanations and causal graphs
  - test_data_quality.py: Data validation and sensor health

- Integration Tests: End-to-end workflow testing
  - test_prediction_pipeline.py: Full prediction workflow
  - test_cmms_flow.py: CMMS integration and work order management

Author: GL-TestEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

__version__ = "1.0.0"
__author__ = "GL-TestEngineer"

# Test configuration
TEST_CONFIG = {
    "coverage_target": 0.85,
    "performance_threshold_ms": 100,
    "async_timeout_seconds": 30,
    "monte_carlo_iterations": 1000,
    "random_seed": 42,
}
