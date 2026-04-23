"""
GL-007 FURNACEPULSE - Test Suite

Comprehensive testing including:
- Unit tests for all calculators (efficiency, hotspot, RUL, draft)
- Integration tests for orchestrator pipelines
- Property-based tests for determinism and provenance
- API endpoint tests with RBAC enforcement
- Compliance tests for NFPA 86 requirements
- Performance benchmarks for real-time processing

Test Categories:
- test_efficiency_calculator.py: Fuel input, SFC, excess air calculations
- test_hotspot_detector.py: TMT monitoring, spatial clustering, alert tiers
- test_rul_predictor.py: Weibull fitting, confidence intervals, maintenance history
- test_nfpa86_compliance.py: Checklist evaluation, evidence packaging
- test_orchestrator.py: Full pipeline integration tests
- test_api.py: REST/GraphQL endpoints, RBAC, error handling

Coverage Target: >85%

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent__ = "GL-007_FurnacePulse"
