"""
CSRD Test Suite
===============

Comprehensive testing for all agents and pipeline components.

Test Coverage Targets:
- IntakeAgent: 90%
- CalculatorAgent: 100% (critical - zero hallucination)
- MaterialityAgent: 80%
- AggregatorAgent: 90%
- ReportingAgent: 85%
- AuditAgent: 95%
- Pipeline Integration: 85%

Run all tests:
    pytest tests/ -v --cov

Run specific agent tests:
    pytest tests/test_calculator_agent.py -v
"""
