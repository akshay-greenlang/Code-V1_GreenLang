"""
Phase 5 Compliance Test Suite

Critical Path Agent Compliance Tests for Regulatory Validation

This test suite ensures that CRITICAL PATH agents maintain:
1. Complete determinism (identical outputs for identical inputs)
2. Zero LLM dependencies (no ChatSession, no API calls)
3. Performance requirements (<10ms execution time)
4. Complete audit trails
5. Proper deprecation warnings for AI versions

These tests are essential for regulatory compliance and must pass 100%.
"""

__version__ = "1.0.0"
