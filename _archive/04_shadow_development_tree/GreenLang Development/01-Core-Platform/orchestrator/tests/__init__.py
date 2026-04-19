# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Tests - GL-FOUND-X-001

Determinism Test Suite for the GreenLang Orchestrator.

This test package verifies:
- Plan determinism (content-addressable plan_id)
- Execution determinism (mock agent outputs)
- Timing independence (DeterministicClock usage)
- Cross-environment consistency (K8s/S3 mocks)
- Hash chain verification

Test Organization:
- conftest.py: Shared fixtures for orchestrator testing
- test_plan_determinism.py: Plan compilation determinism tests
- test_hash_chain.py: Audit event hash chain tests
- test_idempotency.py: Idempotency key and retry tests

Author: GreenLang Team
Version: 1.0.0
"""

__all__ = [
    "conftest",
    "test_plan_determinism",
    "test_hash_chain",
    "test_idempotency",
]
