# -*- coding: utf-8 -*-
"""
Load tests for PII Service - SEC-011 PII Detection/Redaction Enhancements.

This package contains load and performance tests that verify PII Service
can handle production-level throughput:
- Detection throughput (10K+ messages/second)
- Tokenization throughput
- Enforcement latency (P99 < 10ms)
- Streaming throughput
- Vault capacity stress tests

Tests are designed to run with realistic data volumes.
"""

from __future__ import annotations

__all__: list[str] = []
