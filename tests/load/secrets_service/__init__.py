# -*- coding: utf-8 -*-
"""
Load tests for Secrets Service - SEC-006

Performance and stress tests for:
- Read throughput (target: 1000+ reads/sec)
- Concurrent write operations
- Rotation under load
- Cache performance
- Connection pool behavior
- Memory usage under load

Uses pytest-asyncio and custom load generators.
"""
