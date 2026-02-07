# -*- coding: utf-8 -*-
"""
GreenLang Monitoring Load Tests
===============================

Load tests for the Prometheus monitoring stack to verify:
- High cardinality handling (1M+ time series)
- Concurrent query performance (100+ queries)
- Ingestion rate capacity (100K+ samples/sec)

These tests are designed to stress test the monitoring infrastructure
and identify performance bottlenecks.

Requirements:
- Running Prometheus with sufficient resources
- Sufficient disk space for test data
- Test environment isolated from production

Run with: pytest tests/load/monitoring/ -v --timeout=600
"""
