# -*- coding: utf-8 -*-
"""
Load tests for Security Scanning Pipeline - SEC-007

This package contains load and performance tests for:
    - Scan throughput
    - Concurrent scan handling
    - Memory efficiency
    - Response time under load

Performance targets:
    - 1000 findings/second processing
    - <5s average scan time for medium repositories
    - <100MB memory overhead per scan
"""
