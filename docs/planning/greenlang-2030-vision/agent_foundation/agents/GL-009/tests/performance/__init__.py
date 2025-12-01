# -*- coding: utf-8 -*-
"""
Performance tests for GL-009 THERMALIQ (ThermalStorageOptimizer).

This package contains comprehensive performance benchmark tests for the
GL-009 agent handling thermal storage systems including molten salt,
phase change materials (PCM), and hot water tanks.

Test Categories:
    - State of charge calculation performance
    - Thermal loss calculation performance
    - Charge/discharge cycle optimization
    - Exergy calculation benchmarks
    - Batch calculation throughput
    - Cache performance tests
    - Memory stability tests

Performance Targets:
    - State of charge calculation: <1ms
    - Thermal loss calculation: <2ms
    - Batch throughput: >1000 calculations/sec
    - Cache hit latency: <0.1ms
    - Memory growth: <50MB per 10k operations

Author: GL-TestEngineer
Version: 1.0.0
"""
