# -*- coding: utf-8 -*-
"""
GL-007 FURNACEPULSE Determinism Tests Package

This package contains determinism and reproducibility tests for the
FurnacePerformanceOptimizer agent ensuring:
- Bit-perfect reproducibility of all calculations
- Thermal efficiency calculation consistency
- Fuel consumption analysis reproducibility
- Anomaly detection determinism
- Provenance hash consistency
- Floating-point stability with Decimal
- Zero-hallucination verification

Zero-Hallucination Principle:
All thermodynamic and efficiency calculations must produce identical results
when given identical inputs. This is fundamental to regulatory compliance,
audit trail integrity, and industrial process certification.
"""
