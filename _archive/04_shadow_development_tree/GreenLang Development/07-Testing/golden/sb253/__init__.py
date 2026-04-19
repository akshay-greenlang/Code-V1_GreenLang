# -*- coding: utf-8 -*-
"""
SB 253 Golden Test Suite
========================

Comprehensive golden test suite for California SB 253 Climate Disclosure
compliance calculations.

Test Distribution:
    - Scope 1 (Direct): 60 tests
    - Scope 2 (Energy Indirect): 70 tests
    - Scope 3 (Value Chain): 120 tests
    - Verification: 50 tests

Total: 300 golden tests

Accuracy Requirements:
    - Scope 1: +/- 1%
    - Scope 2: +/- 2%
    - Scope 3: +/- 5%

Author: GreenLang Framework Team
Version: 1.0.0
"""

__all__ = [
    "test_scope1_golden",
    "test_scope2_golden",
    "test_scope3_golden",
    "test_verification_golden",
]
