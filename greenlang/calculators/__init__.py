# -*- coding: utf-8 -*-
"""
GreenLang Calculators Package - Backward Compatibility Module

DEPRECATED: This module has been moved to greenlang.calculation.industry
Please update your imports:

  Old: from greenlang.calculators import sb253
  New: from greenlang.calculation.industry import sb253

This package provides regulatory-compliant emission calculators for various
sustainability reporting frameworks including:

- California SB 253 (Climate Corporate Data Accountability Act)
- GHG Protocol Scope 1, 2, 3
- EPA emissions reporting
- CBAM (Carbon Border Adjustment Mechanism)
- EUDR (EU Deforestation Regulation)

All calculators follow the GreenLang Zero-Hallucination principle:
- No LLM in calculation paths
- 100% deterministic calculations
- Full SHA-256 audit trails
- Regulatory-grade precision

Version: 1.0.0
"""

import warnings

warnings.warn(
    "The greenlang.calculators module has been moved to greenlang.calculation.industry. "
    "Please update your imports to use the new location.",
    DeprecationWarning,
    stacklevel=2
)

__version__ = "1.0.0"
