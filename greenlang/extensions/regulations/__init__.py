# -*- coding: utf-8 -*-
"""
greenlang/regulations/__init__.py

Regulatory Compliance Framework

This module provides implementations for various environmental and sustainability
regulations including:
- EUDR (EU Deforestation Regulation)
- CBAM (Carbon Border Adjustment Mechanism)
- CSRD (Corporate Sustainability Reporting Directive)
- And other regulatory frameworks

Each regulation submodule provides:
- Data models for compliance data
- Validation logic for regulatory requirements
- Calculation engines for regulatory metrics
- Reporting utilities

Author: GreenLang Framework Team
"""

from greenlang.regulations import eudr

__all__ = [
    "eudr",
]
