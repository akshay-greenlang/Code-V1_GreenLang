# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft Compliance Module.

This module provides data quality gates and compliance validation for
fuel blending optimization with zero-hallucination governance:

- Critical feed validation (inventory, calorific values, emission factors)
- Timeliness validation (10-minute max for telemetry)
- Completeness validation (95% prices, 100% inventory/contracts)
- Outlier detection with fallback policies

Zero-Hallucination Governance:
- Fail-closed for critical missing data
- No free-form narrative generation
- All outputs traceable to raw records
- Deterministic validation rules

Example:
    >>> from compliance import DataQualityGateRunner
    >>>
    >>> runner = DataQualityGateRunner()
    >>> results = runner.run_all_gates(
    ...     inventory_data=inventory,
    ...     cv_data=calorific_values,
    ...     ef_data=emission_factors,
    ...     price_data=prices
    ... )
    >>> is_blocked, blockers = runner.is_blocked(results)
    >>> if is_blocked:
    ...     raise DataQualityError(f"Blocked by: {blockers}")

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from .data_quality_gates import (
    GateStatus,
    FallbackAction,
    DataCategory,
    GateViolation,
    QualityGateResult,
    CriticalFeedValidator,
    TimelinessValidator,
    CompletenessValidator,
    OutlierDetector,
    DataQualityGateRunner,
)

__all__ = [
    "GateStatus",
    "FallbackAction",
    "DataCategory",
    "GateViolation",
    "QualityGateResult",
    "CriticalFeedValidator",
    "TimelinessValidator",
    "CompletenessValidator",
    "OutlierDetector",
    "DataQualityGateRunner",
]

__version__ = "1.0.0"
