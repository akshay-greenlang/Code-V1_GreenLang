# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - RATA (Relative Accuracy Test Audit) Module

Production-grade RATA test management and calculations per EPA 40 CFR Part 75.

Standards Compliance:
    - EPA 40 CFR Part 75 Appendix A: RATA test procedures
    - EPA 40 CFR Part 75 Appendix B: Relative Accuracy requirements
    - EPA Method 7E: NOx Reference Method
    - EPA Method 6C: SO2 Reference Method

Zero-Hallucination Principle:
    - All RATA calculations use deterministic EPA formulas
    - Complete calculation traces with provenance tracking
    - SHA-256 hashes for audit trail integrity

Example:
    >>> from rata import RATATest, RATARun, RATAResult
    >>> from calculators.rata_calculator import perform_rata
    >>> # Perform RATA calculation
    >>> result = perform_rata(cems_values, rm_values, test_type="standard")

Author: GreenLang GL-010 EmissionsGuardian
Version: 1.0.0
"""

from .schemas import (
    # Enums
    Pollutant,
    RATATestType,
    RATAStatus,
    ReferenceMethod,
    LoadLevel,
    PassFailStatus,
    CalibrationGasLevel,
    # Data Models
    RATATest,
    RATARun,
    RATAResult,
    BiasTestResult,
    CylinderGasAudit,
)

__all__ = [
    # Enums
    "Pollutant",
    "RATATestType",
    "RATAStatus",
    "ReferenceMethod",
    "LoadLevel",
    "PassFailStatus",
    "CalibrationGasLevel",
    # Data Models
    "RATATest",
    "RATARun",
    "RATAResult",
    "BiasTestResult",
    "CylinderGasAudit",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EmissionsGuardian"
