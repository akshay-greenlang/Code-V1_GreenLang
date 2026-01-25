# -*- coding: utf-8 -*-
"""
Scope3ReportingAgent
GL-VCCI Scope 3 Platform

Multi-standard sustainability reporting agent for Scope 3 emissions.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

from .agent import Scope3ReportingAgent
from .models import (
    CompanyInfo,
    EmissionsData,
    EnergyData,
    IntensityMetrics,
    RisksOpportunities,
    TransportData,
    ReportResult,
    ValidationResult,
)
from .config import ReportStandard, ExportFormat, ValidationLevel
from .exceptions import ReportingError, ValidationError, StandardComplianceError

__version__ = "1.0.0"

__all__ = [
    "Scope3ReportingAgent",
    "CompanyInfo",
    "EmissionsData",
    "EnergyData",
    "IntensityMetrics",
    "RisksOpportunities",
    "TransportData",
    "ReportResult",
    "ValidationResult",
    "ReportStandard",
    "ExportFormat",
    "ValidationLevel",
    "ReportingError",
    "ValidationError",
    "StandardComplianceError",
]
