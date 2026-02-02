"""
GL-004 BURNMASTER - Climate Intelligence Module

This module provides emissions reporting and regulatory compliance for combustion
operations, supporting multiple reporting frameworks and jurisdictions.

Components:
    - EmissionsReporter: Core emissions aggregation and reporting engine
    - GHGProtocol: GHG Protocol Scope 1 direct emissions calculations
    - EPAReporting: EPA 40 CFR Part 98 Subpart C reporting
    - EUETSReporting: EU Emissions Trading System compliance
    - TCFDReporting: Task Force on Climate-related Financial Disclosures

Supported Standards:
    - GHG Protocol Corporate Standard (Scope 1)
    - ISO 14064-1:2018 Greenhouse gas inventories
    - EPA 40 CFR Part 98 Subpart C (Stationary Combustion)
    - EU ETS MRR (Monitoring and Reporting Regulation)
    - TCFD Climate-related Financial Disclosures

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from gl_004_burnmaster.climate.emissions_reporter import (
    EmissionsReporter,
    EmissionsReport,
    EmissionRecord,
    ReportingPeriod,
)
from gl_004_burnmaster.climate.ghg_protocol import (
    GHGProtocolCalculator,
    Scope1Emissions,
    EmissionFactor,
    FuelCategory,
)
from gl_004_burnmaster.climate.epa_reporting import (
    EPAReporter,
    SubpartCReport,
    TierMethodology,
)
from gl_004_burnmaster.climate.eu_ets import (
    EUETSReporter,
    ETSReport,
    InstallationCategory,
)
from gl_004_burnmaster.climate.tcfd_reporting import (
    TCFDReporter,
    TCFDMetrics,
    ClimateRiskCategory,
)

__all__ = [
    # Core Reporter
    "EmissionsReporter",
    "EmissionsReport",
    "EmissionRecord",
    "ReportingPeriod",
    # GHG Protocol
    "GHGProtocolCalculator",
    "Scope1Emissions",
    "EmissionFactor",
    "FuelCategory",
    # EPA Reporting
    "EPAReporter",
    "SubpartCReport",
    "TierMethodology",
    # EU ETS
    "EUETSReporter",
    "ETSReport",
    "InstallationCategory",
    # TCFD
    "TCFDReporter",
    "TCFDMetrics",
    "ClimateRiskCategory",
]

__version__ = "1.0.0"
__author__ = "GreenLang Combustion Systems Team"
