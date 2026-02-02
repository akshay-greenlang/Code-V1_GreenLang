"""
EPA Compliance Module - EPA regulatory standards and compliance checking.

This package implements compliance verification for EPA standards including:
    - 40 CFR Part 60 NSPS (New Source Performance Standards)
    - 40 CFR Part 98 GHG Reporting
    - EPA Method 19 Emission Rate Measurement
    - EPA AP-42 Emission Factors

Features:
    - Multi-subpart compliance checking (D, Db, Dc, J)
    - F-factor calculations for emission normalization
    - Actual vs. permit limit comparison
    - Compliance margin analysis
    - SHA-256 provenance tracking

Example:
    >>> from greenlang.compliance.epa import NSPSComplianceChecker, FuelType, FacilityData
    >>> checker = NSPSComplianceChecker()
    >>> facility = FacilityData(
    ...     facility_id="FAC-001",
    ...     equipment_id="BOILER-001",
    ...     boiler_type="fossil_fuel_steam",
    ...     fuel_type="natural_gas",
    ...     heat_input_mmbtu_hr=150.0,
    ... )
    >>> emissions = EmissionsData(
    ...     so2_ppm=10.0,
    ...     nox_ppm=35.0,
    ...     o2_pct=3.5,
    ... )
    >>> result = checker.check_subpart_d(facility, emissions)
"""

from greenlang.compliance.epa.part60_nsps import (
    NSPSComplianceChecker,
    FuelType,
    BoilerType,
    ComplianceStatus,
    EmissionsData,
    FacilityData,
    ComplianceResult,
    FFactorCalculator,
)

__all__ = [
    "NSPSComplianceChecker",
    "FuelType",
    "BoilerType",
    "ComplianceStatus",
    "EmissionsData",
    "FacilityData",
    "ComplianceResult",
    "FFactorCalculator",
]
