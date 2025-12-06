"""
GreenLang Compliance Module - Regulatory compliance checking and reporting.

This package provides compliance verification engines for various regulatory
frameworks including EPA NSPS standards, EU regulations (EUDR, CSRD, Taxonomy),
and other environmental compliance requirements.

Subpackages:
    epa: EPA regulatory compliance (Part 60 NSPS, etc.)
    eu: EU regulatory compliance (EUDR, CSRD, Taxonomy, etc.)

Example:
    >>> from greenlang.compliance.epa import NSPSComplianceChecker
    >>> checker = NSPSComplianceChecker()
    >>> result = checker.check_subpart_d(facility_data, emissions_data)
"""

__all__ = [
    "NSPSComplianceChecker",
]
