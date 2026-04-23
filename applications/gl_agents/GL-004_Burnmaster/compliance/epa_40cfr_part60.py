# -*- coding: utf-8 -*-
"""
EPA 40 CFR Part 60 - New Source Performance Standards (NSPS) Compliance Validator
==================================================================================

This module implements comprehensive compliance validation for EPA 40 CFR Part 60
New Source Performance Standards (NSPS) for steam generating units.

Regulatory Coverage:
    - Subpart D:  Fossil-Fuel-Fired Steam Generators (>250 MMBtu/hr)
                  40 CFR 60.40 - 60.46 (promulgated 12/23/1971)
    - Subpart Da: Electric Utility Steam Generating Units (>250 MMBtu/hr)
                  40 CFR 60.40Da - 60.52Da (promulgated 9/18/1978)
    - Subpart Db: Industrial-Commercial-Institutional Steam Generating Units (>100 MMBtu/hr)
                  40 CFR 60.40b - 60.49b (promulgated 6/19/1984)
    - Subpart Dc: Small Industrial-Commercial-Institutional Steam Generating Units (10-100 MMBtu/hr)
                  40 CFR 60.40c - 60.48c (promulgated 9/12/1990)

Key Features:
    - Emission limits by subpart and fuel type (lb/MMBtu)
    - CEMS monitoring requirements
    - Fuel analysis and recordkeeping requirements
    - Reporting schedules and deviation tracking
    - Exemption and applicability determination
    - Startup/shutdown/malfunction provisions

Reference Documents:
    - 40 CFR Part 60 Subparts D, Da, Db, Dc
    - EPA Method 19 - SO2 Removal and PM, SO2, NOx Emission Rates
    - EPA Performance Specification 2 (SO2/NOx CEMS)
    - EPA Performance Specification 4A (CO CEMS)

Author: GL-RegulatoryIntelligence
Version: 1.0.0
Last Updated: 2025-01-01

Disclaimer:
    This module provides compliance assistance but does not constitute legal advice.
    Always consult the actual CFR text and EPA guidance for regulatory requirements.
    Facilities should work with qualified environmental professionals for compliance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4


# =============================================================================
# ENUMS - REGULATORY CLASSIFICATIONS
# =============================================================================

class NSPSSubpart(str, Enum):
    """
    NSPS Subparts for Steam Generating Units.

    Reference: 40 CFR Part 60 Subparts D, Da, Db, Dc
    """
    SUBPART_D = "D"      # Fossil-fuel-fired steam generators >250 MMBtu/hr (1971)
    SUBPART_Da = "Da"    # Electric utility steam generating units (1978)
    SUBPART_Db = "Db"    # Industrial-commercial-institutional >100 MMBtu/hr (1984)
    SUBPART_Dc = "Dc"    # Small ICI units 10-100 MMBtu/hr (1990)
    NOT_APPLICABLE = "N/A"


class FuelCategory(str, Enum):
    """
    Fuel categories as defined in 40 CFR Part 60.

    Reference: 40 CFR 60.41b, 60.41c definitions
    """
    NATURAL_GAS = "natural_gas"
    DISTILLATE_OIL = "distillate_oil"      # No. 1 and No. 2 fuel oil
    RESIDUAL_OIL = "residual_oil"          # No. 4, 5, 6 fuel oil
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    COAL_LIGNITE = "coal_lignite"
    COAL_ANTHRACITE = "coal_anthracite"
    WOOD = "wood"
    MUNICIPAL_SOLID_WASTE = "msw"
    OTHER_SOLID = "other_solid"
    OTHER_GAS = "other_gas"
    MIXED = "mixed"


class PollutantType(str, Enum):
    """Regulated pollutants under NSPS."""
    NOX = "NOx"         # Nitrogen Oxides (as NO2)
    SO2 = "SO2"         # Sulfur Dioxide
    PM = "PM"           # Particulate Matter (filterable)
    PM_TOTAL = "PM_total"  # Total PM (filterable + condensable)
    OPACITY = "opacity"    # Visible emissions (%)


class ComplianceStatus(str, Enum):
    """Compliance determination status."""
    COMPLIANT = "compliant"
    WARNING = "warning"                  # 80-100% of limit
    EXCEEDANCE = "exceedance"            # Above limit
    CRITICAL = "critical"                # >150% of limit
    EXEMPT = "exempt"                    # Regulatory exemption applies
    NOT_APPLICABLE = "not_applicable"    # Pollutant not regulated for this subpart/fuel
    PENDING_DATA = "pending_data"        # Insufficient data for determination


class MonitoringMethod(str, Enum):
    """
    Emissions monitoring methods per 40 CFR Part 60.

    Reference: 40 CFR 60.13, 60.46b, 60.46c
    """
    CEMS = "cems"                                    # Continuous Emissions Monitoring System
    FUEL_ANALYSIS = "fuel_analysis"                  # Periodic fuel sampling
    STACK_TEST = "stack_test"                        # Performance testing (Reference Methods)
    PREDICTIVE_EMISSIONS_MONITORING = "pems"         # PEMS (40 CFR 60.13(i))
    PARAMETER_MONITORING = "parameter_monitoring"    # Operating parameter monitoring
    EMISSION_FACTOR = "emission_factor"              # AP-42 factors (limited applicability)


class ExemptionType(str, Enum):
    """Types of regulatory exemptions under NSPS."""
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    MALFUNCTION = "malfunction"
    FUEL_SWITCHING = "fuel_switching"
    EMERGENCY = "emergency"
    CONSTRUCTION_DATE = "construction_date"
    CAPACITY_BELOW_THRESHOLD = "capacity_below_threshold"
    FUEL_TYPE_EXEMPTION = "fuel_type_exemption"
    TEMPORARY_FUEL = "temporary_fuel"
    TESTING = "testing"


class ReportingPeriod(str, Enum):
    """Reporting period requirements."""
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    EVENT_BASED = "event_based"


# =============================================================================
# EMISSION LIMITS - 40 CFR PART 60 STANDARDS
# =============================================================================

# Subpart Db Emission Limits (lb/MMBtu)
# Reference: 40 CFR 60.42b, 60.43b, 60.44b
# Applicable to units >100 MMBtu/hr constructed/modified after 6/19/1984

NSPS_LIMITS_SUBPART_Db: Dict[str, Dict[str, float]] = {
    # NOx limits by fuel type (40 CFR 60.44b)
    # For units constructed/modified after 7/9/1997
    "NOx": {
        "natural_gas": 0.10,                # 40 CFR 60.44b(a)(1)
        "distillate_oil": 0.20,             # 40 CFR 60.44b(a)(2)
        "residual_oil": 0.30,               # 40 CFR 60.44b(a)(3)
        "coal_bituminous": 0.50,            # 40 CFR 60.44b(a)(4)
        "coal_subbituminous": 0.50,         # 40 CFR 60.44b(a)(4)
        "coal_lignite": 0.60,               # 40 CFR 60.44b(a)(5)
        "wood": 0.30,                       # 40 CFR 60.44b(a)(6)
        "msw": 0.30,                        # Municipal solid waste
    },
    # SO2 limits (40 CFR 60.42b)
    # Emission rate OR 90% reduction
    "SO2": {
        "natural_gas": 0.00,                # Exempt (negligible sulfur)
        "distillate_oil": 0.50,             # 40 CFR 60.42b(a)
        "residual_oil": 1.20,               # 40 CFR 60.42b(a)
        "coal_bituminous": 1.20,            # 40 CFR 60.42b(a)
        "coal_subbituminous": 1.20,         # 40 CFR 60.42b(a)
        "coal_lignite": 1.20,               # 40 CFR 60.42b(a)
        "wood": 0.00,                       # Exempt (negligible sulfur)
        "msw": 1.20,
    },
    # PM limits (40 CFR 60.43b)
    "PM": {
        "natural_gas": 0.00,                # Exempt (40 CFR 60.43b(h))
        "distillate_oil": 0.03,             # 40 CFR 60.43b(a)(3)
        "residual_oil": 0.05,               # 40 CFR 60.43b(a)(1)
        "coal_bituminous": 0.05,            # 40 CFR 60.43b(a)(1)
        "coal_subbituminous": 0.05,         # 40 CFR 60.43b(a)(1)
        "coal_lignite": 0.07,               # Higher due to ash content
        "wood": 0.10,                       # 40 CFR 60.43b(a)(2)
        "msw": 0.10,
    },
}

# Subpart Dc Emission Limits (lb/MMBtu)
# Reference: 40 CFR 60.42c, 60.43c, 60.44c
# Applicable to units 10-100 MMBtu/hr constructed/modified after 6/9/1989

NSPS_LIMITS_SUBPART_Dc: Dict[str, Dict[str, float]] = {
    # NOx limits - Subpart Dc generally does not have NOx limits
    # Exception: Some state requirements may apply
    "NOx": {
        "natural_gas": None,                # No federal NOx limit
        "distillate_oil": None,
        "residual_oil": None,
        "coal_bituminous": None,
        "coal_subbituminous": None,
        "wood": None,
    },
    # SO2 limits (40 CFR 60.42c)
    "SO2": {
        "natural_gas": 0.00,                # Exempt
        "distillate_oil": 0.50,             # 40 CFR 60.42c(a)
        "residual_oil": 0.50,               # Same as distillate
        "coal_bituminous": 1.20,            # 40 CFR 60.42c(a)
        "coal_subbituminous": 1.20,
        "wood": 0.00,                       # Exempt
    },
    # PM limits (40 CFR 60.43c)
    # Opacity limit of 20% applies per 40 CFR 60.43c(b)
    "PM": {
        "natural_gas": 0.00,                # Exempt
        "distillate_oil": 0.10,             # 40 CFR 60.43c(a)(2)
        "residual_oil": 0.10,
        "coal_bituminous": 0.05,            # 40 CFR 60.43c(a)(1)
        "coal_subbituminous": 0.05,
        "wood": 0.10,
    },
    # Opacity limit applies to all fuel types
    "opacity": {
        "all": 20.0,                        # 20% opacity limit (40 CFR 60.43c(b))
    },
}

# Subpart D Emission Limits (lb/MMBtu)
# Reference: 40 CFR 60.42, 60.43, 60.44
# Original 1971 standards for units >250 MMBtu/hr

NSPS_LIMITS_SUBPART_D: Dict[str, Dict[str, float]] = {
    "PM": {
        "coal": 0.10,                       # 40 CFR 60.42(a)(1)
        "oil": 0.10,                        # 40 CFR 60.42(a)(2)
        "natural_gas": 0.10,                # Same standard applies
        "combined": 0.10,
    },
    "SO2": {
        "coal": 1.20,                       # 40 CFR 60.43(a)(1)
        "oil": 0.80,                        # 40 CFR 60.43(a)(2)
        "natural_gas": 0.00,                # Exempt
    },
    "NOx": {
        "coal": 0.70,                       # 40 CFR 60.44(a)(1)
        "oil": 0.30,                        # 40 CFR 60.44(a)(2)
        "natural_gas": 0.20,                # 40 CFR 60.44(a)(3)
    },
    "opacity": {
        "all": 20.0,                        # 40 CFR 60.42(b)
    },
}

# Subpart Da Emission Limits (lb/MMBtu)
# Reference: 40 CFR 60.42Da, 60.43Da, 60.44Da
# Electric utility units >250 MMBtu/hr

NSPS_LIMITS_SUBPART_Da: Dict[str, Dict[str, float]] = {
    "PM": {
        "coal": 0.03,                       # 40 CFR 60.42Da(a)(1)
        "oil": 0.03,                        # 40 CFR 60.42Da(a)(1)
        "natural_gas": 0.03,
    },
    "SO2": {
        # 95% reduction OR 0.15 lb/MMBtu output limit
        "coal": 1.20,                       # Input-based; OR 90% reduction
        "oil": 0.80,
        "natural_gas": 0.00,                # Exempt
    },
    "NOx": {
        "coal_bituminous": 0.60,            # 40 CFR 60.44Da(a)(1)
        "coal_subbituminous": 0.50,         # 40 CFR 60.44Da(a)(1)
        "coal_lignite": 0.50,
        "oil": 0.30,                        # 40 CFR 60.44Da(b)
        "natural_gas": 0.20,                # 40 CFR 60.44Da(c)
    },
}


# =============================================================================
# APPLICABILITY DATES
# =============================================================================

SUBPART_APPLICABILITY_DATES: Dict[NSPSSubpart, Dict[str, date]] = {
    NSPSSubpart.SUBPART_D: {
        "promulgation_date": date(1971, 12, 23),
        "effective_date": date(1971, 8, 17),
    },
    NSPSSubpart.SUBPART_Da: {
        "promulgation_date": date(1978, 9, 18),
        "effective_date": date(1978, 9, 18),
        "2005_amendments": date(2005, 2, 28),
        "2012_amendments": date(2012, 2, 16),
    },
    NSPSSubpart.SUBPART_Db: {
        "promulgation_date": date(1984, 6, 19),
        "effective_date": date(1984, 6, 19),
        "1997_amendments": date(1997, 7, 9),
        "2006_amendments": date(2006, 2, 1),
    },
    NSPSSubpart.SUBPART_Dc: {
        "promulgation_date": date(1990, 9, 12),
        "effective_date": date(1989, 6, 9),
        "2006_amendments": date(2006, 2, 1),
    },
}


# =============================================================================
# CEMS AND MONITORING REQUIREMENTS
# =============================================================================

@dataclass
class CEMSRequirement:
    """
    CEMS requirements per 40 CFR 60.13 and subpart-specific provisions.

    Reference:
        - 40 CFR 60.13 (General monitoring requirements)
        - 40 CFR 60.46b (Subpart Db compliance provisions)
        - 40 CFR 60.46c (Subpart Dc compliance provisions)
        - Performance Specification 2 (SO2, NOx monitors)
        - Performance Specification 4A (CO monitors)
    """
    pollutant: PollutantType
    required: bool
    capacity_threshold_mmbtu_hr: float
    fuel_types_requiring_cems: List[FuelCategory]
    alternative_monitoring_allowed: bool
    quality_assurance_requirements: str
    minimum_data_availability: float  # Percentage
    averaging_period_hours: float
    reference_method: str
    performance_specification: str
    cfr_reference: str


CEMS_REQUIREMENTS_BY_SUBPART: Dict[NSPSSubpart, List[CEMSRequirement]] = {
    NSPSSubpart.SUBPART_Db: [
        CEMSRequirement(
            pollutant=PollutantType.SO2,
            required=True,
            capacity_threshold_mmbtu_hr=100.0,
            fuel_types_requiring_cems=[
                FuelCategory.COAL_BITUMINOUS,
                FuelCategory.COAL_SUBBITUMINOUS,
                FuelCategory.RESIDUAL_OIL,
            ],
            alternative_monitoring_allowed=True,  # Fuel sulfur monitoring per 60.46b(e)
            quality_assurance_requirements="40 CFR 60, Appendix F, Procedure 1",
            minimum_data_availability=95.0,
            averaging_period_hours=24.0,
            reference_method="EPA Method 6C",
            performance_specification="PS-2",
            cfr_reference="40 CFR 60.46b(a)",
        ),
        CEMSRequirement(
            pollutant=PollutantType.NOX,
            required=True,
            capacity_threshold_mmbtu_hr=100.0,
            fuel_types_requiring_cems=[
                FuelCategory.COAL_BITUMINOUS,
                FuelCategory.COAL_SUBBITUMINOUS,
                FuelCategory.RESIDUAL_OIL,
                FuelCategory.NATURAL_GAS,  # If >250 MMBtu/hr
            ],
            alternative_monitoring_allowed=False,
            quality_assurance_requirements="40 CFR 60, Appendix F, Procedure 1",
            minimum_data_availability=95.0,
            averaging_period_hours=30.0 * 24.0,  # 30-day rolling average
            reference_method="EPA Method 7E",
            performance_specification="PS-2",
            cfr_reference="40 CFR 60.46b(c)",
        ),
        CEMSRequirement(
            pollutant=PollutantType.OPACITY,
            required=True,
            capacity_threshold_mmbtu_hr=100.0,
            fuel_types_requiring_cems=[
                FuelCategory.COAL_BITUMINOUS,
                FuelCategory.COAL_SUBBITUMINOUS,
            ],
            alternative_monitoring_allowed=True,  # Bag leak detection for fabric filters
            quality_assurance_requirements="40 CFR 60, Appendix F, Procedure 3",
            minimum_data_availability=90.0,
            averaging_period_hours=6.0,  # 6-minute averages
            reference_method="EPA Method 9",
            performance_specification="PS-1",
            cfr_reference="40 CFR 60.46b(b)",
        ),
    ],
    NSPSSubpart.SUBPART_Dc: [
        CEMSRequirement(
            pollutant=PollutantType.OPACITY,
            required=True,
            capacity_threshold_mmbtu_hr=30.0,
            fuel_types_requiring_cems=[FuelCategory.COAL_BITUMINOUS],
            alternative_monitoring_allowed=True,
            quality_assurance_requirements="40 CFR 60, Appendix F, Procedure 3",
            minimum_data_availability=90.0,
            averaging_period_hours=6.0,
            reference_method="EPA Method 9",
            performance_specification="PS-1",
            cfr_reference="40 CFR 60.46c(a)",
        ),
    ],
    NSPSSubpart.SUBPART_D: [
        CEMSRequirement(
            pollutant=PollutantType.SO2,
            required=True,
            capacity_threshold_mmbtu_hr=250.0,
            fuel_types_requiring_cems=[
                FuelCategory.COAL_BITUMINOUS,
                FuelCategory.RESIDUAL_OIL,
            ],
            alternative_monitoring_allowed=True,
            quality_assurance_requirements="40 CFR 60, Appendix F",
            minimum_data_availability=90.0,
            averaging_period_hours=24.0,
            reference_method="EPA Method 6",
            performance_specification="PS-2",
            cfr_reference="40 CFR 60.45(a)",
        ),
    ],
    NSPSSubpart.SUBPART_Da: [
        CEMSRequirement(
            pollutant=PollutantType.SO2,
            required=True,
            capacity_threshold_mmbtu_hr=250.0,
            fuel_types_requiring_cems=[
                FuelCategory.COAL_BITUMINOUS,
                FuelCategory.COAL_SUBBITUMINOUS,
                FuelCategory.RESIDUAL_OIL,
            ],
            alternative_monitoring_allowed=False,
            quality_assurance_requirements="40 CFR 60, Appendix F, Procedure 1",
            minimum_data_availability=95.0,
            averaging_period_hours=24.0,
            reference_method="EPA Method 6C",
            performance_specification="PS-2",
            cfr_reference="40 CFR 60.49Da(a)",
        ),
        CEMSRequirement(
            pollutant=PollutantType.NOX,
            required=True,
            capacity_threshold_mmbtu_hr=250.0,
            fuel_types_requiring_cems=[
                FuelCategory.COAL_BITUMINOUS,
                FuelCategory.COAL_SUBBITUMINOUS,
                FuelCategory.RESIDUAL_OIL,
                FuelCategory.NATURAL_GAS,
            ],
            alternative_monitoring_allowed=False,
            quality_assurance_requirements="40 CFR 60, Appendix F, Procedure 1",
            minimum_data_availability=95.0,
            averaging_period_hours=30.0 * 24.0,
            reference_method="EPA Method 7E",
            performance_specification="PS-2",
            cfr_reference="40 CFR 60.49Da(b)",
        ),
        CEMSRequirement(
            pollutant=PollutantType.PM,
            required=True,
            capacity_threshold_mmbtu_hr=250.0,
            fuel_types_requiring_cems=[
                FuelCategory.COAL_BITUMINOUS,
                FuelCategory.COAL_SUBBITUMINOUS,
            ],
            alternative_monitoring_allowed=True,
            quality_assurance_requirements="40 CFR 60, Appendix F",
            minimum_data_availability=95.0,
            averaging_period_hours=24.0,
            reference_method="EPA Method 5",
            performance_specification="PS-11",
            cfr_reference="40 CFR 60.49Da(c)",
        ),
    ],
}


# =============================================================================
# FUEL ANALYSIS REQUIREMENTS
# =============================================================================

@dataclass
class FuelAnalysisRequirement:
    """
    Fuel sampling and analysis requirements.

    Reference:
        - 40 CFR 60.46b(e) - Fuel analysis for SO2 monitoring
        - 40 CFR 60.46c(b) - Fuel analysis for Subpart Dc
        - ASTM D3176 - Coal ultimate analysis
        - ASTM D4294 - Sulfur in petroleum products
    """
    fuel_category: FuelCategory
    parameters: List[str]
    minimum_frequency: str
    astm_methods: List[str]
    recordkeeping_period_years: int
    cfr_reference: str


FUEL_ANALYSIS_REQUIREMENTS: Dict[FuelCategory, FuelAnalysisRequirement] = {
    FuelCategory.COAL_BITUMINOUS: FuelAnalysisRequirement(
        fuel_category=FuelCategory.COAL_BITUMINOUS,
        parameters=["sulfur_content_wt_pct", "heating_value_btu_lb", "ash_content_wt_pct", "moisture_wt_pct"],
        minimum_frequency="Weekly composite OR per shipment (whichever is more frequent)",
        astm_methods=["ASTM D3176", "ASTM D5865", "ASTM D3682"],
        recordkeeping_period_years=2,
        cfr_reference="40 CFR 60.46b(e)(1)",
    ),
    FuelCategory.COAL_SUBBITUMINOUS: FuelAnalysisRequirement(
        fuel_category=FuelCategory.COAL_SUBBITUMINOUS,
        parameters=["sulfur_content_wt_pct", "heating_value_btu_lb", "ash_content_wt_pct", "moisture_wt_pct"],
        minimum_frequency="Weekly composite OR per shipment",
        astm_methods=["ASTM D3176", "ASTM D5865"],
        recordkeeping_period_years=2,
        cfr_reference="40 CFR 60.46b(e)(1)",
    ),
    FuelCategory.RESIDUAL_OIL: FuelAnalysisRequirement(
        fuel_category=FuelCategory.RESIDUAL_OIL,
        parameters=["sulfur_content_wt_pct", "heating_value_btu_gal", "nitrogen_content_wt_pct"],
        minimum_frequency="Per delivery OR weekly composite",
        astm_methods=["ASTM D4294", "ASTM D240", "ASTM D5291"],
        recordkeeping_period_years=2,
        cfr_reference="40 CFR 60.46b(e)(2)",
    ),
    FuelCategory.DISTILLATE_OIL: FuelAnalysisRequirement(
        fuel_category=FuelCategory.DISTILLATE_OIL,
        parameters=["sulfur_content_wt_pct", "heating_value_btu_gal"],
        minimum_frequency="Per delivery (if sulfur content certification not available)",
        astm_methods=["ASTM D4294", "ASTM D240"],
        recordkeeping_period_years=2,
        cfr_reference="40 CFR 60.46b(e)(3)",
    ),
    FuelCategory.NATURAL_GAS: FuelAnalysisRequirement(
        fuel_category=FuelCategory.NATURAL_GAS,
        parameters=["heating_value_btu_scf"],
        minimum_frequency="Monthly (if not metered with BTU correction)",
        astm_methods=["ASTM D1945", "GPA 2261"],
        recordkeeping_period_years=2,
        cfr_reference="40 CFR 60.46b(e)(4)",
    ),
    FuelCategory.WOOD: FuelAnalysisRequirement(
        fuel_category=FuelCategory.WOOD,
        parameters=["moisture_wt_pct", "heating_value_btu_lb"],
        minimum_frequency="Weekly",
        astm_methods=["ASTM E871", "ASTM E711"],
        recordkeeping_period_years=2,
        cfr_reference="40 CFR 60.46b(e)(5)",
    ),
}


# =============================================================================
# RECORDKEEPING REQUIREMENTS
# =============================================================================

@dataclass
class RecordkeepingRequirement:
    """
    Recordkeeping requirements per 40 CFR Part 60.

    Reference:
        - 40 CFR 60.7 (General notification and recordkeeping)
        - 40 CFR 60.48b (Subpart Db recordkeeping)
        - 40 CFR 60.48c (Subpart Dc recordkeeping)
    """
    record_type: str
    description: str
    retention_period_years: int
    required_elements: List[str]
    electronic_submission_allowed: bool
    cfr_reference: str


RECORDKEEPING_REQUIREMENTS: Dict[str, RecordkeepingRequirement] = {
    "operating_log": RecordkeepingRequirement(
        record_type="operating_log",
        description="Daily log of steam generating unit operation",
        retention_period_years=2,
        required_elements=[
            "Date and time of operation",
            "Type and quantity of fuel fired",
            "Heat input rate (MMBtu/hr)",
            "Load (% of rated capacity)",
            "Startup, shutdown, and malfunction events",
        ],
        electronic_submission_allowed=True,
        cfr_reference="40 CFR 60.48b(a)",
    ),
    "emissions_data": RecordkeepingRequirement(
        record_type="emissions_data",
        description="CEMS data and emission calculations",
        retention_period_years=2,
        required_elements=[
            "Hourly emission rates (lb/MMBtu)",
            "24-hour and 30-day rolling averages",
            "Data substitution records",
            "Quality assurance records",
            "Calibration records",
        ],
        electronic_submission_allowed=True,
        cfr_reference="40 CFR 60.48b(b)",
    ),
    "fuel_analysis": RecordkeepingRequirement(
        record_type="fuel_analysis",
        description="Fuel sampling and analysis results",
        retention_period_years=2,
        required_elements=[
            "Date and time of sample collection",
            "Fuel supplier and shipment information",
            "Analytical results (sulfur, heating value, etc.)",
            "Laboratory identification",
            "Calculated emission rates based on fuel analysis",
        ],
        electronic_submission_allowed=True,
        cfr_reference="40 CFR 60.48b(d)",
    ),
    "excess_emissions": RecordkeepingRequirement(
        record_type="excess_emissions",
        description="Excess emissions and monitoring system performance",
        retention_period_years=2,
        required_elements=[
            "Date and time of deviation",
            "Emission rate during deviation",
            "Applicable standard exceeded",
            "Cause of excess emissions",
            "Corrective actions taken",
            "Duration of the deviation",
        ],
        electronic_submission_allowed=True,
        cfr_reference="40 CFR 60.48b(c)",
    ),
    "maintenance_records": RecordkeepingRequirement(
        record_type="maintenance_records",
        description="Equipment maintenance and inspection records",
        retention_period_years=2,
        required_elements=[
            "Date of maintenance activity",
            "Equipment serviced",
            "Nature of maintenance performed",
            "Parts replaced",
            "Person performing maintenance",
        ],
        electronic_submission_allowed=True,
        cfr_reference="40 CFR 60.7(b)",
    ),
}


# =============================================================================
# REPORTING REQUIREMENTS
# =============================================================================

@dataclass
class ReportingRequirement:
    """
    Reporting requirements per 40 CFR Part 60.

    Reference:
        - 40 CFR 60.7 (General notification and reporting)
        - 40 CFR 60.49b (Subpart Db reporting)
        - 40 CFR 60.48c(c) (Subpart Dc reporting)
    """
    report_type: str
    description: str
    frequency: ReportingPeriod
    due_date_description: str
    electronic_reporting_required: bool
    cedri_submission: bool  # Compliance and Emissions Data Reporting Interface
    required_content: List[str]
    cfr_reference: str


REPORTING_REQUIREMENTS: Dict[str, ReportingRequirement] = {
    "excess_emissions_report": ReportingRequirement(
        report_type="excess_emissions_report",
        description="Semi-annual excess emissions and monitoring system performance report",
        frequency=ReportingPeriod.SEMI_ANNUAL,
        due_date_description="Within 30 days after end of each 6-month period",
        electronic_reporting_required=True,
        cedri_submission=True,
        required_content=[
            "Total operating hours during reporting period",
            "Total hours of excess emissions by pollutant",
            "Percent of time in compliance",
            "Nature and cause of excess emissions",
            "Corrective actions taken",
            "CEMS downtime and data availability",
            "Results of any stack tests",
        ],
        cfr_reference="40 CFR 60.7(c)",
    ),
    "initial_notification": ReportingRequirement(
        report_type="initial_notification",
        description="Notification of construction, reconstruction, or modification",
        frequency=ReportingPeriod.EVENT_BASED,
        due_date_description="Within 30 days of commencing construction",
        electronic_reporting_required=False,
        cedri_submission=False,
        required_content=[
            "Facility name, address, and contact information",
            "Description of affected facility",
            "Rated capacity (MMBtu/hr)",
            "Expected startup date",
            "Fuel types to be fired",
            "Emission control equipment",
        ],
        cfr_reference="40 CFR 60.7(a)(1)",
    ),
    "startup_notification": ReportingRequirement(
        report_type="startup_notification",
        description="Notification of actual startup date",
        frequency=ReportingPeriod.EVENT_BASED,
        due_date_description="Within 15 days of actual startup",
        electronic_reporting_required=False,
        cedri_submission=False,
        required_content=[
            "Date of startup",
            "Description of unit",
            "Initial performance test schedule",
        ],
        cfr_reference="40 CFR 60.7(a)(3)",
    ),
    "performance_test_notification": ReportingRequirement(
        report_type="performance_test_notification",
        description="Notification of performance test",
        frequency=ReportingPeriod.EVENT_BASED,
        due_date_description="At least 30 days prior to test",
        electronic_reporting_required=False,
        cedri_submission=False,
        required_content=[
            "Date and time of test",
            "Test methods to be used",
            "Process conditions during test",
        ],
        cfr_reference="40 CFR 60.8(d)",
    ),
    "annual_compliance_certification": ReportingRequirement(
        report_type="annual_compliance_certification",
        description="Annual certification of compliance",
        frequency=ReportingPeriod.ANNUAL,
        due_date_description="By March 1 for previous calendar year",
        electronic_reporting_required=True,
        cedri_submission=True,
        required_content=[
            "Statement of compliance status",
            "Summary of excess emissions",
            "Deviation summary",
            "Fuel consumption summary",
            "Emission summary by pollutant",
        ],
        cfr_reference="40 CFR 60.49b(u)",
    ),
}


# =============================================================================
# DATA CLASSES FOR COMPLIANCE CHECKING
# =============================================================================

@dataclass
class UnitCharacteristics:
    """
    Characteristics of a steam generating unit for NSPS applicability.

    Reference: 40 CFR 60.41b, 60.41c definitions
    """
    unit_id: str
    unit_name: str
    rated_capacity_mmbtu_hr: float
    construction_date: date
    modification_date: Optional[date] = None
    reconstruction_date: Optional[date] = None
    primary_fuel: FuelCategory = FuelCategory.NATURAL_GAS
    secondary_fuels: List[FuelCategory] = field(default_factory=list)
    is_electric_utility: bool = False
    is_cogeneration: bool = False
    serves_institutional: bool = False
    control_devices: List[str] = field(default_factory=list)
    stack_height_ft: Optional[float] = None
    exit_gas_temp_f: Optional[float] = None


@dataclass
class EmissionMeasurement:
    """
    Measured emission data for compliance determination.

    All emission rates should be expressed as lb/MMBtu (heat input basis)
    at reference conditions (dry basis, reference O2 where applicable).
    """
    measurement_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    unit_id: str = ""
    pollutant: PollutantType = PollutantType.NOX
    measured_value_lb_mmbtu: float = 0.0
    averaging_period_hours: float = 1.0
    monitoring_method: MonitoringMethod = MonitoringMethod.CEMS
    reference_o2_pct: float = 3.0
    data_quality_flag: str = "valid"
    cems_data_availability_pct: Optional[float] = None
    stack_test_date: Optional[date] = None
    fuel_type_during_measurement: FuelCategory = FuelCategory.NATURAL_GAS
    heat_input_mmbtu: Optional[float] = None
    operating_load_pct: Optional[float] = None


@dataclass
class ExemptionRecord:
    """
    Record of an exemption period or event.

    Reference: 40 CFR 60.8(c), 63.6(f) SSM provisions
    """
    exemption_id: str = field(default_factory=lambda: str(uuid4()))
    unit_id: str = ""
    exemption_type: ExemptionType = ExemptionType.STARTUP
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_hours: Optional[float] = None
    reason: str = ""
    documented_by: str = ""
    approved: bool = False
    conditions: List[str] = field(default_factory=list)


@dataclass
class DeviationEvent:
    """
    Record of a deviation from applicable standards.

    Reference: 40 CFR 60.7(c) excess emissions reporting
    """
    deviation_id: str = field(default_factory=lambda: str(uuid4()))
    unit_id: str = ""
    pollutant: PollutantType = PollutantType.NOX
    standard_exceeded: str = ""
    applicable_limit_lb_mmbtu: float = 0.0
    measured_value_lb_mmbtu: float = 0.0
    exceedance_pct: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_hours: float = 0.0
    cause_category: str = ""
    cause_description: str = ""
    corrective_actions: List[str] = field(default_factory=list)
    preventive_actions: List[str] = field(default_factory=list)
    reported: bool = False
    report_date: Optional[date] = None


@dataclass
class ComplianceResult:
    """
    Result of NSPS compliance determination.
    """
    result_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    unit_id: str = ""
    applicable_subpart: NSPSSubpart = NSPSSubpart.NOT_APPLICABLE
    pollutant: PollutantType = PollutantType.NOX
    fuel_type: FuelCategory = FuelCategory.NATURAL_GAS
    compliance_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    applicable_limit_lb_mmbtu: Optional[float] = None
    measured_value_lb_mmbtu: Optional[float] = None
    percent_of_limit: Optional[float] = None
    margin_to_limit_lb_mmbtu: Optional[float] = None
    averaging_period_hours: float = 1.0
    data_quality: str = "valid"
    exemptions_applied: List[ExemptionType] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    regulatory_citations: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "result_id": self.result_id,
            "timestamp": self.timestamp.isoformat(),
            "unit_id": self.unit_id,
            "applicable_subpart": self.applicable_subpart.value,
            "pollutant": self.pollutant.value,
            "fuel_type": self.fuel_type.value,
            "compliance_status": self.compliance_status.value,
            "applicable_limit_lb_mmbtu": self.applicable_limit_lb_mmbtu,
            "measured_value_lb_mmbtu": self.measured_value_lb_mmbtu,
            "percent_of_limit": self.percent_of_limit,
            "margin_to_limit_lb_mmbtu": self.margin_to_limit_lb_mmbtu,
            "averaging_period_hours": self.averaging_period_hours,
            "data_quality": self.data_quality,
            "exemptions_applied": [e.value for e in self.exemptions_applied],
            "recommendations": self.recommendations,
            "regulatory_citations": self.regulatory_citations,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class NSPSComplianceReport:
    """
    Comprehensive NSPS compliance report for a unit.
    """
    report_id: str = field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = field(default_factory=datetime.now)
    reporting_period_start: datetime = field(default_factory=datetime.now)
    reporting_period_end: datetime = field(default_factory=datetime.now)
    unit_characteristics: Optional[UnitCharacteristics] = None
    applicable_subpart: NSPSSubpart = NSPSSubpart.NOT_APPLICABLE
    overall_compliance_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    compliance_results: List[ComplianceResult] = field(default_factory=list)
    deviation_events: List[DeviationEvent] = field(default_factory=list)
    exemption_records: List[ExemptionRecord] = field(default_factory=list)
    monitoring_summary: Dict[str, Any] = field(default_factory=dict)
    fuel_consumption_summary: Dict[str, float] = field(default_factory=dict)
    emission_totals: Dict[str, float] = field(default_factory=dict)
    cems_data_availability: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    certifying_official: str = ""
    certification_statement: str = ""


# =============================================================================
# NSPS COMPLIANCE VALIDATOR CLASS
# =============================================================================

class NSPSComplianceValidator:
    """
    EPA 40 CFR Part 60 NSPS Compliance Validator.

    This class provides comprehensive compliance validation for steam generating
    units subject to New Source Performance Standards (NSPS) under Subparts D,
    Da, Db, and Dc.

    Features:
        - Subpart applicability determination
        - Emission limit lookup by subpart, fuel type, and pollutant
        - CEMS and monitoring requirement identification
        - Fuel analysis requirement determination
        - Compliance status calculation
        - Deviation and exemption tracking
        - Compliance report generation

    Example:
        >>> validator = NSPSComplianceValidator()
        >>> unit = UnitCharacteristics(
        ...     unit_id="BLR-001",
        ...     unit_name="Boiler 1",
        ...     rated_capacity_mmbtu_hr=150.0,
        ...     construction_date=date(2020, 1, 1),
        ...     primary_fuel=FuelCategory.NATURAL_GAS,
        ... )
        >>> subpart = validator.determine_applicable_subpart(unit)
        >>> print(f"Applicable subpart: {subpart.value}")
        Applicable subpart: Db

    Reference:
        - 40 CFR Part 60 Subparts D, Da, Db, Dc
        - EPA NSPS General Provisions (40 CFR 60.1-60.19)
    """

    def __init__(self, precision: int = 4):
        """
        Initialize the NSPS Compliance Validator.

        Args:
            precision: Decimal precision for emission calculations (default: 4)
        """
        self.precision = precision
        self._quantize_str = '0.' + '0' * precision

        # Load emission limits
        self.limits_db = NSPS_LIMITS_SUBPART_Db
        self.limits_dc = NSPS_LIMITS_SUBPART_Dc
        self.limits_d = NSPS_LIMITS_SUBPART_D
        self.limits_da = NSPS_LIMITS_SUBPART_Da

        # Load requirements
        self.cems_requirements = CEMS_REQUIREMENTS_BY_SUBPART
        self.fuel_analysis_requirements = FUEL_ANALYSIS_REQUIREMENTS
        self.recordkeeping_requirements = RECORDKEEPING_REQUIREMENTS
        self.reporting_requirements = REPORTING_REQUIREMENTS
        self.applicability_dates = SUBPART_APPLICABILITY_DATES

    def _quantize(self, value: float) -> Decimal:
        """Apply precision rounding (ROUND_HALF_UP for regulatory compliance)."""
        return Decimal(str(value)).quantize(
            Decimal(self._quantize_str), rounding=ROUND_HALF_UP
        )

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # APPLICABILITY DETERMINATION
    # -------------------------------------------------------------------------

    def determine_applicable_subpart(
        self,
        unit: UnitCharacteristics,
    ) -> NSPSSubpart:
        """
        Determine the applicable NSPS subpart for a steam generating unit.

        Applicability Logic:
            1. Check if unit is an electric utility (Subpart Da)
            2. Check construction/modification date against applicability dates
            3. Determine capacity category (D, Db, Dc thresholds)

        Args:
            unit: Characteristics of the steam generating unit

        Returns:
            NSPSSubpart: The applicable NSPS subpart

        Reference:
            - 40 CFR 60.40b (Subpart Db applicability)
            - 40 CFR 60.40c (Subpart Dc applicability)
            - 40 CFR 60.40Da (Subpart Da applicability)

        Example:
            >>> unit = UnitCharacteristics(
            ...     unit_id="BLR-001",
            ...     rated_capacity_mmbtu_hr=150.0,
            ...     construction_date=date(2020, 1, 1),
            ...     is_electric_utility=False,
            ... )
            >>> subpart = validator.determine_applicable_subpart(unit)
            >>> print(subpart)  # NSPSSubpart.SUBPART_Db
        """
        capacity = unit.rated_capacity_mmbtu_hr
        construction = unit.construction_date
        modification = unit.modification_date or date.min

        # Use the most recent applicable date
        applicable_date = max(construction, modification)

        # Check for Subpart Da applicability (electric utilities)
        if unit.is_electric_utility and capacity > 250.0:
            da_effective = self.applicability_dates[NSPSSubpart.SUBPART_Da]["effective_date"]
            if applicable_date >= da_effective:
                return NSPSSubpart.SUBPART_Da

        # Check for Subpart Db applicability (>100 MMBtu/hr)
        if capacity > 100.0:
            db_effective = self.applicability_dates[NSPSSubpart.SUBPART_Db]["effective_date"]
            if applicable_date >= db_effective:
                return NSPSSubpart.SUBPART_Db

            # Check for original Subpart D (>250 MMBtu/hr, pre-1984)
            if capacity > 250.0:
                d_effective = self.applicability_dates[NSPSSubpart.SUBPART_D]["effective_date"]
                if applicable_date >= d_effective:
                    return NSPSSubpart.SUBPART_D

        # Check for Subpart Dc applicability (10-100 MMBtu/hr)
        if 10.0 <= capacity <= 100.0:
            dc_effective = self.applicability_dates[NSPSSubpart.SUBPART_Dc]["effective_date"]
            if applicable_date >= dc_effective:
                return NSPSSubpart.SUBPART_Dc

        # Unit is below thresholds or pre-dates NSPS
        return NSPSSubpart.NOT_APPLICABLE

    def check_construction_date_exemption(
        self,
        unit: UnitCharacteristics,
        subpart: NSPSSubpart,
    ) -> Tuple[bool, str]:
        """
        Check if unit is exempt based on construction/modification date.

        Args:
            unit: Unit characteristics
            subpart: Applicable subpart to check against

        Returns:
            Tuple of (is_exempt, reason)

        Reference: 40 CFR 60.1 definitions of "affected facility"
        """
        if subpart == NSPSSubpart.NOT_APPLICABLE:
            return True, "No NSPS subpart applicable"

        applicable_date = unit.construction_date
        if unit.modification_date and unit.modification_date > applicable_date:
            applicable_date = unit.modification_date

        subpart_dates = self.applicability_dates.get(subpart, {})
        effective_date = subpart_dates.get("effective_date")

        if effective_date and applicable_date < effective_date:
            return True, (
                f"Unit constructed/modified before {subpart.value} effective date "
                f"({effective_date.isoformat()})"
            )

        return False, ""

    # -------------------------------------------------------------------------
    # EMISSION LIMITS LOOKUP
    # -------------------------------------------------------------------------

    def get_emission_limit(
        self,
        subpart: NSPSSubpart,
        pollutant: PollutantType,
        fuel_type: FuelCategory,
    ) -> Optional[float]:
        """
        Get the applicable emission limit for a given subpart, pollutant, and fuel type.

        Args:
            subpart: The applicable NSPS subpart
            pollutant: The pollutant type (NOx, SO2, PM)
            fuel_type: The fuel category being fired

        Returns:
            Emission limit in lb/MMBtu, or None if not applicable

        Reference:
            - 40 CFR 60.42b, 60.43b, 60.44b (Subpart Db limits)
            - 40 CFR 60.42c, 60.43c (Subpart Dc limits)

        Example:
            >>> limit = validator.get_emission_limit(
            ...     NSPSSubpart.SUBPART_Db,
            ...     PollutantType.NOX,
            ...     FuelCategory.NATURAL_GAS,
            ... )
            >>> print(f"NOx limit: {limit} lb/MMBtu")
            NOx limit: 0.1 lb/MMBtu
        """
        # Select the appropriate limits dictionary
        if subpart == NSPSSubpart.SUBPART_Db:
            limits = self.limits_db
        elif subpart == NSPSSubpart.SUBPART_Dc:
            limits = self.limits_dc
        elif subpart == NSPSSubpart.SUBPART_D:
            limits = self.limits_d
        elif subpart == NSPSSubpart.SUBPART_Da:
            limits = self.limits_da
        else:
            return None

        # Get pollutant limits
        pollutant_limits = limits.get(pollutant.value, {})

        # Look up fuel-specific limit
        fuel_key = fuel_type.value

        # Handle fuel type mappings
        if fuel_key not in pollutant_limits:
            # Try broader fuel categories
            if fuel_type in [FuelCategory.COAL_BITUMINOUS, FuelCategory.COAL_SUBBITUMINOUS,
                            FuelCategory.COAL_LIGNITE, FuelCategory.COAL_ANTHRACITE]:
                for key in ["coal_bituminous", "coal_subbituminous", "coal"]:
                    if key in pollutant_limits:
                        return pollutant_limits[key]
            elif fuel_type in [FuelCategory.DISTILLATE_OIL, FuelCategory.RESIDUAL_OIL]:
                for key in ["distillate_oil", "residual_oil", "oil"]:
                    if key in pollutant_limits:
                        return pollutant_limits[key]
            elif "all" in pollutant_limits:
                return pollutant_limits["all"]
            return None

        return pollutant_limits[fuel_key]

    def get_all_limits_for_unit(
        self,
        subpart: NSPSSubpart,
        fuel_type: FuelCategory,
    ) -> Dict[str, Optional[float]]:
        """
        Get all applicable emission limits for a unit.

        Args:
            subpart: The applicable NSPS subpart
            fuel_type: The fuel category

        Returns:
            Dictionary of pollutant -> limit (lb/MMBtu)
        """
        pollutants = [PollutantType.NOX, PollutantType.SO2, PollutantType.PM, PollutantType.OPACITY]
        limits = {}

        for pollutant in pollutants:
            limits[pollutant.value] = self.get_emission_limit(subpart, pollutant, fuel_type)

        return limits

    # -------------------------------------------------------------------------
    # COMPLIANCE CHECKING
    # -------------------------------------------------------------------------

    def check_nsps_compliance(
        self,
        unit_capacity_mmbtu_hr: float,
        fuel_type: str,
        measured_nox_lb_mmbtu: float,
        measured_so2_lb_mmbtu: float,
        measured_pm_lb_mmbtu: float,
        unit_id: str = "UNIT-001",
        construction_date: Optional[date] = None,
        is_electric_utility: bool = False,
        exemptions: Optional[List[ExemptionType]] = None,
    ) -> ComplianceResult:
        """
        Check NSPS compliance for a steam generating unit.

        This is the primary compliance checking method that:
            1. Determines the applicable NSPS subpart
            2. Looks up applicable emission limits
            3. Compares measured values against limits
            4. Generates compliance status and recommendations

        Args:
            unit_capacity_mmbtu_hr: Rated heat input capacity (MMBtu/hr)
            fuel_type: Fuel type string (e.g., "natural_gas", "coal_bituminous")
            measured_nox_lb_mmbtu: Measured NOx emission rate (lb/MMBtu)
            measured_so2_lb_mmbtu: Measured SO2 emission rate (lb/MMBtu)
            measured_pm_lb_mmbtu: Measured PM emission rate (lb/MMBtu)
            unit_id: Unique identifier for the unit
            construction_date: Date of construction (for applicability)
            is_electric_utility: Whether unit is an electric utility
            exemptions: List of applicable exemptions

        Returns:
            ComplianceResult with status and details

        Example:
            >>> result = validator.check_nsps_compliance(
            ...     unit_capacity_mmbtu_hr=150.0,
            ...     fuel_type="natural_gas",
            ...     measured_nox_lb_mmbtu=0.08,
            ...     measured_so2_lb_mmbtu=0.0,
            ...     measured_pm_lb_mmbtu=0.0,
            ... )
            >>> print(f"Status: {result.compliance_status.value}")
            Status: compliant
        """
        exemptions = exemptions or []

        # Create unit characteristics
        unit = UnitCharacteristics(
            unit_id=unit_id,
            unit_name=unit_id,
            rated_capacity_mmbtu_hr=unit_capacity_mmbtu_hr,
            construction_date=construction_date or date(2020, 1, 1),
            primary_fuel=FuelCategory(fuel_type) if fuel_type in [f.value for f in FuelCategory] else FuelCategory.NATURAL_GAS,
            is_electric_utility=is_electric_utility,
        )

        # Determine applicable subpart
        subpart = self.determine_applicable_subpart(unit)

        # Check for construction date exemption
        is_exempt, exempt_reason = self.check_construction_date_exemption(unit, subpart)
        if is_exempt:
            return ComplianceResult(
                unit_id=unit_id,
                applicable_subpart=subpart,
                pollutant=PollutantType.NOX,
                fuel_type=unit.primary_fuel,
                compliance_status=ComplianceStatus.EXEMPT,
                recommendations=[exempt_reason],
                provenance_hash=self._compute_provenance_hash({"exempt": True, "reason": exempt_reason}),
            )

        # Get applicable limits
        nox_limit = self.get_emission_limit(subpart, PollutantType.NOX, unit.primary_fuel)
        so2_limit = self.get_emission_limit(subpart, PollutantType.SO2, unit.primary_fuel)
        pm_limit = self.get_emission_limit(subpart, PollutantType.PM, unit.primary_fuel)

        # Determine overall compliance status (worst case)
        statuses = []
        recommendations = []
        citations = []

        # Check NOx compliance
        if nox_limit is not None and nox_limit > 0:
            nox_pct = (measured_nox_lb_mmbtu / nox_limit) * 100
            nox_status = self._determine_status(nox_pct)
            statuses.append(nox_status)

            if nox_status != ComplianceStatus.COMPLIANT:
                recommendations.append(
                    f"NOx at {nox_pct:.1f}% of limit ({measured_nox_lb_mmbtu:.4f} vs {nox_limit:.4f} lb/MMBtu). "
                    f"Consider combustion optimization or SCR installation."
                )
            citations.append(f"40 CFR 60.44b (Subpart {subpart.value} NOx limit)")

        # Check SO2 compliance
        if so2_limit is not None and so2_limit > 0:
            so2_pct = (measured_so2_lb_mmbtu / so2_limit) * 100
            so2_status = self._determine_status(so2_pct)
            statuses.append(so2_status)

            if so2_status != ComplianceStatus.COMPLIANT:
                recommendations.append(
                    f"SO2 at {so2_pct:.1f}% of limit ({measured_so2_lb_mmbtu:.4f} vs {so2_limit:.4f} lb/MMBtu). "
                    f"Consider fuel switching or FGD installation."
                )
            citations.append(f"40 CFR 60.42b (Subpart {subpart.value} SO2 limit)")

        # Check PM compliance
        if pm_limit is not None and pm_limit > 0:
            pm_pct = (measured_pm_lb_mmbtu / pm_limit) * 100
            pm_status = self._determine_status(pm_pct)
            statuses.append(pm_status)

            if pm_status != ComplianceStatus.COMPLIANT:
                recommendations.append(
                    f"PM at {pm_pct:.1f}% of limit ({measured_pm_lb_mmbtu:.4f} vs {pm_limit:.4f} lb/MMBtu). "
                    f"Check particulate control equipment operation."
                )
            citations.append(f"40 CFR 60.43b (Subpart {subpart.value} PM limit)")

        # Determine worst-case status
        if not statuses:
            overall_status = ComplianceStatus.NOT_APPLICABLE
        else:
            status_priority = {
                ComplianceStatus.COMPLIANT: 0,
                ComplianceStatus.WARNING: 1,
                ComplianceStatus.EXCEEDANCE: 2,
                ComplianceStatus.CRITICAL: 3,
            }
            overall_status = max(statuses, key=lambda s: status_priority.get(s, 0))

        # Calculate margins for NOx (primary pollutant for reporting)
        measured_value = measured_nox_lb_mmbtu
        limit_value = nox_limit
        pct_of_limit = (measured_value / limit_value * 100) if limit_value and limit_value > 0 else None
        margin = (limit_value - measured_value) if limit_value else None

        # Generate provenance hash
        provenance = self._compute_provenance_hash({
            "unit_id": unit_id,
            "capacity": unit_capacity_mmbtu_hr,
            "fuel_type": fuel_type,
            "nox_measured": measured_nox_lb_mmbtu,
            "so2_measured": measured_so2_lb_mmbtu,
            "pm_measured": measured_pm_lb_mmbtu,
            "subpart": subpart.value,
            "status": overall_status.value,
        })

        return ComplianceResult(
            unit_id=unit_id,
            applicable_subpart=subpart,
            pollutant=PollutantType.NOX,
            fuel_type=unit.primary_fuel,
            compliance_status=overall_status,
            applicable_limit_lb_mmbtu=limit_value,
            measured_value_lb_mmbtu=measured_value,
            percent_of_limit=pct_of_limit,
            margin_to_limit_lb_mmbtu=margin,
            exemptions_applied=exemptions,
            recommendations=recommendations,
            regulatory_citations=citations,
            provenance_hash=provenance,
        )

    def _determine_status(self, percent_of_limit: float) -> ComplianceStatus:
        """Determine compliance status based on percentage of limit."""
        if percent_of_limit < 80.0:
            return ComplianceStatus.COMPLIANT
        elif percent_of_limit < 100.0:
            return ComplianceStatus.WARNING
        elif percent_of_limit < 150.0:
            return ComplianceStatus.EXCEEDANCE
        else:
            return ComplianceStatus.CRITICAL

    def check_pollutant_compliance(
        self,
        unit: UnitCharacteristics,
        measurement: EmissionMeasurement,
        exemptions: Optional[List[ExemptionRecord]] = None,
    ) -> ComplianceResult:
        """
        Check compliance for a specific pollutant measurement.

        Args:
            unit: Unit characteristics
            measurement: Emission measurement data
            exemptions: Applicable exemption records

        Returns:
            ComplianceResult for the specific pollutant
        """
        exemptions = exemptions or []

        # Determine applicable subpart
        subpart = self.determine_applicable_subpart(unit)

        # Get applicable limit
        limit = self.get_emission_limit(subpart, measurement.pollutant, measurement.fuel_type_during_measurement)

        # Handle exempt or not applicable cases
        if limit is None:
            return ComplianceResult(
                unit_id=unit.unit_id,
                applicable_subpart=subpart,
                pollutant=measurement.pollutant,
                fuel_type=measurement.fuel_type_during_measurement,
                compliance_status=ComplianceStatus.NOT_APPLICABLE,
                regulatory_citations=[
                    f"No {measurement.pollutant.value} limit applicable under Subpart {subpart.value} "
                    f"for {measurement.fuel_type_during_measurement.value}"
                ],
                provenance_hash=self._compute_provenance_hash({"not_applicable": True}),
            )

        # Check for active exemptions
        applicable_exemptions = []
        for exemption in exemptions:
            if exemption.unit_id == unit.unit_id and exemption.approved:
                if exemption.end_time is None or exemption.end_time > measurement.timestamp:
                    applicable_exemptions.append(exemption.exemption_type)

        if applicable_exemptions:
            return ComplianceResult(
                unit_id=unit.unit_id,
                applicable_subpart=subpart,
                pollutant=measurement.pollutant,
                fuel_type=measurement.fuel_type_during_measurement,
                compliance_status=ComplianceStatus.EXEMPT,
                applicable_limit_lb_mmbtu=limit,
                measured_value_lb_mmbtu=measurement.measured_value_lb_mmbtu,
                exemptions_applied=applicable_exemptions,
                regulatory_citations=["Exemption period applies - see startup/shutdown provisions"],
                provenance_hash=self._compute_provenance_hash({"exempt": True}),
            )

        # Calculate compliance
        percent_of_limit = (measurement.measured_value_lb_mmbtu / limit) * 100 if limit > 0 else 0
        status = self._determine_status(percent_of_limit)
        margin = limit - measurement.measured_value_lb_mmbtu

        # Generate recommendations
        recommendations = []
        if status != ComplianceStatus.COMPLIANT:
            recommendations = self._generate_recommendations(
                measurement.pollutant,
                status,
                percent_of_limit,
                measurement.fuel_type_during_measurement,
            )

        # Regulatory citations
        citations = self._get_regulatory_citations(subpart, measurement.pollutant)

        return ComplianceResult(
            unit_id=unit.unit_id,
            applicable_subpart=subpart,
            pollutant=measurement.pollutant,
            fuel_type=measurement.fuel_type_during_measurement,
            compliance_status=status,
            applicable_limit_lb_mmbtu=limit,
            measured_value_lb_mmbtu=measurement.measured_value_lb_mmbtu,
            percent_of_limit=percent_of_limit,
            margin_to_limit_lb_mmbtu=margin,
            averaging_period_hours=measurement.averaging_period_hours,
            data_quality=measurement.data_quality_flag,
            recommendations=recommendations,
            regulatory_citations=citations,
            provenance_hash=self._compute_provenance_hash({
                "unit_id": unit.unit_id,
                "pollutant": measurement.pollutant.value,
                "measured": measurement.measured_value_lb_mmbtu,
                "limit": limit,
                "status": status.value,
            }),
        )

    def _generate_recommendations(
        self,
        pollutant: PollutantType,
        status: ComplianceStatus,
        percent_of_limit: float,
        fuel_type: FuelCategory,
    ) -> List[str]:
        """Generate recommendations based on compliance status."""
        recommendations = []

        if pollutant == PollutantType.NOX:
            if status == ComplianceStatus.WARNING:
                recommendations.extend([
                    "Monitor NOx trends closely - approaching limit",
                    "Review combustion air distribution",
                    "Consider burner tune-up to optimize air/fuel ratio",
                ])
            elif status in [ComplianceStatus.EXCEEDANCE, ComplianceStatus.CRITICAL]:
                recommendations.extend([
                    "IMMEDIATE ACTION REQUIRED: NOx exceeds NSPS limit",
                    "Reduce load if possible to lower flame temperature",
                    "Check low-NOx burner operation and FGR system",
                    "Consider SCR/SNCR installation for long-term compliance",
                    "Document excess emission event per 40 CFR 60.7(c)",
                ])

        elif pollutant == PollutantType.SO2:
            if status == ComplianceStatus.WARNING:
                recommendations.extend([
                    "Monitor fuel sulfur content closely",
                    "Verify FGD scrubber efficiency",
                ])
            elif status in [ComplianceStatus.EXCEEDANCE, ComplianceStatus.CRITICAL]:
                recommendations.extend([
                    "IMMEDIATE ACTION: SO2 exceeds NSPS limit",
                    "Switch to lower sulfur fuel if available",
                    "Increase FGD reagent injection rate",
                    "Document deviation per 40 CFR 60.7(c)",
                ])

        elif pollutant == PollutantType.PM:
            if status == ComplianceStatus.WARNING:
                recommendations.extend([
                    "Check particulate control equipment operation",
                    "Monitor opacity trends",
                ])
            elif status in [ComplianceStatus.EXCEEDANCE, ComplianceStatus.CRITICAL]:
                recommendations.extend([
                    "IMMEDIATE ACTION: PM exceeds NSPS limit",
                    "Check baghouse/ESP performance",
                    "Verify ash removal system operation",
                    "Document deviation per 40 CFR 60.7(c)",
                ])

        return recommendations

    def _get_regulatory_citations(
        self,
        subpart: NSPSSubpart,
        pollutant: PollutantType,
    ) -> List[str]:
        """Get regulatory citations for the applicable standard."""
        citations = []

        if subpart == NSPSSubpart.SUBPART_Db:
            if pollutant == PollutantType.NOX:
                citations.append("40 CFR 60.44b - Standards for nitrogen oxides")
            elif pollutant == PollutantType.SO2:
                citations.append("40 CFR 60.42b - Standards for sulfur dioxide")
            elif pollutant == PollutantType.PM:
                citations.append("40 CFR 60.43b - Standards for particulate matter")
            citations.append("40 CFR 60.46b - Compliance and monitoring provisions")
            citations.append("40 CFR 60.48b - Recordkeeping and reporting")

        elif subpart == NSPSSubpart.SUBPART_Dc:
            if pollutant == PollutantType.SO2:
                citations.append("40 CFR 60.42c - Standards for sulfur dioxide")
            elif pollutant == PollutantType.PM:
                citations.append("40 CFR 60.43c - Standards for particulate matter")
            citations.append("40 CFR 60.46c - Compliance provisions")

        elif subpart == NSPSSubpart.SUBPART_D:
            citations.append(f"40 CFR 60.4{2 + ['PM', 'SO2', 'NOX'].index(pollutant.value) if pollutant.value in ['PM', 'SO2', 'NOX'] else 2}")

        elif subpart == NSPSSubpart.SUBPART_Da:
            citations.append(f"40 CFR 60.4{2 + ['PM', 'SO2', 'NOX'].index(pollutant.value) if pollutant.value in ['PM', 'SO2', 'NOX'] else 2}Da")

        return citations

    # -------------------------------------------------------------------------
    # MONITORING REQUIREMENTS
    # -------------------------------------------------------------------------

    def get_cems_requirements(
        self,
        unit: UnitCharacteristics,
    ) -> List[CEMSRequirement]:
        """
        Get CEMS requirements for a unit.

        Args:
            unit: Unit characteristics

        Returns:
            List of applicable CEMS requirements

        Reference: 40 CFR 60.13, 60.46b, 60.46c, 60.49Da
        """
        subpart = self.determine_applicable_subpart(unit)

        if subpart not in self.cems_requirements:
            return []

        applicable_reqs = []
        for req in self.cems_requirements[subpart]:
            # Check capacity threshold
            if unit.rated_capacity_mmbtu_hr >= req.capacity_threshold_mmbtu_hr:
                # Check fuel type
                if unit.primary_fuel in req.fuel_types_requiring_cems:
                    applicable_reqs.append(req)

        return applicable_reqs

    def get_fuel_analysis_requirements(
        self,
        unit: UnitCharacteristics,
    ) -> Optional[FuelAnalysisRequirement]:
        """
        Get fuel analysis requirements for a unit.

        Args:
            unit: Unit characteristics

        Returns:
            Applicable fuel analysis requirement, or None

        Reference: 40 CFR 60.46b(e), 60.46c(b)
        """
        return self.fuel_analysis_requirements.get(unit.primary_fuel)

    def get_monitoring_plan_requirements(
        self,
        unit: UnitCharacteristics,
    ) -> Dict[str, Any]:
        """
        Get comprehensive monitoring plan requirements.

        Args:
            unit: Unit characteristics

        Returns:
            Dictionary with monitoring requirements
        """
        subpart = self.determine_applicable_subpart(unit)
        cems_reqs = self.get_cems_requirements(unit)
        fuel_reqs = self.get_fuel_analysis_requirements(unit)

        return {
            "unit_id": unit.unit_id,
            "applicable_subpart": subpart.value,
            "cems_requirements": [
                {
                    "pollutant": req.pollutant.value,
                    "required": req.required,
                    "alternative_allowed": req.alternative_monitoring_allowed,
                    "performance_specification": req.performance_specification,
                    "reference_method": req.reference_method,
                    "data_availability_required": req.minimum_data_availability,
                    "averaging_period_hours": req.averaging_period_hours,
                    "cfr_reference": req.cfr_reference,
                }
                for req in cems_reqs
            ],
            "fuel_analysis_requirements": {
                "fuel_type": fuel_reqs.fuel_category.value if fuel_reqs else None,
                "parameters": fuel_reqs.parameters if fuel_reqs else [],
                "frequency": fuel_reqs.minimum_frequency if fuel_reqs else None,
                "astm_methods": fuel_reqs.astm_methods if fuel_reqs else [],
                "recordkeeping_years": fuel_reqs.recordkeeping_period_years if fuel_reqs else None,
            } if fuel_reqs else None,
            "recordkeeping_requirements": list(self.recordkeeping_requirements.keys()),
            "reporting_requirements": [
                {
                    "type": req.report_type,
                    "frequency": req.frequency.value,
                    "due_date": req.due_date_description,
                    "electronic_required": req.electronic_reporting_required,
                    "cedri_submission": req.cedri_submission,
                }
                for req in self.reporting_requirements.values()
            ],
        }

    # -------------------------------------------------------------------------
    # EXEMPTIONS AND SPECIAL PROVISIONS
    # -------------------------------------------------------------------------

    def check_startup_shutdown_exemption(
        self,
        unit: UnitCharacteristics,
        event_type: ExemptionType,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        duration_hours: Optional[float] = None,
    ) -> ExemptionRecord:
        """
        Create and validate a startup/shutdown exemption record.

        Reference:
            - 40 CFR 60.8(c) - Compliance during startup/shutdown
            - 40 CFR 60.48b(f) - Subpart Db provisions

        Args:
            unit: Unit characteristics
            event_type: Type of exemption (STARTUP, SHUTDOWN, MALFUNCTION)
            start_time: Event start time
            end_time: Event end time (if known)
            duration_hours: Event duration (alternative to end_time)

        Returns:
            ExemptionRecord with validation status
        """
        subpart = self.determine_applicable_subpart(unit)

        # Define maximum exemption durations by subpart and event type
        max_durations = {
            NSPSSubpart.SUBPART_Db: {
                ExemptionType.STARTUP: 4.0,   # 4 hours typical
                ExemptionType.SHUTDOWN: 2.0,  # 2 hours typical
                ExemptionType.MALFUNCTION: 24.0,  # Must be documented
            },
            NSPSSubpart.SUBPART_Dc: {
                ExemptionType.STARTUP: 4.0,
                ExemptionType.SHUTDOWN: 2.0,
                ExemptionType.MALFUNCTION: 24.0,
            },
        }

        max_duration = max_durations.get(subpart, {}).get(event_type, 4.0)

        # Calculate duration
        if duration_hours is None and end_time is not None:
            duration_hours = (end_time - start_time).total_seconds() / 3600.0

        # Determine approval status
        approved = True
        conditions = []

        if duration_hours and duration_hours > max_duration:
            approved = False
            conditions.append(f"Duration ({duration_hours:.1f}h) exceeds typical maximum ({max_duration:.1f}h)")
            conditions.append("Additional documentation and justification required")

        conditions.extend([
            "Must follow manufacturer startup/shutdown procedures",
            "Must minimize emissions during event",
            "Must document in operating log per 40 CFR 60.48b",
        ])

        return ExemptionRecord(
            unit_id=unit.unit_id,
            exemption_type=event_type,
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            reason=f"{event_type.value.capitalize()} event",
            approved=approved,
            conditions=conditions,
        )

    def check_fuel_exemption(
        self,
        unit: UnitCharacteristics,
        pollutant: PollutantType,
    ) -> Tuple[bool, str]:
        """
        Check if unit is exempt from a pollutant standard based on fuel type.

        Args:
            unit: Unit characteristics
            pollutant: Pollutant to check

        Returns:
            Tuple of (is_exempt, reason)

        Reference: 40 CFR 60.43b(h), 60.42b exemptions
        """
        subpart = self.determine_applicable_subpart(unit)
        fuel = unit.primary_fuel

        # Natural gas exemptions
        if fuel == FuelCategory.NATURAL_GAS:
            if pollutant == PollutantType.PM:
                return True, "Natural gas units exempt from PM standards per 40 CFR 60.43b(h)"
            if pollutant == PollutantType.SO2:
                return True, "Natural gas has negligible sulfur - SO2 limit effectively 0"

        # Wood and biomass may have different provisions
        if fuel == FuelCategory.WOOD and subpart == NSPSSubpart.SUBPART_Dc:
            if pollutant == PollutantType.SO2:
                return True, "Wood fuel has negligible sulfur content"

        return False, ""

    # -------------------------------------------------------------------------
    # DEVIATION AND EXCESS EMISSIONS TRACKING
    # -------------------------------------------------------------------------

    def record_deviation(
        self,
        unit: UnitCharacteristics,
        measurement: EmissionMeasurement,
        limit: float,
        cause: str,
        corrective_actions: List[str],
    ) -> DeviationEvent:
        """
        Record an excess emissions deviation event.

        Reference: 40 CFR 60.7(c) - Excess emissions reporting

        Args:
            unit: Unit characteristics
            measurement: The exceedance measurement
            limit: The applicable limit
            cause: Cause of the deviation
            corrective_actions: Actions taken to correct

        Returns:
            DeviationEvent record
        """
        exceedance_pct = ((measurement.measured_value_lb_mmbtu - limit) / limit) * 100 if limit > 0 else 0

        subpart = self.determine_applicable_subpart(unit)

        return DeviationEvent(
            unit_id=unit.unit_id,
            pollutant=measurement.pollutant,
            standard_exceeded=f"40 CFR 60.4{['2','3','4'][['SO2','PM','NOX'].index(measurement.pollutant.value)] if measurement.pollutant.value in ['SO2','PM','NOX'] else '4'}b" if subpart == NSPSSubpart.SUBPART_Db else f"Subpart {subpart.value}",
            applicable_limit_lb_mmbtu=limit,
            measured_value_lb_mmbtu=measurement.measured_value_lb_mmbtu,
            exceedance_pct=exceedance_pct,
            start_time=measurement.timestamp,
            duration_hours=measurement.averaging_period_hours,
            cause_category="equipment_malfunction" if "malfunction" in cause.lower() else "operational",
            cause_description=cause,
            corrective_actions=corrective_actions,
            preventive_actions=self._generate_recommendations(
                measurement.pollutant,
                ComplianceStatus.EXCEEDANCE,
                100 + exceedance_pct,
                measurement.fuel_type_during_measurement,
            ),
        )

    def calculate_excess_emissions_duration(
        self,
        deviations: List[DeviationEvent],
        reporting_period_hours: float,
    ) -> Dict[str, Any]:
        """
        Calculate excess emissions statistics for reporting.

        Reference: 40 CFR 60.7(c) - Excess emissions and monitoring systems performance report

        Args:
            deviations: List of deviation events
            reporting_period_hours: Total hours in reporting period

        Returns:
            Statistics dictionary for semi-annual report
        """
        stats = {
            "reporting_period_hours": reporting_period_hours,
            "by_pollutant": {},
            "total_excess_hours": 0.0,
            "percent_in_compliance": 100.0,
        }

        for pollutant in PollutantType:
            pollutant_deviations = [d for d in deviations if d.pollutant == pollutant]
            excess_hours = sum(d.duration_hours for d in pollutant_deviations)

            stats["by_pollutant"][pollutant.value] = {
                "excess_emission_events": len(pollutant_deviations),
                "total_excess_hours": excess_hours,
                "percent_in_compliance": ((reporting_period_hours - excess_hours) / reporting_period_hours) * 100 if reporting_period_hours > 0 else 100.0,
            }

            stats["total_excess_hours"] += excess_hours

        stats["percent_in_compliance"] = (
            (reporting_period_hours - stats["total_excess_hours"]) / reporting_period_hours
        ) * 100 if reporting_period_hours > 0 else 100.0

        return stats

    # -------------------------------------------------------------------------
    # COMPLIANCE REPORTING
    # -------------------------------------------------------------------------

    def generate_compliance_report(
        self,
        unit: UnitCharacteristics,
        measurements: List[EmissionMeasurement],
        deviations: List[DeviationEvent],
        exemptions: List[ExemptionRecord],
        reporting_period_start: datetime,
        reporting_period_end: datetime,
        fuel_consumption: Dict[str, float],
        cems_availability: Dict[str, float],
    ) -> NSPSComplianceReport:
        """
        Generate a comprehensive NSPS compliance report.

        This generates the data needed for:
            - Semi-annual excess emissions reports (40 CFR 60.7(c))
            - Annual compliance certifications (40 CFR 60.49b(u))

        Args:
            unit: Unit characteristics
            measurements: Emission measurements for the period
            deviations: Deviation events
            exemptions: Exemption records
            reporting_period_start: Start of reporting period
            reporting_period_end: End of reporting period
            fuel_consumption: Fuel consumption by type (MMBtu)
            cems_availability: CEMS data availability by pollutant (%)

        Returns:
            NSPSComplianceReport
        """
        subpart = self.determine_applicable_subpart(unit)

        # Calculate compliance results for each measurement
        compliance_results = []
        for measurement in measurements:
            result = self.check_pollutant_compliance(unit, measurement, exemptions)
            compliance_results.append(result)

        # Determine overall status
        status_priority = {
            ComplianceStatus.COMPLIANT: 0,
            ComplianceStatus.NOT_APPLICABLE: 0,
            ComplianceStatus.EXEMPT: 0,
            ComplianceStatus.WARNING: 1,
            ComplianceStatus.EXCEEDANCE: 2,
            ComplianceStatus.CRITICAL: 3,
            ComplianceStatus.PENDING_DATA: 0,
        }

        if compliance_results:
            overall_status = max(
                compliance_results,
                key=lambda r: status_priority.get(r.compliance_status, 0)
            ).compliance_status
        else:
            overall_status = ComplianceStatus.PENDING_DATA

        # Calculate reporting period hours
        period_hours = (reporting_period_end - reporting_period_start).total_seconds() / 3600.0

        # Calculate excess emissions statistics
        excess_stats = self.calculate_excess_emissions_duration(deviations, period_hours)

        # Generate emission totals from measurements
        emission_totals = {}
        for measurement in measurements:
            pollutant = measurement.pollutant.value
            if pollutant not in emission_totals:
                emission_totals[pollutant] = 0.0
            if measurement.heat_input_mmbtu:
                emission_totals[pollutant] += measurement.measured_value_lb_mmbtu * measurement.heat_input_mmbtu

        # Generate recommendations
        recommendations = []
        if overall_status == ComplianceStatus.EXCEEDANCE:
            recommendations.append("Review and document all excess emission events per 40 CFR 60.7(c)")
            recommendations.append("Submit semi-annual report within 30 days of period end")
        if any(cems_availability.get(p, 100) < 95 for p in cems_availability):
            recommendations.append("CEMS data availability below 95% - review QA/QC procedures")

        # Certification statement
        certification = (
            "I certify that the information contained in this report is accurate and complete "
            "to the best of my knowledge. I am aware that there are significant penalties for "
            "submitting false information, including the possibility of fine and imprisonment "
            "for knowing violations."
        )

        return NSPSComplianceReport(
            reporting_period_start=reporting_period_start,
            reporting_period_end=reporting_period_end,
            unit_characteristics=unit,
            applicable_subpart=subpart,
            overall_compliance_status=overall_status,
            compliance_results=compliance_results,
            deviation_events=deviations,
            exemption_records=exemptions,
            monitoring_summary=excess_stats,
            fuel_consumption_summary=fuel_consumption,
            emission_totals=emission_totals,
            cems_data_availability=cems_availability,
            recommendations=recommendations,
            certification_statement=certification,
        )

    def get_reporting_schedule(
        self,
        unit: UnitCharacteristics,
        reference_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming reporting deadlines for a unit.

        Args:
            unit: Unit characteristics
            reference_date: Current date for calculating deadlines

        Returns:
            List of upcoming reporting deadlines
        """
        subpart = self.determine_applicable_subpart(unit)
        schedules = []

        for report_type, requirement in self.reporting_requirements.items():
            if requirement.frequency == ReportingPeriod.SEMI_ANNUAL:
                # Semi-annual periods end June 30 and December 31
                if reference_date.month <= 6:
                    period_end = date(reference_date.year, 6, 30)
                else:
                    period_end = date(reference_date.year, 12, 31)
                due_date = period_end + timedelta(days=30)

            elif requirement.frequency == ReportingPeriod.ANNUAL:
                # Annual reports typically due by March 1
                due_date = date(reference_date.year + 1, 3, 1)
                period_end = date(reference_date.year, 12, 31)

            else:
                continue

            schedules.append({
                "report_type": report_type,
                "description": requirement.description,
                "period_end": period_end.isoformat(),
                "due_date": due_date.isoformat(),
                "days_until_due": (due_date - reference_date).days,
                "electronic_submission": requirement.electronic_reporting_required,
                "cedri_required": requirement.cedri_submission,
                "cfr_reference": requirement.cfr_reference,
            })

        return sorted(schedules, key=lambda x: x["due_date"])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def check_nsps_compliance(
    unit_capacity_mmbtu_hr: float,
    fuel_type: str,
    measured_nox_lb_mmbtu: float,
    measured_so2_lb_mmbtu: float,
    measured_pm_lb_mmbtu: float,
    **kwargs,
) -> ComplianceResult:
    """
    Convenience function to check NSPS compliance.

    Args:
        unit_capacity_mmbtu_hr: Rated heat input capacity (MMBtu/hr)
        fuel_type: Fuel type string (e.g., "natural_gas")
        measured_nox_lb_mmbtu: Measured NOx (lb/MMBtu)
        measured_so2_lb_mmbtu: Measured SO2 (lb/MMBtu)
        measured_pm_lb_mmbtu: Measured PM (lb/MMBtu)
        **kwargs: Additional arguments passed to validator

    Returns:
        ComplianceResult

    Example:
        >>> result = check_nsps_compliance(
        ...     unit_capacity_mmbtu_hr=150.0,
        ...     fuel_type="natural_gas",
        ...     measured_nox_lb_mmbtu=0.08,
        ...     measured_so2_lb_mmbtu=0.0,
        ...     measured_pm_lb_mmbtu=0.0,
        ... )
        >>> print(result.compliance_status.value)
        compliant
    """
    validator = NSPSComplianceValidator()
    return validator.check_nsps_compliance(
        unit_capacity_mmbtu_hr=unit_capacity_mmbtu_hr,
        fuel_type=fuel_type,
        measured_nox_lb_mmbtu=measured_nox_lb_mmbtu,
        measured_so2_lb_mmbtu=measured_so2_lb_mmbtu,
        measured_pm_lb_mmbtu=measured_pm_lb_mmbtu,
        **kwargs,
    )


def get_emission_limit(
    subpart: str,
    pollutant: str,
    fuel_type: str,
) -> Optional[float]:
    """
    Convenience function to look up emission limit.

    Args:
        subpart: NSPS subpart (e.g., "Db", "Dc")
        pollutant: Pollutant type (e.g., "NOx", "SO2", "PM")
        fuel_type: Fuel category (e.g., "natural_gas", "coal_bituminous")

    Returns:
        Emission limit in lb/MMBtu, or None

    Example:
        >>> limit = get_emission_limit("Db", "NOx", "natural_gas")
        >>> print(f"Limit: {limit} lb/MMBtu")
        Limit: 0.1 lb/MMBtu
    """
    validator = NSPSComplianceValidator()

    try:
        subpart_enum = NSPSSubpart(subpart)
        pollutant_enum = PollutantType(pollutant)
        fuel_enum = FuelCategory(fuel_type)
    except ValueError:
        return None

    return validator.get_emission_limit(subpart_enum, pollutant_enum, fuel_enum)


def determine_applicable_subpart(
    capacity_mmbtu_hr: float,
    construction_date: date,
    is_electric_utility: bool = False,
) -> str:
    """
    Convenience function to determine applicable NSPS subpart.

    Args:
        capacity_mmbtu_hr: Unit rated capacity
        construction_date: Date of construction
        is_electric_utility: Whether unit is an electric utility

    Returns:
        Subpart designation string (e.g., "Db", "Dc", "N/A")

    Example:
        >>> subpart = determine_applicable_subpart(150.0, date(2020, 1, 1))
        >>> print(subpart)
        Db
    """
    validator = NSPSComplianceValidator()

    unit = UnitCharacteristics(
        unit_id="LOOKUP",
        unit_name="Lookup",
        rated_capacity_mmbtu_hr=capacity_mmbtu_hr,
        construction_date=construction_date,
        is_electric_utility=is_electric_utility,
    )

    subpart = validator.determine_applicable_subpart(unit)
    return subpart.value


# =============================================================================
# MAIN - EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import json

    print("=" * 80)
    print("EPA 40 CFR Part 60 NSPS Compliance Validator")
    print("=" * 80)

    # Create validator
    validator = NSPSComplianceValidator()

    # Example 1: Natural gas boiler
    print("\n--- Example 1: Natural Gas Boiler (150 MMBtu/hr) ---")
    result = check_nsps_compliance(
        unit_capacity_mmbtu_hr=150.0,
        fuel_type="natural_gas",
        measured_nox_lb_mmbtu=0.08,
        measured_so2_lb_mmbtu=0.0,
        measured_pm_lb_mmbtu=0.0,
        unit_id="BLR-001",
    )
    print(f"Subpart: {result.applicable_subpart.value}")
    print(f"Status: {result.compliance_status.value}")
    print(f"NOx Limit: {result.applicable_limit_lb_mmbtu} lb/MMBtu")
    print(f"NOx Measured: {result.measured_value_lb_mmbtu} lb/MMBtu")
    print(f"Percent of Limit: {result.percent_of_limit:.1f}%")

    # Example 2: Coal-fired unit
    print("\n--- Example 2: Coal-Fired Boiler (200 MMBtu/hr) ---")
    result2 = check_nsps_compliance(
        unit_capacity_mmbtu_hr=200.0,
        fuel_type="coal_bituminous",
        measured_nox_lb_mmbtu=0.45,
        measured_so2_lb_mmbtu=1.0,
        measured_pm_lb_mmbtu=0.04,
        unit_id="BLR-002",
    )
    print(f"Subpart: {result2.applicable_subpart.value}")
    print(f"Status: {result2.compliance_status.value}")
    for rec in result2.recommendations:
        print(f"  - {rec}")

    # Example 3: Get all limits for a fuel type
    print("\n--- Subpart Db Limits for Natural Gas ---")
    limits = validator.get_all_limits_for_unit(NSPSSubpart.SUBPART_Db, FuelCategory.NATURAL_GAS)
    for pollutant, limit in limits.items():
        print(f"  {pollutant}: {limit} lb/MMBtu" if limit else f"  {pollutant}: Not applicable")

    # Example 4: Monitoring requirements
    print("\n--- Monitoring Requirements for 200 MMBtu/hr Coal Unit ---")
    unit = UnitCharacteristics(
        unit_id="BLR-002",
        unit_name="Coal Boiler 2",
        rated_capacity_mmbtu_hr=200.0,
        construction_date=date(2020, 1, 1),
        primary_fuel=FuelCategory.COAL_BITUMINOUS,
    )
    mon_plan = validator.get_monitoring_plan_requirements(unit)
    print(f"CEMS Required for: {[r['pollutant'] for r in mon_plan['cems_requirements']]}")
    if mon_plan['fuel_analysis_requirements']:
        print(f"Fuel Analysis Frequency: {mon_plan['fuel_analysis_requirements']['frequency']}")

    # Example 5: Reporting schedule
    print("\n--- Upcoming Reporting Deadlines ---")
    schedule = validator.get_reporting_schedule(unit, date.today())
    for item in schedule[:3]:
        print(f"  {item['report_type']}: Due {item['due_date']} ({item['days_until_due']} days)")

    print("\n" + "=" * 80)
    print("Compliance validation complete. See 40 CFR Part 60 for authoritative requirements.")
    print("=" * 80)
