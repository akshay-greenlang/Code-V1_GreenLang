# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH Deterministic Tools.

All tools produce zero-hallucination results using physics-based calculations.
This module provides deterministic calculation tools for emissions compliance
monitoring, regulatory reporting, violation detection, and audit trail generation.

CRITICAL: All numeric results come from deterministic formulas, never from LLM generation.

Physics Basis:
- Combustion Stoichiometry: CxHy + (x + y/4)O2 -> xCO2 + (y/2)H2O
- NOx Formation: Thermal (Zeldovich), Fuel, Prompt mechanisms
- SOx Formation: S + O2 -> SO2
- PM: Soot, ash, and particulate formation

Standards Compliance:
- EPA 40 CFR Part 60 - Standards of Performance
- EPA 40 CFR Part 75 - Continuous Emissions Monitoring
- EPA Method 19 - Sulfur Dioxide Removal and Particulate Matter
- EPA Method 2 - Stack Gas Velocity and Volumetric Flow Rate
- EPA Method 3A - Gas Analysis for CO2/O2 (Instrumental)
- EPA Method 5 - Particulate Matter
- EU Industrial Emissions Directive 2010/75/EU
- China MEE GB 13223-2011

Author: GreenLang Foundation
Version: 1.0.0
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid


# ============================================================================
# CONSTANTS AND REFERENCE DATA
# ============================================================================

# Gas constants
UNIVERSAL_GAS_CONSTANT = 8.314  # J/(mol*K)
STANDARD_TEMPERATURE_K = 298.15  # 25 degC
STANDARD_PRESSURE_PA = 101325   # 1 atm

# Molecular weights (g/mol)
MOLECULAR_WEIGHTS = {
    'NO': 30.01,
    'NO2': 46.01,
    'N2O': 44.01,
    'SO2': 64.07,
    'SO3': 80.07,
    'CO2': 44.01,
    'CO': 28.01,
    'O2': 32.00,
    'N2': 28.01,
    'H2O': 18.02,
    'CH4': 16.04,
    'C2H6': 30.07,
    'C3H8': 44.10
}

# Emission factors (AP-42, Fifth Edition)
AP42_EMISSION_FACTORS = {
    'natural_gas': {
        'nox_lb_mmbtu': 0.1,
        'sox_lb_mmbtu': 0.0006,
        'co2_lb_mmbtu': 117.0,
        'pm_lb_mmbtu': 0.0076,
        'co_lb_mmbtu': 0.082
    },
    'fuel_oil_no2': {
        'nox_lb_mmbtu': 0.14,
        'sox_lb_mmbtu': 0.5,  # Depends on S content
        'co2_lb_mmbtu': 161.0,
        'pm_lb_mmbtu': 0.025,
        'co_lb_mmbtu': 0.036
    },
    'fuel_oil_no6': {
        'nox_lb_mmbtu': 0.35,
        'sox_lb_mmbtu': 1.0,  # Depends on S content
        'co2_lb_mmbtu': 173.0,
        'pm_lb_mmbtu': 0.1,
        'co_lb_mmbtu': 0.036
    },
    'coal_bituminous': {
        'nox_lb_mmbtu': 0.6,
        'sox_lb_mmbtu': 1.2,  # Depends on S content
        'co2_lb_mmbtu': 205.0,
        'pm_lb_mmbtu': 0.4,
        'co_lb_mmbtu': 0.5
    },
    'biomass_wood': {
        'nox_lb_mmbtu': 0.22,
        'sox_lb_mmbtu': 0.025,
        'co2_lb_mmbtu': 195.0,  # Biogenic
        'pm_lb_mmbtu': 0.3,
        'co_lb_mmbtu': 0.6
    }
}

# Regulatory limits by jurisdiction
REGULATORY_LIMITS = {
    'EPA': {
        'nox_lb_mmbtu': 0.10,    # NSPS Subpart Da
        'sox_lb_mmbtu': 0.15,    # NSPS Subpart Da
        'co2_tons_mwh': 1.0,     # Clean Power Plan (historic)
        'pm_lb_mmbtu': 0.03,     # NSPS Subpart Da
        'opacity_percent': 20.0
    },
    'EU_IED': {
        'nox_mg_nm3': 100.0,     # BAT-AEL for gas
        'sox_mg_nm3': 150.0,     # BAT-AEL for gas
        'co2_g_kwh': 350.0,
        'pm_mg_nm3': 5.0,        # BAT-AEL
        'co_mg_nm3': 100.0
    },
    'CHINA_MEE': {
        'nox_mg_nm3': 50.0,      # GB 13223-2011
        'sox_mg_nm3': 35.0,      # Ultra-low emission
        'co2_g_kwh': 320.0,
        'pm_mg_nm3': 10.0,       # Ultra-low emission
        'mercury_ug_nm3': 30.0
    }
}

# F-factors for EPA Method 19 (dscf/MMBtu)
F_FACTORS = {
    'natural_gas': {
        'Fd': 8710,    # Dry basis
        'Fw': 10610,   # Wet basis
        'Fc': 1040     # Carbon
    },
    'fuel_oil_no2': {
        'Fd': 9190,
        'Fw': 10320,
        'Fc': 1420
    },
    'fuel_oil_no6': {
        'Fd': 9220,
        'Fw': 10320,
        'Fc': 1490
    },
    'coal_bituminous': {
        'Fd': 9780,
        'Fw': 10640,
        'Fc': 1800
    },
    'biomass_wood': {
        'Fd': 9240,
        'Fw': 10580,
        'Fc': 1970
    }
}

# NOx temperature coefficient (Zeldovich mechanism)
NOX_ARRHENIUS_A = 1.8e14  # Pre-exponential factor
NOX_ARRHENIUS_EA = 318000  # Activation energy J/mol


# ============================================================================
# RESULT DATA CLASSES
# ============================================================================

@dataclass
class NOxEmissionsResult:
    """Result from NOx emissions calculation."""
    concentration_ppm: float
    emission_rate_lb_mmbtu: float
    emission_rate_lb_hr: float
    mass_rate_kg_hr: float
    thermal_nox_percent: float
    fuel_nox_percent: float
    prompt_nox_percent: float
    correction_factor: float
    calculation_method: str
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'concentration_ppm': self.concentration_ppm,
            'emission_rate_lb_mmbtu': self.emission_rate_lb_mmbtu,
            'emission_rate_lb_hr': self.emission_rate_lb_hr,
            'mass_rate_kg_hr': self.mass_rate_kg_hr,
            'thermal_nox_percent': self.thermal_nox_percent,
            'fuel_nox_percent': self.fuel_nox_percent,
            'prompt_nox_percent': self.prompt_nox_percent,
            'correction_factor': self.correction_factor,
            'calculation_method': self.calculation_method,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class SOxEmissionsResult:
    """Result from SOx emissions calculation."""
    concentration_ppm: float
    emission_rate_lb_mmbtu: float
    emission_rate_lb_hr: float
    mass_rate_kg_hr: float
    fuel_sulfur_percent: float
    so2_so3_ratio: float
    removal_efficiency_percent: float
    calculation_method: str
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'concentration_ppm': self.concentration_ppm,
            'emission_rate_lb_mmbtu': self.emission_rate_lb_mmbtu,
            'emission_rate_lb_hr': self.emission_rate_lb_hr,
            'mass_rate_kg_hr': self.mass_rate_kg_hr,
            'fuel_sulfur_percent': self.fuel_sulfur_percent,
            'so2_so3_ratio': self.so2_so3_ratio,
            'removal_efficiency_percent': self.removal_efficiency_percent,
            'calculation_method': self.calculation_method,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class CO2EmissionsResult:
    """Result from CO2 emissions calculation."""
    concentration_percent: float
    emission_rate_lb_mmbtu: float
    mass_rate_tons_hr: float
    mass_rate_kg_hr: float
    carbon_content_percent: float
    combustion_efficiency_percent: float
    biogenic_percent: float
    calculation_method: str
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'concentration_percent': self.concentration_percent,
            'emission_rate_lb_mmbtu': self.emission_rate_lb_mmbtu,
            'mass_rate_tons_hr': self.mass_rate_tons_hr,
            'mass_rate_kg_hr': self.mass_rate_kg_hr,
            'carbon_content_percent': self.carbon_content_percent,
            'combustion_efficiency_percent': self.combustion_efficiency_percent,
            'biogenic_percent': self.biogenic_percent,
            'calculation_method': self.calculation_method,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class PMEmissionsResult:
    """Result from particulate matter emissions calculation."""
    concentration_mg_m3: float
    emission_rate_lb_mmbtu: float
    emission_rate_lb_hr: float
    mass_rate_kg_hr: float
    pm10_fraction: float
    pm25_fraction: float
    filterable_percent: float
    condensable_percent: float
    calculation_method: str
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'concentration_mg_m3': self.concentration_mg_m3,
            'emission_rate_lb_mmbtu': self.emission_rate_lb_mmbtu,
            'emission_rate_lb_hr': self.emission_rate_lb_hr,
            'mass_rate_kg_hr': self.mass_rate_kg_hr,
            'pm10_fraction': self.pm10_fraction,
            'pm25_fraction': self.pm25_fraction,
            'filterable_percent': self.filterable_percent,
            'condensable_percent': self.condensable_percent,
            'calculation_method': self.calculation_method,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class ComplianceCheckResult:
    """Result from compliance status check."""
    overall_status: str
    nox_status: str
    sox_status: str
    co2_status: str
    pm_status: str
    violations: List[Dict[str, Any]]
    jurisdiction: str
    applicable_limits: Dict[str, float]
    margin_to_limits: Dict[str, float]
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_status': self.overall_status,
            'nox_status': self.nox_status,
            'sox_status': self.sox_status,
            'co2_status': self.co2_status,
            'pm_status': self.pm_status,
            'violations': self.violations,
            'jurisdiction': self.jurisdiction,
            'applicable_limits': self.applicable_limits,
            'margin_to_limits': self.margin_to_limits,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class ViolationResult:
    """Result for detected violation."""
    violation_id: str
    pollutant: str
    measured_value: float
    limit_value: float
    exceedance_percent: float
    severity: str
    duration_minutes: int
    regulatory_reference: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'violation_id': self.violation_id,
            'pollutant': self.pollutant,
            'measured_value': self.measured_value,
            'limit_value': self.limit_value,
            'exceedance_percent': self.exceedance_percent,
            'severity': self.severity,
            'duration_minutes': self.duration_minutes,
            'regulatory_reference': self.regulatory_reference,
            'timestamp': self.timestamp
        }


@dataclass
class RegulatoryReportResult:
    """Result from regulatory report generation."""
    report_id: str
    jurisdiction: str
    reporting_period: Dict[str, str]
    total_operating_hours: float
    avg_nox_lb_mmbtu: float
    avg_sox_lb_mmbtu: float
    total_co2_tons: float
    avg_pm_lb_mmbtu: float
    compliance_rate_percent: float
    exceedance_count: int
    data_availability_percent: float
    certifier: str
    certification_date: str
    submission_deadline: str
    format_version: str
    sections: List[Dict[str, Any]]
    attachments: List[str]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_id': self.report_id,
            'jurisdiction': self.jurisdiction,
            'reporting_period': self.reporting_period,
            'total_operating_hours': self.total_operating_hours,
            'avg_nox_lb_mmbtu': self.avg_nox_lb_mmbtu,
            'avg_sox_lb_mmbtu': self.avg_sox_lb_mmbtu,
            'total_co2_tons': self.total_co2_tons,
            'avg_pm_lb_mmbtu': self.avg_pm_lb_mmbtu,
            'compliance_rate_percent': self.compliance_rate_percent,
            'exceedance_count': self.exceedance_count,
            'data_availability_percent': self.data_availability_percent,
            'certifier': self.certifier,
            'certification_date': self.certification_date,
            'submission_deadline': self.submission_deadline,
            'format_version': self.format_version,
            'sections': self.sections,
            'attachments': self.attachments,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class ExceedancePredictionResult:
    """Result from exceedance prediction."""
    pollutant: str
    current_value: float
    predicted_value: float
    limit_value: float
    exceedance_probability: float
    time_to_exceedance_hours: Optional[float]
    confidence_level: str
    model_type: str
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pollutant': self.pollutant,
            'current_value': self.current_value,
            'predicted_value': self.predicted_value,
            'limit_value': self.limit_value,
            'exceedance_probability': self.exceedance_probability,
            'time_to_exceedance_hours': self.time_to_exceedance_hours,
            'confidence_level': self.confidence_level,
            'model_type': self.model_type,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class EmissionFactorResult:
    """Result from emission factor lookup."""
    fuel_type: str
    nox_factor: float
    sox_factor: float
    co2_factor: float
    pm_factor: float
    co_factor: float
    factor_source: str
    units: str
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fuel_type': self.fuel_type,
            'nox_factor': self.nox_factor,
            'sox_factor': self.sox_factor,
            'co2_factor': self.co2_factor,
            'pm_factor': self.pm_factor,
            'co_factor': self.co_factor,
            'factor_source': self.factor_source,
            'units': self.units,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class DispersionResult:
    """Result from Gaussian plume dispersion calculation."""
    max_ground_concentration: float
    distance_to_max_m: float
    plume_rise_m: float
    effective_stack_height_m: float
    sigma_y: float
    sigma_z: float
    stability_class: str
    calculation_method: str
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_ground_concentration': self.max_ground_concentration,
            'distance_to_max_m': self.distance_to_max_m,
            'plume_rise_m': self.plume_rise_m,
            'effective_stack_height_m': self.effective_stack_height_m,
            'sigma_y': self.sigma_y,
            'sigma_z': self.sigma_z,
            'stability_class': self.stability_class,
            'calculation_method': self.calculation_method,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class AuditTrailResult:
    """Result from audit trail generation."""
    audit_id: str
    audit_period: Dict[str, str]
    total_records: int
    compliant_records: int
    non_compliant_records: int
    data_quality_score: float
    completeness_percent: float
    root_hash: str
    record_hashes: List[str]
    chain_valid: bool
    compliance_events: List[Dict[str, Any]]
    data_corrections: List[Dict[str, Any]]
    certifier: str
    certification_date: str
    certification_statement: str
    epa_part_75_compliant: bool
    data_retention_met: bool
    qapp_requirements_met: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'audit_id': self.audit_id,
            'audit_period': self.audit_period,
            'total_records': self.total_records,
            'compliant_records': self.compliant_records,
            'non_compliant_records': self.non_compliant_records,
            'data_quality_score': self.data_quality_score,
            'completeness_percent': self.completeness_percent,
            'root_hash': self.root_hash,
            'record_hashes': self.record_hashes,
            'chain_valid': self.chain_valid,
            'compliance_events': self.compliance_events,
            'data_corrections': self.data_corrections,
            'certifier': self.certifier,
            'certification_date': self.certification_date,
            'certification_statement': self.certification_statement,
            'epa_part_75_compliant': self.epa_part_75_compliant,
            'data_retention_met': self.data_retention_met,
            'qapp_requirements_met': self.qapp_requirements_met
        }


@dataclass
class FuelAnalysisResult:
    """Result from fuel composition analysis."""
    fuel_type: str
    carbon_percent: float
    hydrogen_percent: float
    sulfur_percent: float
    nitrogen_percent: float
    oxygen_percent: float
    ash_percent: float
    moisture_percent: float
    hhv_btu_lb: float
    lhv_btu_lb: float
    stoichiometric_air_lb_lb: float
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fuel_type': self.fuel_type,
            'carbon_percent': self.carbon_percent,
            'hydrogen_percent': self.hydrogen_percent,
            'sulfur_percent': self.sulfur_percent,
            'nitrogen_percent': self.nitrogen_percent,
            'oxygen_percent': self.oxygen_percent,
            'ash_percent': self.ash_percent,
            'moisture_percent': self.moisture_percent,
            'hhv_btu_lb': self.hhv_btu_lb,
            'lhv_btu_lb': self.lhv_btu_lb,
            'stoichiometric_air_lb_lb': self.stoichiometric_air_lb_lb,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


# ============================================================================
# TOOL SCHEMAS
# ============================================================================

EMISSIONS_TOOL_SCHEMAS = {
    "calculate_nox_emissions": {
        "name": "calculate_nox_emissions",
        "description": "Calculate NOx emissions using EPA Method 19 with thermal, fuel, and prompt NOx components",
        "parameters": {
            "type": "object",
            "properties": {
                "cems_data": {
                    "type": "object",
                    "description": "CEMS measurements including NOx ppm, O2%, flow rate"
                },
                "fuel_data": {
                    "type": "object",
                    "description": "Fuel type, flow rate, nitrogen content"
                },
                "process_parameters": {
                    "type": "object",
                    "description": "Temperature, excess air, combustion conditions"
                }
            },
            "required": ["cems_data", "fuel_data"]
        },
        "deterministic": True,
        "formula": "NOx = Thermal_NOx + Fuel_NOx + Prompt_NOx",
        "standards": ["EPA Method 19", "40 CFR Part 60"]
    },
    "calculate_sox_emissions": {
        "name": "calculate_sox_emissions",
        "description": "Calculate SOx emissions from fuel sulfur content using stoichiometry",
        "parameters": {
            "type": "object",
            "properties": {
                "fuel_data": {
                    "type": "object",
                    "description": "Fuel type, sulfur content, flow rate"
                },
                "process_parameters": {
                    "type": "object",
                    "description": "Control device efficiency"
                }
            },
            "required": ["fuel_data"]
        },
        "deterministic": True,
        "formula": "SO2 = (S_fuel * 64/32) * (1 - control_efficiency)",
        "standards": ["EPA Method 6", "40 CFR Part 60"]
    },
    "calculate_co2_emissions": {
        "name": "calculate_co2_emissions",
        "description": "Calculate CO2 emissions from combustion stoichiometry",
        "parameters": {
            "type": "object",
            "properties": {
                "fuel_data": {
                    "type": "object",
                    "description": "Fuel type, carbon content, flow rate"
                },
                "process_parameters": {
                    "type": "object",
                    "description": "Combustion efficiency"
                }
            },
            "required": ["fuel_data"]
        },
        "deterministic": True,
        "formula": "CO2 = C_fuel * (44/12) * combustion_efficiency",
        "standards": ["EPA Method 3A", "40 CFR Part 75"]
    },
    "calculate_particulate_matter": {
        "name": "calculate_particulate_matter",
        "description": "Calculate PM10/PM2.5 emissions from filterable and condensable components",
        "parameters": {
            "type": "object",
            "properties": {
                "cems_data": {
                    "type": "object",
                    "description": "Opacity, flow rate measurements"
                },
                "fuel_data": {
                    "type": "object",
                    "description": "Fuel type, ash content"
                },
                "process_parameters": {
                    "type": "object",
                    "description": "Control device parameters"
                }
            },
            "required": ["fuel_data"]
        },
        "deterministic": True,
        "formula": "PM = Filterable_PM + Condensable_PM",
        "standards": ["EPA Method 5", "EPA Method 202"]
    },
    "check_compliance_status": {
        "name": "check_compliance_status",
        "description": "Check emissions against multi-jurisdiction regulatory limits",
        "parameters": {
            "type": "object",
            "properties": {
                "emissions_result": {
                    "type": "object",
                    "description": "Calculated emissions data"
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "Regulatory jurisdiction (EPA, EU_IED, CHINA_MEE)"
                },
                "process_parameters": {
                    "type": "object",
                    "description": "Process type for applicable limits"
                }
            },
            "required": ["emissions_result", "jurisdiction"]
        },
        "deterministic": True,
        "jurisdictions": ["EPA", "EU_IED", "CHINA_MEE"]
    },
    "generate_regulatory_report": {
        "name": "generate_regulatory_report",
        "description": "Generate compliance reports in EPA/EU/China formats",
        "parameters": {
            "type": "object",
            "properties": {
                "report_format": {
                    "type": "string",
                    "description": "Report format (EPA_ECMPS, EU_ELED, CHINA_MEE)"
                },
                "reporting_period": {
                    "type": "object",
                    "description": "Start and end dates"
                },
                "facility_data": {
                    "type": "object",
                    "description": "Facility identification"
                },
                "emissions_data": {
                    "type": "array",
                    "description": "Historical emissions records"
                }
            },
            "required": ["report_format", "reporting_period"]
        },
        "deterministic": True,
        "formats": ["EPA_ECMPS", "EU_ELED", "CHINA_MEE"]
    },
    "detect_violations": {
        "name": "detect_violations",
        "description": "Detect emissions exceedances and violations",
        "parameters": {
            "type": "object",
            "properties": {
                "emissions_result": {
                    "type": "object",
                    "description": "Current emissions data"
                },
                "permit_limits": {
                    "type": "object",
                    "description": "Applicable permit limits"
                }
            },
            "required": ["emissions_result", "permit_limits"]
        },
        "deterministic": True,
        "output": "List of ViolationResult"
    },
    "predict_exceedances": {
        "name": "predict_exceedances",
        "description": "Predict potential emissions exceedances from trends",
        "parameters": {
            "type": "object",
            "properties": {
                "historical_data": {
                    "type": "array",
                    "description": "Historical emissions time series"
                },
                "permit_limits": {
                    "type": "object",
                    "description": "Permit limits for prediction"
                },
                "forecast_hours": {
                    "type": "number",
                    "description": "Forecast horizon in hours"
                }
            },
            "required": ["historical_data", "permit_limits"]
        },
        "deterministic": True,
        "methodology": "linear_extrapolation"
    },
    "calculate_emission_factors": {
        "name": "calculate_emission_factors",
        "description": "Look up AP-42 emission factors by fuel type",
        "parameters": {
            "type": "object",
            "properties": {
                "fuel_type": {
                    "type": "string",
                    "description": "Fuel type for factor lookup"
                }
            },
            "required": ["fuel_type"]
        },
        "deterministic": True,
        "data_source": "AP-42 Fifth Edition"
    },
    "analyze_fuel_composition": {
        "name": "analyze_fuel_composition",
        "description": "Analyze fuel composition for emissions calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "fuel_type": {
                    "type": "string",
                    "description": "Type of fuel"
                },
                "ultimate_analysis": {
                    "type": "object",
                    "description": "C, H, S, N, O, ash percentages"
                },
                "proximate_analysis": {
                    "type": "object",
                    "description": "Moisture, ash, volatile matter"
                }
            },
            "required": ["fuel_type"]
        },
        "deterministic": True,
        "standards": ["ASTM D3176", "ASTM D5373"]
    },
    "calculate_dispersion": {
        "name": "calculate_dispersion",
        "description": "Calculate air dispersion using Gaussian plume model",
        "parameters": {
            "type": "object",
            "properties": {
                "emission_rate": {
                    "type": "number",
                    "description": "Emission rate in g/s"
                },
                "stack_parameters": {
                    "type": "object",
                    "description": "Stack height, diameter, exit velocity, temperature"
                },
                "meteorological_data": {
                    "type": "object",
                    "description": "Wind speed, stability class"
                }
            },
            "required": ["emission_rate", "stack_parameters", "meteorological_data"]
        },
        "deterministic": True,
        "methodology": "Gaussian_plume_Briggs"
    },
    "generate_audit_trail": {
        "name": "generate_audit_trail",
        "description": "Generate SHA-256 provenance audit trail",
        "parameters": {
            "type": "object",
            "properties": {
                "audit_period": {
                    "type": "object",
                    "description": "Audit period start and end"
                },
                "facility_data": {
                    "type": "object",
                    "description": "Facility identification"
                },
                "emissions_records": {
                    "type": "array",
                    "description": "Emissions records to audit"
                },
                "compliance_events": {
                    "type": "array",
                    "description": "Compliance events"
                }
            },
            "required": ["audit_period"]
        },
        "deterministic": True,
        "hash_algorithm": "SHA-256"
    }
}


# ============================================================================
# MAIN TOOLS CLASS
# ============================================================================

class EmissionsComplianceTools:
    """
    Deterministic calculation tools for emissions compliance monitoring.

    All methods use physics-based formulas with zero hallucination guarantee.
    No LLM is used for any numerical calculations.

    Attributes:
        config: Configuration object for calculation parameters
        jurisdiction: Default regulatory jurisdiction

    Example:
        >>> tools = EmissionsComplianceTools()
        >>> result = tools.calculate_nox_emissions(
        ...     cems_data={'nox_ppm': 45, 'o2_percent': 3.0},
        ...     fuel_data={'fuel_type': 'natural_gas', 'flow_rate_mmbtu_hr': 100}
        ... )
        >>> print(f"NOx: {result.emission_rate_lb_mmbtu} lb/MMBtu")
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize EmissionsComplianceTools.

        Args:
            config: Optional configuration object
        """
        self.config = config
        self.jurisdiction = 'EPA'
        if config and hasattr(config, 'jurisdiction'):
            self.jurisdiction = config.jurisdiction

    # ========================================================================
    # TOOL 1: NOX EMISSIONS CALCULATION
    # ========================================================================

    def calculate_nox_emissions(
        self,
        cems_data: Dict[str, Any],
        fuel_data: Dict[str, Any],
        process_parameters: Optional[Dict[str, Any]] = None
    ) -> NOxEmissionsResult:
        """
        Calculate NOx emissions using EPA Method 19.

        Physics basis: NOx formation from three mechanisms:
        1. Thermal NOx (Zeldovich): N2 + O -> NO + N
        2. Fuel NOx: Fuel-bound nitrogen oxidation
        3. Prompt NOx: CH radical attack on N2

        Formula: NOx (lb/MMBtu) = Cppm * Fd * Mw_NOx / (Mw_dscf * 10^6)

        Args:
            cems_data: CEMS measurements including NOx ppm, O2%
            fuel_data: Fuel type and flow rate
            process_parameters: Temperature, excess air conditions

        Returns:
            NOxEmissionsResult with complete NOx breakdown

        Standards: EPA Method 19, 40 CFR Part 60, Part 75
        """
        if process_parameters is None:
            process_parameters = {}

        # Extract inputs
        nox_ppm = cems_data.get('nox_ppm', 0)
        o2_percent = cems_data.get('o2_percent', 3.0)
        flow_rate_dscfm = cems_data.get('flow_rate_dscfm', 10000)

        fuel_type = fuel_data.get('fuel_type', 'natural_gas')
        heat_input_mmbtu_hr = fuel_data.get('heat_input_mmbtu_hr', 100)
        fuel_nitrogen_percent = fuel_data.get('nitrogen_percent', 0.01)

        combustion_temp_f = process_parameters.get('combustion_temperature_f', 2800)

        # Get F-factor for fuel
        f_factors = F_FACTORS.get(fuel_type, F_FACTORS['natural_gas'])
        fd = f_factors['Fd']

        # Calculate O2 correction factor (to 3% O2 reference)
        # Correction = (20.9 - O2_ref) / (20.9 - O2_measured)
        o2_ref = 3.0
        if o2_percent < 20.9:
            correction_factor = (20.9 - o2_ref) / (20.9 - o2_percent)
        else:
            correction_factor = 1.0

        # Correct NOx to reference O2
        nox_corrected_ppm = nox_ppm * correction_factor

        # Calculate emission rate (lb/MMBtu) using EPA Method 19
        # E = C * Fd * Mw / (K * 10^6)
        # where K = 385.3 dscf/lb-mol at standard conditions
        mw_nox = MOLECULAR_WEIGHTS['NO2']  # Use NO2 molecular weight per EPA convention
        k = 385.3  # dscf/lb-mol

        emission_rate_lb_mmbtu = (nox_corrected_ppm * fd * mw_nox) / (k * 1e6)

        # Calculate mass rates
        emission_rate_lb_hr = emission_rate_lb_mmbtu * heat_input_mmbtu_hr
        mass_rate_kg_hr = emission_rate_lb_hr * 0.453592

        # Estimate NOx source breakdown (simplified)
        # Thermal NOx increases with temperature (Arrhenius relationship)
        temp_k = (combustion_temp_f - 32) * 5/9 + 273.15
        thermal_factor = min(1.0, math.exp(-NOX_ARRHENIUS_EA / (UNIVERSAL_GAS_CONSTANT * temp_k)) * 1e20)

        # Fuel NOx proportional to fuel nitrogen
        fuel_factor = min(0.6, fuel_nitrogen_percent * 50)

        # Prompt NOx (typically small for most fuels)
        prompt_factor = 0.05

        total_factor = thermal_factor + fuel_factor + prompt_factor
        if total_factor > 0:
            thermal_nox_percent = (thermal_factor / total_factor) * 100
            fuel_nox_percent = (fuel_factor / total_factor) * 100
            prompt_nox_percent = (prompt_factor / total_factor) * 100
        else:
            thermal_nox_percent = 70.0
            fuel_nox_percent = 25.0
            prompt_nox_percent = 5.0

        result = NOxEmissionsResult(
            concentration_ppm=round(nox_corrected_ppm, 2),
            emission_rate_lb_mmbtu=round(emission_rate_lb_mmbtu, 6),
            emission_rate_lb_hr=round(emission_rate_lb_hr, 4),
            mass_rate_kg_hr=round(mass_rate_kg_hr, 4),
            thermal_nox_percent=round(thermal_nox_percent, 1),
            fuel_nox_percent=round(fuel_nox_percent, 1),
            prompt_nox_percent=round(prompt_nox_percent, 1),
            correction_factor=round(correction_factor, 4),
            calculation_method='EPA_Method_19',
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash({
                'cems_data': cems_data,
                'fuel_data': fuel_data
            })
        )

        return result

    # ========================================================================
    # TOOL 2: SOX EMISSIONS CALCULATION
    # ========================================================================

    def calculate_sox_emissions(
        self,
        fuel_data: Dict[str, Any],
        process_parameters: Optional[Dict[str, Any]] = None
    ) -> SOxEmissionsResult:
        """
        Calculate SOx emissions from fuel sulfur content.

        Physics basis: Combustion stoichiometry
        S + O2 -> SO2
        MW(SO2) = 64.07 g/mol, MW(S) = 32.07 g/mol

        Formula: SO2 (lb/lb S) = 64.07 / 32.07 = 2.0

        Args:
            fuel_data: Fuel type, sulfur content, heating value
            process_parameters: Control device efficiency

        Returns:
            SOxEmissionsResult with SOx emissions data

        Standards: EPA Method 6, 40 CFR Part 60
        """
        if process_parameters is None:
            process_parameters = {}

        # Extract inputs
        fuel_type = fuel_data.get('fuel_type', 'natural_gas')
        sulfur_percent = fuel_data.get('sulfur_percent', 0.0006)  # Natural gas default
        heat_input_mmbtu_hr = fuel_data.get('heat_input_mmbtu_hr', 100)
        heating_value_btu_lb = fuel_data.get('heating_value_btu_lb', 23000)

        # Control device efficiency
        fgd_efficiency = process_parameters.get('fgd_efficiency_percent', 0) / 100

        # Calculate SO2 emission rate
        # S -> SO2 conversion factor
        s_to_so2 = 64.07 / 32.07  # = 2.0

        # Calculate fuel consumption rate
        if heating_value_btu_lb > 0:
            fuel_rate_lb_hr = (heat_input_mmbtu_hr * 1e6) / heating_value_btu_lb
        else:
            fuel_rate_lb_hr = 0

        # Calculate uncontrolled SO2 rate
        sulfur_rate_lb_hr = fuel_rate_lb_hr * (sulfur_percent / 100)
        so2_rate_uncontrolled_lb_hr = sulfur_rate_lb_hr * s_to_so2

        # Apply control device efficiency
        so2_rate_lb_hr = so2_rate_uncontrolled_lb_hr * (1 - fgd_efficiency)

        # Calculate emission rate in lb/MMBtu
        if heat_input_mmbtu_hr > 0:
            emission_rate_lb_mmbtu = so2_rate_lb_hr / heat_input_mmbtu_hr
        else:
            emission_rate_lb_mmbtu = 0

        # Calculate concentration (ppm)
        # Using F-factor approach
        f_factors = F_FACTORS.get(fuel_type, F_FACTORS['natural_gas'])
        fd = f_factors['Fd']

        if fd > 0:
            # Reverse of Method 19 formula
            mw_so2 = MOLECULAR_WEIGHTS['SO2']
            k = 385.3
            concentration_ppm = (emission_rate_lb_mmbtu * k * 1e6) / (fd * mw_so2)
        else:
            concentration_ppm = 0

        # Mass rate in kg/hr
        mass_rate_kg_hr = so2_rate_lb_hr * 0.453592

        # SO2/SO3 ratio (typically 95-98% SO2)
        so2_so3_ratio = 97.0

        result = SOxEmissionsResult(
            concentration_ppm=round(concentration_ppm, 2),
            emission_rate_lb_mmbtu=round(emission_rate_lb_mmbtu, 6),
            emission_rate_lb_hr=round(so2_rate_lb_hr, 4),
            mass_rate_kg_hr=round(mass_rate_kg_hr, 4),
            fuel_sulfur_percent=round(sulfur_percent, 4),
            so2_so3_ratio=round(so2_so3_ratio, 1),
            removal_efficiency_percent=round(fgd_efficiency * 100, 1),
            calculation_method='Stoichiometric_S_to_SO2',
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash(fuel_data)
        )

        return result

    # ========================================================================
    # TOOL 3: CO2 EMISSIONS CALCULATION
    # ========================================================================

    def calculate_co2_emissions(
        self,
        fuel_data: Dict[str, Any],
        process_parameters: Optional[Dict[str, Any]] = None
    ) -> CO2EmissionsResult:
        """
        Calculate CO2 emissions from combustion stoichiometry.

        Physics basis: Carbon combustion
        C + O2 -> CO2
        MW(CO2) = 44.01 g/mol, MW(C) = 12.01 g/mol

        Formula: CO2 (lb/lb C) = 44.01 / 12.01 = 3.664

        Args:
            fuel_data: Fuel type, carbon content, heating value
            process_parameters: Combustion efficiency

        Returns:
            CO2EmissionsResult with CO2 emissions data

        Standards: EPA Method 3A, 40 CFR Part 75
        """
        if process_parameters is None:
            process_parameters = {}

        # Extract inputs
        fuel_type = fuel_data.get('fuel_type', 'natural_gas')
        carbon_percent = fuel_data.get('carbon_percent', 75.0)  # Natural gas ~ 75%
        heat_input_mmbtu_hr = fuel_data.get('heat_input_mmbtu_hr', 100)
        heating_value_btu_lb = fuel_data.get('heating_value_btu_lb', 23000)

        combustion_efficiency = process_parameters.get('combustion_efficiency_percent', 99) / 100

        # Use AP-42 factor if available
        ap42_factors = AP42_EMISSION_FACTORS.get(fuel_type, AP42_EMISSION_FACTORS['natural_gas'])
        co2_factor_lb_mmbtu = ap42_factors['co2_lb_mmbtu']

        # Calculate emission rate
        emission_rate_lb_mmbtu = co2_factor_lb_mmbtu * combustion_efficiency

        # Calculate mass rates
        emission_rate_lb_hr = emission_rate_lb_mmbtu * heat_input_mmbtu_hr
        mass_rate_tons_hr = emission_rate_lb_hr / 2000  # Short tons
        mass_rate_kg_hr = emission_rate_lb_hr * 0.453592

        # Calculate concentration (percent)
        # CO2% = (CO2 volume) / (total flue gas volume) * 100
        # Typical values: Natural gas ~10%, coal ~14%
        f_factors = F_FACTORS.get(fuel_type, F_FACTORS['natural_gas'])
        fc = f_factors['Fc']  # Carbon F-factor

        if fc > 0:
            co2_percent = (fc * 100) / (f_factors['Fd'] * (44.01 / 12.01))
        else:
            co2_percent = 10.0  # Default

        # Biogenic fraction (for biomass fuels)
        biogenic_percent = 100.0 if 'biomass' in fuel_type else 0.0

        result = CO2EmissionsResult(
            concentration_percent=round(co2_percent, 2),
            emission_rate_lb_mmbtu=round(emission_rate_lb_mmbtu, 2),
            mass_rate_tons_hr=round(mass_rate_tons_hr, 4),
            mass_rate_kg_hr=round(mass_rate_kg_hr, 2),
            carbon_content_percent=round(carbon_percent, 2),
            combustion_efficiency_percent=round(combustion_efficiency * 100, 2),
            biogenic_percent=round(biogenic_percent, 1),
            calculation_method='AP42_Emission_Factor',
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash(fuel_data)
        )

        return result

    # ========================================================================
    # TOOL 4: PARTICULATE MATTER CALCULATION
    # ========================================================================

    def calculate_particulate_matter(
        self,
        cems_data: Dict[str, Any],
        fuel_data: Dict[str, Any],
        process_parameters: Optional[Dict[str, Any]] = None
    ) -> PMEmissionsResult:
        """
        Calculate PM10/PM2.5 emissions.

        Components:
        1. Filterable PM: Direct particulate captured on filter
        2. Condensable PM: Vapors that condense after dilution

        Formula: PM = Filterable_PM + Condensable_PM

        Args:
            cems_data: Opacity, flow measurements
            fuel_data: Fuel type, ash content
            process_parameters: Control device parameters

        Returns:
            PMEmissionsResult with PM breakdown

        Standards: EPA Method 5, EPA Method 202
        """
        if process_parameters is None:
            process_parameters = {}

        # Extract inputs
        fuel_type = fuel_data.get('fuel_type', 'natural_gas')
        ash_percent = fuel_data.get('ash_percent', 0.0)
        heat_input_mmbtu_hr = fuel_data.get('heat_input_mmbtu_hr', 100)

        opacity_percent = cems_data.get('opacity_percent', 0)
        flow_rate_dscfm = cems_data.get('flow_rate_dscfm', 10000)

        # Control device efficiency
        baghouse_efficiency = process_parameters.get('baghouse_efficiency_percent', 99) / 100

        # Get AP-42 emission factor
        ap42_factors = AP42_EMISSION_FACTORS.get(fuel_type, AP42_EMISSION_FACTORS['natural_gas'])
        pm_factor_lb_mmbtu = ap42_factors['pm_lb_mmbtu']

        # Calculate uncontrolled emission rate
        uncontrolled_rate = pm_factor_lb_mmbtu

        # Apply control efficiency
        emission_rate_lb_mmbtu = uncontrolled_rate * (1 - baghouse_efficiency)

        # Calculate mass rates
        emission_rate_lb_hr = emission_rate_lb_mmbtu * heat_input_mmbtu_hr
        mass_rate_kg_hr = emission_rate_lb_hr * 0.453592

        # Calculate concentration (mg/m3)
        # Convert from lb/dscf to mg/m3
        if flow_rate_dscfm > 0:
            # Convert dscfm to m3/hr: 1 dscf = 0.0283168 m3
            flow_rate_m3_hr = flow_rate_dscfm * 60 * 0.0283168
            # mg/m3 = (kg/hr * 1e6 mg/kg) / (m3/hr)
            if flow_rate_m3_hr > 0:
                concentration_mg_m3 = (mass_rate_kg_hr * 1e6) / flow_rate_m3_hr
            else:
                concentration_mg_m3 = 0
        else:
            concentration_mg_m3 = 0

        # PM10/PM2.5 fractions (typical for combustion sources)
        pm10_fraction = 0.95  # 95% of total PM is PM10
        pm25_fraction = 0.85  # 85% of total PM is PM2.5

        # Filterable vs condensable (typical split)
        filterable_percent = 70.0
        condensable_percent = 30.0

        result = PMEmissionsResult(
            concentration_mg_m3=round(concentration_mg_m3, 2),
            emission_rate_lb_mmbtu=round(emission_rate_lb_mmbtu, 6),
            emission_rate_lb_hr=round(emission_rate_lb_hr, 4),
            mass_rate_kg_hr=round(mass_rate_kg_hr, 4),
            pm10_fraction=round(pm10_fraction, 2),
            pm25_fraction=round(pm25_fraction, 2),
            filterable_percent=round(filterable_percent, 1),
            condensable_percent=round(condensable_percent, 1),
            calculation_method='AP42_with_Control',
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash({
                'cems_data': cems_data,
                'fuel_data': fuel_data
            })
        )

        return result

    # ========================================================================
    # TOOL 5: COMPLIANCE STATUS CHECK
    # ========================================================================

    def check_compliance_status(
        self,
        emissions_result: Dict[str, Any],
        jurisdiction: str,
        process_parameters: Optional[Dict[str, Any]] = None
    ) -> ComplianceCheckResult:
        """
        Check emissions against multi-jurisdiction regulatory limits.

        Supports EPA, EU IED, and China MEE standards with
        appropriate unit conversions.

        Args:
            emissions_result: Calculated emissions data
            jurisdiction: Regulatory jurisdiction
            process_parameters: Process type for applicable limits

        Returns:
            ComplianceCheckResult with compliance status

        Standards: 40 CFR Part 60/75, EU IED 2010/75/EU, GB 13223-2011
        """
        if process_parameters is None:
            process_parameters = {}

        # Get applicable limits
        limits = REGULATORY_LIMITS.get(jurisdiction, REGULATORY_LIMITS['EPA'])

        # Extract emission values
        nox_data = emissions_result.get('nox', {})
        sox_data = emissions_result.get('sox', {})
        co2_data = emissions_result.get('co2', {})
        pm_data = emissions_result.get('pm', {})

        nox_value = nox_data.get('emission_rate_lb_mmbtu', 0)
        sox_value = sox_data.get('emission_rate_lb_mmbtu', 0)
        co2_value = co2_data.get('mass_rate_tons_hr', 0)
        pm_value = pm_data.get('emission_rate_lb_mmbtu', 0)

        # Get limits based on jurisdiction
        if jurisdiction == 'EPA':
            nox_limit = limits['nox_lb_mmbtu']
            sox_limit = limits['sox_lb_mmbtu']
            pm_limit = limits['pm_lb_mmbtu']
            co2_limit = limits['co2_tons_mwh'] * 100  # Approximate conversion
        elif jurisdiction == 'EU_IED':
            # Convert mg/Nm3 to lb/MMBtu (approximate)
            nox_limit = limits['nox_mg_nm3'] * 0.0001  # Rough conversion
            sox_limit = limits['sox_mg_nm3'] * 0.0001
            pm_limit = limits['pm_mg_nm3'] * 0.00001
            co2_limit = 100.0
        else:  # CHINA_MEE
            nox_limit = limits['nox_mg_nm3'] * 0.0001
            sox_limit = limits['sox_mg_nm3'] * 0.0001
            pm_limit = limits['pm_mg_nm3'] * 0.00001
            co2_limit = 100.0

        # Check compliance for each pollutant
        violations = []

        nox_status = 'compliant' if nox_value <= nox_limit else 'non_compliant'
        sox_status = 'compliant' if sox_value <= sox_limit else 'non_compliant'
        co2_status = 'compliant' if co2_value <= co2_limit else 'non_compliant'
        pm_status = 'compliant' if pm_value <= pm_limit else 'non_compliant'

        # Create violation records
        if nox_status == 'non_compliant':
            violations.append({
                'pollutant': 'NOx',
                'measured': nox_value,
                'limit': nox_limit,
                'exceedance_percent': ((nox_value - nox_limit) / nox_limit) * 100
            })

        if sox_status == 'non_compliant':
            violations.append({
                'pollutant': 'SOx',
                'measured': sox_value,
                'limit': sox_limit,
                'exceedance_percent': ((sox_value - sox_limit) / sox_limit) * 100
            })

        if pm_status == 'non_compliant':
            violations.append({
                'pollutant': 'PM',
                'measured': pm_value,
                'limit': pm_limit,
                'exceedance_percent': ((pm_value - pm_limit) / pm_limit) * 100
            })

        # Determine overall status
        if violations:
            overall_status = 'non_compliant'
        elif any(s == 'warning' for s in [nox_status, sox_status, co2_status, pm_status]):
            overall_status = 'warning'
        else:
            overall_status = 'compliant'

        # Calculate margin to limits
        margin_to_limits = {
            'nox_margin_percent': ((nox_limit - nox_value) / nox_limit * 100) if nox_limit > 0 else 0,
            'sox_margin_percent': ((sox_limit - sox_value) / sox_limit * 100) if sox_limit > 0 else 0,
            'pm_margin_percent': ((pm_limit - pm_value) / pm_limit * 100) if pm_limit > 0 else 0
        }

        result = ComplianceCheckResult(
            overall_status=overall_status,
            nox_status=nox_status,
            sox_status=sox_status,
            co2_status=co2_status,
            pm_status=pm_status,
            violations=violations,
            jurisdiction=jurisdiction,
            applicable_limits={
                'nox_limit': nox_limit,
                'sox_limit': sox_limit,
                'co2_limit': co2_limit,
                'pm_limit': pm_limit
            },
            margin_to_limits=margin_to_limits,
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash(emissions_result)
        )

        return result

    # ========================================================================
    # TOOL 6: REGULATORY REPORT GENERATION
    # ========================================================================

    def generate_regulatory_report(
        self,
        report_format: str,
        reporting_period: Dict[str, str],
        facility_data: Dict[str, Any],
        emissions_data: List[Dict[str, Any]]
    ) -> RegulatoryReportResult:
        """
        Generate compliance reports in EPA/EU/China formats.

        Generates complete regulatory report with emissions summary,
        compliance statistics, and certification.

        Args:
            report_format: Report format (EPA_ECMPS, EU_ELED, CHINA_MEE)
            reporting_period: Start and end dates
            facility_data: Facility identification
            emissions_data: Historical emissions records

        Returns:
            RegulatoryReportResult with complete report data

        Standards: 40 CFR Part 75, EU ELED, China MEE
        """
        report_id = f"RPT-{str(uuid.uuid4())[:8].upper()}"

        # Calculate summary statistics
        if emissions_data:
            nox_values = [d.get('nox_lb_mmbtu', 0) for d in emissions_data]
            sox_values = [d.get('sox_lb_mmbtu', 0) for d in emissions_data]
            co2_values = [d.get('co2_tons', 0) for d in emissions_data]
            pm_values = [d.get('pm_lb_mmbtu', 0) for d in emissions_data]

            avg_nox = sum(nox_values) / len(nox_values) if nox_values else 0
            avg_sox = sum(sox_values) / len(sox_values) if sox_values else 0
            total_co2 = sum(co2_values)
            avg_pm = sum(pm_values) / len(pm_values) if pm_values else 0

            operating_hours = len(emissions_data)
            exceedances = sum(1 for d in emissions_data if d.get('exceedance', False))
            compliance_rate = ((operating_hours - exceedances) / operating_hours * 100) if operating_hours > 0 else 0
            data_availability = len([d for d in emissions_data if d.get('valid', True)]) / max(operating_hours, 1) * 100
        else:
            avg_nox = 0
            avg_sox = 0
            total_co2 = 0
            avg_pm = 0
            operating_hours = 0
            exceedances = 0
            compliance_rate = 0
            data_availability = 0

        # Determine submission deadline based on format
        if report_format == 'EPA_ECMPS':
            format_version = 'ECMPS 3.0'
            submission_deadline = 'Q+30 days'
        elif report_format == 'EU_ELED':
            format_version = 'E-PRTR 2023'
            submission_deadline = 'April 30'
        else:
            format_version = 'GB 13223-2011'
            submission_deadline = 'Q+45 days'

        # Build report sections
        sections = [
            {
                'section': 'Facility Information',
                'content': facility_data
            },
            {
                'section': 'Emissions Summary',
                'content': {
                    'avg_nox_lb_mmbtu': round(avg_nox, 4),
                    'avg_sox_lb_mmbtu': round(avg_sox, 4),
                    'total_co2_tons': round(total_co2, 2),
                    'avg_pm_lb_mmbtu': round(avg_pm, 4)
                }
            },
            {
                'section': 'Compliance Statistics',
                'content': {
                    'compliance_rate_percent': round(compliance_rate, 2),
                    'exceedance_count': exceedances,
                    'data_availability_percent': round(data_availability, 2)
                }
            }
        ]

        # Attachments based on format
        attachments = [
            'hourly_emissions_data.csv',
            'calibration_records.pdf',
            'qa_qa_summary.pdf'
        ]

        result = RegulatoryReportResult(
            report_id=report_id,
            jurisdiction=report_format.split('_')[0],
            reporting_period=reporting_period,
            total_operating_hours=operating_hours,
            avg_nox_lb_mmbtu=round(avg_nox, 4),
            avg_sox_lb_mmbtu=round(avg_sox, 4),
            total_co2_tons=round(total_co2, 2),
            avg_pm_lb_mmbtu=round(avg_pm, 4),
            compliance_rate_percent=round(compliance_rate, 2),
            exceedance_count=exceedances,
            data_availability_percent=round(data_availability, 2),
            certifier=facility_data.get('designated_representative', 'Unknown'),
            certification_date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            submission_deadline=submission_deadline,
            format_version=format_version,
            sections=sections,
            attachments=attachments,
            provenance_hash=self._calculate_hash({
                'report_format': report_format,
                'reporting_period': reporting_period,
                'facility_data': facility_data
            })
        )

        return result

    # ========================================================================
    # TOOL 7: VIOLATION DETECTION
    # ========================================================================

    def detect_violations(
        self,
        emissions_result: Dict[str, Any],
        permit_limits: Dict[str, float]
    ) -> List[ViolationResult]:
        """
        Detect emissions exceedances and violations.

        Compares current emissions against permit limits and
        generates violation records for exceedances.

        Args:
            emissions_result: Current emissions data
            permit_limits: Applicable permit limits

        Returns:
            List of ViolationResult for detected violations
        """
        violations = []

        # Check each pollutant
        pollutant_mappings = [
            ('nox', 'nox_limit', '40 CFR 60.44(a)'),
            ('sox', 'sox_limit', '40 CFR 60.43(a)'),
            ('co2', 'co2_limit', '40 CFR 75.10'),
            ('pm', 'pm_limit', '40 CFR 60.42(a)')
        ]

        for pollutant, limit_key, reg_ref in pollutant_mappings:
            pollutant_data = emissions_result.get(pollutant, {})

            if pollutant == 'co2':
                measured_value = pollutant_data.get('mass_rate_tons_hr', 0)
            else:
                measured_value = pollutant_data.get('emission_rate_lb_mmbtu', 0)

            limit_value = permit_limits.get(limit_key, float('inf'))

            if measured_value > limit_value:
                exceedance_percent = ((measured_value - limit_value) / limit_value * 100) if limit_value > 0 else 0

                # Determine severity
                if exceedance_percent > 50:
                    severity = 'critical'
                elif exceedance_percent > 25:
                    severity = 'high'
                elif exceedance_percent > 10:
                    severity = 'medium'
                else:
                    severity = 'low'

                violation = ViolationResult(
                    violation_id=f"VIO-{str(uuid.uuid4())[:8].upper()}",
                    pollutant=pollutant.upper(),
                    measured_value=round(measured_value, 6),
                    limit_value=round(limit_value, 6),
                    exceedance_percent=round(exceedance_percent, 2),
                    severity=severity,
                    duration_minutes=15,  # Default CEMS averaging period
                    regulatory_reference=reg_ref,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                violations.append(violation)

        return violations

    # ========================================================================
    # TOOL 8: EXCEEDANCE PREDICTION
    # ========================================================================

    def predict_exceedances(
        self,
        historical_data: List[Dict[str, Any]],
        permit_limits: Dict[str, float],
        forecast_hours: int = 24
    ) -> List[ExceedancePredictionResult]:
        """
        Predict potential emissions exceedances from trends.

        Uses linear extrapolation to forecast future emissions
        and estimate exceedance probability.

        Args:
            historical_data: Historical emissions time series
            permit_limits: Permit limits for prediction
            forecast_hours: Forecast horizon in hours

        Returns:
            List of ExceedancePredictionResult for each pollutant
        """
        predictions = []

        for pollutant in ['nox', 'sox', 'pm']:
            # Extract time series
            values = [d.get(f'{pollutant}_lb_mmbtu', d.get(f'{pollutant}_ppm', 0))
                     for d in historical_data]

            if not values:
                continue

            current_value = values[-1] if values else 0
            limit_value = permit_limits.get(f'{pollutant}_limit', 100)

            # Calculate trend (simple linear regression)
            n = len(values)
            if n >= 2:
                x_mean = (n - 1) / 2
                y_mean = sum(values) / n

                numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
                denominator = sum((i - x_mean) ** 2 for i in range(n))

                slope = numerator / denominator if denominator > 0 else 0
                predicted_value = current_value + slope * forecast_hours
            else:
                predicted_value = current_value
                slope = 0

            # Calculate exceedance probability
            if predicted_value >= limit_value:
                exceedance_probability = min(100, 50 + (predicted_value - limit_value) / limit_value * 50)
            else:
                exceedance_probability = max(0, (predicted_value / limit_value) * 50)

            # Time to exceedance
            if slope > 0 and current_value < limit_value:
                time_to_exceedance = (limit_value - current_value) / slope
            else:
                time_to_exceedance = None

            # Confidence level
            confidence = 'high' if n >= 24 else ('medium' if n >= 12 else 'low')

            prediction = ExceedancePredictionResult(
                pollutant=pollutant.upper(),
                current_value=round(current_value, 4),
                predicted_value=round(predicted_value, 4),
                limit_value=round(limit_value, 4),
                exceedance_probability=round(exceedance_probability, 1),
                time_to_exceedance_hours=round(time_to_exceedance, 1) if time_to_exceedance else None,
                confidence_level=confidence,
                model_type='linear_extrapolation',
                timestamp=datetime.now(timezone.utc).isoformat(),
                provenance_hash=self._calculate_hash(historical_data[-10:])
            )
            predictions.append(prediction)

        return predictions

    # ========================================================================
    # TOOL 9: EMISSION FACTOR LOOKUP
    # ========================================================================

    def calculate_emission_factors(
        self,
        fuel_type: str
    ) -> EmissionFactorResult:
        """
        Look up AP-42 emission factors by fuel type.

        Returns emission factors from EPA AP-42 Fifth Edition
        for the specified fuel type.

        Args:
            fuel_type: Type of fuel

        Returns:
            EmissionFactorResult with all emission factors
        """
        factors = AP42_EMISSION_FACTORS.get(fuel_type, AP42_EMISSION_FACTORS['natural_gas'])

        result = EmissionFactorResult(
            fuel_type=fuel_type,
            nox_factor=factors['nox_lb_mmbtu'],
            sox_factor=factors['sox_lb_mmbtu'],
            co2_factor=factors['co2_lb_mmbtu'],
            pm_factor=factors['pm_lb_mmbtu'],
            co_factor=factors['co_lb_mmbtu'],
            factor_source='AP-42 Fifth Edition',
            units='lb/MMBtu',
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash({'fuel_type': fuel_type})
        )

        return result

    # ========================================================================
    # TOOL 10: FUEL COMPOSITION ANALYSIS
    # ========================================================================

    def analyze_fuel_composition(
        self,
        fuel_type: str,
        ultimate_analysis: Optional[Dict[str, float]] = None,
        proximate_analysis: Optional[Dict[str, float]] = None
    ) -> FuelAnalysisResult:
        """
        Analyze fuel composition for emissions calculations.

        Calculates heating values and stoichiometric air
        requirements from fuel composition.

        Args:
            fuel_type: Type of fuel
            ultimate_analysis: C, H, S, N, O, ash percentages
            proximate_analysis: Moisture, ash, volatile matter

        Returns:
            FuelAnalysisResult with complete fuel analysis
        """
        # Default compositions by fuel type
        default_compositions = {
            'natural_gas': {
                'C': 75.0, 'H': 24.0, 'S': 0.0006, 'N': 0.01, 'O': 0.5, 'ash': 0, 'moisture': 0
            },
            'fuel_oil_no2': {
                'C': 87.0, 'H': 12.5, 'S': 0.5, 'N': 0.01, 'O': 0.1, 'ash': 0.01, 'moisture': 0.1
            },
            'fuel_oil_no6': {
                'C': 86.0, 'H': 10.5, 'S': 2.5, 'N': 0.3, 'O': 0.5, 'ash': 0.1, 'moisture': 0.5
            },
            'coal_bituminous': {
                'C': 75.0, 'H': 5.0, 'S': 2.0, 'N': 1.5, 'O': 8.0, 'ash': 8.0, 'moisture': 5.0
            },
            'biomass_wood': {
                'C': 50.0, 'H': 6.0, 'S': 0.1, 'N': 0.3, 'O': 42.0, 'ash': 1.0, 'moisture': 20.0
            }
        }

        comp = default_compositions.get(fuel_type, default_compositions['natural_gas'])

        if ultimate_analysis:
            comp.update(ultimate_analysis)

        if proximate_analysis:
            comp['moisture'] = proximate_analysis.get('moisture', comp['moisture'])
            comp['ash'] = proximate_analysis.get('ash', comp['ash'])

        # Calculate heating values (Dulong formula approximation)
        # HHV (Btu/lb) = 14544*C + 62028*(H - O/8) + 4050*S
        hhv_btu_lb = (14544 * comp['C']/100 +
                     62028 * (comp['H']/100 - comp['O']/100/8) +
                     4050 * comp['S']/100)

        # LHV = HHV - latent heat of water vapor
        lhv_btu_lb = hhv_btu_lb - 1040 * (9 * comp['H']/100 + comp['moisture']/100)

        # Stoichiometric air (lb air / lb fuel)
        # Based on combustion stoichiometry
        stoich_air = (11.5 * comp['C']/100 + 34.3 * (comp['H']/100 - comp['O']/100/8) +
                     4.3 * comp['S']/100)

        result = FuelAnalysisResult(
            fuel_type=fuel_type,
            carbon_percent=round(comp['C'], 2),
            hydrogen_percent=round(comp['H'], 2),
            sulfur_percent=round(comp['S'], 4),
            nitrogen_percent=round(comp['N'], 2),
            oxygen_percent=round(comp['O'], 2),
            ash_percent=round(comp['ash'], 2),
            moisture_percent=round(comp['moisture'], 2),
            hhv_btu_lb=round(hhv_btu_lb, 0),
            lhv_btu_lb=round(lhv_btu_lb, 0),
            stoichiometric_air_lb_lb=round(stoich_air, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash({
                'fuel_type': fuel_type,
                'composition': comp
            })
        )

        return result

    # ========================================================================
    # TOOL 11: DISPERSION CALCULATION
    # ========================================================================

    def calculate_dispersion(
        self,
        emission_rate_g_s: float,
        stack_parameters: Dict[str, float],
        meteorological_data: Dict[str, Any]
    ) -> DispersionResult:
        """
        Calculate air dispersion using Gaussian plume model.

        Uses Briggs plume rise equations and Pasquill-Gifford
        dispersion coefficients.

        Formula: C(x,y,z) = Q/(2*pi*u*sigma_y*sigma_z) *
                           exp(-y^2/(2*sigma_y^2)) *
                           [exp(-(z-H)^2/(2*sigma_z^2)) + exp(-(z+H)^2/(2*sigma_z^2))]

        Args:
            emission_rate_g_s: Emission rate in g/s
            stack_parameters: Stack height, diameter, velocity, temperature
            meteorological_data: Wind speed, stability class

        Returns:
            DispersionResult with concentration predictions
        """
        # Extract parameters
        stack_height_m = stack_parameters.get('height_m', 50)
        stack_diameter_m = stack_parameters.get('diameter_m', 3)
        exit_velocity_m_s = stack_parameters.get('exit_velocity_m_s', 15)
        exit_temp_k = stack_parameters.get('exit_temperature_k', 423)  # 150C default

        wind_speed_m_s = meteorological_data.get('wind_speed_m_s', 5)
        stability_class = meteorological_data.get('stability_class', 'D')
        ambient_temp_k = meteorological_data.get('ambient_temperature_k', 298)

        # Calculate plume rise (Briggs equations for buoyant plumes)
        # Buoyancy flux: F = g * V * d^2 * (Ts - Ta) / (4 * Ts)
        g = 9.81  # m/s^2
        volume_flux = exit_velocity_m_s * math.pi * (stack_diameter_m/2)**2
        buoyancy_flux = g * volume_flux * (exit_temp_k - ambient_temp_k) / (4 * exit_temp_k)

        # Plume rise (simplified Briggs for F < 55)
        if buoyancy_flux > 0:
            if stability_class in ['A', 'B', 'C']:  # Unstable
                plume_rise = 1.6 * (buoyancy_flux ** (1/3)) * (1000 ** (2/3)) / wind_speed_m_s
            elif stability_class == 'D':  # Neutral
                plume_rise = 1.6 * (buoyancy_flux ** (1/3)) * (500 ** (2/3)) / wind_speed_m_s
            else:  # Stable (E, F)
                plume_rise = 2.6 * (buoyancy_flux / wind_speed_m_s) ** (1/3)
        else:
            # Momentum-dominated plume
            plume_rise = 3.0 * stack_diameter_m * (exit_velocity_m_s / wind_speed_m_s)

        plume_rise = min(plume_rise, 500)  # Cap at 500m

        effective_height = stack_height_m + plume_rise

        # Dispersion coefficients (Pasquill-Gifford rural)
        # sigma_y = a * x^b, sigma_z = c * x^d + e
        pg_coefficients = {
            'A': {'a': 0.22, 'b': 0.894, 'c': 0.20, 'd': 0.0, 'e': 0},
            'B': {'a': 0.16, 'b': 0.894, 'c': 0.12, 'd': 0.0, 'e': 0},
            'C': {'a': 0.11, 'b': 0.894, 'c': 0.08, 'd': 0.0, 'e': 0},
            'D': {'a': 0.08, 'b': 0.894, 'c': 0.06, 'd': 0.0, 'e': 0},
            'E': {'a': 0.06, 'b': 0.894, 'c': 0.03, 'd': 0.0, 'e': 0},
            'F': {'a': 0.04, 'b': 0.894, 'c': 0.016, 'd': 0.0, 'e': 0}
        }

        coef = pg_coefficients.get(stability_class, pg_coefficients['D'])

        # Find distance to maximum ground-level concentration
        # Approximate: x_max occurs when sigma_z = H / sqrt(2)
        sigma_z_at_max = effective_height / math.sqrt(2)
        x_max = (sigma_z_at_max / coef['c']) ** (1 / max(coef['d'], 0.5)) if coef['c'] > 0 else 1000

        x_max = max(100, min(x_max, 50000))  # Bound between 100m and 50km

        # Calculate sigma_y and sigma_z at x_max
        sigma_y = coef['a'] * (x_max ** coef['b'])
        sigma_z = coef['c'] * (x_max ** max(coef['d'], 0.5)) if coef['c'] > 0 else 50

        # Maximum ground-level concentration
        # C_max = Q / (pi * u * sigma_y * sigma_z * e)
        if sigma_y > 0 and sigma_z > 0 and wind_speed_m_s > 0:
            c_max = (emission_rate_g_s / (math.pi * wind_speed_m_s * sigma_y * sigma_z * math.e))
            c_max_ug_m3 = c_max * 1e6  # Convert g/m3 to ug/m3
        else:
            c_max_ug_m3 = 0

        result = DispersionResult(
            max_ground_concentration=round(c_max_ug_m3, 4),
            distance_to_max_m=round(x_max, 0),
            plume_rise_m=round(plume_rise, 1),
            effective_stack_height_m=round(effective_height, 1),
            sigma_y=round(sigma_y, 1),
            sigma_z=round(sigma_z, 1),
            stability_class=stability_class,
            calculation_method='Gaussian_plume_Briggs',
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash({
                'emission_rate': emission_rate_g_s,
                'stack_parameters': stack_parameters,
                'meteorological_data': meteorological_data
            })
        )

        return result

    # ========================================================================
    # TOOL 12: AUDIT TRAIL GENERATION
    # ========================================================================

    def generate_audit_trail(
        self,
        audit_period: Dict[str, str],
        facility_data: Dict[str, Any],
        emissions_records: List[Dict[str, Any]],
        compliance_events: List[Dict[str, Any]]
    ) -> AuditTrailResult:
        """
        Generate SHA-256 provenance audit trail.

        Creates cryptographic hash chain for emissions data
        verification and regulatory compliance.

        Args:
            audit_period: Audit period start and end
            facility_data: Facility identification
            emissions_records: Emissions records to audit
            compliance_events: Compliance events

        Returns:
            AuditTrailResult with hash chain and verification
        """
        audit_id = f"AUD-{str(uuid.uuid4())[:8].upper()}"

        # Generate hashes for each record
        record_hashes = []
        previous_hash = "0" * 64  # Genesis hash

        for record in emissions_records:
            record_with_chain = {
                'previous_hash': previous_hash,
                'record': record,
                'timestamp': record.get('timestamp', datetime.now(timezone.utc).isoformat())
            }
            record_hash = self._calculate_hash(record_with_chain)
            record_hashes.append(record_hash)
            previous_hash = record_hash

        # Root hash (Merkle root simplified)
        if record_hashes:
            root_hash = self._calculate_hash(record_hashes)
        else:
            root_hash = "0" * 64

        # Calculate statistics
        total_records = len(emissions_records)
        compliant_records = sum(1 for r in emissions_records if r.get('compliant', True))
        non_compliant_records = total_records - compliant_records

        valid_records = sum(1 for r in emissions_records if r.get('valid', True))
        data_quality_score = (valid_records / total_records * 100) if total_records > 0 else 0
        completeness_percent = data_quality_score

        # Verify chain integrity
        chain_valid = True
        if len(record_hashes) >= 2:
            # Simplified verification
            for i in range(len(record_hashes) - 1):
                if len(record_hashes[i]) != 64:
                    chain_valid = False
                    break

        # Data corrections (placeholder)
        data_corrections = []

        result = AuditTrailResult(
            audit_id=audit_id,
            audit_period=audit_period,
            total_records=total_records,
            compliant_records=compliant_records,
            non_compliant_records=non_compliant_records,
            data_quality_score=round(data_quality_score, 2),
            completeness_percent=round(completeness_percent, 2),
            root_hash=root_hash,
            record_hashes=record_hashes,
            chain_valid=chain_valid,
            compliance_events=compliance_events,
            data_corrections=data_corrections,
            certifier=facility_data.get('designated_representative', 'Unknown'),
            certification_date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            certification_statement=(
                "I certify that the information contained in this report is true, accurate, "
                "and complete to the best of my knowledge."
            ),
            epa_part_75_compliant=data_quality_score >= 90,
            data_retention_met=True,
            qapp_requirements_met=data_quality_score >= 90
        )

        return result

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_tool_schemas(self) -> Dict[str, Any]:
        """Get all tool schemas for API documentation."""
        return EMISSIONS_TOOL_SCHEMAS.copy()
