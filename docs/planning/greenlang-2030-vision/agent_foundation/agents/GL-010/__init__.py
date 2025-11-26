# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH - EmissionsComplianceAgent Package.

Zero-hallucination emissions compliance monitoring for industrial processes.
This package provides comprehensive emissions monitoring including NOx, SOx,
CO2, and PM calculations, multi-jurisdiction regulatory compliance (EPA, EU IED,
China MEE), violation detection, predictive analytics, and audit trail generation.

Emissions Coverage:
- NOx (Nitrogen Oxides): Thermal + Fuel + Prompt NOx mechanisms
- SOx (Sulfur Oxides): Fuel sulfur-based stoichiometry
- CO2 (Carbon Dioxide): Combustion stoichiometry
- PM (Particulate Matter): PM10/PM2.5 fractions

Standards Compliance:
- EPA 40 CFR Part 60 - Standards of Performance for New Stationary Sources
- EPA 40 CFR Part 75 - Continuous Emissions Monitoring
- EPA Method 19 - Sulfur Dioxide Removal and Particulate Matter
- EU Industrial Emissions Directive 2010/75/EU
- China MEE GB 13223-2011 - Emission Standards

Example:
    >>> from gl_010 import EmissionsComplianceOrchestrator, EmissionsComplianceConfig
    >>> config = EmissionsComplianceConfig(jurisdiction="EPA")
    >>> orchestrator = EmissionsComplianceOrchestrator(config)
    >>> result = await orchestrator.execute({
    ...     'operation_mode': 'monitor',
    ...     'cems_data': {'nox_ppm': 45, 'o2_percent': 3.0},
    ...     'fuel_data': {'fuel_type': 'natural_gas', 'heat_input_mmbtu_hr': 100}
    ... })
    >>> print(f"NOx: {result['emissions']['nox_ppm']} ppm")
    >>> print(f"Compliance: {result['compliance_status']}")

Modules:
    emissions_compliance_orchestrator: Main orchestrator class
    tools: Deterministic calculation tools
    config: Pydantic configuration classes
    main: FastAPI application entry point
    greenlang: Determinism utilities

Author: GreenLang Foundation
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Foundation"
__agent_id__ = "GL-010"
__codename__ = "EMISSIONWATCH"

# Core exports
from .config import (
    EmissionsComplianceConfig,
    NOxConfig,
    SOxConfig,
    CO2Config,
    PMConfig,
    RegulatoryLimitsConfig,
    CEMSConfig,
    AlertConfig,
    ReportingConfig,
    IntegrationConfig,
    CacheConfig,
    MonitoringConfig,
    Jurisdiction,
    PollutantType,
    AlertSeverity,
    ReportFormat,
    FuelType,
    ControlDeviceType,
    create_config,
    load_config_from_file
)

from .tools import (
    EmissionsComplianceTools,
    NOxEmissionsResult,
    SOxEmissionsResult,
    CO2EmissionsResult,
    PMEmissionsResult,
    ComplianceCheckResult,
    ViolationResult,
    RegulatoryReportResult,
    ExceedancePredictionResult,
    EmissionFactorResult,
    DispersionResult,
    AuditTrailResult,
    FuelAnalysisResult,
    EMISSIONS_TOOL_SCHEMAS,
    AP42_EMISSION_FACTORS,
    REGULATORY_LIMITS,
    F_FACTORS,
    MOLECULAR_WEIGHTS
)

from .emissions_compliance_orchestrator import (
    EmissionsComplianceOrchestrator,
    OperationMode,
    ComplianceStatus,
    ValidationStatus,
    DataQualityCode,
    ThreadSafeCache,
    PerformanceMetrics,
    RetryHandler,
    create_orchestrator
)

# Greenlang utilities
try:
    from .greenlang import (
        DeterministicClock,
        DeterminismValidator,
        deterministic_uuid,
        calculate_provenance_hash,
        create_emissions_uuid,
        create_audit_hash
    )
except ImportError:
    # Greenlang module may not be available
    DeterministicClock = None
    DeterminismValidator = None
    deterministic_uuid = None
    calculate_provenance_hash = None
    create_emissions_uuid = None
    create_audit_hash = None

# FastAPI app (optional import)
try:
    from .main import app
except ImportError:
    app = None

# All public exports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent_id__",
    "__codename__",

    # Configuration
    "EmissionsComplianceConfig",
    "NOxConfig",
    "SOxConfig",
    "CO2Config",
    "PMConfig",
    "RegulatoryLimitsConfig",
    "CEMSConfig",
    "AlertConfig",
    "ReportingConfig",
    "IntegrationConfig",
    "CacheConfig",
    "MonitoringConfig",
    "Jurisdiction",
    "PollutantType",
    "AlertSeverity",
    "ReportFormat",
    "FuelType",
    "ControlDeviceType",
    "create_config",
    "load_config_from_file",

    # Tools
    "EmissionsComplianceTools",
    "NOxEmissionsResult",
    "SOxEmissionsResult",
    "CO2EmissionsResult",
    "PMEmissionsResult",
    "ComplianceCheckResult",
    "ViolationResult",
    "RegulatoryReportResult",
    "ExceedancePredictionResult",
    "EmissionFactorResult",
    "DispersionResult",
    "AuditTrailResult",
    "FuelAnalysisResult",
    "EMISSIONS_TOOL_SCHEMAS",
    "AP42_EMISSION_FACTORS",
    "REGULATORY_LIMITS",
    "F_FACTORS",
    "MOLECULAR_WEIGHTS",

    # Orchestrator
    "EmissionsComplianceOrchestrator",
    "OperationMode",
    "ComplianceStatus",
    "ValidationStatus",
    "DataQualityCode",
    "ThreadSafeCache",
    "PerformanceMetrics",
    "RetryHandler",
    "create_orchestrator",

    # Greenlang utilities
    "DeterministicClock",
    "DeterminismValidator",
    "deterministic_uuid",
    "calculate_provenance_hash",
    "create_emissions_uuid",
    "create_audit_hash",

    # FastAPI app
    "app"
]


def get_agent_info() -> dict:
    """
    Get agent information.

    Returns:
        Dictionary with agent identification and capabilities
    """
    return {
        "agent_id": __agent_id__,
        "codename": __codename__,
        "full_name": "EmissionsComplianceAgent",
        "version": __version__,
        "author": __author__,
        "description": "Zero-hallucination emissions compliance monitoring for industrial processes",
        "deterministic": True,
        "standards": [
            "EPA 40 CFR Part 60",
            "EPA 40 CFR Part 75",
            "EPA Method 19",
            "EU IED 2010/75/EU",
            "China MEE GB 13223-2011"
        ],
        "jurisdictions": ["EPA", "EU_IED", "CHINA_MEE", "CARB", "TCEQ"],
        "pollutants": ["NOx", "SOx", "CO2", "PM", "CO", "Opacity"],
        "operation_modes": [mode.value for mode in OperationMode],
        "tool_count": len(EMISSIONS_TOOL_SCHEMAS),
        "capabilities": [
            "Real-time CEMS monitoring",
            "Multi-jurisdiction compliance checking (EPA, EU, China)",
            "NOx/SOx/CO2/PM emissions calculations",
            "Violation detection and alerting",
            "Exceedance prediction",
            "Regulatory report generation (ECMPS, E-PRTR)",
            "Compliance audit trail with SHA-256 provenance",
            "Gaussian plume dispersion modeling",
            "Data quality validation (EPA Part 75)"
        ]
    }
