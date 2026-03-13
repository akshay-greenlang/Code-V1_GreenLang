# -*- coding: utf-8 -*-
"""
AGENT-EUDR-039: Customs Declaration Support

EUDR customs declaration lifecycle management including CN code mapping,
HS code validation, SAD form generation, country of origin verification,
customs value calculation (CIF/FOB), EUDR Article 4 compliance checking,
and customs authority submission via NCTS/AIS/ICS2 interfaces. Provides
production-grade capabilities for managing the full lifecycle of customs
declarations for all 7 EUDR-regulated commodities (cattle, cocoa, coffee,
oil palm, rubber, soya, wood) as required by EU 2023/1115 Articles 4, 5,
6, 12, 31 and EU UCC Regulation 952/2013.

The agent sits alongside the Due Diligence Statement Creator (EUDR-032),
the Reference Number Generator (EUDR-034), and the EU Information System
Interface (EUDR-035), providing dedicated customs declaration support
that integrates with supply chain traceability (EUDR-001), geolocation
verification (EUDR-002), and country risk evaluation (EUDR-016).

Core capabilities:
    1. CNCodeMapper              -- Maps EUDR commodities to 8-digit EU
       Combined Nomenclature codes with tariff rate lookup per EU CN
       Regulation 2658/87 and EUDR Annex I
    2. HSCodeValidator           -- Validates 6-digit WCO Harmonized System
       codes for global trade compatibility and identifies EUDR-regulated
       products per WCO HS Convention
    3. DeclarationGenerator      -- Generates EU Single Administrative
       Document (SAD) forms and customs declarations per UCC Delegated
       Regulation (EU) 2015/2446 with all mandatory boxes
    4. OriginValidator           -- Verifies country of origin against
       supply chain traceability data and determines preferential origin
       eligibility (GSP, FTA) per EU UCC origin rules
    5. ValueCalculator           -- Calculates CIF/FOB customs values with
       currency conversion (ECB rates), Incoterms adjustments, and customs
       value determination per WTO Valuation Agreement
    6. ComplianceChecker         -- Verifies EUDR Article 4 compliance
       including DDS reference presence, deforestation-free status,
       legality, geolocation, supply chain, and risk assessment checks
    7. CustomsInterface          -- Submits declarations to NCTS/AIS/ICS2,
       receives Movement Reference Numbers (MRN), tracks clearance status
       with retry logic and exponential backoff

Foundational modules:
    - config.py       -- CustomsDeclarationSupportConfig with GL_EUDR_CDS_
      env var support (60+ settings)
    - models.py       -- Pydantic v2 data models with 12 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 42 Prometheus self-monitoring metrics (gl_eudr_cds_)

Agent ID: GL-EUDR-CDS-039
Module: greenlang.agents.eudr.customs_declaration_support
PRD: PRD-AGENT-EUDR-039
Regulation: EU 2023/1115 (EUDR) Articles 4, 5, 6, 12, 31;
            EU UCC 952/2013; EU CN 2658/87; WTO Valuation Agreement

Example:
    >>> from greenlang.agents.eudr.customs_declaration_support import (
    ...     CustomsDeclarationSupportConfig,
    ...     get_config,
    ...     CommodityType,
    ...     DeclarationType,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.default_port_of_entry)
    NLRTM

    >>> from greenlang.agents.eudr.customs_declaration_support import (
    ...     CustomsDeclarationService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-039 Customs Declaration Support (GL-EUDR-CDS-039)
Status: Production Ready
"""
from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-CDS-039"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "CustomsDeclarationSupportConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (12) --
    "DeclarationStatus",
    "DeclarationType",
    "CommodityType",
    "CustomsSystem",
    "Incoterms",
    "TariffType",
    "ComplianceCheckType",
    "VerificationStatus",
    "PortType",
    "DeclarationPurpose",
    "SubmissionStatus",
    "AuditAction",
    # -- Core Models (15+) --
    "CustomsDeclaration",
    "CNCodeMapping",
    "HSCode",
    "TariffCalculation",
    "CountryOriginVerification",
    "SubmissionLog",
    "ComplianceCheck",
    "PortOfEntry",
    "DeclarationLine",
    "SADForm",
    "ValueDeclaration",
    "QuantityDeclaration",
    "DeclarationSummary",
    "AuditEntry",
    "HealthStatus",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    "EUDR_COMMODITY_CN_CODES",
    "MRN_PATTERN",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics (counters) --
    "record_declaration_created",
    "record_declaration_submitted",
    "record_declaration_cleared",
    "record_declaration_rejected",
    "record_compliance_check_passed",
    "record_compliance_check_failed",
    "record_tariff_calculated",
    "record_cn_code_mapped",
    "record_hs_code_validated",
    "record_origin_verification",
    "record_value_calculation",
    "record_submission_retried",
    "record_mrn_assigned",
    "record_sad_form_generated",
    "record_amendment",
    "record_currency_conversion",
    # -- Metrics (histograms) --
    "observe_declaration_generation_duration",
    "observe_submission_duration",
    "observe_compliance_check_duration",
    "observe_tariff_calculation_duration",
    "observe_value_calculation_duration",
    "observe_cn_code_mapping_duration",
    "observe_hs_code_validation_duration",
    "observe_origin_verification_duration",
    "observe_sad_generation_duration",
    "observe_clearance_duration",
    "observe_currency_conversion_duration",
    "observe_customs_response_duration",
    # -- Metrics (gauges) --
    "set_pending_declarations",
    "set_declarations_awaiting_clearance",
    "set_average_tariff_rate",
    "set_total_customs_value_eur",
    "set_total_duty_amount_eur",
    "set_active_submissions",
    "set_compliance_pass_rate",
    "set_declarations_by_status",
    "set_declarations_by_commodity",
    "set_exchange_rate_eur_usd",
    "set_exchange_rate_eur_gbp",
    "set_exchange_rate_eur_jpy",
    "set_submission_error_rate",
    "set_average_clearance_time_hours",
    # -- Engines (7) --
    "CNCodeMapper",
    "HSCodeValidator",
    "DeclarationGenerator",
    "OriginValidator",
    "ValueCalculator",
    "ComplianceChecker",
    "CustomsInterface",
    # -- Service Facade --
    "CustomsDeclarationService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "CustomsDeclarationSupportConfig": ("config", "CustomsDeclarationSupportConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations (12)
    "DeclarationStatus": ("models", "DeclarationStatus"),
    "DeclarationType": ("models", "DeclarationType"),
    "CommodityType": ("models", "CommodityType"),
    "CustomsSystem": ("models", "CustomsSystem"),
    "Incoterms": ("models", "Incoterms"),
    "TariffType": ("models", "TariffType"),
    "ComplianceCheckType": ("models", "ComplianceCheckType"),
    "VerificationStatus": ("models", "VerificationStatus"),
    "PortType": ("models", "PortType"),
    "DeclarationPurpose": ("models", "DeclarationPurpose"),
    "SubmissionStatus": ("models", "SubmissionStatus"),
    "AuditAction": ("models", "AuditAction"),
    # Core Models (15+)
    "CustomsDeclaration": ("models", "CustomsDeclaration"),
    "CNCodeMapping": ("models", "CNCodeMapping"),
    "HSCode": ("models", "HSCode"),
    "TariffCalculation": ("models", "TariffCalculation"),
    "CountryOriginVerification": ("models", "CountryOriginVerification"),
    "SubmissionLog": ("models", "SubmissionLog"),
    "ComplianceCheck": ("models", "ComplianceCheck"),
    "PortOfEntry": ("models", "PortOfEntry"),
    "DeclarationLine": ("models", "DeclarationLine"),
    "SADForm": ("models", "SADForm"),
    "ValueDeclaration": ("models", "ValueDeclaration"),
    "QuantityDeclaration": ("models", "QuantityDeclaration"),
    "DeclarationSummary": ("models", "DeclarationSummary"),
    "AuditEntry": ("models", "AuditEntry"),
    "HealthStatus": ("models", "HealthStatus"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    "EUDR_COMMODITY_CN_CODES": ("models", "EUDR_COMMODITY_CN_CODES"),
    "MRN_PATTERN": ("models", "MRN_PATTERN"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters)
    "record_declaration_created": ("metrics", "record_declaration_created"),
    "record_declaration_submitted": ("metrics", "record_declaration_submitted"),
    "record_declaration_cleared": ("metrics", "record_declaration_cleared"),
    "record_declaration_rejected": ("metrics", "record_declaration_rejected"),
    "record_compliance_check_passed": ("metrics", "record_compliance_check_passed"),
    "record_compliance_check_failed": ("metrics", "record_compliance_check_failed"),
    "record_tariff_calculated": ("metrics", "record_tariff_calculated"),
    "record_cn_code_mapped": ("metrics", "record_cn_code_mapped"),
    "record_hs_code_validated": ("metrics", "record_hs_code_validated"),
    "record_origin_verification": ("metrics", "record_origin_verification"),
    "record_value_calculation": ("metrics", "record_value_calculation"),
    "record_submission_retried": ("metrics", "record_submission_retried"),
    "record_mrn_assigned": ("metrics", "record_mrn_assigned"),
    "record_sad_form_generated": ("metrics", "record_sad_form_generated"),
    "record_amendment": ("metrics", "record_amendment"),
    "record_currency_conversion": ("metrics", "record_currency_conversion"),
    # Metrics (histograms)
    "observe_declaration_generation_duration": ("metrics", "observe_declaration_generation_duration"),
    "observe_submission_duration": ("metrics", "observe_submission_duration"),
    "observe_compliance_check_duration": ("metrics", "observe_compliance_check_duration"),
    "observe_tariff_calculation_duration": ("metrics", "observe_tariff_calculation_duration"),
    "observe_value_calculation_duration": ("metrics", "observe_value_calculation_duration"),
    "observe_cn_code_mapping_duration": ("metrics", "observe_cn_code_mapping_duration"),
    "observe_hs_code_validation_duration": ("metrics", "observe_hs_code_validation_duration"),
    "observe_origin_verification_duration": ("metrics", "observe_origin_verification_duration"),
    "observe_sad_generation_duration": ("metrics", "observe_sad_generation_duration"),
    "observe_clearance_duration": ("metrics", "observe_clearance_duration"),
    "observe_currency_conversion_duration": ("metrics", "observe_currency_conversion_duration"),
    "observe_customs_response_duration": ("metrics", "observe_customs_response_duration"),
    # Metrics (gauges)
    "set_pending_declarations": ("metrics", "set_pending_declarations"),
    "set_declarations_awaiting_clearance": ("metrics", "set_declarations_awaiting_clearance"),
    "set_average_tariff_rate": ("metrics", "set_average_tariff_rate"),
    "set_total_customs_value_eur": ("metrics", "set_total_customs_value_eur"),
    "set_total_duty_amount_eur": ("metrics", "set_total_duty_amount_eur"),
    "set_active_submissions": ("metrics", "set_active_submissions"),
    "set_compliance_pass_rate": ("metrics", "set_compliance_pass_rate"),
    "set_declarations_by_status": ("metrics", "set_declarations_by_status"),
    "set_declarations_by_commodity": ("metrics", "set_declarations_by_commodity"),
    "set_exchange_rate_eur_usd": ("metrics", "set_exchange_rate_eur_usd"),
    "set_exchange_rate_eur_gbp": ("metrics", "set_exchange_rate_eur_gbp"),
    "set_exchange_rate_eur_jpy": ("metrics", "set_exchange_rate_eur_jpy"),
    "set_submission_error_rate": ("metrics", "set_submission_error_rate"),
    "set_average_clearance_time_hours": ("metrics", "set_average_clearance_time_hours"),
    # Engines (7)
    "CNCodeMapper": ("cn_code_mapper", "CNCodeMapper"),
    "HSCodeValidator": ("hs_code_validator", "HSCodeValidator"),
    "DeclarationGenerator": ("declaration_generator", "DeclarationGenerator"),
    "OriginValidator": ("origin_validator", "OriginValidator"),
    "ValueCalculator": ("value_calculator", "ValueCalculator"),
    "ComplianceChecker": ("compliance_checker", "ComplianceChecker"),
    "CustomsInterface": ("customs_interface", "CustomsInterface"),
    # Service Facade
    "CustomsDeclarationService": ("setup", "CustomsDeclarationService"),
    "get_service": ("setup", "get_service"),
    "reset_service": ("setup", "reset_service"),
    "lifespan": ("setup", "lifespan"),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports.

    Enables ``from greenlang.agents.eudr.customs_declaration_support import X``
    without eagerly loading all submodules at package import time.

    Args:
        name: Attribute name to look up.

    Returns:
        The lazily imported object.

    Raises:
        AttributeError: If the name is not a known export.
    """
    if name in _LAZY_IMPORTS:
        module_suffix, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(
            f"greenlang.agents.eudr.customs_declaration_support.{module_suffix}"
        )
        return getattr(mod, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string.

    Returns:
        Version string in semver format (e.g. "1.0.0").

    Example:
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata.

    Returns:
        Dictionary with agent_id, version, regulation references,
        engine listing, and capability summary for the Customs
        Declaration Support agent.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-CDS-039'
        >>> info["engine_count"]
        7
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Customs Declaration Support",
        "prd": "PRD-AGENT-EUDR-039",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["4", "5", "6", "12", "31"],
        "supplementary_regulations": [
            "EU Union Customs Code (UCC) 952/2013",
            "EU Combined Nomenclature Regulation 2658/87",
            "WTO Customs Valuation Agreement",
            "WCO Harmonized System Convention",
        ],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "eudr_commodities": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
        "customs_systems": ["NCTS", "AIS", "ICS2"],
        "supported_currencies": [
            "EUR", "USD", "GBP", "JPY", "CHF", "BRL", "IDR",
        ],
        "incoterms": [
            "EXW", "FCA", "FAS", "FOB", "CFR", "CIF",
            "CPT", "CIP", "DAP", "DPU", "DDP",
        ],
        "engines": [
            "CNCodeMapper",
            "HSCodeValidator",
            "DeclarationGenerator",
            "OriginValidator",
            "ValueCalculator",
            "ComplianceChecker",
            "CustomsInterface",
        ],
        "engine_count": 7,
        "enum_count": 12,
        "core_model_count": 15,
        "metrics_count": 42,
        "db_prefix": "gl_eudr_cds_",
        "metrics_prefix": "gl_eudr_cds_",
        "env_prefix": "GL_EUDR_CDS_",
    }
