# -*- coding: utf-8 -*-
"""
Supplier Risk Scorer Reference Data - AGENT-EUDR-017

Comprehensive reference data for the Supplier Risk Scorer Agent covering
supplier risk benchmarks, certification schemes, and EUDR document requirements.

This package provides three reference data modules:

1. supplier_risk_database:
   - SAMPLE_SUPPLIERS: 30+ sample supplier profiles with supplier_id, name,
     type, country, commodities, and baseline risk indicators
   - RISK_FACTOR_BENCHMARKS: benchmark thresholds for each of the 8 risk
     factors (geographic_sourcing, compliance_history, documentation_quality,
     certification_status, traceability_completeness, financial_stability,
     environmental_performance, social_compliance)
   - INDUSTRY_AVERAGES: average risk scores by commodity and region
   - PEER_GROUP_DEFINITIONS: peer group criteria (same commodity, same region, same size)
   - NON_CONFORMANCE_SEVERITY_MATRIX: mapping of issue types to severity levels
     (minor, major, critical)

2. certification_schemes:
   - CERTIFICATION_SCHEMES: dict of 8 major schemes (FSC, PEFC, RSPO,
     Rainforest Alliance, UTZ, Organic, Fair Trade, ISCC) with metadata
     (name, type, commodities_covered, regions, validity_period,
     chain_of_custody_types, accreditation_body, website)
   - SCHEME_EQUIVALENCES: mapping of equivalent certifications across schemes
   - ACCREDITED_CERTIFICATION_BODIES: list of accredited CBs per scheme
   - CERTIFICATION_REQUIREMENTS: per-scheme requirements (documents, audits, frequency)
   - COMMODITY_SCHEME_MAPPING: which schemes apply to which EUDR commodities

3. document_requirements:
   - EUDR_REQUIRED_DOCUMENTS: per-commodity list of required document types
     (geolocation, DDS reference, product description, quantity declaration,
     harvest date, compliance declaration, certificate, trade license,
     phytosanitary) with descriptions
   - DOCUMENT_TEMPLATES: template metadata for each document type
   - VALIDATION_RULES: per-document validation rules (required fields, formats, ranges)
   - EXPIRY_POLICIES: expiry periods for each document type
   - LANGUAGE_REQUIREMENTS: accepted languages per EU member state

All reference data uses ISO 3166-1 alpha-3 country codes and is designed
for deterministic, zero-hallucination supplier risk scoring per EU 2023/1115.

Example:
    >>> from greenlang.agents.eudr.supplier_risk_scorer.reference_data import (
    ...     SAMPLE_SUPPLIERS,
    ...     CERTIFICATION_SCHEMES,
    ...     EUDR_REQUIRED_DOCUMENTS,
    ...     get_supplier,
    ...     get_scheme,
    ...     get_required_docs,
    ... )
    >>>
    >>> # Get sample supplier
    >>> supplier = get_supplier("SUP-BRA-001")
    >>> assert supplier["country"] == "BRA"
    >>>
    >>> # Get certification scheme
    >>> fsc_scheme = get_scheme("FSC")
    >>> assert "wood" in fsc_scheme["commodities_covered"]
    >>>
    >>> # Get required documents for commodity
    >>> docs = get_required_docs("cattle")
    >>> assert "geolocation" in [d["type"] for d in docs]

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-017
Agent ID: GL-EUDR-SRS-017
Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 11, 31
Status: Production Ready
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module 1: Supplier Risk Database
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.supplier_risk_scorer.reference_data.supplier_risk_database import (
    DATA_VERSION as SUPPLIER_DATA_VERSION,
    DATA_SOURCES as SUPPLIER_DATA_SOURCES,
    SAMPLE_SUPPLIERS,
    RISK_FACTOR_BENCHMARKS,
    INDUSTRY_AVERAGES,
    PEER_GROUP_DEFINITIONS,
    NON_CONFORMANCE_SEVERITY_MATRIX,
    get_supplier,
    get_benchmarks,
    get_industry_average,
    get_peer_group,
    get_nc_severity,
)

# ---------------------------------------------------------------------------
# Module 2: Certification Schemes
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.supplier_risk_scorer.reference_data.certification_schemes import (
    DATA_VERSION as CERT_DATA_VERSION,
    DATA_SOURCES as CERT_DATA_SOURCES,
    CERTIFICATION_SCHEMES,
    SCHEME_EQUIVALENCES,
    ACCREDITED_CERTIFICATION_BODIES,
    CERTIFICATION_REQUIREMENTS,
    COMMODITY_SCHEME_MAPPING,
    get_scheme,
    get_equivalences,
    is_accredited,
    get_requirements,
    get_schemes_for_commodity,
)

# ---------------------------------------------------------------------------
# Module 3: Document Requirements
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.supplier_risk_scorer.reference_data.document_requirements import (
    DATA_VERSION as DOC_DATA_VERSION,
    DATA_SOURCES as DOC_DATA_SOURCES,
    EUDR_REQUIRED_DOCUMENTS,
    DOCUMENT_TEMPLATES,
    VALIDATION_RULES,
    EXPIRY_POLICIES,
    LANGUAGE_REQUIREMENTS,
    get_required_docs,
    get_template,
    get_validation_rules,
    get_expiry_policy,
    get_language_requirements,
)


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    # Supplier risk database
    "SUPPLIER_DATA_VERSION",
    "SUPPLIER_DATA_SOURCES",
    "SAMPLE_SUPPLIERS",
    "RISK_FACTOR_BENCHMARKS",
    "INDUSTRY_AVERAGES",
    "PEER_GROUP_DEFINITIONS",
    "NON_CONFORMANCE_SEVERITY_MATRIX",
    "get_supplier",
    "get_benchmarks",
    "get_industry_average",
    "get_peer_group",
    "get_nc_severity",
    # Certification schemes
    "CERT_DATA_VERSION",
    "CERT_DATA_SOURCES",
    "CERTIFICATION_SCHEMES",
    "SCHEME_EQUIVALENCES",
    "ACCREDITED_CERTIFICATION_BODIES",
    "CERTIFICATION_REQUIREMENTS",
    "COMMODITY_SCHEME_MAPPING",
    "get_scheme",
    "get_equivalences",
    "is_accredited",
    "get_requirements",
    "get_schemes_for_commodity",
    # Document requirements
    "DOC_DATA_VERSION",
    "DOC_DATA_SOURCES",
    "EUDR_REQUIRED_DOCUMENTS",
    "DOCUMENT_TEMPLATES",
    "VALIDATION_RULES",
    "EXPIRY_POLICIES",
    "LANGUAGE_REQUIREMENTS",
    "get_required_docs",
    "get_template",
    "get_validation_rules",
    "get_expiry_policy",
    "get_language_requirements",
]
