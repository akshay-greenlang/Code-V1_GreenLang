# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-012: Document Authentication Agent

Provides built-in reference datasets for document authentication operations:
    - document_templates: Known document template specifications per type
      per country for document classification.  Covers 20+ templates across
      COO (12 countries), phytosanitary (IPPC), BOL, RSPO, FSC, ISCC,
      Fairtrade, UTZ/RA, LTR, DDS, and SSD document types.
    - trusted_cas: Trusted certificate authority registry for chain
      validation, organized by category (eIDAS TSPs, document signing,
      government, certification bodies) with pinned issuer mappings.
    - fraud_rules: 15 deterministic fraud detection rule definitions
      (FRD-001 through FRD-015) with severity levels, thresholds, and
      applicable document types.  Also includes required document sets
      per EUDR commodity for completeness validation.

These datasets enable deterministic, zero-hallucination document
authentication without external API dependencies.  All data is
version-tracked and provenance-auditable per EU 2023/1115 Article 14
and eIDAS Regulation (EU) No 910/2014.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication (GL-EUDR-DAV-012)
Agent ID: GL-EUDR-DAV-012
"""

from greenlang.agents.eudr.document_authentication.reference_data.document_templates import (
    DOCUMENT_TEMPLATES,
    TEMPLATE_TYPE_INDEX,
    TOTAL_DOCUMENT_TYPES,
    TOTAL_TEMPLATES,
    get_all_templates,
    get_countries_for_type,
    get_supported_document_types,
    get_template,
    get_templates_for_type,
)
from greenlang.agents.eudr.document_authentication.reference_data.fraud_rules import (
    DOC_TYPE_RULE_INDEX,
    FRAUD_RULE_INDEX,
    FRAUD_RULES,
    FRAUD_RULES_BY_PATTERN,
    FRAUD_RULES_BY_SEVERITY,
    REQUIRED_DOCUMENTS_BY_COMMODITY,
    TOTAL_COMMODITIES,
    TOTAL_FRAUD_RULES,
    get_all_commodities,
    get_all_rules,
    get_enabled_rules,
    get_required_documents,
    get_rule,
    get_rule_ids,
    get_rules_by_pattern,
    get_rules_by_severity,
    get_rules_for_document_type,
    get_severity_distribution,
)
from greenlang.agents.eudr.document_authentication.reference_data.trusted_cas import (
    CA_CATEGORY_INDEX,
    PINNED_ISSUERS,
    TOTAL_CATEGORIES,
    TOTAL_TRUSTED_CAS,
    TRUSTED_CAS,
    get_all_categories,
    get_ca_by_name,
    get_cas_by_category,
    get_intermediate_cas,
    get_pinned_issuers,
    get_pinned_issuers_for_standard,
    get_root_cas,
    get_trusted_cas,
    is_trusted_ca,
)

__all__ = [
    # -- document_templates --
    "DOCUMENT_TEMPLATES",
    "TEMPLATE_TYPE_INDEX",
    "TOTAL_DOCUMENT_TYPES",
    "TOTAL_TEMPLATES",
    "get_template",
    "get_templates_for_type",
    "get_all_templates",
    "get_supported_document_types",
    "get_countries_for_type",
    # -- trusted_cas --
    "TRUSTED_CAS",
    "CA_CATEGORY_INDEX",
    "PINNED_ISSUERS",
    "TOTAL_TRUSTED_CAS",
    "TOTAL_CATEGORIES",
    "get_trusted_cas",
    "get_cas_by_category",
    "get_ca_by_name",
    "get_root_cas",
    "get_intermediate_cas",
    "get_pinned_issuers",
    "get_pinned_issuers_for_standard",
    "get_all_categories",
    "is_trusted_ca",
    # -- fraud_rules --
    "FRAUD_RULES",
    "FRAUD_RULE_INDEX",
    "FRAUD_RULES_BY_SEVERITY",
    "FRAUD_RULES_BY_PATTERN",
    "REQUIRED_DOCUMENTS_BY_COMMODITY",
    "DOC_TYPE_RULE_INDEX",
    "TOTAL_FRAUD_RULES",
    "TOTAL_COMMODITIES",
    "get_rule",
    "get_all_rules",
    "get_enabled_rules",
    "get_rules_by_severity",
    "get_rules_by_pattern",
    "get_rules_for_document_type",
    "get_required_documents",
    "get_all_commodities",
    "get_rule_ids",
    "get_severity_distribution",
]
