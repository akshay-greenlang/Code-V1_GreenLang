# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-024: Third-Party Audit Manager

Static reference data for audit checklists, competent authority profiles,
certification scheme standards, and non-conformance classification rules.

Modules:
    - audit_checklists: EUDR and certification scheme audit criteria
    - competent_authorities: 27 EU Member State authority profiles
    - certification_schemes: FSC/PEFC/RSPO/RA/ISCC standards
    - non_conformance_rules: NC severity classification rule sets

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

try:
    from greenlang.agents.eudr.third_party_audit_manager.reference_data.audit_checklists import (
        EUDR_CHECKLIST_CRITERIA,
        FSC_CHECKLIST_CRITERIA,
        PEFC_CHECKLIST_CRITERIA,
        RSPO_CHECKLIST_CRITERIA,
        RA_CHECKLIST_CRITERIA,
        ISCC_CHECKLIST_CRITERIA,
        get_checklist_by_type,
    )
except ImportError:
    EUDR_CHECKLIST_CRITERIA = None
    get_checklist_by_type = None

try:
    from greenlang.agents.eudr.third_party_audit_manager.reference_data.competent_authorities import (
        EU_COMPETENT_AUTHORITIES,
        get_authority_by_member_state,
    )
except ImportError:
    EU_COMPETENT_AUTHORITIES = None
    get_authority_by_member_state = None

try:
    from greenlang.agents.eudr.third_party_audit_manager.reference_data.certification_schemes import (
        CERTIFICATION_SCHEME_PROFILES,
        get_scheme_profile,
    )
except ImportError:
    CERTIFICATION_SCHEME_PROFILES = None
    get_scheme_profile = None

try:
    from greenlang.agents.eudr.third_party_audit_manager.reference_data.non_conformance_rules import (
        NC_CLASSIFICATION_RULES,
        get_rules_by_severity,
    )
except ImportError:
    NC_CLASSIFICATION_RULES = None
    get_rules_by_severity = None

__all__ = [
    "EUDR_CHECKLIST_CRITERIA",
    "FSC_CHECKLIST_CRITERIA",
    "PEFC_CHECKLIST_CRITERIA",
    "RSPO_CHECKLIST_CRITERIA",
    "RA_CHECKLIST_CRITERIA",
    "ISCC_CHECKLIST_CRITERIA",
    "get_checklist_by_type",
    "EU_COMPETENT_AUTHORITIES",
    "get_authority_by_member_state",
    "CERTIFICATION_SCHEME_PROFILES",
    "get_scheme_profile",
    "NC_CLASSIFICATION_RULES",
    "get_rules_by_severity",
]
