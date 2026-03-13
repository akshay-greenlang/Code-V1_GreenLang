# -*- coding: utf-8 -*-
"""
Reference Data Package for Legal Compliance Verifier - AGENT-EUDR-023

Provides pre-loaded reference data for deterministic legal compliance
verification across EUDR Article 2(40) eight legislation categories.

Modules:
    - article_2_40_categories: 8 legislation category definitions
    - country_legal_frameworks: 27 country legal frameworks
    - certification_schemes: FSC/PEFC/RSPO/RA/ISCC specifications
    - red_flag_indicators: 40 red flag definitions

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from greenlang.agents.eudr.legal_compliance_verifier.reference_data.article_2_40_categories import (
    LEGISLATION_CATEGORIES,
    get_category_definition,
)
from greenlang.agents.eudr.legal_compliance_verifier.reference_data.country_legal_frameworks import (
    COUNTRY_FRAMEWORKS,
    get_country_framework,
    SUPPORTED_COUNTRIES,
)
from greenlang.agents.eudr.legal_compliance_verifier.reference_data.certification_schemes import (
    CERTIFICATION_SCHEMES,
    EUDR_EQUIVALENCE_MATRIX,
    get_scheme_spec,
)
from greenlang.agents.eudr.legal_compliance_verifier.reference_data.red_flag_indicators import (
    RED_FLAG_INDICATORS,
    get_red_flag_definition,
)

__all__ = [
    "LEGISLATION_CATEGORIES",
    "get_category_definition",
    "COUNTRY_FRAMEWORKS",
    "get_country_framework",
    "SUPPORTED_COUNTRIES",
    "CERTIFICATION_SCHEMES",
    "EUDR_EQUIVALENCE_MATRIX",
    "get_scheme_spec",
    "RED_FLAG_INDICATORS",
    "get_red_flag_definition",
]
