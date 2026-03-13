# -*- coding: utf-8 -*-
"""
Certification Scheme Reference Data - AGENT-EUDR-023

Defines 5 certification schemes (FSC, PEFC, RSPO, Rainforest Alliance, ISCC)
with their sub-schemes, covered commodities, chain-of-custody models,
certificate number format patterns, and EUDR Article 2(40) equivalence mapping.

Zero-Hallucination: EUDR equivalence mapping is based on published scheme
standards and expert analysis of requirement overlap with Article 2(40)
categories. Coverage levels: FULL, PARTIAL, NONE.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Certification scheme specifications
# ---------------------------------------------------------------------------

CERTIFICATION_SCHEMES: Dict[str, Dict[str, Any]] = {
    "fsc_fm": {
        "scheme_name": "Forest Stewardship Council",
        "sub_scheme": "Forest Management",
        "code": "FSC-FM",
        "commodities": ["wood"],
        "coc_models": ["transfer", "percentage", "credit"],
        "cert_number_pattern": r"^FSC-C\d{6}$",
        "cert_number_example": "FSC-C123456",
        "api_url": "https://info.fsc.org/api/v1",
        "max_validity_years": 5,
        "annual_surveillance": True,
    },
    "fsc_coc": {
        "scheme_name": "Forest Stewardship Council",
        "sub_scheme": "Chain of Custody",
        "code": "FSC-CoC",
        "commodities": ["wood"],
        "coc_models": ["transfer", "percentage", "credit"],
        "cert_number_pattern": r"^FSC-C\d{6}$",
        "cert_number_example": "FSC-C654321",
        "api_url": "https://info.fsc.org/api/v1",
        "max_validity_years": 5,
        "annual_surveillance": True,
    },
    "fsc_cw": {
        "scheme_name": "Forest Stewardship Council",
        "sub_scheme": "Controlled Wood",
        "code": "FSC-CW",
        "commodities": ["wood"],
        "coc_models": ["controlled_sources"],
        "cert_number_pattern": r"^FSC-C\d{6}$",
        "cert_number_example": "FSC-C111111",
        "api_url": "https://info.fsc.org/api/v1",
        "max_validity_years": 5,
        "annual_surveillance": True,
    },
    "pefc_sfm": {
        "scheme_name": "PEFC",
        "sub_scheme": "Sustainable Forest Management",
        "code": "PEFC-SFM",
        "commodities": ["wood"],
        "coc_models": ["physical_separation", "percentage"],
        "cert_number_pattern": r"^PEFC/\d{2}-\d{2}-\d{2}/\d+$",
        "cert_number_example": "PEFC/01-00-01/12345",
        "api_url": "https://pefc.org/api/v1",
        "max_validity_years": 5,
        "annual_surveillance": True,
    },
    "pefc_coc": {
        "scheme_name": "PEFC",
        "sub_scheme": "Chain of Custody",
        "code": "PEFC-CoC",
        "commodities": ["wood"],
        "coc_models": ["physical_separation", "percentage"],
        "cert_number_pattern": r"^PEFC/\d{2}-\d{2}-\d{2}/\d+$",
        "cert_number_example": "PEFC/01-31-01/67890",
        "api_url": "https://pefc.org/api/v1",
        "max_validity_years": 5,
        "annual_surveillance": True,
    },
    "rspo_pc": {
        "scheme_name": "Roundtable on Sustainable Palm Oil",
        "sub_scheme": "Principles and Criteria",
        "code": "RSPO-P&C",
        "commodities": ["oil_palm"],
        "coc_models": ["identity_preserved", "segregated", "mass_balance"],
        "cert_number_pattern": r"^RSPO-\d{7}$",
        "cert_number_example": "RSPO-1234567",
        "api_url": "https://rspo.org/api/v1",
        "max_validity_years": 5,
        "annual_surveillance": True,
    },
    "rspo_scc": {
        "scheme_name": "Roundtable on Sustainable Palm Oil",
        "sub_scheme": "Supply Chain Certification",
        "code": "RSPO-SCC",
        "commodities": ["oil_palm"],
        "coc_models": ["identity_preserved", "segregated", "mass_balance"],
        "cert_number_pattern": r"^RSPO-\d{7}$",
        "cert_number_example": "RSPO-7654321",
        "api_url": "https://rspo.org/api/v1",
        "max_validity_years": 5,
        "annual_surveillance": True,
    },
    "rspo_is": {
        "scheme_name": "Roundtable on Sustainable Palm Oil",
        "sub_scheme": "Independent Smallholder",
        "code": "RSPO-IS",
        "commodities": ["oil_palm"],
        "coc_models": ["mass_balance"],
        "cert_number_pattern": r"^RSPO-\d{7}$",
        "cert_number_example": "RSPO-0000001",
        "api_url": "https://rspo.org/api/v1",
        "max_validity_years": 3,
        "annual_surveillance": True,
    },
    "ra_sa": {
        "scheme_name": "Rainforest Alliance",
        "sub_scheme": "Sustainable Agriculture",
        "code": "RA-SA",
        "commodities": ["cocoa", "coffee"],
        "coc_models": ["segregated", "mass_balance"],
        "cert_number_pattern": r"^RA-\d{6,8}$",
        "cert_number_example": "RA-12345678",
        "api_url": "https://www.rainforest-alliance.org/api/v1",
        "max_validity_years": 3,
        "annual_surveillance": True,
    },
    "ra_coc": {
        "scheme_name": "Rainforest Alliance",
        "sub_scheme": "Chain of Custody",
        "code": "RA-CoC",
        "commodities": ["cocoa", "coffee"],
        "coc_models": ["segregated", "mass_balance"],
        "cert_number_pattern": r"^RA-\d{6,8}$",
        "cert_number_example": "RA-87654321",
        "api_url": "https://www.rainforest-alliance.org/api/v1",
        "max_validity_years": 3,
        "annual_surveillance": True,
    },
    "iscc_eu": {
        "scheme_name": "ISCC",
        "sub_scheme": "ISCC EU",
        "code": "ISCC-EU",
        "commodities": ["soya", "oil_palm"],
        "coc_models": ["physical_segregation", "mass_balance"],
        "cert_number_pattern": r"^ISCC-\w+-\d+$",
        "cert_number_example": "ISCC-EU-12345",
        "api_url": "https://www.iscc-system.org/api/v1",
        "max_validity_years": 1,
        "annual_surveillance": True,
    },
    "iscc_plus": {
        "scheme_name": "ISCC",
        "sub_scheme": "ISCC PLUS",
        "code": "ISCC-PLUS",
        "commodities": ["soya", "oil_palm"],
        "coc_models": ["physical_segregation", "mass_balance"],
        "cert_number_pattern": r"^ISCC-\w+-\d+$",
        "cert_number_example": "ISCC-PLUS-67890",
        "api_url": "https://www.iscc-system.org/api/v1",
        "max_validity_years": 1,
        "annual_surveillance": True,
    },
}

# ---------------------------------------------------------------------------
# EUDR Article 2(40) equivalence matrix
# Coverage levels: "full", "partial", "none"
# Per Architecture Spec Appendix C
# ---------------------------------------------------------------------------

EUDR_EQUIVALENCE_MATRIX: Dict[str, Dict[str, str]] = {
    "fsc_fm": {
        "land_use_rights": "full",
        "environmental_protection": "full",
        "forest_related_rules": "full",
        "third_party_rights": "full",
        "labour_rights": "full",
        "tax_and_royalty": "none",
        "trade_and_customs": "none",
        "anti_corruption": "partial",
    },
    "fsc_coc": {
        "land_use_rights": "partial",
        "environmental_protection": "none",
        "forest_related_rules": "partial",
        "third_party_rights": "none",
        "labour_rights": "partial",
        "tax_and_royalty": "none",
        "trade_and_customs": "partial",
        "anti_corruption": "none",
    },
    "fsc_cw": {
        "land_use_rights": "partial",
        "environmental_protection": "partial",
        "forest_related_rules": "partial",
        "third_party_rights": "partial",
        "labour_rights": "partial",
        "tax_and_royalty": "none",
        "trade_and_customs": "none",
        "anti_corruption": "none",
    },
    "pefc_sfm": {
        "land_use_rights": "full",
        "environmental_protection": "full",
        "forest_related_rules": "full",
        "third_party_rights": "partial",
        "labour_rights": "partial",
        "tax_and_royalty": "none",
        "trade_and_customs": "none",
        "anti_corruption": "partial",
    },
    "pefc_coc": {
        "land_use_rights": "none",
        "environmental_protection": "none",
        "forest_related_rules": "partial",
        "third_party_rights": "none",
        "labour_rights": "none",
        "tax_and_royalty": "none",
        "trade_and_customs": "partial",
        "anti_corruption": "none",
    },
    "rspo_pc": {
        "land_use_rights": "full",
        "environmental_protection": "full",
        "forest_related_rules": "partial",
        "third_party_rights": "full",
        "labour_rights": "full",
        "tax_and_royalty": "none",
        "trade_and_customs": "none",
        "anti_corruption": "partial",
    },
    "rspo_scc": {
        "land_use_rights": "none",
        "environmental_protection": "none",
        "forest_related_rules": "none",
        "third_party_rights": "none",
        "labour_rights": "partial",
        "tax_and_royalty": "none",
        "trade_and_customs": "partial",
        "anti_corruption": "none",
    },
    "rspo_is": {
        "land_use_rights": "partial",
        "environmental_protection": "partial",
        "forest_related_rules": "none",
        "third_party_rights": "partial",
        "labour_rights": "partial",
        "tax_and_royalty": "none",
        "trade_and_customs": "none",
        "anti_corruption": "none",
    },
    "ra_sa": {
        "land_use_rights": "partial",
        "environmental_protection": "full",
        "forest_related_rules": "full",
        "third_party_rights": "partial",
        "labour_rights": "full",
        "tax_and_royalty": "none",
        "trade_and_customs": "none",
        "anti_corruption": "partial",
    },
    "ra_coc": {
        "land_use_rights": "none",
        "environmental_protection": "none",
        "forest_related_rules": "none",
        "third_party_rights": "none",
        "labour_rights": "partial",
        "tax_and_royalty": "none",
        "trade_and_customs": "partial",
        "anti_corruption": "none",
    },
    "iscc_eu": {
        "land_use_rights": "partial",
        "environmental_protection": "full",
        "forest_related_rules": "partial",
        "third_party_rights": "none",
        "labour_rights": "partial",
        "tax_and_royalty": "none",
        "trade_and_customs": "partial",
        "anti_corruption": "none",
    },
    "iscc_plus": {
        "land_use_rights": "partial",
        "environmental_protection": "full",
        "forest_related_rules": "partial",
        "third_party_rights": "none",
        "labour_rights": "partial",
        "tax_and_royalty": "none",
        "trade_and_customs": "partial",
        "anti_corruption": "none",
    },
}


def get_scheme_spec(scheme_key: str) -> Optional[Dict[str, Any]]:
    """Get the specification for a certification scheme.

    Args:
        scheme_key: Scheme key (e.g. "fsc_fm", "rspo_pc").

    Returns:
        Scheme specification dict or None if not found.

    Example:
        >>> spec = get_scheme_spec("fsc_fm")
        >>> assert spec["scheme_name"] == "Forest Stewardship Council"
    """
    return CERTIFICATION_SCHEMES.get(scheme_key)


def get_eudr_coverage(scheme_key: str) -> Optional[Dict[str, str]]:
    """Get the EUDR equivalence coverage for a certification scheme.

    Args:
        scheme_key: Scheme key (e.g. "fsc_fm").

    Returns:
        Dict mapping category to coverage level, or None.

    Example:
        >>> coverage = get_eudr_coverage("fsc_fm")
        >>> assert coverage["land_use_rights"] == "full"
    """
    return EUDR_EQUIVALENCE_MATRIX.get(scheme_key)


def get_schemes_for_commodity(commodity: str) -> List[str]:
    """Get all certification schemes applicable to a commodity.

    Args:
        commodity: EUDR commodity type (e.g. "wood", "oil_palm").

    Returns:
        List of scheme keys that cover the commodity.

    Example:
        >>> schemes = get_schemes_for_commodity("wood")
        >>> assert "fsc_fm" in schemes
    """
    result: List[str] = []
    for key, spec in CERTIFICATION_SCHEMES.items():
        if commodity in spec["commodities"]:
            result.append(key)
    return result
