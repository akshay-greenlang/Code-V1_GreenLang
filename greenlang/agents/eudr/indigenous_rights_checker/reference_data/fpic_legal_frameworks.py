# -*- coding: utf-8 -*-
"""
Per-Country FPIC Legal Framework Requirements - AGENT-EUDR-021

Authoritative reference data for FPIC legal requirements in EUDR
commodity-producing countries. Maps national legislation, constitutional
provisions, and regulatory frameworks that govern Free, Prior and
Informed Consent for indigenous and tribal peoples.

Per PRD Section 4.3: 8 major producing countries with FPIC-specific
legal frameworks are documented. Each entry includes legal basis,
applicable EUDR commodities, FPIC process requirements, validation
rules for country-specific FPIC verification, and compliance thresholds.

Example:
    >>> from greenlang.agents.eudr.indigenous_rights_checker.reference_data.fpic_legal_frameworks import (
    ...     FPIC_LEGAL_FRAMEWORKS,
    ...     get_fpic_requirements,
    ... )
    >>> brazil = get_fpic_requirements("BR")
    >>> print(brazil["legal_basis"])
    ['Federal Constitution Art. 231', 'ILO 169 (ratified 2002)', 'FUNAI regulatory framework']

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# FPIC Legal Framework Requirements by Country
# Per PRD Section 4.3: 8 major EUDR commodity-producing countries
# ---------------------------------------------------------------------------

FPIC_LEGAL_FRAMEWORKS: Dict[str, Dict[str, Any]] = {
    "BR": {
        "country_name": "Brazil",
        "country_code": "BR",
        "legal_basis": [
            "Federal Constitution Art. 231",
            "ILO 169 (ratified 2002)",
            "FUNAI regulatory framework",
        ],
        "constitutional_protection": True,
        "ilo_169_ratified": True,
        "fpic_legally_required": True,
        "applicable_commodities": [
            "cattle", "soya", "cocoa", "coffee", "wood",
        ],
        "governing_authority": "FUNAI (Fundacao Nacional dos Povos Indigenas)",
        "consultation_protocol": "funai_consultation",
        "minimum_consultation_period_days": 90,
        "community_representation_requirement": (
            "Community leaders recognized by FUNAI and community "
            "internal governance structures. Must include caciques "
            "and community assembly representatives."
        ),
        "information_disclosure_requirements": [
            "Project description in Portuguese and indigenous language",
            "Environmental impact assessment (EIA/RIMA)",
            "Social impact assessment",
            "Economic analysis with benefit-sharing options",
            "Maps showing project area relative to territory",
        ],
        "consent_documentation_requirements": [
            "Community assembly minutes with attendee list",
            "Signed consent form by recognized representatives",
            "FUNAI oversight certification",
            "Translation of all documents into indigenous language",
        ],
        "coercion_indicators": [
            "Consent obtained without FUNAI oversight",
            "Consent during active judicial proceeding",
            "Consent obtained within 30 days of first contact",
            "Benefits conditioned on consent",
        ],
        "additional_validation_rules": {
            "funai_certification_required": True,
            "environmental_license_prerequisite": True,
            "community_assembly_minimum_quorum": 0.5,
            "monitoring_period_years": 5,
        },
        "key_judicial_precedents": [
            "STF - Raposa Serra do Sol (2009)",
            "STF - Marco Temporal controversy (2023)",
        ],
    },
    "ID": {
        "country_name": "Indonesia",
        "country_code": "ID",
        "legal_basis": [
            "Constitutional Court Decision 35/2012",
            "AMAN customary territory recognition",
            "FPIC in RSPO Principles & Criteria",
        ],
        "constitutional_protection": True,
        "ilo_169_ratified": False,
        "fpic_legally_required": True,
        "applicable_commodities": [
            "palm_oil", "rubber", "wood", "cocoa", "coffee",
        ],
        "governing_authority": "BPN (Badan Pertanahan Nasional) / AMAN",
        "consultation_protocol": "padiatapa_framework",
        "minimum_consultation_period_days": 60,
        "community_representation_requirement": (
            "Adat community leaders (tetua adat) with mandate "
            "from Musyawarah Adat (customary assembly). AMAN "
            "verification of community governance structure."
        ),
        "information_disclosure_requirements": [
            "Project description in Bahasa Indonesia and local language",
            "AMDAL (environmental impact analysis)",
            "Social impact assessment (ANDAL)",
            "Land use plan with community territory overlay",
            "RSPO New Planting Procedure (if palm oil)",
        ],
        "consent_documentation_requirements": [
            "Musyawarah Adat (customary assembly) minutes",
            "Signed agreement by tetua adat",
            "Witness statements from community members",
            "AMAN verification letter (where applicable)",
        ],
        "coercion_indicators": [
            "Consent without Musyawarah Adat convening",
            "Military or police presence during consultation",
            "Consent obtained through village head without adat leaders",
            "Land compensation offered before FPIC process",
        ],
        "additional_validation_rules": {
            "amdal_required": True,
            "rspo_npp_for_palm_oil": True,
            "community_mapping_required": True,
            "monitoring_period_years": 3,
        },
        "key_judicial_precedents": [
            "MK 35/PUU-X/2012 (Constitutional Court customary forests)",
        ],
    },
    "CO": {
        "country_name": "Colombia",
        "country_code": "CO",
        "legal_basis": [
            "Constitution Art. 330",
            "ILO 169 (ratified 1991)",
            "Constitutional Court T-129/2011",
            "Decreto 1320/1998",
        ],
        "constitutional_protection": True,
        "ilo_169_ratified": True,
        "fpic_legally_required": True,
        "applicable_commodities": ["coffee", "palm_oil", "cocoa", "wood"],
        "governing_authority": "Ministerio del Interior - Direccion de Consulta Previa",
        "consultation_protocol": "consulta_previa_decreto_1320",
        "minimum_consultation_period_days": 120,
        "community_representation_requirement": (
            "Authorities of Resguardo Indigena, Cabildo Indigena, "
            "or recognized indigenous organization. Must be verified "
            "by Ministerio del Interior."
        ),
        "information_disclosure_requirements": [
            "Project description in Spanish and indigenous language",
            "Environmental impact assessment (EIA)",
            "Social and cultural impact assessment",
            "Territorial impact analysis",
            "Alternative options analysis",
        ],
        "consent_documentation_requirements": [
            "Consulta Previa act (acta de consulta)",
            "Protocol agreement (protocolo de consulta)",
            "Ministry of Interior certification",
            "Community assembly minutes",
        ],
        "coercion_indicators": [
            "Consultation without Ministry oversight",
            "Consent during active armed conflict in territory",
            "Consultation conducted in Spanish only",
            "Benefits contingent on project approval",
        ],
        "additional_validation_rules": {
            "ministry_certification_required": True,
            "cultural_impact_assessment_required": True,
            "monitoring_period_years": 5,
        },
        "key_judicial_precedents": [
            "T-129/2011 (FPIC for mining in indigenous territories)",
            "T-769/2009 (mandatory consultation for infrastructure)",
        ],
    },
    "PE": {
        "country_name": "Peru",
        "country_code": "PE",
        "legal_basis": [
            "ILO 169 (ratified 1994)",
            "Prior Consultation Law 29785 (2011)",
            "Reglamento DS 001-2012-MC",
        ],
        "constitutional_protection": True,
        "ilo_169_ratified": True,
        "fpic_legally_required": True,
        "applicable_commodities": ["coffee", "cocoa", "wood", "palm_oil"],
        "governing_authority": "Ministerio de Cultura - Viceministerio de Interculturalidad",
        "consultation_protocol": "ley_29785_consulta_previa",
        "minimum_consultation_period_days": 90,
        "community_representation_requirement": (
            "Presidents of Comunidades Nativas or Comunidades "
            "Campesinas, AIDESEP regional organizations, with "
            "Ministry of Culture verification."
        ),
        "information_disclosure_requirements": [
            "Project information in Spanish and indigenous language",
            "Environmental impact assessment (EIA)",
            "Plan de consulta (consultation plan)",
            "Rights information document",
            "Alternative project options",
        ],
        "consent_documentation_requirements": [
            "Acta de consulta previa",
            "Agreement document (acuerdo)",
            "Ministry of Culture process report",
            "Community assembly verification",
        ],
        "coercion_indicators": [
            "Consultation without Ministry of Culture oversight",
            "Agreement signed under protest from community members",
            "Insufficient translation into indigenous language",
            "Process completed in less than 90 days",
        ],
        "additional_validation_rules": {
            "ministry_certification_required": True,
            "database_registration_required": True,
            "monitoring_period_years": 5,
        },
        "key_judicial_precedents": [
            "TC 06316-2008-PA (right to prior consultation)",
        ],
    },
    "CD": {
        "country_name": "Democratic Republic of Congo",
        "country_code": "CD",
        "legal_basis": [
            "Forest Code Art. 7",
            "Community Forest Concession framework",
            "FPIC in FLEGT VPA process",
        ],
        "constitutional_protection": False,
        "ilo_169_ratified": False,
        "fpic_legally_required": False,
        "applicable_commodities": ["wood", "cocoa", "coffee"],
        "governing_authority": "ICCN (Institut Congolais pour la Conservation de la Nature)",
        "consultation_protocol": "community_forest_consultation",
        "minimum_consultation_period_days": 60,
        "community_representation_requirement": (
            "Traditional chiefs (chefs coutumiers) and community "
            "forest management committees. Clan-based representation."
        ),
        "information_disclosure_requirements": [
            "Project description in French and local language",
            "Environmental impact study",
            "Community rights documentation",
        ],
        "consent_documentation_requirements": [
            "Community assembly minutes (proces-verbal)",
            "Chief signature and community seal",
            "Provincial authority attestation",
        ],
        "coercion_indicators": [
            "Consent obtained during armed conflict",
            "Consultation without local language translation",
            "Payment to chiefs without community benefit sharing",
        ],
        "additional_validation_rules": {
            "provincial_attestation_required": True,
            "monitoring_period_years": 3,
        },
        "key_judicial_precedents": [],
    },
    "CI": {
        "country_name": "Cote d'Ivoire",
        "country_code": "CI",
        "legal_basis": [
            "Land Law 98-750",
            "Rural Land Code 2019",
            "RSPO FPIC requirements (for palm oil)",
        ],
        "constitutional_protection": False,
        "ilo_169_ratified": False,
        "fpic_legally_required": False,
        "applicable_commodities": ["cocoa", "coffee", "rubber", "palm_oil"],
        "governing_authority": "AFOR (Agence Fonciere Rurale)",
        "consultation_protocol": "customary_land_consultation",
        "minimum_consultation_period_days": 45,
        "community_representation_requirement": (
            "Village chiefs, terre committee members, and recognized "
            "community elders. Matrilineal clan leaders in Akan areas."
        ),
        "information_disclosure_requirements": [
            "Project description in French and local language",
            "Land use impact analysis",
            "Community boundary maps",
        ],
        "consent_documentation_requirements": [
            "Village assembly minutes",
            "Chief and committee signatures",
            "Sous-prefet attestation",
        ],
        "coercion_indicators": [
            "Consent without village assembly",
            "Individual land sales bypassing community governance",
            "Consultation only with male leadership in matrilineal areas",
        ],
        "additional_validation_rules": {
            "sous_prefet_attestation": True,
            "monitoring_period_years": 3,
        },
        "key_judicial_precedents": [],
    },
    "GH": {
        "country_name": "Ghana",
        "country_code": "GH",
        "legal_basis": [
            "Constitution Art. 267",
            "Stool Land management framework",
            "Cocoa Forest REDD+ Programme FPIC protocol",
        ],
        "constitutional_protection": True,
        "ilo_169_ratified": False,
        "fpic_legally_required": False,
        "applicable_commodities": ["cocoa", "wood"],
        "governing_authority": "Lands Commission / Office of Administrator of Stool Lands",
        "consultation_protocol": "stool_land_consultation",
        "minimum_consultation_period_days": 30,
        "community_representation_requirement": (
            "Traditional authorities (chiefs, queen mothers), "
            "Stool Land administrators, and community elders. "
            "Must include both male and female representatives."
        ),
        "information_disclosure_requirements": [
            "Project description in English and local language",
            "Land impact assessment",
            "Benefit-sharing proposal",
        ],
        "consent_documentation_requirements": [
            "Stool council minutes",
            "Chief and queen mother signatures",
            "District Assembly endorsement",
        ],
        "coercion_indicators": [
            "Consent without queen mother involvement",
            "Stool land alienation without community awareness",
        ],
        "additional_validation_rules": {
            "stool_council_approval": True,
            "monitoring_period_years": 3,
        },
        "key_judicial_precedents": [],
    },
    "MY": {
        "country_name": "Malaysia",
        "country_code": "MY",
        "legal_basis": [
            "Native Customary Rights (NCR) under Sarawak Land Code",
            "MSPO FPIC requirements",
            "Federal Constitution Art. 153",
        ],
        "constitutional_protection": True,
        "ilo_169_ratified": False,
        "fpic_legally_required": True,
        "applicable_commodities": ["palm_oil", "rubber", "wood"],
        "governing_authority": "Sarawak Land and Survey Department / JAKOA",
        "consultation_protocol": "ncr_consultation",
        "minimum_consultation_period_days": 60,
        "community_representation_requirement": (
            "Tuai Rumah (longhouse chiefs), Penghulu (paramount "
            "chiefs), and Village Development Committee (JKKK). "
            "Must include affected NCR holders."
        ),
        "information_disclosure_requirements": [
            "Project description in Bahasa Malaysia and Iban/local language",
            "Environmental impact assessment",
            "NCR area survey and maps",
            "MSPO assessment report (for palm oil)",
        ],
        "consent_documentation_requirements": [
            "Community meeting minutes",
            "Tuai Rumah and Penghulu signatures",
            "NCR verification from Land and Survey Department",
        ],
        "coercion_indicators": [
            "Consent without NCR survey completion",
            "Consultation only with Penghulu bypassing longhouse chiefs",
            "Land development before consent formalized",
        ],
        "additional_validation_rules": {
            "ncr_survey_required": True,
            "mspo_for_palm_oil": True,
            "monitoring_period_years": 3,
        },
        "key_judicial_precedents": [
            "Nor Nyawai v Borneo Pulp (2001) - NCR recognition",
        ],
    },
}


def get_fpic_requirements(country_code: str) -> Dict[str, Any]:
    """Get FPIC legal requirements for a specific country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Dictionary with FPIC legal framework data, or empty dict if
        no specific framework is configured.

    Example:
        >>> req = get_fpic_requirements("BR")
        >>> req["fpic_legally_required"]
        True
    """
    return FPIC_LEGAL_FRAMEWORKS.get(country_code.upper(), {})


def is_fpic_legally_required(country_code: str) -> bool:
    """Check if FPIC is legally required in a specific country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        True if FPIC is legally required, False otherwise.

    Example:
        >>> is_fpic_legally_required("BR")
        True
        >>> is_fpic_legally_required("GH")
        False
    """
    framework = FPIC_LEGAL_FRAMEWORKS.get(country_code.upper(), {})
    return framework.get("fpic_legally_required", False)


def get_consultation_protocol(country_code: str) -> Optional[str]:
    """Get the consultation protocol identifier for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Protocol identifier string, or None if not configured.

    Example:
        >>> get_consultation_protocol("BR")
        'funai_consultation'
    """
    framework = FPIC_LEGAL_FRAMEWORKS.get(country_code.upper(), {})
    return framework.get("consultation_protocol")


def get_minimum_consultation_days(country_code: str) -> int:
    """Get minimum consultation period in days for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Minimum consultation period in days (default 90 if not configured).

    Example:
        >>> get_minimum_consultation_days("BR")
        90
    """
    framework = FPIC_LEGAL_FRAMEWORKS.get(country_code.upper(), {})
    return framework.get("minimum_consultation_period_days", 90)


def get_countries_with_fpic_requirement() -> List[str]:
    """Get all country codes where FPIC is legally required.

    Returns:
        List of ISO 3166-1 alpha-2 country codes.

    Example:
        >>> countries = get_countries_with_fpic_requirement()
        >>> assert "BR" in countries
    """
    return [
        code
        for code, framework in FPIC_LEGAL_FRAMEWORKS.items()
        if framework.get("fpic_legally_required", False)
    ]
