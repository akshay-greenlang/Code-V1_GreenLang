# -*- coding: utf-8 -*-
"""
Protected Area Data Sources Reference - AGENT-EUDR-022

Metadata for authoritative protected area data sources including WDPA/
Protected Planet, national registries (ICMBio, KLHK, ICCN, SERNANP,
SINAP), and supplementary databases (KBA, UNESCO WH, Ramsar).

Author: GreenLang Platform Team
Date: March 2026
"""

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# WDPA Primary Source
# ---------------------------------------------------------------------------

WDPA_SOURCE: Dict[str, Any] = {
    "name": "World Database on Protected Areas",
    "abbreviation": "WDPA",
    "organization": "UNEP-WCMC / IUCN",
    "url": "https://www.protectedplanet.net",
    "update_frequency": "monthly",
    "coverage": "270,000+ protected areas across 245 countries",
    "coordinate_system": "WGS84 (EPSG:4326)",
    "formats": ["GeoPackage", "Shapefile", "GeoJSON"],
    "license": "Non-commercial use with attribution",
    "api_endpoint": "https://api.protectedplanet.net/v3",
    "fields": [
        "WDPAID", "WDPA_PID", "PA_DEF", "NAME", "ORIG_NAME", "DESIG",
        "DESIG_ENG", "DESIG_TYPE", "IUCN_CAT", "INT_CRIT", "MARINE",
        "REP_M_AREA", "GIS_M_AREA", "REP_AREA", "GIS_AREA", "NO_TAKE",
        "NO_TK_AREA", "STATUS", "STATUS_YR", "GOV_TYPE", "OWN_TYPE",
        "MANG_AUTH", "MANG_PLAN", "VERIF", "METADATAID", "SUB_LOC",
        "PARENT_ISO3", "ISO3",
    ],
}

# ---------------------------------------------------------------------------
# Supplementary Sources
# ---------------------------------------------------------------------------

KBA_SOURCE: Dict[str, Any] = {
    "name": "Key Biodiversity Areas Database",
    "abbreviation": "KBA",
    "organization": "BirdLife International / IUCN",
    "url": "https://www.keybiodiversityareas.org",
    "coverage": "16,000+ globally significant biodiversity sites",
    "update_frequency": "annual",
}

UNESCO_WH_SOURCE: Dict[str, Any] = {
    "name": "UNESCO World Heritage List",
    "abbreviation": "UNESCO_WH",
    "organization": "UNESCO World Heritage Centre",
    "url": "https://whc.unesco.org",
    "coverage": "1,199 World Heritage Sites (natural and mixed)",
    "update_frequency": "annual (post-Committee session)",
}

RAMSAR_SOURCE: Dict[str, Any] = {
    "name": "Ramsar Wetlands of International Importance",
    "abbreviation": "RAMSAR",
    "organization": "Ramsar Convention Secretariat",
    "url": "https://www.ramsar.org",
    "coverage": "2,500+ Wetlands of International Importance",
    "update_frequency": "continuous",
}

# ---------------------------------------------------------------------------
# National Registries (Major EUDR commodity-producing countries)
# ---------------------------------------------------------------------------

NATIONAL_REGISTRIES: Dict[str, Dict[str, Any]] = {
    "BRA": {
        "name": "ICMBio / CNUC - National Registry of Conservation Units",
        "country": "Brazil",
        "iso3": "BRA",
        "organization": "Instituto Chico Mendes de Conservacao da Biodiversidade",
        "url": "https://www.gov.br/icmbio",
        "coverage": "2,400+ Conservation Units (federal, state, municipal)",
        "legal_basis": "Lei 9.985/2000 (SNUC - National System of Conservation Units)",
        "update_frequency": "quarterly",
        "key_categories": [
            "Estacao Ecologica (Ecological Station - IUCN Ia)",
            "Reserva Biologica (Biological Reserve - IUCN Ia)",
            "Parque Nacional (National Park - IUCN II)",
            "Reserva Extrativista (Extractive Reserve - IUCN VI)",
            "Reserva de Desenvolvimento Sustentavel (Sustainable Development - IUCN VI)",
            "Floresta Nacional (National Forest - IUCN VI)",
            "Area de Protecao Ambiental (Environmental Protection Area - IUCN V)",
        ],
    },
    "IDN": {
        "name": "KLHK - Ministry of Environment and Forestry",
        "country": "Indonesia",
        "iso3": "IDN",
        "organization": "Kementerian Lingkungan Hidup dan Kehutanan",
        "url": "https://www.menlhk.go.id",
        "coverage": "560+ protected areas (Kawasan Konservasi)",
        "legal_basis": "UU 5/1990 (Conservation of Living Natural Resources)",
        "update_frequency": "biannual",
        "key_categories": [
            "Taman Nasional (National Park - IUCN II)",
            "Cagar Alam (Nature Reserve - IUCN Ia)",
            "Suaka Margasatwa (Wildlife Sanctuary - IUCN IV)",
            "Taman Wisata Alam (Nature Recreation Park - IUCN V)",
            "Taman Buru (Game Reserve - IUCN VI)",
        ],
    },
    "COD": {
        "name": "ICCN - Congolese Institute for Nature Conservation",
        "country": "Democratic Republic of Congo",
        "iso3": "COD",
        "organization": "Institut Congolais pour la Conservation de la Nature",
        "url": "https://www.iccnrdc.cd",
        "coverage": "90+ protected areas including 5 World Heritage Sites",
        "legal_basis": "Loi 14/003 du 11 fevrier 2014",
        "update_frequency": "annual",
        "key_categories": [
            "Parc National (National Park - IUCN II)",
            "Reserve de Faune (Fauna Reserve - IUCN IV)",
            "Domaine de Chasse (Hunting Domain - IUCN VI)",
        ],
    },
    "PER": {
        "name": "SERNANP - National Service of Natural Protected Areas",
        "country": "Peru",
        "iso3": "PER",
        "organization": "Servicio Nacional de Areas Naturales Protegidas por el Estado",
        "url": "https://www.sernanp.gob.pe",
        "coverage": "76 Natural Protected Areas covering 22.5M hectares",
        "legal_basis": "Ley 26834 (Natural Protected Areas Law)",
        "update_frequency": "quarterly",
        "key_categories": [
            "Parque Nacional (National Park - IUCN II)",
            "Santuario Nacional (National Sanctuary - IUCN III)",
            "Reserva Nacional (National Reserve - IUCN VI)",
            "Zona Reservada (Reserved Zone - proposed)",
        ],
    },
    "COL": {
        "name": "SINAP - National System of Protected Areas",
        "country": "Colombia",
        "iso3": "COL",
        "organization": "Parques Nacionales Naturales de Colombia",
        "url": "https://www.parquesnacionales.gov.co",
        "coverage": "59+ National Natural Parks covering 23M+ hectares",
        "legal_basis": "Decreto 2372 de 2010",
        "update_frequency": "biannual",
        "key_categories": [
            "Parque Nacional Natural (National Natural Park - IUCN II)",
            "Santuario de Flora y Fauna (Flora and Fauna Sanctuary - IUCN III)",
            "Area Natural Unica (Unique Natural Area - IUCN III)",
            "Reserva Nacional Natural (National Natural Reserve - IUCN IV)",
        ],
    },
    "CMR": {
        "name": "MINFOF - Ministry of Forests and Wildlife",
        "country": "Cameroon",
        "iso3": "CMR",
        "organization": "Ministere des Forets et de la Faune",
        "url": "https://www.minfof.cm",
        "coverage": "40+ protected areas including 10 National Parks",
        "legal_basis": "Loi 94/01 du 20 janvier 1994",
        "update_frequency": "annual",
    },
    "CIV": {
        "name": "OIPR - Office Ivoirien des Parcs et Reserves",
        "country": "Cote d'Ivoire",
        "iso3": "CIV",
        "organization": "Office Ivoirien des Parcs et Reserves",
        "url": "https://www.oipr.ci",
        "coverage": "14 protected areas covering 2.1M hectares",
        "legal_basis": "Loi 2002-102 relative aux parcs nationaux",
        "update_frequency": "annual",
    },
    "GHA": {
        "name": "Wildlife Division - Forestry Commission",
        "country": "Ghana",
        "iso3": "GHA",
        "organization": "Forestry Commission of Ghana",
        "url": "https://www.fcghana.org",
        "coverage": "290+ forest reserves and wildlife protected areas",
        "legal_basis": "Wildlife Conservation Regulations 1971 (LI 685)",
        "update_frequency": "annual",
    },
}

ALL_SOURCES: Dict[str, Dict[str, Any]] = {
    "wdpa": WDPA_SOURCE,
    "kba": KBA_SOURCE,
    "unesco_wh": UNESCO_WH_SOURCE,
    "ramsar": RAMSAR_SOURCE,
}


def get_source_metadata(source_key: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific data source.

    Args:
        source_key: Source key (wdpa, kba, unesco_wh, ramsar) or ISO3 code.

    Returns:
        Source metadata dictionary or None if not found.

    Example:
        >>> meta = get_source_metadata("wdpa")
        >>> assert meta["abbreviation"] == "WDPA"
    """
    if source_key in ALL_SOURCES:
        return ALL_SOURCES[source_key]
    if source_key.upper() in NATIONAL_REGISTRIES:
        return NATIONAL_REGISTRIES[source_key.upper()]
    return None
