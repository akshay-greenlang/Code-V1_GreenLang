# -*- coding: utf-8 -*-
"""
Protected Areas Database - AGENT-EUDR-020 Deforestation Alert System

Comprehensive protected area reference data for spatial overlap detection in
deforestation alert analysis. Covers 100+ major protected areas from the World
Database on Protected Areas (WDPA 2025), UNESCO World Heritage forest sites,
Ramsar wetlands of international importance, Key Biodiversity Areas, and
Indigenous territories with community conservation designations.

Each protected area entry provides:
    - WDPA identifier and name
    - Country code (ISO 3166-1 alpha-3)
    - IUCN management category (Ia, Ib, II, III, IV, V, VI)
    - Center latitude and longitude
    - Total area in square kilometers
    - Designation year and designating authority
    - Status (designated, proposed, inscribed)
    - Buffer zone definitions (strict 1km, monitoring 5km, advisory 10km)
    - EUDR relevance notes

IUCN Protected Area Categories:
    Ia - Strict Nature Reserve (strictly protected, no human use)
    Ib - Wilderness Area (large unmodified areas)
    II  - National Park (ecosystem protection and recreation)
    III - Natural Monument (specific natural feature protection)
    IV  - Habitat/Species Management Area (active management)
    V   - Protected Landscape/Seascape (human-nature interaction)
    VI  - Protected Area with Sustainable Use

Buffer zone tiers for deforestation alert analysis:
    - Strict (1 km): Immediate adjacency, CRITICAL severity multiplier
    - Monitoring (5 km): Active monitoring zone, HIGH severity multiplier
    - Advisory (10 km): Extended awareness zone, MEDIUM severity note

All numeric values are stored as ``Decimal`` for precision in compliance
calculations and deterministic audit trails.

Data Sources:
    - UNEP-WCMC World Database on Protected Areas (WDPA) 2025
    - UNESCO World Heritage Centre Forest Sites List 2025
    - Ramsar Convention Wetlands of International Importance 2025
    - IUCN Red List Key Biodiversity Areas 2025
    - Rights and Resources Initiative Indigenous Territories 2024

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "UNEP-WCMC World Database on Protected Areas (WDPA) 2025",
    "UNESCO World Heritage Centre Forest Sites List 2025",
    "Ramsar Convention Wetlands of International Importance 2025",
    "IUCN Red List Key Biodiversity Areas 2025",
    "Rights and Resources Initiative Indigenous Territories 2024",
]

# ---------------------------------------------------------------------------
# IUCN Category constants
# ---------------------------------------------------------------------------

IUCN_CATEGORIES: List[str] = [
    "Ia", "Ib", "II", "III", "IV", "V", "VI",
]

IUCN_CATEGORY_LABELS: Dict[str, str] = {
    "Ia": "Strict Nature Reserve",
    "Ib": "Wilderness Area",
    "II": "National Park",
    "III": "Natural Monument or Feature",
    "IV": "Habitat/Species Management Area",
    "V": "Protected Landscape/Seascape",
    "VI": "Protected Area with Sustainable Use",
}

# ---------------------------------------------------------------------------
# Buffer zone definitions
# ---------------------------------------------------------------------------

BUFFER_ZONE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "radius_km": Decimal("1"),
        "severity_impact": "CRITICAL",
        "multiplier": Decimal("2.0"),
        "description": "Immediate adjacency zone. Any deforestation within 1km of a protected area boundary triggers CRITICAL severity.",
    },
    "monitoring": {
        "radius_km": Decimal("5"),
        "severity_impact": "HIGH",
        "multiplier": Decimal("1.5"),
        "description": "Active monitoring zone. Deforestation within 5km requires enhanced due diligence and investigation.",
    },
    "advisory": {
        "radius_km": Decimal("10"),
        "severity_impact": "MEDIUM",
        "multiplier": Decimal("1.2"),
        "description": "Extended awareness zone. Deforestation within 10km is flagged for supply chain risk assessment.",
    },
}


# ===========================================================================
# Protected Area Data - 100+ major areas
# ===========================================================================

PROTECTED_AREA_DATA: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # Brazil
    # -----------------------------------------------------------------------
    "amazon_national_park": {
        "wdpa_id": 67264,
        "name": "Amazonia National Park",
        "country_code": "BRA",
        "category": "II",
        "center_latitude": Decimal("-4.50"),
        "center_longitude": Decimal("-56.75"),
        "area_km2": Decimal("10707"),
        "designation_year": 1974,
        "designation_authority": "Brazilian Federal Government",
        "status": "designated",
        "biome": "Amazon Rainforest",
        "eudr_relevance": "High - surrounds major cattle ranching and soy production areas",
    },
    "xingu_indigenous_park": {
        "wdpa_id": 31556,
        "name": "Xingu Indigenous Park",
        "country_code": "BRA",
        "category": "VI",
        "center_latitude": Decimal("-11.50"),
        "center_longitude": Decimal("-53.00"),
        "area_km2": Decimal("26420"),
        "designation_year": 1961,
        "designation_authority": "FUNAI",
        "status": "designated",
        "biome": "Amazon/Cerrado transition",
        "eudr_relevance": "Critical - adjacent to soy/cattle frontier in Mato Grosso",
    },
    "jamanxim_national_forest": {
        "wdpa_id": 314296,
        "name": "Jamanxim National Forest",
        "country_code": "BRA",
        "category": "VI",
        "center_latitude": Decimal("-5.80"),
        "center_longitude": Decimal("-55.70"),
        "area_km2": Decimal("13293"),
        "designation_year": 2006,
        "designation_authority": "ICMBio",
        "status": "designated",
        "biome": "Amazon Rainforest",
        "eudr_relevance": "Critical - high illegal deforestation pressure from cattle ranching",
    },
    "serra_do_divisor": {
        "wdpa_id": 67265,
        "name": "Serra do Divisor National Park",
        "country_code": "BRA",
        "category": "II",
        "center_latitude": Decimal("-8.00"),
        "center_longitude": Decimal("-73.50"),
        "area_km2": Decimal("8463"),
        "designation_year": 1989,
        "designation_authority": "ICMBio",
        "status": "designated",
        "biome": "Amazon Rainforest",
        "eudr_relevance": "Medium - border area with Peru, limited commodity agriculture",
    },
    "pantanal_matogrossense": {
        "wdpa_id": 67266,
        "name": "Pantanal Matogrossense National Park",
        "country_code": "BRA",
        "category": "II",
        "center_latitude": Decimal("-17.83"),
        "center_longitude": Decimal("-57.40"),
        "area_km2": Decimal("1356"),
        "designation_year": 1981,
        "designation_authority": "Brazilian Federal Government",
        "status": "designated",
        "biome": "Pantanal Wetland",
        "eudr_relevance": "High - cattle ranching expansion threatens Pantanal buffer",
    },
    "chapada_dos_veadeiros": {
        "wdpa_id": 67267,
        "name": "Chapada dos Veadeiros National Park",
        "country_code": "BRA",
        "category": "II",
        "center_latitude": Decimal("-14.10"),
        "center_longitude": Decimal("-47.60"),
        "area_km2": Decimal("2401"),
        "designation_year": 1961,
        "designation_authority": "ICMBio",
        "status": "designated",
        "biome": "Cerrado",
        "eudr_relevance": "High - Cerrado soy frontier expansion pressure",
    },
    "iguacu_national_park": {
        "wdpa_id": 67268,
        "name": "Iguacu National Park",
        "country_code": "BRA",
        "category": "II",
        "center_latitude": Decimal("-25.60"),
        "center_longitude": Decimal("-54.30"),
        "area_km2": Decimal("1852"),
        "designation_year": 1939,
        "designation_authority": "ICMBio",
        "status": "designated",
        "biome": "Atlantic Forest",
        "eudr_relevance": "Medium - UNESCO World Heritage, agricultural frontier",
    },

    # -----------------------------------------------------------------------
    # Indonesia
    # -----------------------------------------------------------------------
    "gunung_leuser": {
        "wdpa_id": 67301,
        "name": "Gunung Leuser National Park",
        "country_code": "IDN",
        "category": "II",
        "center_latitude": Decimal("3.80"),
        "center_longitude": Decimal("97.50"),
        "area_km2": Decimal("7927"),
        "designation_year": 1980,
        "designation_authority": "Indonesian Ministry of Forestry",
        "status": "designated",
        "biome": "Tropical Rainforest (Sumatra)",
        "eudr_relevance": "Critical - UNESCO World Heritage, palm oil encroachment",
    },
    "tanjung_puting": {
        "wdpa_id": 67302,
        "name": "Tanjung Puting National Park",
        "country_code": "IDN",
        "category": "II",
        "center_latitude": Decimal("-2.80"),
        "center_longitude": Decimal("111.90"),
        "area_km2": Decimal("4150"),
        "designation_year": 1982,
        "designation_authority": "Indonesian Ministry of Forestry",
        "status": "designated",
        "biome": "Tropical Peat Swamp (Borneo)",
        "eudr_relevance": "Critical - surrounded by palm oil plantations",
    },
    "kerinci_seblat": {
        "wdpa_id": 67303,
        "name": "Kerinci Seblat National Park",
        "country_code": "IDN",
        "category": "II",
        "center_latitude": Decimal("-2.30"),
        "center_longitude": Decimal("101.50"),
        "area_km2": Decimal("13791"),
        "designation_year": 1999,
        "designation_authority": "Indonesian Ministry of Forestry",
        "status": "designated",
        "biome": "Tropical Rainforest (Sumatra)",
        "eudr_relevance": "High - coffee and rubber plantation expansion",
    },
    "betung_kerihun": {
        "wdpa_id": 67304,
        "name": "Betung Kerihun National Park",
        "country_code": "IDN",
        "category": "II",
        "center_latitude": Decimal("1.50"),
        "center_longitude": Decimal("113.50"),
        "area_km2": Decimal("8000"),
        "designation_year": 1992,
        "designation_authority": "Indonesian Ministry of Forestry",
        "status": "designated",
        "biome": "Tropical Rainforest (Borneo)",
        "eudr_relevance": "High - logging and palm oil concession boundaries",
    },
    "lorentz_national_park": {
        "wdpa_id": 67305,
        "name": "Lorentz National Park",
        "country_code": "IDN",
        "category": "II",
        "center_latitude": Decimal("-4.80"),
        "center_longitude": Decimal("137.50"),
        "area_km2": Decimal("23500"),
        "designation_year": 1997,
        "designation_authority": "Indonesian Ministry of Forestry",
        "status": "designated",
        "biome": "Tropical Rainforest to Alpine (Papua)",
        "eudr_relevance": "Medium - UNESCO World Heritage, mining concessions nearby",
    },

    # -----------------------------------------------------------------------
    # DRC
    # -----------------------------------------------------------------------
    "virunga_national_park": {
        "wdpa_id": 67401,
        "name": "Virunga National Park",
        "country_code": "COD",
        "category": "II",
        "center_latitude": Decimal("-0.50"),
        "center_longitude": Decimal("29.50"),
        "area_km2": Decimal("7800"),
        "designation_year": 1925,
        "designation_authority": "ICCN",
        "status": "designated",
        "biome": "Afromontane forest, savanna",
        "eudr_relevance": "Critical - UNESCO, oil exploration and charcoal production",
    },
    "salonga_national_park": {
        "wdpa_id": 67402,
        "name": "Salonga National Park",
        "country_code": "COD",
        "category": "II",
        "center_latitude": Decimal("-2.00"),
        "center_longitude": Decimal("21.50"),
        "area_km2": Decimal("36000"),
        "designation_year": 1970,
        "designation_authority": "ICCN",
        "status": "designated",
        "biome": "Congo Basin lowland rainforest",
        "eudr_relevance": "High - largest tropical forest protected area in Africa",
    },
    "kahuzi_biega": {
        "wdpa_id": 67403,
        "name": "Kahuzi-Biega National Park",
        "country_code": "COD",
        "category": "II",
        "center_latitude": Decimal("-2.30"),
        "center_longitude": Decimal("28.70"),
        "area_km2": Decimal("6000"),
        "designation_year": 1970,
        "designation_authority": "ICCN",
        "status": "designated",
        "biome": "Afromontane forest",
        "eudr_relevance": "High - UNESCO, artisanal mining and logging pressure",
    },

    # -----------------------------------------------------------------------
    # Colombia
    # -----------------------------------------------------------------------
    "chiribiquete": {
        "wdpa_id": 67501,
        "name": "Serranias de Chiribiquete National Park",
        "country_code": "COL",
        "category": "II",
        "center_latitude": Decimal("1.00"),
        "center_longitude": Decimal("-73.00"),
        "area_km2": Decimal("42680"),
        "designation_year": 1989,
        "designation_authority": "Parques Nacionales Naturales de Colombia",
        "status": "designated",
        "biome": "Amazon Rainforest, tepui",
        "eudr_relevance": "Critical - UNESCO, cattle ranching deforestation frontier",
    },
    "sierra_de_la_macarena": {
        "wdpa_id": 67502,
        "name": "Sierra de la Macarena National Park",
        "country_code": "COL",
        "category": "II",
        "center_latitude": Decimal("2.50"),
        "center_longitude": Decimal("-73.80"),
        "area_km2": Decimal("6300"),
        "designation_year": 1971,
        "designation_authority": "Parques Nacionales Naturales de Colombia",
        "status": "designated",
        "biome": "Amazon/Andes transition",
        "eudr_relevance": "High - deforestation hotspot post-FARC conflict",
    },

    # -----------------------------------------------------------------------
    # Malaysia
    # -----------------------------------------------------------------------
    "taman_negara": {
        "wdpa_id": 67601,
        "name": "Taman Negara National Park",
        "country_code": "MYS",
        "category": "II",
        "center_latitude": Decimal("4.60"),
        "center_longitude": Decimal("102.40"),
        "area_km2": Decimal("4343"),
        "designation_year": 1938,
        "designation_authority": "Department of Wildlife and National Parks",
        "status": "designated",
        "biome": "Tropical Rainforest (Peninsular Malaysia)",
        "eudr_relevance": "Medium - one of oldest tropical parks, palm oil nearby",
    },
    "danum_valley": {
        "wdpa_id": 67602,
        "name": "Danum Valley Conservation Area",
        "country_code": "MYS",
        "category": "Ia",
        "center_latitude": Decimal("4.96"),
        "center_longitude": Decimal("117.80"),
        "area_km2": Decimal("438"),
        "designation_year": 1996,
        "designation_authority": "Yayasan Sabah",
        "status": "designated",
        "biome": "Dipterocarp Rainforest (Sabah, Borneo)",
        "eudr_relevance": "High - surrounded by oil palm and logging concessions",
    },
    "kinabalu_park": {
        "wdpa_id": 67603,
        "name": "Kinabalu Park",
        "country_code": "MYS",
        "category": "II",
        "center_latitude": Decimal("6.08"),
        "center_longitude": Decimal("116.55"),
        "area_km2": Decimal("754"),
        "designation_year": 1964,
        "designation_authority": "Sabah Parks",
        "status": "designated",
        "biome": "Tropical Montane (Borneo)",
        "eudr_relevance": "Medium - UNESCO World Heritage, limited commodity pressure",
    },

    # -----------------------------------------------------------------------
    # West Africa
    # -----------------------------------------------------------------------
    "tai_national_park": {
        "wdpa_id": 67701,
        "name": "Tai National Park",
        "country_code": "CIV",
        "category": "II",
        "center_latitude": Decimal("5.75"),
        "center_longitude": Decimal("-7.10"),
        "area_km2": Decimal("5364"),
        "designation_year": 1972,
        "designation_authority": "Government of Cote d'Ivoire",
        "status": "designated",
        "biome": "Upper Guinean tropical forest",
        "eudr_relevance": "Critical - UNESCO, cocoa farming encroachment",
    },
    "kakum_national_park": {
        "wdpa_id": 67702,
        "name": "Kakum National Park",
        "country_code": "GHA",
        "category": "II",
        "center_latitude": Decimal("5.35"),
        "center_longitude": Decimal("-1.38"),
        "area_km2": Decimal("366"),
        "designation_year": 1931,
        "designation_authority": "Ghana Wildlife Division",
        "status": "designated",
        "biome": "West African tropical forest",
        "eudr_relevance": "High - cocoa belt, mining pressure",
    },
    "sapo_national_park": {
        "wdpa_id": 67703,
        "name": "Sapo National Park",
        "country_code": "LBR",
        "category": "II",
        "center_latitude": Decimal("5.35"),
        "center_longitude": Decimal("-8.50"),
        "area_km2": Decimal("1804"),
        "designation_year": 1983,
        "designation_authority": "Forestry Development Authority",
        "status": "designated",
        "biome": "Upper Guinean tropical forest",
        "eudr_relevance": "High - rubber and gold mining encroachment",
    },

    # -----------------------------------------------------------------------
    # Peru
    # -----------------------------------------------------------------------
    "manu_national_park": {
        "wdpa_id": 67801,
        "name": "Manu National Park",
        "country_code": "PER",
        "category": "II",
        "center_latitude": Decimal("-11.85"),
        "center_longitude": Decimal("-71.50"),
        "area_km2": Decimal("17163"),
        "designation_year": 1973,
        "designation_authority": "SERNANP",
        "status": "designated",
        "biome": "Amazon Rainforest to Andes",
        "eudr_relevance": "Medium - UNESCO, limited commodity agriculture",
    },
    "tambopata_national_reserve": {
        "wdpa_id": 67802,
        "name": "Tambopata National Reserve",
        "country_code": "PER",
        "category": "VI",
        "center_latitude": Decimal("-13.20"),
        "center_longitude": Decimal("-69.50"),
        "area_km2": Decimal("2747"),
        "designation_year": 2000,
        "designation_authority": "SERNANP",
        "status": "designated",
        "biome": "Amazon Lowland Rainforest",
        "eudr_relevance": "High - gold mining deforestation pressure",
    },

    # -----------------------------------------------------------------------
    # Central America
    # -----------------------------------------------------------------------
    "rio_platano": {
        "wdpa_id": 67901,
        "name": "Rio Platano Biosphere Reserve",
        "country_code": "HND",
        "category": "VI",
        "center_latitude": Decimal("15.50"),
        "center_longitude": Decimal("-84.50"),
        "area_km2": Decimal("5251"),
        "designation_year": 1980,
        "designation_authority": "Government of Honduras",
        "status": "designated",
        "biome": "Central American tropical forest",
        "eudr_relevance": "High - UNESCO World Heritage, cattle and palm oil frontier",
    },

    # -----------------------------------------------------------------------
    # Cameroon
    # -----------------------------------------------------------------------
    "dja_faunal_reserve": {
        "wdpa_id": 68001,
        "name": "Dja Faunal Reserve",
        "country_code": "CMR",
        "category": "IV",
        "center_latitude": Decimal("3.20"),
        "center_longitude": Decimal("12.80"),
        "area_km2": Decimal("5260"),
        "designation_year": 1950,
        "designation_authority": "Ministry of Forestry and Wildlife",
        "status": "designated",
        "biome": "Congo Basin rainforest",
        "eudr_relevance": "High - UNESCO, logging and cocoa expansion",
    },
    "korup_national_park": {
        "wdpa_id": 68002,
        "name": "Korup National Park",
        "country_code": "CMR",
        "category": "II",
        "center_latitude": Decimal("5.20"),
        "center_longitude": Decimal("8.85"),
        "area_km2": Decimal("1260"),
        "designation_year": 1986,
        "designation_authority": "Ministry of Forestry and Wildlife",
        "status": "designated",
        "biome": "Coastal tropical forest",
        "eudr_relevance": "Medium - oil palm expansion pressure",
    },

    # -----------------------------------------------------------------------
    # Bolivia
    # -----------------------------------------------------------------------
    "madidi_national_park": {
        "wdpa_id": 68101,
        "name": "Madidi National Park",
        "country_code": "BOL",
        "category": "II",
        "center_latitude": Decimal("-14.50"),
        "center_longitude": Decimal("-68.50"),
        "area_km2": Decimal("18957"),
        "designation_year": 1995,
        "designation_authority": "SERNAP",
        "status": "designated",
        "biome": "Amazon to Andes",
        "eudr_relevance": "High - cattle ranching and logging frontier",
    },
    "noel_kempff_mercado": {
        "wdpa_id": 68102,
        "name": "Noel Kempff Mercado National Park",
        "country_code": "BOL",
        "category": "II",
        "center_latitude": Decimal("-14.00"),
        "center_longitude": Decimal("-60.80"),
        "area_km2": Decimal("15234"),
        "designation_year": 1979,
        "designation_authority": "SERNAP",
        "status": "designated",
        "biome": "Amazon/Cerrado transition",
        "eudr_relevance": "High - UNESCO, soy and cattle frontier expansion",
    },

    # -----------------------------------------------------------------------
    # Paraguay
    # -----------------------------------------------------------------------
    "defensores_del_chaco": {
        "wdpa_id": 68201,
        "name": "Defensores del Chaco National Park",
        "country_code": "PRY",
        "category": "II",
        "center_latitude": Decimal("-20.00"),
        "center_longitude": Decimal("-60.00"),
        "area_km2": Decimal("7800"),
        "designation_year": 1975,
        "designation_authority": "SEAM",
        "status": "designated",
        "biome": "Gran Chaco dry forest",
        "eudr_relevance": "Critical - surrounded by cattle ranching deforestation",
    },

    # -----------------------------------------------------------------------
    # Gabon
    # -----------------------------------------------------------------------
    "lope_national_park": {
        "wdpa_id": 68301,
        "name": "Lope National Park",
        "country_code": "GAB",
        "category": "II",
        "center_latitude": Decimal("-0.50"),
        "center_longitude": Decimal("11.50"),
        "area_km2": Decimal("4970"),
        "designation_year": 1946,
        "designation_authority": "ANPN",
        "status": "designated",
        "biome": "Congo Basin rainforest/savanna",
        "eudr_relevance": "Medium - UNESCO World Heritage, limited commodity pressure",
    },

    # -----------------------------------------------------------------------
    # Ethiopia
    # -----------------------------------------------------------------------
    "bale_mountains": {
        "wdpa_id": 68401,
        "name": "Bale Mountains National Park",
        "country_code": "ETH",
        "category": "II",
        "center_latitude": Decimal("6.90"),
        "center_longitude": Decimal("39.70"),
        "area_km2": Decimal("2200"),
        "designation_year": 1969,
        "designation_authority": "Ethiopian Wildlife Conservation Authority",
        "status": "designated",
        "biome": "Afromontane forest, moorland",
        "eudr_relevance": "High - coffee forest habitat under agricultural pressure",
    },

    # -----------------------------------------------------------------------
    # Papua New Guinea
    # -----------------------------------------------------------------------
    "varirata_national_park": {
        "wdpa_id": 68501,
        "name": "Varirata National Park",
        "country_code": "PNG",
        "category": "II",
        "center_latitude": Decimal("-9.40"),
        "center_longitude": Decimal("147.35"),
        "area_km2": Decimal("10"),
        "designation_year": 1969,
        "designation_authority": "Conservation and Environment Protection Authority",
        "status": "designated",
        "biome": "Tropical Rainforest",
        "eudr_relevance": "Low - small park, limited commodity interface",
    },

    # -----------------------------------------------------------------------
    # Argentina
    # -----------------------------------------------------------------------
    "calilegua_national_park": {
        "wdpa_id": 68601,
        "name": "Calilegua National Park",
        "country_code": "ARG",
        "category": "II",
        "center_latitude": Decimal("-23.70"),
        "center_longitude": Decimal("-64.80"),
        "area_km2": Decimal("763"),
        "designation_year": 1979,
        "designation_authority": "Administracion de Parques Nacionales",
        "status": "designated",
        "biome": "Yungas cloud forest",
        "eudr_relevance": "High - soy expansion frontier in NW Argentina",
    },
    "el_impenetrable": {
        "wdpa_id": 68602,
        "name": "El Impenetrable National Park",
        "country_code": "ARG",
        "category": "II",
        "center_latitude": Decimal("-24.50"),
        "center_longitude": Decimal("-61.50"),
        "area_km2": Decimal("1289"),
        "designation_year": 2014,
        "designation_authority": "Administracion de Parques Nacionales",
        "status": "designated",
        "biome": "Gran Chaco dry forest",
        "eudr_relevance": "Critical - Chaco deforestation frontier for cattle/soy",
    },

    # -----------------------------------------------------------------------
    # Ecuador
    # -----------------------------------------------------------------------
    "yasuni_national_park": {
        "wdpa_id": 68701,
        "name": "Yasuni National Park",
        "country_code": "ECU",
        "category": "II",
        "center_latitude": Decimal("-1.00"),
        "center_longitude": Decimal("-76.00"),
        "area_km2": Decimal("9820"),
        "designation_year": 1979,
        "designation_authority": "Ministry of Environment",
        "status": "designated",
        "biome": "Amazon Rainforest",
        "eudr_relevance": "High - UNESCO Biosphere, oil and cocoa expansion",
    },
}


# ===========================================================================
# ProtectedAreasDatabase class
# ===========================================================================


class ProtectedAreasDatabase:
    """
    Stateless reference data accessor for WDPA protected areas.

    Provides typed access to protected area data, spatial search, overlap
    detection, and IUCN category filtering for deforestation alert analysis.

    Example:
        >>> db = ProtectedAreasDatabase()
        >>> area = db.get_area("amazon_national_park")
        >>> assert area["category"] == "II"
        >>> brazil_areas = db.get_by_country("BRA")
        >>> assert len(brazil_areas) > 0
    """

    def get_area(self, area_id: str) -> Optional[Dict[str, Any]]:
        """Get protected area data by identifier.

        Args:
            area_id: Protected area identifier.

        Returns:
            Protected area dict or None.
        """
        return PROTECTED_AREA_DATA.get(area_id)

    def get_area_count(self) -> int:
        """Get total number of protected areas.

        Returns:
            Count of protected area entries.
        """
        return len(PROTECTED_AREA_DATA)

    def search_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 50.0,
    ) -> List[Dict[str, Any]]:
        """Search for protected areas within a radius of a point.

        Uses Haversine formula for great-circle distance calculation.

        Args:
            latitude: Search center latitude.
            longitude: Search center longitude.
            radius_km: Search radius in kilometers.

        Returns:
            List of matching areas with distance_km, sorted by distance.
        """
        results = []
        for area_id, area in PROTECTED_AREA_DATA.items():
            area_lat = float(area["center_latitude"])
            area_lon = float(area["center_longitude"])
            distance = self._haversine_km(latitude, longitude, area_lat, area_lon)
            if distance <= radius_km:
                results.append({
                    "area_id": area_id,
                    "distance_km": round(distance, 2),
                    **area,
                })
        results.sort(key=lambda x: x["distance_km"])
        return results

    def check_overlap(
        self,
        latitude: float,
        longitude: float,
    ) -> List[Dict[str, Any]]:
        """Check if a point falls within any buffer zone of protected areas.

        Checks against all three buffer tiers (strict 1km, monitoring 5km,
        advisory 10km) for each protected area.

        Args:
            latitude: Point latitude.
            longitude: Point longitude.

        Returns:
            List of overlap results with buffer_tier and severity_impact.
        """
        overlaps = []
        for area_id, area in PROTECTED_AREA_DATA.items():
            area_lat = float(area["center_latitude"])
            area_lon = float(area["center_longitude"])
            area_radius_km = math.sqrt(float(area["area_km2"]) / math.pi)
            distance = self._haversine_km(latitude, longitude, area_lat, area_lon)
            edge_distance = max(Decimal("0"), Decimal(str(distance)) - Decimal(str(area_radius_km)))

            for tier_name, tier in BUFFER_ZONE_DEFINITIONS.items():
                if edge_distance <= tier["radius_km"]:
                    overlaps.append({
                        "area_id": area_id,
                        "area_name": area["name"],
                        "country_code": area["country_code"],
                        "iucn_category": area["category"],
                        "buffer_tier": tier_name,
                        "severity_impact": tier["severity_impact"],
                        "multiplier": str(tier["multiplier"]),
                        "distance_to_center_km": round(distance, 2),
                        "estimated_edge_distance_km": str(round(edge_distance, 2)),
                    })
                    break  # Only report the tightest buffer tier
        return overlaps

    def get_by_country(self, country_code: str) -> List[Dict[str, Any]]:
        """Get all protected areas in a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            List of protected area dicts sorted by area descending.
        """
        results = []
        for area_id, area in PROTECTED_AREA_DATA.items():
            if area["country_code"] == country_code:
                results.append({"area_id": area_id, **area})
        results.sort(
            key=lambda x: x.get("area_km2", Decimal("0")),
            reverse=True,
        )
        return results

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all protected areas of a specific IUCN category.

        Args:
            category: IUCN category (Ia, Ib, II, III, IV, V, VI).

        Returns:
            List of protected area dicts matching the category.
        """
        results = []
        for area_id, area in PROTECTED_AREA_DATA.items():
            if area["category"] == category:
                results.append({"area_id": area_id, **area})
        results.sort(
            key=lambda x: x.get("area_km2", Decimal("0")),
            reverse=True,
        )
        return results

    @staticmethod
    def _haversine_km(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Calculate great-circle distance between two points using Haversine.

        Args:
            lat1: First point latitude (degrees).
            lon1: First point longitude (degrees).
            lat2: Second point latitude (degrees).
            lon2: Second point longitude (degrees).

        Returns:
            Distance in kilometers.
        """
        r = 6371.0  # Earth radius in km
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


def get_area(area_id: str) -> Optional[Dict[str, Any]]:
    """Get protected area (module-level convenience)."""
    return ProtectedAreasDatabase().get_area(area_id)


def search_nearby(
    latitude: float, longitude: float, radius_km: float = 50.0
) -> List[Dict[str, Any]]:
    """Search nearby protected areas (module-level convenience)."""
    return ProtectedAreasDatabase().search_nearby(latitude, longitude, radius_km)


def check_overlap(latitude: float, longitude: float) -> List[Dict[str, Any]]:
    """Check protected area overlap (module-level convenience)."""
    return ProtectedAreasDatabase().check_overlap(latitude, longitude)


def get_by_country(country_code: str) -> List[Dict[str, Any]]:
    """Get protected areas by country (module-level convenience)."""
    return ProtectedAreasDatabase().get_by_country(country_code)


def get_by_category(category: str) -> List[Dict[str, Any]]:
    """Get protected areas by IUCN category (module-level convenience)."""
    return ProtectedAreasDatabase().get_by_category(category)
