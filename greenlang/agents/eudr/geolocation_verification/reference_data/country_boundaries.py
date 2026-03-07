# -*- coding: utf-8 -*-
"""
Country Boundary Reference Data - AGENT-EUDR-002

Provides simplified bounding box data for 100+ countries relevant to EUDR
commodity supply chains. Used for fast country-match validation of GPS
coordinates without requiring external GIS services.

Each entry is a tuple: (lat_min, lat_max, lon_min, lon_max)

Data source: Natural Earth 1:110m Admin 0 boundaries (simplified)

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ISO 3166-1 alpha-2 -> (lat_min, lat_max, lon_min, lon_max)
COUNTRY_BOUNDING_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    # ---- EUDR High-Priority Countries (Major commodity producers) ----
    # South America
    "BR": (-33.75, 5.27, -73.99, -34.79),    # Brazil
    "CO": (-4.23, 13.39, -79.00, -66.87),     # Colombia
    "PE": (-18.35, -0.04, -81.33, -68.65),    # Peru
    "EC": (-5.01, 1.68, -81.08, -75.19),      # Ecuador
    "BO": (-22.90, -9.68, -69.64, -57.45),    # Bolivia
    "PY": (-27.61, -19.29, -62.65, -54.26),   # Paraguay
    "AR": (-55.06, -21.78, -73.57, -53.64),   # Argentina
    "UY": (-35.00, -30.09, -58.44, -53.07),   # Uruguay
    "VE": (0.63, 12.20, -73.38, -59.80),      # Venezuela
    "GY": (1.17, 8.56, -61.39, -56.48),       # Guyana
    "SR": (1.83, 6.01, -58.07, -53.98),       # Suriname
    "GF": (2.11, 5.78, -54.60, -51.61),       # French Guiana

    # Central America & Caribbean
    "GT": (13.74, 17.82, -92.23, -88.22),     # Guatemala
    "HN": (12.98, 16.51, -89.35, -83.11),     # Honduras
    "NI": (10.71, 15.03, -87.69, -82.56),     # Nicaragua
    "CR": (8.03, 11.22, -85.95, -82.55),      # Costa Rica
    "PA": (7.20, 9.65, -83.05, -77.17),       # Panama
    "MX": (14.53, 32.72, -118.40, -86.70),    # Mexico

    # West Africa (Cocoa/Coffee belt)
    "GH": (4.74, 11.17, -3.26, 1.20),         # Ghana
    "CI": (4.36, 10.74, -8.60, -2.49),        # Ivory Coast (Cote d'Ivoire)
    "CM": (1.65, 13.08, 8.49, 16.19),         # Cameroon
    "NG": (4.27, 13.89, 2.69, 14.68),         # Nigeria
    "TG": (6.10, 11.14, -0.15, 1.81),         # Togo
    "GN": (7.19, 12.68, -14.93, -7.64),       # Guinea
    "SL": (6.93, 10.00, -13.30, -10.27),      # Sierra Leone
    "LR": (4.34, 8.55, -11.49, -7.37),        # Liberia
    "SN": (12.31, 16.69, -17.54, -11.36),     # Senegal
    "BF": (9.39, 15.08, -5.52, 2.40),         # Burkina Faso

    # Central Africa (Cocoa/Wood)
    "CD": (-13.46, 5.39, 12.18, 31.31),       # DR Congo
    "CG": (-5.03, 3.70, 11.21, 18.65),        # Republic of Congo
    "GA": (-3.98, 2.33, 8.70, 14.50),         # Gabon
    "GQ": (-1.47, 3.77, 5.61, 11.34),         # Equatorial Guinea
    "CF": (2.22, 11.00, 14.42, 27.46),        # Central African Republic

    # East Africa (Coffee)
    "ET": (3.40, 14.89, 32.99, 48.00),        # Ethiopia
    "KE": (-4.68, 5.02, 33.91, 41.91),        # Kenya
    "TZ": (-11.75, -1.00, 29.33, 40.44),      # Tanzania
    "UG": (-1.48, 4.23, 29.57, 35.00),        # Uganda
    "RW": (-2.84, -1.05, 28.86, 30.90),       # Rwanda
    "BI": (-4.47, -2.31, 29.00, 30.85),       # Burundi

    # Southeast Asia (Palm Oil/Rubber)
    "ID": (-11.01, 5.91, 95.01, 141.02),      # Indonesia
    "MY": (0.85, 7.36, 99.64, 119.28),        # Malaysia
    "TH": (5.61, 20.46, 97.34, 105.64),       # Thailand
    "VN": (8.56, 23.39, 102.14, 109.47),      # Vietnam
    "MM": (9.78, 28.54, 92.19, 101.17),       # Myanmar
    "KH": (10.41, 14.69, 102.34, 107.63),     # Cambodia
    "LA": (13.91, 22.50, 100.08, 107.64),     # Laos
    "PH": (4.59, 21.12, 116.93, 126.60),      # Philippines
    "PG": (-11.66, -0.87, 140.84, 157.04),    # Papua New Guinea
    "LK": (5.92, 9.84, 79.65, 81.88),         # Sri Lanka
    "IN": (6.75, 35.99, 68.16, 97.42),        # India

    # Europe (EU importers)
    "DE": (47.27, 55.06, 5.87, 15.04),        # Germany
    "FR": (41.36, 51.09, -5.14, 9.56),        # France
    "NL": (50.75, 53.47, 3.36, 7.21),         # Netherlands
    "BE": (49.50, 51.50, 2.55, 6.41),         # Belgium
    "IT": (36.65, 47.09, 6.63, 18.52),        # Italy
    "ES": (35.95, 43.79, -9.30, 4.33),        # Spain
    "PT": (36.96, 42.15, -9.50, -6.19),       # Portugal
    "AT": (46.37, 49.02, 9.53, 17.16),        # Austria
    "PL": (49.00, 54.84, 14.12, 24.15),       # Poland
    "SE": (55.34, 69.06, 11.11, 24.17),       # Sweden
    "FI": (59.81, 70.09, 20.55, 31.58),       # Finland
    "DK": (54.56, 57.75, 8.09, 15.19),        # Denmark
    "IE": (51.42, 55.39, -10.48, -5.99),      # Ireland
    "CZ": (48.55, 51.06, 12.09, 18.86),       # Czech Republic
    "RO": (43.62, 48.27, 20.26, 29.76),       # Romania
    "BG": (41.24, 44.23, 22.36, 28.61),       # Bulgaria
    "GR": (34.80, 41.75, 19.37, 29.64),       # Greece
    "HU": (45.74, 48.59, 16.11, 22.90),       # Hungary
    "HR": (42.39, 46.55, 13.49, 19.43),       # Croatia
    "SK": (47.73, 49.60, 16.83, 22.57),       # Slovakia
    "SI": (45.42, 46.88, 13.38, 16.61),       # Slovenia
    "LT": (53.90, 56.45, 20.93, 26.84),       # Lithuania
    "LV": (55.67, 58.08, 20.97, 28.24),       # Latvia
    "EE": (57.52, 59.68, 21.83, 28.21),       # Estonia
    "LU": (49.45, 50.18, 5.73, 6.53),         # Luxembourg
    "MT": (35.81, 36.08, 14.18, 14.57),       # Malta
    "CY": (34.57, 35.70, 32.27, 34.60),       # Cyprus
    "GB": (49.87, 60.86, -8.18, 1.77),        # United Kingdom
    "CH": (45.82, 47.81, 5.96, 10.49),        # Switzerland
    "NO": (57.96, 71.19, 4.65, 31.08),        # Norway

    # Other major producers/traders
    "CN": (18.16, 53.56, 73.50, 134.77),      # China
    "US": (24.52, 49.38, -124.73, -66.95),    # United States (contiguous)
    "AU": (-43.64, -10.06, 113.15, 153.64),   # Australia
    "RU": (41.19, 81.86, 19.64, 180.00),      # Russia
    "ZA": (-34.84, -22.13, 16.46, 32.89),     # South Africa
    "EG": (22.00, 31.67, 24.70, 36.87),       # Egypt
    "MA": (27.67, 35.93, -13.17, -1.00),      # Morocco
    "TR": (35.82, 42.11, 25.66, 44.82),       # Turkey
    "JP": (24.25, 45.52, 122.93, 153.99),     # Japan
    "KR": (33.19, 38.61, 124.61, 131.87),     # South Korea
}


def is_coordinate_in_country(
    lat: float, lon: float, country_code: str
) -> bool:
    """Check if a coordinate falls within a country's bounding box.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        country_code: ISO 3166-1 alpha-2 country code (uppercase).

    Returns:
        True if the coordinate is within the country's bounding box.
        False if the country code is unknown or coordinate is outside.
    """
    bbox = COUNTRY_BOUNDING_BOXES.get(country_code.upper())
    if bbox is None:
        return False
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def get_country_for_coordinate(
    lat: float, lon: float
) -> Optional[str]:
    """Find the country code for a given coordinate using bounding boxes.

    Uses simple bounding box containment. May return None or an incorrect
    match in border areas where bounding boxes overlap. For precise
    country identification, use PostGIS point-in-polygon queries.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        ISO 3166-1 alpha-2 country code, or None if no match found.
    """
    matches: List[str] = []
    for code, bbox in COUNTRY_BOUNDING_BOXES.items():
        lat_min, lat_max, lon_min, lon_max = bbox
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            matches.append(code)

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # If multiple matches (overlapping bounding boxes), return the
    # smallest bbox (most specific match)
    smallest = min(
        matches,
        key=lambda c: (
            (COUNTRY_BOUNDING_BOXES[c][1] - COUNTRY_BOUNDING_BOXES[c][0])
            * (COUNTRY_BOUNDING_BOXES[c][3] - COUNTRY_BOUNDING_BOXES[c][2])
        ),
    )
    return smallest
