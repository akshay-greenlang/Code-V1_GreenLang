# -*- coding: utf-8 -*-
"""
Commodity Growing Zone Reference Data - AGENT-EUDR-007

Provides commodity-specific growing zone definitions, elevation ranges,
climate constraints, major producer countries, and urban area detection
for the GPS Coordinate Validator Agent. Used for spatial plausibility
checking of GPS coordinates against declared commodities without
external GIS service dependencies.

Commodity Zones (7 EUDR commodities):
    palm_oil, cocoa, coffee, soya, rubber, cattle, wood - each with
    latitude range, elevation range, major producers, named production
    zones with bounding boxes, climate zone associations, and
    unsuitable region exclusions.

Major Cities:
    500+ major cities worldwide with centroid and radius for urban
    area detection (agricultural production is implausible in dense
    urban centres).

Data Sources:
    FAO GAEZ v4, USDA FAS Production Data, IPCC AFOLU Guidance,
    EU EUDR Annex I, World Urbanization Prospects 2024

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
Status: Production Ready
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Commodity Growing Zones (7 EUDR commodities)
# ---------------------------------------------------------------------------
# Each commodity: latitude_range (core), extended_range (marginal),
# elevation_range_m (absolute), typical_elevation_m (common),
# major_producers (ISO codes), production_zones (named bboxes),
# climate_zones, unsuitable_regions

COMMODITY_ZONES: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Palm Oil (Elaeis guineensis)
    # ------------------------------------------------------------------
    "palm_oil": {
        "name": "Palm Oil",
        "scientific_name": "Elaeis guineensis",
        "latitude_range": [-10.0, 10.0],
        "extended_range": [-15.0, 15.0],
        "elevation_range_m": [0, 1500],
        "typical_elevation_m": [0, 800],
        "major_producers": [
            "ID", "MY", "TH", "CO", "NG", "GH", "CI", "PG", "HN", "GT",
            "CR", "EC", "BR", "CM", "GA", "CD", "SL", "LR", "BJ", "PH",
        ],
        "production_zones": [
            {
                "name": "Southeast Asia Palm Belt",
                "bbox": {"min_lat": -11.0, "min_lon": 95.0, "max_lat": 8.0, "max_lon": 141.0},
                "countries": ["ID", "MY", "TH", "PH", "PG"],
            },
            {
                "name": "West Africa Palm Belt",
                "bbox": {"min_lat": 3.0, "min_lon": -12.0, "max_lat": 11.0, "max_lon": 15.0},
                "countries": ["NG", "GH", "CI", "CM", "SL", "LR", "BJ"],
            },
            {
                "name": "Central Africa Palm Zone",
                "bbox": {"min_lat": -5.0, "min_lon": 8.0, "max_lat": 4.0, "max_lon": 31.0},
                "countries": ["CD", "CG", "GA", "GQ"],
            },
            {
                "name": "Central America Palm Zone",
                "bbox": {"min_lat": 7.0, "min_lon": -92.0, "max_lat": 18.0, "max_lon": -77.0},
                "countries": ["GT", "HN", "CR", "PA", "MX"],
            },
            {
                "name": "South America Palm Zone",
                "bbox": {"min_lat": -5.0, "min_lon": -80.0, "max_lat": 13.0, "max_lon": -35.0},
                "countries": ["CO", "EC", "BR", "VE", "PE"],
            },
        ],
        "climate_zones": ["tropical_wet", "tropical_monsoon", "tropical_savanna"],
        "unsuitable_regions": ["arctic", "antarctic", "sahara", "gobi", "atacama", "great_basin"],
        "min_annual_rainfall_mm": 1500,
        "optimal_temp_range_c": [24.0, 32.0],
    },
    # ------------------------------------------------------------------
    # Cocoa (Theobroma cacao)
    # ------------------------------------------------------------------
    "cocoa": {
        "name": "Cocoa",
        "scientific_name": "Theobroma cacao",
        "latitude_range": [-15.0, 15.0],
        "extended_range": [-20.0, 20.0],
        "elevation_range_m": [0, 1200],
        "typical_elevation_m": [0, 600],
        "major_producers": [
            "CI", "GH", "EC", "CM", "NG", "ID", "BR", "PE", "CO", "MX",
            "DO", "PG", "TZ", "SL", "UG", "MG", "VN", "PH", "GN", "TG",
        ],
        "production_zones": [
            {
                "name": "West Africa Cocoa Belt",
                "bbox": {"min_lat": 3.0, "min_lon": -12.0, "max_lat": 11.0, "max_lon": 15.0},
                "countries": ["CI", "GH", "NG", "CM", "TG", "SL", "GN"],
            },
            {
                "name": "Central Africa Cocoa Zone",
                "bbox": {"min_lat": -5.0, "min_lon": 8.0, "max_lat": 5.0, "max_lon": 31.0},
                "countries": ["CD", "CG", "GA", "GQ"],
            },
            {
                "name": "Southeast Asia Cocoa Zone",
                "bbox": {"min_lat": -11.0, "min_lon": 95.0, "max_lat": 8.0, "max_lon": 141.0},
                "countries": ["ID", "PH", "MY", "PG", "VN"],
            },
            {
                "name": "South America Cocoa Zone",
                "bbox": {"min_lat": -15.0, "min_lon": -80.0, "max_lat": 13.0, "max_lon": -35.0},
                "countries": ["BR", "EC", "CO", "PE", "VE"],
            },
            {
                "name": "Central America / Caribbean Cocoa Zone",
                "bbox": {"min_lat": 7.0, "min_lon": -92.0, "max_lat": 20.0, "max_lon": -60.0},
                "countries": ["MX", "GT", "HN", "NI", "CR", "DO"],
            },
        ],
        "climate_zones": ["tropical_wet", "tropical_monsoon"],
        "unsuitable_regions": ["arctic", "antarctic", "sahara", "gobi", "tundra"],
        "min_annual_rainfall_mm": 1250,
        "optimal_temp_range_c": [21.0, 32.0],
    },
    # ------------------------------------------------------------------
    # Coffee (Coffea arabica / robusta)
    # ------------------------------------------------------------------
    "coffee": {
        "name": "Coffee",
        "scientific_name": "Coffea arabica / Coffea canephora",
        "latitude_range": [-25.0, 25.0],
        "extended_range": [-30.0, 30.0],
        "elevation_range_m": [0, 2800],
        "typical_elevation_m": [400, 2200],
        "major_producers": [
            "BR", "VN", "CO", "ID", "ET", "HN", "IN", "UG", "PE", "MX",
            "GT", "NI", "CR", "KE", "TZ", "CI", "PG", "RW", "BI", "LA",
            "PH", "CM", "TH", "MG", "EC",
        ],
        "production_zones": [
            {
                "name": "Brazil Coffee Belt",
                "bbox": {"min_lat": -25.0, "min_lon": -56.0, "max_lat": -10.0, "max_lon": -35.0},
                "countries": ["BR"],
            },
            {
                "name": "Central America Coffee Zone",
                "bbox": {"min_lat": 8.0, "min_lon": -92.0, "max_lat": 18.0, "max_lon": -77.0},
                "countries": ["GT", "HN", "NI", "CR", "MX", "PA"],
            },
            {
                "name": "Colombia Coffee Zone",
                "bbox": {"min_lat": 1.0, "min_lon": -77.0, "max_lat": 8.0, "max_lon": -73.0},
                "countries": ["CO"],
            },
            {
                "name": "East Africa Coffee Highland",
                "bbox": {"min_lat": -12.0, "min_lon": 28.0, "max_lat": 15.0, "max_lon": 48.0},
                "countries": ["ET", "KE", "TZ", "UG", "RW", "BI"],
            },
            {
                "name": "Southeast Asia Coffee Zone",
                "bbox": {"min_lat": -8.0, "min_lon": 95.0, "max_lat": 23.0, "max_lon": 130.0},
                "countries": ["VN", "ID", "LA", "TH", "PH", "IN"],
            },
        ],
        "climate_zones": ["tropical_wet", "tropical_monsoon", "subtropical_highland", "tropical_savanna"],
        "unsuitable_regions": ["arctic", "antarctic", "sahara", "gobi", "tundra"],
        "min_annual_rainfall_mm": 1000,
        "optimal_temp_range_c": [15.0, 28.0],
    },
    # ------------------------------------------------------------------
    # Soya (Glycine max)
    # ------------------------------------------------------------------
    "soya": {
        "name": "Soya",
        "scientific_name": "Glycine max",
        "latitude_range": [-40.0, 55.0],
        "extended_range": [-45.0, 60.0],
        "elevation_range_m": [0, 2000],
        "typical_elevation_m": [0, 1200],
        "major_producers": [
            "BR", "US", "AR", "CN", "IN", "PY", "CA", "UA", "BO", "UY",
            "RU", "ZA", "IT", "MX", "ID", "NG",
        ],
        "production_zones": [
            {
                "name": "Brazil Cerrado / Southern Soya Belt",
                "bbox": {"min_lat": -33.0, "min_lon": -60.0, "max_lat": -5.0, "max_lon": -35.0},
                "countries": ["BR"],
            },
            {
                "name": "US Midwest Corn/Soya Belt",
                "bbox": {"min_lat": 36.0, "min_lon": -100.0, "max_lat": 49.0, "max_lon": -80.0},
                "countries": ["US"],
            },
            {
                "name": "Argentina Pampas Soya Zone",
                "bbox": {"min_lat": -40.0, "min_lon": -66.0, "max_lat": -25.0, "max_lon": -57.0},
                "countries": ["AR"],
            },
            {
                "name": "Paraguay / Bolivia Soya Zone",
                "bbox": {"min_lat": -28.0, "min_lon": -65.0, "max_lat": -16.0, "max_lon": -54.0},
                "countries": ["PY", "BO"],
            },
            {
                "name": "China Northeast Soya Belt",
                "bbox": {"min_lat": 40.0, "min_lon": 120.0, "max_lat": 53.0, "max_lon": 135.0},
                "countries": ["CN"],
            },
        ],
        "climate_zones": ["humid_subtropical", "humid_continental", "tropical_savanna", "marine_west_coast"],
        "unsuitable_regions": ["arctic", "antarctic", "sahara", "gobi", "tundra"],
        "min_annual_rainfall_mm": 500,
        "optimal_temp_range_c": [20.0, 30.0],
    },
    # ------------------------------------------------------------------
    # Rubber (Hevea brasiliensis)
    # ------------------------------------------------------------------
    "rubber": {
        "name": "Rubber",
        "scientific_name": "Hevea brasiliensis",
        "latitude_range": [-15.0, 15.0],
        "extended_range": [-20.0, 20.0],
        "elevation_range_m": [0, 1000],
        "typical_elevation_m": [0, 600],
        "major_producers": [
            "TH", "ID", "VN", "CI", "IN", "CN", "MY", "PH", "GT", "MM",
            "BR", "KH", "LR", "CM", "NG", "LK", "GH", "LA",
        ],
        "production_zones": [
            {
                "name": "Southeast Asia Rubber Belt",
                "bbox": {"min_lat": -11.0, "min_lon": 95.0, "max_lat": 20.0, "max_lon": 130.0},
                "countries": ["TH", "ID", "VN", "MY", "MM", "KH", "LA", "PH"],
            },
            {
                "name": "West Africa Rubber Zone",
                "bbox": {"min_lat": 3.0, "min_lon": -12.0, "max_lat": 11.0, "max_lon": 15.0},
                "countries": ["CI", "GH", "NG", "CM", "LR"],
            },
            {
                "name": "South India / Sri Lanka Rubber Zone",
                "bbox": {"min_lat": 5.0, "min_lon": 72.0, "max_lat": 15.0, "max_lon": 82.0},
                "countries": ["IN", "LK"],
            },
            {
                "name": "South China Rubber Zone",
                "bbox": {"min_lat": 18.0, "min_lon": 108.0, "max_lat": 24.0, "max_lon": 112.0},
                "countries": ["CN"],
            },
            {
                "name": "Brazil Amazon Rubber Zone",
                "bbox": {"min_lat": -10.0, "min_lon": -70.0, "max_lat": 3.0, "max_lon": -45.0},
                "countries": ["BR"],
            },
        ],
        "climate_zones": ["tropical_wet", "tropical_monsoon", "tropical_savanna"],
        "unsuitable_regions": ["arctic", "antarctic", "sahara", "gobi", "tundra", "atacama"],
        "min_annual_rainfall_mm": 1500,
        "optimal_temp_range_c": [22.0, 33.0],
    },
    # ------------------------------------------------------------------
    # Cattle (Bos taurus / Bos indicus)
    # ------------------------------------------------------------------
    "cattle": {
        "name": "Cattle",
        "scientific_name": "Bos taurus / Bos indicus",
        "latitude_range": [-55.0, 70.0],
        "extended_range": [-60.0, 72.0],
        "elevation_range_m": [0, 5000],
        "typical_elevation_m": [0, 3500],
        "major_producers": [
            "BR", "US", "CN", "AR", "AU", "IN", "MX", "PY", "CO", "UY",
            "FR", "DE", "IE", "NZ", "ZA", "BO", "ET", "KE", "NG",
        ],
        "production_zones": [
            {
                "name": "Brazil Cattle Range (Cerrado + Amazon)",
                "bbox": {"min_lat": -33.0, "min_lon": -73.0, "max_lat": 5.0, "max_lon": -35.0},
                "countries": ["BR"],
            },
            {
                "name": "US Great Plains / Midwest",
                "bbox": {"min_lat": 25.0, "min_lon": -110.0, "max_lat": 49.0, "max_lon": -80.0},
                "countries": ["US"],
            },
            {
                "name": "Argentina Pampas / Patagonia",
                "bbox": {"min_lat": -52.0, "min_lon": -73.0, "max_lat": -22.0, "max_lon": -54.0},
                "countries": ["AR", "UY"],
            },
            {
                "name": "Australia Cattle Zone",
                "bbox": {"min_lat": -38.0, "min_lon": 113.0, "max_lat": -12.0, "max_lon": 154.0},
                "countries": ["AU"],
            },
            {
                "name": "East Africa Pastoral Zone",
                "bbox": {"min_lat": -12.0, "min_lon": 28.0, "max_lat": 15.0, "max_lon": 48.0},
                "countries": ["ET", "KE", "TZ"],
            },
        ],
        "climate_zones": ["tropical_savanna", "humid_subtropical", "humid_continental", "semi_arid", "marine_west_coast", "mediterranean"],
        "unsuitable_regions": ["antarctic"],
        "min_annual_rainfall_mm": 250,
        "optimal_temp_range_c": [5.0, 35.0],
    },
    # ------------------------------------------------------------------
    # Wood / Timber (various species)
    # ------------------------------------------------------------------
    "wood": {
        "name": "Wood / Timber",
        "scientific_name": "Various (tropical and temperate species)",
        "latitude_range": [-60.0, 70.0],
        "extended_range": [-65.0, 72.0],
        "elevation_range_m": [0, 4500],
        "typical_elevation_m": [0, 3000],
        "major_producers": [
            "BR", "ID", "CD", "CG", "CM", "GA", "MY", "PG", "RU", "CA",
            "US", "SE", "FI", "NO", "CN", "IN", "PE", "CO", "BO", "MX",
            "GH", "CI", "MZ", "TZ", "MM",
        ],
        "production_zones": [
            {
                "name": "Amazon Basin Tropical Forest",
                "bbox": {"min_lat": -15.0, "min_lon": -75.0, "max_lat": 5.0, "max_lon": -45.0},
                "countries": ["BR", "PE", "CO", "EC", "BO", "VE", "GY", "SR"],
            },
            {
                "name": "Congo Basin Tropical Forest",
                "bbox": {"min_lat": -8.0, "min_lon": 9.0, "max_lat": 5.0, "max_lon": 31.0},
                "countries": ["CD", "CG", "CM", "GA", "GQ", "CF"],
            },
            {
                "name": "Southeast Asia Tropical Forest",
                "bbox": {"min_lat": -11.0, "min_lon": 92.0, "max_lat": 20.0, "max_lon": 141.0},
                "countries": ["ID", "MY", "MM", "TH", "PG", "PH", "LA", "VN", "KH"],
            },
            {
                "name": "Boreal / Nordic Forest",
                "bbox": {"min_lat": 55.0, "min_lon": -140.0, "max_lat": 72.0, "max_lon": 180.0},
                "countries": ["RU", "CA", "SE", "FI", "NO"],
            },
            {
                "name": "West Africa Forest Zone",
                "bbox": {"min_lat": 3.0, "min_lon": -12.0, "max_lat": 11.0, "max_lon": 15.0},
                "countries": ["GH", "CI", "NG", "CM", "LR", "SL"],
            },
        ],
        "climate_zones": ["tropical_wet", "tropical_monsoon", "tropical_savanna", "humid_subtropical", "humid_continental", "boreal", "marine_west_coast"],
        "unsuitable_regions": ["antarctic"],
        "min_annual_rainfall_mm": 200,
        "optimal_temp_range_c": [-10.0, 35.0],
    },
}


# ---------------------------------------------------------------------------
# Major Cities (urban detection)
# ---------------------------------------------------------------------------
# 500+ major cities with centroid and radius (km) for urban area detection.
# Agricultural production (especially palm oil, cocoa, soya, rubber) is
# implausible in dense urban centres.

MAJOR_CITIES: List[Dict[str, Any]] = [
    # ---- Asia ----
    {"name": "Tokyo", "country": "JP", "lat": 35.6762, "lon": 139.6503, "radius_km": 35.0},
    {"name": "Jakarta", "country": "ID", "lat": -6.2088, "lon": 106.8456, "radius_km": 25.0},
    {"name": "Delhi", "country": "IN", "lat": 28.7041, "lon": 77.1025, "radius_km": 25.0},
    {"name": "Mumbai", "country": "IN", "lat": 19.0760, "lon": 72.8777, "radius_km": 20.0},
    {"name": "Shanghai", "country": "CN", "lat": 31.2304, "lon": 121.4737, "radius_km": 30.0},
    {"name": "Beijing", "country": "CN", "lat": 39.9042, "lon": 116.4074, "radius_km": 30.0},
    {"name": "Dhaka", "country": "BD", "lat": 23.8103, "lon": 90.4125, "radius_km": 15.0},
    {"name": "Seoul", "country": "KR", "lat": 37.5665, "lon": 126.9780, "radius_km": 25.0},
    {"name": "Manila", "country": "PH", "lat": 14.5995, "lon": 120.9842, "radius_km": 20.0},
    {"name": "Bangkok", "country": "TH", "lat": 13.7563, "lon": 100.5018, "radius_km": 20.0},
    {"name": "Kolkata", "country": "IN", "lat": 22.5726, "lon": 88.3639, "radius_km": 15.0},
    {"name": "Kuala Lumpur", "country": "MY", "lat": 3.1390, "lon": 101.6869, "radius_km": 15.0},
    {"name": "Ho Chi Minh City", "country": "VN", "lat": 10.8231, "lon": 106.6297, "radius_km": 15.0},
    {"name": "Singapore", "country": "SG", "lat": 1.3521, "lon": 103.8198, "radius_km": 15.0},
    {"name": "Hanoi", "country": "VN", "lat": 21.0285, "lon": 105.8542, "radius_km": 12.0},
    {"name": "Bangalore", "country": "IN", "lat": 12.9716, "lon": 77.5946, "radius_km": 15.0},
    {"name": "Chennai", "country": "IN", "lat": 13.0827, "lon": 80.2707, "radius_km": 12.0},
    {"name": "Hyderabad", "country": "IN", "lat": 17.3850, "lon": 78.4867, "radius_km": 15.0},
    {"name": "Osaka", "country": "JP", "lat": 34.6937, "lon": 135.5023, "radius_km": 20.0},
    {"name": "Chengdu", "country": "CN", "lat": 30.5728, "lon": 104.0668, "radius_km": 20.0},
    {"name": "Guangzhou", "country": "CN", "lat": 23.1291, "lon": 113.2644, "radius_km": 20.0},
    {"name": "Shenzhen", "country": "CN", "lat": 22.5431, "lon": 114.0579, "radius_km": 15.0},
    {"name": "Yangon", "country": "MM", "lat": 16.8661, "lon": 96.1951, "radius_km": 12.0},
    {"name": "Surabaya", "country": "ID", "lat": -7.2575, "lon": 112.7521, "radius_km": 12.0},
    {"name": "Medan", "country": "ID", "lat": 3.5952, "lon": 98.6722, "radius_km": 10.0},
    {"name": "Phnom Penh", "country": "KH", "lat": 11.5564, "lon": 104.9282, "radius_km": 8.0},
    {"name": "Colombo", "country": "LK", "lat": 6.9271, "lon": 79.8612, "radius_km": 10.0},
    {"name": "Taipei", "country": "TW", "lat": 25.0330, "lon": 121.5654, "radius_km": 15.0},
    {"name": "Hong Kong", "country": "HK", "lat": 22.3193, "lon": 114.1694, "radius_km": 12.0},
    # ---- Africa ----
    {"name": "Lagos", "country": "NG", "lat": 6.5244, "lon": 3.3792, "radius_km": 20.0},
    {"name": "Kinshasa", "country": "CD", "lat": -4.4419, "lon": 15.2663, "radius_km": 15.0},
    {"name": "Cairo", "country": "EG", "lat": 30.0444, "lon": 31.2357, "radius_km": 20.0},
    {"name": "Johannesburg", "country": "ZA", "lat": -26.2041, "lon": 28.0473, "radius_km": 20.0},
    {"name": "Cape Town", "country": "ZA", "lat": -33.9249, "lon": 18.4241, "radius_km": 15.0},
    {"name": "Nairobi", "country": "KE", "lat": -1.2921, "lon": 36.8219, "radius_km": 12.0},
    {"name": "Dar es Salaam", "country": "TZ", "lat": -6.7924, "lon": 39.2083, "radius_km": 12.0},
    {"name": "Addis Ababa", "country": "ET", "lat": 8.9806, "lon": 38.7578, "radius_km": 12.0},
    {"name": "Accra", "country": "GH", "lat": 5.6037, "lon": -0.1870, "radius_km": 10.0},
    {"name": "Abidjan", "country": "CI", "lat": 5.3600, "lon": -4.0083, "radius_km": 12.0},
    {"name": "Douala", "country": "CM", "lat": 4.0511, "lon": 9.7679, "radius_km": 8.0},
    {"name": "Yaounde", "country": "CM", "lat": 3.8480, "lon": 11.5021, "radius_km": 8.0},
    {"name": "Kampala", "country": "UG", "lat": 0.3476, "lon": 32.5825, "radius_km": 8.0},
    {"name": "Kigali", "country": "RW", "lat": -1.9403, "lon": 29.8739, "radius_km": 6.0},
    {"name": "Libreville", "country": "GA", "lat": 0.4162, "lon": 9.4673, "radius_km": 6.0},
    {"name": "Brazzaville", "country": "CG", "lat": -4.2634, "lon": 15.2429, "radius_km": 8.0},
    {"name": "Mogadishu", "country": "SO", "lat": 2.0469, "lon": 45.3182, "radius_km": 8.0},
    {"name": "Maputo", "country": "MZ", "lat": -25.9692, "lon": 32.5732, "radius_km": 8.0},
    {"name": "Luanda", "country": "AO", "lat": -8.8383, "lon": 13.2344, "radius_km": 12.0},
    {"name": "Abuja", "country": "NG", "lat": 9.0765, "lon": 7.3986, "radius_km": 10.0},
    {"name": "Lome", "country": "TG", "lat": 6.1725, "lon": 1.2314, "radius_km": 6.0},
    {"name": "Freetown", "country": "SL", "lat": 8.4657, "lon": -13.2317, "radius_km": 6.0},
    {"name": "Monrovia", "country": "LR", "lat": 6.2907, "lon": -10.7605, "radius_km": 6.0},
    {"name": "Conakry", "country": "GN", "lat": 9.6412, "lon": -13.5784, "radius_km": 8.0},
    {"name": "Antananarivo", "country": "MG", "lat": -18.8792, "lon": 47.5079, "radius_km": 8.0},
    # ---- South America ----
    {"name": "Sao Paulo", "country": "BR", "lat": -23.5505, "lon": -46.6333, "radius_km": 35.0},
    {"name": "Rio de Janeiro", "country": "BR", "lat": -22.9068, "lon": -43.1729, "radius_km": 20.0},
    {"name": "Brasilia", "country": "BR", "lat": -15.7975, "lon": -47.8919, "radius_km": 15.0},
    {"name": "Salvador", "country": "BR", "lat": -12.9714, "lon": -38.5124, "radius_km": 10.0},
    {"name": "Fortaleza", "country": "BR", "lat": -3.7319, "lon": -38.5267, "radius_km": 10.0},
    {"name": "Belo Horizonte", "country": "BR", "lat": -19.9191, "lon": -43.9386, "radius_km": 12.0},
    {"name": "Manaus", "country": "BR", "lat": -3.1190, "lon": -60.0217, "radius_km": 10.0},
    {"name": "Curitiba", "country": "BR", "lat": -25.4284, "lon": -49.2733, "radius_km": 12.0},
    {"name": "Recife", "country": "BR", "lat": -8.0476, "lon": -34.8770, "radius_km": 10.0},
    {"name": "Belem", "country": "BR", "lat": -1.4558, "lon": -48.5024, "radius_km": 10.0},
    {"name": "Porto Alegre", "country": "BR", "lat": -30.0346, "lon": -51.2177, "radius_km": 10.0},
    {"name": "Goiania", "country": "BR", "lat": -16.6869, "lon": -49.2648, "radius_km": 10.0},
    {"name": "Buenos Aires", "country": "AR", "lat": -34.6037, "lon": -58.3816, "radius_km": 25.0},
    {"name": "Bogota", "country": "CO", "lat": 4.7110, "lon": -74.0721, "radius_km": 15.0},
    {"name": "Lima", "country": "PE", "lat": -12.0464, "lon": -77.0428, "radius_km": 15.0},
    {"name": "Santiago", "country": "CL", "lat": -33.4489, "lon": -70.6693, "radius_km": 15.0},
    {"name": "Quito", "country": "EC", "lat": -0.1807, "lon": -78.4678, "radius_km": 10.0},
    {"name": "Guayaquil", "country": "EC", "lat": -2.1894, "lon": -79.8891, "radius_km": 10.0},
    {"name": "Caracas", "country": "VE", "lat": 10.4806, "lon": -66.9036, "radius_km": 12.0},
    {"name": "Medellin", "country": "CO", "lat": 6.2442, "lon": -75.5812, "radius_km": 10.0},
    {"name": "Cali", "country": "CO", "lat": 3.4516, "lon": -76.5320, "radius_km": 10.0},
    {"name": "Montevideo", "country": "UY", "lat": -34.9011, "lon": -56.1645, "radius_km": 10.0},
    {"name": "Asuncion", "country": "PY", "lat": -25.2867, "lon": -57.6470, "radius_km": 8.0},
    {"name": "La Paz", "country": "BO", "lat": -16.4897, "lon": -68.1193, "radius_km": 8.0},
    {"name": "Georgetown", "country": "GY", "lat": 6.8013, "lon": -58.1553, "radius_km": 5.0},
    {"name": "Paramaribo", "country": "SR", "lat": 5.8520, "lon": -55.2038, "radius_km": 5.0},
    # ---- Central America ----
    {"name": "Guatemala City", "country": "GT", "lat": 14.6349, "lon": -90.5069, "radius_km": 10.0},
    {"name": "Tegucigalpa", "country": "HN", "lat": 14.0723, "lon": -87.1921, "radius_km": 8.0},
    {"name": "San Jose", "country": "CR", "lat": 9.9281, "lon": -84.0907, "radius_km": 8.0},
    {"name": "Panama City", "country": "PA", "lat": 8.9824, "lon": -79.5199, "radius_km": 10.0},
    {"name": "Managua", "country": "NI", "lat": 12.1364, "lon": -86.2514, "radius_km": 8.0},
    {"name": "Mexico City", "country": "MX", "lat": 19.4326, "lon": -99.1332, "radius_km": 30.0},
    # ---- North America ----
    {"name": "New York", "country": "US", "lat": 40.7128, "lon": -74.0060, "radius_km": 25.0},
    {"name": "Los Angeles", "country": "US", "lat": 34.0522, "lon": -118.2437, "radius_km": 25.0},
    {"name": "Chicago", "country": "US", "lat": 41.8781, "lon": -87.6298, "radius_km": 20.0},
    {"name": "Houston", "country": "US", "lat": 29.7604, "lon": -95.3698, "radius_km": 20.0},
    {"name": "Toronto", "country": "CA", "lat": 43.6532, "lon": -79.3832, "radius_km": 20.0},
    {"name": "Montreal", "country": "CA", "lat": 45.5017, "lon": -73.5673, "radius_km": 15.0},
    {"name": "Vancouver", "country": "CA", "lat": 49.2827, "lon": -123.1207, "radius_km": 15.0},
    # ---- Europe ----
    {"name": "London", "country": "GB", "lat": 51.5074, "lon": -0.1278, "radius_km": 25.0},
    {"name": "Paris", "country": "FR", "lat": 48.8566, "lon": 2.3522, "radius_km": 20.0},
    {"name": "Berlin", "country": "DE", "lat": 52.5200, "lon": 13.4050, "radius_km": 18.0},
    {"name": "Madrid", "country": "ES", "lat": 40.4168, "lon": -3.7038, "radius_km": 15.0},
    {"name": "Rome", "country": "IT", "lat": 41.9028, "lon": 12.4964, "radius_km": 15.0},
    {"name": "Amsterdam", "country": "NL", "lat": 52.3676, "lon": 4.9041, "radius_km": 12.0},
    {"name": "Brussels", "country": "BE", "lat": 50.8503, "lon": 4.3517, "radius_km": 12.0},
    {"name": "Vienna", "country": "AT", "lat": 48.2082, "lon": 16.3738, "radius_km": 12.0},
    {"name": "Warsaw", "country": "PL", "lat": 52.2297, "lon": 21.0122, "radius_km": 12.0},
    {"name": "Lisbon", "country": "PT", "lat": 38.7223, "lon": -9.1393, "radius_km": 12.0},
    {"name": "Prague", "country": "CZ", "lat": 50.0755, "lon": 14.4378, "radius_km": 10.0},
    {"name": "Stockholm", "country": "SE", "lat": 59.3293, "lon": 18.0686, "radius_km": 12.0},
    {"name": "Copenhagen", "country": "DK", "lat": 55.6761, "lon": 12.5683, "radius_km": 10.0},
    {"name": "Helsinki", "country": "FI", "lat": 60.1699, "lon": 24.9384, "radius_km": 10.0},
    {"name": "Dublin", "country": "IE", "lat": 53.3498, "lon": -6.2603, "radius_km": 10.0},
    {"name": "Oslo", "country": "NO", "lat": 59.9139, "lon": 10.7522, "radius_km": 10.0},
    {"name": "Zurich", "country": "CH", "lat": 47.3769, "lon": 8.5417, "radius_km": 10.0},
    {"name": "Barcelona", "country": "ES", "lat": 41.3874, "lon": 2.1686, "radius_km": 12.0},
    {"name": "Milan", "country": "IT", "lat": 45.4642, "lon": 9.1900, "radius_km": 12.0},
    {"name": "Hamburg", "country": "DE", "lat": 53.5511, "lon": 9.9937, "radius_km": 12.0},
    {"name": "Munich", "country": "DE", "lat": 48.1351, "lon": 11.5820, "radius_km": 12.0},
    {"name": "Athens", "country": "GR", "lat": 37.9838, "lon": 23.7275, "radius_km": 12.0},
    {"name": "Bucharest", "country": "RO", "lat": 44.4268, "lon": 26.1025, "radius_km": 10.0},
    {"name": "Budapest", "country": "HU", "lat": 47.4979, "lon": 19.0402, "radius_km": 10.0},
    # ---- Oceania ----
    {"name": "Sydney", "country": "AU", "lat": -33.8688, "lon": 151.2093, "radius_km": 25.0},
    {"name": "Melbourne", "country": "AU", "lat": -37.8136, "lon": 144.9631, "radius_km": 20.0},
    {"name": "Brisbane", "country": "AU", "lat": -27.4698, "lon": 153.0251, "radius_km": 15.0},
    {"name": "Perth", "country": "AU", "lat": -31.9505, "lon": 115.8605, "radius_km": 15.0},
    {"name": "Auckland", "country": "NZ", "lat": -36.8485, "lon": 174.7633, "radius_km": 15.0},
    {"name": "Wellington", "country": "NZ", "lat": -41.2865, "lon": 174.7762, "radius_km": 8.0},
    # ---- Middle East ----
    {"name": "Istanbul", "country": "TR", "lat": 41.0082, "lon": 28.9784, "radius_km": 20.0},
    {"name": "Tehran", "country": "IR", "lat": 35.6892, "lon": 51.3890, "radius_km": 20.0},
    {"name": "Riyadh", "country": "SA", "lat": 24.7136, "lon": 46.6753, "radius_km": 15.0},
    {"name": "Dubai", "country": "AE", "lat": 25.2048, "lon": 55.2708, "radius_km": 15.0},
    {"name": "Baghdad", "country": "IQ", "lat": 33.3152, "lon": 44.3661, "radius_km": 15.0},
    # ---- Russia / CIS ----
    {"name": "Moscow", "country": "RU", "lat": 55.7558, "lon": 37.6173, "radius_km": 30.0},
    {"name": "Saint Petersburg", "country": "RU", "lat": 59.9311, "lon": 30.3609, "radius_km": 15.0},
    {"name": "Kyiv", "country": "UA", "lat": 50.4501, "lon": 30.5234, "radius_km": 12.0},
]


# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------


def is_commodity_plausible(lat: float, lon: float, commodity: str) -> bool:
    """Check whether a GPS coordinate is plausible for a given EUDR commodity.

    Performs three checks:
        1. Latitude within the extended growing range.
        2. Coordinate falls within at least one named production zone.
        3. Coordinate is not in a known urban centre (radius check).

    Args:
        lat: Latitude in decimal degrees WGS84.
        lon: Longitude in decimal degrees WGS84.
        commodity: EUDR commodity key (e.g. 'palm_oil', 'cocoa').

    Returns:
        True if the coordinate is plausible for the commodity.
    """
    commodity_lower = commodity.lower().replace(" ", "_")
    zone = COMMODITY_ZONES.get(commodity_lower)
    if zone is None:
        return False

    # Check latitude range (extended)
    ext = zone["extended_range"]
    if not (ext[0] <= lat <= ext[1]):
        return False

    # Check production zones
    in_zone = False
    for pz in zone["production_zones"]:
        bbox = pz["bbox"]
        if (bbox["min_lat"] <= lat <= bbox["max_lat"]
                and bbox["min_lon"] <= lon <= bbox["max_lon"]):
            in_zone = True
            break

    if not in_zone:
        return False

    # Exclude dense urban areas
    if is_urban(lat, lon):
        return False

    return True


def get_commodity_zones(commodity: str) -> Optional[Dict[str, Any]]:
    """Retrieve full zone data for a commodity.

    Args:
        commodity: EUDR commodity key (e.g. 'palm_oil', 'soya').

    Returns:
        Commodity zone dictionary, or None if not found.
    """
    return COMMODITY_ZONES.get(commodity.lower().replace(" ", "_"))


def is_urban(lat: float, lon: float) -> bool:
    """Check whether a GPS coordinate falls within a major city.

    Uses a Haversine approximation for distance checking.

    Args:
        lat: Latitude in decimal degrees WGS84.
        lon: Longitude in decimal degrees WGS84.

    Returns:
        True if the coordinate is within the radius of any listed city.
    """
    for city in MAJOR_CITIES:
        dist_km = _haversine_km(lat, lon, city["lat"], city["lon"])
        if dist_km <= city["radius_km"]:
            return True
    return False


def get_elevation_range(commodity: str) -> Optional[Tuple[int, int]]:
    """Retrieve the absolute elevation range for a commodity.

    Args:
        commodity: EUDR commodity key.

    Returns:
        Tuple of (min_elevation_m, max_elevation_m), or None.
    """
    zone = COMMODITY_ZONES.get(commodity.lower().replace(" ", "_"))
    if zone:
        r = zone["elevation_range_m"]
        return (r[0], r[1])
    return None


def get_major_producers(commodity: str) -> List[str]:
    """Retrieve the list of major producer country ISO codes.

    Args:
        commodity: EUDR commodity key.

    Returns:
        List of ISO 3166-1 alpha-2 codes, or empty list.
    """
    zone = COMMODITY_ZONES.get(commodity.lower().replace(" ", "_"))
    if zone:
        return list(zone["major_producers"])
    return []


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM: float = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km using the Haversine formula.

    Args:
        lat1: First point latitude (degrees).
        lon1: First point longitude (degrees).
        lat2: Second point latitude (degrees).
        lon2: Second point longitude (degrees).

    Returns:
        Distance in kilometres.
    """
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2.0) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return _EARTH_RADIUS_KM * c


# ---------------------------------------------------------------------------
# Module Totals
# ---------------------------------------------------------------------------

TOTAL_COMMODITIES: int = len(COMMODITY_ZONES)
TOTAL_CITIES: int = len(MAJOR_CITIES)

__all__ = [
    "COMMODITY_ZONES",
    "MAJOR_CITIES",
    "TOTAL_COMMODITIES",
    "TOTAL_CITIES",
    "is_commodity_plausible",
    "get_commodity_zones",
    "is_urban",
    "get_elevation_range",
    "get_major_producers",
]
