# -*- coding: utf-8 -*-
"""
Deforestation Alert System Reference Data - AGENT-EUDR-020

Comprehensive reference data for the Deforestation Alert System Agent covering
satellite data source specifications with spectral band configurations, global
deforestation hotspot regions with FAO data and driver analysis, WDPA protected
areas with IUCN category classifications, and country-level forest cover
statistics for 180+ countries with carbon stock estimates.

This package provides four reference data modules:

1. satellite_sources (SatelliteSourceDatabase):
   - SATELLITE_SOURCE_DATA: 5 satellite sources (Sentinel-2, Landsat 8/9,
     GLAD, Hansen GFC, RADD) with resolution, revisit, spectral bands,
     coverage regions, and cloud-free revisit estimates
   - Spectral index formulas (NDVI, EVI, NBR, NDMI, SAVI)
   - Coverage maps per source and tropical region
   - Data availability timelines (1972-present)
   - Helper functions: get_source_specs(), get_coverage(), get_bands(),
     get_revisit_rate(), get_availability_timeline()

2. deforestation_hotspots (DeforestationHotspotsDatabase):
   - HOTSPOT_DATA: 30+ global deforestation hotspot regions with
     coordinates, area_at_risk_ha, annual_loss_rate_ha, and drivers
   - COUNTRY_DEFORESTATION_RATES: EUDR-relevant country rates
   - COMMODITY_LINKAGES: EUDR commodity-deforestation associations
   - HISTORICAL_TRENDS: 2000-2025 deforestation trends per region
   - Helper functions: get_hotspot(), get_country_rate(), get_drivers(),
     get_commodity_linkage(), get_trend()

3. protected_areas (ProtectedAreasDatabase):
   - PROTECTED_AREA_DATA: 100+ major WDPA protected areas with IUCN
     categories (Ia/Ib/II/III/IV/V/VI), center coordinates, area_km2,
     designation year, and buffer zone specifications
   - UNESCO World Heritage forest sites
   - Ramsar wetlands and Key Biodiversity Areas
   - Buffer zone definitions (1km strict, 5km monitoring, 10km advisory)
   - Helper functions: get_area(), search_nearby(), check_overlap(),
     get_by_country(), get_by_category()

4. country_forest_data (CountryForestDatabase):
   - COUNTRY_FOREST_DATA: 180+ countries with total_forest_ha,
     forest_pct_of_land, annual_change_rate, primary_forest_pct,
     plantation_pct, and natural_regeneration_pct
   - Hansen tree cover thresholds (10/15/20/25/30% canopy)
   - Carbon stock estimates (above/below ground, soil, dead wood, litter)
   - Country-specific forest definitions (min area, canopy %, tree height)
   - Helper functions: get_forest_stats(), get_cover_change(),
     get_carbon_stock(), get_forest_definition(), compare_countries()

All reference data uses ISO 3166-1 alpha-3 country codes, Decimal values
for regulatory-grade precision, and is designed for deterministic,
zero-hallucination deforestation alert analysis per EU 2023/1115 Articles
2, 9, 10, 11, and 31.

Data Sources:
    - ESA Copernicus Sentinel-2 Mission Specifications
    - USGS Landsat 8/9 OLI/TIRS Specifications
    - University of Maryland GLAD Forest Change Alerts
    - Hansen/UMD/Google/USGS/NASA Global Forest Change v1.10
    - Wageningen University RADD Forest Disturbance Alert System
    - FAO Global Forest Resources Assessment 2025
    - UNEP-WCMC World Database on Protected Areas 2025
    - Global Forest Watch Dashboard 2025
    - IPCC 2006/2019 Refinement Guidelines Vol 4

Example:
    >>> from greenlang.agents.eudr.deforestation_alert_system.reference_data import (
    ...     SatelliteSourceDatabase,
    ...     DeforestationHotspotsDatabase,
    ...     ProtectedAreasDatabase,
    ...     CountryForestDatabase,
    ...     get_all_databases,
    ...     validate_all_databases,
    ... )
    >>>
    >>> # Get satellite source specs
    >>> sat_db = SatelliteSourceDatabase()
    >>> sentinel2 = sat_db.get_source_specs("sentinel2")
    >>> assert sentinel2["resolution_m"] == 10
    >>>
    >>> # Get deforestation hotspot
    >>> hotspot_db = DeforestationHotspotsDatabase()
    >>> amazon = hotspot_db.get_hotspot("amazon_arc_of_deforestation")
    >>> assert amazon is not None
    >>>
    >>> # Check protected area overlap
    >>> protected_db = ProtectedAreasDatabase()
    >>> areas = protected_db.get_by_country("BRA")
    >>> assert len(areas) > 0
    >>>
    >>> # Get country forest stats
    >>> forest_db = CountryForestDatabase()
    >>> brazil = forest_db.get_forest_stats("BRA")
    >>> assert brazil is not None

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020
Agent ID: GL-EUDR-DAS-020
Regulation: EU 2023/1115 (EUDR) Articles 2, 9, 10, 11, 31
Status: Production Ready
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module 1: Satellite Sources Database
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.deforestation_alert_system.reference_data.satellite_sources import (
    DATA_VERSION as SATELLITE_DATA_VERSION,
    DATA_SOURCES as SATELLITE_DATA_SOURCES,
    SATELLITE_SOURCE_DATA,
    SPECTRAL_INDEX_FORMULAS,
    COVERAGE_REGIONS,
    SatelliteSourceDatabase,
    get_source_specs,
    get_coverage,
    get_bands,
    get_revisit_rate,
    get_availability_timeline,
)

# ---------------------------------------------------------------------------
# Module 2: Deforestation Hotspots Database
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.deforestation_alert_system.reference_data.deforestation_hotspots import (
    DATA_VERSION as HOTSPOT_DATA_VERSION,
    DATA_SOURCES as HOTSPOT_DATA_SOURCES,
    HOTSPOT_DATA,
    COUNTRY_DEFORESTATION_RATES,
    COMMODITY_LINKAGES,
    DEFORESTATION_DRIVERS,
    DeforestationHotspotsDatabase,
    get_hotspot,
    get_country_rate,
    get_drivers,
    get_commodity_linkage,
    get_trend,
)

# ---------------------------------------------------------------------------
# Module 3: Protected Areas Database
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.deforestation_alert_system.reference_data.protected_areas import (
    DATA_VERSION as PROTECTED_AREA_DATA_VERSION,
    DATA_SOURCES as PROTECTED_AREA_DATA_SOURCES,
    IUCN_CATEGORIES,
    IUCN_CATEGORY_LABELS,
    PROTECTED_AREA_DATA,
    BUFFER_ZONE_DEFINITIONS,
    ProtectedAreasDatabase,
    get_area,
    search_nearby,
    check_overlap,
    get_by_country,
    get_by_category,
)

# ---------------------------------------------------------------------------
# Module 4: Country Forest Database
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.deforestation_alert_system.reference_data.country_forest_data import (
    DATA_VERSION as FOREST_DATA_VERSION,
    DATA_SOURCES as FOREST_DATA_SOURCES,
    COUNTRY_FOREST_DATA,
    HANSEN_THRESHOLDS,
    FOREST_DEFINITIONS,
    CountryForestDatabase,
    get_forest_stats,
    get_cover_change,
    get_carbon_stock,
    get_forest_definition,
    compare_countries,
)


# ===========================================================================
# Cross-module utility functions
# ===========================================================================


def get_all_databases() -> dict:
    """Instantiate and return all four reference data database accessors.

    Returns:
        Dict mapping database names to their class instances.

    Example:
        >>> databases = get_all_databases()
        >>> sat_db = databases["satellite_sources"]
        >>> specs = sat_db.get_source_specs("sentinel2")
    """
    return {
        "satellite_sources": SatelliteSourceDatabase(),
        "deforestation_hotspots": DeforestationHotspotsDatabase(),
        "protected_areas": ProtectedAreasDatabase(),
        "country_forest_data": CountryForestDatabase(),
    }


def validate_all_databases() -> dict:
    """Validate that all reference data modules loaded correctly.

    Checks that each database has data entries and that key EUDR-relevant
    entries are present in each dataset.

    Returns:
        Dict with per-module validation status and statistics.

    Example:
        >>> results = validate_all_databases()
        >>> assert results["overall_status"] == "valid"
        >>> assert results["satellite_sources"]["source_count"] > 0
    """
    results: dict = {
        "overall_status": "valid",
        "errors": [],
    }

    # Validate Satellite Sources Database
    try:
        sat_db = SatelliteSourceDatabase()
        sat_count = sat_db.get_source_count()
        sat_sentinel2 = sat_db.get_source_specs("sentinel2")
        results["satellite_sources"] = {
            "status": "valid",
            "source_count": sat_count,
            "data_version": SATELLITE_DATA_VERSION,
            "sentinel2_present": sat_sentinel2 is not None,
        }
    except Exception as e:
        results["satellite_sources"] = {"status": "error", "error": str(e)}
        results["errors"].append(f"SatelliteSources: {e}")
        results["overall_status"] = "invalid"

    # Validate Deforestation Hotspots Database
    try:
        hotspot_db = DeforestationHotspotsDatabase()
        hotspot_count = hotspot_db.get_hotspot_count()
        amazon = hotspot_db.get_hotspot("amazon_arc_of_deforestation")
        results["deforestation_hotspots"] = {
            "status": "valid",
            "hotspot_count": hotspot_count,
            "data_version": HOTSPOT_DATA_VERSION,
            "amazon_hotspot_present": amazon is not None,
        }
    except Exception as e:
        results["deforestation_hotspots"] = {"status": "error", "error": str(e)}
        results["errors"].append(f"DeforestationHotspots: {e}")
        results["overall_status"] = "invalid"

    # Validate Protected Areas Database
    try:
        protected_db = ProtectedAreasDatabase()
        protected_count = protected_db.get_area_count()
        brazil_areas = protected_db.get_by_country("BRA")
        results["protected_areas"] = {
            "status": "valid",
            "area_count": protected_count,
            "data_version": PROTECTED_AREA_DATA_VERSION,
            "brazil_areas_present": len(brazil_areas) > 0,
            "iucn_categories_count": len(IUCN_CATEGORIES),
        }
    except Exception as e:
        results["protected_areas"] = {"status": "error", "error": str(e)}
        results["errors"].append(f"ProtectedAreas: {e}")
        results["overall_status"] = "invalid"

    # Validate Country Forest Database
    try:
        forest_db = CountryForestDatabase()
        forest_count = forest_db.get_country_count()
        brazil_forest = forest_db.get_forest_stats("BRA")
        results["country_forest_data"] = {
            "status": "valid",
            "country_count": forest_count,
            "data_version": FOREST_DATA_VERSION,
            "brazil_data_present": brazil_forest is not None,
            "hansen_thresholds_count": len(HANSEN_THRESHOLDS),
        }
    except Exception as e:
        results["country_forest_data"] = {"status": "error", "error": str(e)}
        results["errors"].append(f"CountryForestData: {e}")
        results["overall_status"] = "invalid"

    return results


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    # Satellite Sources Database
    "SATELLITE_DATA_VERSION",
    "SATELLITE_DATA_SOURCES",
    "SATELLITE_SOURCE_DATA",
    "SPECTRAL_INDEX_FORMULAS",
    "COVERAGE_REGIONS",
    "SatelliteSourceDatabase",
    "get_source_specs",
    "get_coverage",
    "get_bands",
    "get_revisit_rate",
    "get_availability_timeline",
    # Deforestation Hotspots Database
    "HOTSPOT_DATA_VERSION",
    "HOTSPOT_DATA_SOURCES",
    "HOTSPOT_DATA",
    "COUNTRY_DEFORESTATION_RATES",
    "COMMODITY_LINKAGES",
    "DEFORESTATION_DRIVERS",
    "DeforestationHotspotsDatabase",
    "get_hotspot",
    "get_country_rate",
    "get_drivers",
    "get_commodity_linkage",
    "get_trend",
    # Protected Areas Database
    "PROTECTED_AREA_DATA_VERSION",
    "PROTECTED_AREA_DATA_SOURCES",
    "IUCN_CATEGORIES",
    "IUCN_CATEGORY_LABELS",
    "PROTECTED_AREA_DATA",
    "BUFFER_ZONE_DEFINITIONS",
    "ProtectedAreasDatabase",
    "get_area",
    "search_nearby",
    "check_overlap",
    "get_by_country",
    "get_by_category",
    # Country Forest Database
    "FOREST_DATA_VERSION",
    "FOREST_DATA_SOURCES",
    "COUNTRY_FOREST_DATA",
    "HANSEN_THRESHOLDS",
    "FOREST_DEFINITIONS",
    "CountryForestDatabase",
    "get_forest_stats",
    "get_cover_change",
    "get_carbon_stock",
    "get_forest_definition",
    "compare_countries",
    # Cross-module utilities
    "get_all_databases",
    "validate_all_databases",
]
