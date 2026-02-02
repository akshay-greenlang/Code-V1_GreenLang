"""
Emission Factor Service

This module provides a comprehensive service for loading, caching, and retrieving
emission factors from the GreenLang factor database. It supports version pinning,
fallback hierarchies, and multi-source factor resolution.

Features:
- Load factors from JSON database files
- Cache layer for performance
- Fallback hierarchy (region -> country -> global)
- Version pinning for reproducibility
- Multi-source factor resolution
- GWP conversion support
"""

import json
import logging
from pathlib import Path
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from functools import lru_cache
from dataclasses import dataclass

from .models import (
    EmissionFactor,
    EmissionFactorSource,
    GWPSet,
    EmissionScope,
    EmissionCategory,
    GridEmissionFactor,
    STANDARD_GWP_AR6_100YR,
    STANDARD_GWP_AR5_100YR
)

logger = logging.getLogger(__name__)


@dataclass
class FactorLookupResult:
    """Result of a factor lookup operation."""
    factor: Optional[EmissionFactor]
    found: bool
    source_used: Optional[str] = None
    fallback_used: bool = False
    fallback_level: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class EmissionFactorService:
    """
    Service for loading and retrieving emission factors.

    This service provides:
    - Factor lookup by fuel type, region, and year
    - Version pinning support for reproducibility
    - Cache layer for performance
    - Fallback hierarchy for missing regional data
    - Multi-source factor resolution with priority ordering
    """

    # Default source priority order
    DEFAULT_SOURCE_PRIORITY = [
        EmissionFactorSource.EPA,
        EmissionFactorSource.DEFRA,
        EmissionFactorSource.IEA,
        EmissionFactorSource.IPCC,
        EmissionFactorSource.ECOINVENT
    ]

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_enabled: bool = True,
        default_gwp_set: GWPSet = GWPSet.AR5,
        source_priority: Optional[List[EmissionFactorSource]] = None
    ):
        """
        Initialize the EmissionFactorService.

        Args:
            data_dir: Path to emission factor data directory
            cache_enabled: Whether to enable caching
            default_gwp_set: Default GWP set to use (AR5, AR6)
            source_priority: Priority order for factor sources
        """
        if data_dir is None:
            # Default to the emission_factors directory relative to this file
            self.data_dir = Path(__file__).parent / "emission_factors"
        else:
            self.data_dir = Path(data_dir)

        self.cache_enabled = cache_enabled
        self.default_gwp_set = default_gwp_set
        self.source_priority = source_priority or self.DEFAULT_SOURCE_PRIORITY

        # Internal caches
        self._factor_cache: Dict[str, EmissionFactor] = {}
        self._source_data: Dict[str, Dict] = {}
        self._grid_factors: Dict[str, GridEmissionFactor] = {}

        # Load all factor databases
        self._load_all_sources()

    def _load_all_sources(self) -> None:
        """Load all emission factor databases from disk."""
        logger.info(f"Loading emission factors from {self.data_dir}")

        # Load EPA factors
        self._load_source_data("epa")

        # Load DEFRA factors
        self._load_source_data("defra")

        # Load IPCC factors
        self._load_source_data("ipcc")

        # Load IEA factors
        self._load_source_data("iea")

        logger.info(f"Loaded {len(self._factor_cache)} emission factors")

    def _load_source_data(self, source: str) -> None:
        """Load data files for a specific source."""
        source_dir = self.data_dir / source

        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return

        for json_file in source_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._source_data[f"{source}/{json_file.stem}"] = data
                    self._index_factors(source, json_file.stem, data)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

    def _index_factors(self, source: str, file_name: str, data: Dict) -> None:
        """Index individual factors from loaded data."""
        # Handle different data structures based on source and file type
        if source == "iea":
            self._index_iea_factors(data)
        elif source == "epa":
            self._index_epa_factors(data, file_name)
        elif source == "defra":
            self._index_defra_factors(data, file_name)
        elif source == "ipcc":
            self._index_ipcc_factors(data, file_name)

    def _index_iea_factors(self, data: Dict) -> None:
        """Index IEA grid emission factors."""
        # Index country-specific factors
        if "countries" in data:
            for region_name, countries in data["countries"].items():
                for country_data in countries:
                    factor_id = country_data.get("id")
                    if factor_id:
                        grid_factor = GridEmissionFactor(
                            id=factor_id,
                            country=country_data.get("country", ""),
                            country_code=country_data.get("iso2", country_data.get("country_code", "")),
                            co2_factor=Decimal(str(country_data.get("co2_factor", 0))),
                            renewable_share_pct=country_data.get("renewable_share_pct"),
                            notes=country_data.get("notes")
                        )
                        self._grid_factors[factor_id] = grid_factor
                        self._grid_factors[grid_factor.country_code.lower()] = grid_factor

        # Index regional averages
        if "regional_averages" in data:
            for region_data in data["regional_averages"]:
                factor_id = region_data.get("id")
                if factor_id:
                    grid_factor = GridEmissionFactor(
                        id=factor_id,
                        country=region_data.get("region", ""),
                        country_code=region_data.get("region", "").lower().replace(" ", "_"),
                        co2_factor=Decimal(str(region_data.get("co2_factor", 0))),
                        notes=region_data.get("notes")
                    )
                    self._grid_factors[factor_id] = grid_factor

    def _index_epa_factors(self, data: Dict, file_name: str) -> None:
        """Index EPA emission factors."""
        if "factors" in data:
            factors_list = data["factors"]
            if isinstance(factors_list, list):
                for factor_data in factors_list:
                    self._index_single_factor(factor_data, EmissionFactorSource.EPA, data.get("metadata", {}))

    def _index_defra_factors(self, data: Dict, file_name: str) -> None:
        """Index DEFRA emission factors."""
        # Handle nested structures in DEFRA data
        for category_name, category_data in data.items():
            if isinstance(category_data, list):
                for factor_data in category_data:
                    self._index_single_factor(factor_data, EmissionFactorSource.DEFRA, data.get("metadata", {}))
            elif isinstance(category_data, dict):
                for sub_category, sub_data in category_data.items():
                    if isinstance(sub_data, list):
                        for factor_data in sub_data:
                            self._index_single_factor(factor_data, EmissionFactorSource.DEFRA, data.get("metadata", {}))

    def _index_ipcc_factors(self, data: Dict, file_name: str) -> None:
        """Index IPCC emission factors and GWP values."""
        # Handle GWP values
        for category_name, category_data in data.items():
            if isinstance(category_data, list):
                for factor_data in category_data:
                    if "id" in factor_data:
                        self._index_single_factor(factor_data, EmissionFactorSource.IPCC, data.get("metadata", {}))

    def _index_single_factor(
        self,
        factor_data: Dict,
        source: EmissionFactorSource,
        metadata: Dict
    ) -> None:
        """Index a single emission factor."""
        factor_id = factor_data.get("id")
        if not factor_id:
            return

        # Determine the primary value
        value = factor_data.get("co2e_factor") or factor_data.get("co2_factor") or factor_data.get("gwp_ar6_100yr", 0)

        try:
            emission_factor = EmissionFactor(
                id=factor_id,
                value=Decimal(str(value)),
                unit=factor_data.get("unit", factor_data.get("co2_unit", "kg CO2e")),
                source=source,
                source_document=metadata.get("source_document"),
                source_url=metadata.get("source_url"),
                year=metadata.get("version", "2024")[:4] if metadata.get("version") else 2024,
                region=metadata.get("region", "global"),
                fuel_type=factor_data.get("fuel_type") or factor_data.get("fuel_name"),
                co2_factor=Decimal(str(factor_data["co2_factor"])) if factor_data.get("co2_factor") else None,
                ch4_factor=Decimal(str(factor_data["ch4_factor"])) if factor_data.get("ch4_factor") else None,
                n2o_factor=Decimal(str(factor_data["n2o_factor"])) if factor_data.get("n2o_factor") else None,
                co2e_factor=Decimal(str(factor_data["co2e_factor"])) if factor_data.get("co2e_factor") else None,
                gwp_set=GWPSet(metadata.get("gwp_set", "AR5")) if metadata.get("gwp_set") else GWPSet.AR5,
                notes=factor_data.get("notes")
            )

            self._factor_cache[factor_id] = emission_factor

            # Also index by fuel type + region for easier lookup
            if emission_factor.fuel_type:
                lookup_key = f"{emission_factor.fuel_type.lower()}_{emission_factor.region.lower()}"
                if lookup_key not in self._factor_cache:
                    self._factor_cache[lookup_key] = emission_factor

        except Exception as e:
            logger.debug(f"Could not index factor {factor_id}: {e}")

    def get_factor(
        self,
        fuel_type: str,
        region: str = "global",
        year: Optional[int] = None,
        source: Optional[EmissionFactorSource] = None,
        gwp_set: Optional[GWPSet] = None
    ) -> FactorLookupResult:
        """
        Get emission factor by fuel type, region, and year.

        Args:
            fuel_type: Type of fuel (e.g., "natural_gas", "diesel")
            region: Geographic region or country code
            year: Data year (optional, defaults to latest)
            source: Specific source to use (optional)
            gwp_set: GWP set to use (optional)

        Returns:
            FactorLookupResult with the found factor or None
        """
        gwp_set = gwp_set or self.default_gwp_set

        # Normalize inputs
        fuel_type_norm = fuel_type.lower().replace(" ", "_").replace("-", "_")
        region_norm = region.lower().replace(" ", "_")

        # Try direct lookup first
        lookup_key = f"{fuel_type_norm}_{region_norm}"
        if lookup_key in self._factor_cache:
            return FactorLookupResult(
                factor=self._factor_cache[lookup_key],
                found=True,
                source_used=str(self._factor_cache[lookup_key].source.value)
            )

        # Try by specific source priority
        sources_to_try = [source] if source else self.source_priority
        for src in sources_to_try:
            factor_id = f"ef://{src.value}/stationary/{fuel_type_norm}/{year or 2024}"
            if factor_id in self._factor_cache:
                return FactorLookupResult(
                    factor=self._factor_cache[factor_id],
                    found=True,
                    source_used=src.value
                )

        # Try fallback hierarchy: region -> country -> global
        fallback_regions = self._get_fallback_hierarchy(region_norm)
        for fallback_region in fallback_regions:
            fallback_key = f"{fuel_type_norm}_{fallback_region}"
            if fallback_key in self._factor_cache:
                return FactorLookupResult(
                    factor=self._factor_cache[fallback_key],
                    found=True,
                    source_used=str(self._factor_cache[fallback_key].source.value),
                    fallback_used=True,
                    fallback_level=fallback_region,
                    warnings=[f"Using fallback region '{fallback_region}' instead of '{region}'"]
                )

        return FactorLookupResult(
            factor=None,
            found=False,
            warnings=[f"No emission factor found for {fuel_type} in {region}"]
        )

    def get_factor_by_id(self, factor_id: str) -> FactorLookupResult:
        """
        Get emission factor by its unique ID.

        Args:
            factor_id: Unique factor identifier (e.g., ef://epa/stationary/natural_gas/2024)

        Returns:
            FactorLookupResult with the found factor or None
        """
        if factor_id in self._factor_cache:
            return FactorLookupResult(
                factor=self._factor_cache[factor_id],
                found=True,
                source_used=str(self._factor_cache[factor_id].source.value)
            )

        return FactorLookupResult(
            factor=None,
            found=False,
            warnings=[f"Factor not found: {factor_id}"]
        )

    def get_grid_factor(
        self,
        country_code: str,
        year: Optional[int] = None,
        method: str = "location-based"
    ) -> FactorLookupResult:
        """
        Get electricity grid emission factor for a country.

        Args:
            country_code: ISO 2-letter country code (e.g., "US", "GB", "DE")
            year: Data year (optional)
            method: "location-based" or "market-based"

        Returns:
            FactorLookupResult with the grid emission factor
        """
        country_code_norm = country_code.lower()

        # Try direct lookup
        if country_code_norm in self._grid_factors:
            grid_factor = self._grid_factors[country_code_norm]
            # Convert to EmissionFactor for consistent return type
            emission_factor = EmissionFactor(
                id=grid_factor.id,
                value=grid_factor.co2_factor,
                unit=grid_factor.unit,
                source=grid_factor.source,
                region=grid_factor.country,
                country_code=grid_factor.country_code,
                year=grid_factor.year,
                notes=grid_factor.notes
            )
            return FactorLookupResult(
                factor=emission_factor,
                found=True,
                source_used="IEA"
            )

        # Try with ef:// prefix
        factor_id = f"ef://iea/grid/{country_code_norm}/2024"
        if factor_id in self._grid_factors:
            grid_factor = self._grid_factors[factor_id]
            emission_factor = EmissionFactor(
                id=grid_factor.id,
                value=grid_factor.co2_factor,
                unit=grid_factor.unit,
                source=grid_factor.source,
                region=grid_factor.country,
                country_code=grid_factor.country_code,
                year=grid_factor.year,
                notes=grid_factor.notes
            )
            return FactorLookupResult(
                factor=emission_factor,
                found=True,
                source_used="IEA"
            )

        # Try fallback to regional average
        regional_fallback = self._get_regional_fallback_for_country(country_code_norm)
        if regional_fallback and regional_fallback in self._grid_factors:
            grid_factor = self._grid_factors[regional_fallback]
            emission_factor = EmissionFactor(
                id=grid_factor.id,
                value=grid_factor.co2_factor,
                unit=grid_factor.unit,
                source=grid_factor.source,
                region=grid_factor.country,
                year=grid_factor.year,
                notes=grid_factor.notes
            )
            return FactorLookupResult(
                factor=emission_factor,
                found=True,
                source_used="IEA",
                fallback_used=True,
                fallback_level="regional",
                warnings=[f"Using regional average for {country_code}"]
            )

        return FactorLookupResult(
            factor=None,
            found=False,
            warnings=[f"No grid factor found for country: {country_code}"]
        )

    def _get_fallback_hierarchy(self, region: str) -> List[str]:
        """Get fallback region hierarchy for a given region."""
        region_lower = region.lower()

        # Country-specific fallbacks
        country_to_region = {
            "us": ["north_america", "oecd", "global"],
            "uk": ["europe", "eu27", "oecd", "global"],
            "gb": ["europe", "eu27", "oecd", "global"],
            "de": ["europe", "eu27", "oecd", "global"],
            "fr": ["europe", "eu27", "oecd", "global"],
            "cn": ["asia_pacific", "non_oecd", "global"],
            "in": ["asia_pacific", "non_oecd", "global"],
            "jp": ["asia_pacific", "oecd", "global"],
            "au": ["oceania", "oecd", "global"],
            "br": ["latin_america", "non_oecd", "global"],
        }

        return country_to_region.get(region_lower, ["global"])

    def _get_regional_fallback_for_country(self, country_code: str) -> Optional[str]:
        """Get regional average fallback for a country."""
        country_to_region = {
            # European countries
            "at": "ef://iea/grid/eu27/2024",
            "be": "ef://iea/grid/eu27/2024",
            "bg": "ef://iea/grid/eu27/2024",
            "hr": "ef://iea/grid/eu27/2024",
            "cy": "ef://iea/grid/eu27/2024",
            "cz": "ef://iea/grid/eu27/2024",
            # Add more mappings as needed
        }
        return country_to_region.get(country_code.lower())

    def get_gwp(
        self,
        gas: str,
        time_horizon: int = 100,
        gwp_set: Optional[GWPSet] = None
    ) -> Optional[Decimal]:
        """
        Get Global Warming Potential for a greenhouse gas.

        Args:
            gas: Gas name or formula (e.g., "CH4", "CO2", "SF6")
            time_horizon: GWP time horizon (20 or 100 years)
            gwp_set: IPCC assessment report (AR5 or AR6)

        Returns:
            GWP value as Decimal, or None if not found
        """
        gwp_set = gwp_set or self.default_gwp_set
        gas_upper = gas.upper()

        gwp_lookup = STANDARD_GWP_AR6_100YR if gwp_set == GWPSet.AR6 else STANDARD_GWP_AR5_100YR

        if gas_upper in gwp_lookup:
            gwp_value = gwp_lookup[gas_upper]
            if time_horizon == 20 and gwp_value.gwp_20yr:
                return gwp_value.gwp_20yr
            return gwp_value.gwp_100yr

        return None

    def list_factors(
        self,
        source: Optional[EmissionFactorSource] = None,
        category: Optional[EmissionCategory] = None,
        region: Optional[str] = None,
        limit: int = 100
    ) -> List[EmissionFactor]:
        """
        List emission factors with optional filtering.

        Args:
            source: Filter by source
            category: Filter by category
            region: Filter by region
            limit: Maximum number of results

        Returns:
            List of matching EmissionFactor objects
        """
        results = []

        for factor in self._factor_cache.values():
            if source and factor.source != source:
                continue
            if category and factor.category != category:
                continue
            if region and factor.region.lower() != region.lower():
                continue

            results.append(factor)

            if len(results) >= limit:
                break

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded emission factors.

        Returns:
            Dictionary with factor counts and coverage information
        """
        source_counts = {}
        region_counts = {}
        category_counts = {}

        for factor in self._factor_cache.values():
            source_name = factor.source.value if isinstance(factor.source, EmissionFactorSource) else str(factor.source)
            source_counts[source_name] = source_counts.get(source_name, 0) + 1

            region_counts[factor.region] = region_counts.get(factor.region, 0) + 1

            if factor.category:
                cat_name = factor.category.value if isinstance(factor.category, EmissionCategory) else str(factor.category)
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        return {
            "total_factors": len(self._factor_cache),
            "total_grid_factors": len(self._grid_factors),
            "factors_by_source": source_counts,
            "factors_by_region": region_counts,
            "factors_by_category": category_counts,
            "sources_loaded": list(self._source_data.keys()),
            "default_gwp_set": self.default_gwp_set.value
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._factor_cache.clear()
        self._grid_factors.clear()
        self._source_data.clear()
        logger.info("Emission factor cache cleared")

    def reload(self) -> None:
        """Reload all emission factors from disk."""
        self.clear_cache()
        self._load_all_sources()
        logger.info("Emission factors reloaded")


# Singleton instance for convenience
_service_instance: Optional[EmissionFactorService] = None


def get_emission_factor_service(
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> EmissionFactorService:
    """
    Get or create the EmissionFactorService singleton.

    Args:
        data_dir: Path to data directory (only used on first call)
        **kwargs: Additional arguments for service initialization

    Returns:
        EmissionFactorService instance
    """
    global _service_instance

    if _service_instance is None:
        _service_instance = EmissionFactorService(data_dir=data_dir, **kwargs)

    return _service_instance
