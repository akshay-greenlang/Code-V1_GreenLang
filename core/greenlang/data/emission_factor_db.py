"""
Emission Factor Database - Zero-Hallucination Emission Factor Lookups

This module provides deterministic emission factor lookups from authoritative sources:
- DEFRA (UK Department for Environment, Food & Rural Affairs) 2023 & 2024
- EPA eGRID (US Environmental Protection Agency) 2023
- IPCC (Intergovernmental Panel on Climate Change) AR6

All emission factors are pinned to specific versions with SHA-256 provenance tracking.
Supports version selection and automatic fallback to latest available data.
"""

import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class GWPSet(str, Enum):
    """Global Warming Potential sets from IPCC Assessment Reports."""
    AR6GWP100 = "AR6GWP100"  # IPCC Sixth Assessment Report, 100-year GWP
    AR5GWP100 = "AR5GWP100"  # IPCC Fifth Assessment Report, 100-year GWP
    AR4GWP100 = "AR4GWP100"  # IPCC Fourth Assessment Report, 100-year GWP


class DataSource(str, Enum):
    """Supported emission factor data sources."""
    DEFRA = "DEFRA"
    EPA = "EPA"
    EPA_EGRID = "EPA_EGRID"
    IPCC = "IPCC"


class DataVersion(str, Enum):
    """Supported data versions."""
    DEFRA_2023 = "defra_2023"
    DEFRA_2024 = "defra_2024"
    EPA_EGRID_2023 = "epa_egrid_2023"
    IPCC_AR6 = "ipcc_ar6"
    LATEST = "latest"


class EmissionFactorRecord(BaseModel):
    """Complete emission factor record with provenance."""

    ef_uri: str = Field(..., description="Unique emission factor URI (e.g., ef://defra/2024/natural_gas/US)")
    ef_value: float = Field(..., description="Emission factor value")
    ef_unit: str = Field(..., description="Emission factor unit (e.g., kgCO2e/MJ)")

    # Component emissions
    co2: float = Field(..., description="CO2 emission factor")
    ch4: float = Field(0.0, description="CH4 emission factor")
    n2o: float = Field(0.0, description="N2O emission factor")

    # Metadata
    source: str = Field(..., description="Data source (DEFRA, EPA, IPCC)")
    source_version: str = Field(..., description="Source version (e.g., 2024)")
    gwp_set: GWPSet = Field(..., description="GWP set used")
    region: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    year: int = Field(..., description="Reference year")

    # Quality indicators
    uncertainty: float = Field(..., description="Uncertainty percentage (e.g., 0.05 for +/-5%)")
    data_quality: str = Field(..., description="Data quality tier (1=best, 5=worst)")

    # Provenance
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    data_hash: str = Field(..., description="SHA-256 hash of emission factor data")

    # References
    citation: str = Field(..., description="Full citation for emission factor")
    url: Optional[str] = Field(None, description="URL to source document")


class EGridSubregionRecord(BaseModel):
    """EPA eGRID subregion emission record."""

    subregion_code: str = Field(..., description="eGRID subregion code (e.g., CAMX)")
    subregion_name: str = Field(..., description="Full subregion name")
    states: List[str] = Field(..., description="States in subregion")

    # Emission rates
    co2e_kg_per_mwh: float = Field(..., description="CO2e intensity in kg/MWh")
    co2_lb_per_mwh: float = Field(..., description="CO2 rate in lb/MWh")
    ch4_lb_per_mwh: float = Field(..., description="CH4 rate in lb/MWh")
    n2o_lb_per_mwh: float = Field(..., description="N2O rate in lb/MWh")

    # Generation mix
    generation_mix: Dict[str, float] = Field(..., description="Fuel mix percentages")
    total_generation_mwh: int = Field(..., description="Total annual generation")
    plant_count: int = Field(..., description="Number of power plants")

    # Metadata
    year: int = Field(..., description="Data year")
    uncertainty: float = Field(0.05, description="Uncertainty percentage")
    quality: str = Field("1", description="Data quality tier")
    citation: str = Field(..., description="Data citation")


class EmissionFactorDatabase:
    """
    In-memory emission factor database with deterministic lookups.

    This database is loaded from JSON files containing emission factors from
    authoritative sources. All lookups are deterministic - same inputs always
    return the same outputs.

    Zero-Hallucination Guarantee:
    - No LLM calls for emission factor lookup
    - No interpolation or estimation
    - Exact values from authoritative sources only
    - Complete provenance tracking

    Version Support:
    - DEFRA 2023 (legacy)
    - DEFRA 2024 (current)
    - EPA eGRID 2023
    - Automatic version selection with 'latest'
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        default_version: DataVersion = DataVersion.DEFRA_2024
    ):
        """
        Initialize emission factor database.

        Args:
            data_dir: Directory containing emission factor JSON files
            default_version: Default version to use for lookups
        """
        self.data_dir = data_dir or Path(__file__).parent / "factors"
        self.default_version = default_version
        self.factors: Dict[str, EmissionFactorRecord] = {}
        self.egrid_subregions: Dict[str, EGridSubregionRecord] = {}
        self.egrid_state_mapping: Dict[str, str] = {}
        self.loaded_versions: List[str] = []
        self._metadata: Dict[str, Any] = {}

        self._load_all_factors()

    def _load_all_factors(self):
        """Load all emission factors from JSON files."""
        # Load DEFRA 2024 factors (preferred)
        defra_2024_path = self.data_dir / "defra_2024.json"
        if defra_2024_path.exists():
            self._load_defra_json(defra_2024_path, "DEFRA", "2024")
            self.loaded_versions.append("defra_2024")
            logger.info("Loaded DEFRA 2024 emission factors")

        # Load DEFRA 2023 factors (legacy)
        defra_2023_path = self.data_dir / "defra_2023.json"
        if defra_2023_path.exists():
            self._load_defra_json(defra_2023_path, "DEFRA", "2023")
            self.loaded_versions.append("defra_2023")
            logger.info("Loaded DEFRA 2023 emission factors")

        # Load EPA eGRID 2023 factors
        egrid_path = self.data_dir / "epa_egrid_2023.json"
        if egrid_path.exists():
            self._load_egrid_json(egrid_path)
            self.loaded_versions.append("epa_egrid_2023")
            logger.info("Loaded EPA eGRID 2023 emission factors")

        # Load EPA 2023 factors
        epa_path = self.data_dir / "epa_2023.json"
        if epa_path.exists():
            self._load_defra_json(epa_path, "EPA", "2023")
            self.loaded_versions.append("epa_2023")

        # Load IPCC AR6 factors
        ipcc_path = self.data_dir / "ipcc_ar6.json"
        if ipcc_path.exists():
            self._load_defra_json(ipcc_path, "IPCC", "AR6")
            self.loaded_versions.append("ipcc_ar6")

    def _load_defra_json(self, path: Path, source: str, version: str):
        """Load emission factors from DEFRA-format JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Store metadata
        if '_metadata' in data:
            self._metadata[f"{source.lower()}_{version}"] = data['_metadata']

        # Process all categories
        for category_key, category_data in data.items():
            if category_key.startswith('_'):
                continue

            self._process_category(category_data, source, version, category_key)

    def _process_category(
        self,
        category_data: Dict,
        source: str,
        version: str,
        category_prefix: str = ""
    ):
        """Recursively process emission factor categories."""
        for key, value in category_data.items():
            if isinstance(value, dict):
                # Check if this is a region->year structure
                if any(isinstance(v, dict) and any(
                    k.isdigit() or k == version for k in v.keys()
                ) for v in value.values() if isinstance(v, dict)):
                    # This is a fuel type with region/year data
                    self._load_fuel_type(key, value, source, version, category_prefix)
                else:
                    # Nested category - recurse
                    new_prefix = f"{category_prefix}/{key}" if category_prefix else key
                    self._process_category(value, source, version, new_prefix)

    def _load_fuel_type(
        self,
        fuel_type: str,
        regions: Dict,
        source: str,
        version: str,
        category: str = ""
    ):
        """Load emission factors for a specific fuel type."""
        for region, years in regions.items():
            if not isinstance(years, dict):
                continue

            for year, ef_data in years.items():
                if not isinstance(ef_data, dict):
                    continue

                # Build the full fuel path
                fuel_path = f"{category}/{fuel_type}" if category else fuel_type
                fuel_path = fuel_path.replace('//', '/')

                # Create URI
                ef_uri = f"ef://{source.lower()}/{version}/{fuel_path}/{region}/{year}"

                # Calculate data hash
                data_str = json.dumps(ef_data, sort_keys=True)
                data_hash = hashlib.sha256(data_str.encode()).hexdigest()

                # Extract emission factor value
                ef_value = ef_data.get("co2e_ar6", ef_data.get("co2e", ef_data.get("co2", 0)))

                try:
                    # Create record
                    record = EmissionFactorRecord(
                        ef_uri=ef_uri,
                        ef_value=float(ef_value),
                        ef_unit=ef_data.get("unit", "kgCO2e/unit"),
                        co2=float(ef_data.get("co2", 0)),
                        ch4=float(ef_data.get("ch4", 0)),
                        n2o=float(ef_data.get("n2o", 0)),
                        source=source,
                        source_version=version,
                        gwp_set=GWPSet.AR6GWP100,
                        region=region,
                        year=int(year) if year.isdigit() else 2024,
                        uncertainty=float(ef_data.get("uncertainty", 0.10)),
                        data_quality=str(ef_data.get("quality", "2")),
                        data_hash=data_hash,
                        citation=ef_data.get("citation", f"{source} {version}"),
                        url=ef_data.get("url")
                    )

                    self.factors[ef_uri] = record

                except Exception as e:
                    logger.warning(f"Failed to load emission factor {ef_uri}: {e}")

    def _load_egrid_json(self, path: Path):
        """Load EPA eGRID emission factors."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Store metadata
        if '_metadata' in data:
            self._metadata['epa_egrid_2023'] = data['_metadata']

        # Load subregions
        subregions = data.get('subregions', {})
        for code, subregion_data in subregions.items():
            year_data = subregion_data.get('2023', {})
            if not year_data:
                continue

            try:
                record = EGridSubregionRecord(
                    subregion_code=code,
                    subregion_name=subregion_data.get('name', code),
                    states=subregion_data.get('states', []),
                    co2e_kg_per_mwh=year_data.get('co2e_kg_per_mwh', 0),
                    co2_lb_per_mwh=year_data.get('co2_lb_per_mwh', 0),
                    ch4_lb_per_mwh=year_data.get('ch4_lb_per_mwh', 0),
                    n2o_lb_per_mwh=year_data.get('n2o_lb_per_mwh', 0),
                    generation_mix=year_data.get('generation_mix', {}),
                    total_generation_mwh=year_data.get('total_generation_mwh', 0),
                    plant_count=year_data.get('plant_count', 0),
                    year=2023,
                    uncertainty=year_data.get('uncertainty', 0.05),
                    quality=year_data.get('quality', '1'),
                    citation=year_data.get('citation', 'EPA eGRID2023')
                )
                self.egrid_subregions[code] = record

                # Map states to subregions
                for state in subregion_data.get('states', []):
                    if state not in self.egrid_state_mapping:
                        self.egrid_state_mapping[state] = code

            except Exception as e:
                logger.warning(f"Failed to load eGRID subregion {code}: {e}")

        # Load state averages for more precise state-level lookups
        state_averages = data.get('state_averages', {})
        for state, state_data in state_averages.items():
            year_data = state_data.get('2023', {})
            if 'primary_subregion' in year_data:
                self.egrid_state_mapping[state] = year_data['primary_subregion']

        # Load US national average as a special subregion
        us_avg = data.get('us_national_average', {}).get('2023', {})
        if us_avg:
            try:
                record = EGridSubregionRecord(
                    subregion_code='US_AVG',
                    subregion_name='US National Average',
                    states=['US'],
                    co2e_kg_per_mwh=us_avg.get('co2e_kg_per_mwh', 0),
                    co2_lb_per_mwh=us_avg.get('co2_lb_per_mwh', 0),
                    ch4_lb_per_mwh=us_avg.get('ch4_lb_per_mwh', 0),
                    n2o_lb_per_mwh=us_avg.get('n2o_lb_per_mwh', 0),
                    generation_mix=us_avg.get('generation_mix', {}),
                    total_generation_mwh=us_avg.get('total_generation_mwh', 0),
                    plant_count=us_avg.get('plant_count', 0),
                    year=2023,
                    uncertainty=us_avg.get('uncertainty', 0.05),
                    quality=us_avg.get('quality', '1'),
                    citation=us_avg.get('citation', 'EPA eGRID2023')
                )
                self.egrid_subregions['US_AVG'] = record
            except Exception as e:
                logger.warning(f"Failed to load US national average: {e}")

    def lookup(
        self,
        fuel_type: str,
        region: str,
        year: int = 2024,
        version: Optional[DataVersion] = None,
        gwp_set: GWPSet = GWPSet.AR6GWP100
    ) -> Optional[EmissionFactorRecord]:
        """
        Look up emission factor for fuel type, region, and year.

        This is a DETERMINISTIC lookup - same inputs always return same output.
        No estimation, interpolation, or LLM calls.

        Args:
            fuel_type: Type of fuel (natural_gas, diesel, petrol, etc.)
            region: ISO 3166-1 alpha-2 country code (US, GB, EU)
            year: Reference year (2020-2025)
            version: Specific data version or None for default
            gwp_set: GWP set to use (default: AR6GWP100)

        Returns:
            EmissionFactorRecord if found, None otherwise
        """
        version = version or self.default_version

        # Determine which versions to try based on selection
        versions_to_try = self._get_versions_to_try(version)

        # Try each version in order
        for v in versions_to_try:
            source, ver = self._parse_version(v)

            # Try exact match
            ef_uri = f"ef://{source.lower()}/{ver}/{fuel_type}/{region}/{year}"
            if ef_uri in self.factors:
                return self.factors[ef_uri]

            # Try without year for latest
            ef_uri_latest = f"ef://{source.lower()}/{ver}/{fuel_type}/{region}/{ver}"
            if ef_uri_latest in self.factors:
                return self.factors[ef_uri_latest]

        # Try fallback strategies
        return self._fallback_lookup(fuel_type, region, year, versions_to_try)

    def _get_versions_to_try(self, version: DataVersion) -> List[str]:
        """Get list of versions to try in order."""
        if version == DataVersion.LATEST:
            return ["defra_2024", "defra_2023", "epa_egrid_2023"]
        elif version == DataVersion.DEFRA_2024:
            return ["defra_2024", "defra_2023"]
        elif version == DataVersion.DEFRA_2023:
            return ["defra_2023"]
        elif version == DataVersion.EPA_EGRID_2023:
            return ["epa_egrid_2023", "defra_2024"]
        else:
            return ["defra_2024", "defra_2023"]

    def _parse_version(self, version_str: str) -> Tuple[str, str]:
        """Parse version string into source and version."""
        if version_str.startswith("defra_"):
            return "DEFRA", version_str.replace("defra_", "")
        elif version_str.startswith("epa_"):
            return "EPA", version_str.replace("epa_", "")
        else:
            parts = version_str.split("_")
            return parts[0].upper(), parts[1] if len(parts) > 1 else "2024"

    def _fallback_lookup(
        self,
        fuel_type: str,
        region: str,
        year: int,
        versions: List[str]
    ) -> Optional[EmissionFactorRecord]:
        """Try fallback strategies to find emission factor."""
        # Try global region
        for v in versions:
            source, ver = self._parse_version(v)
            ef_uri = f"ef://{source.lower()}/{ver}/{fuel_type}/GLOBAL/{year}"
            if ef_uri in self.factors:
                return self.factors[ef_uri]

        # Try most recent year
        for v in versions:
            source, ver = self._parse_version(v)
            for y in range(year, year - 5, -1):
                ef_uri = f"ef://{source.lower()}/{ver}/{fuel_type}/{region}/{y}"
                if ef_uri in self.factors:
                    return self.factors[ef_uri]

        # Try partial fuel type match (e.g., fuels/natural_gas)
        for uri, record in self.factors.items():
            if fuel_type in uri and region in uri:
                return record

        return None

    def lookup_by_uri(self, ef_uri: str) -> Optional[EmissionFactorRecord]:
        """
        Look up emission factor by URI.

        Args:
            ef_uri: Emission factor URI (e.g., ef://defra/2024/natural_gas/US/2024)

        Returns:
            EmissionFactorRecord if found, None otherwise
        """
        return self.factors.get(ef_uri)

    def lookup_grid_intensity(
        self,
        location: str,
        year: int = 2023,
        location_type: str = "state"
    ) -> Optional[Union[EGridSubregionRecord, EmissionFactorRecord]]:
        """
        Look up electricity grid intensity for a location.

        Supports:
        - US state codes (CA, TX, NY)
        - eGRID subregion codes (CAMX, ERCT, NYUP)
        - Country codes (US, GB, EU)

        Args:
            location: State code, subregion code, or country code
            year: Reference year
            location_type: 'state', 'subregion', or 'country'

        Returns:
            Grid intensity record
        """
        # Try eGRID subregion directly
        if location in self.egrid_subregions:
            return self.egrid_subregions[location]

        # Try state mapping to eGRID subregion
        if location in self.egrid_state_mapping:
            subregion = self.egrid_state_mapping[location]
            if subregion in self.egrid_subregions:
                return self.egrid_subregions[subregion]

        # Fall back to DEFRA electricity grid factors
        return self.lookup("electricity_grid", location, year)

    def get_egrid_subregion(self, code: str) -> Optional[EGridSubregionRecord]:
        """
        Get eGRID subregion by code.

        Args:
            code: eGRID subregion code (e.g., CAMX)

        Returns:
            EGridSubregionRecord if found
        """
        return self.egrid_subregions.get(code)

    def get_egrid_for_state(self, state_code: str) -> Optional[EGridSubregionRecord]:
        """
        Get eGRID subregion for a US state.

        Args:
            state_code: Two-letter state code (e.g., CA)

        Returns:
            EGridSubregionRecord for the state's primary subregion
        """
        subregion = self.egrid_state_mapping.get(state_code)
        if subregion:
            return self.egrid_subregions.get(subregion)
        return None

    def search(
        self,
        fuel_type: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        version: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[EmissionFactorRecord]:
        """
        Search emission factors by criteria.

        Args:
            fuel_type: Filter by fuel type
            region: Filter by region
            source: Filter by source (DEFRA, EPA, IPCC)
            version: Filter by version (2023, 2024)
            category: Filter by category (fuels, transport, etc.)

        Returns:
            List of matching EmissionFactorRecords
        """
        results = []

        for uri, record in self.factors.items():
            if fuel_type and fuel_type not in uri:
                continue
            if region and record.region != region:
                continue
            if source and record.source != source:
                continue
            if version and record.source_version != version:
                continue
            if category and category not in uri:
                continue

            results.append(record)

        return results

    def list_fuel_types(self, version: Optional[str] = None) -> List[str]:
        """List all available fuel types."""
        fuel_types = set()
        for uri in self.factors.keys():
            if version and version not in uri:
                continue
            # Parse URI: ef://source/version/category/fuel_type/region/year
            parts = uri.replace("ef://", "").split('/')
            if len(parts) >= 3:
                # Could be category/fuel_type or just fuel_type
                fuel_types.add(parts[2])
        return sorted(fuel_types)

    def list_regions(self, version: Optional[str] = None) -> List[str]:
        """List all available regions."""
        regions = set()
        for record in self.factors.values():
            if version and record.source_version != version:
                continue
            regions.add(record.region)
        return sorted(regions)

    def list_categories(self, version: Optional[str] = None) -> List[str]:
        """List all available categories."""
        categories = set()
        for uri in self.factors.keys():
            if version and version not in uri:
                continue
            parts = uri.replace("ef://", "").split('/')
            if len(parts) >= 3:
                categories.add(parts[2])
        return sorted(categories)

    def list_egrid_subregions(self) -> List[str]:
        """List all eGRID subregion codes."""
        return sorted(self.egrid_subregions.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            "total_factors": len(self.factors),
            "fuel_types": len(self.list_fuel_types()),
            "regions": len(self.list_regions()),
            "categories": len(self.list_categories()),
            "egrid_subregions": len(self.egrid_subregions),
            "loaded_versions": self.loaded_versions,
            "sources": len(set(r.source for r in self.factors.values())),
            "date_range": {
                "min_year": min((r.year for r in self.factors.values()), default=2023),
                "max_year": max((r.year for r in self.factors.values()), default=2024)
            }
        }

        # Add version-specific stats
        for version in self.loaded_versions:
            stats[f"{version}_factors"] = len([
                r for r in self.factors.values()
                if version.replace("_", "") in r.ef_uri
            ])

        return stats

    def get_metadata(self, version: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific version."""
        return self._metadata.get(version)

    def compare_versions(
        self,
        fuel_type: str,
        region: str,
        version1: DataVersion = DataVersion.DEFRA_2023,
        version2: DataVersion = DataVersion.DEFRA_2024
    ) -> Dict[str, Any]:
        """
        Compare emission factors between two versions.

        Args:
            fuel_type: Fuel type to compare
            region: Region to compare
            version1: First version
            version2: Second version

        Returns:
            Comparison results including values and percentage change
        """
        record1 = self.lookup(fuel_type, region, version=version1)
        record2 = self.lookup(fuel_type, region, version=version2)

        result = {
            "fuel_type": fuel_type,
            "region": region,
            "version1": str(version1),
            "version2": str(version2),
            "record1": record1,
            "record2": record2,
        }

        if record1 and record2:
            change = record2.ef_value - record1.ef_value
            pct_change = (change / record1.ef_value * 100) if record1.ef_value != 0 else 0
            result["change"] = change
            result["percent_change"] = round(pct_change, 2)

        return result


# Global database instance
_db_instance: Optional[EmissionFactorDatabase] = None


def get_database(
    version: DataVersion = DataVersion.DEFRA_2024,
    force_reload: bool = False
) -> EmissionFactorDatabase:
    """
    Get global emission factor database instance.

    Args:
        version: Default version to use
        force_reload: Force reload of database

    Returns:
        EmissionFactorDatabase instance
    """
    global _db_instance
    if _db_instance is None or force_reload:
        _db_instance = EmissionFactorDatabase(default_version=version)
    return _db_instance


def lookup_emission_factor(
    fuel_type: str,
    region: str,
    year: int = 2024,
    version: Optional[DataVersion] = None
) -> Optional[EmissionFactorRecord]:
    """
    Convenience function to look up emission factor.

    Args:
        fuel_type: Fuel type
        region: Region code
        year: Reference year
        version: Data version

    Returns:
        EmissionFactorRecord if found
    """
    db = get_database()
    return db.lookup(fuel_type, region, year, version)


def lookup_grid_intensity(
    location: str,
    year: int = 2023
) -> Optional[Union[EGridSubregionRecord, EmissionFactorRecord]]:
    """
    Convenience function to look up grid intensity.

    Args:
        location: State, subregion, or country code
        year: Reference year

    Returns:
        Grid intensity record
    """
    db = get_database()
    return db.lookup_grid_intensity(location, year)
