# -*- coding: utf-8 -*-
"""
greenlang/data/emission_factor_database.py

EmissionFactorDatabase v2 - Multi-gas emission factor management with provenance

This module provides:
- Multi-gas emission factors (CO2, CH4, N2O breakdown)
- Full provenance tracking (source, version, dates)
- Data quality scoring
- Backward compatibility with v1 scalar factors
- Migration from v1 to v2 format

Example:
    >>> from greenlang.data.emission_factor_database import EmissionFactorDatabase
    >>>
    >>> # Initialize with v2 factors
    >>> db = EmissionFactorDatabase()
    >>>
    >>> # Get v2 factor record (multi-gas)
    >>> factor = db.get_factor_record("diesel", "gallons", "US")
    >>> print(factor.vectors.CO2)  # 10.18 kg CO2/gallon
    >>> print(factor.vectors.CH4)  # 0.00082 kg CH4/gallon
    >>> print(factor.gwp_100yr.co2e_total)  # 10.21 kg CO2e/gallon
    >>>
    >>> # Backward compatible v1 API (returns scalar)
    >>> co2e = db.get_factor("diesel", "gallons", "US")
    >>> print(co2e)  # 10.21 (same as v1)
"""

from typing import Optional, Dict, List, Tuple
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import logging

from .emission_factor_record import (
    EmissionFactorRecord,
    GHGVectors,
    GWPValues,
    DataQualityScore,
    SourceProvenance,
    LicenseInfo,
    GeographyLevel,
    Scope,
    Boundary,
    Methodology,
    GWPSet,
    HeatingValueBasis,
)
from greenlang.cache import get_global_cache, EmissionFactorCache
from . import wtt_emission_factors

logger = logging.getLogger(__name__)


class EmissionFactorDatabase:
    """
    v2 Emission factor database with multi-gas support and provenance tracking.

    Features:
    - Multi-gas breakdown (CO2, CH4, N2O)
    - Full provenance (source, version, dates)
    - Data quality scoring (DQS)
    - Query by scope, boundary, GWP set
    - Backward compatible with v1 API
    - In-memory + file-based storage
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
    ):
        """
        Initialize emission factor database.

        Args:
            data_dir: Directory containing factor JSON files. If None, use built-in defaults.
            enable_cache: Enable caching for factor lookups (default: True)
            cache_size: Maximum cache size (default: 1000 entries)
            cache_ttl: Cache TTL in seconds (default: 3600 = 1 hour)
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.factors: Dict[str, EmissionFactorRecord] = {}
        self.enable_cache = enable_cache

        # Initialize cache
        if enable_cache:
            self.cache = get_global_cache(
                max_size=cache_size,
                ttl_seconds=cache_ttl,
                enable_stats=True,
            )
        else:
            self.cache = None

        # Load built-in v2 factors
        self._load_default_factors_v2()

        # Load custom factors from directory (if provided)
        if self.data_dir and self.data_dir.exists():
            self._load_custom_factors()

        # Warm cache with common factors
        if self.enable_cache:
            self._warm_cache()

    # ==================== V2 API (ENHANCED) ====================

    # Valid scope/boundary combinations per GHG Protocol
    # Scope 1: Direct emissions from owned/controlled sources
    # Scope 2: Indirect emissions from purchased energy
    # Scope 3: All other indirect emissions in value chain (15 categories)
    VALID_SCOPE_BOUNDARIES: Dict[str, List[str]] = {
        "1": ["combustion", "process", "fugitive", "mobile"],  # Scope 1: Direct emissions
        "2": ["location-based", "market-based", "combustion"],  # Scope 2: Purchased energy
        "3": [
            # Upstream categories (1-8)
            "wtt", "wtw", "upstream", "downstream",
            "cradle_to_gate", "cradle_to_grave",
            "purchased_goods_services",  # Category 1
            "capital_goods",              # Category 2
            "fuel_energy_activities",     # Category 3
            "upstream_transport",         # Category 4
            "waste_operations",           # Category 5
            "business_travel",            # Category 6
            "employee_commuting",         # Category 7
            "upstream_leased_assets",     # Category 8
            # Downstream categories (9-15)
            "downstream_transport",       # Category 9
            "processing_sold_products",   # Category 10
            "use_sold_products",          # Category 11
            "eol_sold_products",          # Category 12
            "downstream_leased_assets",   # Category 13
            "franchises",                 # Category 14
            "investments",                # Category 15
        ],
    }

    # Scope 3 category descriptions for validation errors
    SCOPE3_CATEGORIES: Dict[str, str] = {
        "purchased_goods_services": "Category 1: Purchased Goods and Services",
        "capital_goods": "Category 2: Capital Goods",
        "fuel_energy_activities": "Category 3: Fuel and Energy Related Activities",
        "upstream_transport": "Category 4: Upstream Transportation and Distribution",
        "waste_operations": "Category 5: Waste Generated in Operations",
        "business_travel": "Category 6: Business Travel",
        "employee_commuting": "Category 7: Employee Commuting",
        "upstream_leased_assets": "Category 8: Upstream Leased Assets",
        "downstream_transport": "Category 9: Downstream Transportation and Distribution",
        "processing_sold_products": "Category 10: Processing of Sold Products",
        "use_sold_products": "Category 11: Use of Sold Products",
        "eol_sold_products": "Category 12: End-of-Life Treatment of Sold Products",
        "downstream_leased_assets": "Category 13: Downstream Leased Assets",
        "franchises": "Category 14: Franchises",
        "investments": "Category 15: Investments",
        "wtt": "Well-to-Tank (upstream fuel production)",
        "wtw": "Well-to-Wheel (full lifecycle)",
        "upstream": "General upstream emissions",
        "downstream": "General downstream emissions",
        "cradle_to_gate": "Cradle-to-Gate lifecycle",
        "cradle_to_grave": "Cradle-to-Grave lifecycle",
    }

    def _validate_scope_boundary(self, scope: str, boundary: str) -> None:
        """
        Validate that scope and boundary combination is valid per GHG Protocol.

        GHG Protocol Scope Definitions:
        - Scope 1: Direct GHG emissions from sources owned or controlled by the company
          (e.g., on-site combustion, company vehicles, process emissions)
        - Scope 2: Indirect GHG emissions from purchased electricity, steam, heating, cooling
          (location-based or market-based accounting)
        - Scope 3: All other indirect emissions in the value chain (15 categories)
          (both upstream and downstream)

        Args:
            scope: GHG scope ("1", "2", or "3")
            boundary: Emission boundary

        Raises:
            ValueError: If invalid scope/boundary combination with detailed error message
        """
        # Normalize boundary for comparison (case-insensitive, underscores for spaces/hyphens)
        boundary_normalized = boundary.lower().strip().replace(" ", "_").replace("-", "_")

        # Check if scope is valid
        if scope not in self.VALID_SCOPE_BOUNDARIES:
            raise ValueError(
                f"Invalid scope '{scope}'. Must be one of: 1 (Direct), 2 (Energy), 3 (Value Chain)"
            )

        # Get valid boundaries for this scope
        valid_boundaries = self.VALID_SCOPE_BOUNDARIES[scope]

        # Check if boundary is valid for this scope
        if boundary_normalized not in valid_boundaries:
            # Provide helpful error message with scope description
            scope_descriptions = {
                "1": "Direct emissions (on-site combustion, process, fugitive, mobile sources)",
                "2": "Indirect from purchased energy (location-based, market-based)",
                "3": "Value chain (15 categories: purchased goods, transport, travel, etc.)",
            }

            # For Scope 3, provide category-specific help
            if scope == "3":
                categories_help = "\n".join(
                    f"  - {k}: {v}" for k, v in self.SCOPE3_CATEGORIES.items()
                )
                raise ValueError(
                    f"Invalid Scope 3 boundary: '{boundary}'.\n"
                    f"Scope 3 covers value chain emissions with these valid boundaries:\n"
                    f"{categories_help}"
                )

            raise ValueError(
                f"Invalid scope/boundary combination: Scope {scope} + '{boundary}'.\n"
                f"Scope {scope}: {scope_descriptions[scope]}\n"
                f"Valid boundaries for Scope {scope}: {valid_boundaries}"
            )

    def get_factor_record(
        self,
        fuel_type: str,
        unit: str,
        geography: str = "US",
        scope: str = "1",
        boundary: str = "combustion",
        gwp_set: str = "IPCC_AR6_100",
        as_of_date: Optional[date] = None,
    ) -> Optional[EmissionFactorRecord]:
        """
        Get full emission factor record (v2 API with caching).

        Args:
            fuel_type: Fuel type (diesel, natural_gas, electricity, etc.)
            unit: Unit (gallons, kWh, therms, etc.) - case insensitive
            geography: ISO country code or region (US, EU, UK, etc.)
            scope: GHG scope ("1", "2", or "3")
            boundary: Emission boundary (combustion, WTT, WTW)
            gwp_set: GWP reference set (IPCC_AR6_100, IPCC_AR6_20, etc.)
            as_of_date: Historical query date (for reproducibility)

        Returns:
            EmissionFactorRecord with multi-gas vectors and provenance,
            or None if not found

        Raises:
            ValueError: If invalid scope/boundary combination
        """
        # Validate scope/boundary combination
        self._validate_scope_boundary(scope, boundary)

        # Normalize unit to lowercase for consistent lookups (fixes kWh vs kwh mismatch)
        unit = unit.lower()

        # Check cache first (if enabled and not historical query)
        if self.enable_cache and self.cache and not as_of_date:
            cached_factor = self.cache.get(
                fuel_type, unit, geography, scope, boundary, gwp_set
            )
            if cached_factor:
                return cached_factor

        # Handle WTT and WTW boundaries (computed on-the-fly)
        if boundary in ["WTT", "WTW"]:
            factor = self._get_wtt_or_wtw_factor(
                fuel_type, unit, geography, scope, boundary, gwp_set
            )

            # Cache the computed factor
            if factor and self.enable_cache and self.cache and not as_of_date:
                self.cache.put(fuel_type, unit, factor, geography, scope, boundary, gwp_set)

            return factor

        # Build lookup key for combustion boundary
        key = self._make_key(fuel_type, unit, geography, scope, boundary, gwp_set)

        # Check if exists
        if key not in self.factors:
            # Try fallback: relax constraints
            key = self._find_best_match(fuel_type, unit, geography, scope, boundary, gwp_set)

        if key and key in self.factors:
            factor = self.factors[key]

            # Check validity period (for historical queries)
            if as_of_date:
                if not factor.is_valid_on(as_of_date):
                    logger.warning(
                        f"Factor {key} not valid on {as_of_date}. "
                        f"Valid: {factor.valid_from} to {factor.valid_to or 'current'}"
                    )
                    return None

            # Cache the factor (if caching enabled and not historical)
            if self.enable_cache and self.cache and not as_of_date:
                self.cache.put(fuel_type, unit, factor, geography, scope, boundary, gwp_set)

            return factor

        return None

    def add_factor_record(self, factor: EmissionFactorRecord):
        """
        Add emission factor record to database.

        Args:
            factor: EmissionFactorRecord to add
        """
        key = self._make_key(
            factor.fuel_type,
            factor.unit,
            factor.geography,
            factor.scope.value,
            factor.boundary.value,
            factor.gwp_100yr.gwp_set.value,
        )

        self.factors[key] = factor
        logger.info(f"Added factor: {key} ({factor.gwp_100yr.co2e_total:.4f} kgCO2e/unit)")

    def list_factors(
        self,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
    ) -> List[EmissionFactorRecord]:
        """
        List all factors matching criteria.

        Args:
            fuel_type: Filter by fuel type (optional) - case insensitive
            geography: Filter by geography (optional)

        Returns:
            List of matching EmissionFactorRecord objects
        """
        results = []

        # Normalize fuel_type for case-insensitive comparison
        fuel_type_lower = fuel_type.lower() if fuel_type else None

        for factor in self.factors.values():
            # Apply filters with case-insensitive fuel_type comparison
            if fuel_type_lower and factor.fuel_type.lower() != fuel_type_lower:
                continue
            if geography and factor.geography != geography:
                continue

            results.append(factor)

        return results

    # ==================== V1 API (BACKWARD COMPATIBLE) ====================

    def get_factor(
        self,
        fuel_type: str,
        unit: str,
        region: str = "US",
    ) -> Optional[float]:
        """
        Get emission factor as scalar (v1 backward compatible API).

        Args:
            fuel_type: Fuel type
            unit: Unit
            region: Region/geography

        Returns:
            CO2e emission factor (kg CO2e / unit) as float,
            or None if not found
        """
        # Try v2 lookup with default params
        factor_record = self.get_factor_record(
            fuel_type=fuel_type,
            unit=unit,
            geography=region,
            scope="1",  # Default to Scope 1
            boundary="combustion",  # Default to combustion
            gwp_set="IPCC_AR6_100",  # Default to 100-year GWP
        )

        if factor_record:
            return factor_record.gwp_100yr.co2e_total

        return None

    def get_available_regions(self) -> List[str]:
        """Get list of available regions (v1 compatible)."""
        regions = set()
        for factor in self.factors.values():
            regions.add(factor.geography)
        return sorted(list(regions))

    def get_available_fuels(self, region: str = "US") -> List[str]:
        """Get list of available fuel types for region (v1 compatible)."""
        fuels = set()
        for factor in self.factors.values():
            if factor.geography == region:
                fuels.add(factor.fuel_type)
        return sorted(list(fuels))

    def get_available_units(self, fuel_type: str, region: str = "US") -> List[str]:
        """Get list of available units for fuel type (v1 compatible)."""
        units = set()
        # Normalize fuel_type for case-insensitive comparison
        fuel_type_lower = fuel_type.lower()
        for factor in self.factors.values():
            if factor.fuel_type.lower() == fuel_type_lower and factor.geography == region:
                units.add(factor.unit)
        return sorted(list(units))

    # ==================== INTERNAL METHODS ====================

    def _get_wtt_or_wtw_factor(
        self,
        fuel_type: str,
        unit: str,
        geography: str,
        scope: str,
        boundary: str,
        gwp_set: str,
    ) -> Optional[EmissionFactorRecord]:
        """
        Compute WTT or WTW factor by combining combustion and upstream emissions.

        WTT (Well-to-Tank) = upstream only (extraction, refining, transport)
        WTW (Well-to-Wheel) = WTT + combustion

        Args:
            fuel_type: Fuel type
            unit: Unit
            geography: Geography
            scope: Scope
            boundary: "WTT" or "WTW"
            gwp_set: GWP set

        Returns:
            EmissionFactorRecord with WTT or WTW vectors, or None if data unavailable
        """
        # Get combustion factor (for WTW)
        combustion_factor = None
        if boundary == "WTW":
            combustion_factor = self.get_factor_record(
                fuel_type=fuel_type,
                unit=unit,
                geography=geography,
                scope=scope,
                boundary="combustion",
                gwp_set=gwp_set,
            )

            if not combustion_factor:
                logger.warning(
                    f"No combustion factor found for {fuel_type} ({unit}) in {geography}. "
                    "Cannot compute WTW."
                )
                return None

        # Get WTT factor from wtt_emission_factors module
        wtt_co2e, wtt_source = wtt_emission_factors.get_wtt_factor(
            fuel_type, unit, geography
        )

        if wtt_co2e == 0.0:
            # No WTT data available, try estimation
            if combustion_factor:
                wtt_co2e = wtt_emission_factors.estimate_wtt_factor(
                    fuel_type, combustion_factor.gwp_100yr.co2e_total
                )
                wtt_source = f"Estimated using typical ratio ({wtt_emission_factors.TYPICAL_WTT_RATIOS.get(fuel_type, 0.15)*100:.0f}%)"
            else:
                logger.warning(
                    f"No WTT data available for {fuel_type} ({unit}) in {geography}"
                )
                return None

        # Decompose WTT CO2e into gas vectors based on fuel type
        # Different fuels have different upstream emission profiles:
        # - Natural gas: ~20% CH4 (methane leakage from extraction/transport)
        # - Coal: ~10% CH4 (mining emissions)
        # - Biofuels: ~5% N2O (agricultural emissions from feedstock)
        # - Petroleum (diesel, gasoline): ~95% CO2, ~5% CH4 (refining/transport)
        wtt_vectors = self._decompose_wtt_emissions(
            fuel_type=fuel_type,
            wtt_co2e=wtt_co2e,
            gwp_set=gwp_set,
        )

        # Compute final vectors based on boundary
        if boundary == "WTT":
            # WTT only
            final_vectors = wtt_vectors
            boundary_enum = Boundary.WTT
            factor_id_suffix = "WTT"
            description = f"Well-to-Tank (upstream) emissions for {fuel_type}"
        else:  # WTW
            # WTW = combustion + WTT
            final_vectors = GHGVectors(
                CO2=combustion_factor.vectors.CO2 + wtt_vectors.CO2,
                CH4=combustion_factor.vectors.CH4 + wtt_vectors.CH4,
                N2O=combustion_factor.vectors.N2O + wtt_vectors.N2O,
            )
            boundary_enum = Boundary.WTW
            factor_id_suffix = "WTW"
            description = f"Well-to-Wheel (full lifecycle) emissions for {fuel_type}"

        # Build factor record
        base_factor = combustion_factor if combustion_factor else self.get_factor_record(
            fuel_type, unit, geography, scope, "combustion", gwp_set
        )

        if not base_factor:
            return None

        # Create WTT/WTW factor record
        wtt_wtw_factor = EmissionFactorRecord(
            factor_id=f"EF:{geography}:{fuel_type}:{unit}:{date.today().year}:v1:{factor_id_suffix}",
            fuel_type=fuel_type,
            unit=unit,
            geography=geography,
            geography_level=base_factor.geography_level,
            vectors=final_vectors,
            gwp_100yr=base_factor.gwp_100yr,  # Recompute in post_init
            gwp_20yr=base_factor.gwp_20yr,
            scope=base_factor.scope,
            boundary=boundary_enum,
            provenance=SourceProvenance(
                source_org="GreenLang",
                source_publication=f"Computed {boundary} factor from combustion + {wtt_source}",
                source_year=date.today().year,
                methodology=Methodology.IPCC_TIER_1,
                version="v1",
                citation=f"Combustion: {base_factor.provenance.source_org}; WTT: {wtt_source}",
            ),
            valid_from=base_factor.valid_from,
            valid_to=base_factor.valid_to,
            uncertainty_95ci=base_factor.uncertainty_95ci * 1.2,  # Higher uncertainty for computed factors
            dqs=DataQualityScore(
                temporal=base_factor.dqs.temporal,
                geographical=base_factor.dqs.geographical - 1,  # Lower quality (computed)
                technological=base_factor.dqs.technological - 1,
                representativeness=base_factor.dqs.representativeness - 1,
                methodological=base_factor.dqs.methodological - 1,
            ),
            license_info=base_factor.license_info,
            heating_value_basis=base_factor.heating_value_basis,
            compliance_frameworks=base_factor.compliance_frameworks,
            tags=base_factor.tags + [boundary.lower()],
            notes=description,
        )

        return wtt_wtw_factor

    def _decompose_wtt_emissions(
        self,
        fuel_type: str,
        wtt_co2e: float,
        gwp_set: str,
    ) -> GHGVectors:
        """
        Decompose WTT (Well-to-Tank) CO2e into individual gas components.

        Different fuel types have different upstream emission profiles:
        - Natural gas: ~20% of WTT CO2e from CH4 (methane leakage in extraction/transport)
        - Coal: ~10% of WTT CO2e from CH4 (mining emissions)
        - Biofuels: ~5% of WTT CO2e from N2O (agricultural emissions from feedstock)
        - Petroleum (diesel, gasoline, fuel_oil): ~95% CO2, ~5% CH4 (refining/transport)
        - Electricity: ~100% CO2 (grid losses, no direct CH4/N2O in upstream)

        The decomposition uses GWP values to back-calculate individual gas amounts
        from the total CO2e value. This is the inverse of the CO2e calculation:
            CO2e = CO2 + (CH4 * CH4_GWP) + (N2O * N2O_GWP)

        Args:
            fuel_type: Fuel type (natural_gas, coal, diesel, etc.)
            wtt_co2e: Total WTT emissions in kg CO2e
            gwp_set: GWP reference set (determines CH4 and N2O GWP values)

        Returns:
            GHGVectors with decomposed CO2, CH4, and N2O values in kg
        """
        # Determine GWP values based on gwp_set
        # AR6 100-year vs 20-year GWP values
        if "20" in gwp_set:
            ch4_gwp = 84   # AR6 20-year GWP for CH4
            n2o_gwp = 273  # AR6 GWP for N2O (same for 20yr and 100yr)
        else:
            ch4_gwp = 28   # AR6 100-year GWP for CH4
            n2o_gwp = 273  # AR6 GWP for N2O

        # Define WTT emission profiles by fuel category
        # Format: (CO2_fraction, CH4_fraction, N2O_fraction) of total CO2e
        # Note: These fractions represent the CO2e contribution from each gas,
        # not the mass fraction. We back-calculate mass using GWP.

        # WTT emission profiles based on literature and regulatory guidance
        wtt_profiles = {
            # Natural gas: Significant CH4 from methane leakage (2-3% leakage rate typical)
            # Sources: EPA, IPCC, academic studies on methane emissions
            "natural_gas": {"CO2_frac": 0.80, "CH4_frac": 0.20, "N2O_frac": 0.00},

            # Coal: CH4 from mining operations (coal seam methane)
            # Sources: EPA coal mine methane data, IPCC Guidelines
            "coal": {"CO2_frac": 0.90, "CH4_frac": 0.10, "N2O_frac": 0.00},

            # Petroleum products: Small CH4 from refining/transport, mostly CO2
            # Sources: EPA refinery data, IPCC petroleum guidelines
            "diesel": {"CO2_frac": 0.95, "CH4_frac": 0.05, "N2O_frac": 0.00},
            "gasoline": {"CO2_frac": 0.95, "CH4_frac": 0.05, "N2O_frac": 0.00},
            "fuel_oil": {"CO2_frac": 0.95, "CH4_frac": 0.05, "N2O_frac": 0.00},
            "kerosene": {"CO2_frac": 0.95, "CH4_frac": 0.05, "N2O_frac": 0.00},
            "jet_fuel": {"CO2_frac": 0.95, "CH4_frac": 0.05, "N2O_frac": 0.00},
            "lpg": {"CO2_frac": 0.92, "CH4_frac": 0.08, "N2O_frac": 0.00},
            "propane": {"CO2_frac": 0.92, "CH4_frac": 0.08, "N2O_frac": 0.00},

            # Biofuels: N2O from agricultural feedstock production (fertilizer use)
            # Sources: IPCC biofuel guidelines, lifecycle assessments
            "biodiesel": {"CO2_frac": 0.90, "CH4_frac": 0.05, "N2O_frac": 0.05},
            "bioethanol": {"CO2_frac": 0.88, "CH4_frac": 0.05, "N2O_frac": 0.07},
            "ethanol": {"CO2_frac": 0.88, "CH4_frac": 0.05, "N2O_frac": 0.07},
            "biogas": {"CO2_frac": 0.70, "CH4_frac": 0.25, "N2O_frac": 0.05},
            "biomass": {"CO2_frac": 0.85, "CH4_frac": 0.08, "N2O_frac": 0.07},
            "wood": {"CO2_frac": 0.88, "CH4_frac": 0.07, "N2O_frac": 0.05},
            "wood_pellets": {"CO2_frac": 0.88, "CH4_frac": 0.07, "N2O_frac": 0.05},

            # Electricity: Grid transmission/distribution losses are mostly CO2
            # Sources: EPA eGRID, IEA data
            "electricity": {"CO2_frac": 1.00, "CH4_frac": 0.00, "N2O_frac": 0.00},

            # Default for unknown fuels: Conservative all-CO2 assumption
            "default": {"CO2_frac": 1.00, "CH4_frac": 0.00, "N2O_frac": 0.00},
        }

        # Get profile for this fuel type, fallback to default
        profile = wtt_profiles.get(fuel_type.lower(), wtt_profiles["default"])

        # Extract fractions
        co2_frac = profile["CO2_frac"]
        ch4_frac = profile["CH4_frac"]
        n2o_frac = profile["N2O_frac"]

        # Calculate CO2e contribution from each gas
        co2_co2e = wtt_co2e * co2_frac  # CO2 contribution in CO2e (same as mass)
        ch4_co2e = wtt_co2e * ch4_frac  # CH4 contribution in CO2e
        n2o_co2e = wtt_co2e * n2o_frac  # N2O contribution in CO2e

        # Back-calculate mass of each gas from CO2e using GWP
        # CO2e = mass * GWP, therefore mass = CO2e / GWP
        # For CO2, GWP = 1, so CO2 mass = CO2 CO2e contribution
        co2_kg = co2_co2e  # GWP of CO2 = 1
        ch4_kg = ch4_co2e / ch4_gwp if ch4_gwp > 0 else 0.0
        n2o_kg = n2o_co2e / n2o_gwp if n2o_gwp > 0 else 0.0

        logger.debug(
            f"WTT decomposition for {fuel_type}: "
            f"CO2={co2_kg:.6f}kg, CH4={ch4_kg:.6f}kg, N2O={n2o_kg:.6f}kg "
            f"(profile: CO2={co2_frac:.0%}, CH4={ch4_frac:.0%}, N2O={n2o_frac:.0%})"
        )

        return GHGVectors(
            CO2=co2_kg,
            CH4=ch4_kg,
            N2O=n2o_kg,
        )

    def _make_key(
        self,
        fuel_type: str,
        unit: str,
        geography: str,
        scope: str,
        boundary: str,
        gwp_set: str,
    ) -> str:
        """
        Generate unique lookup key.

        All components are normalized for consistent lookups:
        - unit: lowercase (fixes kWh vs kwh mismatch)
        - boundary: lowercase
        - fuel_type: lowercase
        """
        # Normalize all components to ensure consistent key generation
        return f"{geography}:{fuel_type.lower()}:{unit.lower()}:{scope}:{boundary.lower()}:{gwp_set}"

    def _find_best_match(
        self,
        fuel_type: str,
        unit: str,
        geography: str,
        scope: str,
        boundary: str,
        gwp_set: str,
    ) -> Optional[str]:
        """
        Find best matching factor using fallback strategy.

        Fallback order:
        1. Exact match
        2. Relax GWP set (try AR6_100)
        3. Relax boundary (try combustion)
        4. Relax scope (try scope 1)
        5. Try regional fallback (US if not found)
        """
        # Try exact match first
        key = self._make_key(fuel_type, unit, geography, scope, boundary, gwp_set)
        if key in self.factors:
            return key

        # Fallback 1: Default GWP set
        if gwp_set != "IPCC_AR6_100":
            key = self._make_key(fuel_type, unit, geography, scope, boundary, "IPCC_AR6_100")
            if key in self.factors:
                logger.debug(f"Using fallback GWP set: IPCC_AR6_100")
                return key

        # Fallback 2: Default boundary
        if boundary != "combustion":
            key = self._make_key(fuel_type, unit, geography, scope, "combustion", gwp_set)
            if key in self.factors:
                logger.debug(f"Using fallback boundary: combustion")
                return key

        # Fallback 3: Default scope
        if scope != "1":
            key = self._make_key(fuel_type, unit, geography, "1", boundary, gwp_set)
            if key in self.factors:
                logger.debug(f"Using fallback scope: 1")
                return key

        # Fallback 4: US as default region
        if geography != "US":
            key = self._make_key(fuel_type, unit, "US", scope, boundary, gwp_set)
            if key in self.factors:
                logger.warning(f"Using fallback geography: US (requested: {geography})")
                return key

        return None

    def _load_default_factors_v2(self):
        """
        Load built-in v2 emission factors with multi-gas breakdown.

        These are migrated from v1 defaults using IPCC decomposition ratios.
        """

        # ==================== US FACTORS ====================

        # US Diesel (EPA 2024)
        self.add_factor_record(
            EmissionFactorRecord(
                factor_id="EF:US:diesel:2024:v1",
                fuel_type="diesel",
                unit="gallons",
                geography="US",
                geography_level=GeographyLevel.COUNTRY,
                vectors=GHGVectors(
                    CO2=10.18,      # kg CO2 per gallon
                    CH4=0.00082,    # kg CH4 per gallon
                    N2O=0.000164,   # kg N2O per gallon
                ),
                gwp_100yr=GWPValues(
                    gwp_set=GWPSet.IPCC_AR6_100,
                    CH4_gwp=28,
                    N2O_gwp=273,
                ),
                gwp_20yr=GWPValues(
                    gwp_set=GWPSet.IPCC_AR6_20,
                    CH4_gwp=84,
                    N2O_gwp=273,
                ),
                scope=Scope.SCOPE_1,
                boundary=Boundary.COMBUSTION,
                provenance=SourceProvenance(
                    source_org="EPA",
                    source_publication="Emission Factors for Greenhouse Gas Inventories 2024",
                    source_year=2024,
                    source_url="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
                    methodology=Methodology.IPCC_TIER_1,
                    version="v1",
                ),
                valid_from=date(2024, 1, 1),
                valid_to=date(2024, 12, 31),
                uncertainty_95ci=0.05,  # ±5%
                dqs=DataQualityScore(
                    temporal=5,
                    geographical=4,
                    technological=4,
                    representativeness=4,
                    methodological=5,
                ),
                license_info=LicenseInfo(
                    license="CC0-1.0",
                    redistribution_allowed=True,
                    commercial_use_allowed=True,
                    attribution_required=False,
                    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                ),
                heating_value_basis=HeatingValueBasis.HHV,
                reference_temperature_c=15.0,
                compliance_frameworks=["GHG_Protocol", "IPCC_2006", "EPA_MRR"],
                tags=["fossil", "transport", "stationary", "tier1"],
            )
        )

        # US Diesel (liters)
        self.add_factor_record(
            self._convert_unit_factor(
                self.get_factor_record("diesel", "gallons", "US"),
                new_unit="liters",
                conversion_factor=3.78541,  # gallons to liters
            )
        )

        # US Natural Gas (EPA 2024)
        self.add_factor_record(
            EmissionFactorRecord(
                factor_id="EF:US:natural_gas:2024:v1",
                fuel_type="natural_gas",
                unit="therms",
                geography="US",
                geography_level=GeographyLevel.COUNTRY,
                vectors=GHGVectors(
                    CO2=5.29,       # kg CO2 per therm
                    CH4=0.00096,    # kg CH4 per therm (includes fugitive)
                    N2O=0.0001,     # kg N2O per therm
                ),
                gwp_100yr=GWPValues(
                    gwp_set=GWPSet.IPCC_AR6_100,
                    CH4_gwp=28,
                    N2O_gwp=273,
                ),
                scope=Scope.SCOPE_1,
                boundary=Boundary.COMBUSTION,
                provenance=SourceProvenance(
                    source_org="EPA",
                    source_publication="Emission Factors for Greenhouse Gas Inventories 2024",
                    source_year=2024,
                    source_url="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
                    methodology=Methodology.IPCC_TIER_1,
                    version="v1",
                ),
                valid_from=date(2024, 1, 1),
                valid_to=date(2024, 12, 31),
                uncertainty_95ci=0.08,  # ±8%
                dqs=DataQualityScore(
                    temporal=5,
                    geographical=4,
                    technological=4,
                    representativeness=4,
                    methodological=5,
                ),
                license_info=LicenseInfo(
                    license="CC0-1.0",
                    redistribution_allowed=True,
                    commercial_use_allowed=True,
                    attribution_required=False,
                ),
                compliance_frameworks=["GHG_Protocol", "IPCC_2006", "EPA_MRR"],
                tags=["fossil", "stationary", "tier1"],
            )
        )

        # US Electricity (EPA eGRID 2024)
        self.add_factor_record(
            EmissionFactorRecord(
                factor_id="EF:US:electricity:2024:v1",
                fuel_type="electricity",
                unit="kWh",
                geography="US",
                geography_level=GeographyLevel.COUNTRY,
                vectors=GHGVectors(
                    CO2=0.381,      # kg CO2 per kWh (US grid average)
                    CH4=0.000035,   # kg CH4 per kWh
                    N2O=0.000006,   # kg N2O per kWh
                ),
                gwp_100yr=GWPValues(
                    gwp_set=GWPSet.IPCC_AR6_100,
                    CH4_gwp=28,
                    N2O_gwp=273,
                ),
                scope=Scope.SCOPE_2,
                boundary=Boundary.COMBUSTION,
                provenance=SourceProvenance(
                    source_org="EPA",
                    source_publication="eGRID 2024 Annual Report",
                    source_year=2024,
                    source_url="https://www.epa.gov/egrid",
                    methodology=Methodology.IPCC_TIER_2,
                    version="v1",
                ),
                valid_from=date(2024, 1, 1),
                valid_to=date(2024, 12, 31),
                uncertainty_95ci=0.06,  # ±6%
                dqs=DataQualityScore(
                    temporal=5,
                    geographical=3,  # Country average (not state-specific)
                    technological=3,  # Grid mix (not plant-specific)
                    representativeness=4,
                    methodological=5,
                ),
                license_info=LicenseInfo(
                    license="CC0-1.0",
                    redistribution_allowed=True,
                    commercial_use_allowed=True,
                    attribution_required=False,
                ),
                compliance_frameworks=["GHG_Protocol", "IPCC_2006", "EPA_MRR"],
                tags=["electricity", "grid", "scope2"],
                notes="US grid average emission factor. State-specific factors available via eGRID subregions.",
            )
        )

        # US Gasoline
        self.add_factor_record(
            EmissionFactorRecord(
                factor_id="EF:US:gasoline:2024:v1",
                fuel_type="gasoline",
                unit="gallons",
                geography="US",
                geography_level=GeographyLevel.COUNTRY,
                vectors=GHGVectors(
                    CO2=8.76,
                    CH4=0.00070,
                    N2O=0.000140,
                ),
                gwp_100yr=GWPValues(
                    gwp_set=GWPSet.IPCC_AR6_100,
                    CH4_gwp=28,
                    N2O_gwp=273,
                ),
                scope=Scope.SCOPE_1,
                boundary=Boundary.COMBUSTION,
                provenance=SourceProvenance(
                    source_org="EPA",
                    source_publication="Emission Factors for Greenhouse Gas Inventories 2024",
                    source_year=2024,
                    methodology=Methodology.IPCC_TIER_1,
                ),
                valid_from=date(2024, 1, 1),
                uncertainty_95ci=0.05,
                dqs=DataQualityScore(
                    temporal=5, geographical=4, technological=4,
                    representativeness=4, methodological=5
                ),
                license_info=LicenseInfo(
                    license="CC0-1.0",
                    redistribution_allowed=True,
                    commercial_use_allowed=True,
                    attribution_required=False,
                ),
            )
        )

        # US Coal
        self.add_factor_record(
            EmissionFactorRecord(
                factor_id="EF:US:coal:2024:v1",
                fuel_type="coal",
                unit="tons",
                geography="US",
                geography_level=GeographyLevel.COUNTRY,
                vectors=GHGVectors(
                    CO2=2080.0,     # kg CO2 per ton
                    CH4=0.120,      # kg CH4 per ton
                    N2O=0.032,      # kg N2O per ton
                ),
                gwp_100yr=GWPValues(
                    gwp_set=GWPSet.IPCC_AR6_100,
                    CH4_gwp=28,
                    N2O_gwp=273,
                ),
                scope=Scope.SCOPE_1,
                boundary=Boundary.COMBUSTION,
                provenance=SourceProvenance(
                    source_org="EPA",
                    source_publication="Emission Factors for Greenhouse Gas Inventories 2024",
                    source_year=2024,
                    methodology=Methodology.IPCC_TIER_1,
                ),
                valid_from=date(2024, 1, 1),
                uncertainty_95ci=0.10,
                dqs=DataQualityScore(
                    temporal=5, geographical=4, technological=3,
                    representativeness=4, methodological=5
                ),
                license_info=LicenseInfo(
                    license="CC0-1.0",
                    redistribution_allowed=True,
                    commercial_use_allowed=True,
                    attribution_required=False,
                ),
                tags=["fossil", "stationary", "coal"],
            )
        )

        # ==================== EU FACTORS ====================

        # EU Electricity (IEA 2024)
        self.add_factor_record(
            EmissionFactorRecord(
                factor_id="EF:EU:electricity:2024:v1",
                fuel_type="electricity",
                unit="kWh",
                geography="EU",
                geography_level=GeographyLevel.CONTINENT,
                vectors=GHGVectors(
                    CO2=0.229,
                    CH4=0.000028,
                    N2O=0.000005,
                ),
                gwp_100yr=GWPValues(
                    gwp_set=GWPSet.IPCC_AR6_100,
                    CH4_gwp=28,
                    N2O_gwp=273,
                ),
                scope=Scope.SCOPE_2,
                boundary=Boundary.COMBUSTION,
                provenance=SourceProvenance(
                    source_org="IEA",
                    source_publication="Emissions Factors 2024",
                    source_year=2024,
                    methodology=Methodology.IPCC_TIER_1,
                ),
                valid_from=date(2024, 1, 1),
                uncertainty_95ci=0.12,
                dqs=DataQualityScore(
                    temporal=4, geographical=3, technological=3,
                    representativeness=4, methodological=4
                ),
                license_info=LicenseInfo(
                    license="CC-BY-4.0",
                    redistribution_allowed=True,
                    commercial_use_allowed=True,
                    attribution_required=True,
                    license_url="https://creativecommons.org/licenses/by/4.0/",
                ),
            )
        )

        # ==================== UK FACTORS ====================

        # UK Electricity (BEIS 2024)
        self.add_factor_record(
            EmissionFactorRecord(
                factor_id="EF:UK:electricity:2024:v1",
                fuel_type="electricity",
                unit="kWh",
                geography="UK",
                geography_level=GeographyLevel.COUNTRY,
                vectors=GHGVectors(
                    CO2=0.208,
                    CH4=0.000022,
                    N2O=0.000004,
                ),
                gwp_100yr=GWPValues(
                    gwp_set=GWPSet.IPCC_AR6_100,
                    CH4_gwp=28,
                    N2O_gwp=273,
                ),
                scope=Scope.SCOPE_2,
                boundary=Boundary.COMBUSTION,
                provenance=SourceProvenance(
                    source_org="UK DESNZ",
                    source_publication="UK Government GHG Conversion Factors 2024",
                    source_year=2024,
                    source_url="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
                    methodology=Methodology.IPCC_TIER_1,
                ),
                valid_from=date(2024, 6, 1),
                valid_to=date(2025, 5, 31),
                uncertainty_95ci=0.08,
                dqs=DataQualityScore(
                    temporal=5, geographical=4, technological=3,
                    representativeness=4, methodological=5
                ),
                license_info=LicenseInfo(
                    license="OGL-UK-3.0",
                    redistribution_allowed=True,
                    commercial_use_allowed=True,
                    attribution_required=True,
                    license_url="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
                ),
                compliance_frameworks=["GHG_Protocol", "SECR", "ISO14064"],
            )
        )

        logger.info(f"Loaded {len(self.factors)} default emission factors (v2)")

    def _convert_unit_factor(
        self,
        source_factor: EmissionFactorRecord,
        new_unit: str,
        conversion_factor: float,
    ) -> EmissionFactorRecord:
        """
        Create new factor record with different unit.

        Args:
            source_factor: Original factor
            new_unit: New unit
            conversion_factor: Conversion multiplier (source_unit to new_unit)

        Returns:
            New EmissionFactorRecord with converted values
        """
        # Create new factor ID
        new_factor_id = source_factor.factor_id.replace(
            f":{source_factor.unit}:", f":{new_unit}:"
        )

        # Convert vectors
        new_vectors = GHGVectors(
            CO2=source_factor.vectors.CO2 / conversion_factor,
            CH4=source_factor.vectors.CH4 / conversion_factor,
            N2O=source_factor.vectors.N2O / conversion_factor,
        )

        # Create new factor
        return EmissionFactorRecord(
            factor_id=new_factor_id,
            fuel_type=source_factor.fuel_type,
            unit=new_unit,
            geography=source_factor.geography,
            geography_level=source_factor.geography_level,
            region_hint=source_factor.region_hint,
            vectors=new_vectors,
            gwp_100yr=source_factor.gwp_100yr,
            gwp_20yr=source_factor.gwp_20yr,
            scope=source_factor.scope,
            boundary=source_factor.boundary,
            provenance=source_factor.provenance,
            valid_from=source_factor.valid_from,
            valid_to=source_factor.valid_to,
            uncertainty_95ci=source_factor.uncertainty_95ci,
            dqs=source_factor.dqs,
            license_info=source_factor.license_info,
            heating_value_basis=source_factor.heating_value_basis,
            compliance_frameworks=source_factor.compliance_frameworks,
            tags=source_factor.tags,
            notes=f"Unit converted from {source_factor.unit} using factor {conversion_factor:.6f}",
        )

    def _load_custom_factors(self):
        """Load custom emission factors from JSON files in data directory."""
        if not self.data_dir or not self.data_dir.exists():
            return

        # Find all .json files
        factor_files = list(self.data_dir.glob("**/*.json"))

        for file_path in factor_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Check if single factor or list of factors
                if isinstance(data, list):
                    for factor_data in data:
                        factor = EmissionFactorRecord.from_dict(factor_data)
                        self.add_factor_record(factor)
                else:
                    factor = EmissionFactorRecord.from_dict(data)
                    self.add_factor_record(factor)

                logger.info(f"Loaded custom factors from {file_path}")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

    def save_to_directory(self, output_dir: str):
        """
        Save all factors to JSON files in directory.

        Organizes as: {geography}/{fuel_type}_{year}_{version}.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Group by geography
        by_geography: Dict[str, List[EmissionFactorRecord]] = {}
        for factor in self.factors.values():
            if factor.geography not in by_geography:
                by_geography[factor.geography] = []
            by_geography[factor.geography].append(factor)

        # Write files
        for geography, factors in by_geography.items():
            geo_dir = output_path / geography
            geo_dir.mkdir(exist_ok=True)

            for factor in factors:
                # Extract year from factor_id or valid_from
                year = factor.provenance.source_year
                version = factor.provenance.version

                filename = f"{factor.fuel_type}_{factor.unit}_{year}_{version}.json"
                file_path = geo_dir / filename

                with open(file_path, "w") as f:
                    f.write(factor.to_json())

        logger.info(f"Saved {len(self.factors)} factors to {output_dir}")

    # ==================== CACHE MANAGEMENT ====================

    def _warm_cache(self):
        """
        Warm cache with most common emission factors.

        Pre-loads frequently accessed factors to achieve high hit rate from start.
        """
        if not self.enable_cache or not self.cache:
            return

        # Common factors to pre-load (based on typical usage patterns)
        common_lookups = [
            # US factors (most common)
            ("diesel", "gallons", "US", "1", "combustion", "IPCC_AR6_100"),
            ("gasoline", "gallons", "US", "1", "combustion", "IPCC_AR6_100"),
            ("natural_gas", "therms", "US", "1", "combustion", "IPCC_AR6_100"),
            ("electricity", "kWh", "US", "2", "combustion", "IPCC_AR6_100"),
            ("propane", "gallons", "US", "1", "combustion", "IPCC_AR6_100"),
            ("fuel_oil", "gallons", "US", "1", "combustion", "IPCC_AR6_100"),
            # UK factors
            ("electricity", "kWh", "UK", "2", "combustion", "IPCC_AR6_100"),
            ("natural_gas", "kWh", "UK", "1", "combustion", "IPCC_AR6_100"),
            ("diesel", "liters", "UK", "1", "combustion", "IPCC_AR6_100"),
        ]

        warmed = 0
        for params in common_lookups:
            fuel_type, unit, geography, scope, boundary, gwp_set = params
            factor = self.get_factor_record(fuel_type, unit, geography, scope, boundary, gwp_set)
            if factor:
                warmed += 1

        logger.info(f"Cache warmed with {warmed} common factors")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate_pct, size, etc.
            Returns empty dict if caching disabled.
        """
        if not self.enable_cache or not self.cache:
            return {
                "enabled": False,
                "message": "Caching is disabled",
            }

        stats = self.cache.get_stats()
        stats["enabled"] = True
        return stats

    def clear_cache(self):
        """Clear all cache entries."""
        if self.enable_cache and self.cache:
            self.cache.clear()
            logger.info("Cache cleared")

    def invalidate_cache(
        self,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
    ):
        """
        Invalidate cache entries matching criteria.

        Args:
            fuel_type: Invalidate entries for this fuel type (optional)
            geography: Invalidate entries for this geography (optional)
        """
        if self.enable_cache and self.cache:
            count = self.cache.invalidate(fuel_type, geography)
            logger.info(f"Invalidated {count} cache entries")


# ==================== BACKWARD COMPATIBILITY (v1) ====================

# Keep old EmissionFactors class for v1 compatibility
class EmissionFactors:
    """
    DEPRECATED: v1 emission factors (scalar CO2e only).

    This class is maintained for backward compatibility.
    New code should use EmissionFactorDatabase.
    """

    def __init__(self, custom_factors_path: Optional[str] = None):
        logger.warning(
            "EmissionFactors (v1) is deprecated. Use EmissionFactorDatabase (v2) instead."
        )

        # Initialize v2 database
        self._db = EmissionFactorDatabase()

    def get_factor(
        self, fuel_type: str, unit: str, region: str = "US"
    ) -> Optional[float]:
        """Get emission factor (v1 API - scalar CO2e)."""
        return self._db.get_factor(fuel_type, unit, region)

    def get_available_regions(self) -> list:
        return self._db.get_available_regions()

    def get_available_fuels(self, region: str = "US") -> list:
        return self._db.get_available_fuels(region)

    def get_available_units(self, fuel_type: str, region: str = "US") -> list:
        return self._db.get_available_units(fuel_type, region)
