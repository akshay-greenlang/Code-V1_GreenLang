"""
Emission Factor Database for GL-003 UNIFIEDSTEAM

Provides comprehensive emission factor management for fuel-specific and
grid-specific CO2e calculations. Supports Scope 1, 2, and 3 emissions
with versioned factor tables and audit trails.

Reference Sources:
    - EPA Emission Factors Hub (2024)
    - DEFRA GHG Conversion Factors
    - IEA Emission Factors by Country
    - IPCC Guidelines for National GHG Inventories

Author: GL-003 Climate Intelligence Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class EmissionScope(Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"  # Direct emissions from owned/controlled sources
    SCOPE_2 = "scope_2"  # Indirect emissions from purchased energy
    SCOPE_3 = "scope_3"  # Other indirect emissions in value chain


class FuelType(Enum):
    """Standard fuel types for steam generation."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    DIESEL = "diesel"
    PROPANE = "propane"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    COAL_LIGNITE = "coal_lignite"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_BAGASSE = "biomass_bagasse"
    WASTE_HEAT = "waste_heat"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"
    REFINERY_GAS = "refinery_gas"
    COKE_OVEN_GAS = "coke_oven_gas"
    BLAST_FURNACE_GAS = "blast_furnace_gas"


class GridRegion(Enum):
    """Grid regions for electricity emission factors."""
    # US EPA eGRID regions
    US_WECC = "us_wecc"
    US_ERCOT = "us_ercot"
    US_SERC = "us_serc"
    US_MRO = "us_mro"
    US_RFC = "us_rfc"
    US_NPCC = "us_npcc"
    US_FRCC = "us_frcc"
    US_AVERAGE = "us_average"
    # European regions
    EU_WEST = "eu_west"
    EU_EAST = "eu_east"
    EU_NORTH = "eu_north"
    EU_AVERAGE = "eu_average"
    # Asia-Pacific
    ASIA_CHINA = "asia_china"
    ASIA_INDIA = "asia_india"
    ASIA_JAPAN = "asia_japan"
    ASIA_SOUTHEAST = "asia_southeast"
    # Other
    GLOBAL_AVERAGE = "global_average"
    CUSTOM = "custom"


class EmissionFactorUnit(Enum):
    """Units for emission factors."""
    KG_CO2E_PER_GJ = "kg_CO2e/GJ"
    KG_CO2E_PER_KWH = "kg_CO2e/kWh"
    KG_CO2E_PER_MMBTU = "kg_CO2e/MMBtu"
    KG_CO2E_PER_KG = "kg_CO2e/kg"
    KG_CO2E_PER_LITER = "kg_CO2e/L"
    KG_CO2E_PER_M3 = "kg_CO2e/m3"
    KG_CO2E_PER_TON = "kg_CO2e/ton"


@dataclass
class EmissionFactor:
    """
    Emission factor with full provenance and uncertainty.

    Attributes:
        fuel_type: Type of fuel or energy source
        co2_factor: CO2 emission factor (kg CO2/unit)
        ch4_factor: CH4 emission factor (kg CH4/unit)
        n2o_factor: N2O emission factor (kg N2O/unit)
        co2e_factor: Combined CO2e factor using GWP values
        unit: Unit of measurement
        source: Data source reference
        version: Factor version (year or revision)
        effective_date: Date factor became effective
        expiry_date: Date factor expires (if applicable)
        uncertainty_pct: Uncertainty percentage (+/-)
        gwp_co2: Global Warming Potential for CO2 (always 1)
        gwp_ch4: Global Warming Potential for CH4
        gwp_n2o: Global Warming Potential for N2O
        notes: Additional notes or caveats
    """
    fuel_type: FuelType
    co2_factor: Decimal
    ch4_factor: Decimal
    n2o_factor: Decimal
    co2e_factor: Decimal
    unit: EmissionFactorUnit
    source: str
    version: str
    effective_date: datetime
    expiry_date: Optional[datetime] = None
    uncertainty_pct: Decimal = Decimal("5.0")
    gwp_co2: int = 1
    gwp_ch4: int = 28  # AR5 100-year GWP
    gwp_n2o: int = 265  # AR5 100-year GWP
    notes: str = ""

    def __post_init__(self):
        """Validate and compute CO2e if not provided."""
        if self.co2e_factor == Decimal("0"):
            # Calculate CO2e from component factors
            self.co2e_factor = (
                self.co2_factor * self.gwp_co2 +
                self.ch4_factor * self.gwp_ch4 +
                self.n2o_factor * self.gwp_n2o
            )

    def get_uncertainty_bounds(self) -> Tuple[Decimal, Decimal]:
        """Return 95% confidence interval bounds."""
        uncertainty = self.co2e_factor * (self.uncertainty_pct / Decimal("100"))
        return (
            self.co2e_factor - uncertainty * Decimal("1.96"),
            self.co2e_factor + uncertainty * Decimal("1.96")
        )

    def is_valid(self, check_date: Optional[datetime] = None) -> bool:
        """Check if factor is valid for the given date."""
        check_date = check_date or datetime.now(timezone.utc)
        if check_date < self.effective_date:
            return False
        if self.expiry_date and check_date > self.expiry_date:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fuel_type": self.fuel_type.value,
            "co2_factor": str(self.co2_factor),
            "ch4_factor": str(self.ch4_factor),
            "n2o_factor": str(self.n2o_factor),
            "co2e_factor": str(self.co2e_factor),
            "unit": self.unit.value,
            "source": self.source,
            "version": self.version,
            "effective_date": self.effective_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "uncertainty_pct": str(self.uncertainty_pct),
            "gwp_ch4": self.gwp_ch4,
            "gwp_n2o": self.gwp_n2o,
            "notes": self.notes,
        }


@dataclass
class GridEmissionFactor:
    """
    Grid electricity emission factor.

    Attributes:
        region: Grid region identifier
        co2e_factor: Combined CO2e factor (kg CO2e/kWh)
        source: Data source reference
        version: Factor version (year)
        generation_mix: Dictionary of generation sources and percentages
    """
    region: GridRegion
    co2e_factor: Decimal
    source: str
    version: str
    effective_date: datetime
    generation_mix: Dict[str, Decimal] = field(default_factory=dict)
    uncertainty_pct: Decimal = Decimal("10.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "region": self.region.value,
            "co2e_factor": str(self.co2e_factor),
            "source": self.source,
            "version": self.version,
            "effective_date": self.effective_date.isoformat(),
            "generation_mix": {k: str(v) for k, v in self.generation_mix.items()},
            "uncertainty_pct": str(self.uncertainty_pct),
        }


class EmissionFactorDatabase:
    """
    Comprehensive emission factor database with versioning and audit trail.

    Provides EPA, DEFRA, and IEA emission factors for steam system
    climate impact calculations. Supports custom factors with validation.

    Example:
        >>> db = EmissionFactorDatabase()
        >>> factor = db.get_fuel_factor(FuelType.NATURAL_GAS)
        >>> print(f"Natural gas: {factor.co2e_factor} {factor.unit.value}")
    """

    # Standard fuel emission factors (EPA 2024)
    # Units: kg CO2e per GJ (gross calorific value)
    DEFAULT_FUEL_FACTORS: Dict[FuelType, Dict[str, Any]] = {
        FuelType.NATURAL_GAS: {
            "co2": Decimal("56.1"),
            "ch4": Decimal("0.001"),
            "n2o": Decimal("0.0001"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("3.0"),
        },
        FuelType.FUEL_OIL_NO2: {
            "co2": Decimal("73.96"),
            "ch4": Decimal("0.003"),
            "n2o": Decimal("0.0006"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("5.0"),
        },
        FuelType.FUEL_OIL_NO6: {
            "co2": Decimal("77.37"),
            "ch4": Decimal("0.003"),
            "n2o": Decimal("0.0006"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("5.0"),
        },
        FuelType.DIESEL: {
            "co2": Decimal("74.14"),
            "ch4": Decimal("0.003"),
            "n2o": Decimal("0.0006"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("5.0"),
        },
        FuelType.PROPANE: {
            "co2": Decimal("63.07"),
            "ch4": Decimal("0.001"),
            "n2o": Decimal("0.0001"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("3.0"),
        },
        FuelType.COAL_BITUMINOUS: {
            "co2": Decimal("94.6"),
            "ch4": Decimal("0.011"),
            "n2o": Decimal("0.0016"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("7.0"),
        },
        FuelType.COAL_SUBBITUMINOUS: {
            "co2": Decimal("97.17"),
            "ch4": Decimal("0.011"),
            "n2o": Decimal("0.0016"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("7.0"),
        },
        FuelType.COAL_LIGNITE: {
            "co2": Decimal("101.0"),
            "ch4": Decimal("0.011"),
            "n2o": Decimal("0.0016"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("10.0"),
        },
        FuelType.BIOMASS_WOOD: {
            "co2": Decimal("0.0"),  # Biogenic CO2 typically excluded
            "ch4": Decimal("0.030"),
            "n2o": Decimal("0.004"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("15.0"),
            "notes": "Biogenic CO2 excluded per GHG Protocol",
        },
        FuelType.BIOMASS_BAGASSE: {
            "co2": Decimal("0.0"),
            "ch4": Decimal("0.030"),
            "n2o": Decimal("0.004"),
            "source": "EPA Emission Factors Hub 2024",
            "uncertainty": Decimal("15.0"),
            "notes": "Biogenic CO2 excluded per GHG Protocol",
        },
        FuelType.WASTE_HEAT: {
            "co2": Decimal("0.0"),
            "ch4": Decimal("0.0"),
            "n2o": Decimal("0.0"),
            "source": "GHG Protocol",
            "uncertainty": Decimal("0.0"),
            "notes": "Waste heat recovery has no direct emissions",
        },
        FuelType.HYDROGEN: {
            "co2": Decimal("0.0"),
            "ch4": Decimal("0.0"),
            "n2o": Decimal("0.0"),
            "source": "GHG Protocol",
            "uncertainty": Decimal("0.0"),
            "notes": "Green hydrogen; gray/blue H2 factors vary",
        },
        FuelType.REFINERY_GAS: {
            "co2": Decimal("64.2"),
            "ch4": Decimal("0.001"),
            "n2o": Decimal("0.0001"),
            "source": "API Compendium 2021",
            "uncertainty": Decimal("10.0"),
        },
        FuelType.COKE_OVEN_GAS: {
            "co2": Decimal("44.4"),
            "ch4": Decimal("0.001"),
            "n2o": Decimal("0.0001"),
            "source": "IPCC 2006 Guidelines",
            "uncertainty": Decimal("10.0"),
        },
        FuelType.BLAST_FURNACE_GAS: {
            "co2": Decimal("260.0"),
            "ch4": Decimal("0.001"),
            "n2o": Decimal("0.0001"),
            "source": "IPCC 2006 Guidelines",
            "uncertainty": Decimal("10.0"),
        },
    }

    # Grid emission factors (kg CO2e/kWh)
    DEFAULT_GRID_FACTORS: Dict[GridRegion, Dict[str, Any]] = {
        GridRegion.US_WECC: {
            "co2e": Decimal("0.322"),
            "source": "EPA eGRID 2022",
        },
        GridRegion.US_ERCOT: {
            "co2e": Decimal("0.396"),
            "source": "EPA eGRID 2022",
        },
        GridRegion.US_SERC: {
            "co2e": Decimal("0.428"),
            "source": "EPA eGRID 2022",
        },
        GridRegion.US_MRO: {
            "co2e": Decimal("0.456"),
            "source": "EPA eGRID 2022",
        },
        GridRegion.US_RFC: {
            "co2e": Decimal("0.412"),
            "source": "EPA eGRID 2022",
        },
        GridRegion.US_NPCC: {
            "co2e": Decimal("0.227"),
            "source": "EPA eGRID 2022",
        },
        GridRegion.US_FRCC: {
            "co2e": Decimal("0.389"),
            "source": "EPA eGRID 2022",
        },
        GridRegion.US_AVERAGE: {
            "co2e": Decimal("0.386"),
            "source": "EPA eGRID 2022",
        },
        GridRegion.EU_WEST: {
            "co2e": Decimal("0.233"),
            "source": "IEA 2023",
        },
        GridRegion.EU_EAST: {
            "co2e": Decimal("0.412"),
            "source": "IEA 2023",
        },
        GridRegion.EU_NORTH: {
            "co2e": Decimal("0.047"),
            "source": "IEA 2023",
        },
        GridRegion.EU_AVERAGE: {
            "co2e": Decimal("0.276"),
            "source": "IEA 2023",
        },
        GridRegion.ASIA_CHINA: {
            "co2e": Decimal("0.555"),
            "source": "IEA 2023",
        },
        GridRegion.ASIA_INDIA: {
            "co2e": Decimal("0.708"),
            "source": "IEA 2023",
        },
        GridRegion.ASIA_JAPAN: {
            "co2e": Decimal("0.457"),
            "source": "IEA 2023",
        },
        GridRegion.ASIA_SOUTHEAST: {
            "co2e": Decimal("0.512"),
            "source": "IEA 2023",
        },
        GridRegion.GLOBAL_AVERAGE: {
            "co2e": Decimal("0.436"),
            "source": "IEA 2023",
        },
    }

    def __init__(
        self,
        custom_fuel_factors: Optional[Dict[FuelType, EmissionFactor]] = None,
        custom_grid_factors: Optional[Dict[GridRegion, GridEmissionFactor]] = None,
        version: str = "2024.1",
    ):
        """
        Initialize emission factor database.

        Args:
            custom_fuel_factors: Override default fuel factors
            custom_grid_factors: Override default grid factors
            version: Database version identifier
        """
        self.version = version
        self.created_at = datetime.now(timezone.utc)

        # Build fuel factor database
        self._fuel_factors: Dict[FuelType, EmissionFactor] = {}
        self._build_default_fuel_factors()

        # Apply custom fuel factors
        if custom_fuel_factors:
            for fuel_type, factor in custom_fuel_factors.items():
                self._fuel_factors[fuel_type] = factor
                logger.info(f"Custom fuel factor applied for {fuel_type.value}")

        # Build grid factor database
        self._grid_factors: Dict[GridRegion, GridEmissionFactor] = {}
        self._build_default_grid_factors()

        # Apply custom grid factors
        if custom_grid_factors:
            for region, factor in custom_grid_factors.items():
                self._grid_factors[region] = factor
                logger.info(f"Custom grid factor applied for {region.value}")

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
        self._log_action("initialize", {"version": version})

    def _build_default_fuel_factors(self):
        """Build default fuel emission factors from EPA data."""
        effective_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

        for fuel_type, data in self.DEFAULT_FUEL_FACTORS.items():
            factor = EmissionFactor(
                fuel_type=fuel_type,
                co2_factor=data["co2"],
                ch4_factor=data["ch4"],
                n2o_factor=data["n2o"],
                co2e_factor=Decimal("0"),  # Will be calculated
                unit=EmissionFactorUnit.KG_CO2E_PER_GJ,
                source=data["source"],
                version=self.version,
                effective_date=effective_date,
                uncertainty_pct=data.get("uncertainty", Decimal("5.0")),
                notes=data.get("notes", ""),
            )
            self._fuel_factors[fuel_type] = factor

    def _build_default_grid_factors(self):
        """Build default grid emission factors from EPA/IEA data."""
        effective_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

        for region, data in self.DEFAULT_GRID_FACTORS.items():
            factor = GridEmissionFactor(
                region=region,
                co2e_factor=data["co2e"],
                source=data["source"],
                version=self.version,
                effective_date=effective_date,
            )
            self._grid_factors[region] = factor

    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log action to audit trail."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details,
        }
        self._audit_log.append(entry)

    def get_fuel_factor(
        self,
        fuel_type: FuelType,
        check_date: Optional[datetime] = None,
    ) -> EmissionFactor:
        """
        Get emission factor for a fuel type.

        Args:
            fuel_type: Type of fuel
            check_date: Date for validity check

        Returns:
            EmissionFactor for the specified fuel

        Raises:
            KeyError: If fuel type not found
            ValueError: If factor is not valid for the date
        """
        if fuel_type not in self._fuel_factors:
            raise KeyError(f"Emission factor not found for fuel type: {fuel_type.value}")

        factor = self._fuel_factors[fuel_type]

        if not factor.is_valid(check_date):
            raise ValueError(
                f"Emission factor for {fuel_type.value} is not valid for "
                f"date {check_date or 'now'}"
            )

        self._log_action("get_fuel_factor", {"fuel_type": fuel_type.value})
        return factor

    def get_grid_factor(
        self,
        region: GridRegion,
        check_date: Optional[datetime] = None,
    ) -> GridEmissionFactor:
        """
        Get grid emission factor for a region.

        Args:
            region: Grid region
            check_date: Date for validity check

        Returns:
            GridEmissionFactor for the specified region
        """
        if region not in self._grid_factors:
            raise KeyError(f"Grid factor not found for region: {region.value}")

        self._log_action("get_grid_factor", {"region": region.value})
        return self._grid_factors[region]

    def calculate_fuel_emissions(
        self,
        fuel_type: FuelType,
        energy_gj: Decimal,
        include_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate emissions from fuel consumption.

        Args:
            fuel_type: Type of fuel
            energy_gj: Energy consumed in GJ
            include_uncertainty: Include uncertainty bounds

        Returns:
            Dictionary with CO2, CH4, N2O, and CO2e emissions
        """
        factor = self.get_fuel_factor(fuel_type)

        co2_kg = factor.co2_factor * energy_gj
        ch4_kg = factor.ch4_factor * energy_gj
        n2o_kg = factor.n2o_factor * energy_gj
        co2e_kg = factor.co2e_factor * energy_gj

        result = {
            "fuel_type": fuel_type.value,
            "energy_gj": str(energy_gj),
            "co2_kg": str(co2_kg.quantize(Decimal("0.01"), ROUND_HALF_UP)),
            "ch4_kg": str(ch4_kg.quantize(Decimal("0.0001"), ROUND_HALF_UP)),
            "n2o_kg": str(n2o_kg.quantize(Decimal("0.00001"), ROUND_HALF_UP)),
            "co2e_kg": str(co2e_kg.quantize(Decimal("0.01"), ROUND_HALF_UP)),
            "factor_source": factor.source,
            "factor_version": factor.version,
            "calculation_hash": self._compute_hash(
                fuel_type.value, str(energy_gj), factor.version
            ),
        }

        if include_uncertainty:
            lower, upper = factor.get_uncertainty_bounds()
            result["co2e_lower_kg"] = str(
                (lower * energy_gj).quantize(Decimal("0.01"), ROUND_HALF_UP)
            )
            result["co2e_upper_kg"] = str(
                (upper * energy_gj).quantize(Decimal("0.01"), ROUND_HALF_UP)
            )
            result["uncertainty_pct"] = str(factor.uncertainty_pct)

        self._log_action("calculate_fuel_emissions", {
            "fuel_type": fuel_type.value,
            "energy_gj": str(energy_gj),
            "co2e_kg": result["co2e_kg"],
        })

        return result

    def calculate_electricity_emissions(
        self,
        region: GridRegion,
        energy_kwh: Decimal,
        include_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate emissions from electricity consumption.

        Args:
            region: Grid region
            energy_kwh: Electricity consumed in kWh
            include_uncertainty: Include uncertainty bounds

        Returns:
            Dictionary with CO2e emissions
        """
        factor = self.get_grid_factor(region)
        co2e_kg = factor.co2e_factor * energy_kwh

        result = {
            "region": region.value,
            "energy_kwh": str(energy_kwh),
            "co2e_kg": str(co2e_kg.quantize(Decimal("0.01"), ROUND_HALF_UP)),
            "factor_source": factor.source,
            "factor_version": factor.version,
            "calculation_hash": self._compute_hash(
                region.value, str(energy_kwh), factor.version
            ),
        }

        if include_uncertainty:
            uncertainty = factor.uncertainty_pct / Decimal("100")
            result["co2e_lower_kg"] = str(
                (co2e_kg * (1 - uncertainty * Decimal("1.96"))).quantize(
                    Decimal("0.01"), ROUND_HALF_UP
                )
            )
            result["co2e_upper_kg"] = str(
                (co2e_kg * (1 + uncertainty * Decimal("1.96"))).quantize(
                    Decimal("0.01"), ROUND_HALF_UP
                )
            )
            result["uncertainty_pct"] = str(factor.uncertainty_pct)

        return result

    def _compute_hash(self, *args) -> str:
        """Compute deterministic hash for audit trail."""
        data = "|".join(str(a) for a in args)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def list_available_fuels(self) -> List[str]:
        """List all available fuel types."""
        return [f.value for f in self._fuel_factors.keys()]

    def list_available_regions(self) -> List[str]:
        """List all available grid regions."""
        return [r.value for r in self._grid_factors.keys()]

    def export_factors(self) -> Dict[str, Any]:
        """Export all factors for external use or backup."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "fuel_factors": {
                ft.value: f.to_dict() for ft, f in self._fuel_factors.items()
            },
            "grid_factors": {
                r.value: f.to_dict() for r, f in self._grid_factors.items()
            },
        }

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log entries."""
        return self._audit_log.copy()
