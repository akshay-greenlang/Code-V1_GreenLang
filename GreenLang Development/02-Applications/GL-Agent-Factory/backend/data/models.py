"""
Emission Factor Data Models

This module defines the data models for emission factors used throughout
the GreenLang platform. These models support version pinning, uncertainty
tracking, and audit trails for reproducible GHG calculations.

Sources supported:
- EPA (US Environmental Protection Agency)
- DEFRA (UK Department for Environment, Food and Rural Affairs)
- IPCC (Intergovernmental Panel on Climate Change)
- IEA (International Energy Agency)
- Ecoinvent (LCA Database)
- GaBi (LCA Database)
- WSA (World Steel Association)
- IAI (International Aluminium Institute)
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, List, Any
import hashlib
import json


class EmissionFactorSource(Enum):
    """Authoritative emission factor sources."""
    EPA = "epa"
    DEFRA = "defra"
    IPCC = "ipcc"
    IEA = "iea"
    ECOINVENT = "ecoinvent"
    GABI = "gabi"
    WSA = "wsa"
    IAI = "iai"
    CUSTOM = "custom"


class GWPSet(Enum):
    """IPCC Assessment Report GWP sets."""
    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"


class EmissionScope(Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class EmissionCategory(Enum):
    """Emission factor categories."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    ELECTRICITY = "electricity"
    FUGITIVE = "fugitive"
    INDUSTRIAL_PROCESS = "industrial_process"
    TRANSPORT = "transport"
    WASTE = "waste"
    AGRICULTURE = "agriculture"
    PURCHASED_GOODS = "purchased_goods"
    CAPITAL_GOODS = "capital_goods"
    WELL_TO_TANK = "well_to_tank"
    BUSINESS_TRAVEL = "business_travel"
    EMPLOYEE_COMMUTING = "employee_commuting"
    FREIGHT = "freight"
    MATERIALS = "materials"


class UnitType(Enum):
    """Unit type categories."""
    ENERGY = "energy"
    MASS = "mass"
    VOLUME = "volume"
    DISTANCE = "distance"
    AREA = "area"
    EMISSION = "emission"
    CURRENCY = "currency"


@dataclass
class UncertaintyRange:
    """Represents uncertainty bounds for emission factors."""
    lower_bound: Optional[Decimal] = None
    upper_bound: Optional[Decimal] = None
    lower_pct: Optional[float] = None
    upper_pct: Optional[float] = None
    confidence_level: float = 0.95
    methodology: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "lower_bound": str(self.lower_bound) if self.lower_bound else None,
            "upper_bound": str(self.upper_bound) if self.upper_bound else None,
            "lower_pct": self.lower_pct,
            "upper_pct": self.upper_pct,
            "confidence_level": self.confidence_level,
            "methodology": self.methodology
        }


@dataclass
class EmissionFactor:
    """
    Core emission factor model supporting all GHG calculation needs.

    Attributes:
        id: Unique identifier in format ef://source/category/fuel/year
        value: The emission factor value
        unit: Unit of measurement (e.g., kg CO2e/kWh)
        source: Data source (EPA, DEFRA, IPCC, etc.)
        year: Data year for the factor
        region: Geographic region/country
        uncertainty_lower: Lower uncertainty bound (optional)
        uncertainty_upper: Upper uncertainty bound (optional)
        gwp_set: Which IPCC GWP values were used (AR5, AR6)
        last_updated: When this factor was last updated
        checksum: SHA-256 hash for data integrity
    """

    # Core identification
    id: str

    # Value and measurement
    value: Decimal
    unit: str

    # Source tracking
    source: EmissionFactorSource
    source_document: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None

    # Temporal attributes
    year: int = field(default_factory=lambda: datetime.now().year)
    data_year: Optional[int] = None  # Year the underlying data represents
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    # Geographic attributes
    region: str = "global"
    country_code: Optional[str] = None
    sub_region: Optional[str] = None

    # Classification
    scope: Optional[EmissionScope] = None
    category: Optional[EmissionCategory] = None
    fuel_type: Optional[str] = None
    activity_type: Optional[str] = None

    # GHG breakdown
    co2_factor: Optional[Decimal] = None
    ch4_factor: Optional[Decimal] = None
    n2o_factor: Optional[Decimal] = None
    co2e_factor: Optional[Decimal] = None
    biogenic_co2_factor: Optional[Decimal] = None

    # Uncertainty
    uncertainty_lower: Optional[Decimal] = None
    uncertainty_upper: Optional[Decimal] = None
    uncertainty: Optional[UncertaintyRange] = None

    # GWP settings
    gwp_set: GWPSet = GWPSet.AR5

    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)
    checksum: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Audit
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    def __post_init__(self):
        """Calculate checksum after initialization."""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of core factor data."""
        data = {
            "id": self.id,
            "value": str(self.value),
            "unit": self.unit,
            "source": self.source.value if isinstance(self.source, EmissionFactorSource) else self.source,
            "year": self.year,
            "region": self.region,
            "gwp_set": self.gwp_set.value if isinstance(self.gwp_set, GWPSet) else self.gwp_set
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def verify_checksum(self) -> bool:
        """Verify data integrity using checksum."""
        return self.checksum == self._calculate_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "value": str(self.value),
            "unit": self.unit,
            "source": self.source.value if isinstance(self.source, EmissionFactorSource) else self.source,
            "source_document": self.source_document,
            "source_url": self.source_url,
            "source_version": self.source_version,
            "year": self.year,
            "data_year": self.data_year,
            "region": self.region,
            "country_code": self.country_code,
            "scope": self.scope.value if self.scope else None,
            "category": self.category.value if self.category else None,
            "fuel_type": self.fuel_type,
            "co2_factor": str(self.co2_factor) if self.co2_factor else None,
            "ch4_factor": str(self.ch4_factor) if self.ch4_factor else None,
            "n2o_factor": str(self.n2o_factor) if self.n2o_factor else None,
            "co2e_factor": str(self.co2e_factor) if self.co2e_factor else None,
            "biogenic_co2_factor": str(self.biogenic_co2_factor) if self.biogenic_co2_factor else None,
            "uncertainty_lower": str(self.uncertainty_lower) if self.uncertainty_lower else None,
            "uncertainty_upper": str(self.uncertainty_upper) if self.uncertainty_upper else None,
            "gwp_set": self.gwp_set.value if isinstance(self.gwp_set, GWPSet) else self.gwp_set,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "checksum": self.checksum,
            "notes": self.notes,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmissionFactor':
        """Create EmissionFactor from dictionary."""
        # Convert enums
        source = EmissionFactorSource(data.get("source", "custom"))
        gwp_set = GWPSet(data.get("gwp_set", "AR5"))
        scope = EmissionScope(data["scope"]) if data.get("scope") else None
        category = EmissionCategory(data["category"]) if data.get("category") else None

        # Parse datetime
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        elif last_updated is None:
            last_updated = datetime.utcnow()

        return cls(
            id=data["id"],
            value=Decimal(str(data["value"])),
            unit=data["unit"],
            source=source,
            source_document=data.get("source_document"),
            source_url=data.get("source_url"),
            source_version=data.get("source_version"),
            year=data.get("year", datetime.now().year),
            data_year=data.get("data_year"),
            region=data.get("region", "global"),
            country_code=data.get("country_code"),
            scope=scope,
            category=category,
            fuel_type=data.get("fuel_type"),
            co2_factor=Decimal(str(data["co2_factor"])) if data.get("co2_factor") else None,
            ch4_factor=Decimal(str(data["ch4_factor"])) if data.get("ch4_factor") else None,
            n2o_factor=Decimal(str(data["n2o_factor"])) if data.get("n2o_factor") else None,
            co2e_factor=Decimal(str(data["co2e_factor"])) if data.get("co2e_factor") else None,
            biogenic_co2_factor=Decimal(str(data["biogenic_co2_factor"])) if data.get("biogenic_co2_factor") else None,
            uncertainty_lower=Decimal(str(data["uncertainty_lower"])) if data.get("uncertainty_lower") else None,
            uncertainty_upper=Decimal(str(data["uncertainty_upper"])) if data.get("uncertainty_upper") else None,
            gwp_set=gwp_set,
            last_updated=last_updated,
            checksum=data.get("checksum"),
            notes=data.get("notes"),
            tags=data.get("tags", [])
        )


@dataclass
class GWPValue:
    """
    Global Warming Potential value for a specific gas.

    Attributes:
        gas_name: Name of the greenhouse gas
        chemical_formula: Chemical formula (e.g., CH4, N2O)
        gwp_100yr: 100-year GWP value
        gwp_20yr: 20-year GWP value (optional)
        assessment_report: IPCC assessment report (AR4, AR5, AR6)
        atmospheric_lifetime_years: Atmospheric lifetime
    """
    gas_name: str
    chemical_formula: str
    gwp_100yr: Decimal
    gwp_20yr: Optional[Decimal] = None
    assessment_report: GWPSet = GWPSet.AR6
    atmospheric_lifetime_years: Optional[float] = None
    cas_number: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "gas_name": self.gas_name,
            "chemical_formula": self.chemical_formula,
            "gwp_100yr": str(self.gwp_100yr),
            "gwp_20yr": str(self.gwp_20yr) if self.gwp_20yr else None,
            "assessment_report": self.assessment_report.value,
            "atmospheric_lifetime_years": self.atmospheric_lifetime_years,
            "cas_number": self.cas_number
        }


@dataclass
class GridEmissionFactor:
    """
    Electricity grid emission factor for a specific region.

    Attributes:
        id: Unique identifier
        country: Country name
        country_code: ISO country code
        co2_factor: CO2 emission factor
        unit: Unit of measurement (typically kg CO2/kWh)
        source: Data source
        year: Data year
        method: Location-based or market-based
        renewable_share_pct: Percentage of renewable energy
    """
    id: str
    country: str
    country_code: str
    co2_factor: Decimal
    unit: str = "kg CO2/kWh"
    source: EmissionFactorSource = EmissionFactorSource.IEA
    year: int = 2024
    data_year: int = 2022
    method: str = "location-based"
    renewable_share_pct: Optional[float] = None
    ch4_factor: Optional[Decimal] = None
    n2o_factor: Optional[Decimal] = None
    co2e_factor: Optional[Decimal] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "country": self.country,
            "country_code": self.country_code,
            "co2_factor": str(self.co2_factor),
            "co2e_factor": str(self.co2e_factor) if self.co2e_factor else str(self.co2_factor),
            "unit": self.unit,
            "source": self.source.value,
            "year": self.year,
            "data_year": self.data_year,
            "method": self.method,
            "renewable_share_pct": self.renewable_share_pct,
            "notes": self.notes
        }


@dataclass
class MaterialEmissionFactor:
    """
    Emission factor for purchased materials/products.

    Supports Scope 3 Category 1 (Purchased Goods) calculations.
    """
    id: str
    material_name: str
    material_category: str
    co2e_factor: Decimal
    unit: str  # e.g., kg CO2e/kg, kg CO2e/unit
    source: EmissionFactorSource
    year: int
    region: str = "global"
    lifecycle_stage: str = "cradle-to-gate"
    uncertainty_pct: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "material_name": self.material_name,
            "material_category": self.material_category,
            "co2e_factor": str(self.co2e_factor),
            "unit": self.unit,
            "source": self.source.value,
            "year": self.year,
            "region": self.region,
            "lifecycle_stage": self.lifecycle_stage,
            "uncertainty_pct": self.uncertainty_pct,
            "notes": self.notes
        }


@dataclass
class TransportEmissionFactor:
    """
    Emission factor for transportation activities.

    Supports both passenger travel and freight transport.
    """
    id: str
    mode: str  # road, rail, air, sea
    vehicle_type: str
    fuel_type: Optional[str] = None
    co2e_factor: Decimal = Decimal("0")
    unit: str = "kg CO2e/km"  # or kg CO2e/tonne-km for freight
    source: EmissionFactorSource = EmissionFactorSource.DEFRA
    year: int = 2024
    region: str = "global"
    load_factor: Optional[float] = None
    occupancy: Optional[float] = None
    well_to_tank_factor: Optional[Decimal] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "mode": self.mode,
            "vehicle_type": self.vehicle_type,
            "fuel_type": self.fuel_type,
            "co2e_factor": str(self.co2e_factor),
            "unit": self.unit,
            "source": self.source.value,
            "year": self.year,
            "region": self.region,
            "load_factor": self.load_factor,
            "occupancy": self.occupancy,
            "well_to_tank_factor": str(self.well_to_tank_factor) if self.well_to_tank_factor else None,
            "notes": self.notes
        }


@dataclass
class EmissionFactorVersion:
    """
    Version tracking for emission factor updates.

    Enables audit trails and version pinning for calculations.
    """
    factor_id: str
    version: str
    effective_date: datetime
    previous_value: Optional[Decimal] = None
    new_value: Decimal = Decimal("0")
    change_reason: Optional[str] = None
    changed_by: Optional[str] = None
    approved_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "factor_id": self.factor_id,
            "version": self.version,
            "effective_date": self.effective_date.isoformat(),
            "previous_value": str(self.previous_value) if self.previous_value else None,
            "new_value": str(self.new_value),
            "change_reason": self.change_reason,
            "changed_by": self.changed_by,
            "approved_by": self.approved_by
        }


# Standard GWP values from IPCC AR6
STANDARD_GWP_AR6_100YR = {
    "CO2": GWPValue("Carbon Dioxide", "CO2", Decimal("1"), Decimal("1"), GWPSet.AR6, None, "124-38-9"),
    "CH4": GWPValue("Methane", "CH4", Decimal("27.9"), Decimal("82.5"), GWPSet.AR6, 11.8, "74-82-8"),
    "N2O": GWPValue("Nitrous Oxide", "N2O", Decimal("273"), Decimal("273"), GWPSet.AR6, 109, "10024-97-2"),
    "SF6": GWPValue("Sulfur Hexafluoride", "SF6", Decimal("25200"), Decimal("18300"), GWPSet.AR6, 3200, "2551-62-4"),
    "NF3": GWPValue("Nitrogen Trifluoride", "NF3", Decimal("17400"), Decimal("13400"), GWPSet.AR6, 569, "7783-54-2"),
    "HFC-134a": GWPValue("HFC-134a", "CH2FCF3", Decimal("1530"), Decimal("4140"), GWPSet.AR6, 14, "811-97-2"),
    "HFC-32": GWPValue("HFC-32", "CH2F2", Decimal("771"), Decimal("2693"), GWPSet.AR6, 5.4, "75-10-5"),
    "R-410A": GWPValue("R-410A", "R-410A", Decimal("2256"), Decimal("4717"), GWPSet.AR6),
}

STANDARD_GWP_AR5_100YR = {
    "CO2": GWPValue("Carbon Dioxide", "CO2", Decimal("1"), Decimal("1"), GWPSet.AR5),
    "CH4": GWPValue("Methane", "CH4", Decimal("28"), Decimal("84"), GWPSet.AR5, 12.4, "74-82-8"),
    "N2O": GWPValue("Nitrous Oxide", "N2O", Decimal("265"), Decimal("264"), GWPSet.AR5, 121, "10024-97-2"),
    "SF6": GWPValue("Sulfur Hexafluoride", "SF6", Decimal("23500"), Decimal("17500"), GWPSet.AR5, 3200, "2551-62-4"),
}
