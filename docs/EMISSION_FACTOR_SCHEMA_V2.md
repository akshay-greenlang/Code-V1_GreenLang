# EmissionFactorRecord Schema v2 - Design Specification

**Version:** 2.0.0
**Date:** 2025-10-24
**Owner:** GreenLang Framework Team
**Status:** Design Phase

---

## Executive Summary

This document defines the **EmissionFactorRecord v2 schema**, the core data structure for storing emission factors with multi-gas breakdown, full provenance tracking, data quality scoring, and regulatory compliance metadata.

**Key Enhancements from v1:**
- ✅ Multi-gas vectors (CO2, CH4, N2O separated)
- ✅ Full provenance (source, methodology, version, dates)
- ✅ Data quality scoring (5-dimension DQS)
- ✅ Licensing and redistribution metadata
- ✅ Uncertainty quantification (95% CI)
- ✅ Multiple GWP horizon support (AR6 100yr, AR6 20yr)
- ✅ Boundary specification (combustion, WTT, WTW)
- ✅ Technical parameters (HHV/LHV, temperature, biogenic flag)

---

## 1. Python Dataclass Definition

```python
"""
greenlang/data/emission_factor_record.py

EmissionFactorRecord v2 - Enhanced emission factor with multi-gas and provenance
"""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional, Dict, List, Literal
from enum import Enum
import json
import hashlib


class GeographyLevel(str, Enum):
    """Geographic specificity of the emission factor"""
    GLOBAL = "global"
    CONTINENT = "continent"
    COUNTRY = "country"
    STATE = "state"
    GRID_ZONE = "grid_zone"
    CITY = "city"
    FACILITY = "facility"


class Scope(str, Enum):
    """GHG Protocol emission scope"""
    SCOPE_1 = "1"  # Direct emissions
    SCOPE_2 = "2"  # Purchased electricity, steam, heating, cooling
    SCOPE_3 = "3"  # Indirect (value chain)


class Boundary(str, Enum):
    """Emission boundary / lifecycle stage"""
    COMBUSTION = "combustion"  # Direct combustion only (tank-to-wheel)
    WTT = "WTT"  # Well-to-tank (upstream)
    WTW = "WTW"  # Well-to-wheel (full lifecycle)
    CRADLE_TO_GATE = "cradle_to_gate"
    CRADLE_TO_GRAVE = "cradle_to_grave"


class Methodology(str, Enum):
    """Calculation methodology"""
    IPCC_TIER_1 = "IPCC_Tier_1"
    IPCC_TIER_2 = "IPCC_Tier_2"
    IPCC_TIER_3 = "IPCC_Tier_3"
    DIRECT_MEASUREMENT = "direct_measurement"
    LCA = "lifecycle_assessment"
    HYBRID = "hybrid"
    SPEND_BASED = "spend_based"


class GWPSet(str, Enum):
    """Global Warming Potential reference set"""
    IPCC_AR6_100 = "IPCC_AR6_100"  # Sixth Assessment, 100-year horizon
    IPCC_AR6_20 = "IPCC_AR6_20"    # Sixth Assessment, 20-year horizon
    IPCC_AR5_100 = "IPCC_AR5_100"  # Fifth Assessment, 100-year
    IPCC_SAR_100 = "IPCC_SAR_100"  # Second Assessment (legacy)


class HeatingValueBasis(str, Enum):
    """Heating value basis for fuels"""
    HHV = "HHV"  # Higher Heating Value (gross)
    LHV = "LHV"  # Lower Heating Value (net)


class DataQualityRating(str, Enum):
    """Overall data quality rating"""
    EXCELLENT = "excellent"      # DQS ≥ 4.5
    HIGH_QUALITY = "high_quality"  # DQS ≥ 4.0
    GOOD = "good"               # DQS ≥ 3.5
    MODERATE = "moderate"       # DQS ≥ 3.0
    LOW = "low"                 # DQS < 3.0


@dataclass
class GHGVectors:
    """Individual greenhouse gas emission vectors (kg per unit)"""

    # Primary gases (always present)
    CO2: float  # Carbon dioxide (kg/unit)
    CH4: float  # Methane (kg/unit)
    N2O: float  # Nitrous oxide (kg/unit)

    # Optional gases (for specific processes)
    HFCs: Optional[float] = None  # Hydrofluorocarbons
    PFCs: Optional[float] = None  # Perfluorocarbons
    SF6: Optional[float] = None   # Sulfur hexafluoride
    NF3: Optional[float] = None   # Nitrogen trifluoride

    # Biogenic carbon (reported separately per GHGP)
    biogenic_CO2: Optional[float] = None

    def __post_init__(self):
        """Validate non-negative values"""
        for field_name in ['CO2', 'CH4', 'N2O', 'HFCs', 'PFCs', 'SF6', 'NF3', 'biogenic_CO2']:
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")


@dataclass
class GWPValues:
    """Global Warming Potential values used for CO2e calculation"""

    gwp_set: GWPSet
    CH4_gwp: float  # e.g., 28 for AR6 100yr, 84 for AR6 20yr
    N2O_gwp: float  # e.g., 273 for AR6
    HFCs_gwp: Optional[float] = None
    PFCs_gwp: Optional[float] = None
    SF6_gwp: Optional[float] = None
    NF3_gwp: Optional[float] = None

    # Pre-calculated CO2-equivalent
    co2e_total: float = field(init=False)

    def calculate_co2e(self, vectors: GHGVectors) -> float:
        """Calculate total CO2e from vectors and GWP"""
        co2e = vectors.CO2  # CO2 has GWP = 1
        co2e += vectors.CH4 * self.CH4_gwp
        co2e += vectors.N2O * self.N2O_gwp

        if vectors.HFCs and self.HFCs_gwp:
            co2e += vectors.HFCs * self.HFCs_gwp
        if vectors.PFCs and self.PFCs_gwp:
            co2e += vectors.PFCs * self.PFCs_gwp
        if vectors.SF6 and self.SF6_gwp:
            co2e += vectors.SF6 * self.SF6_gwp
        if vectors.NF3 and self.NF3_gwp:
            co2e += vectors.NF3 * self.NF3_gwp

        return co2e


@dataclass
class DataQualityScore:
    """GHG Protocol 5-dimension data quality scoring (1-5 scale)"""

    # Each dimension scored 1 (lowest) to 5 (highest)
    temporal: int           # How recent is the data?
    geographical: int       # How geographically specific?
    technological: int      # How technology-specific?
    representativeness: int # How representative of actual conditions?
    methodological: int     # How rigorous is the methodology?

    # Calculated fields
    overall_score: float = field(init=False)
    rating: DataQualityRating = field(init=False)

    # Optional dimension weights (default: equal weighting)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'temporal': 0.2,
        'geographical': 0.2,
        'technological': 0.2,
        'representativeness': 0.2,
        'methodological': 0.2
    })

    def __post_init__(self):
        """Validate scores and calculate overall"""
        for dimension in ['temporal', 'geographical', 'technological',
                          'representativeness', 'methodological']:
            score = getattr(self, dimension)
            if not 1 <= score <= 5:
                raise ValueError(f"{dimension} score must be 1-5, got {score}")

        # Calculate weighted average
        self.overall_score = (
            self.temporal * self.weights['temporal'] +
            self.geographical * self.weights['geographical'] +
            self.technological * self.weights['technological'] +
            self.representativeness * self.weights['representativeness'] +
            self.methodological * self.weights['methodological']
        )

        # Assign rating
        if self.overall_score >= 4.5:
            self.rating = DataQualityRating.EXCELLENT
        elif self.overall_score >= 4.0:
            self.rating = DataQualityRating.HIGH_QUALITY
        elif self.overall_score >= 3.5:
            self.rating = DataQualityRating.GOOD
        elif self.overall_score >= 3.0:
            self.rating = DataQualityRating.MODERATE
        else:
            self.rating = DataQualityRating.LOW


@dataclass
class SourceProvenance:
    """Source attribution and provenance tracking"""

    source_org: str              # e.g., "EPA", "IEA", "IPCC"
    source_publication: str      # Full publication name
    source_year: int            # Publication year
    source_url: Optional[str] = None
    source_doi: Optional[str] = None

    methodology: Methodology
    methodology_description: Optional[str] = None

    # Version tracking
    version: str = "v1"         # Factor version (for same time period)
    supersedes: Optional[str] = None  # Previous factor_id if this is a correction

    # Citation
    citation: str = field(init=False)

    def __post_init__(self):
        """Generate citation"""
        self.citation = f"{self.source_org} ({self.source_year}). {self.source_publication}."
        if self.source_doi:
            self.citation += f" DOI: {self.source_doi}"
        elif self.source_url:
            self.citation += f" URL: {self.source_url}"


@dataclass
class LicenseInfo:
    """Licensing and redistribution metadata"""

    license: str                    # e.g., "CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0", "proprietary"
    redistribution_allowed: bool
    commercial_use_allowed: bool
    attribution_required: bool

    license_url: Optional[str] = None
    license_text: Optional[str] = None

    # Usage restrictions
    restrictions: List[str] = field(default_factory=list)
    # e.g., ["internal_use_only", "no_derivative_works", "share_alike"]


@dataclass
class EmissionFactorRecord:
    """
    Complete emission factor record with multi-gas, provenance, and quality metadata.

    This is the v2 schema supporting:
    - Multi-gas breakdown (CO2, CH4, N2O, etc.)
    - Multiple GWP horizons (AR6 100yr, 20yr)
    - Full source provenance
    - Data quality scoring
    - Licensing metadata
    - Regulatory compliance

    Example:
        >>> factor = EmissionFactorRecord(
        ...     factor_id="EF:US:diesel:2025:v1",
        ...     fuel_type="diesel",
        ...     unit="gallons",
        ...     geography="US",
        ...     geography_level=GeographyLevel.COUNTRY,
        ...     vectors=GHGVectors(CO2=10.18, CH4=0.00082, N2O=0.000164),
        ...     gwp_100yr=GWPValues(
        ...         gwp_set=GWPSet.IPCC_AR6_100,
        ...         CH4_gwp=28,
        ...         N2O_gwp=273
        ...     ),
        ...     scope=Scope.SCOPE_1,
        ...     boundary=Boundary.COMBUSTION,
        ...     provenance=SourceProvenance(
        ...         source_org="EPA",
        ...         source_publication="Emission Factors for GHG Inventories 2024",
        ...         source_year=2024,
        ...         methodology=Methodology.IPCC_TIER_1
        ...     ),
        ...     valid_from=date(2024, 1, 1),
        ...     uncertainty_95ci=0.05,
        ...     dqs=DataQualityScore(
        ...         temporal=5, geographical=4, technological=4,
        ...         representativeness=4, methodological=5
        ...     ),
        ...     license_info=LicenseInfo(
        ...         license="CC0-1.0",
        ...         redistribution_allowed=True,
        ...         commercial_use_allowed=True,
        ...         attribution_required=False
        ...     )
        ... )
    """

    # ==================== IDENTITY ====================
    factor_id: str                # Unique ID: "EF:{geo}:{fuel}:{year}:{version}"
    fuel_type: str               # Standardized fuel name (lowercase, underscores)
    unit: str                    # Emission factor unit denominator (e.g., "gallons", "kWh")

    # ==================== GEOGRAPHY ====================
    geography: str               # ISO country code (US, UK, IN) or region (EU, ASIA)
    geography_level: GeographyLevel
    region_hint: Optional[str] = None  # Sub-national (e.g., "CA" for California, "TX" for Texas)

    # ==================== EMISSION VECTORS ====================
    vectors: GHGVectors          # Individual gas quantities (kg/unit)

    # CO2-equivalent for different GWP horizons
    gwp_100yr: GWPValues         # Standard 100-year (required)
    gwp_20yr: Optional[GWPValues] = None  # Optional 20-year horizon

    # ==================== SCOPE & BOUNDARY ====================
    scope: Scope
    boundary: Boundary

    # ==================== PROVENANCE ====================
    provenance: SourceProvenance

    # ==================== VALIDITY ====================
    valid_from: date             # Start of validity period
    valid_to: Optional[date] = None  # End of validity (None = current/no expiry)

    # ==================== QUALITY ====================
    uncertainty_95ci: float      # Uncertainty as ±X (e.g., 0.05 = ±5%)
    dqs: DataQualityScore

    # ==================== LICENSING ====================
    license_info: LicenseInfo

    # ==================== TECHNICAL DETAILS ====================
    heating_value_basis: Optional[HeatingValueBasis] = None
    reference_temperature_c: Optional[float] = None  # For liquids
    pressure_bar: Optional[float] = None  # For gases
    moisture_content_pct: Optional[float] = None  # For biomass
    ash_content_pct: Optional[float] = None  # For solid fuels
    biogenic_flag: bool = False  # Is this a biogenic fuel?

    # ==================== COMPLIANCE MARKERS ====================
    # Regulatory frameworks this factor complies with
    compliance_frameworks: List[str] = field(default_factory=lambda: [
        "GHG_Protocol",  # GHG Protocol Corporate Standard
        "IPCC_2006",     # IPCC 2006 Guidelines
    ])

    # ==================== METADATA ====================
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "greenlang_system"
    notes: Optional[str] = None

    # Tags for searching/filtering
    tags: List[str] = field(default_factory=list)

    # ==================== CALCULATED FIELDS ====================
    content_hash: str = field(init=False)  # SHA-256 of factor data

    def __post_init__(self):
        """Post-initialization validation and calculations"""

        # Calculate CO2e totals
        self.gwp_100yr.co2e_total = self.gwp_100yr.calculate_co2e(self.vectors)
        if self.gwp_20yr:
            self.gwp_20yr.co2e_total = self.gwp_20yr.calculate_co2e(self.vectors)

        # Generate content hash
        self.content_hash = self._generate_hash()

        # Validate factor_id format
        if not self.factor_id.startswith("EF:"):
            raise ValueError(f"factor_id must start with 'EF:', got {self.factor_id}")

        # Validate uncertainty is positive
        if self.uncertainty_95ci < 0:
            raise ValueError(f"uncertainty_95ci must be non-negative, got {self.uncertainty_95ci}")

        # Validate date range
        if self.valid_to and self.valid_to < self.valid_from:
            raise ValueError(f"valid_to ({self.valid_to}) must be after valid_from ({self.valid_from})")

    def _generate_hash(self) -> str:
        """Generate SHA-256 hash of factor data for provenance"""
        # Include only the data fields, not metadata
        hash_data = {
            'factor_id': self.factor_id,
            'fuel_type': self.fuel_type,
            'unit': self.unit,
            'geography': self.geography,
            'vectors': asdict(self.vectors),
            'gwp_100yr': {
                'CH4_gwp': self.gwp_100yr.CH4_gwp,
                'N2O_gwp': self.gwp_100yr.N2O_gwp
            },
            'scope': self.scope.value,
            'boundary': self.boundary.value,
            'valid_from': self.valid_from.isoformat()
        }

        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON serialization)"""
        data = asdict(self)

        # Convert enums to values
        data['geography_level'] = self.geography_level.value
        data['scope'] = self.scope.value
        data['boundary'] = self.boundary.value
        data['gwp_100yr']['gwp_set'] = self.gwp_100yr.gwp_set.value
        if self.gwp_20yr:
            data['gwp_20yr']['gwp_set'] = self.gwp_20yr.gwp_set.value
        data['provenance']['methodology'] = self.provenance.methodology.value
        if self.heating_value_basis:
            data['heating_value_basis'] = self.heating_value_basis.value
        data['dqs']['rating'] = self.dqs.rating.value

        # Convert dates to ISO format
        data['valid_from'] = self.valid_from.isoformat()
        if self.valid_to:
            data['valid_to'] = self.valid_to.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()

        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> 'EmissionFactorRecord':
        """Create from dictionary"""
        # Convert string enums back to enum types
        data['geography_level'] = GeographyLevel(data['geography_level'])
        data['scope'] = Scope(data['scope'])
        data['boundary'] = Boundary(data['boundary'])

        # Convert nested objects
        data['vectors'] = GHGVectors(**data['vectors'])

        gwp_100_data = data['gwp_100yr']
        gwp_100_data['gwp_set'] = GWPSet(gwp_100_data['gwp_set'])
        data['gwp_100yr'] = GWPValues(**gwp_100_data)

        if data.get('gwp_20yr'):
            gwp_20_data = data['gwp_20yr']
            gwp_20_data['gwp_set'] = GWPSet(gwp_20_data['gwp_set'])
            data['gwp_20yr'] = GWPValues(**gwp_20_data)

        prov_data = data['provenance']
        prov_data['methodology'] = Methodology(prov_data['methodology'])
        data['provenance'] = SourceProvenance(**prov_data)

        dqs_data = data['dqs']
        data['dqs'] = DataQualityScore(**dqs_data)

        data['license_info'] = LicenseInfo(**data['license_info'])

        if data.get('heating_value_basis'):
            data['heating_value_basis'] = HeatingValueBasis(data['heating_value_basis'])

        # Convert date strings to date objects
        data['valid_from'] = date.fromisoformat(data['valid_from'])
        if data.get('valid_to'):
            data['valid_to'] = date.fromisoformat(data['valid_to'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'EmissionFactorRecord':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))

    def is_valid_on(self, check_date: date) -> bool:
        """Check if factor is valid on a given date"""
        if check_date < self.valid_from:
            return False
        if self.valid_to and check_date > self.valid_to:
            return False
        return True

    def is_redistributable(self) -> bool:
        """Check if this factor can be redistributed to customers"""
        return self.license_info.redistribution_allowed

    def get_co2e(self, gwp_horizon: str = "100yr") -> float:
        """Get CO2e value for specified GWP horizon"""
        if gwp_horizon == "100yr":
            return self.gwp_100yr.co2e_total
        elif gwp_horizon == "20yr":
            if not self.gwp_20yr:
                raise ValueError("20-year GWP not available for this factor")
            return self.gwp_20yr.co2e_total
        else:
            raise ValueError(f"Unknown GWP horizon: {gwp_horizon}")

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"EmissionFactorRecord(factor_id='{self.factor_id}', "
            f"fuel_type='{self.fuel_type}', "
            f"co2e_100yr={self.gwp_100yr.co2e_total:.4f} kg/unit, "
            f"dqs={self.dqs.overall_score:.2f})"
        )
```

---

## 2. Example Instances

### Example 1: US Diesel (EPA)
```python
from datetime import date

us_diesel = EmissionFactorRecord(
    factor_id="EF:US:diesel:2024:v1",
    fuel_type="diesel",
    unit="gallons",
    geography="US",
    geography_level=GeographyLevel.COUNTRY,

    vectors=GHGVectors(
        CO2=10.18,
        CH4=0.00082,
        N2O=0.000164
    ),

    gwp_100yr=GWPValues(
        gwp_set=GWPSet.IPCC_AR6_100,
        CH4_gwp=28,
        N2O_gwp=273
    ),

    gwp_20yr=GWPValues(
        gwp_set=GWPSet.IPCC_AR6_20,
        CH4_gwp=84,
        N2O_gwp=273
    ),

    scope=Scope.SCOPE_1,
    boundary=Boundary.COMBUSTION,

    provenance=SourceProvenance(
        source_org="EPA",
        source_publication="Emission Factors for Greenhouse Gas Inventories",
        source_year=2024,
        source_url="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
        methodology=Methodology.IPCC_TIER_1,
        version="v1"
    ),

    valid_from=date(2024, 1, 1),
    valid_to=date(2024, 12, 31),

    uncertainty_95ci=0.05,  # ±5%

    dqs=DataQualityScore(
        temporal=5,           # Current year data
        geographical=4,       # Country-level (not state-specific)
        technological=4,      # Fuel-specific, not equipment
        representativeness=4, # Representative US average
        methodological=5      # IPCC Tier 1 (peer-reviewed)
    ),

    license_info=LicenseInfo(
        license="CC0-1.0",
        redistribution_allowed=True,
        commercial_use_allowed=True,
        attribution_required=False,
        license_url="https://creativecommons.org/publicdomain/zero/1.0/"
    ),

    heating_value_basis=HeatingValueBasis.HHV,
    reference_temperature_c=15.0,

    compliance_frameworks=["GHG_Protocol", "IPCC_2006", "EPA_MRR"],

    tags=["fossil", "transport", "stationary", "tier1"]
)

# Access data
print(us_diesel)
# EmissionFactorRecord(factor_id='EF:US:diesel:2024:v1', fuel_type='diesel', co2e_100yr=10.2101 kg/unit, dqs=4.40)

print(f"CO2e (100yr): {us_diesel.get_co2e('100yr')} kg/gallon")
# CO2e (100yr): 10.2101 kg/gallon

print(f"CO2e (20yr): {us_diesel.get_co2e('20yr')} kg/gallon")
# CO2e (20yr): 10.2569 kg/gallon (higher due to CH4 GWP=84)

print(f"Citation: {us_diesel.provenance.citation}")
# Citation: EPA (2024). Emission Factors for Greenhouse Gas Inventories. URL: https://...

print(f"Redistributable: {us_diesel.is_redistributable()}")
# Redistributable: True
```

### Example 2: India Grid Electricity
```python
india_electricity = EmissionFactorRecord(
    factor_id="EF:IN:electricity:2024:v1",
    fuel_type="electricity",
    unit="kWh",
    geography="IN",
    geography_level=GeographyLevel.COUNTRY,

    vectors=GHGVectors(
        CO2=0.708,   # India grid is coal-heavy
        CH4=0.00015,
        N2O=0.000008
    ),

    gwp_100yr=GWPValues(
        gwp_set=GWPSet.IPCC_AR6_100,
        CH4_gwp=28,
        N2O_gwp=273
    ),

    scope=Scope.SCOPE_2,
    boundary=Boundary.COMBUSTION,

    provenance=SourceProvenance(
        source_org="IEA",
        source_publication="Emissions Factors 2024",
        source_year=2024,
        source_url="https://www.iea.org/data-and-statistics",
        methodology=Methodology.IPCC_TIER_1,
        version="v1"
    ),

    valid_from=date(2024, 1, 1),

    uncertainty_95ci=0.12,  # ±12% (grid mix varies)

    dqs=DataQualityScore(
        temporal=4,           # 2024 data (recent)
        geographical=3,       # Country-level (not state-specific)
        technological=3,      # Grid average (not plant-specific)
        representativeness=4, # Representative of national grid
        methodological=4      # IEA standard methodology
    ),

    license_info=LicenseInfo(
        license="CC-BY-4.0",
        redistribution_allowed=True,
        commercial_use_allowed=True,
        attribution_required=True,
        license_url="https://creativecommons.org/licenses/by/4.0/"
    ),

    compliance_frameworks=["GHG_Protocol", "IPCC_2006", "CSRD"],

    tags=["electricity", "grid", "scope2", "india"],

    notes="India grid emission factor based on 2024 fuel mix. State-specific factors available on request."
)
```

### Example 3: UK Natural Gas with WTT
```python
uk_natural_gas_wtt = EmissionFactorRecord(
    factor_id="EF:UK:natural_gas:WTT:2024:v1",
    fuel_type="natural_gas",
    unit="kWh",
    geography="UK",
    geography_level=GeographyLevel.COUNTRY,

    vectors=GHGVectors(
        CO2=0.0232,   # WTT only (extraction, processing, transport)
        CH4=0.00045,  # Higher CH4 from fugitive emissions
        N2O=0.000001
    ),

    gwp_100yr=GWPValues(
        gwp_set=GWPSet.IPCC_AR6_100,
        CH4_gwp=28,
        N2O_gwp=273
    ),

    scope=Scope.SCOPE_3,
    boundary=Boundary.WTT,

    provenance=SourceProvenance(
        source_org="UK DESNZ",
        source_publication="UK Government GHG Conversion Factors for Company Reporting 2024",
        source_year=2024,
        source_url="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
        methodology=Methodology.LCA,
        methodology_description="Lifecycle assessment of natural gas supply chain",
        version="v1"
    ),

    valid_from=date(2024, 6, 1),
    valid_to=date(2025, 5, 31),

    uncertainty_95ci=0.25,  # ±25% (supply chain variability)

    dqs=DataQualityScore(
        temporal=5,           # Current year
        geographical=4,       # UK-specific
        technological=3,      # Generic supply chain
        representativeness=3, # UK average (varies by source)
        methodological=4      # Defra LCA methodology
    ),

    license_info=LicenseInfo(
        license="OGL-UK-3.0",
        redistribution_allowed=True,
        commercial_use_allowed=True,
        attribution_required=True,
        license_url="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
    ),

    compliance_frameworks=["GHG_Protocol_Scope3", "ISO14064", "SECR"],

    tags=["natural_gas", "wtt", "scope3", "upstream"],

    notes="Well-to-tank emissions only. Add combustion factor for total (WTW)."
)
```

---

## 3. JSON Schema (for validation)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "EmissionFactorRecord",
  "description": "Enhanced emission factor with multi-gas and provenance (v2)",
  "type": "object",
  "required": [
    "factor_id", "fuel_type", "unit", "geography", "geography_level",
    "vectors", "gwp_100yr", "scope", "boundary", "provenance",
    "valid_from", "uncertainty_95ci", "dqs", "license_info"
  ],
  "properties": {
    "factor_id": {
      "type": "string",
      "pattern": "^EF:[A-Z]{2,10}:[a-z_]+:[0-9]{4}:v[0-9]+$",
      "description": "Unique factor ID in format EF:GEO:FUEL:YEAR:VERSION"
    },
    "fuel_type": {
      "type": "string",
      "pattern": "^[a-z_]+$",
      "description": "Standardized fuel name (lowercase, underscores)"
    },
    "unit": {
      "type": "string",
      "description": "Emission factor unit denominator"
    },
    "geography": {
      "type": "string",
      "description": "ISO country code or region"
    },
    "geography_level": {
      "type": "string",
      "enum": ["global", "continent", "country", "state", "grid_zone", "city", "facility"]
    },
    "region_hint": {
      "type": "string",
      "description": "Sub-national region (optional)"
    },
    "vectors": {
      "type": "object",
      "required": ["CO2", "CH4", "N2O"],
      "properties": {
        "CO2": {"type": "number", "minimum": 0},
        "CH4": {"type": "number", "minimum": 0},
        "N2O": {"type": "number", "minimum": 0},
        "HFCs": {"type": "number", "minimum": 0},
        "PFCs": {"type": "number", "minimum": 0},
        "SF6": {"type": "number", "minimum": 0},
        "NF3": {"type": "number", "minimum": 0},
        "biogenic_CO2": {"type": "number", "minimum": 0}
      }
    },
    "gwp_100yr": {
      "type": "object",
      "required": ["gwp_set", "CH4_gwp", "N2O_gwp"],
      "properties": {
        "gwp_set": {
          "type": "string",
          "enum": ["IPCC_AR6_100", "IPCC_AR6_20", "IPCC_AR5_100", "IPCC_SAR_100"]
        },
        "CH4_gwp": {"type": "number", "minimum": 0},
        "N2O_gwp": {"type": "number", "minimum": 0}
      }
    },
    "scope": {
      "type": "string",
      "enum": ["1", "2", "3"]
    },
    "boundary": {
      "type": "string",
      "enum": ["combustion", "WTT", "WTW", "cradle_to_gate", "cradle_to_grave"]
    },
    "provenance": {
      "type": "object",
      "required": ["source_org", "source_publication", "source_year", "methodology", "version"],
      "properties": {
        "source_org": {"type": "string"},
        "source_publication": {"type": "string"},
        "source_year": {"type": "integer", "minimum": 1990, "maximum": 2100},
        "source_url": {"type": "string", "format": "uri"},
        "source_doi": {"type": "string"},
        "methodology": {
          "type": "string",
          "enum": ["IPCC_Tier_1", "IPCC_Tier_2", "IPCC_Tier_3", "direct_measurement", "lifecycle_assessment", "hybrid", "spend_based"]
        },
        "methodology_description": {"type": "string"},
        "version": {"type": "string", "pattern": "^v[0-9]+$"}
      }
    },
    "valid_from": {
      "type": "string",
      "format": "date"
    },
    "valid_to": {
      "type": "string",
      "format": "date"
    },
    "uncertainty_95ci": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Uncertainty as decimal (0.05 = ±5%)"
    },
    "dqs": {
      "type": "object",
      "required": ["temporal", "geographical", "technological", "representativeness", "methodological"],
      "properties": {
        "temporal": {"type": "integer", "minimum": 1, "maximum": 5},
        "geographical": {"type": "integer", "minimum": 1, "maximum": 5},
        "technological": {"type": "integer", "minimum": 1, "maximum": 5},
        "representativeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "methodological": {"type": "integer", "minimum": 1, "maximum": 5}
      }
    },
    "license_info": {
      "type": "object",
      "required": ["license", "redistribution_allowed", "commercial_use_allowed", "attribution_required"],
      "properties": {
        "license": {"type": "string"},
        "redistribution_allowed": {"type": "boolean"},
        "commercial_use_allowed": {"type": "boolean"},
        "attribution_required": {"type": "boolean"},
        "license_url": {"type": "string", "format": "uri"},
        "restrictions": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    },
    "compliance_frameworks": {
      "type": "array",
      "items": {"type": "string"}
    },
    "tags": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
```

---

## 4. Storage Formats

### Option A: JSON Files (Simple, Version Control Friendly)
```
factors/
├── US/
│   ├── diesel_2024_v1.json
│   ├── natural_gas_2024_v1.json
│   └── electricity_2024_v1.json
├── EU/
│   └── ...
└── index.json  # Searchable index
```

### Option B: SQLite Database (Fast Queries)
```sql
CREATE TABLE emission_factors (
    factor_id TEXT PRIMARY KEY,
    fuel_type TEXT NOT NULL,
    unit TEXT NOT NULL,
    geography TEXT NOT NULL,
    geography_level TEXT NOT NULL,

    -- Store nested objects as JSON
    vectors JSON NOT NULL,
    gwp_100yr JSON NOT NULL,
    gwp_20yr JSON,

    scope TEXT NOT NULL,
    boundary TEXT NOT NULL,

    provenance JSON NOT NULL,

    valid_from DATE NOT NULL,
    valid_to DATE,

    uncertainty_95ci REAL NOT NULL,
    dqs JSON NOT NULL,

    license_info JSON NOT NULL,

    -- Full record as JSON
    record_json JSON NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_fuel_geo ON emission_factors(fuel_type, geography);
CREATE INDEX idx_valid_dates ON emission_factors(valid_from, valid_to);
CREATE INDEX idx_scope ON emission_factors(scope);
```

### Option C: PostgreSQL (Enterprise, Audit Trail)
Same as SQLite but add:
- Version history table
- Audit log table
- Row-level security
- JSONB columns for efficient querying

---

## 5. Migration from v1 to v2

```python
def migrate_v1_to_v2(v1_factor: Dict) -> EmissionFactorRecord:
    """
    Migrate v1 emission factor (single CO2e value) to v2 (multi-gas).

    Strategy:
    1. Use CO2e value as total
    2. Apply IPCC default CH4/N2O ratios to decompose
    3. Mark with low DQS (estimated, not measured)
    4. Add notes about migration
    """

    # v1 format: {"fuel_type": "diesel", "kWh": 10.21}
    # We need to estimate CO2, CH4, N2O from CO2e

    co2e_total = v1_factor["value"]  # e.g., 10.21 kg/gallon

    # IPCC default ratios for diesel combustion
    # Typical: CO2 ~99.6%, CH4 ~0.008%, N2O ~0.016% of CO2e
    CH4_gwp = 28  # AR6 100yr
    N2O_gwp = 273

    # Reverse calculate (approximation)
    # co2e = CO2 + CH4*28 + N2O*273
    # Assume CH4 = 0.00008 * CO2, N2O = 0.000016 * CO2
    # co2e = CO2 * (1 + 0.00008*28 + 0.000016*273) ≈ CO2 * 1.0066
    CO2_estimate = co2e_total / 1.0066
    CH4_estimate = CO2_estimate * 0.00008
    N2O_estimate = CO2_estimate * 0.000016

    return EmissionFactorRecord(
        factor_id=f"EF:MIGRATED:{v1_factor['fuel_type']}:2024:v1",
        fuel_type=v1_factor["fuel_type"],
        unit=v1_factor["unit"],
        geography="UNKNOWN",
        geography_level=GeographyLevel.GLOBAL,

        vectors=GHGVectors(
            CO2=CO2_estimate,
            CH4=CH4_estimate,
            N2O=N2O_estimate
        ),

        gwp_100yr=GWPValues(
            gwp_set=GWPSet.IPCC_AR6_100,
            CH4_gwp=28,
            N2O_gwp=273
        ),

        scope=Scope.SCOPE_1,  # Assume Scope 1
        boundary=Boundary.COMBUSTION,

        provenance=SourceProvenance(
            source_org="LEGACY",
            source_publication="Migrated from v1 factors",
            source_year=2024,
            methodology=Methodology.IPCC_TIER_1,
            version="v1"
        ),

        valid_from=date(2020, 1, 1),

        uncertainty_95ci=0.25,  # High uncertainty (estimated)

        dqs=DataQualityScore(
            temporal=2,           # Unknown age
            geographical=1,       # Unknown geography
            technological=2,      # Generic
            representativeness=2, # Unknown
            methodological=2      # Estimated, not measured
        ),

        license_info=LicenseInfo(
            license="proprietary",
            redistribution_allowed=False,
            commercial_use_allowed=True,
            attribution_required=False
        ),

        notes="MIGRATED FROM V1: Multi-gas breakdown estimated using IPCC default ratios. Replace with measured factors when available."
    )
```

---

## 6. Next Steps

1. ✅ **Implement dataclass** (this document)
2. **Create test suite** (test_emission_factor_record.py)
3. **Implement database layer** (EmissionFactorDatabase class)
4. **Build migration script** (migrate_v1_to_v2.py)
5. **Integrate with FuelAgentAI** (update lookup_emission_factor tool)
6. **Populate with real data** (EPA, IPCC, BEIS)

---

**Document Status:** READY FOR REVIEW
**Estimated Implementation:** 2-3 days
**Dependencies:** None (foundational schema)
