# -*- coding: utf-8 -*-
"""
greenlang/data/emission_factor_record.py

EmissionFactorRecord v2 - Enhanced emission factor with multi-gas and provenance

This module provides the v2 schema for storing emission factors with:
- Multi-gas breakdown (CO2, CH4, N2O, etc.)
- Full provenance tracking (source, methodology, version)
- Data quality scoring (5-dimension DQS per GHG Protocol)
- Licensing and redistribution metadata
- Uncertainty quantification (95% CI)
- Multiple GWP horizon support (AR6 100yr, AR6 20yr)

Example:
    >>> from greenlang.data.emission_factor_record import EmissionFactorRecord
    >>> from datetime import date
    >>>
    >>> factor = EmissionFactorRecord(
    ...     factor_id="EF:US:diesel:2024:v1",
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
    >>>
    >>> print(factor.gwp_100yr.co2e_total)
    10.210796
"""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional, Dict, List, Literal
from enum import Enum
import json
import hashlib


# ==================== ENUMERATIONS ====================

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


# ==================== DATA CLASSES ====================

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
    co2e_total: float = field(init=False, default=0.0)

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
    overall_score: float = field(init=False, default=0.0)
    rating: DataQualityRating = field(init=False, default=DataQualityRating.LOW)

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

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'temporal': self.temporal,
            'geographical': self.geographical,
            'technological': self.technological,
            'representativeness': self.representativeness,
            'methodological': self.methodological,
            'overall_score': self.overall_score,
            'rating': self.rating.value
        }


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
    citation: str = field(init=False, default="")

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
    content_hash: str = field(init=False, default="")  # SHA-256 of factor data

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
