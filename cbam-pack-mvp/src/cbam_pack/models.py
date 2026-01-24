"""
CBAM Pack Data Models

Pydantic models for all data entities as specified in PRD Section 10.
Uses Decimal for all numeric values to ensure deterministic calculations.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================

class Quarter(str, Enum):
    """Reporting quarter."""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class Unit(str, Enum):
    """Supported units of measurement."""
    KG = "kg"
    TONNES = "tonnes"


class MethodType(str, Enum):
    """Emission calculation method type."""
    SUPPLIER_SPECIFIC = "supplier_specific"
    DEFAULT = "default"


class ClaimType(str, Enum):
    """Type of claim in the audit bundle."""
    EMISSION = "emission"
    QUANTITY = "quantity"
    INTENSITY = "intensity"
    TOTAL_EMISSIONS = "total_emissions"
    DIRECT_EMISSIONS = "direct_emissions"
    INDIRECT_EMISSIONS = "indirect_emissions"


class AssumptionType(str, Enum):
    """Type of assumption made during calculation."""
    DEFAULT_FACTOR = "default_factor"
    MISSING_DATA_FILL = "missing_data_fill"
    UNIT_INFERENCE = "unit_inference"
    METHOD_SELECTION = "method_selection"


class FactorPolicy(str, Enum):
    """Emission factor selection policy."""
    DEFAULTS_FIRST = "defaults_first"
    SUPPLIER_REQUIRED = "supplier_required"


class AggregationPolicy(str, Enum):
    """Aggregation policy for multiple installations."""
    BY_PRODUCT_ORIGIN = "by_product_origin"
    PRESERVE_DETAIL = "preserve_detail"


class ValidationStrictness(str, Enum):
    """Validation strictness level."""
    STRICT = "strict"
    LENIENT = "lenient"


# =============================================================================
# ISO 3166-1 Alpha-2 Country Codes (subset for CBAM)
# =============================================================================

VALID_COUNTRY_CODES = {
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", "AU",
    "AW", "AX", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BL",
    "BM", "BN", "BO", "BQ", "BR", "BS", "BT", "BV", "BW", "BY", "BZ", "CA", "CC",
    "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN", "CO", "CR", "CU", "CV",
    "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE", "EG",
    "EH", "ER", "ES", "ET", "FI", "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD",
    "GE", "GF", "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GS", "GT",
    "GU", "GW", "GY", "HK", "HM", "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM",
    "IN", "IO", "IQ", "IR", "IS", "IT", "JE", "JM", "JO", "JP", "KE", "KG", "KH",
    "KI", "KM", "KN", "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK",
    "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH",
    "MK", "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW",
    "MX", "MY", "MZ", "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR",
    "NU", "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM", "PN", "PR",
    "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW", "SA", "SB", "SC",
    "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS",
    "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK", "TL",
    "TM", "TN", "TO", "TR", "TT", "TV", "TW", "TZ", "UA", "UG", "UM", "US", "UY",
    "UZ", "VA", "VC", "VE", "VG", "VI", "VN", "VU", "WF", "WS", "YE", "YT", "ZA",
    "ZM", "ZW",
}


# =============================================================================
# Input Models
# =============================================================================

class ImportLineItem(BaseModel):
    """
    A single line item from the import ledger.

    Represents one import transaction of CBAM-regulated goods.
    """
    line_id: str = Field(..., description="Unique identifier for the line")
    quarter: Quarter = Field(..., description="Reporting quarter")
    year: int = Field(..., ge=2023, le=2030, description="Reporting year")
    cn_code: str = Field(..., description="8-digit Combined Nomenclature code")
    product_description: str = Field(..., min_length=1, description="Description of goods")
    country_of_origin: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    quantity: Decimal = Field(..., gt=0, description="Amount of goods imported")
    unit: Unit = Field(..., description="Unit of measurement (kg or tonnes)")

    # Optional supplier-specific data
    supplier_id: Optional[str] = Field(None, description="Supplier identifier")
    installation_id: Optional[str] = Field(None, description="Production installation ID")
    supplier_direct_emissions: Optional[Decimal] = Field(
        None, ge=0, description="Supplier-provided direct emission intensity (tCO2e/tonne)"
    )
    supplier_indirect_emissions: Optional[Decimal] = Field(
        None, ge=0, description="Supplier-provided indirect emission intensity (tCO2e/tonne)"
    )
    supplier_certificate_ref: Optional[str] = Field(
        None, description="Reference to supplier certificate"
    )

    @field_validator("cn_code")
    @classmethod
    def validate_cn_code(cls, v: str) -> str:
        """Validate CN code is 8 digits and supported (72xx, 73xx, 76xx)."""
        v = v.strip()
        if len(v) != 8:
            raise ValueError(f"CN code must be 8 digits, got {len(v)}")
        if not v.isdigit():
            raise ValueError("CN code must contain only digits")
        prefix = v[:2]
        if prefix not in ("72", "73", "76"):
            raise ValueError(
                f"CN code must start with 72, 73, or 76 for Steel/Aluminum. Got: {prefix}"
            )
        return v

    @field_validator("country_of_origin")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate ISO 3166-1 alpha-2 country code."""
        v = v.strip().upper()
        if len(v) != 2:
            raise ValueError("Country code must be 2 characters")
        if v not in VALID_COUNTRY_CODES:
            raise ValueError(f"Invalid ISO 3166-1 alpha-2 country code: {v}")
        return v

    model_config = {
        "json_encoders": {Decimal: str},
        "extra": "forbid",
    }


# =============================================================================
# Configuration Models
# =============================================================================

class Address(BaseModel):
    """Postal address."""
    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    postal_code: str = Field(..., min_length=1)
    country: str = Field(..., min_length=2, max_length=2)

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in VALID_COUNTRY_CODES:
            raise ValueError(f"Invalid country code: {v}")
        return v


class Contact(BaseModel):
    """Contact information."""
    name: str = Field(..., min_length=1)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    phone: Optional[str] = None


class Declarant(BaseModel):
    """CBAM declarant (importer) information."""
    name: str = Field(..., min_length=1)
    eori_number: str = Field(..., min_length=15, max_length=17)
    address: Address
    contact: Contact


class Representative(BaseModel):
    """Authorized customs representative."""
    name: str = Field(..., min_length=1)
    eori_number: str = Field(..., min_length=15, max_length=17)


class ReportingPeriod(BaseModel):
    """CBAM reporting period."""
    quarter: Quarter
    year: int = Field(..., ge=2023, le=2030)


class Settings(BaseModel):
    """CBAM pack settings."""
    factor_policy: FactorPolicy = FactorPolicy.DEFAULTS_FIRST
    aggregation: AggregationPolicy = AggregationPolicy.BY_PRODUCT_ORIGIN
    xml_schema_version: str = "2.0.0"
    validation_strictness: ValidationStrictness = ValidationStrictness.STRICT


class CBAMConfig(BaseModel):
    """Complete CBAM pack configuration."""
    version: str = "1.0"
    declarant: Declarant
    reporting_period: ReportingPeriod
    representative: Optional[Representative] = None
    settings: Settings = Field(default_factory=Settings)

    model_config = {"extra": "forbid"}


# =============================================================================
# Result Models
# =============================================================================

class EmissionsResult(BaseModel):
    """Emissions calculation result for a single line item."""
    line_id: str
    direct_emissions_tco2e: Decimal = Field(..., description="Direct emissions in tCO2e")
    indirect_emissions_tco2e: Decimal = Field(..., description="Indirect emissions in tCO2e")
    total_emissions_tco2e: Decimal = Field(..., description="Total emissions in tCO2e")
    emissions_intensity: Decimal = Field(..., description="Emission intensity (tCO2e/tonne)")
    method_direct: MethodType = Field(..., description="Method used for direct emissions")
    method_indirect: MethodType = Field(..., description="Method used for indirect emissions")
    factor_direct_ref: str = Field(..., description="Direct emission factor ID")
    factor_indirect_ref: str = Field(..., description="Indirect emission factor ID")
    uncertainty_range: Optional[dict] = None

    model_config = {"json_encoders": {Decimal: str}}


class AggregatedResult(BaseModel):
    """Aggregated emissions by CN code and country."""
    cn_code: str
    country_of_origin: str
    total_quantity_tonnes: Decimal
    total_direct_emissions_tco2e: Decimal
    total_indirect_emissions_tco2e: Decimal
    total_emissions_tco2e: Decimal
    weighted_intensity: Decimal
    line_count: int
    method_used: str  # "default_values" or "supplier_specific" or "mixed"

    model_config = {"json_encoders": {Decimal: str}}


# =============================================================================
# Audit Bundle Models
# =============================================================================

class EvidenceRef(BaseModel):
    """Reference to evidence file."""
    evidence_id: UUID = Field(default_factory=uuid4)
    file_id: str
    file_hash: str = Field(..., description="SHA-256 hash of file")
    file_name: str
    row_pointer: Optional[int] = None
    field_pointer: Optional[str] = None
    snippet: Optional[str] = None


class Claim(BaseModel):
    """A claim about an emission value with full provenance."""
    claim_id: UUID = Field(default_factory=uuid4)
    value: float  # Changed to float for JSON serialization
    unit: str
    claim_type: ClaimType
    method: Optional[str] = None
    confidence: Optional[str] = None  # HIGH, MEDIUM, LOW
    methodology: Optional[str] = None
    period: Optional[str] = None
    scope: Optional[str] = None
    evidence_refs: list[str] = Field(default_factory=list)
    factor_ref: Optional[str] = None
    transformation_refs: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    applies_to: Optional[str] = None  # line_id

    model_config = {
        "json_encoders": {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }
    }


class Assumption(BaseModel):
    """An assumption made during calculation."""
    assumption_id: UUID = Field(default_factory=uuid4)
    type: AssumptionType
    description: str
    rationale: str
    applies_to: list[str] = Field(default_factory=list)  # line_ids or claim_ids
    factor_ref: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }
    }


class LineageNode(BaseModel):
    """Node in the lineage graph."""
    id: str
    type: str  # "input", "transformation", "output"
    file: Optional[str] = None
    hash: Optional[str] = None
    agent: Optional[str] = None
    operation: Optional[str] = None


class LineageEdge(BaseModel):
    """Edge in the lineage graph."""
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")

    model_config = {"populate_by_name": True}


class LineageGraph(BaseModel):
    """Complete lineage graph showing data flow."""
    nodes: list[LineageNode] = Field(default_factory=list)
    edges: list[LineageEdge] = Field(default_factory=list)


class GapItem(BaseModel):
    """A single gap identified in data quality."""
    gap_id: str
    gap_type: str
    applies_to: list[str]
    supplier_id: Optional[str] = None
    description: str
    recommended_action: str


class GapReport(BaseModel):
    """Report of data quality gaps and improvement opportunities."""
    summary: dict
    gaps: list[GapItem] = Field(default_factory=list)


class InputFileRef(BaseModel):
    """Reference to an input file with hash."""
    file_name: str
    file_hash: str


class OutputFileRef(BaseModel):
    """Reference to an output file with hash."""
    file_name: str
    file_hash: str


class RuntimeInfo(BaseModel):
    """Runtime environment information."""
    greenlang_version: str
    python_version: str
    platform: str


class RunManifest(BaseModel):
    """Complete run manifest for reproducibility."""
    run_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pack_name: str = "cbam"
    pack_version: str = "1.0.0"
    factor_library_version: str
    xml_schema_version: str
    config_hash: str
    input_files: list[InputFileRef] = Field(default_factory=list)
    output_files: list[OutputFileRef] = Field(default_factory=list)
    runtime: RuntimeInfo
    agents_executed: list[str] = Field(default_factory=list)
    execution_time_seconds: Optional[float] = None
    statistics: Optional[dict] = None

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }
    }
