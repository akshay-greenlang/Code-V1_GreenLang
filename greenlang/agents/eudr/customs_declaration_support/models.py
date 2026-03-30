# -*- coding: utf-8 -*-
"""
Customs Declaration Support Models - AGENT-EUDR-039

Pydantic v2 models for customs declaration lifecycle management including
CN code mapping, HS code validation, declaration generation (SAD format),
country of origin verification, customs value calculation, EUDR compliance
checking, and customs authority submission interfaces.

All models use Decimal for monetary values and tariff rates to ensure
deterministic, bit-perfect reproducibility in customs calculations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-039 Customs Declaration Support (GL-EUDR-CDS-039)
Regulation: EU 2023/1115 (EUDR) Articles 4, 5, 6, 12, 31;
            EU UCC 952/2013; EU CN Regulation 2658/87
Status: Production Ready
"""
from __future__ import annotations

import enum
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from greenlang.schemas import GreenLangBase


# ---------------------------------------------------------------------------
# Enums (12)
# ---------------------------------------------------------------------------


class DeclarationStatus(str, enum.Enum):
    """Lifecycle status of a customs declaration."""

    PENDING = "pending"
    DRAFT = "draft"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    UNDER_REVIEW = "under_review"
    CLEARED = "cleared"
    RELEASED = "released"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    AMENDED = "amended"


class DeclarationType(str, enum.Enum):
    """Type of customs declaration per UCC."""

    IMPORT = "import"
    EXPORT = "export"
    TRANSIT = "transit"
    RE_EXPORT = "re_export"
    TEMPORARY_ADMISSION = "temporary_admission"
    INWARD_PROCESSING = "inward_processing"


class CommodityType(str, enum.Enum):
    """Seven EUDR-regulated commodities per Article 1(1) and Annex I."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class CustomsSystem(str, enum.Enum):
    """Supported EU customs IT systems (internal uppercase values)."""

    NCTS = "NCTS"    # New Computerised Transit System
    AIS = "AIS"      # Automated Import System
    ICS2 = "ICS2"    # Import Control System 2


class CustomsSystemType(str, enum.Enum):
    """Supported EU customs IT systems (lowercase values for API)."""

    NCTS = "ncts"
    AIS = "ais"
    ICS2 = "ics2"


class Incoterms(str, enum.Enum):
    """ICC Incoterms 2020 for customs valuation."""

    EXW = "EXW"   # Ex Works
    FCA = "FCA"   # Free Carrier
    FAS = "FAS"   # Free Alongside Ship
    FOB = "FOB"   # Free on Board
    CFR = "CFR"   # Cost and Freight
    CIF = "CIF"   # Cost, Insurance, Freight
    CPT = "CPT"   # Carriage Paid To
    CIP = "CIP"   # Carriage and Insurance Paid To
    DAP = "DAP"   # Delivered at Place
    DPU = "DPU"   # Delivered at Place Unloaded
    DDP = "DDP"   # Delivered Duty Paid


class TariffType(str, enum.Enum):
    """Types of tariff duty applied."""

    CONVENTIONAL = "conventional"
    PREFERENTIAL = "preferential"
    AUTONOMOUS = "autonomous"
    ANTI_DUMPING = "anti_dumping"
    COUNTERVAILING = "countervailing"
    ADDITIONAL = "additional"
    ZERO_RATE = "zero_rate"


class ComplianceCheckType(str, enum.Enum):
    """Types of EUDR compliance checks for customs."""

    DDS_REFERENCE = "dds_reference"
    DEFORESTATION_FREE = "deforestation_free"
    LEGALITY = "legality"
    GEOLOCATION = "geolocation"
    SUPPLY_CHAIN = "supply_chain"
    RISK_ASSESSMENT = "risk_assessment"
    COUNTRY_BENCHMARKING = "country_benchmarking"


class VerificationStatus(str, enum.Enum):
    """Status of a verification check."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


class ComplianceStatus(str, enum.Enum):
    """Compliance status for EUDR customs declarations."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    PARTIALLY_COMPLIANT = "partially_compliant"


class PortType(str, enum.Enum):
    """Type of port of entry."""

    SEA = "sea"
    AIR = "air"
    LAND = "land"
    RAIL = "rail"
    INLAND_WATERWAY = "inland_waterway"


class DeclarationPurpose(str, enum.Enum):
    """Purpose of the customs declaration."""

    FREE_CIRCULATION = "free_circulation"
    CUSTOMS_WAREHOUSING = "customs_warehousing"
    TRANSIT = "transit"
    TEMPORARY_STORAGE = "temporary_storage"
    OUTWARD_PROCESSING = "outward_processing"
    INWARD_PROCESSING = "inward_processing"
    END_USE = "end_use"


class SubmissionStatus(str, enum.Enum):
    """Status of declaration submission to customs authority."""

    QUEUED = "queued"
    TRANSMITTING = "transmitting"
    TRANSMITTED = "transmitted"
    ACKNOWLEDGED = "acknowledged"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"
    TIMEOUT = "timeout"


class AuditAction(str, enum.Enum):
    """Audit trail action types for customs declaration events."""

    CREATE = "create"
    UPDATE = "update"
    VALIDATE = "validate"
    SUBMIT = "submit"
    ACCEPT = "accept"
    REJECT = "reject"
    CLEAR = "clear"
    RELEASE = "release"
    AMEND = "amend"
    CANCEL = "cancel"
    MAP_CN_CODE = "map_cn_code"
    CALCULATE_VALUE = "calculate_value"
    CHECK_COMPLIANCE = "check_compliance"
    VERIFY_ORIGIN = "verify_origin"


class MRNStatus(str, enum.Enum):
    """Status of Movement Reference Number (MRN) assignment."""

    GENERATED = "generated"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NOT_ASSIGNED = "not_assigned"
    ASSIGNED = "assigned"
    VALIDATED = "validated"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class RiskLevel(str, enum.Enum):
    """Risk level classification for customs declarations."""

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class OriginVerificationResult(str, enum.Enum):
    """Result of origin verification check."""

    VERIFIED = "verified"
    MISMATCH = "mismatch"
    UNVERIFIED = "unverified"
    NOT_VERIFIED = "not_verified"
    DISCREPANCY = "discrepancy"
    PENDING = "pending"
    INSUFFICIENT_DATA = "insufficient_data"


class ValidationResult(str, enum.Enum):
    """Result of a validation / compliance check."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-CDS-039"
AGENT_VERSION = "1.0.0"

# EUDR Annex I - 7 regulated commodities with representative CN codes
# These are the primary 8-digit CN codes; actual mappings are more extensive
EUDR_COMMODITY_CN_CODES: Dict[str, List[str]] = {
    "cattle": [
        "01022110",  # Live pure-bred breeding cattle
        "01022190",  # Other live cattle, <80 kg
        "01022921",  # Live cattle for slaughter, 80-160 kg
        "01022929",  # Other live cattle, 80-160 kg
        "01022941",  # Live cattle for slaughter, 160-300 kg
        "02011000",  # Carcasses and half-carcasses of bovine
        "02012020",  # Bone-in cuts of bovine
        "02013000",  # Boneless meat of bovine
        "02021000",  # Frozen carcasses of bovine
        "41015010",  # Whole raw hides of bovine
    ],
    "cocoa": [
        "18010000",  # Cocoa beans, whole or broken
        "18020000",  # Cocoa shells, husks, skins
        "18031000",  # Cocoa paste, not defatted
        "18032000",  # Cocoa paste, wholly or partly defatted
        "18040000",  # Cocoa butter, fat and oil
        "18050000",  # Cocoa powder, not sweetened
        "18061015",  # Cocoa powder, sweetened, <5% sucrose
        "18062010",  # Chocolate in blocks >2kg
        "18063100",  # Filled chocolate blocks/slabs/bars
        "18069011",  # Chocolate not filled, in blocks
    ],
    "coffee": [
        "09011100",  # Coffee, not roasted, not decaffeinated
        "09011200",  # Coffee, not roasted, decaffeinated
        "09012100",  # Coffee, roasted, not decaffeinated
        "09012200",  # Coffee, roasted, decaffeinated
        "09019010",  # Coffee husks and skins
        "09019090",  # Coffee substitutes containing coffee
        "21011100",  # Extracts, essences of coffee
        "21011200",  # Coffee preparations with extracts
    ],
    "oil_palm": [
        "15111000",  # Crude palm oil
        "15119011",  # Palm oil fractions, solid
        "15119019",  # Other palm oil fractions, solid
        "15119091",  # Palm oil, refined, liquid
        "15119099",  # Other palm oil fractions, liquid
        "15132110",  # Crude palm kernel oil
        "15132190",  # Palm kernel oil fractions
        "15132911",  # Solid palm kernel oil fractions
        "23066000",  # Palm kernel oil-cake
        "38260010",  # Palm oil-based biodiesel
    ],
    "rubber": [
        "40011000",  # Natural rubber latex
        "40012100",  # Smoked sheets of natural rubber
        "40012200",  # Technically specified rubber (TSNR)
        "40012900",  # Other forms of natural rubber
        "40013000",  # Balata, gutta-percha
        "40021100",  # Styrene-butadiene rubber latex
        "40111000",  # New pneumatic tyres, for motor cars
        "40112010",  # New pneumatic tyres for buses/lorries
        "40119300",  # New pneumatic tyres for aircraft
    ],
    "soya": [
        "12011000",  # Soya bean seeds for sowing
        "12019000",  # Other soya beans
        "15071000",  # Crude soya-bean oil
        "15079010",  # Soya-bean oil, refined
        "15079090",  # Other soya-bean oil fractions
        "23040000",  # Soya-bean oil-cake and meal
        "21031000",  # Soy sauce
        "20060031",  # Soya beans preserved by sugar
    ],
    "wood": [
        "44011100",  # Fuel wood, coniferous, in logs
        "44011200",  # Fuel wood, non-coniferous
        "44012100",  # Wood chips, coniferous
        "44012200",  # Wood chips, non-coniferous
        "44032100",  # Coniferous wood in the rough, treated
        "44039100",  # Oak wood in the rough
        "44071100",  # Coniferous wood, sawn, >6mm thick
        "44079100",  # Oak wood, sawn, >6mm thick
        "47010000",  # Mechanical wood pulp
        "48010000",  # Newsprint in rolls or sheets
    ],
}

# MRN format: 2-digit year + 2-letter country + 13 alphanumeric + 1 check = 18 chars
MRN_PATTERN = re.compile(
    r"^[0-9]{2}[A-Z]{2}[A-Z0-9]{13}[A-Z0-9]$"
)


# ---------------------------------------------------------------------------
# Pydantic Models (15+)
# ---------------------------------------------------------------------------


class CNCodeMapping(GreenLangBase):
    """Mapping between EUDR commodity and EU Combined Nomenclature code."""

    cn_code: str = Field(
        ..., min_length=8, max_length=10,
        description="8-digit EU Combined Nomenclature code",
    )
    commodity: CommodityType = Field(
        ..., description="EUDR commodity classification",
        alias="commodity_type",  # Support legacy field name
    )
    description: str = Field(
        default="", description="CN code description",
    )
    chapter: str = Field(
        default="", description="CN chapter (first 2 digits)",
    )
    heading: str = Field(
        default="", description="CN heading (first 4 digits)",
    )
    hs_code: str = Field(
        default="", description="HS subheading (first 6 digits)",
        alias="subheading",  # Support legacy field name
    )
    duty_rate: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Third-country duty rate percentage",
        alias="duty_rate_percent",  # Support legacy field name
    )
    unit: str = Field(
        default="", description="Supplementary unit (kg, l, p/st)",
        alias="supplementary_unit",  # Support legacy field name
    )
    taric_code: str = Field(
        default="", description="10-digit TARIC code if applicable",
    )
    is_eudr_regulated: bool = Field(
        default=True,
        description="Whether this CN code falls under EUDR scope",
    )
    effective_date: Optional[str] = Field(
        default=None, description="Date from which this mapping is effective",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata",
    )
    provenance_hash: str = Field(
        default="", description="Provenance hash for traceability",
    )

    @field_validator("cn_code")
    @classmethod
    def validate_cn_code_format(cls, v: str) -> str:
        """Validate CN code is 8-10 digits."""
        if not re.match(r"^\d{8,10}$", v):
            raise ValueError(
                f"CN code must be 8-10 digits, got '{v}'"
            )
        return v

    model_config = {"frozen": False, "extra": "ignore", "populate_by_name": True}


class HSCode(GreenLangBase):
    """World Customs Organization Harmonized System code."""

    hs_code: str = Field(
        ..., min_length=6, max_length=6,
        description="6-digit HS code per WCO",
    )
    description: str = Field(
        default="", description="HS code description",
    )
    chapter: Any = Field(
        default="", description="HS chapter (first 2 digits or int)",
    )
    heading: str = Field(
        default="", description="HS heading (first 4 digits)",
    )
    subheading: str = Field(
        default="", description="HS subheading (first 6 digits)",
    )
    is_valid: bool = Field(
        default=True, description="Whether this HS code is currently valid",
    )
    eudr_regulated: bool = Field(
        default=False, description="Whether this HS code falls under EUDR scope",
    )
    notes: str = Field(
        default="", description="Notes on this HS code",
    )
    wco_version: str = Field(
        default="2022", description="WCO HS nomenclature version",
    )
    commodity: Optional[CommodityType] = Field(
        default=None,
        description="Associated EUDR commodity if applicable",
        alias="eudr_commodity",
    )
    cn_code_mappings: List[str] = Field(
        default_factory=list,
        description="Associated 8-digit CN codes",
        alias="cn_codes",
    )
    provenance_hash: str = Field(
        default="", description="Provenance hash for traceability",
    )

    @field_validator("hs_code")
    @classmethod
    def validate_hs_code_format(cls, v: str) -> str:
        """Validate HS code is exactly 6 digits."""
        if not re.match(r"^\d{6}$", v):
            raise ValueError(
                f"HS code must be exactly 6 digits, got '{v}'"
            )
        return v

    model_config = {"frozen": False, "extra": "ignore", "populate_by_name": True}


class TariffCalculation(GreenLangBase):
    """Result of a customs tariff calculation."""

    calculation_id: str = Field(
        ..., description="Unique calculation identifier",
    )
    declaration_id: str = Field(
        default="", description="Associated declaration identifier",
    )
    cn_code: str = Field(
        default="", description="CN code for the tariff lookup",
    )
    total_customs_value: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total customs value",
    )
    total_duty_amount: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total duty amount",
    )
    total_vat_amount: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total VAT amount",
    )
    total_payable: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total payable (duty + VAT)",
    )
    currency: Any = Field(
        default=None, description="Currency code",
    )
    exchange_rate: Decimal = Field(
        default=Decimal("1.0"), gt=0,
        description="Exchange rate to EUR",
    )
    exchange_rate_date: Optional[datetime] = Field(
        default=None, description="Date of exchange rate",
    )
    calculation_method: str = Field(
        default="standard", description="Calculation method used",
    )
    tariff_type: TariffType = Field(
        default=TariffType.CONVENTIONAL,
        description="Type of tariff applied",
    )
    duty_rate_percent: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Duty rate as percentage",
    )
    customs_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Customs value in EUR",
    )
    duty_amount_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Calculated duty amount in EUR",
    )
    vat_rate_percent: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Applicable VAT rate percentage",
    )
    vat_amount_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Calculated VAT amount in EUR",
    )
    total_charges_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total customs charges in EUR (duty + VAT + additional)",
    )
    anti_dumping_duty_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Anti-dumping duty if applicable",
    )
    preferential_rate_applied: bool = Field(
        default=False,
        description="Whether a preferential rate was applied",
    )
    origin_country: str = Field(
        default="", description="Country of origin for tariff determination",
    )
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class CountryOriginVerification(GreenLangBase):
    """Verification of country of origin against supply chain data."""

    verification_id: str = Field(
        default="", description="Unique verification identifier",
    )
    declaration_id: str = Field(
        default="", description="Associated declaration identifier",
    )
    declared_origin: str = Field(
        default="", description="Declared country of origin (ISO 3166-1)",
    )
    supply_chain_origins: List[str] = Field(
        default_factory=list,
        description="Origins from supply chain traceability data",
    )
    dds_reference: str = Field(
        default="", description="DDS reference number",
    )
    result: Optional[OriginVerificationResult] = Field(
        default=None,
        description="Verification result (verified/mismatch/unverified)",
    )
    confidence_score: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Confidence score (0-100)",
    )
    verification_method: str = Field(
        default="", description="Method used for verification",
    )
    mismatch_details: str = Field(
        default="", description="Details of any mismatch found",
    )
    # Legacy fields
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Verification result status (legacy)",
    )
    origin_certificate_ref: str = Field(
        default="",
        description="Certificate of origin reference number",
    )
    preferential_origin: bool = Field(
        default=False,
        description="Whether preferential origin status applies",
    )
    gsp_eligible: bool = Field(
        default=False,
        description="Whether GSP preferential treatment applies",
    )
    country_risk_level: str = Field(
        default="standard",
        description="Country risk level (low/standard/high)",
    )
    discrepancies: List[str] = Field(
        default_factory=list,
        description="Discrepancies found between declared and actual origin",
    )
    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class SubmissionLog(GreenLangBase):
    """Log entry for a customs system submission attempt."""

    submission_id: str = Field(
        ..., description="Unique submission identifier",
    )
    declaration_id: str = Field(
        ..., description="Associated declaration identifier",
    )
    customs_system: CustomsSystem = Field(
        ..., description="Target customs system",
    )
    submission_status: SubmissionStatus = Field(
        default=SubmissionStatus.QUEUED,
        description="Current submission status",
    )
    mrn: str = Field(
        default="",
        description="Movement Reference Number (assigned by customs)",
    )
    error_code: str = Field(
        default="", description="Error code if submission failed",
    )
    error_message: str = Field(
        default="", description="Error message if submission failed",
    )
    retry_count: int = Field(
        default=0, ge=0, description="Number of retry attempts",
    )
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    acknowledged_at: Optional[datetime] = None
    response_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw response from customs system",
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ComplianceCheck(GreenLangBase):
    """Result of an EUDR compliance check for customs declaration."""

    check_id: str = Field(
        default="", description="Unique check identifier",
    )
    declaration_id: str = Field(
        default="", description="Associated declaration identifier",
    )
    check_type: str = Field(
        default="", description="Type of compliance check performed",
    )
    result: Optional[ValidationResult] = Field(
        default=None,
        description="Validation result (pass/fail/warning/not_applicable)",
    )
    message: str = Field(
        default="", description="Check result message",
    )
    severity: str = Field(
        default="info", description="Severity level (info/warning/critical)",
    )
    suggested_fix: str = Field(
        default="", description="Suggested remediation action",
    )
    dds_reference: str = Field(
        default="",
        description="Due Diligence Statement reference number",
    )
    article_reference: str = Field(
        default="",
        description="EUDR article reference for this check",
    )
    status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Check result status (legacy)",
    )
    dds_reference_number: str = Field(
        default="",
        description="Due Diligence Statement reference number (legacy)",
    )
    details: str = Field(
        default="", description="Detailed check result description",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence references",
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class PortOfEntry(GreenLangBase):
    """EU port of entry / customs office information."""

    port_code: str = Field(
        ..., description="Port/customs office code (UN/LOCODE)",
    )
    port_name: str = Field(
        default="", description="Port name",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="EU member state code (ISO 3166-1 alpha-2)",
    )
    port_type: PortType = Field(
        default=PortType.SEA, description="Type of port",
    )
    customs_office_code: str = Field(
        default="", description="Customs office reference number",
    )
    timezone: str = Field(
        default="", description="Port timezone",
    )
    ncts_enabled: bool = Field(
        default=False, description="Whether NCTS is enabled at this port",
    )
    ais_enabled: bool = Field(
        default=False, description="Whether AIS is enabled at this port",
    )
    is_active: bool = Field(
        default=True, description="Whether this port is active",
    )
    supported_commodities: List[CommodityType] = Field(
        default_factory=list,
        description="EUDR commodities handled at this port",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional port information",
    )
    provenance_hash: str = Field(
        default="", description="Provenance hash for traceability",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class DeclarationLine(GreenLangBase):
    """A single line item in a customs declaration."""

    line_number: int = Field(
        ..., ge=1, le=99,
        description="Item number (1-99 per SAD form)",
    )
    cn_code: str = Field(
        ..., min_length=8, max_length=10,
        description="8-digit CN code",
    )
    commodity_type: CommodityType = Field(
        ..., description="EUDR commodity type",
    )
    description: str = Field(
        ..., description="Goods description",
    )
    country_of_origin: str = Field(
        ..., min_length=2, max_length=3,
        description="Country of origin (ISO 3166-1)",
    )
    net_mass_kg: Decimal = Field(
        ..., gt=0, description="Net mass in kilograms",
    )
    gross_mass_kg: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Gross mass in kilograms",
    )
    supplementary_units: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Supplementary units quantity",
    )
    statistical_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Statistical value in EUR",
    )
    customs_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Customs value in EUR",
    )
    duty_amount_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Duty amount in EUR",
    )
    dds_reference_number: str = Field(
        default="",
        description="DDS reference number per EUDR Article 4(2)",
    )
    lot_number: str = Field(
        default="", description="Production lot number for traceability",
    )
    preference_code: str = Field(
        default="", description="Customs preference code (box 36)",
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ValueDeclaration(GreenLangBase):
    """Customs value declaration per WTO Valuation Agreement."""

    value_id: str = Field(
        ..., description="Unique value declaration identifier",
    )
    incoterms: Incoterms = Field(
        default=Incoterms.CIF,
        description="Incoterms basis for transaction value",
    )
    transaction_value: Decimal = Field(
        ..., gt=0,
        description="Transaction value in original currency",
    )
    currency: str = Field(
        default="EUR", description="Original currency code",
    )
    exchange_rate: Decimal = Field(
        default=Decimal("1.0000"), gt=0,
        description="Exchange rate to EUR",
    )
    freight_cost: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Freight/transport cost",
    )
    insurance_cost: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Insurance cost",
    )
    loading_cost: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Loading/handling cost",
    )
    adjustments: Decimal = Field(
        default=Decimal("0"),
        description="Other adjustments (positive or negative)",
    )
    cif_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Calculated CIF value in EUR",
    )
    fob_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Calculated FOB value in EUR",
    )
    customs_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Final customs value in EUR",
    )
    valuation_method: str = Field(
        default="transaction_value",
        description="WTO valuation method used (1-6)",
    )
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class QuantityDeclaration(GreenLangBase):
    """Quantity declaration for goods in a customs entry."""

    net_mass_kg: Decimal = Field(
        ..., gt=0, description="Net mass in kilograms",
    )
    gross_mass_kg: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Gross mass in kilograms",
    )
    packages_count: int = Field(
        default=1, ge=1, description="Number of packages",
    )
    package_type: str = Field(
        default="CT", description="Package type code (UN/ECE Rec 21)",
    )
    container_number: str = Field(
        default="", description="Container number if applicable",
    )
    marks_and_numbers: str = Field(
        default="", description="Marks and numbers on packages",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class SADForm(GreenLangBase):
    """Single Administrative Document (SAD) form for EU customs.

    Represents the standard EU customs declaration form with all
    mandatory boxes per UCC Delegated Regulation (EU) 2015/2446.
    Supports both box-numbered field names (box1_*, box2_*, ...) for
    test compatibility and descriptive field names for readability.
    """

    # Internal identifiers
    sad_id: str = Field(
        default="", description="Internal SAD form identifier",
    )
    form_id: str = Field(
        default="", description="Internal form identifier (alias)",
    )
    declaration_id: str = Field(
        default="", description="Associated declaration identifier",
    )
    form_type: str = Field(
        default="IM", description="Form type (IM/EX/TR)",
    )

    # Box-numbered fields for test compatibility
    box1_declaration_type: str = Field(
        default="IM", description="Box 1: Declaration type",
    )
    box2_consignor: str = Field(
        default="", description="Box 2: Consignor/Exporter",
    )
    box3_forms_count: int = Field(
        default=1, description="Box 3: Number of forms",
    )
    box5_items_count: int = Field(
        default=1, description="Box 5: Number of items",
    )
    box6_total_packages: int = Field(
        default=0, description="Box 6: Total packages",
    )
    box8_consignee: str = Field(
        default="", description="Box 8: Consignee/Importer",
    )
    box8_eori: str = Field(
        default="", description="Box 8: Consignee EORI",
    )
    box11_trading_country: str = Field(
        default="", description="Box 11: Trading country",
    )
    box14_declarant: str = Field(
        default="", description="Box 14: Declarant",
    )
    box14_eori: str = Field(
        default="", description="Box 14: Declarant EORI",
    )
    box15_country_of_dispatch: str = Field(
        default="", description="Box 15: Country of dispatch",
    )
    box17_country_of_destination: str = Field(
        default="", description="Box 17: Country of destination",
    )
    box22_currency_and_total: str = Field(
        default="", description="Box 22: Currency and total invoiced amount",
    )
    box25_mode_of_transport: Any = Field(
        default=1, description="Box 25: Mode of transport code",
    )
    box26_inland_mode_of_transport: Any = Field(
        default=3, description="Box 26: Inland mode of transport",
    )
    box29_office_of_entry: str = Field(
        default="", description="Box 29: Customs office of entry",
    )
    box31_package_description: str = Field(
        default="", description="Box 31: Package description",
    )
    box33_commodity_code: str = Field(
        default="", description="Box 33: Commodity code (CN code)",
    )
    box34_country_of_origin: str = Field(
        default="", description="Box 34: Country of origin",
    )
    box35_gross_weight: Decimal = Field(
        default=Decimal("0"), ge=0, description="Box 35: Gross weight",
    )
    box37_procedure: str = Field(
        default="", description="Box 37: Procedure code",
    )
    box38_net_weight: Decimal = Field(
        default=Decimal("0"), ge=0, description="Box 38: Net weight",
    )
    box44_additional_info: str = Field(
        default="", description="Box 44: Additional information",
    )
    box46_statistical_value: Decimal = Field(
        default=Decimal("0"), ge=0, description="Box 46: Statistical value",
    )
    box47_duty_calculation: str = Field(
        default="", description="Box 47: Duty calculation summary",
    )
    box54_date_and_signature: str = Field(
        default="", description="Box 54: Date and signature",
    )

    # EUDR-specific fields
    eudr_dds_reference: str = Field(
        default="", description="EUDR DDS reference number",
    )
    eudr_compliance_status: str = Field(
        default="", description="EUDR compliance status",
    )

    # Legacy descriptive fields (kept for backward compat)
    declaration_type: DeclarationType = Field(
        default=DeclarationType.IMPORT,
    )
    consignor_name: str = Field(default="", description="Consignor name")
    consignor_address: str = Field(default="", description="Consignor address")
    consignor_country: str = Field(default="", description="Consignor country")
    consignor_eori: str = Field(default="", description="Consignor EORI number")
    consignee_name: str = Field(default="", description="Consignee name")
    consignee_address: str = Field(default="", description="Consignee address")
    consignee_country: str = Field(default="", description="Consignee country")
    consignee_eori: str = Field(default="", description="Consignee EORI number")
    declarant_name: str = Field(default="", description="Declarant name")
    declarant_eori: str = Field(default="", description="Declarant EORI number")
    country_of_dispatch: str = Field(default="", description="Country of dispatch")
    country_of_destination: str = Field(default="", description="Country of destination")
    delivery_terms: Incoterms = Field(default=Incoterms.CIF, description="Delivery terms")
    invoice_currency: str = Field(default="EUR", description="Invoice currency")
    invoice_total: Decimal = Field(default=Decimal("0"), ge=0, description="Total invoiced amount")
    transport_mode: str = Field(default="1", description="Mode of transport code")
    customs_office_entry: str = Field(default="", description="Customs office of entry code")
    line_items: List[DeclarationLine] = Field(default_factory=list, description="Declaration line items")
    dds_reference_numbers: List[str] = Field(default_factory=list, description="DDS reference numbers")
    additional_documents: List[Dict[str, str]] = Field(default_factory=list, description="Additional documents")
    total_statistical_value_eur: Decimal = Field(default=Decimal("0"), ge=0)
    total_duty_eur: Decimal = Field(default=Decimal("0"), ge=0)
    total_vat_eur: Decimal = Field(default=Decimal("0"), ge=0)
    total_charges_eur: Decimal = Field(default=Decimal("0"), ge=0)

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class CustomsDeclaration(GreenLangBase):
    """A complete customs declaration record.

    Top-level model representing the full lifecycle of a customs
    declaration for EUDR-regulated commodities, including the SAD form,
    compliance checks, submission logs, and clearance status.
    """

    declaration_id: str = Field(
        ..., description="Unique declaration identifier",
    )
    operator_id: str = Field(
        default="", description="EUDR operator identifier",
    )
    operator_name: str = Field(
        default="", description="Operator name",
    )
    operator_eori: str = Field(
        default="", description="Operator EORI number",
    )
    consignee_name: str = Field(
        default="", description="Consignee name",
    )
    consignee_eori: str = Field(
        default="", description="Consignee EORI number",
    )
    declaration_type: DeclarationType = Field(
        default=DeclarationType.IMPORT,
    )
    status: DeclarationStatus = Field(
        default=DeclarationStatus.DRAFT,
    )
    purpose: DeclarationPurpose = Field(
        default=DeclarationPurpose.FREE_CIRCULATION,
    )
    mrn: str = Field(
        default="",
        description="Movement Reference Number (18 chars, assigned by customs)",
    )
    lrn: str = Field(
        default="",
        description="Local Reference Number (assigned by declarant)",
    )
    # Customs details
    port_of_entry: Any = Field(
        default=None, description="Port of entry code or PortOfEntry object",
    )
    customs_office_code: str = Field(
        default="", description="Customs office code",
    )
    incoterms: Any = Field(
        default=None, description="Incoterms basis",
    )
    total_value: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total declared value",
    )
    currency: Any = Field(
        default=None, description="Currency code",
    )
    total_gross_weight: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total gross weight",
    )
    total_net_weight: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total net weight",
    )
    weight_unit: str = Field(
        default="kg", description="Unit of weight",
    )
    # Commodity details
    commodities: List[CommodityType] = Field(
        default_factory=list,
        description="EUDR commodity types in this declaration",
    )
    cn_codes: List[str] = Field(
        default_factory=list,
        description="CN codes in this declaration",
    )
    hs_codes: List[str] = Field(
        default_factory=list,
        description="HS codes in this declaration",
    )
    country_of_origin: str = Field(
        default="", description="Country of origin",
    )
    country_of_dispatch: str = Field(
        default="", description="Country of dispatch",
    )
    # DDS & EUDR
    dds_reference: str = Field(
        default="", description="DDS reference number",
    )
    dds_reference_numbers: List[str] = Field(
        default_factory=list,
        description="DDS reference numbers linked to this declaration",
    )
    eudr_compliance_passed: bool = Field(
        default=False,
        description="Whether all EUDR compliance checks passed",
    )
    # Rejection
    rejection_reason: str = Field(
        default="", description="Reason for rejection",
    )
    rejected_at: Optional[datetime] = None
    # Sub-documents
    sad_form: Optional[SADForm] = Field(
        default=None, description="Generated SAD form",
    )
    value_declaration: Optional[ValueDeclaration] = Field(
        default=None, description="Customs value declaration",
    )
    quantity_declaration: Optional[QuantityDeclaration] = Field(
        default=None, description="Quantity declaration",
    )
    compliance_checks: List[ComplianceCheck] = Field(
        default_factory=list,
        description="EUDR compliance check results",
    )
    submission_logs: List[SubmissionLog] = Field(
        default_factory=list,
        description="Customs submission attempt history",
    )
    origin_verifications: List[CountryOriginVerification] = Field(
        default_factory=list,
        description="Country of origin verification results",
    )
    tariff_calculations: List[TariffCalculation] = Field(
        default_factory=list,
        description="Tariff calculation results",
    )
    customs_system: CustomsSystem = Field(
        default=CustomsSystem.AIS,
        description="Target customs system",
    )
    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    submitted_at: Optional[datetime] = None
    cleared_at: Optional[datetime] = None
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    provenance_hash: str = ""

    @field_validator("mrn")
    @classmethod
    def validate_mrn_format(cls, v: str) -> str:
        """Validate MRN format if provided.

        MRN format: YYCCxxxxxxxxxxxxxc (2-digit year, 2-letter country,
        13 alphanumeric characters, 1 check character = 18 total).
        """
        if v and not MRN_PATTERN.match(v):
            raise ValueError(
                f"Invalid MRN format: '{v}'. "
                "Expected: YYCCxxxxxxxxxxxxxc (18 alphanumeric chars)"
            )
        return v

    model_config = {"frozen": False, "extra": "ignore"}


class DeclarationSummary(GreenLangBase):
    """Summary view of a customs declaration for listing endpoints."""

    declaration_id: str = Field(..., description="Declaration identifier")
    operator_id: str = Field(..., description="Operator identifier")
    status: DeclarationStatus
    declaration_type: DeclarationType
    mrn: str = Field(default="", description="MRN if assigned")
    commodity_types: List[CommodityType] = Field(
        default_factory=list, description="Commodities in declaration",
    )
    total_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total customs value in EUR",
    )
    total_charges_eur: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total customs charges in EUR",
    )
    line_count: int = Field(
        default=0, ge=0, description="Number of line items",
    )
    eudr_compliance_passed: bool = False
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    model_config = {"frozen": False, "extra": "ignore"}


class AuditEntry(GreenLangBase):
    """An audit trail entry for customs declaration events."""

    entry_id: str = Field(..., description="Unique audit entry identifier")
    entity_type: str = Field(
        ..., description="Entity type (declaration, submission, etc.)",
    )
    entity_id: str = Field(
        ..., description="Entity identifier being audited",
    )
    action: AuditAction
    actor: str = Field(
        ..., description="User or system performing the action",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the action",
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(GreenLangBase):
    """Health check response for the Customs Declaration Support agent."""

    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0

    model_config = {"frozen": False, "extra": "ignore"}


class TariffLineItem(GreenLangBase):
    """A single tariff line item for duty calculation."""

    line_number: int = Field(
        ..., ge=1, description="Line item number",
    )
    cn_code: str = Field(
        ..., description="CN code for this line",
    )
    description: str = Field(
        default="", description="Tariff description",
    )
    quantity: Decimal = Field(
        default=Decimal("0"), ge=0, description="Quantity of goods",
    )
    unit: str = Field(
        default="", description="Unit of measurement (kg, l, p/st)",
    )
    unit_price: Decimal = Field(
        default=Decimal("0"), ge=0, description="Unit price",
    )
    total_value: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total value",
    )
    currency: Any = Field(
        default=None, description="Currency code",
    )
    duty_rate: Decimal = Field(
        default=Decimal("0"), ge=0, description="Duty rate percentage",
    )
    duty_amount: Decimal = Field(
        default=Decimal("0"), ge=0, description="Calculated duty amount",
    )
    vat_rate: Decimal = Field(
        default=Decimal("0"), ge=0, description="VAT rate percentage",
    )
    vat_amount: Decimal = Field(
        default=Decimal("0"), ge=0, description="Calculated VAT amount",
    )
    origin_country: str = Field(
        default="", description="Country of origin",
    )
    tariff_type: TariffType = Field(
        default=TariffType.CONVENTIONAL, description="Type of tariff",
    )
    customs_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Customs value in EUR",
    )
    duty_amount_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Duty amount in EUR",
    )
    duty_rate_percent: Decimal = Field(
        default=Decimal("0"), ge=0, description="Duty rate percentage (alias)",
    )
    provenance_hash: str = Field(
        default="", description="Provenance hash",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class DutyCalculation(GreenLangBase):
    """Detailed duty calculation breakdown."""

    cn_code: str = Field(
        default="", description="CN code for this duty calculation",
    )
    customs_value: Decimal = Field(
        default=Decimal("0"), ge=0, description="Customs value",
    )
    duty_rate: Decimal = Field(
        default=Decimal("0"), ge=0, description="Duty rate percentage",
    )
    duty_amount: Decimal = Field(
        default=Decimal("0"), ge=0, description="Calculated duty amount",
    )
    preferential_rate: Optional[Decimal] = Field(
        default=None, description="Preferential duty rate",
    )
    preferential_origin: str = Field(
        default="", description="Preferential origin country code",
    )
    anti_dumping_duty: Decimal = Field(
        default=Decimal("0"), ge=0, description="Anti-dumping duty",
    )
    countervailing_duty: Decimal = Field(
        default=Decimal("0"), ge=0, description="Countervailing duty",
    )
    total_duty: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total duty amount",
    )
    calculation_id: str = Field(
        default="", description="Unique calculation identifier",
    )
    line_items: List[TariffLineItem] = Field(
        default_factory=list, description="Tariff line items",
    )
    total_customs_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total customs value in EUR",
    )
    total_duty_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total duty in EUR",
    )
    total_vat_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total VAT in EUR",
    )
    vat_rate_percent: Decimal = Field(
        default=Decimal("0"), ge=0, description="VAT rate percentage",
    )
    total_charges_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total charges in EUR",
    )
    calculation_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class CustomsInterfaceResponse(GreenLangBase):
    """Response from a customs system interface (NCTS/AIS/ICS2)."""

    system: Optional[CustomsSystemType] = Field(
        default=None, description="Source customs system",
    )
    request_id: str = Field(
        default="", description="Request identifier",
    )
    mrn: str = Field(
        default="", description="Assigned MRN",
    )
    status: str = Field(
        default="", description="Response status string",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    response_code: str = Field(
        default="", description="Customs system response code",
    )
    response_message: str = Field(
        default="", description="Customs system response message",
    )
    errors: List[str] = Field(
        default_factory=list, description="Error messages",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), ge=0, description="Processing time in ms",
    )
    response_id: str = Field(
        default="", description="Response identifier (legacy)",
    )
    customs_system_legacy: Optional[CustomsSystem] = Field(
        default=None, description="Source customs system (legacy)",
        alias="customs_system_internal",
    )
    success: bool = Field(
        default=False, description="Whether submission was successful",
    )
    message: str = Field(
        default="", description="Response message",
    )
    error_code: str = Field(
        default="", description="Error code if failed",
    )
    error_details: str = Field(
        default="", description="Detailed error message",
    )
    raw_response: Dict[str, Any] = Field(
        default_factory=dict, description="Raw response payload",
    )
    provenance_hash: str = Field(
        default="", description="Provenance hash",
    )

    model_config = {"frozen": False, "extra": "ignore", "populate_by_name": True}


# ---------------------------------------------------------------------------
# Additional Constants
# ---------------------------------------------------------------------------

# HS Chapters for EUDR-regulated commodities (per Annex I)
EUDR_HS_CHAPTERS = {
    "01": "Live animals",
    "02": "Meat and edible meat offal",
    "04": "Dairy produce; birds' eggs; natural honey",
    "09": "Coffee, tea, mate and spices",
    "12": "Oil seeds; miscellaneous grains, seeds and fruit",
    "15": "Animal or vegetable fats and oils",
    "18": "Cocoa and cocoa preparations",
    "20": "Preparations of vegetables, fruit, nuts",
    "21": "Miscellaneous edible preparations",
    "23": "Residues and waste from food industries",
    "38": "Miscellaneous chemical products",
    "40": "Rubber and articles thereof",
    "41": "Raw hides and skins (other than furskins) and leather",
    "44": "Wood and articles of wood; wood charcoal",
    "47": "Pulp of wood or other fibrous cellulosic material",
    "48": "Paper and paperboard; articles of paper pulp",
}

# MRN format as regex string (for compatibility with tests expecting string)
MRN_FORMAT_REGEX = r"^[0-9]{2}[A-Z]{2}[A-Z0-9]{13}[A-Z0-9]$"

# Supported Incoterms list
SUPPORTED_INCOTERMS = [
    "EXW", "FCA", "FAS", "FOB", "CFR", "CIF",
    "CPT", "CIP", "DAP", "DPU", "DDP"
]

# ---------------------------------------------------------------------------
# Aliases for backward compatibility with tests
# ---------------------------------------------------------------------------

# Constants
EUDR_CN_CODE_MAPPINGS = EUDR_COMMODITY_CN_CODES
EUDR_REGULATED_COMMODITIES = [CommodityType(c) for c in EUDR_COMMODITY_CN_CODES.keys()]

# Enum aliases (ComplianceStatus and CustomsSystemType are now real classes above)
IncotermsType = Incoterms  # IncotermsType -> Incoterms

# Model aliases
HSCodeInfo = HSCode  # HSCodeInfo -> HSCode
OriginVerification = CountryOriginVerification  # OriginVerification -> CountryOriginVerification

# Additional enums needed by tests
class CurrencyCode(str, enum.Enum):
    """ISO 4217 currency codes commonly used in customs declarations."""

    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CNY = "CNY"
    CAD = "CAD"
    AUD = "AUD"
    INR = "INR"
    BRL = "BRL"
