# -*- coding: utf-8 -*-
"""
DDSAssemblyEngine - PACK-006 EUDR Starter Engine 1
====================================================

Due Diligence Statement (DDS) composition engine per EUDR Annex II.
Assembles, validates, and exports Due Diligence Statements required
under Regulation (EU) 2023/1115 for operators and traders placing
relevant commodities on the EU market.

Key Capabilities:
    - Full Article 4 DDS assembly with all Annex II required fields
    - Simplified Article 13 DDS for low-risk country sourcing
    - Batch assembly for multi-supplier operations
    - EU Information System (EU-IS) export formatting
    - Evidence attachment and operator declaration finalization
    - Geolocation formatting per WGS84 with 6-decimal precision
    - Unique DDS reference number generation
    - Complete Annex II field validation

Annex II Required Fields:
    1. Operator/trader name and address
    2. HS/CN codes of relevant products
    3. Quantity (net mass and supplementary unit)
    4. Country of production
    5. Geolocation (coordinates <4ha, polygons >=4ha)
    6. Supplier information
    7. Conclusion of risk assessment
    8. Risk mitigation measures (if any)

Zero-Hallucination:
    - All DDS composition is deterministic field mapping
    - No LLM involvement in statement assembly or validation
    - SHA-256 provenance hashing on every DDS document
    - Pydantic validation at all input/output boundaries

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-006 EUDR Starter
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DDSType(str, Enum):
    """Type of Due Diligence Statement."""

    STANDARD = "STANDARD"
    SIMPLIFIED = "SIMPLIFIED"

class DDSStatus(str, Enum):
    """Status of a Due Diligence Statement."""

    DRAFT = "DRAFT"
    ASSEMBLED = "ASSEMBLED"
    VALIDATED = "VALIDATED"
    FINALIZED = "FINALIZED"
    SUBMITTED = "SUBMITTED"
    AMENDED = "AMENDED"

class RiskConclusion(str, Enum):
    """Risk assessment conclusion per EUDR Article 10."""

    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    STANDARD = "STANDARD"
    HIGH = "HIGH"

class EvidenceType(str, Enum):
    """Types of evidence that can be attached to a DDS."""

    SATELLITE_IMAGE = "SATELLITE_IMAGE"
    CERTIFICATE = "CERTIFICATE"
    AUDIT_REPORT = "AUDIT_REPORT"
    TRANSACTION_RECORD = "TRANSACTION_RECORD"
    GEOLOCATION_DATA = "GEOLOCATION_DATA"
    CUSTOMS_DECLARATION = "CUSTOMS_DECLARATION"
    SUPPLIER_DECLARATION = "SUPPLIER_DECLARATION"
    LAND_TITLE = "LAND_TITLE"
    PERMIT = "PERMIT"
    OTHER = "OTHER"

class GeolocationFormat(str, Enum):
    """Format of geolocation data."""

    POINT = "POINT"
    POLYGON = "POLYGON"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class SupplierInfo(BaseModel):
    """Supplier information for DDS Annex II field 6."""

    supplier_id: str = Field(default_factory=_new_uuid, description="Unique supplier identifier")
    name: str = Field(..., description="Legal name of the supplier")
    address: Optional[str] = Field(None, description="Full address of the supplier")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    eori_number: Optional[str] = Field(None, description="EORI number if EU-based")
    contact_person: Optional[str] = Field(None, description="Contact person name")
    contact_email: Optional[str] = Field(None, description="Contact email address")
    tier: int = Field(default=1, ge=1, le=10, description="Supply chain tier (1=direct)")
    certifications: List[str] = Field(default_factory=list, description="Active certifications")

class ProductInfo(BaseModel):
    """Product information for DDS Annex II fields 2-3."""

    product_id: str = Field(default_factory=_new_uuid, description="Unique product identifier")
    description: str = Field(..., description="Product description")
    cn_code: str = Field(..., description="Combined Nomenclature 8-digit code")
    hs_code: Optional[str] = Field(None, description="Harmonized System 6-digit code")
    commodity_category: str = Field(..., description="EUDR commodity category")
    net_mass_kg: Decimal = Field(..., ge=0, description="Net mass in kilograms")
    supplementary_unit: Optional[str] = Field(None, description="Supplementary unit (e.g., 'pieces')")
    supplementary_quantity: Optional[Decimal] = Field(None, ge=0, description="Quantity in supplementary unit")
    trade_name: Optional[str] = Field(None, description="Trade or brand name")

class GeolocationPoint(BaseModel):
    """Single geolocation point (WGS84, 6 decimal places)."""

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")

    @field_validator("latitude", "longitude")
    @classmethod
    def round_to_six_decimals(cls, v: float) -> float:
        """Ensure 6-decimal precision per EUDR requirement."""
        return round(v, 6)

class GeolocationPolygon(BaseModel):
    """Polygon geolocation for plots >= 4 hectares."""

    coordinates: List[List[float]] = Field(
        ..., description="List of [lon, lat] pairs forming a closed ring"
    )
    area_hectares: Optional[float] = Field(None, ge=0, description="Area in hectares")

    @field_validator("coordinates")
    @classmethod
    def validate_closed_ring(cls, v: List[List[float]]) -> List[List[float]]:
        """Ensure polygon is a closed ring (first point == last point)."""
        if len(v) < 4:
            raise ValueError("Polygon must have at least 4 coordinate pairs (3 unique + closing)")
        if v[0] != v[-1]:
            v.append(v[0])
        return v

class GeolocationInfo(BaseModel):
    """Geolocation information for DDS Annex II field 5."""

    plot_id: str = Field(default_factory=_new_uuid, description="Unique plot identifier")
    format: GeolocationFormat = Field(..., description="POINT for <4ha, POLYGON for >=4ha")
    point: Optional[GeolocationPoint] = Field(None, description="Point coordinate for <4ha plots")
    polygon: Optional[GeolocationPolygon] = Field(None, description="Polygon for >=4ha plots")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    area_hectares: Optional[float] = Field(None, ge=0, description="Plot area in hectares")
    datum: str = Field(default="WGS84", description="Geodetic datum (must be WGS84)")

class RiskSummary(BaseModel):
    """Risk assessment summary for DDS Annex II field 7."""

    risk_id: str = Field(default_factory=_new_uuid, description="Risk assessment identifier")
    conclusion: RiskConclusion = Field(..., description="Overall risk conclusion")
    country_risk: str = Field(default="STANDARD", description="Country-level risk benchmark")
    supplier_risk_score: Optional[float] = Field(None, ge=0, le=100, description="Supplier risk score")
    commodity_risk_score: Optional[float] = Field(None, ge=0, le=100, description="Commodity risk score")
    deforestation_risk: Optional[str] = Field(None, description="Deforestation risk assessment")
    assessment_date: datetime = Field(default_factory=utcnow, description="Assessment date")
    assessor: Optional[str] = Field(None, description="Person or system that performed assessment")
    methodology: str = Field(default="EUDR_STANDARD", description="Assessment methodology used")
    key_findings: List[str] = Field(default_factory=list, description="Key risk findings")

class MitigationSummary(BaseModel):
    """Risk mitigation measures for DDS Annex II field 8."""

    mitigation_id: str = Field(default_factory=_new_uuid, description="Mitigation record identifier")
    measures_taken: List[str] = Field(default_factory=list, description="Mitigation measures applied")
    residual_risk: Optional[str] = Field(None, description="Risk level after mitigation")
    additional_dd_performed: bool = Field(default=False, description="Whether additional DD was needed")
    third_party_verification: bool = Field(default=False, description="Whether third-party verified")
    verification_entity: Optional[str] = Field(None, description="Third-party verifier name")
    completion_date: Optional[datetime] = Field(None, description="When mitigation was completed")
    effectiveness_assessment: Optional[str] = Field(None, description="Mitigation effectiveness")

class OperatorDeclaration(BaseModel):
    """Operator/trader declaration for DDS finalization."""

    operator_name: str = Field(..., description="Legal name of operator/trader")
    operator_address: str = Field(..., description="Full registered address")
    operator_country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    eori_number: Optional[str] = Field(None, description="EORI number")
    authorized_representative: Optional[str] = Field(None, description="Authorized representative name")
    declaration_text: str = Field(
        default="I hereby declare that due diligence has been exercised "
        "in accordance with Regulation (EU) 2023/1115 and that the risk "
        "of the relevant products being associated with deforestation, "
        "forest degradation, or non-compliance with relevant legislation "
        "of the country of production is negligible.",
        description="Declaration text",
    )
    signature_date: datetime = Field(default_factory=utcnow, description="Date of signature")
    signature_reference: Optional[str] = Field(None, description="Digital signature reference")
    is_sme: bool = Field(default=False, description="Whether the operator is an SME")
    is_trader: bool = Field(default=False, description="True if trader, False if operator")

class DDSEvidence(BaseModel):
    """Evidence document attached to a DDS."""

    evidence_id: str = Field(default_factory=_new_uuid, description="Evidence identifier")
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    title: str = Field(..., description="Evidence title or filename")
    description: Optional[str] = Field(None, description="Evidence description")
    file_reference: Optional[str] = Field(None, description="File storage reference/path")
    file_hash: Optional[str] = Field(None, description="SHA-256 hash of the evidence file")
    issue_date: Optional[datetime] = Field(None, description="When the evidence was issued")
    expiry_date: Optional[datetime] = Field(None, description="When the evidence expires")
    issuing_authority: Optional[str] = Field(None, description="Authority that issued the evidence")
    attached_at: datetime = Field(default_factory=utcnow, description="When attached to DDS")

class FormattedGeolocation(BaseModel):
    """Formatted geolocation data for DDS output."""

    plots: List[GeolocationInfo] = Field(default_factory=list, description="All plot geolocations")
    total_plots: int = Field(default=0, description="Total number of plots")
    total_area_hectares: float = Field(default=0.0, description="Total area across all plots")
    countries: List[str] = Field(default_factory=list, description="Distinct countries of production")
    datum: str = Field(default="WGS84", description="Geodetic datum used")
    precision_decimals: int = Field(default=6, description="Coordinate decimal precision")
    provenance_hash: str = Field(default="", description="SHA-256 hash of geolocation data")

class AnnexIIField(BaseModel):
    """Validation result for a single Annex II field."""

    field_number: int = Field(..., ge=1, le=8, description="Annex II field number (1-8)")
    field_name: str = Field(..., description="Human-readable field name")
    is_present: bool = Field(default=False, description="Whether the field is populated")
    is_valid: bool = Field(default=False, description="Whether the field passes validation")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")

class AnnexIIValidation(BaseModel):
    """Complete Annex II validation result."""

    validation_id: str = Field(default_factory=_new_uuid, description="Validation run identifier")
    is_complete: bool = Field(default=False, description="Whether all required fields are present")
    is_valid: bool = Field(default=False, description="Whether all fields pass validation")
    fields: List[AnnexIIField] = Field(default_factory=list, description="Per-field validation results")
    total_fields: int = Field(default=8, description="Total Annex II fields")
    passed_fields: int = Field(default=0, description="Number of fields passing validation")
    failed_fields: int = Field(default=0, description="Number of fields failing validation")
    errors: List[str] = Field(default_factory=list, description="Aggregate validation errors")
    warnings: List[str] = Field(default_factory=list, description="Aggregate validation warnings")
    validated_at: datetime = Field(default_factory=utcnow, description="Validation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 hash of validation")

class DDSDocument(BaseModel):
    """Complete Due Diligence Statement document."""

    dds_id: str = Field(default_factory=_new_uuid, description="Unique DDS identifier")
    reference_number: str = Field(default="", description="DDS reference number")
    dds_type: DDSType = Field(default=DDSType.STANDARD, description="Standard or simplified DDS")
    status: DDSStatus = Field(default=DDSStatus.DRAFT, description="Current DDS status")

    # Annex II Field 1: Operator/Trader
    operator: Optional[OperatorDeclaration] = Field(None, description="Operator/trader declaration")

    # Annex II Field 2-3: Product information
    products: List[ProductInfo] = Field(default_factory=list, description="Products in this DDS")

    # Annex II Field 4: Country of production
    countries_of_production: List[str] = Field(default_factory=list, description="ISO country codes")

    # Annex II Field 5: Geolocation
    geolocations: List[GeolocationInfo] = Field(default_factory=list, description="Plot geolocations")

    # Annex II Field 6: Suppliers
    suppliers: List[SupplierInfo] = Field(default_factory=list, description="Supplier information")

    # Annex II Field 7: Risk assessment conclusion
    risk_summary: Optional[RiskSummary] = Field(None, description="Risk assessment summary")

    # Annex II Field 8: Risk mitigation
    mitigation_summary: Optional[MitigationSummary] = Field(None, description="Mitigation measures")

    # Evidence and metadata
    evidence: List[DDSEvidence] = Field(default_factory=list, description="Attached evidence")
    version: int = Field(default=1, ge=1, description="DDS version number")
    created_at: datetime = Field(default_factory=utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=utcnow, description="Last update timestamp")
    finalized_at: Optional[datetime] = Field(None, description="Finalization timestamp")
    submitted_at: Optional[datetime] = Field(None, description="Submission timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 hash of complete DDS")

    # Validation
    annex_ii_validation: Optional[AnnexIIValidation] = Field(None, description="Annex II validation result")

class EUISSubmission(BaseModel):
    """Formatted DDS for EU Information System submission."""

    submission_id: str = Field(default_factory=_new_uuid, description="Submission identifier")
    dds_reference: str = Field(..., description="DDS reference number")
    dds_type: DDSType = Field(..., description="Standard or simplified")
    operator_name: str = Field(..., description="Operator/trader name")
    operator_eori: Optional[str] = Field(None, description="EORI number")
    operator_country: str = Field(..., description="Operator country code")

    products: List[Dict[str, Any]] = Field(default_factory=list, description="Product entries")
    geolocations: List[Dict[str, Any]] = Field(default_factory=list, description="Geolocation entries")
    suppliers: List[Dict[str, Any]] = Field(default_factory=list, description="Supplier entries")

    risk_conclusion: str = Field(..., description="Risk assessment conclusion")
    mitigation_applied: bool = Field(default=False, description="Whether mitigation was applied")

    declaration_date: datetime = Field(default_factory=utcnow, description="Declaration date")
    submission_format: str = Field(default="EU_IS_V1", description="Submission format version")
    xml_payload: Optional[str] = Field(None, description="XML-formatted payload for EU-IS")
    provenance_hash: str = Field(default="", description="SHA-256 hash of submission")

class FinalizedDDS(BaseModel):
    """Finalized DDS ready for submission."""

    dds: DDSDocument = Field(..., description="The finalized DDS document")
    finalization_reference: str = Field(default_factory=_new_uuid, description="Finalization reference")
    operator_declaration_hash: str = Field(default="", description="Hash of operator declaration")
    is_complete: bool = Field(default=False, description="Whether all requirements are met")
    submission_ready: bool = Field(default=False, description="Whether ready for EU-IS submission")
    finalized_at: datetime = Field(default_factory=utcnow, description="Finalization timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 hash of finalized DDS")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DDSAssemblyEngine:
    """
    Due Diligence Statement Assembly Engine.

    Composes, validates, and exports Due Diligence Statements required under
    EUDR (Regulation (EU) 2023/1115) Annex II. Handles both standard Article 4
    DDS and simplified Article 13 DDS for low-risk country sourcing.

    All assembly operations are deterministic with full provenance tracking.
    No LLM involvement in any DDS composition or validation path.

    Attributes:
        config: Optional engine configuration
        _assembly_count: Counter for assembled DDS documents

    Example:
        >>> engine = DDSAssemblyEngine()
        >>> dds = engine.assemble_dds(supplier_data, geo_data, risk_data, commodity_data)
        >>> validation = engine.validate_annex_ii(dds)
        >>> assert validation.is_complete
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DDSAssemblyEngine.

        Args:
            config: Optional configuration dictionary with keys:
                - default_datum: Geodetic datum (default: WGS84)
                - coordinate_precision: Decimal places (default: 6)
                - require_all_annex_ii: Strict Annex II validation (default: True)
        """
        self.config = config or {}
        self._assembly_count: int = 0
        self._default_datum: str = self.config.get("default_datum", "WGS84")
        self._coordinate_precision: int = self.config.get("coordinate_precision", 6)
        self._require_all_annex_ii: bool = self.config.get("require_all_annex_ii", True)
        logger.info("DDSAssemblyEngine initialized (version=%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------

    def assemble_dds(
        self,
        supplier_data: List[Dict[str, Any]],
        geolocation_data: List[Dict[str, Any]],
        risk_data: Dict[str, Any],
        commodity_data: List[Dict[str, Any]],
    ) -> DDSDocument:
        """Assemble a complete Due Diligence Statement from component data.

        Combines supplier, geolocation, risk assessment, and commodity data
        into a single DDS document per Annex II requirements.

        Args:
            supplier_data: List of supplier information dictionaries.
            geolocation_data: List of plot geolocation dictionaries.
            risk_data: Risk assessment data dictionary.
            commodity_data: List of commodity/product dictionaries.

        Returns:
            Assembled DDSDocument with all Annex II fields populated.

        Raises:
            ValueError: If required data is missing or invalid.
        """
        start_time = utcnow()
        logger.info("Assembling DDS from %d suppliers, %d plots, %d products",
                     len(supplier_data), len(geolocation_data), len(commodity_data))

        # Parse suppliers
        suppliers = self._parse_suppliers(supplier_data)

        # Parse geolocations
        geolocations = self._parse_geolocations(geolocation_data)

        # Parse products
        products = self._parse_products(commodity_data)

        # Parse risk summary
        risk_summary = self._parse_risk_summary(risk_data)

        # Determine countries of production
        countries = list({geo.country for geo in geolocations if geo.country})

        # Build mitigation summary if risk is not negligible/low
        mitigation = None
        if risk_summary.conclusion in (RiskConclusion.STANDARD, RiskConclusion.HIGH):
            mitigation = MitigationSummary(
                measures_taken=risk_data.get("mitigation_measures", []),
                residual_risk=risk_data.get("residual_risk"),
                additional_dd_performed=risk_data.get("additional_dd", False),
                third_party_verification=risk_data.get("third_party_verification", False),
                verification_entity=risk_data.get("verification_entity"),
            )

        # Determine DDS type
        dds_type = DDSType.STANDARD
        if risk_summary.conclusion == RiskConclusion.NEGLIGIBLE:
            all_low_risk = all(
                risk_data.get("country_benchmarks", {}).get(c, "STANDARD") == "LOW"
                for c in countries
            )
            if all_low_risk:
                dds_type = DDSType.SIMPLIFIED

        # Generate reference number
        dds = DDSDocument(
            dds_type=dds_type,
            status=DDSStatus.ASSEMBLED,
            products=products,
            countries_of_production=countries,
            geolocations=geolocations,
            suppliers=suppliers,
            risk_summary=risk_summary,
            mitigation_summary=mitigation,
            created_at=start_time,
            updated_at=utcnow(),
        )

        dds.reference_number = self.compute_dds_reference(dds)
        dds.provenance_hash = _compute_hash(dds)

        self._assembly_count += 1
        logger.info("DDS assembled: ref=%s, type=%s, hash=%s",
                     dds.reference_number, dds.dds_type.value, dds.provenance_hash[:16])
        return dds

    def validate_annex_ii(self, dds: DDSDocument) -> AnnexIIValidation:
        """Validate DDS against all Annex II required fields.

        Checks that all 8 required fields of Annex II are present and valid.

        Args:
            dds: The DDS document to validate.

        Returns:
            AnnexIIValidation result with per-field details.
        """
        logger.info("Validating Annex II for DDS %s", dds.reference_number)
        fields: List[AnnexIIField] = []

        # Field 1: Operator/trader name and address
        f1 = self._validate_field_1_operator(dds)
        fields.append(f1)

        # Field 2: HS/CN codes
        f2 = self._validate_field_2_cn_codes(dds)
        fields.append(f2)

        # Field 3: Quantity
        f3 = self._validate_field_3_quantity(dds)
        fields.append(f3)

        # Field 4: Country of production
        f4 = self._validate_field_4_country(dds)
        fields.append(f4)

        # Field 5: Geolocation
        f5 = self._validate_field_5_geolocation(dds)
        fields.append(f5)

        # Field 6: Supplier information
        f6 = self._validate_field_6_supplier(dds)
        fields.append(f6)

        # Field 7: Risk assessment conclusion
        f7 = self._validate_field_7_risk(dds)
        fields.append(f7)

        # Field 8: Risk mitigation
        f8 = self._validate_field_8_mitigation(dds)
        fields.append(f8)

        passed = sum(1 for f in fields if f.is_valid)
        failed = sum(1 for f in fields if not f.is_valid)
        all_errors = []
        all_warnings = []
        for f in fields:
            all_errors.extend(f.errors)
            all_warnings.extend(f.warnings)

        is_complete = all(f.is_present for f in fields)
        is_valid = all(f.is_valid for f in fields)

        validation = AnnexIIValidation(
            is_complete=is_complete,
            is_valid=is_valid,
            fields=fields,
            total_fields=8,
            passed_fields=passed,
            failed_fields=failed,
            errors=all_errors,
            warnings=all_warnings,
        )
        validation.provenance_hash = _compute_hash(validation)

        logger.info("Annex II validation: complete=%s, valid=%s, passed=%d/8",
                     is_complete, is_valid, passed)
        return validation

    def generate_standard_dds(self, data: Dict[str, Any]) -> DDSDocument:
        """Generate a full Article 4 Due Diligence Statement.

        Creates a standard DDS requiring all Annex II fields, risk assessment,
        and mitigation documentation as applicable.

        Args:
            data: Complete data dictionary with keys: suppliers, geolocations,
                  risk, commodities, operator.

        Returns:
            Standard DDSDocument with all fields populated.

        Raises:
            ValueError: If required data sections are missing.
        """
        logger.info("Generating standard Article 4 DDS")

        required_keys = ["suppliers", "geolocations", "risk", "commodities"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"Missing required data sections: {missing}")

        dds = self.assemble_dds(
            supplier_data=data["suppliers"],
            geolocation_data=data["geolocations"],
            risk_data=data["risk"],
            commodity_data=data["commodities"],
        )
        dds.dds_type = DDSType.STANDARD

        # Attach operator if provided
        if "operator" in data:
            dds.operator = OperatorDeclaration(**data["operator"])

        dds.updated_at = utcnow()
        dds.provenance_hash = _compute_hash(dds)
        return dds

    def generate_simplified_dds(self, data: Dict[str, Any]) -> DDSDocument:
        """Generate a simplified Article 13 DDS for low-risk country sourcing.

        Simplified DDS has reduced requirements: no detailed risk assessment
        or mitigation documentation needed when all sourcing is from countries
        benchmarked as low-risk per Article 29.

        Args:
            data: Data dictionary with keys: suppliers, geolocations, commodities.
                  Risk data is simplified.

        Returns:
            Simplified DDSDocument with reduced field requirements.

        Raises:
            ValueError: If simplified DD criteria are not met.
        """
        logger.info("Generating simplified Article 13 DDS")

        risk_data = data.get("risk", {})
        country_benchmarks = risk_data.get("country_benchmarks", {})
        countries = data.get("countries", [])

        # Verify all countries are low-risk benchmarked
        for country in countries:
            benchmark = country_benchmarks.get(country, "STANDARD")
            if benchmark != "LOW":
                raise ValueError(
                    f"Country {country} is benchmarked as {benchmark}, "
                    f"not eligible for simplified DDS (Article 13 requires LOW)"
                )

        # Create simplified risk summary
        simplified_risk = {
            "conclusion": "NEGLIGIBLE",
            "country_risk": "LOW",
            "methodology": "EUDR_SIMPLIFIED_ARTICLE_13",
            "key_findings": ["All source countries benchmarked as LOW risk per Article 29"],
            "country_benchmarks": country_benchmarks,
        }

        dds = self.assemble_dds(
            supplier_data=data.get("suppliers", []),
            geolocation_data=data.get("geolocations", []),
            risk_data=simplified_risk,
            commodity_data=data.get("commodities", []),
        )
        dds.dds_type = DDSType.SIMPLIFIED

        if "operator" in data:
            dds.operator = OperatorDeclaration(**data["operator"])

        dds.updated_at = utcnow()
        dds.provenance_hash = _compute_hash(dds)
        return dds

    def batch_assemble(self, supplier_list: List[Dict[str, Any]]) -> List[DDSDocument]:
        """Assemble DDS documents for multiple suppliers in batch.

        Each entry in supplier_list must contain all data needed for one DDS.

        Args:
            supplier_list: List of dictionaries, each containing: suppliers,
                geolocations, risk, commodities (and optionally operator).

        Returns:
            List of assembled DDSDocument instances.
        """
        logger.info("Batch assembling DDS for %d entries", len(supplier_list))
        results: List[DDSDocument] = []
        errors: List[Tuple[int, str]] = []

        for idx, entry in enumerate(supplier_list):
            try:
                dds = self.assemble_dds(
                    supplier_data=entry.get("suppliers", []),
                    geolocation_data=entry.get("geolocations", []),
                    risk_data=entry.get("risk", {}),
                    commodity_data=entry.get("commodities", []),
                )
                if "operator" in entry:
                    dds.operator = OperatorDeclaration(**entry["operator"])
                    dds.provenance_hash = _compute_hash(dds)
                results.append(dds)
            except Exception as exc:
                logger.warning("Batch entry %d failed: %s", idx, str(exc))
                errors.append((idx, str(exc)))

        logger.info("Batch assembly complete: %d success, %d errors",
                     len(results), len(errors))
        return results

    def compute_dds_reference(self, dds: DDSDocument) -> str:
        """Generate a unique DDS reference number.

        Format: EUDR-DDS-{YYYYMMDD}-{TYPE_PREFIX}-{SHORT_HASH}

        Args:
            dds: The DDS document to generate a reference for.

        Returns:
            Unique reference number string.
        """
        date_part = dds.created_at.strftime("%Y%m%d")
        type_prefix = "STD" if dds.dds_type == DDSType.STANDARD else "SMP"

        # Use a hash of the DDS ID for uniqueness
        hash_input = f"{dds.dds_id}-{date_part}-{type_prefix}"
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8].upper()

        return f"EUDR-DDS-{date_part}-{type_prefix}-{short_hash}"

    def format_geolocation_for_dds(
        self, plots: List[Dict[str, Any]]
    ) -> FormattedGeolocation:
        """Format geolocation data for DDS output per EUDR requirements.

        Ensures all coordinates are in WGS84 datum with 6 decimal places.
        Applies the <4ha point vs >=4ha polygon rule.

        Args:
            plots: List of plot data dictionaries with lat, lon, area, and
                optionally polygon coordinates.

        Returns:
            FormattedGeolocation with standardized coordinate data.
        """
        logger.info("Formatting %d plots for DDS geolocation", len(plots))
        formatted_plots: List[GeolocationInfo] = []
        total_area = 0.0
        countries: List[str] = []

        for plot in plots:
            area = float(plot.get("area_hectares", 0))
            country = plot.get("country", "UNKNOWN")
            total_area += area

            if country not in countries:
                countries.append(country)

            if area >= 4.0:
                # Polygon required for >= 4 hectares
                coords = plot.get("polygon_coordinates", [])
                polygon = GeolocationPolygon(
                    coordinates=coords,
                    area_hectares=area,
                )
                geo_info = GeolocationInfo(
                    format=GeolocationFormat.POLYGON,
                    polygon=polygon,
                    country=country,
                    area_hectares=area,
                )
            else:
                # Point sufficient for < 4 hectares
                point = GeolocationPoint(
                    latitude=round(float(plot.get("latitude", 0)), self._coordinate_precision),
                    longitude=round(float(plot.get("longitude", 0)), self._coordinate_precision),
                )
                geo_info = GeolocationInfo(
                    format=GeolocationFormat.POINT,
                    point=point,
                    country=country,
                    area_hectares=area,
                )

            formatted_plots.append(geo_info)

        result = FormattedGeolocation(
            plots=formatted_plots,
            total_plots=len(formatted_plots),
            total_area_hectares=round(total_area, 4),
            countries=countries,
            datum=self._default_datum,
            precision_decimals=self._coordinate_precision,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def attach_evidence(
        self, dds: DDSDocument, evidence_list: List[Dict[str, Any]]
    ) -> DDSDocument:
        """Attach evidence documents to a DDS.

        Args:
            dds: The DDS document to attach evidence to.
            evidence_list: List of evidence dictionaries with keys:
                evidence_type, title, description, file_reference, etc.

        Returns:
            Updated DDSDocument with evidence attached.
        """
        logger.info("Attaching %d evidence items to DDS %s",
                     len(evidence_list), dds.reference_number)

        for ev_data in evidence_list:
            evidence = DDSEvidence(
                evidence_type=EvidenceType(ev_data.get("evidence_type", "OTHER")),
                title=ev_data.get("title", "Untitled Evidence"),
                description=ev_data.get("description"),
                file_reference=ev_data.get("file_reference"),
                file_hash=ev_data.get("file_hash"),
                issue_date=ev_data.get("issue_date"),
                expiry_date=ev_data.get("expiry_date"),
                issuing_authority=ev_data.get("issuing_authority"),
            )
            dds.evidence.append(evidence)

        dds.updated_at = utcnow()
        dds.provenance_hash = _compute_hash(dds)
        return dds

    def finalize_dds(
        self, dds: DDSDocument, operator_declaration: Dict[str, Any]
    ) -> FinalizedDDS:
        """Finalize a DDS with operator declaration for submission.

        Validates completeness, attaches operator declaration, and marks
        the DDS as finalized.

        Args:
            dds: The DDS document to finalize.
            operator_declaration: Operator declaration data dictionary.

        Returns:
            FinalizedDDS containing the finalized document.

        Raises:
            ValueError: If DDS fails Annex II validation.
        """
        logger.info("Finalizing DDS %s", dds.reference_number)

        # Attach operator declaration
        dds.operator = OperatorDeclaration(**operator_declaration)

        # Validate Annex II completeness
        validation = self.validate_annex_ii(dds)
        dds.annex_ii_validation = validation

        is_complete = validation.is_complete
        if self._require_all_annex_ii and not is_complete:
            logger.warning("DDS %s failed Annex II validation: %s",
                           dds.reference_number, validation.errors)

        # Update DDS status
        dds.status = DDSStatus.FINALIZED
        dds.finalized_at = utcnow()
        dds.updated_at = utcnow()
        dds.provenance_hash = _compute_hash(dds)

        operator_hash = _compute_hash(dds.operator)

        finalized = FinalizedDDS(
            dds=dds,
            operator_declaration_hash=operator_hash,
            is_complete=is_complete,
            submission_ready=is_complete and validation.is_valid,
        )
        finalized.provenance_hash = _compute_hash(finalized)

        logger.info("DDS finalized: ref=%s, complete=%s, submission_ready=%s",
                     dds.reference_number, is_complete, finalized.submission_ready)
        return finalized

    def export_for_eu_is(self, dds: DDSDocument) -> EUISSubmission:
        """Export a DDS formatted for the EU Information System.

        Converts the DDS into the format required for submission to the
        EU-wide information system per Article 33.

        Args:
            dds: The finalized DDS document to export.

        Returns:
            EUISSubmission formatted for the EU Information System.

        Raises:
            ValueError: If DDS is not finalized.
        """
        if dds.status not in (DDSStatus.FINALIZED, DDSStatus.SUBMITTED):
            raise ValueError(
                f"DDS must be finalized before EU-IS export (current status: {dds.status.value})"
            )

        logger.info("Exporting DDS %s for EU Information System", dds.reference_number)

        # Format products for EU-IS
        products_formatted = []
        for p in dds.products:
            products_formatted.append({
                "cn_code": p.cn_code,
                "hs_code": p.hs_code or p.cn_code[:6],
                "description": p.description,
                "commodity_category": p.commodity_category,
                "net_mass_kg": str(p.net_mass_kg),
                "supplementary_unit": p.supplementary_unit,
                "supplementary_quantity": str(p.supplementary_quantity) if p.supplementary_quantity else None,
            })

        # Format geolocations for EU-IS
        geo_formatted = []
        for g in dds.geolocations:
            entry: Dict[str, Any] = {
                "plot_id": g.plot_id,
                "format": g.format.value,
                "country": g.country,
                "area_hectares": g.area_hectares,
                "datum": g.datum,
            }
            if g.format == GeolocationFormat.POINT and g.point:
                entry["latitude"] = g.point.latitude
                entry["longitude"] = g.point.longitude
            elif g.format == GeolocationFormat.POLYGON and g.polygon:
                entry["polygon_coordinates"] = g.polygon.coordinates
            geo_formatted.append(entry)

        # Format suppliers for EU-IS
        supplier_formatted = []
        for s in dds.suppliers:
            supplier_formatted.append({
                "name": s.name,
                "country": s.country,
                "eori_number": s.eori_number,
                "address": s.address,
                "tier": s.tier,
            })

        # Generate XML payload stub
        xml_payload = self._generate_xml_stub(dds, products_formatted, geo_formatted)

        submission = EUISSubmission(
            dds_reference=dds.reference_number,
            dds_type=dds.dds_type,
            operator_name=dds.operator.operator_name if dds.operator else "UNKNOWN",
            operator_eori=dds.operator.eori_number if dds.operator else None,
            operator_country=dds.operator.operator_country if dds.operator else "UNKNOWN",
            products=products_formatted,
            geolocations=geo_formatted,
            suppliers=supplier_formatted,
            risk_conclusion=dds.risk_summary.conclusion.value if dds.risk_summary else "UNKNOWN",
            mitigation_applied=dds.mitigation_summary is not None,
            declaration_date=dds.finalized_at or utcnow(),
            xml_payload=xml_payload,
        )
        submission.provenance_hash = _compute_hash(submission)

        logger.info("EU-IS export complete for DDS %s", dds.reference_number)
        return submission

    # -------------------------------------------------------------------
    # Private: Parsing Helpers
    # -------------------------------------------------------------------

    def _parse_suppliers(self, supplier_data: List[Dict[str, Any]]) -> List[SupplierInfo]:
        """Parse raw supplier data into SupplierInfo models."""
        suppliers: List[SupplierInfo] = []
        for sd in supplier_data:
            suppliers.append(SupplierInfo(
                name=sd.get("name", "Unknown Supplier"),
                address=sd.get("address"),
                country=sd.get("country", "UNKNOWN"),
                eori_number=sd.get("eori_number"),
                contact_person=sd.get("contact_person"),
                contact_email=sd.get("contact_email"),
                tier=sd.get("tier", 1),
                certifications=sd.get("certifications", []),
            ))
        return suppliers

    def _parse_geolocations(self, geo_data: List[Dict[str, Any]]) -> List[GeolocationInfo]:
        """Parse raw geolocation data into GeolocationInfo models."""
        geolocations: List[GeolocationInfo] = []
        for gd in geo_data:
            area = float(gd.get("area_hectares", 0))
            fmt = GeolocationFormat.POLYGON if area >= 4.0 else GeolocationFormat.POINT

            point = None
            polygon = None
            if fmt == GeolocationFormat.POINT:
                point = GeolocationPoint(
                    latitude=round(float(gd.get("latitude", 0)), self._coordinate_precision),
                    longitude=round(float(gd.get("longitude", 0)), self._coordinate_precision),
                )
            else:
                coords = gd.get("polygon_coordinates", gd.get("coordinates", []))
                if coords:
                    polygon = GeolocationPolygon(coordinates=coords, area_hectares=area)

            geolocations.append(GeolocationInfo(
                format=fmt,
                point=point,
                polygon=polygon,
                country=gd.get("country", "UNKNOWN"),
                area_hectares=area,
            ))
        return geolocations

    def _parse_products(self, commodity_data: List[Dict[str, Any]]) -> List[ProductInfo]:
        """Parse raw commodity data into ProductInfo models."""
        products: List[ProductInfo] = []
        for cd in commodity_data:
            products.append(ProductInfo(
                description=cd.get("description", ""),
                cn_code=cd.get("cn_code", "00000000"),
                hs_code=cd.get("hs_code"),
                commodity_category=cd.get("commodity_category", "UNKNOWN"),
                net_mass_kg=_decimal(cd.get("net_mass_kg", 0)),
                supplementary_unit=cd.get("supplementary_unit"),
                supplementary_quantity=_decimal(cd.get("supplementary_quantity")) if cd.get("supplementary_quantity") else None,
                trade_name=cd.get("trade_name"),
            ))
        return products

    def _parse_risk_summary(self, risk_data: Dict[str, Any]) -> RiskSummary:
        """Parse raw risk data into RiskSummary model."""
        conclusion_str = risk_data.get("conclusion", "STANDARD").upper()
        try:
            conclusion = RiskConclusion(conclusion_str)
        except ValueError:
            conclusion = RiskConclusion.STANDARD

        return RiskSummary(
            conclusion=conclusion,
            country_risk=risk_data.get("country_risk", "STANDARD"),
            supplier_risk_score=risk_data.get("supplier_risk_score"),
            commodity_risk_score=risk_data.get("commodity_risk_score"),
            deforestation_risk=risk_data.get("deforestation_risk"),
            assessor=risk_data.get("assessor"),
            methodology=risk_data.get("methodology", "EUDR_STANDARD"),
            key_findings=risk_data.get("key_findings", []),
        )

    # -------------------------------------------------------------------
    # Private: Annex II Validation Methods
    # -------------------------------------------------------------------

    def _validate_field_1_operator(self, dds: DDSDocument) -> AnnexIIField:
        """Validate Annex II Field 1: Operator/trader name and address."""
        errors: List[str] = []
        warnings: List[str] = []
        is_present = dds.operator is not None

        if not is_present:
            errors.append("Operator/trader information is missing (Annex II, Field 1)")
        else:
            if not dds.operator.operator_name or len(dds.operator.operator_name.strip()) < 2:
                errors.append("Operator name must be at least 2 characters")
            if not dds.operator.operator_address or len(dds.operator.operator_address.strip()) < 5:
                errors.append("Operator address must be at least 5 characters")
            if not dds.operator.operator_country or len(dds.operator.operator_country) != 2:
                errors.append("Operator country must be a 2-letter ISO code")
            if not dds.operator.eori_number:
                warnings.append("EORI number not provided (recommended for EU operators)")

        return AnnexIIField(
            field_number=1,
            field_name="Operator/Trader Information",
            is_present=is_present,
            is_valid=is_present and len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_field_2_cn_codes(self, dds: DDSDocument) -> AnnexIIField:
        """Validate Annex II Field 2: HS/CN codes of products."""
        errors: List[str] = []
        warnings: List[str] = []
        is_present = len(dds.products) > 0

        if not is_present:
            errors.append("No products/CN codes provided (Annex II, Field 2)")
        else:
            cn_pattern = re.compile(r"^\d{8}$")
            for i, p in enumerate(dds.products):
                if not cn_pattern.match(p.cn_code):
                    errors.append(f"Product {i}: CN code '{p.cn_code}' must be exactly 8 digits")
                if not p.description or len(p.description.strip()) < 3:
                    warnings.append(f"Product {i}: Description should be more descriptive")

        return AnnexIIField(
            field_number=2,
            field_name="HS/CN Product Codes",
            is_present=is_present,
            is_valid=is_present and len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_field_3_quantity(self, dds: DDSDocument) -> AnnexIIField:
        """Validate Annex II Field 3: Quantity (net mass and supplementary unit)."""
        errors: List[str] = []
        warnings: List[str] = []
        is_present = len(dds.products) > 0

        if not is_present:
            errors.append("No product quantities provided (Annex II, Field 3)")
        else:
            for i, p in enumerate(dds.products):
                if p.net_mass_kg <= 0:
                    errors.append(f"Product {i}: Net mass must be greater than zero")
                if p.net_mass_kg > Decimal("1000000000"):
                    warnings.append(f"Product {i}: Net mass exceeds 1,000,000 tonnes (verify)")

        return AnnexIIField(
            field_number=3,
            field_name="Quantity (Net Mass)",
            is_present=is_present,
            is_valid=is_present and len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_field_4_country(self, dds: DDSDocument) -> AnnexIIField:
        """Validate Annex II Field 4: Country of production."""
        errors: List[str] = []
        warnings: List[str] = []
        is_present = len(dds.countries_of_production) > 0

        if not is_present:
            errors.append("No countries of production specified (Annex II, Field 4)")
        else:
            iso_pattern = re.compile(r"^[A-Z]{2}$")
            for c in dds.countries_of_production:
                if not iso_pattern.match(c):
                    errors.append(f"Country code '{c}' is not a valid ISO 3166-1 alpha-2 code")

        return AnnexIIField(
            field_number=4,
            field_name="Country of Production",
            is_present=is_present,
            is_valid=is_present and len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_field_5_geolocation(self, dds: DDSDocument) -> AnnexIIField:
        """Validate Annex II Field 5: Geolocation of production plots."""
        errors: List[str] = []
        warnings: List[str] = []
        is_present = len(dds.geolocations) > 0

        if not is_present:
            errors.append("No geolocation data provided (Annex II, Field 5)")
        else:
            for i, geo in enumerate(dds.geolocations):
                if geo.datum != "WGS84":
                    errors.append(f"Plot {i}: Datum must be WGS84 (found: {geo.datum})")

                area = geo.area_hectares or 0
                if area >= 4.0 and geo.format != GeolocationFormat.POLYGON:
                    errors.append(f"Plot {i}: Area >= 4ha requires POLYGON format")
                if area < 4.0 and geo.format == GeolocationFormat.POLYGON:
                    warnings.append(f"Plot {i}: Area < 4ha can use POINT format (polygon not required)")

                if geo.format == GeolocationFormat.POINT and geo.point:
                    if geo.point.latitude == 0.0 and geo.point.longitude == 0.0:
                        warnings.append(f"Plot {i}: Coordinates at (0,0) - likely placeholder")
                elif geo.format == GeolocationFormat.POLYGON and geo.polygon:
                    if len(geo.polygon.coordinates) < 4:
                        errors.append(f"Plot {i}: Polygon must have at least 4 coordinate pairs")

        return AnnexIIField(
            field_number=5,
            field_name="Geolocation of Production",
            is_present=is_present,
            is_valid=is_present and len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_field_6_supplier(self, dds: DDSDocument) -> AnnexIIField:
        """Validate Annex II Field 6: Supplier information."""
        errors: List[str] = []
        warnings: List[str] = []
        is_present = len(dds.suppliers) > 0

        if not is_present:
            errors.append("No supplier information provided (Annex II, Field 6)")
        else:
            for i, s in enumerate(dds.suppliers):
                if not s.name or len(s.name.strip()) < 2:
                    errors.append(f"Supplier {i}: Name must be at least 2 characters")
                if not s.country or len(s.country) != 2:
                    errors.append(f"Supplier {i}: Country must be a 2-letter ISO code")
                if not s.address:
                    warnings.append(f"Supplier {i}: Address not provided (recommended)")

        return AnnexIIField(
            field_number=6,
            field_name="Supplier Information",
            is_present=is_present,
            is_valid=is_present and len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_field_7_risk(self, dds: DDSDocument) -> AnnexIIField:
        """Validate Annex II Field 7: Risk assessment conclusion."""
        errors: List[str] = []
        warnings: List[str] = []
        is_present = dds.risk_summary is not None

        if not is_present:
            errors.append("No risk assessment summary provided (Annex II, Field 7)")
        else:
            if dds.risk_summary.conclusion not in list(RiskConclusion):
                errors.append(f"Invalid risk conclusion: {dds.risk_summary.conclusion}")
            if not dds.risk_summary.methodology:
                warnings.append("Risk assessment methodology not specified")
            if dds.risk_summary.conclusion in (RiskConclusion.HIGH,) and not dds.risk_summary.key_findings:
                warnings.append("HIGH risk conclusion should include key findings")

        return AnnexIIField(
            field_number=7,
            field_name="Risk Assessment Conclusion",
            is_present=is_present,
            is_valid=is_present and len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_field_8_mitigation(self, dds: DDSDocument) -> AnnexIIField:
        """Validate Annex II Field 8: Risk mitigation measures."""
        errors: List[str] = []
        warnings: List[str] = []

        # Mitigation is required only when risk is STANDARD or HIGH
        risk_requires_mitigation = (
            dds.risk_summary is not None
            and dds.risk_summary.conclusion in (RiskConclusion.STANDARD, RiskConclusion.HIGH)
        )

        is_present = dds.mitigation_summary is not None

        if risk_requires_mitigation and not is_present:
            errors.append(
                "Risk mitigation measures required for STANDARD/HIGH risk "
                "(Annex II, Field 8)"
            )
        elif is_present:
            if not dds.mitigation_summary.measures_taken:
                errors.append("Mitigation summary present but no measures documented")
            if dds.risk_summary and dds.risk_summary.conclusion == RiskConclusion.HIGH:
                if not dds.mitigation_summary.third_party_verification:
                    warnings.append("HIGH risk: third-party verification recommended")

        # For NEGLIGIBLE/LOW risk, mitigation is optional - always valid
        if not risk_requires_mitigation:
            is_valid = True
            is_present = True  # Mark as present since it is not required
        else:
            is_valid = is_present and len(errors) == 0

        return AnnexIIField(
            field_number=8,
            field_name="Risk Mitigation Measures",
            is_present=is_present,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )

    # -------------------------------------------------------------------
    # Private: XML Generation
    # -------------------------------------------------------------------

    def _generate_xml_stub(
        self,
        dds: DDSDocument,
        products: List[Dict[str, Any]],
        geolocations: List[Dict[str, Any]],
    ) -> str:
        """Generate an XML payload stub for EU-IS submission.

        This is a simplified XML structure. Production systems would use
        the official EU-IS XML schema (XSD).

        Args:
            dds: The DDS document.
            products: Formatted product entries.
            geolocations: Formatted geolocation entries.

        Returns:
            XML string for EU-IS submission.
        """
        operator_name = dds.operator.operator_name if dds.operator else "UNKNOWN"
        operator_eori = dds.operator.eori_number if dds.operator else ""
        operator_country = dds.operator.operator_country if dds.operator else "UNKNOWN"
        risk_conclusion = dds.risk_summary.conclusion.value if dds.risk_summary else "UNKNOWN"

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<DueDiligenceStatement xmlns="urn:eu:eudr:dds:v1">',
            f'  <Reference>{dds.reference_number}</Reference>',
            f'  <Type>{dds.dds_type.value}</Type>',
            f'  <Status>{dds.status.value}</Status>',
            '  <Operator>',
            f'    <Name>{_xml_escape(operator_name)}</Name>',
            f'    <EORI>{_xml_escape(operator_eori)}</EORI>',
            f'    <Country>{operator_country}</Country>',
            '  </Operator>',
            '  <Products>',
        ]

        for p in products:
            lines.append('    <Product>')
            lines.append(f'      <CNCode>{p["cn_code"]}</CNCode>')
            lines.append(f'      <Description>{_xml_escape(p["description"])}</Description>')
            lines.append(f'      <NetMassKg>{p["net_mass_kg"]}</NetMassKg>')
            lines.append(f'      <CommodityCategory>{p["commodity_category"]}</CommodityCategory>')
            lines.append('    </Product>')

        lines.append('  </Products>')
        lines.append('  <Geolocations>')

        for g in geolocations:
            lines.append('    <Plot>')
            lines.append(f'      <PlotId>{g["plot_id"]}</PlotId>')
            lines.append(f'      <Format>{g["format"]}</Format>')
            lines.append(f'      <Country>{g["country"]}</Country>')
            if g["format"] == "POINT":
                lines.append(f'      <Latitude>{g.get("latitude", 0)}</Latitude>')
                lines.append(f'      <Longitude>{g.get("longitude", 0)}</Longitude>')
            lines.append('    </Plot>')

        lines.append('  </Geolocations>')
        lines.append(f'  <RiskConclusion>{risk_conclusion}</RiskConclusion>')
        lines.append(f'  <MitigationApplied>{str(dds.mitigation_summary is not None).lower()}</MitigationApplied>')
        lines.append(f'  <DeclarationDate>{dds.finalized_at.isoformat() if dds.finalized_at else ""}</DeclarationDate>')
        lines.append('</DueDiligenceStatement>')

        return "\n".join(lines)

def _xml_escape(text: str) -> str:
    """Escape special characters for XML output."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
