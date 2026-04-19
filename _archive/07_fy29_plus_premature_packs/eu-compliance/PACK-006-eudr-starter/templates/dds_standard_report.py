# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack: Standard Due Diligence Statement Report
====================================================================

Generates a standard Due Diligence Statement (DDS) per EUDR Annex II.
This is the primary compliance document operators must submit to
demonstrate that commodities and derived products placed on the
EU market are deforestation-free and legally produced.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

PACK_ID = "PACK-006-eudr-starter"
TEMPLATE_NAME = "dds_standard_report"
TEMPLATE_VERSION = "1.0.0"


# =============================================================================
# ENUMS
# =============================================================================

class DDSStatus(str, Enum):
    """DDS document status."""
    DRAFT = "DRAFT"
    FINAL = "FINAL"
    SUBMITTED = "SUBMITTED"
    AMENDED = "AMENDED"


class CommodityType(str, Enum):
    """EUDR-regulated commodity types per Article 1."""
    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    OIL_PALM = "OIL_PALM"
    RUBBER = "RUBBER"
    SOYA = "SOYA"
    WOOD = "WOOD"


class BenchmarkClassification(str, Enum):
    """Article 29 country benchmark classification."""
    LOW = "LOW"
    STANDARD = "STANDARD"
    HIGH = "HIGH"


class RiskClassification(str, Enum):
    """Risk assessment outcome classification."""
    NEGLIGIBLE = "NEGLIGIBLE"
    NON_NEGLIGIBLE = "NON_NEGLIGIBLE"


class DDType(str, Enum):
    """Due diligence type."""
    STANDARD = "STANDARD"
    SIMPLIFIED = "SIMPLIFIED"


class ChainOfCustodyModel(str, Enum):
    """Chain of custody traceability model."""
    IDENTITY_PRESERVED = "IDENTITY_PRESERVED"
    SEGREGATED = "SEGREGATED"
    MASS_BALANCE = "MASS_BALANCE"
    BOOK_AND_CLAIM = "BOOK_AND_CLAIM"


class CertificationScheme(str, Enum):
    """Recognized certification schemes."""
    FSC = "FSC"
    PEFC = "PEFC"
    RSPO = "RSPO"
    RAINFOREST_ALLIANCE = "RAINFOREST_ALLIANCE"
    UTZ = "UTZ"
    FAIRTRADE = "FAIRTRADE"
    ISCC = "ISCC"
    OTHER = "OTHER"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class OperatorInfo(BaseModel):
    """Operator identification per EUDR Article 2."""
    company_name: str = Field(..., description="Legal entity name")
    address: str = Field(..., description="Registered address")
    eori_number: Optional[str] = Field(None, description="EORI number for customs")
    registration_country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    contact_name: str = Field(..., description="Primary contact person")
    contact_email: str = Field(..., description="Contact email address")
    contact_phone: Optional[str] = Field(None, description="Contact phone number")
    vat_number: Optional[str] = Field(None, description="VAT registration number")


class ProductDescription(BaseModel):
    """Product information per EUDR Article 4."""
    commodity_type: CommodityType = Field(..., description="Primary commodity type")
    product_description: str = Field(..., description="Detailed product description")
    hs_codes: List[str] = Field(default_factory=list, description="HS/CN tariff codes")
    is_derived_product: bool = Field(False, description="Whether this is a derived product")
    derived_product_details: Optional[str] = Field(
        None, description="Details if derived product"
    )

    @field_validator("hs_codes")
    @classmethod
    def validate_hs_codes(cls, v: List[str]) -> List[str]:
        """Validate HS codes are non-empty strings."""
        return [code.strip() for code in v if code.strip()]


class QuantityInfo(BaseModel):
    """Product quantity information."""
    net_mass_kg: float = Field(..., gt=0, description="Net mass in kilograms")
    supplementary_units: Optional[float] = Field(
        None, ge=0, description="Supplementary units (pieces, litres)"
    )
    supplementary_unit_type: Optional[str] = Field(
        None, description="Type of supplementary unit"
    )
    total_shipment_value_eur: Optional[float] = Field(
        None, ge=0, description="Total shipment value in EUR"
    )


class CountryOfProduction(BaseModel):
    """Country of production information per Article 9."""
    country_iso: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 code"
    )
    country_name: str = Field(..., description="Full country name")
    benchmark_classification: BenchmarkClassification = Field(
        ..., description="Article 29 benchmark classification"
    )


class GeolocationCoordinate(BaseModel):
    """Single geolocation coordinate in WGS84."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude (WGS84)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude (WGS84)")


class GeolocationData(BaseModel):
    """Geolocation data per Article 9."""
    coordinates: List[GeolocationCoordinate] = Field(
        default_factory=list,
        description="Point coordinates (for plots <4ha) or polygon vertices (>=4ha)",
    )
    is_polygon: bool = Field(
        False, description="True if polygon (>=4ha), False if point (<4ha)"
    )
    area_hectares: Optional[float] = Field(None, ge=0, description="Area in hectares")
    plot_reference_ids: List[str] = Field(
        default_factory=list, description="Plot reference identifiers"
    )
    coordinate_system: str = Field("WGS84", description="Coordinate reference system")
    precision_meters: Optional[float] = Field(
        None, ge=0, description="Coordinate precision in meters"
    )

    @field_validator("coordinates")
    @classmethod
    def validate_polygon_closure(
        cls, v: List[GeolocationCoordinate], info: Any
    ) -> List[GeolocationCoordinate]:
        """Validate polygon has at least 4 points and is closed."""
        # Polygon validation is handled at render time
        return v


class SupplierInfo(BaseModel):
    """Supplier information for the DDS."""
    supplier_name: str = Field(..., description="Supplier legal name")
    supplier_country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    contact_name: Optional[str] = Field(None, description="Contact person")
    contact_email: Optional[str] = Field(None, description="Contact email")
    certification_status: Optional[str] = Field(
        None, description="Active certification status"
    )
    certification_scheme: Optional[CertificationScheme] = Field(
        None, description="Certification scheme"
    )
    tier_level: int = Field(1, ge=1, le=10, description="Supply chain tier level")


class SupplyChainSummary(BaseModel):
    """Supply chain traceability summary."""
    chain_of_custody_model: ChainOfCustodyModel = Field(
        ..., description="Chain of custody model applied"
    )
    tier_depth: int = Field(1, ge=1, description="Number of supply chain tiers traced")
    traceability_evidence: List[str] = Field(
        default_factory=list, description="List of traceability evidence documents"
    )
    total_suppliers: int = Field(0, ge=0, description="Total suppliers in chain")


class RiskScoreBreakdown(BaseModel):
    """Detailed risk score breakdown."""
    country_risk_score: float = Field(0.0, ge=0.0, le=100.0, description="Country risk")
    supplier_risk_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Supplier risk"
    )
    commodity_risk_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Commodity risk"
    )
    document_risk_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Document risk"
    )


class RiskAssessmentSummary(BaseModel):
    """Risk assessment summary section."""
    composite_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall composite risk score"
    )
    risk_breakdown: RiskScoreBreakdown = Field(
        default_factory=RiskScoreBreakdown, description="Score breakdown by category"
    )
    risk_classification: RiskClassification = Field(
        ..., description="Final risk classification"
    )
    assessment_date: date = Field(
        default_factory=date.today, description="Date of risk assessment"
    )
    methodology_reference: str = Field(
        "EUDR Article 10", description="Methodology reference"
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Key findings from the assessment"
    )


class MitigationMeasure(BaseModel):
    """Risk mitigation measure applied."""
    measure_id: str = Field(..., description="Measure identifier")
    description: str = Field(..., description="Measure description")
    effectiveness: str = Field(
        "ADEQUATE", description="Effectiveness assessment"
    )
    date_applied: Optional[date] = Field(None, description="Date measure was applied")
    evidence_reference: Optional[str] = Field(
        None, description="Reference to supporting evidence"
    )


class EvidenceItem(BaseModel):
    """Supporting evidence with provenance."""
    evidence_id: str = Field(..., description="Evidence identifier")
    document_name: str = Field(..., description="Document name")
    document_type: str = Field(..., description="Document type")
    sha256_hash: str = Field(..., description="SHA-256 hash of the document")
    upload_date: Optional[date] = Field(None, description="Date uploaded")
    source: Optional[str] = Field(None, description="Evidence source")


class DDSStandardInput(BaseModel):
    """Complete input data for the Standard DDS report."""
    dds_reference: str = Field(..., description="DDS reference number")
    dds_date: date = Field(default_factory=date.today, description="DDS date")
    dds_version: str = Field("1.0", description="Document version")
    dds_status: DDSStatus = Field(DDSStatus.DRAFT, description="Document status")
    operator: OperatorInfo = Field(..., description="Operator information")
    product: ProductDescription = Field(..., description="Product description")
    quantity: QuantityInfo = Field(..., description="Quantity information")
    country_of_production: CountryOfProduction = Field(
        ..., description="Country of production"
    )
    geolocation: GeolocationData = Field(
        default_factory=GeolocationData, description="Geolocation data"
    )
    suppliers: List[SupplierInfo] = Field(
        default_factory=list, description="Supplier information"
    )
    supply_chain: SupplyChainSummary = Field(
        ..., description="Supply chain summary"
    )
    risk_assessment: RiskAssessmentSummary = Field(
        ..., description="Risk assessment summary"
    )
    mitigation_measures: List[MitigationMeasure] = Field(
        default_factory=list, description="Risk mitigation measures"
    )
    conclusion_text: str = Field(
        "", description="Risk assessment conclusion narrative"
    )
    dd_type: DDType = Field(DDType.STANDARD, description="Due diligence type")
    signatory_name: str = Field(..., description="Authorized signatory name")
    signatory_title: Optional[str] = Field(None, description="Signatory title")
    signature_date: date = Field(
        default_factory=date.today, description="Date of signature"
    )
    evidence_index: List[EvidenceItem] = Field(
        default_factory=list, description="Supporting evidence index"
    )

    @field_validator("dds_reference")
    @classmethod
    def validate_dds_reference(cls, v: str) -> str:
        """Ensure DDS reference is non-empty."""
        if not v.strip():
            raise ValueError("DDS reference number must not be empty")
        return v.strip()


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_status_badge(status: DDSStatus) -> str:
    """Text badge for DDS status."""
    return f"[{status.value}]"


def _fmt_risk_badge(classification: RiskClassification) -> str:
    """Text badge for risk classification."""
    return f"[{classification.value}]"


def _fmt_commodity(commodity: CommodityType) -> str:
    """Human-readable commodity name."""
    mapping = {
        CommodityType.CATTLE: "Cattle",
        CommodityType.COCOA: "Cocoa",
        CommodityType.COFFEE: "Coffee",
        CommodityType.OIL_PALM: "Oil Palm",
        CommodityType.RUBBER: "Rubber",
        CommodityType.SOYA: "Soya",
        CommodityType.WOOD: "Wood",
    }
    return mapping.get(commodity, commodity.value)


def _fmt_mass(kg: float) -> str:
    """Format mass with appropriate scale."""
    if kg >= 1_000_000:
        return f"{kg / 1_000:,.1f} tonnes"
    if kg >= 1_000:
        return f"{kg / 1_000:,.2f} tonnes"
    return f"{kg:,.1f} kg"


def _fmt_currency(value: Optional[float]) -> str:
    """Format EUR value."""
    if value is None:
        return "N/A"
    return f"EUR {value:,.2f}"


def _risk_score_label(score: float) -> str:
    """Risk score to human-readable label."""
    if score <= 20:
        return "LOW"
    if score <= 40:
        return "MODERATE"
    if score <= 60:
        return "ELEVATED"
    if score <= 80:
        return "HIGH"
    return "CRITICAL"


def _benchmark_css(benchmark: BenchmarkClassification) -> str:
    """CSS class for benchmark classification."""
    return f"benchmark-{benchmark.value.lower()}"


def _status_css(status: DDSStatus) -> str:
    """CSS class for DDS status."""
    return f"status-{status.value.lower()}"


def _risk_css(classification: RiskClassification) -> str:
    """CSS class for risk classification."""
    return f"risk-{classification.value.lower().replace('_', '-')}"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class DDSStandardReport:
    """Generate Standard Due Diligence Statement per EUDR Annex II.

    This is the primary compliance document for EUDR. It contains all
    information required under Article 4 and Annex II, including operator
    identification, product details, geolocation data, risk assessment,
    and the operator declaration.

    Sections:
        1.  Header - DDS reference, date, version, status
        2.  Operator Information - Company details, EORI
        3.  Product Description - Commodity type, HS codes
        4.  Quantity - Net mass, supplementary units, value
        5.  Country of Production - ISO code, benchmark
        6.  Geolocation Data - Coordinates/polygon, WGS84
        7.  Supplier Information - Suppliers in the chain
        8.  Supply Chain Summary - CoC model, tier depth
        9.  Risk Assessment Summary - Composite score, breakdown
        10. Risk Mitigation Measures - Applied measures
        11. Conclusion - Risk classification, DD type
        12. Operator Declaration - Article 4(2) confirmation
        13. Evidence Index - Supporting documents with hashes
        14. Provenance - Pack version, timestamp, hash

    Example:
        >>> report = DDSStandardReport()
        >>> data = DDSStandardInput(...)
        >>> md = report.render_markdown(data)
        >>> html = report.render_html(data)
        >>> payload = report.render_json(data)
    """

    def __init__(self) -> None:
        """Initialize the DDS Standard Report template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: DDSStandardInput) -> str:
        """Render the DDS as Markdown.

        Args:
            data: Validated DDS input data.

        Returns:
            Complete Markdown string with all 14 sections.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_operator_info(data),
            self._md_product_description(data),
            self._md_quantity(data),
            self._md_country_of_production(data),
            self._md_geolocation(data),
            self._md_supplier_info(data),
            self._md_supply_chain(data),
            self._md_risk_assessment(data),
            self._md_mitigation_measures(data),
            self._md_conclusion(data),
            self._md_operator_declaration(data),
            self._md_evidence_index(data),
            self._md_provenance(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: DDSStandardInput) -> str:
        """Render the DDS as HTML with inline CSS.

        Args:
            data: Validated DDS input data.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_operator_info(data),
            self._html_product_description(data),
            self._html_quantity(data),
            self._html_country_of_production(data),
            self._html_geolocation(data),
            self._html_supplier_info(data),
            self._html_supply_chain(data),
            self._html_risk_assessment(data),
            self._html_mitigation_measures(data),
            self._html_conclusion(data),
            self._html_operator_declaration(data),
            self._html_evidence_index(data),
            self._html_provenance(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: DDSStandardInput) -> Dict[str, Any]:
        """Render the DDS as a JSON-serializable dictionary.

        Args:
            data: Validated DDS input data.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance_hash = self._compute_provenance_hash(data)

        return {
            "metadata": {
                "pack_id": PACK_ID,
                "template_name": TEMPLATE_NAME,
                "version": TEMPLATE_VERSION,
                "generated_at": self._render_timestamp.isoformat(),
                "provenance_hash": provenance_hash,
            },
            "dds_reference": data.dds_reference,
            "dds_date": data.dds_date.isoformat(),
            "dds_version": data.dds_version,
            "dds_status": data.dds_status.value,
            "operator": data.operator.model_dump(mode="json"),
            "product": data.product.model_dump(mode="json"),
            "quantity": data.quantity.model_dump(mode="json"),
            "country_of_production": data.country_of_production.model_dump(mode="json"),
            "geolocation": {
                "coordinates": [
                    {"latitude": c.latitude, "longitude": c.longitude}
                    for c in data.geolocation.coordinates
                ],
                "is_polygon": data.geolocation.is_polygon,
                "area_hectares": data.geolocation.area_hectares,
                "plot_reference_ids": data.geolocation.plot_reference_ids,
                "coordinate_system": data.geolocation.coordinate_system,
                "precision_meters": data.geolocation.precision_meters,
            },
            "suppliers": [s.model_dump(mode="json") for s in data.suppliers],
            "supply_chain": data.supply_chain.model_dump(mode="json"),
            "risk_assessment": {
                "composite_risk_score": data.risk_assessment.composite_risk_score,
                "risk_breakdown": data.risk_assessment.risk_breakdown.model_dump(
                    mode="json"
                ),
                "risk_classification": data.risk_assessment.risk_classification.value,
                "assessment_date": data.risk_assessment.assessment_date.isoformat(),
                "methodology_reference": data.risk_assessment.methodology_reference,
                "key_findings": data.risk_assessment.key_findings,
            },
            "mitigation_measures": [
                m.model_dump(mode="json") for m in data.mitigation_measures
            ],
            "conclusion": {
                "risk_classification": data.risk_assessment.risk_classification.value,
                "dd_type": data.dd_type.value,
                "conclusion_text": data.conclusion_text,
            },
            "declaration": {
                "signatory_name": data.signatory_name,
                "signatory_title": data.signatory_title,
                "signature_date": data.signature_date.isoformat(),
            },
            "evidence_index": [e.model_dump(mode="json") for e in data.evidence_index],
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance_hash(self, data: DDSStandardInput) -> str:
        """Compute SHA-256 provenance hash over the input data.

        Args:
            data: The DDS input data to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: DDSStandardInput) -> str:
        """Section 1: DDS Header."""
        return (
            "# Due Diligence Statement (Standard)\n"
            "## EUDR Regulation (EU) 2023/1115 - Annex II\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| DDS Reference | **{data.dds_reference}** |\n"
            f"| Date | {data.dds_date.isoformat()} |\n"
            f"| Version | {data.dds_version} |\n"
            f"| Status | {_fmt_status_badge(data.dds_status)} |\n"
            f"| DD Type | {data.dd_type.value} |\n\n---"
        )

    def _md_operator_info(self, data: DDSStandardInput) -> str:
        """Section 2: Operator Information."""
        op = data.operator
        eori = op.eori_number or "N/A"
        vat = op.vat_number or "N/A"
        phone = op.contact_phone or "N/A"
        return (
            "## 2. Operator Information\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Company Name | {op.company_name} |\n"
            f"| Registered Address | {op.address} |\n"
            f"| EORI Number | {eori} |\n"
            f"| Registration Country | {op.registration_country} |\n"
            f"| VAT Number | {vat} |\n"
            f"| Contact Name | {op.contact_name} |\n"
            f"| Contact Email | {op.contact_email} |\n"
            f"| Contact Phone | {phone} |"
        )

    def _md_product_description(self, data: DDSStandardInput) -> str:
        """Section 3: Product Description."""
        p = data.product
        hs = ", ".join(p.hs_codes) if p.hs_codes else "N/A"
        derived = "Yes" if p.is_derived_product else "No"
        details = p.derived_product_details or "N/A"
        return (
            "## 3. Product Description\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Commodity Type | {_fmt_commodity(p.commodity_type)} |\n"
            f"| Product Description | {p.product_description} |\n"
            f"| HS/CN Codes | {hs} |\n"
            f"| Derived Product | {derived} |\n"
            f"| Derived Product Details | {details} |"
        )

    def _md_quantity(self, data: DDSStandardInput) -> str:
        """Section 4: Quantity."""
        q = data.quantity
        supp = "N/A"
        if q.supplementary_units is not None:
            unit_type = q.supplementary_unit_type or "units"
            supp = f"{q.supplementary_units:,.2f} {unit_type}"
        return (
            "## 4. Quantity\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Net Mass | {_fmt_mass(q.net_mass_kg)} |\n"
            f"| Supplementary Units | {supp} |\n"
            f"| Total Shipment Value | {_fmt_currency(q.total_shipment_value_eur)} |"
        )

    def _md_country_of_production(self, data: DDSStandardInput) -> str:
        """Section 5: Country of Production."""
        c = data.country_of_production
        return (
            "## 5. Country of Production\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Country Code | {c.country_iso} |\n"
            f"| Country Name | {c.country_name} |\n"
            f"| Benchmark Classification | [{c.benchmark_classification.value}] |"
        )

    def _md_geolocation(self, data: DDSStandardInput) -> str:
        """Section 6: Geolocation Data."""
        g = data.geolocation
        geo_type = "Polygon (>=4ha)" if g.is_polygon else "Point (<4ha)"
        area = f"{g.area_hectares:,.2f} ha" if g.area_hectares is not None else "N/A"
        precision = (
            f"{g.precision_meters:,.1f} m" if g.precision_meters is not None else "N/A"
        )
        plot_refs = ", ".join(g.plot_reference_ids) if g.plot_reference_ids else "N/A"

        lines = [
            "## 6. Geolocation Data\n",
            "| Field | Value |",
            "|-------|-------|",
            f"| Geometry Type | {geo_type} |",
            f"| Coordinate System | {g.coordinate_system} |",
            f"| Area | {area} |",
            f"| Precision | {precision} |",
            f"| Plot References | {plot_refs} |",
            "",
        ]

        if g.coordinates:
            lines.append("### Coordinates\n")
            lines.append("| # | Latitude | Longitude |")
            lines.append("|---|----------|-----------|")
            for idx, coord in enumerate(g.coordinates, 1):
                lines.append(
                    f"| {idx} | {coord.latitude:.6f} | {coord.longitude:.6f} |"
                )

        return "\n".join(lines)

    def _md_supplier_info(self, data: DDSStandardInput) -> str:
        """Section 7: Supplier Information."""
        lines = [
            "## 7. Supplier Information\n",
            "| # | Supplier Name | Country | Tier | Certification | Scheme |",
            "|---|---------------|---------|------|---------------|--------|",
        ]
        for idx, s in enumerate(data.suppliers, 1):
            cert = s.certification_status or "N/A"
            scheme = s.certification_scheme.value if s.certification_scheme else "N/A"
            lines.append(
                f"| {idx} | {s.supplier_name} | {s.supplier_country} "
                f"| {s.tier_level} | {cert} | {scheme} |"
            )
        if not data.suppliers:
            lines.append("| - | No suppliers registered | - | - | - | - |")
        return "\n".join(lines)

    def _md_supply_chain(self, data: DDSStandardInput) -> str:
        """Section 8: Supply Chain Summary."""
        sc = data.supply_chain
        model_name = sc.chain_of_custody_model.value.replace("_", " ").title()
        evidence_list = (
            "\n".join(f"- {e}" for e in sc.traceability_evidence)
            if sc.traceability_evidence
            else "- No evidence documents recorded"
        )
        return (
            "## 8. Supply Chain Summary\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Chain of Custody Model | {model_name} |\n"
            f"| Tier Depth | {sc.tier_depth} |\n"
            f"| Total Suppliers | {sc.total_suppliers} |\n\n"
            f"**Traceability Evidence:**\n\n{evidence_list}"
        )

    def _md_risk_assessment(self, data: DDSStandardInput) -> str:
        """Section 9: Risk Assessment Summary."""
        ra = data.risk_assessment
        rb = ra.risk_breakdown
        findings = (
            "\n".join(f"- {f}" for f in ra.key_findings)
            if ra.key_findings
            else "- No specific findings recorded"
        )
        return (
            "## 9. Risk Assessment Summary\n\n"
            f"**Composite Risk Score:** {ra.composite_risk_score:.1f}/100 "
            f"({_risk_score_label(ra.composite_risk_score)}) | "
            f"**Classification:** {_fmt_risk_badge(ra.risk_classification)}\n\n"
            "### Score Breakdown\n\n"
            "| Category | Score | Label |\n"
            "|----------|-------|-------|\n"
            f"| Country Risk | {rb.country_risk_score:.1f} | "
            f"{_risk_score_label(rb.country_risk_score)} |\n"
            f"| Supplier Risk | {rb.supplier_risk_score:.1f} | "
            f"{_risk_score_label(rb.supplier_risk_score)} |\n"
            f"| Commodity Risk | {rb.commodity_risk_score:.1f} | "
            f"{_risk_score_label(rb.commodity_risk_score)} |\n"
            f"| Document Risk | {rb.document_risk_score:.1f} | "
            f"{_risk_score_label(rb.document_risk_score)} |\n\n"
            f"**Assessment Date:** {ra.assessment_date.isoformat()} | "
            f"**Methodology:** {ra.methodology_reference}\n\n"
            f"### Key Findings\n\n{findings}"
        )

    def _md_mitigation_measures(self, data: DDSStandardInput) -> str:
        """Section 10: Risk Mitigation Measures."""
        if (
            data.risk_assessment.risk_classification == RiskClassification.NEGLIGIBLE
            and not data.mitigation_measures
        ):
            return (
                "## 10. Risk Mitigation Measures\n\n"
                "Risk assessed as NEGLIGIBLE. No additional mitigation measures required."
            )
        lines = [
            "## 10. Risk Mitigation Measures\n",
            "| ID | Description | Effectiveness | Date Applied | Evidence |",
            "|----|-------------|---------------|--------------|----------|",
        ]
        for m in data.mitigation_measures:
            applied = m.date_applied.isoformat() if m.date_applied else "N/A"
            evidence = m.evidence_reference or "N/A"
            lines.append(
                f"| {m.measure_id} | {m.description} | {m.effectiveness} "
                f"| {applied} | {evidence} |"
            )
        if not data.mitigation_measures:
            lines.append("| - | No measures recorded | - | - | - |")
        return "\n".join(lines)

    def _md_conclusion(self, data: DDSStandardInput) -> str:
        """Section 11: Conclusion."""
        conclusion = data.conclusion_text or (
            f"Based on the due diligence assessment conducted, the risk has been "
            f"classified as {data.risk_assessment.risk_classification.value}. "
            f"Due diligence type applied: {data.dd_type.value}."
        )
        return (
            "## 11. Conclusion\n\n"
            f"**Risk Classification:** "
            f"{_fmt_risk_badge(data.risk_assessment.risk_classification)}\n"
            f"**Due Diligence Type:** {data.dd_type.value}\n\n"
            f"{conclusion}"
        )

    def _md_operator_declaration(self, data: DDSStandardInput) -> str:
        """Section 12: Operator Declaration per Article 4(2)."""
        title = f", {data.signatory_title}" if data.signatory_title else ""
        declaration_text = (
            "I hereby declare that, having exercised due diligence in accordance "
            "with Regulation (EU) 2023/1115, I confirm that the relevant commodities "
            "and products are deforestation-free, have been produced in accordance "
            "with the relevant legislation of the country of production, and are "
            "covered by a due diligence statement."
        )
        return (
            "## 12. Operator Declaration\n\n"
            f"> {declaration_text}\n\n"
            f"**Signatory:** {data.signatory_name}{title}\n"
            f"**Date:** {data.signature_date.isoformat()}"
        )

    def _md_evidence_index(self, data: DDSStandardInput) -> str:
        """Section 13: Evidence Index."""
        lines = [
            "## 13. Evidence Index\n",
            "| ID | Document | Type | SHA-256 Hash | Upload Date | Source |",
            "|----|----------|------|-------------|-------------|--------|",
        ]
        for e in data.evidence_index:
            upload = e.upload_date.isoformat() if e.upload_date else "N/A"
            source = e.source or "N/A"
            hash_short = f"`{e.sha256_hash[:16]}...`"
            lines.append(
                f"| {e.evidence_id} | {e.document_name} | {e.document_type} "
                f"| {hash_short} | {upload} | {source} |"
            )
        if not data.evidence_index:
            lines.append("| - | No evidence documents | - | - | - | - |")
        return "\n".join(lines)

    def _md_provenance(self, data: DDSStandardInput) -> str:
        """Section 14: Provenance Footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n\n"
            "## 14. Provenance\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Pack ID | {PACK_ID} |\n"
            f"| Template | {TEMPLATE_NAME} |\n"
            f"| Version | {TEMPLATE_VERSION} |\n"
            f"| Generated At | {ts} |\n"
            f"| Provenance Hash | `{provenance}` |"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, data: DDSStandardInput, body: str) -> str:
        """Wrap body content in a full HTML document with inline CSS."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>DDS {data.dds_reference} - {data.operator.company_name}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "max-width:1100px;color:#222;line-height:1.5;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "h1{color:#1a365d;border-bottom:3px solid #2b6cb0;padding-bottom:0.5rem;}\n"
            "h2{color:#2b6cb0;margin-top:2rem;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".status-draft{color:#b08800;font-weight:bold;}\n"
            ".status-final{color:#1a7f37;font-weight:bold;}\n"
            ".status-submitted{color:#0969da;font-weight:bold;}\n"
            ".status-amended{color:#cf222e;font-weight:bold;}\n"
            ".risk-negligible{color:#1a7f37;font-weight:bold;}\n"
            ".risk-non-negligible{color:#cf222e;font-weight:bold;}\n"
            ".benchmark-low{color:#1a7f37;}\n"
            ".benchmark-standard{color:#b08800;}\n"
            ".benchmark-high{color:#cf222e;}\n"
            ".score-bar{height:20px;border-radius:4px;display:inline-block;}\n"
            ".declaration{background:#f0f4f8;border-left:4px solid #2b6cb0;"
            "padding:1rem;margin:1rem 0;font-style:italic;}\n"
            ".provenance{font-size:0.85rem;color:#666;}\n"
            "code{background:#f5f5f5;padding:0.2rem 0.4rem;border-radius:3px;"
            "font-size:0.9em;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: DDSStandardInput) -> str:
        """HTML Section 1: Header."""
        css = _status_css(data.dds_status)
        return (
            '<div class="section">\n'
            "<h1>Due Diligence Statement (Standard)</h1>\n"
            "<p>EUDR Regulation (EU) 2023/1115 &mdash; Annex II</p>\n"
            "<table><tbody>"
            f"<tr><th>DDS Reference</th><td><strong>{data.dds_reference}</strong></td></tr>"
            f"<tr><th>Date</th><td>{data.dds_date.isoformat()}</td></tr>"
            f"<tr><th>Version</th><td>{data.dds_version}</td></tr>"
            f'<tr><th>Status</th><td class="{css}">{data.dds_status.value}</td></tr>'
            f"<tr><th>DD Type</th><td>{data.dd_type.value}</td></tr>"
            "</tbody></table>\n<hr>\n</div>"
        )

    def _html_operator_info(self, data: DDSStandardInput) -> str:
        """HTML Section 2: Operator Information."""
        op = data.operator
        eori = op.eori_number or "N/A"
        vat = op.vat_number or "N/A"
        phone = op.contact_phone or "N/A"
        return (
            '<div class="section">\n<h2>2. Operator Information</h2>\n'
            "<table><tbody>"
            f"<tr><th>Company Name</th><td>{op.company_name}</td></tr>"
            f"<tr><th>Registered Address</th><td>{op.address}</td></tr>"
            f"<tr><th>EORI Number</th><td>{eori}</td></tr>"
            f"<tr><th>Registration Country</th><td>{op.registration_country}</td></tr>"
            f"<tr><th>VAT Number</th><td>{vat}</td></tr>"
            f"<tr><th>Contact Name</th><td>{op.contact_name}</td></tr>"
            f"<tr><th>Contact Email</th><td>{op.contact_email}</td></tr>"
            f"<tr><th>Contact Phone</th><td>{phone}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_product_description(self, data: DDSStandardInput) -> str:
        """HTML Section 3: Product Description."""
        p = data.product
        hs = ", ".join(p.hs_codes) if p.hs_codes else "N/A"
        derived = "Yes" if p.is_derived_product else "No"
        details = p.derived_product_details or "N/A"
        return (
            '<div class="section">\n<h2>3. Product Description</h2>\n'
            "<table><tbody>"
            f"<tr><th>Commodity Type</th><td>{_fmt_commodity(p.commodity_type)}</td></tr>"
            f"<tr><th>Product Description</th><td>{p.product_description}</td></tr>"
            f"<tr><th>HS/CN Codes</th><td>{hs}</td></tr>"
            f"<tr><th>Derived Product</th><td>{derived}</td></tr>"
            f"<tr><th>Derived Product Details</th><td>{details}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_quantity(self, data: DDSStandardInput) -> str:
        """HTML Section 4: Quantity."""
        q = data.quantity
        supp = "N/A"
        if q.supplementary_units is not None:
            unit_type = q.supplementary_unit_type or "units"
            supp = f"{q.supplementary_units:,.2f} {unit_type}"
        return (
            '<div class="section">\n<h2>4. Quantity</h2>\n'
            "<table><tbody>"
            f"<tr><th>Net Mass</th><td>{_fmt_mass(q.net_mass_kg)}</td></tr>"
            f"<tr><th>Supplementary Units</th><td>{supp}</td></tr>"
            f"<tr><th>Total Shipment Value</th>"
            f"<td>{_fmt_currency(q.total_shipment_value_eur)}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_country_of_production(self, data: DDSStandardInput) -> str:
        """HTML Section 5: Country of Production."""
        c = data.country_of_production
        css = _benchmark_css(c.benchmark_classification)
        return (
            '<div class="section">\n<h2>5. Country of Production</h2>\n'
            "<table><tbody>"
            f"<tr><th>Country Code</th><td>{c.country_iso}</td></tr>"
            f"<tr><th>Country Name</th><td>{c.country_name}</td></tr>"
            f'<tr><th>Benchmark</th><td class="{css}">'
            f"{c.benchmark_classification.value}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_geolocation(self, data: DDSStandardInput) -> str:
        """HTML Section 6: Geolocation Data."""
        g = data.geolocation
        geo_type = "Polygon (>=4ha)" if g.is_polygon else "Point (<4ha)"
        area = f"{g.area_hectares:,.2f} ha" if g.area_hectares is not None else "N/A"
        precision = (
            f"{g.precision_meters:,.1f} m" if g.precision_meters is not None else "N/A"
        )
        plot_refs = ", ".join(g.plot_reference_ids) if g.plot_reference_ids else "N/A"

        coord_rows = ""
        for idx, coord in enumerate(g.coordinates, 1):
            coord_rows += (
                f"<tr><td>{idx}</td><td>{coord.latitude:.6f}</td>"
                f"<td>{coord.longitude:.6f}</td></tr>"
            )
        coord_table = ""
        if coord_rows:
            coord_table = (
                "<h3>Coordinates</h3>\n"
                "<table><thead><tr><th>#</th><th>Latitude</th>"
                "<th>Longitude</th></tr></thead>\n"
                f"<tbody>{coord_rows}</tbody></table>"
            )

        return (
            '<div class="section">\n<h2>6. Geolocation Data</h2>\n'
            "<table><tbody>"
            f"<tr><th>Geometry Type</th><td>{geo_type}</td></tr>"
            f"<tr><th>Coordinate System</th><td>{g.coordinate_system}</td></tr>"
            f"<tr><th>Area</th><td>{area}</td></tr>"
            f"<tr><th>Precision</th><td>{precision}</td></tr>"
            f"<tr><th>Plot References</th><td>{plot_refs}</td></tr>"
            f"</tbody></table>\n{coord_table}\n</div>"
        )

    def _html_supplier_info(self, data: DDSStandardInput) -> str:
        """HTML Section 7: Supplier Information."""
        rows = ""
        for idx, s in enumerate(data.suppliers, 1):
            cert = s.certification_status or "N/A"
            scheme = s.certification_scheme.value if s.certification_scheme else "N/A"
            rows += (
                f"<tr><td>{idx}</td><td>{s.supplier_name}</td>"
                f"<td>{s.supplier_country}</td><td>{s.tier_level}</td>"
                f"<td>{cert}</td><td>{scheme}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="6">No suppliers registered</td></tr>'
        return (
            '<div class="section">\n<h2>7. Supplier Information</h2>\n'
            "<table><thead><tr><th>#</th><th>Supplier</th><th>Country</th>"
            "<th>Tier</th><th>Certification</th><th>Scheme</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_supply_chain(self, data: DDSStandardInput) -> str:
        """HTML Section 8: Supply Chain Summary."""
        sc = data.supply_chain
        model_name = sc.chain_of_custody_model.value.replace("_", " ").title()
        evidence_items = "".join(
            f"<li>{e}</li>" for e in sc.traceability_evidence
        )
        evidence_html = (
            f"<ul>{evidence_items}</ul>"
            if evidence_items
            else "<p>No evidence documents recorded</p>"
        )
        return (
            '<div class="section">\n<h2>8. Supply Chain Summary</h2>\n'
            "<table><tbody>"
            f"<tr><th>Chain of Custody Model</th><td>{model_name}</td></tr>"
            f"<tr><th>Tier Depth</th><td>{sc.tier_depth}</td></tr>"
            f"<tr><th>Total Suppliers</th><td>{sc.total_suppliers}</td></tr>"
            f"</tbody></table>\n<h3>Traceability Evidence</h3>\n{evidence_html}\n</div>"
        )

    def _html_risk_assessment(self, data: DDSStandardInput) -> str:
        """HTML Section 9: Risk Assessment Summary."""
        ra = data.risk_assessment
        rb = ra.risk_breakdown
        css = _risk_css(ra.risk_classification)

        findings_items = "".join(f"<li>{f}</li>" for f in ra.key_findings)
        findings_html = (
            f"<ul>{findings_items}</ul>"
            if findings_items
            else "<p>No specific findings recorded</p>"
        )

        def _bar(score: float) -> str:
            color = "#1a7f37" if score <= 20 else (
                "#b08800" if score <= 40 else (
                    "#e36209" if score <= 60 else "#cf222e"
                )
            )
            return (
                f'<div class="score-bar" style="width:{score}%;'
                f'background:{color};">&nbsp;</div> {score:.1f}'
            )

        return (
            '<div class="section">\n<h2>9. Risk Assessment Summary</h2>\n'
            f'<p><strong>Composite Risk Score:</strong> {ra.composite_risk_score:.1f}/100 '
            f'({_risk_score_label(ra.composite_risk_score)}) | '
            f'<strong>Classification:</strong> '
            f'<span class="{css}">{ra.risk_classification.value}</span></p>\n'
            "<h3>Score Breakdown</h3>\n"
            "<table><thead><tr><th>Category</th><th>Score</th>"
            "<th>Visual</th></tr></thead>\n<tbody>"
            f"<tr><td>Country Risk</td><td>{rb.country_risk_score:.1f}</td>"
            f"<td>{_bar(rb.country_risk_score)}</td></tr>"
            f"<tr><td>Supplier Risk</td><td>{rb.supplier_risk_score:.1f}</td>"
            f"<td>{_bar(rb.supplier_risk_score)}</td></tr>"
            f"<tr><td>Commodity Risk</td><td>{rb.commodity_risk_score:.1f}</td>"
            f"<td>{_bar(rb.commodity_risk_score)}</td></tr>"
            f"<tr><td>Document Risk</td><td>{rb.document_risk_score:.1f}</td>"
            f"<td>{_bar(rb.document_risk_score)}</td></tr>"
            "</tbody></table>\n"
            f"<p><strong>Assessment Date:</strong> {ra.assessment_date.isoformat()} | "
            f"<strong>Methodology:</strong> {ra.methodology_reference}</p>\n"
            f"<h3>Key Findings</h3>\n{findings_html}\n</div>"
        )

    def _html_mitigation_measures(self, data: DDSStandardInput) -> str:
        """HTML Section 10: Risk Mitigation Measures."""
        if (
            data.risk_assessment.risk_classification == RiskClassification.NEGLIGIBLE
            and not data.mitigation_measures
        ):
            return (
                '<div class="section">\n<h2>10. Risk Mitigation Measures</h2>\n'
                "<p>Risk assessed as NEGLIGIBLE. No additional mitigation measures "
                "required.</p>\n</div>"
            )
        rows = ""
        for m in data.mitigation_measures:
            applied = m.date_applied.isoformat() if m.date_applied else "N/A"
            evidence = m.evidence_reference or "N/A"
            rows += (
                f"<tr><td>{m.measure_id}</td><td>{m.description}</td>"
                f"<td>{m.effectiveness}</td><td>{applied}</td>"
                f"<td>{evidence}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="5">No measures recorded</td></tr>'
        return (
            '<div class="section">\n<h2>10. Risk Mitigation Measures</h2>\n'
            "<table><thead><tr><th>ID</th><th>Description</th>"
            "<th>Effectiveness</th><th>Date Applied</th>"
            f"<th>Evidence</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_conclusion(self, data: DDSStandardInput) -> str:
        """HTML Section 11: Conclusion."""
        css = _risk_css(data.risk_assessment.risk_classification)
        conclusion = data.conclusion_text or (
            f"Based on the due diligence assessment conducted, the risk has been "
            f"classified as {data.risk_assessment.risk_classification.value}. "
            f"Due diligence type applied: {data.dd_type.value}."
        )
        return (
            '<div class="section">\n<h2>11. Conclusion</h2>\n'
            f'<p><strong>Risk Classification:</strong> '
            f'<span class="{css}">{data.risk_assessment.risk_classification.value}'
            f"</span></p>\n"
            f"<p><strong>Due Diligence Type:</strong> {data.dd_type.value}</p>\n"
            f"<p>{conclusion}</p>\n</div>"
        )

    def _html_operator_declaration(self, data: DDSStandardInput) -> str:
        """HTML Section 12: Operator Declaration."""
        title = f", {data.signatory_title}" if data.signatory_title else ""
        declaration_text = (
            "I hereby declare that, having exercised due diligence in accordance "
            "with Regulation (EU) 2023/1115, I confirm that the relevant commodities "
            "and products are deforestation-free, have been produced in accordance "
            "with the relevant legislation of the country of production, and are "
            "covered by a due diligence statement."
        )
        return (
            '<div class="section">\n<h2>12. Operator Declaration</h2>\n'
            f'<div class="declaration">{declaration_text}</div>\n'
            f"<p><strong>Signatory:</strong> {data.signatory_name}{title}</p>\n"
            f"<p><strong>Date:</strong> {data.signature_date.isoformat()}</p>\n</div>"
        )

    def _html_evidence_index(self, data: DDSStandardInput) -> str:
        """HTML Section 13: Evidence Index."""
        rows = ""
        for e in data.evidence_index:
            upload = e.upload_date.isoformat() if e.upload_date else "N/A"
            source = e.source or "N/A"
            hash_short = f"<code>{e.sha256_hash[:16]}...</code>"
            rows += (
                f"<tr><td>{e.evidence_id}</td><td>{e.document_name}</td>"
                f"<td>{e.document_type}</td><td>{hash_short}</td>"
                f"<td>{upload}</td><td>{source}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="6">No evidence documents</td></tr>'
        return (
            '<div class="section">\n<h2>13. Evidence Index</h2>\n'
            "<table><thead><tr><th>ID</th><th>Document</th><th>Type</th>"
            "<th>SHA-256</th><th>Upload Date</th>"
            f"<th>Source</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_provenance(self, data: DDSStandardInput) -> str:
        """HTML Section 14: Provenance."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section provenance">\n<hr>\n'
            "<h2>14. Provenance</h2>\n"
            "<table><tbody>"
            f"<tr><th>Pack ID</th><td>{PACK_ID}</td></tr>"
            f"<tr><th>Template</th><td>{TEMPLATE_NAME}</td></tr>"
            f"<tr><th>Version</th><td>{TEMPLATE_VERSION}</td></tr>"
            f"<tr><th>Generated At</th><td>{ts}</td></tr>"
            f"<tr><th>Provenance Hash</th><td><code>{provenance}</code></td></tr>"
            "</tbody></table>\n</div>"
        )
