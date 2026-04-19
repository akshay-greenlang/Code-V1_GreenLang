# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack: Simplified Due Diligence Statement Report
======================================================================

Generates a simplified Due Diligence Statement (DDS) for products
sourced from low-risk countries per EUDR Article 13. This streamlined
template reduces geolocation requirements to country-level only and
omits detailed risk assessment narrative while maintaining full
compliance with the simplified procedure requirements.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

PACK_ID = "PACK-006-eudr-starter"
TEMPLATE_NAME = "dds_simplified_report"
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
    """EUDR-regulated commodity types."""
    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    OIL_PALM = "OIL_PALM"
    RUBBER = "RUBBER"
    SOYA = "SOYA"
    WOOD = "WOOD"


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


class ChainOfCustodyModel(str, Enum):
    """Chain of custody traceability model."""
    IDENTITY_PRESERVED = "IDENTITY_PRESERVED"
    SEGREGATED = "SEGREGATED"
    MASS_BALANCE = "MASS_BALANCE"
    BOOK_AND_CLAIM = "BOOK_AND_CLAIM"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SimplifiedOperatorInfo(BaseModel):
    """Operator identification for simplified DDS."""
    company_name: str = Field(..., description="Legal entity name")
    address: str = Field(..., description="Registered address")
    eori_number: Optional[str] = Field(None, description="EORI number")
    registration_country: str = Field(..., description="ISO 3166-1 alpha-2")
    contact_name: str = Field(..., description="Primary contact")
    contact_email: str = Field(..., description="Contact email")


class SimplifiedProduct(BaseModel):
    """Product information for simplified DDS."""
    commodity_type: CommodityType = Field(..., description="Commodity type")
    product_description: str = Field(..., description="Product description")
    hs_codes: List[str] = Field(default_factory=list, description="HS/CN codes")
    is_derived_product: bool = Field(False, description="Derived product flag")

    @field_validator("hs_codes")
    @classmethod
    def clean_hs_codes(cls, v: List[str]) -> List[str]:
        """Strip whitespace from HS codes."""
        return [code.strip() for code in v if code.strip()]


class SimplifiedQuantity(BaseModel):
    """Quantity information for simplified DDS."""
    net_mass_kg: float = Field(..., gt=0, description="Net mass in kilograms")
    supplementary_units: Optional[float] = Field(None, ge=0, description="Supp. units")
    supplementary_unit_type: Optional[str] = Field(None, description="Unit type")
    total_shipment_value_eur: Optional[float] = Field(None, ge=0, description="EUR value")


class SimplifiedCountryInfo(BaseModel):
    """Country of production for simplified DDS (low-risk only)."""
    country_iso: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2"
    )
    country_name: str = Field(..., description="Full country name")
    low_risk_confirmation: bool = Field(
        True, description="Confirmation that country is classified as low-risk"
    )
    benchmark_reference: str = Field(
        "Article 29 - LOW",
        description="Reference to the benchmark classification",
    )


class SimplifiedSupplierInfo(BaseModel):
    """Streamlined supplier information."""
    supplier_name: str = Field(..., description="Supplier legal name")
    supplier_country: str = Field(..., description="ISO 3166-1 alpha-2")
    certification_scheme: Optional[CertificationScheme] = Field(
        None, description="Active certification"
    )
    tier_level: int = Field(1, ge=1, le=10, description="Supply chain tier")


class SimplifiedSupplyChain(BaseModel):
    """Streamlined supply chain summary."""
    chain_of_custody_model: ChainOfCustodyModel = Field(
        ..., description="Chain of custody model"
    )
    tier_depth: int = Field(1, ge=1, description="Tiers traced")
    total_suppliers: int = Field(0, ge=0, description="Total suppliers")


class SimplifiedEvidenceItem(BaseModel):
    """Evidence item for simplified DDS."""
    evidence_id: str = Field(..., description="Evidence identifier")
    document_name: str = Field(..., description="Document name")
    document_type: str = Field(..., description="Document type")
    sha256_hash: str = Field(..., description="SHA-256 hash")


class DDSSimplifiedInput(BaseModel):
    """Complete input data for the Simplified DDS report."""
    dds_reference: str = Field(..., description="DDS reference number")
    dds_date: date = Field(default_factory=date.today, description="DDS date")
    dds_version: str = Field("1.0", description="Document version")
    dds_status: DDSStatus = Field(DDSStatus.DRAFT, description="Status")
    operator: SimplifiedOperatorInfo = Field(..., description="Operator info")
    product: SimplifiedProduct = Field(..., description="Product description")
    quantity: SimplifiedQuantity = Field(..., description="Quantity")
    country_of_production: SimplifiedCountryInfo = Field(
        ..., description="Country (must be low-risk)"
    )
    suppliers: List[SimplifiedSupplierInfo] = Field(
        default_factory=list, description="Suppliers"
    )
    supply_chain: SimplifiedSupplyChain = Field(
        ..., description="Supply chain summary"
    )
    conclusion_text: str = Field(
        "", description="Brief conclusion narrative"
    )
    signatory_name: str = Field(..., description="Authorized signatory")
    signatory_title: Optional[str] = Field(None, description="Signatory title")
    signature_date: date = Field(
        default_factory=date.today, description="Signature date"
    )
    evidence_index: List[SimplifiedEvidenceItem] = Field(
        default_factory=list, description="Evidence index"
    )

    @field_validator("dds_reference")
    @classmethod
    def validate_reference(cls, v: str) -> str:
        """Ensure reference is non-empty."""
        if not v.strip():
            raise ValueError("DDS reference must not be empty")
        return v.strip()


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

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


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class DDSSimplifiedReport:
    """Generate Simplified Due Diligence Statement per EUDR Article 13.

    This template is used when all products originate from countries
    classified as low-risk under Article 29. The simplified procedure
    reduces requirements for geolocation (country-level only), omits
    detailed risk assessment, and streamlines supplier documentation.

    Sections:
        1. Header - DDS reference, date, version, status
        2. Operator Information - Company details, EORI
        3. Product Description - Commodity, HS codes
        4. Quantity - Net mass, value
        5. Country of Production - Low-risk country confirmation
        6. Supplier Information - Streamlined supplier list
        7. Supply Chain Summary - CoC model, tier depth
        8. Conclusion - Simplified risk conclusion
        9. Operator Declaration - Article 4(2) / Article 13 confirmation
        10. Evidence Index - Supporting documents
        11. Provenance - Pack version, timestamp, hash

    Example:
        >>> report = DDSSimplifiedReport()
        >>> data = DDSSimplifiedInput(...)
        >>> md = report.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize the Simplified DDS Report template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: DDSSimplifiedInput) -> str:
        """Render the simplified DDS as Markdown.

        Args:
            data: Validated simplified DDS input data.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_operator_info(data),
            self._md_product_description(data),
            self._md_quantity(data),
            self._md_country_of_production(data),
            self._md_supplier_info(data),
            self._md_supply_chain(data),
            self._md_conclusion(data),
            self._md_operator_declaration(data),
            self._md_evidence_index(data),
            self._md_provenance(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: DDSSimplifiedInput) -> str:
        """Render the simplified DDS as HTML.

        Args:
            data: Validated simplified DDS input data.

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
            self._html_supplier_info(data),
            self._html_supply_chain(data),
            self._html_conclusion(data),
            self._html_operator_declaration(data),
            self._html_evidence_index(data),
            self._html_provenance(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: DDSSimplifiedInput) -> Dict[str, Any]:
        """Render the simplified DDS as a JSON-serializable dictionary.

        Args:
            data: Validated simplified DDS input data.

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
            "dd_type": "SIMPLIFIED",
            "operator": data.operator.model_dump(mode="json"),
            "product": data.product.model_dump(mode="json"),
            "quantity": data.quantity.model_dump(mode="json"),
            "country_of_production": data.country_of_production.model_dump(mode="json"),
            "suppliers": [s.model_dump(mode="json") for s in data.suppliers],
            "supply_chain": data.supply_chain.model_dump(mode="json"),
            "conclusion": {
                "dd_type": "SIMPLIFIED",
                "risk_classification": "NEGLIGIBLE",
                "conclusion_text": data.conclusion_text or (
                    "Products sourced exclusively from low-risk countries per "
                    "Article 29. Simplified due diligence applied per Article 13."
                ),
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

    def _compute_provenance_hash(self, data: DDSSimplifiedInput) -> str:
        """Compute SHA-256 provenance hash over the input data.

        Args:
            data: The simplified DDS input data to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: DDSSimplifiedInput) -> str:
        """Section 1: DDS Header."""
        return (
            "# Due Diligence Statement (Simplified)\n"
            "## EUDR Regulation (EU) 2023/1115 - Article 13\n\n"
            "> **Simplified procedure for products from low-risk countries**\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| DDS Reference | **{data.dds_reference}** |\n"
            f"| Date | {data.dds_date.isoformat()} |\n"
            f"| Version | {data.dds_version} |\n"
            f"| Status | [{data.dds_status.value}] |\n"
            f"| DD Type | SIMPLIFIED |\n\n---"
        )

    def _md_operator_info(self, data: DDSSimplifiedInput) -> str:
        """Section 2: Operator Information."""
        op = data.operator
        eori = op.eori_number or "N/A"
        return (
            "## 2. Operator Information\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Company Name | {op.company_name} |\n"
            f"| Registered Address | {op.address} |\n"
            f"| EORI Number | {eori} |\n"
            f"| Registration Country | {op.registration_country} |\n"
            f"| Contact Name | {op.contact_name} |\n"
            f"| Contact Email | {op.contact_email} |"
        )

    def _md_product_description(self, data: DDSSimplifiedInput) -> str:
        """Section 3: Product Description."""
        p = data.product
        hs = ", ".join(p.hs_codes) if p.hs_codes else "N/A"
        derived = "Yes" if p.is_derived_product else "No"
        return (
            "## 3. Product Description\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Commodity Type | {_fmt_commodity(p.commodity_type)} |\n"
            f"| Product Description | {p.product_description} |\n"
            f"| HS/CN Codes | {hs} |\n"
            f"| Derived Product | {derived} |"
        )

    def _md_quantity(self, data: DDSSimplifiedInput) -> str:
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

    def _md_country_of_production(self, data: DDSSimplifiedInput) -> str:
        """Section 5: Country of Production (simplified - country level only)."""
        c = data.country_of_production
        confirmed = "CONFIRMED" if c.low_risk_confirmation else "UNCONFIRMED"
        return (
            "## 5. Country of Production\n\n"
            "> **Note:** Under Article 13, detailed geolocation data is not required "
            "for products from low-risk countries. Country-level identification "
            "is sufficient.\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Country Code | {c.country_iso} |\n"
            f"| Country Name | {c.country_name} |\n"
            f"| Low-Risk Status | [{confirmed}] |\n"
            f"| Benchmark Reference | {c.benchmark_reference} |"
        )

    def _md_supplier_info(self, data: DDSSimplifiedInput) -> str:
        """Section 6: Supplier Information (streamlined)."""
        lines = [
            "## 6. Supplier Information\n",
            "| # | Supplier Name | Country | Tier | Certification |",
            "|---|---------------|---------|------|---------------|",
        ]
        for idx, s in enumerate(data.suppliers, 1):
            cert = s.certification_scheme.value if s.certification_scheme else "N/A"
            lines.append(
                f"| {idx} | {s.supplier_name} | {s.supplier_country} "
                f"| {s.tier_level} | {cert} |"
            )
        if not data.suppliers:
            lines.append("| - | No suppliers registered | - | - | - |")
        return "\n".join(lines)

    def _md_supply_chain(self, data: DDSSimplifiedInput) -> str:
        """Section 7: Supply Chain Summary."""
        sc = data.supply_chain
        model_name = sc.chain_of_custody_model.value.replace("_", " ").title()
        return (
            "## 7. Supply Chain Summary\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Chain of Custody Model | {model_name} |\n"
            f"| Tier Depth | {sc.tier_depth} |\n"
            f"| Total Suppliers | {sc.total_suppliers} |"
        )

    def _md_conclusion(self, data: DDSSimplifiedInput) -> str:
        """Section 8: Conclusion."""
        conclusion = data.conclusion_text or (
            "Products sourced exclusively from low-risk countries per "
            "Article 29. Simplified due diligence applied per Article 13. "
            "Risk is assessed as negligible based on the country benchmark."
        )
        return (
            "## 8. Conclusion\n\n"
            "**Risk Classification:** [NEGLIGIBLE]\n"
            "**Due Diligence Type:** SIMPLIFIED\n\n"
            f"{conclusion}"
        )

    def _md_operator_declaration(self, data: DDSSimplifiedInput) -> str:
        """Section 9: Operator Declaration."""
        title = f", {data.signatory_title}" if data.signatory_title else ""
        declaration = (
            "I hereby declare that, in accordance with the simplified due diligence "
            "procedure under Article 13 of Regulation (EU) 2023/1115, the relevant "
            "commodities and products originate from countries classified as low-risk "
            "under Article 29. I confirm that the products are deforestation-free "
            "and have been produced in accordance with the relevant legislation "
            "of the country of production."
        )
        return (
            "## 9. Operator Declaration\n\n"
            f"> {declaration}\n\n"
            f"**Signatory:** {data.signatory_name}{title}\n"
            f"**Date:** {data.signature_date.isoformat()}"
        )

    def _md_evidence_index(self, data: DDSSimplifiedInput) -> str:
        """Section 10: Evidence Index."""
        lines = [
            "## 10. Evidence Index\n",
            "| ID | Document | Type | SHA-256 Hash |",
            "|----|----------|------|-------------|",
        ]
        for e in data.evidence_index:
            hash_short = f"`{e.sha256_hash[:16]}...`"
            lines.append(
                f"| {e.evidence_id} | {e.document_name} | {e.document_type} "
                f"| {hash_short} |"
            )
        if not data.evidence_index:
            lines.append("| - | No evidence documents | - | - |")
        return "\n".join(lines)

    def _md_provenance(self, data: DDSSimplifiedInput) -> str:
        """Section 11: Provenance Footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n\n"
            "## 11. Provenance\n\n"
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

    def _wrap_html(self, data: DDSSimplifiedInput, body: str) -> str:
        """Wrap body content in a full HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Simplified DDS {data.dds_reference} - "
            f"{data.operator.company_name}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "max-width:1000px;color:#222;line-height:1.5;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "h1{color:#1a7f37;border-bottom:3px solid #2da44e;padding-bottom:0.5rem;}\n"
            "h2{color:#1a7f37;margin-top:2rem;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".simplified-badge{background:#d1fae5;color:#1a7f37;padding:0.3rem 0.8rem;"
            "border-radius:4px;font-weight:bold;display:inline-block;margin:0.5rem 0;}\n"
            ".low-risk{color:#1a7f37;font-weight:bold;}\n"
            ".note-box{background:#f0fdf4;border-left:4px solid #2da44e;"
            "padding:1rem;margin:1rem 0;font-style:italic;}\n"
            ".declaration{background:#f0f4f8;border-left:4px solid #2b6cb0;"
            "padding:1rem;margin:1rem 0;font-style:italic;}\n"
            ".provenance{font-size:0.85rem;color:#666;}\n"
            "code{background:#f5f5f5;padding:0.2rem 0.4rem;border-radius:3px;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 1: Header."""
        return (
            '<div class="section">\n'
            "<h1>Due Diligence Statement (Simplified)</h1>\n"
            "<p>EUDR Regulation (EU) 2023/1115 &mdash; Article 13</p>\n"
            '<div class="simplified-badge">SIMPLIFIED PROCEDURE</div>\n'
            "<table><tbody>"
            f"<tr><th>DDS Reference</th>"
            f"<td><strong>{data.dds_reference}</strong></td></tr>"
            f"<tr><th>Date</th><td>{data.dds_date.isoformat()}</td></tr>"
            f"<tr><th>Version</th><td>{data.dds_version}</td></tr>"
            f"<tr><th>Status</th><td>{data.dds_status.value}</td></tr>"
            f"<tr><th>DD Type</th><td>SIMPLIFIED</td></tr>"
            "</tbody></table>\n<hr>\n</div>"
        )

    def _html_operator_info(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 2: Operator Information."""
        op = data.operator
        eori = op.eori_number or "N/A"
        return (
            '<div class="section">\n<h2>2. Operator Information</h2>\n'
            "<table><tbody>"
            f"<tr><th>Company Name</th><td>{op.company_name}</td></tr>"
            f"<tr><th>Registered Address</th><td>{op.address}</td></tr>"
            f"<tr><th>EORI Number</th><td>{eori}</td></tr>"
            f"<tr><th>Registration Country</th><td>{op.registration_country}</td></tr>"
            f"<tr><th>Contact Name</th><td>{op.contact_name}</td></tr>"
            f"<tr><th>Contact Email</th><td>{op.contact_email}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_product_description(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 3: Product Description."""
        p = data.product
        hs = ", ".join(p.hs_codes) if p.hs_codes else "N/A"
        derived = "Yes" if p.is_derived_product else "No"
        return (
            '<div class="section">\n<h2>3. Product Description</h2>\n'
            "<table><tbody>"
            f"<tr><th>Commodity Type</th><td>{_fmt_commodity(p.commodity_type)}</td></tr>"
            f"<tr><th>Product Description</th><td>{p.product_description}</td></tr>"
            f"<tr><th>HS/CN Codes</th><td>{hs}</td></tr>"
            f"<tr><th>Derived Product</th><td>{derived}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_quantity(self, data: DDSSimplifiedInput) -> str:
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

    def _html_country_of_production(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 5: Country of Production."""
        c = data.country_of_production
        confirmed = "CONFIRMED" if c.low_risk_confirmation else "UNCONFIRMED"
        return (
            '<div class="section">\n<h2>5. Country of Production</h2>\n'
            '<div class="note-box">Under Article 13, detailed geolocation data '
            "is not required for products from low-risk countries. Country-level "
            "identification is sufficient.</div>\n"
            "<table><tbody>"
            f"<tr><th>Country Code</th><td>{c.country_iso}</td></tr>"
            f"<tr><th>Country Name</th><td>{c.country_name}</td></tr>"
            f'<tr><th>Low-Risk Status</th><td class="low-risk">{confirmed}</td></tr>'
            f"<tr><th>Benchmark Reference</th><td>{c.benchmark_reference}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_supplier_info(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 6: Supplier Information."""
        rows = ""
        for idx, s in enumerate(data.suppliers, 1):
            cert = s.certification_scheme.value if s.certification_scheme else "N/A"
            rows += (
                f"<tr><td>{idx}</td><td>{s.supplier_name}</td>"
                f"<td>{s.supplier_country}</td><td>{s.tier_level}</td>"
                f"<td>{cert}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="5">No suppliers registered</td></tr>'
        return (
            '<div class="section">\n<h2>6. Supplier Information</h2>\n'
            "<table><thead><tr><th>#</th><th>Supplier</th><th>Country</th>"
            "<th>Tier</th><th>Certification</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_supply_chain(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 7: Supply Chain Summary."""
        sc = data.supply_chain
        model_name = sc.chain_of_custody_model.value.replace("_", " ").title()
        return (
            '<div class="section">\n<h2>7. Supply Chain Summary</h2>\n'
            "<table><tbody>"
            f"<tr><th>Chain of Custody Model</th><td>{model_name}</td></tr>"
            f"<tr><th>Tier Depth</th><td>{sc.tier_depth}</td></tr>"
            f"<tr><th>Total Suppliers</th><td>{sc.total_suppliers}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_conclusion(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 8: Conclusion."""
        conclusion = data.conclusion_text or (
            "Products sourced exclusively from low-risk countries per "
            "Article 29. Simplified due diligence applied per Article 13. "
            "Risk is assessed as negligible based on the country benchmark."
        )
        return (
            '<div class="section">\n<h2>8. Conclusion</h2>\n'
            '<p><strong>Risk Classification:</strong> '
            '<span class="low-risk">NEGLIGIBLE</span></p>\n'
            "<p><strong>Due Diligence Type:</strong> SIMPLIFIED</p>\n"
            f"<p>{conclusion}</p>\n</div>"
        )

    def _html_operator_declaration(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 9: Operator Declaration."""
        title = f", {data.signatory_title}" if data.signatory_title else ""
        declaration = (
            "I hereby declare that, in accordance with the simplified due diligence "
            "procedure under Article 13 of Regulation (EU) 2023/1115, the relevant "
            "commodities and products originate from countries classified as low-risk "
            "under Article 29. I confirm that the products are deforestation-free "
            "and have been produced in accordance with the relevant legislation "
            "of the country of production."
        )
        return (
            '<div class="section">\n<h2>9. Operator Declaration</h2>\n'
            f'<div class="declaration">{declaration}</div>\n'
            f"<p><strong>Signatory:</strong> {data.signatory_name}{title}</p>\n"
            f"<p><strong>Date:</strong> {data.signature_date.isoformat()}</p>\n</div>"
        )

    def _html_evidence_index(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 10: Evidence Index."""
        rows = ""
        for e in data.evidence_index:
            hash_short = f"<code>{e.sha256_hash[:16]}...</code>"
            rows += (
                f"<tr><td>{e.evidence_id}</td><td>{e.document_name}</td>"
                f"<td>{e.document_type}</td><td>{hash_short}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="4">No evidence documents</td></tr>'
        return (
            '<div class="section">\n<h2>10. Evidence Index</h2>\n'
            "<table><thead><tr><th>ID</th><th>Document</th>"
            "<th>Type</th><th>SHA-256</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_provenance(self, data: DDSSimplifiedInput) -> str:
        """HTML Section 11: Provenance."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section provenance">\n<hr>\n'
            "<h2>11. Provenance</h2>\n"
            "<table><tbody>"
            f"<tr><th>Pack ID</th><td>{PACK_ID}</td></tr>"
            f"<tr><th>Template</th><td>{TEMPLATE_NAME}</td></tr>"
            f"<tr><th>Version</th><td>{TEMPLATE_VERSION}</td></tr>"
            f"<tr><th>Generated At</th><td>{ts}</td></tr>"
            f"<tr><th>Provenance Hash</th><td><code>{provenance}</code></td></tr>"
            "</tbody></table>\n</div>"
        )
