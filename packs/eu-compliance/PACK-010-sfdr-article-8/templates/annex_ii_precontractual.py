"""
AnnexIIPrecontractualTemplate - SFDR RTS Annex II pre-contractual disclosure.

This module implements the Annex II pre-contractual disclosure template for
PACK-010 SFDR Article 8 products. It generates the mandatory pre-contractual
information required under SFDR Delegated Regulation (EU) 2022/1288,
covering 12 disclosure sections from product identification through
engagement policies.

The template follows the official RTS Annex II format for financial products
that promote environmental or social characteristics (Article 8).

Example:
    >>> template = AnnexIIPrecontractualTemplate()
    >>> data = PrecontractualData(
    ...     product_info=ProductInfo(product_name="Green Bond Fund", ...),
    ...     es_characteristics=ESCharacteristics(...),
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class ProductInfo(BaseModel):
    """Product identification details."""

    product_name: str = Field(..., min_length=1, description="Legal product name")
    isin: str = Field("", description="ISIN code (if applicable)")
    lei: str = Field("", description="Legal Entity Identifier")
    sfdr_classification: str = Field(
        "article_8",
        description="SFDR classification: article_8, article_8_plus, article_9",
    )
    fund_type: str = Field(
        "", description="Fund type (UCITS, AIF, pension, insurance, etc.)"
    )
    currency: str = Field("EUR", description="Base currency")
    management_company: str = Field("", description="Management company name")
    domicile: str = Field("", description="Fund domicile country")

    @field_validator("sfdr_classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        """Validate SFDR classification is an Article 8 variant."""
        allowed = {"article_8", "article_8_plus", "article_9"}
        if v not in allowed:
            raise ValueError(
                f"sfdr_classification must be one of {allowed}, got '{v}'"
            )
        return v


class ESCharacteristics(BaseModel):
    """Environmental and social characteristics promoted by the product."""

    environmental: List[str] = Field(
        default_factory=list,
        description="Environmental characteristics promoted",
    )
    social: List[str] = Field(
        default_factory=list,
        description="Social characteristics promoted",
    )
    binding_elements: List[str] = Field(
        default_factory=list,
        description="Binding elements of the investment strategy",
    )
    sustainability_indicators: List[str] = Field(
        default_factory=list,
        description="Sustainability indicators used to measure attainment",
    )


class AssetAllocation(BaseModel):
    """Proportion of investments in asset allocation."""

    sustainable_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage of sustainable investments",
    )
    taxonomy_aligned_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage taxonomy-aligned",
    )
    other_environmental_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage other environmental objectives",
    )
    social_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage social objectives",
    )
    not_sustainable_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage not qualifying as sustainable",
    )
    cash_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage cash and cash equivalents",
    )
    hedging_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage hedging instruments",
    )

    @field_validator("not_sustainable_pct")
    @classmethod
    def check_total_reasonable(cls, v: float, info: Any) -> float:
        """Warn if allocation clearly exceeds 100 percent."""
        return v


class InvestmentStrategy(BaseModel):
    """Investment strategy details."""

    approach: str = Field(
        "", description="Description of the investment approach"
    )
    binding_commitments: List[str] = Field(
        default_factory=list,
        description="Binding commitments in the investment strategy",
    )
    exclusions: List[str] = Field(
        default_factory=list,
        description="Exclusion criteria applied",
    )
    engagement_policy: str = Field(
        "",
        description="Description of engagement policy applied",
    )
    good_governance_assessment: str = Field(
        "",
        description="How good governance practices of investee companies are assessed",
    )
    minimum_safeguards: str = Field(
        "",
        description="Minimum safeguards applied (OECD Guidelines, UNGPs, etc.)",
    )


class TaxonomyDisclosure(BaseModel):
    """EU Taxonomy alignment disclosure."""

    minimum_alignment_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Minimum committed taxonomy alignment percentage",
    )
    objective_breakdown: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Breakdown by environmental objective: {objective, pct}",
    )
    transitional_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage in transitional activities",
    )
    enabling_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage in enabling activities",
    )
    fossil_gas_included: bool = Field(
        False, description="Whether investments in fossil gas taxonomy activities are included"
    )
    nuclear_included: bool = Field(
        False, description="Whether investments in nuclear taxonomy activities are included"
    )


class DNSHApproach(BaseModel):
    """Do No Significant Harm approach."""

    methodology: str = Field(
        "", description="Methodology for DNSH assessment"
    )
    indicators_considered: List[str] = Field(
        default_factory=list,
        description="PAI indicators considered in DNSH assessment",
    )
    thresholds: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Thresholds applied: {indicator, threshold, unit}",
    )
    alignment_with_oecd_ungp: str = Field(
        "",
        description="How investments are aligned with OECD Guidelines and UN Guiding Principles",
    )


class DataSourceEntry(BaseModel):
    """Individual data source entry."""

    source_name: str = Field("", description="Name of data source")
    provider: str = Field("", description="Data provider")
    coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Coverage percentage"
    )
    data_type: str = Field("", description="Type of data (reported, estimated, modeled)")


class DataSources(BaseModel):
    """Data sources and processing methodology."""

    sources: List[DataSourceEntry] = Field(
        default_factory=list, description="List of data sources"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations in data and methodologies",
    )
    estimation_methodology: str = Field(
        "",
        description="Methodology for estimated or modeled data",
    )
    data_quality_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Overall data quality score"
    )


class DueDiligenceProcess(BaseModel):
    """Due diligence process description."""

    description: str = Field(
        "", description="Description of due diligence process"
    )
    internal_resources: List[str] = Field(
        default_factory=list, description="Internal resources dedicated"
    )
    external_resources: List[str] = Field(
        default_factory=list, description="External resources and providers"
    )
    frequency: str = Field(
        "", description="Frequency of due diligence reviews"
    )


class PrecontractualData(BaseModel):
    """Complete input data for Annex II pre-contractual disclosure."""

    product_info: ProductInfo
    es_characteristics: ESCharacteristics = Field(default_factory=ESCharacteristics)
    asset_allocation: AssetAllocation = Field(default_factory=AssetAllocation)
    investment_strategy: InvestmentStrategy = Field(default_factory=InvestmentStrategy)
    taxonomy_disclosure: TaxonomyDisclosure = Field(default_factory=TaxonomyDisclosure)
    dnsh_approach: DNSHApproach = Field(default_factory=DNSHApproach)
    data_sources: DataSources = Field(default_factory=DataSources)
    due_diligence: DueDiligenceProcess = Field(default_factory=DueDiligenceProcess)
    reference_benchmark: Optional[str] = Field(
        None, description="Designated reference benchmark (if any)"
    )
    reference_benchmark_description: str = Field(
        "",
        description="How the reference benchmark is aligned with E/S characteristics",
    )


# ---------------------------------------------------------------------------
#  Template Configuration
# ---------------------------------------------------------------------------

class TemplateConfig(BaseModel):
    """Configuration for Annex II template rendering."""

    currency: str = Field("EUR", description="Currency code")
    language: str = Field("en", description="Output language code")
    include_taxonomy_diagrams: bool = Field(
        True, description="Include taxonomy alignment diagrams"
    )
    include_asset_allocation_chart: bool = Field(
        True, description="Include asset allocation visual"
    )
    disclaimer_text: str = Field(
        "",
        description="Custom disclaimer text to append",
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class AnnexIIPrecontractualTemplate:
    """
    SFDR RTS Annex II pre-contractual disclosure template for Article 8 products.

    Generates the mandatory pre-contractual information required under
    SFDR Delegated Regulation (EU) 2022/1288 Annex II. Covers 12 sections
    from product identification through engagement policies.

    Attributes:
        config: Optional configuration dictionary.
        PACK_ID: Pack identifier (PACK-010).
        TEMPLATE_NAME: Template identifier.
        VERSION: Template version.

    Example:
        >>> template = AnnexIIPrecontractualTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "Pre-contractual Disclosure" in md
    """

    PACK_ID = "PACK-010"
    TEMPLATE_NAME = "annex_ii_precontractual"
    VERSION = "1.0"

    SFDR_ARTICLE = "Article 8"
    REGULATION_REF = "Regulation (EU) 2019/2088"
    RTS_REF = "Delegated Regulation (EU) 2022/1288, Annex II"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize AnnexIIPrecontractualTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    #  Public render dispatcher
    # ------------------------------------------------------------------ #

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render pre-contractual disclosure in the specified format.

        Args:
            data: Report data dictionary matching PrecontractualData schema.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered content as string (markdown/html) or dict (json).

        Raises:
            ValueError: If format is not supported.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'markdown', 'html', or 'json'.")

    # ------------------------------------------------------------------ #
    #  Markdown rendering
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the Annex II pre-contractual disclosure as Markdown.

        Args:
            data: Report data dictionary matching PrecontractualData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header(data))
        sections.append(self._md_section_1_product_id(data))
        sections.append(self._md_section_2_es_characteristics(data))
        sections.append(self._md_section_3_no_sustainable_objective(data))
        sections.append(self._md_section_4_investment_strategy(data))
        sections.append(self._md_section_5_proportion(data))
        sections.append(self._md_section_6_min_sustainable(data))
        sections.append(self._md_section_7_taxonomy_alignment(data))
        sections.append(self._md_section_8_dnsh(data))
        sections.append(self._md_section_9_data_sources(data))
        sections.append(self._md_section_10_limitations(data))
        sections.append(self._md_section_11_due_diligence(data))
        sections.append(self._md_section_12_engagement(data))

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the Annex II pre-contractual disclosure as self-contained HTML.

        Args:
            data: Report data dictionary matching PrecontractualData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_section_1_product_id(data))
        sections.append(self._html_section_2_es_characteristics(data))
        sections.append(self._html_section_3_no_sustainable_objective(data))
        sections.append(self._html_section_4_investment_strategy(data))
        sections.append(self._html_section_5_proportion(data))
        sections.append(self._html_section_6_min_sustainable(data))
        sections.append(self._html_section_7_taxonomy_alignment(data))
        sections.append(self._html_section_8_dnsh(data))
        sections.append(self._html_section_9_data_sources(data))
        sections.append(self._html_section_10_limitations(data))
        sections.append(self._html_section_11_due_diligence(data))
        sections.append(self._html_section_12_engagement(data))

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Annex II Pre-contractual Disclosure",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the Annex II pre-contractual disclosure as structured JSON.

        Args:
            data: Report data dictionary matching PrecontractualData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        pi = data.get("product_info", {})
        report: Dict[str, Any] = {
            "report_type": "sfdr_annex_ii_precontractual",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "regulation": self.REGULATION_REF,
            "rts_annex": self.RTS_REF,
            "product_identification": {
                "product_name": pi.get("product_name", ""),
                "isin": pi.get("isin", ""),
                "lei": pi.get("lei", ""),
                "sfdr_classification": pi.get("sfdr_classification", "article_8"),
                "fund_type": pi.get("fund_type", ""),
                "currency": pi.get("currency", "EUR"),
                "management_company": pi.get("management_company", ""),
                "domicile": pi.get("domicile", ""),
            },
            "es_characteristics": data.get("es_characteristics", {}),
            "no_sustainable_investment_objective": self._json_no_sustainable_obj(data),
            "investment_strategy": data.get("investment_strategy", {}),
            "asset_allocation": data.get("asset_allocation", {}),
            "minimum_sustainable_investment": self._json_min_sustainable(data),
            "taxonomy_alignment": data.get("taxonomy_disclosure", {}),
            "dnsh_approach": data.get("dnsh_approach", {}),
            "data_sources": self._json_data_sources(data),
            "limitations": self._json_limitations(data),
            "due_diligence": data.get("due_diligence", {}),
            "engagement_policies": self._json_engagement(data),
            "reference_benchmark": {
                "designated": data.get("reference_benchmark") is not None,
                "name": data.get("reference_benchmark", ""),
                "description": data.get("reference_benchmark_description", ""),
            },
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown document header."""
        pi = data.get("product_info", {})
        name = pi.get("product_name", "Unknown Product")
        return (
            f"# Pre-contractual Disclosure ({self.SFDR_ARTICLE})\n\n"
            f"**{self.RTS_REF}**\n\n"
            f"**Product:** {name}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_product_id(self, data: Dict[str, Any]) -> str:
        """Section 1: Product name and identifier."""
        pi = data.get("product_info", {})
        lines = [
            "## 1. Product Identification\n",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Product Name** | {pi.get('product_name', 'N/A')} |",
            f"| **ISIN** | {pi.get('isin', 'N/A') or 'N/A'} |",
            f"| **LEI** | {pi.get('lei', 'N/A') or 'N/A'} |",
            f"| **SFDR Classification** | {self._format_classification(pi.get('sfdr_classification', ''))} |",
            f"| **Fund Type** | {pi.get('fund_type', 'N/A') or 'N/A'} |",
            f"| **Currency** | {pi.get('currency', 'EUR')} |",
            f"| **Management Company** | {pi.get('management_company', 'N/A') or 'N/A'} |",
            f"| **Domicile** | {pi.get('domicile', 'N/A') or 'N/A'} |",
        ]
        return "\n".join(lines)

    def _md_section_2_es_characteristics(self, data: Dict[str, Any]) -> str:
        """Section 2: Environmental/social characteristics promoted."""
        esc = data.get("es_characteristics", {})
        env_list = esc.get("environmental", [])
        soc_list = esc.get("social", [])
        binding = esc.get("binding_elements", [])
        indicators = esc.get("sustainability_indicators", [])

        lines = [
            "## 2. Environmental/Social Characteristics\n",
            "This financial product promotes the following environmental "
            "and/or social characteristics:\n",
        ]

        if env_list:
            lines.append("### Environmental Characteristics\n")
            for item in env_list:
                lines.append(f"- {item}")
            lines.append("")

        if soc_list:
            lines.append("### Social Characteristics\n")
            for item in soc_list:
                lines.append(f"- {item}")
            lines.append("")

        if binding:
            lines.append("### Binding Elements\n")
            for item in binding:
                lines.append(f"- {item}")
            lines.append("")

        if indicators:
            lines.append("### Sustainability Indicators\n")
            lines.append("| # | Indicator |")
            lines.append("|---|-----------|")
            for i, ind in enumerate(indicators, 1):
                lines.append(f"| {i} | {ind} |")

        return "\n".join(lines)

    def _md_section_3_no_sustainable_objective(self, data: Dict[str, Any]) -> str:
        """Section 3: 'No sustainable investment objective' disclaimer."""
        pi = data.get("product_info", {})
        classification = pi.get("sfdr_classification", "article_8")
        aa = data.get("asset_allocation", {})
        sustainable_pct = aa.get("sustainable_pct", 0.0)

        lines = [
            "## 3. Sustainable Investment Objective Statement\n",
        ]

        if classification == "article_8":
            lines.append(
                "**This financial product promotes environmental or social "
                "characteristics, but does not have as its objective sustainable "
                "investment.**\n"
            )
        elif classification == "article_8_plus":
            lines.append(
                "**This financial product promotes environmental or social "
                "characteristics and, while it does not have sustainable "
                "investment as its objective, it will have a minimum proportion "
                f"of {sustainable_pct:.1f}% of sustainable investments.**\n"
            )
            lines.append(
                "The sustainable investments contribute to environmental and/or "
                "social objectives and do not significantly harm any of those "
                "objectives (DNSH principle)."
            )

        return "\n".join(lines)

    def _md_section_4_investment_strategy(self, data: Dict[str, Any]) -> str:
        """Section 4: Investment strategy and binding elements."""
        strat = data.get("investment_strategy", {})
        lines = [
            "## 4. Investment Strategy\n",
        ]

        approach = strat.get("approach", "")
        if approach:
            lines.append(f"### Approach\n\n{approach}\n")

        commitments = strat.get("binding_commitments", [])
        if commitments:
            lines.append("### Binding Commitments\n")
            for c in commitments:
                lines.append(f"- {c}")
            lines.append("")

        exclusions = strat.get("exclusions", [])
        if exclusions:
            lines.append("### Exclusion Criteria\n")
            for e in exclusions:
                lines.append(f"- {e}")
            lines.append("")

        governance = strat.get("good_governance_assessment", "")
        if governance:
            lines.append(f"### Good Governance Assessment\n\n{governance}\n")

        safeguards = strat.get("minimum_safeguards", "")
        if safeguards:
            lines.append(f"### Minimum Safeguards\n\n{safeguards}\n")

        return "\n".join(lines)

    def _md_section_5_proportion(self, data: Dict[str, Any]) -> str:
        """Section 5: Proportion of investments."""
        aa = data.get("asset_allocation", {})
        lines = [
            "## 5. Proportion of Investments\n",
            "The following table shows the planned asset allocation:\n",
            "| Category | Proportion |",
            "|----------|-----------|",
            f"| Sustainable Investments | {aa.get('sustainable_pct', 0.0):.1f}% |",
            f"|   - Taxonomy-aligned | {aa.get('taxonomy_aligned_pct', 0.0):.1f}% |",
            f"|   - Other environmental | {aa.get('other_environmental_pct', 0.0):.1f}% |",
            f"|   - Social | {aa.get('social_pct', 0.0):.1f}% |",
            f"| Not sustainable | {aa.get('not_sustainable_pct', 0.0):.1f}% |",
            f"|   - Cash/equivalents | {aa.get('cash_pct', 0.0):.1f}% |",
            f"|   - Hedging | {aa.get('hedging_pct', 0.0):.1f}% |",
            "",
        ]

        # ASCII chart
        lines.append("### Asset Allocation Breakdown\n")
        lines.append("```")
        chart_items = [
            ("Taxonomy-aligned", aa.get("taxonomy_aligned_pct", 0.0)),
            ("Other env.", aa.get("other_environmental_pct", 0.0)),
            ("Social", aa.get("social_pct", 0.0)),
            ("Not sustainable", aa.get("not_sustainable_pct", 0.0)),
        ]
        for label, pct in chart_items:
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            lines.append(f"  {label:20s} [{bar:<50s}] {pct:.1f}%")
        lines.append("```")

        return "\n".join(lines)

    def _md_section_6_min_sustainable(self, data: Dict[str, Any]) -> str:
        """Section 6: Minimum sustainable investment."""
        aa = data.get("asset_allocation", {})
        sustainable_pct = aa.get("sustainable_pct", 0.0)
        taxonomy_pct = aa.get("taxonomy_aligned_pct", 0.0)
        other_env_pct = aa.get("other_environmental_pct", 0.0)
        social_pct = aa.get("social_pct", 0.0)

        lines = [
            "## 6. Minimum Sustainable Investment\n",
        ]

        if sustainable_pct > 0:
            lines.append(
                f"This product commits to a minimum of **{sustainable_pct:.1f}%** "
                "sustainable investments, broken down as follows:\n"
            )
            lines.append(f"- **Taxonomy-aligned:** {taxonomy_pct:.1f}%")
            lines.append(f"- **Other environmental objectives:** {other_env_pct:.1f}%")
            lines.append(f"- **Social objectives:** {social_pct:.1f}%")
        else:
            lines.append(
                "This product does not commit to a minimum proportion of "
                "sustainable investments."
            )

        return "\n".join(lines)

    def _md_section_7_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Section 7: EU Taxonomy alignment proportion."""
        td = data.get("taxonomy_disclosure", {})
        min_pct = td.get("minimum_alignment_pct", 0.0)
        objectives = td.get("objective_breakdown", [])
        transitional = td.get("transitional_pct", 0.0)
        enabling = td.get("enabling_pct", 0.0)
        fossil_gas = td.get("fossil_gas_included", False)
        nuclear = td.get("nuclear_included", False)

        lines = [
            "## 7. EU Taxonomy Alignment\n",
            f"**Minimum taxonomy alignment commitment:** {min_pct:.1f}%\n",
        ]

        # Fossil gas / nuclear disclosure (mandatory since Jan 2023)
        lines.append("### Fossil Gas and Nuclear Activities\n")
        lines.append("| Activity Type | Included |")
        lines.append("|---------------|----------|")
        lines.append(f"| Fossil gas | {'Yes' if fossil_gas else 'No'} |")
        lines.append(f"| Nuclear energy | {'Yes' if nuclear else 'No'} |")
        lines.append("")

        if objectives:
            lines.append("### Breakdown by Environmental Objective\n")
            lines.append("| Objective | Alignment (%) |")
            lines.append("|-----------|--------------|")
            for obj in objectives:
                lines.append(
                    f"| {obj.get('objective', '')} | {obj.get('pct', 0.0):.1f}% |"
                )
            lines.append("")

        lines.append(f"**Transitional activities:** {transitional:.1f}%\n")
        lines.append(f"**Enabling activities:** {enabling:.1f}%")

        return "\n".join(lines)

    def _md_section_8_dnsh(self, data: Dict[str, Any]) -> str:
        """Section 8: DNSH approach."""
        dnsh = data.get("dnsh_approach", {})
        methodology = dnsh.get("methodology", "")
        indicators = dnsh.get("indicators_considered", [])
        thresholds = dnsh.get("thresholds", [])
        oecd_ungp = dnsh.get("alignment_with_oecd_ungp", "")

        lines = [
            "## 8. Do No Significant Harm (DNSH)\n",
        ]

        if methodology:
            lines.append(f"### Methodology\n\n{methodology}\n")

        if indicators:
            lines.append("### PAI Indicators Considered\n")
            for ind in indicators:
                lines.append(f"- {ind}")
            lines.append("")

        if thresholds:
            lines.append("### Thresholds Applied\n")
            lines.append("| Indicator | Threshold | Unit |")
            lines.append("|-----------|-----------|------|")
            for t in thresholds:
                lines.append(
                    f"| {t.get('indicator', '')} | "
                    f"{t.get('threshold', '')} | "
                    f"{t.get('unit', '')} |"
                )
            lines.append("")

        if oecd_ungp:
            lines.append(
                f"### Alignment with OECD Guidelines and UN Guiding Principles\n\n{oecd_ungp}"
            )

        return "\n".join(lines)

    def _md_section_9_data_sources(self, data: Dict[str, Any]) -> str:
        """Section 9: Data sources and processing."""
        ds = data.get("data_sources", {})
        sources = ds.get("sources", [])
        estimation = ds.get("estimation_methodology", "")
        quality = ds.get("data_quality_score")

        lines = [
            "## 9. Data Sources and Processing\n",
        ]

        if sources:
            lines.append("| Source | Provider | Coverage | Data Type |")
            lines.append("|--------|----------|----------|-----------|")
            for s in sources:
                name = s.get("source_name", "")
                provider = s.get("provider", "")
                coverage = s.get("coverage_pct", 0.0)
                dtype = s.get("data_type", "")
                lines.append(f"| {name} | {provider} | {coverage:.1f}% | {dtype} |")
            lines.append("")

        if estimation:
            lines.append(f"### Estimation Methodology\n\n{estimation}\n")

        if quality is not None:
            lines.append(f"**Data Quality Score:** {quality:.1f}/100")

        return "\n".join(lines)

    def _md_section_10_limitations(self, data: Dict[str, Any]) -> str:
        """Section 10: Limitations to methodologies and data."""
        ds = data.get("data_sources", {})
        limitations = ds.get("limitations", [])

        lines = [
            "## 10. Limitations to Methodologies and Data\n",
        ]

        if limitations:
            for lim in limitations:
                lines.append(f"- {lim}")
        else:
            lines.append("No material limitations have been identified.")

        return "\n".join(lines)

    def _md_section_11_due_diligence(self, data: Dict[str, Any]) -> str:
        """Section 11: Due diligence process."""
        dd = data.get("due_diligence", {})
        description = dd.get("description", "")
        internal = dd.get("internal_resources", [])
        external = dd.get("external_resources", [])
        frequency = dd.get("frequency", "")

        lines = [
            "## 11. Due Diligence\n",
        ]

        if description:
            lines.append(f"{description}\n")

        if internal:
            lines.append("### Internal Resources\n")
            for r in internal:
                lines.append(f"- {r}")
            lines.append("")

        if external:
            lines.append("### External Resources\n")
            for r in external:
                lines.append(f"- {r}")
            lines.append("")

        if frequency:
            lines.append(f"**Review Frequency:** {frequency}")

        return "\n".join(lines)

    def _md_section_12_engagement(self, data: Dict[str, Any]) -> str:
        """Section 12: Engagement policies."""
        strat = data.get("investment_strategy", {})
        engagement = strat.get("engagement_policy", "")
        benchmark = data.get("reference_benchmark")
        benchmark_desc = data.get("reference_benchmark_description", "")

        lines = [
            "## 12. Engagement Policies\n",
        ]

        if engagement:
            lines.append(f"{engagement}\n")
        else:
            lines.append(
                "The engagement policy is described in the product's "
                "investment strategy documentation.\n"
            )

        lines.append("### Designated Reference Benchmark\n")
        if benchmark:
            lines.append(f"**Benchmark:** {benchmark}\n")
            if benchmark_desc:
                lines.append(f"{benchmark_desc}")
        else:
            lines.append("No reference benchmark has been designated for this product.")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_product_id(self, data: Dict[str, Any]) -> str:
        """Build HTML product identification section."""
        pi = data.get("product_info", {})
        return (
            '<div class="section">'
            "<h2>1. Product Identification</h2>"
            '<table class="info-table">'
            f"<tr><td><strong>Product Name</strong></td><td>{_esc(pi.get('product_name', 'N/A'))}</td></tr>"
            f"<tr><td><strong>ISIN</strong></td><td>{_esc(pi.get('isin', '') or 'N/A')}</td></tr>"
            f"<tr><td><strong>LEI</strong></td><td>{_esc(pi.get('lei', '') or 'N/A')}</td></tr>"
            f"<tr><td><strong>SFDR Classification</strong></td>"
            f"<td>{_esc(self._format_classification(pi.get('sfdr_classification', '')))}</td></tr>"
            f"<tr><td><strong>Fund Type</strong></td><td>{_esc(pi.get('fund_type', '') or 'N/A')}</td></tr>"
            f"<tr><td><strong>Currency</strong></td><td>{_esc(pi.get('currency', 'EUR'))}</td></tr>"
            f"<tr><td><strong>Management Company</strong></td>"
            f"<td>{_esc(pi.get('management_company', '') or 'N/A')}</td></tr>"
            f"<tr><td><strong>Domicile</strong></td><td>{_esc(pi.get('domicile', '') or 'N/A')}</td></tr>"
            "</table></div>"
        )

    def _html_section_2_es_characteristics(self, data: Dict[str, Any]) -> str:
        """Build HTML E/S characteristics section."""
        esc = data.get("es_characteristics", {})
        env_list = esc.get("environmental", [])
        soc_list = esc.get("social", [])
        binding = esc.get("binding_elements", [])

        parts = ['<div class="section"><h2>2. Environmental/Social Characteristics</h2>']

        if env_list:
            parts.append("<h3>Environmental</h3><ul>")
            for item in env_list:
                parts.append(f"<li>{_esc(item)}</li>")
            parts.append("</ul>")

        if soc_list:
            parts.append("<h3>Social</h3><ul>")
            for item in soc_list:
                parts.append(f"<li>{_esc(item)}</li>")
            parts.append("</ul>")

        if binding:
            parts.append("<h3>Binding Elements</h3><ul>")
            for item in binding:
                parts.append(f"<li>{_esc(item)}</li>")
            parts.append("</ul>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_3_no_sustainable_objective(self, data: Dict[str, Any]) -> str:
        """Build HTML 'no sustainable objective' disclaimer section."""
        pi = data.get("product_info", {})
        classification = pi.get("sfdr_classification", "article_8")
        aa = data.get("asset_allocation", {})
        sustainable_pct = aa.get("sustainable_pct", 0.0)

        parts = ['<div class="section disclaimer-box">']
        parts.append("<h2>3. Sustainable Investment Objective Statement</h2>")

        if classification == "article_8":
            parts.append(
                '<p class="disclaimer"><strong>This financial product promotes '
                "environmental or social characteristics, but does not have as "
                "its objective sustainable investment.</strong></p>"
            )
        elif classification == "article_8_plus":
            parts.append(
                '<p class="disclaimer"><strong>This financial product promotes '
                "environmental or social characteristics and, while it does not "
                "have sustainable investment as its objective, it will have a "
                f"minimum proportion of {sustainable_pct:.1f}% of sustainable "
                "investments.</strong></p>"
            )

        parts.append("</div>")
        return "".join(parts)

    def _html_section_4_investment_strategy(self, data: Dict[str, Any]) -> str:
        """Build HTML investment strategy section."""
        strat = data.get("investment_strategy", {})
        parts = ['<div class="section"><h2>4. Investment Strategy</h2>']

        approach = strat.get("approach", "")
        if approach:
            parts.append(f"<h3>Approach</h3><p>{_esc(approach)}</p>")

        commitments = strat.get("binding_commitments", [])
        if commitments:
            parts.append("<h3>Binding Commitments</h3><ul>")
            for c in commitments:
                parts.append(f"<li>{_esc(c)}</li>")
            parts.append("</ul>")

        exclusions = strat.get("exclusions", [])
        if exclusions:
            parts.append("<h3>Exclusion Criteria</h3><ul>")
            for e in exclusions:
                parts.append(f"<li>{_esc(e)}</li>")
            parts.append("</ul>")

        governance = strat.get("good_governance_assessment", "")
        if governance:
            parts.append(f"<h3>Good Governance</h3><p>{_esc(governance)}</p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_5_proportion(self, data: Dict[str, Any]) -> str:
        """Build HTML asset allocation proportion section."""
        aa = data.get("asset_allocation", {})
        parts = ['<div class="section"><h2>5. Proportion of Investments</h2>']
        parts.append('<table class="data-table">')
        parts.append("<tr><th>Category</th><th>Proportion</th><th>Visual</th></tr>")

        items = [
            ("Taxonomy-aligned", aa.get("taxonomy_aligned_pct", 0.0), "#2ecc71"),
            ("Other environmental", aa.get("other_environmental_pct", 0.0), "#27ae60"),
            ("Social", aa.get("social_pct", 0.0), "#3498db"),
            ("Not sustainable", aa.get("not_sustainable_pct", 0.0), "#95a5a6"),
        ]

        for label, pct, color in items:
            bar_width = max(int(pct * 2), 0)
            parts.append(
                f"<tr><td>{_esc(label)}</td><td>{pct:.1f}%</td>"
                f'<td><div style="background:{color};width:{bar_width}px;'
                f'height:16px;border-radius:3px;"></div></td></tr>'
            )

        parts.append("</table></div>")
        return "".join(parts)

    def _html_section_6_min_sustainable(self, data: Dict[str, Any]) -> str:
        """Build HTML minimum sustainable investment section."""
        aa = data.get("asset_allocation", {})
        sustainable_pct = aa.get("sustainable_pct", 0.0)
        parts = ['<div class="section"><h2>6. Minimum Sustainable Investment</h2>']

        if sustainable_pct > 0:
            parts.append(
                f"<p>This product commits to a minimum of <strong>{sustainable_pct:.1f}%</strong> "
                "sustainable investments.</p>"
            )
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Category</th><th>Minimum %</th></tr>")
            parts.append(
                f"<tr><td>Taxonomy-aligned</td><td>{aa.get('taxonomy_aligned_pct', 0.0):.1f}%</td></tr>"
            )
            parts.append(
                f"<tr><td>Other environmental</td><td>{aa.get('other_environmental_pct', 0.0):.1f}%</td></tr>"
            )
            parts.append(
                f"<tr><td>Social</td><td>{aa.get('social_pct', 0.0):.1f}%</td></tr>"
            )
            parts.append("</table>")
        else:
            parts.append(
                "<p>This product does not commit to a minimum proportion of "
                "sustainable investments.</p>"
            )

        parts.append("</div>")
        return "".join(parts)

    def _html_section_7_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Build HTML taxonomy alignment section."""
        td = data.get("taxonomy_disclosure", {})
        min_pct = td.get("minimum_alignment_pct", 0.0)
        objectives = td.get("objective_breakdown", [])
        fossil_gas = td.get("fossil_gas_included", False)
        nuclear = td.get("nuclear_included", False)

        parts = ['<div class="section"><h2>7. EU Taxonomy Alignment</h2>']
        parts.append(f"<p><strong>Minimum taxonomy alignment:</strong> {min_pct:.1f}%</p>")

        parts.append("<h3>Fossil Gas and Nuclear</h3>")
        parts.append('<table class="data-table">')
        parts.append("<tr><th>Activity Type</th><th>Included</th></tr>")
        parts.append(f"<tr><td>Fossil gas</td><td>{'Yes' if fossil_gas else 'No'}</td></tr>")
        parts.append(f"<tr><td>Nuclear energy</td><td>{'Yes' if nuclear else 'No'}</td></tr>")
        parts.append("</table>")

        if objectives:
            parts.append("<h3>By Environmental Objective</h3>")
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Objective</th><th>Alignment (%)</th></tr>")
            for obj in objectives:
                parts.append(
                    f"<tr><td>{_esc(obj.get('objective', ''))}</td>"
                    f"<td>{obj.get('pct', 0.0):.1f}%</td></tr>"
                )
            parts.append("</table>")

        parts.append(
            f"<p>Transitional: {td.get('transitional_pct', 0.0):.1f}% | "
            f"Enabling: {td.get('enabling_pct', 0.0):.1f}%</p>"
        )
        parts.append("</div>")
        return "".join(parts)

    def _html_section_8_dnsh(self, data: Dict[str, Any]) -> str:
        """Build HTML DNSH section."""
        dnsh = data.get("dnsh_approach", {})
        parts = ['<div class="section"><h2>8. Do No Significant Harm (DNSH)</h2>']

        methodology = dnsh.get("methodology", "")
        if methodology:
            parts.append(f"<p>{_esc(methodology)}</p>")

        indicators = dnsh.get("indicators_considered", [])
        if indicators:
            parts.append("<h3>PAI Indicators Considered</h3><ul>")
            for ind in indicators:
                parts.append(f"<li>{_esc(ind)}</li>")
            parts.append("</ul>")

        thresholds = dnsh.get("thresholds", [])
        if thresholds:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Indicator</th><th>Threshold</th><th>Unit</th></tr>")
            for t in thresholds:
                parts.append(
                    f"<tr><td>{_esc(str(t.get('indicator', '')))}</td>"
                    f"<td>{_esc(str(t.get('threshold', '')))}</td>"
                    f"<td>{_esc(str(t.get('unit', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_9_data_sources(self, data: Dict[str, Any]) -> str:
        """Build HTML data sources section."""
        ds = data.get("data_sources", {})
        sources = ds.get("sources", [])

        parts = ['<div class="section"><h2>9. Data Sources and Processing</h2>']

        if sources:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Source</th><th>Provider</th><th>Coverage</th><th>Type</th></tr>")
            for s in sources:
                parts.append(
                    f"<tr><td>{_esc(s.get('source_name', ''))}</td>"
                    f"<td>{_esc(s.get('provider', ''))}</td>"
                    f"<td>{s.get('coverage_pct', 0.0):.1f}%</td>"
                    f"<td>{_esc(s.get('data_type', ''))}</td></tr>"
                )
            parts.append("</table>")

        estimation = ds.get("estimation_methodology", "")
        if estimation:
            parts.append(f"<h3>Estimation Methodology</h3><p>{_esc(estimation)}</p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_10_limitations(self, data: Dict[str, Any]) -> str:
        """Build HTML limitations section."""
        ds = data.get("data_sources", {})
        limitations = ds.get("limitations", [])

        parts = ['<div class="section"><h2>10. Limitations</h2>']
        if limitations:
            parts.append("<ul>")
            for lim in limitations:
                parts.append(f"<li>{_esc(lim)}</li>")
            parts.append("</ul>")
        else:
            parts.append("<p>No material limitations identified.</p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_11_due_diligence(self, data: Dict[str, Any]) -> str:
        """Build HTML due diligence section."""
        dd = data.get("due_diligence", {})
        parts = ['<div class="section"><h2>11. Due Diligence</h2>']

        desc = dd.get("description", "")
        if desc:
            parts.append(f"<p>{_esc(desc)}</p>")

        internal = dd.get("internal_resources", [])
        if internal:
            parts.append("<h3>Internal Resources</h3><ul>")
            for r in internal:
                parts.append(f"<li>{_esc(r)}</li>")
            parts.append("</ul>")

        external = dd.get("external_resources", [])
        if external:
            parts.append("<h3>External Resources</h3><ul>")
            for r in external:
                parts.append(f"<li>{_esc(r)}</li>")
            parts.append("</ul>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_12_engagement(self, data: Dict[str, Any]) -> str:
        """Build HTML engagement policies section."""
        strat = data.get("investment_strategy", {})
        engagement = strat.get("engagement_policy", "")
        benchmark = data.get("reference_benchmark")

        parts = ['<div class="section"><h2>12. Engagement Policies</h2>']
        if engagement:
            parts.append(f"<p>{_esc(engagement)}</p>")

        parts.append("<h3>Designated Reference Benchmark</h3>")
        if benchmark:
            parts.append(f"<p><strong>{_esc(benchmark)}</strong></p>")
            desc = data.get("reference_benchmark_description", "")
            if desc:
                parts.append(f"<p>{_esc(desc)}</p>")
        else:
            parts.append("<p>No reference benchmark designated.</p>")

        parts.append("</div>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  JSON helpers
    # ------------------------------------------------------------------ #

    def _json_no_sustainable_obj(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON for sustainable objective statement."""
        pi = data.get("product_info", {})
        aa = data.get("asset_allocation", {})
        classification = pi.get("sfdr_classification", "article_8")
        return {
            "has_sustainable_objective": False,
            "article_type": classification,
            "minimum_sustainable_pct": aa.get("sustainable_pct", 0.0),
            "statement": (
                "Promotes E/S characteristics but does not have sustainable "
                "investment as its objective."
                if classification == "article_8"
                else f"Promotes E/S characteristics with minimum "
                f"{aa.get('sustainable_pct', 0.0):.1f}% sustainable investments."
            ),
        }

    def _json_min_sustainable(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON for minimum sustainable investment."""
        aa = data.get("asset_allocation", {})
        return {
            "total_pct": aa.get("sustainable_pct", 0.0),
            "taxonomy_aligned_pct": aa.get("taxonomy_aligned_pct", 0.0),
            "other_environmental_pct": aa.get("other_environmental_pct", 0.0),
            "social_pct": aa.get("social_pct", 0.0),
        }

    def _json_data_sources(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON for data sources."""
        ds = data.get("data_sources", {})
        return {
            "sources": ds.get("sources", []),
            "estimation_methodology": ds.get("estimation_methodology", ""),
            "data_quality_score": ds.get("data_quality_score"),
        }

    def _json_limitations(self, data: Dict[str, Any]) -> List[str]:
        """Build JSON for limitations."""
        ds = data.get("data_sources", {})
        return ds.get("limitations", [])

    def _json_engagement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON for engagement policies."""
        strat = data.get("investment_strategy", {})
        return {
            "engagement_policy": strat.get("engagement_policy", ""),
            "reference_benchmark": data.get("reference_benchmark"),
            "reference_benchmark_description": data.get(
                "reference_benchmark_description", ""
            ),
        }

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    def _format_classification(self, classification: str) -> str:
        """Format SFDR classification for display."""
        mapping = {
            "article_8": "Article 8 (Light Green)",
            "article_8_plus": "Article 8+ (Light Green with Sustainable Investment)",
            "article_9": "Article 9 (Dark Green)",
        }
        return mapping.get(classification, classification)

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown footer with provenance."""
        return (
            "---\n\n"
            f"*Report generated by GreenLang {self.PACK_ID} | "
            f"Template: {self.TEMPLATE_NAME} v{self.VERSION}*\n\n"
            f"*Generated: {self.generated_at}*\n\n"
            f"**Provenance Hash (SHA-256):** `{provenance_hash}`"
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>\n"
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; "
            "color: #2c3e50; line-height: 1.6; max-width: 1000px; margin: 40px auto; }\n"
            "h1 { color: #1a5276; border-bottom: 3px solid #2ecc71; padding-bottom: 10px; }\n"
            "h2 { color: #1a5276; margin-top: 30px; border-bottom: 1px solid #bdc3c7; "
            "padding-bottom: 5px; }\n"
            "h3 { color: #2c3e50; }\n"
            ".section { margin-bottom: 30px; padding: 15px; "
            "background: #fafafa; border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".disclaimer-box { background: #fef9e7; border-color: #f1c40f; }\n"
            ".info-table, .data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".info-table td, .data-table td, .data-table th { padding: 8px 12px; "
            "border: 1px solid #ddd; }\n"
            ".data-table th { background: #2c3e50; color: white; text-align: left; }\n"
            ".info-table tr:nth-child(even), .data-table tr:nth-child(even) { "
            "background: #f2f3f4; }\n"
            ".disclaimer { font-weight: bold; color: #7d6608; }\n"
            ".provenance { margin-top: 40px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            ".footer { margin-top: 30px; font-size: 0.85em; color: #7f8c8d; "
            "border-top: 1px solid #bdc3c7; padding-top: 10px; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f'<p><strong>{self.RTS_REF}</strong></p>\n'
            f'<p>Pack: {self.PACK_ID} | Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f"Generated: {self.generated_at}</p>\n"
            f"{body}\n"
            f'<div class="provenance">Provenance Hash (SHA-256): {provenance_hash}</div>\n'
            f'<div class="footer">Generated by GreenLang {self.PACK_ID} | '
            f'{self.TEMPLATE_NAME} v{self.VERSION}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n"
            "</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        """Compute SHA-256 provenance hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
#  Module-level HTML escaping utility
# ---------------------------------------------------------------------------

def _esc(value: str) -> str:
    """Escape HTML special characters."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
