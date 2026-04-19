"""
AnnexIIIPrecontractualTemplate - SFDR RTS Annex III pre-contractual disclosure.

This module implements the Annex III pre-contractual disclosure template for
PACK-011 SFDR Article 9 products. It generates the mandatory pre-contractual
information required under SFDR Delegated Regulation (EU) 2022/1288,
Annex III for financial products that have sustainable investment as their
objective (Article 9).

The template follows the official RTS Annex III format with 10 mandatory
sections: sustainable objective, investment strategy, proportions
(near 100%), taxonomy alignment, monitoring, data sources, limitations,
due diligence, engagement policies, and designated benchmark.

Article 9 products must demonstrate that their entire investment allocation
(excluding cash/hedging) contributes to a sustainable investment objective,
either environmental (including taxonomy-aligned) or social.

Example:
    >>> template = AnnexIIIPrecontractualTemplate()
    >>> data = AnnexIIIPrecontractualData(
    ...     product_info=Article9ProductInfo(product_name="Climate Impact Fund", ...),
    ...     sustainable_objective=SustainableObjective(...),
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

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

class Article9ProductInfo(BaseModel):
    """Product identification details for Article 9 products."""

    product_name: str = Field(..., min_length=1, description="Legal product name")
    isin: str = Field("", description="ISIN code (if applicable)")
    lei: str = Field("", description="Legal Entity Identifier")
    sfdr_classification: str = Field(
        "article_9",
        description="SFDR classification: article_9 or article_9_transitional",
    )
    fund_type: str = Field(
        "", description="Fund type (UCITS, AIF, pension, insurance, etc.)"
    )
    currency: str = Field("EUR", description="Base currency")
    management_company: str = Field("", description="Management company name")
    domicile: str = Field("", description="Fund domicile country")
    inception_date: str = Field("", description="Fund inception date (YYYY-MM-DD)")
    target_aum: Optional[float] = Field(
        None, ge=0.0, description="Target AUM in base currency"
    )

    @field_validator("sfdr_classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        """Validate SFDR classification is Article 9."""
        allowed = {"article_9", "article_9_transitional"}
        if v not in allowed:
            raise ValueError(
                f"sfdr_classification must be one of {allowed}, got '{v}'"
            )
        return v


class SustainableObjective(BaseModel):
    """Sustainable investment objective of the Article 9 product."""

    objective_type: str = Field(
        "environmental",
        description="environmental, social, or environmental_and_social",
    )
    objective_description: str = Field(
        "",
        description="Detailed description of the sustainable investment objective",
    )
    environmental_objectives: List[str] = Field(
        default_factory=list,
        description="Specific environmental objectives pursued",
    )
    social_objectives: List[str] = Field(
        default_factory=list,
        description="Specific social objectives pursued",
    )
    sdg_alignment: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="SDG alignment: {sdg_number, sdg_name, relevance}",
    )
    paris_alignment: bool = Field(
        False, description="Whether the product targets Paris Agreement alignment"
    )
    carbon_reduction_target: Optional[str] = Field(
        None,
        description="Carbon reduction target (e.g. 'Net zero by 2050')",
    )
    index_designation: Optional[str] = Field(
        None,
        description="Designated EU Climate Benchmark index (CTB/PAB) if applicable",
    )

    @field_validator("objective_type")
    @classmethod
    def validate_objective_type(cls, v: str) -> str:
        """Validate objective type."""
        allowed = {"environmental", "social", "environmental_and_social"}
        if v not in allowed:
            raise ValueError(f"objective_type must be one of {allowed}, got '{v}'")
        return v


class Article9Strategy(BaseModel):
    """Investment strategy for Article 9 product."""

    approach: str = Field(
        "", description="Description of the investment approach"
    )
    binding_commitments: List[str] = Field(
        default_factory=list,
        description="Binding commitments to sustainable investment objective",
    )
    exclusions: List[str] = Field(
        default_factory=list,
        description="Exclusion criteria applied (mandatory and additional)",
    )
    positive_screening: List[str] = Field(
        default_factory=list,
        description="Positive screening criteria for sustainable investments",
    )
    engagement_policy: str = Field(
        "",
        description="Description of engagement policy applied",
    )
    good_governance_assessment: str = Field(
        "",
        description="How good governance practices are assessed",
    )
    minimum_safeguards: str = Field(
        "",
        description="Minimum safeguards applied (OECD Guidelines, UNGPs, etc.)",
    )
    stewardship_code: str = Field(
        "",
        description="Stewardship code or proxy voting policy reference",
    )


class Article9Proportions(BaseModel):
    """Proportion of investments for Article 9 (near 100% sustainable)."""

    sustainable_total_pct: float = Field(
        100.0, ge=0.0, le=100.0,
        description="Total sustainable investment percentage (target ~100%)",
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
        description="Percentage not qualifying as sustainable (cash/hedging only)",
    )
    cash_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage cash and cash equivalents",
    )
    hedging_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage hedging instruments",
    )
    enabling_activities_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage in enabling activities (taxonomy)",
    )
    transitional_activities_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage in transitional activities (taxonomy)",
    )


class Article9TaxonomyAlignment(BaseModel):
    """EU Taxonomy alignment disclosure for Article 9."""

    minimum_alignment_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Minimum committed taxonomy alignment percentage",
    )
    objective_breakdown: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Breakdown by environmental objective: {objective, pct}",
    )
    substantial_contribution_criteria: List[str] = Field(
        default_factory=list,
        description="Substantial contribution criteria applied",
    )
    dnsh_criteria: List[str] = Field(
        default_factory=list,
        description="DNSH criteria applied per taxonomy objective",
    )
    minimum_social_safeguards: str = Field(
        "",
        description="Minimum social safeguards compliance description",
    )
    fossil_gas_included: bool = Field(
        False, description="Whether investments in fossil gas taxonomy activities are included"
    )
    nuclear_included: bool = Field(
        False, description="Whether investments in nuclear taxonomy activities are included"
    )
    third_party_verification: bool = Field(
        False, description="Whether taxonomy alignment is third-party verified"
    )


class MonitoringApproach(BaseModel):
    """Monitoring approach for sustainable investment objective."""

    methodology: str = Field(
        "", description="Methodology for ongoing monitoring"
    )
    frequency: str = Field(
        "quarterly", description="Monitoring frequency"
    )
    kpis: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="KPIs monitored: {name, target, unit, frequency}",
    )
    escalation_process: str = Field(
        "", description="Process when KPIs breach thresholds"
    )
    reporting_to_investors: str = Field(
        "", description="How monitoring results are reported to investors"
    )


class Article9DNSHApproach(BaseModel):
    """Do No Significant Harm approach for Article 9."""

    methodology: str = Field(
        "", description="Methodology for DNSH assessment"
    )
    pai_indicators_considered: List[str] = Field(
        default_factory=list,
        description="PAI indicators considered in DNSH assessment",
    )
    thresholds: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Thresholds applied: {indicator, threshold, unit}",
    )
    alignment_with_oecd_ungp: str = Field(
        "",
        description="How investments align with OECD Guidelines and UN Guiding Principles",
    )
    mandatory_pai_coverage: float = Field(
        100.0, ge=0.0, le=100.0,
        description="Percentage of mandatory PAI indicators addressed",
    )


class Article9DataSource(BaseModel):
    """Individual data source entry for Article 9."""

    source_name: str = Field("", description="Name of data source")
    provider: str = Field("", description="Data provider")
    coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Coverage percentage"
    )
    data_type: str = Field(
        "", description="Type of data (reported, estimated, modeled)"
    )
    update_frequency: str = Field(
        "", description="Data update frequency"
    )
    quality_rating: str = Field(
        "", description="Data quality rating (high, medium, low)"
    )


class Article9DataSources(BaseModel):
    """Data sources and processing methodology for Article 9."""

    sources: List[Article9DataSource] = Field(
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
    proportion_estimated: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage of data that is estimated vs. reported",
    )


class Article9DueDiligence(BaseModel):
    """Due diligence process for Article 9."""

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
    esg_integration: str = Field(
        "", description="How ESG is integrated in the investment process"
    )
    controversies_monitoring: str = Field(
        "", description="How ESG controversies are monitored"
    )


class Article9Benchmark(BaseModel):
    """Designated reference benchmark for Article 9."""

    has_benchmark: bool = Field(
        False, description="Whether a reference benchmark is designated"
    )
    benchmark_name: str = Field(
        "", description="Name of the designated benchmark"
    )
    benchmark_type: str = Field(
        "", description="CTB (Climate Transition Benchmark) or PAB (Paris-Aligned Benchmark)"
    )
    benchmark_provider: str = Field(
        "", description="Benchmark index provider"
    )
    alignment_description: str = Field(
        "",
        description="How the benchmark aligns with the sustainable investment objective",
    )
    methodology_summary: str = Field(
        "", description="Summary of benchmark methodology"
    )
    deviation_policy: str = Field(
        "",
        description="How deviations from the benchmark are managed",
    )
    no_benchmark_rationale: str = Field(
        "",
        description="If no benchmark, explanation of how objective is attained",
    )


class AnnexIIIPrecontractualData(BaseModel):
    """Complete input data for Annex III pre-contractual disclosure (Article 9)."""

    product_info: Article9ProductInfo
    sustainable_objective: SustainableObjective = Field(
        default_factory=SustainableObjective
    )
    investment_strategy: Article9Strategy = Field(
        default_factory=Article9Strategy
    )
    proportions: Article9Proportions = Field(
        default_factory=Article9Proportions
    )
    taxonomy_alignment: Article9TaxonomyAlignment = Field(
        default_factory=Article9TaxonomyAlignment
    )
    monitoring: MonitoringApproach = Field(
        default_factory=MonitoringApproach
    )
    dnsh_approach: Article9DNSHApproach = Field(
        default_factory=Article9DNSHApproach
    )
    data_sources: Article9DataSources = Field(
        default_factory=Article9DataSources
    )
    due_diligence: Article9DueDiligence = Field(
        default_factory=Article9DueDiligence
    )
    benchmark: Article9Benchmark = Field(
        default_factory=Article9Benchmark
    )


# ---------------------------------------------------------------------------
#  Template Configuration
# ---------------------------------------------------------------------------

class TemplateConfig(BaseModel):
    """Configuration for Annex III template rendering."""

    currency: str = Field("EUR", description="Currency code")
    language: str = Field("en", description="Output language code")
    include_taxonomy_diagrams: bool = Field(
        True, description="Include taxonomy alignment diagrams"
    )
    include_asset_allocation_chart: bool = Field(
        True, description="Include asset allocation visual"
    )
    include_sdg_mapping: bool = Field(
        True, description="Include SDG mapping in output"
    )
    disclaimer_text: str = Field(
        "",
        description="Custom disclaimer text to append",
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class AnnexIIIPrecontractualTemplate:
    """
    SFDR RTS Annex III pre-contractual disclosure template for Article 9 products.

    Generates the mandatory pre-contractual information required under
    SFDR Delegated Regulation (EU) 2022/1288 Annex III. Covers 10 sections
    for financial products that have sustainable investment as their objective.

    The header states: "This financial product has sustainable investment
    as its objective."

    Attributes:
        config: Optional configuration dictionary.
        PACK_ID: Pack identifier (PACK-011).
        TEMPLATE_NAME: Template identifier.
        VERSION: Template version.

    Example:
        >>> template = AnnexIIIPrecontractualTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "sustainable investment as its objective" in md
    """

    PACK_ID = "PACK-011"
    TEMPLATE_NAME = "annex_iii_precontractual"
    VERSION = "1.0"

    SFDR_ARTICLE = "Article 9"
    REGULATION_REF = "Regulation (EU) 2019/2088"
    RTS_REF = "Delegated Regulation (EU) 2022/1288, Annex III"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize AnnexIIIPrecontractualTemplate.

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
            data: Report data dictionary matching AnnexIIIPrecontractualData schema.
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
        Render the Annex III pre-contractual disclosure as Markdown.

        Args:
            data: Report data dictionary matching AnnexIIIPrecontractualData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header(data))
        sections.append(self._md_section_1_objective(data))
        sections.append(self._md_section_2_strategy(data))
        sections.append(self._md_section_3_proportions(data))
        sections.append(self._md_section_4_taxonomy_alignment(data))
        sections.append(self._md_section_5_monitoring(data))
        sections.append(self._md_section_6_dnsh(data))
        sections.append(self._md_section_7_data_sources(data))
        sections.append(self._md_section_8_limitations(data))
        sections.append(self._md_section_9_due_diligence(data))
        sections.append(self._md_section_10_engagement_benchmark(data))

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the Annex III pre-contractual disclosure as self-contained HTML.

        Args:
            data: Report data dictionary matching AnnexIIIPrecontractualData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_section_1_objective(data))
        sections.append(self._html_section_2_strategy(data))
        sections.append(self._html_section_3_proportions(data))
        sections.append(self._html_section_4_taxonomy_alignment(data))
        sections.append(self._html_section_5_monitoring(data))
        sections.append(self._html_section_6_dnsh(data))
        sections.append(self._html_section_7_data_sources(data))
        sections.append(self._html_section_8_limitations(data))
        sections.append(self._html_section_9_due_diligence(data))
        sections.append(self._html_section_10_engagement_benchmark(data))

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Annex III Pre-contractual Disclosure (Article 9)",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the Annex III pre-contractual disclosure as structured JSON.

        Args:
            data: Report data dictionary matching AnnexIIIPrecontractualData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        pi = data.get("product_info", {})
        obj = data.get("sustainable_objective", {})
        strat = data.get("investment_strategy", {})
        prop = data.get("proportions", {})
        tax = data.get("taxonomy_alignment", {})
        mon = data.get("monitoring", {})
        dnsh = data.get("dnsh_approach", {})
        ds = data.get("data_sources", {})
        dd = data.get("due_diligence", {})
        bm = data.get("benchmark", {})

        report: Dict[str, Any] = {
            "report_type": "sfdr_annex_iii_precontractual",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "regulation": self.REGULATION_REF,
            "rts_annex": self.RTS_REF,
            "sfdr_article": self.SFDR_ARTICLE,
            "product_identification": {
                "product_name": pi.get("product_name", ""),
                "isin": pi.get("isin", ""),
                "lei": pi.get("lei", ""),
                "sfdr_classification": pi.get("sfdr_classification", "article_9"),
                "fund_type": pi.get("fund_type", ""),
                "currency": pi.get("currency", "EUR"),
                "management_company": pi.get("management_company", ""),
                "domicile": pi.get("domicile", ""),
                "inception_date": pi.get("inception_date", ""),
            },
            "sustainable_objective": {
                "type": obj.get("objective_type", "environmental"),
                "description": obj.get("objective_description", ""),
                "environmental_objectives": obj.get("environmental_objectives", []),
                "social_objectives": obj.get("social_objectives", []),
                "sdg_alignment": obj.get("sdg_alignment", []),
                "paris_alignment": obj.get("paris_alignment", False),
                "carbon_reduction_target": obj.get("carbon_reduction_target"),
                "index_designation": obj.get("index_designation"),
            },
            "investment_strategy": {
                "approach": strat.get("approach", ""),
                "binding_commitments": strat.get("binding_commitments", []),
                "exclusions": strat.get("exclusions", []),
                "positive_screening": strat.get("positive_screening", []),
                "good_governance_assessment": strat.get("good_governance_assessment", ""),
                "minimum_safeguards": strat.get("minimum_safeguards", ""),
            },
            "proportions": {
                "sustainable_total_pct": prop.get("sustainable_total_pct", 100.0),
                "taxonomy_aligned_pct": prop.get("taxonomy_aligned_pct", 0.0),
                "other_environmental_pct": prop.get("other_environmental_pct", 0.0),
                "social_pct": prop.get("social_pct", 0.0),
                "not_sustainable_pct": prop.get("not_sustainable_pct", 0.0),
                "cash_pct": prop.get("cash_pct", 0.0),
                "hedging_pct": prop.get("hedging_pct", 0.0),
            },
            "taxonomy_alignment": {
                "minimum_alignment_pct": tax.get("minimum_alignment_pct", 0.0),
                "objective_breakdown": tax.get("objective_breakdown", []),
                "fossil_gas_included": tax.get("fossil_gas_included", False),
                "nuclear_included": tax.get("nuclear_included", False),
                "third_party_verification": tax.get("third_party_verification", False),
            },
            "monitoring": {
                "methodology": mon.get("methodology", ""),
                "frequency": mon.get("frequency", "quarterly"),
                "kpis": mon.get("kpis", []),
                "escalation_process": mon.get("escalation_process", ""),
            },
            "dnsh_approach": {
                "methodology": dnsh.get("methodology", ""),
                "pai_indicators_considered": dnsh.get("pai_indicators_considered", []),
                "thresholds": dnsh.get("thresholds", []),
                "alignment_with_oecd_ungp": dnsh.get("alignment_with_oecd_ungp", ""),
                "mandatory_pai_coverage": dnsh.get("mandatory_pai_coverage", 100.0),
            },
            "data_sources": {
                "sources": ds.get("sources", []),
                "estimation_methodology": ds.get("estimation_methodology", ""),
                "data_quality_score": ds.get("data_quality_score"),
                "proportion_estimated": ds.get("proportion_estimated", 0.0),
            },
            "limitations": ds.get("limitations", []),
            "due_diligence": {
                "description": dd.get("description", ""),
                "internal_resources": dd.get("internal_resources", []),
                "external_resources": dd.get("external_resources", []),
                "frequency": dd.get("frequency", ""),
                "esg_integration": dd.get("esg_integration", ""),
            },
            "benchmark": {
                "has_benchmark": bm.get("has_benchmark", False),
                "benchmark_name": bm.get("benchmark_name", ""),
                "benchmark_type": bm.get("benchmark_type", ""),
                "alignment_description": bm.get("alignment_description", ""),
                "no_benchmark_rationale": bm.get("no_benchmark_rationale", ""),
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
            f"> **This financial product has sustainable investment as its objective.**\n\n"
            f"**Product:** {name}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_objective(self, data: Dict[str, Any]) -> str:
        """Section 1: Sustainable investment objective."""
        pi = data.get("product_info", {})
        obj = data.get("sustainable_objective", {})
        obj_type = obj.get("objective_type", "environmental")
        obj_desc = obj.get("objective_description", "")
        env_objs = obj.get("environmental_objectives", [])
        soc_objs = obj.get("social_objectives", [])
        sdgs = obj.get("sdg_alignment", [])
        paris = obj.get("paris_alignment", False)
        carbon_target = obj.get("carbon_reduction_target")
        index = obj.get("index_designation")

        lines: List[str] = [
            "## 1. Sustainable Investment Objective\n",
            "| Field | Value |",
            "|-------|-------|",
            f"| **Product Name** | {pi.get('product_name', 'N/A')} |",
            f"| **ISIN** | {pi.get('isin', 'N/A') or 'N/A'} |",
            f"| **LEI** | {pi.get('lei', 'N/A') or 'N/A'} |",
            f"| **Classification** | {self._format_classification(pi.get('sfdr_classification', ''))} |",
            f"| **Fund Type** | {pi.get('fund_type', 'N/A') or 'N/A'} |",
            f"| **Currency** | {pi.get('currency', 'EUR')} |",
            f"| **Management Company** | {pi.get('management_company', 'N/A') or 'N/A'} |",
            f"| **Domicile** | {pi.get('domicile', 'N/A') or 'N/A'} |",
            "",
        ]

        lines.append(
            f"### Objective Type: {self._format_objective_type(obj_type)}\n"
        )

        if obj_desc:
            lines.append(f"{obj_desc}\n")

        if paris:
            lines.append(
                "**Paris Agreement Alignment:** This product targets alignment "
                "with the objectives of the Paris Agreement.\n"
            )
        if carbon_target:
            lines.append(f"**Carbon Reduction Target:** {carbon_target}\n")

        if index:
            lines.append(
                f"**Designated EU Climate Benchmark:** {index}\n"
            )

        if env_objs:
            lines.append("### Environmental Objectives\n")
            for item in env_objs:
                lines.append(f"- {item}")
            lines.append("")

        if soc_objs:
            lines.append("### Social Objectives\n")
            for item in soc_objs:
                lines.append(f"- {item}")
            lines.append("")

        if sdgs:
            lines.append("### SDG Alignment\n")
            lines.append("| SDG | Name | Relevance |")
            lines.append("|-----|------|-----------|")
            for sdg in sdgs:
                lines.append(
                    f"| SDG {sdg.get('sdg_number', '')} | "
                    f"{sdg.get('sdg_name', '')} | "
                    f"{sdg.get('relevance', '')} |"
                )

        return "\n".join(lines)

    def _md_section_2_strategy(self, data: Dict[str, Any]) -> str:
        """Section 2: Investment strategy and binding elements."""
        strat = data.get("investment_strategy", {})
        lines: List[str] = [
            "## 2. Investment Strategy\n",
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

        positive = strat.get("positive_screening", [])
        if positive:
            lines.append("### Positive Screening Criteria\n")
            for p in positive:
                lines.append(f"- {p}")
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

        stewardship = strat.get("stewardship_code", "")
        if stewardship:
            lines.append(f"### Stewardship Code\n\n{stewardship}\n")

        return "\n".join(lines)

    def _md_section_3_proportions(self, data: Dict[str, Any]) -> str:
        """Section 3: Proportion of investments (near 100% sustainable)."""
        prop = data.get("proportions", {})
        sustainable = prop.get("sustainable_total_pct", 100.0)
        taxonomy = prop.get("taxonomy_aligned_pct", 0.0)
        other_env = prop.get("other_environmental_pct", 0.0)
        social = prop.get("social_pct", 0.0)
        not_sust = prop.get("not_sustainable_pct", 0.0)
        cash = prop.get("cash_pct", 0.0)
        hedging = prop.get("hedging_pct", 0.0)
        enabling = prop.get("enabling_activities_pct", 0.0)
        transitional = prop.get("transitional_activities_pct", 0.0)

        lines: List[str] = [
            "## 3. Proportion of Investments\n",
            "**This Article 9 product has sustainable investment as its objective.**\n",
            f"A minimum of **{sustainable:.1f}%** of this product's investments "
            "are sustainable investments. The remaining investments are used "
            "exclusively for hedging and liquidity management purposes.\n",
            "| Category | Proportion |",
            "|----------|-----------|",
            f"| **Sustainable Investments** | **{sustainable:.1f}%** |",
            f"|   - Taxonomy-aligned | {taxonomy:.1f}% |",
            f"|   - Other environmental | {other_env:.1f}% |",
            f"|   - Social | {social:.1f}% |",
            f"| **Not sustainable (cash/hedging)** | **{not_sust:.1f}%** |",
            f"|   - Cash/equivalents | {cash:.1f}% |",
            f"|   - Hedging | {hedging:.1f}% |",
            "",
            f"| Taxonomy Activity Type | Proportion |",
            f"|------------------------|-----------|",
            f"| Enabling activities | {enabling:.1f}% |",
            f"| Transitional activities | {transitional:.1f}% |",
            "",
        ]

        # ASCII chart
        lines.append("### Asset Allocation Breakdown\n")
        lines.append("```")
        chart_items = [
            ("Taxonomy-aligned", taxonomy),
            ("Other environ.", other_env),
            ("Social", social),
            ("Cash/Hedging", not_sust),
        ]
        for label, pct in chart_items:
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            lines.append(f"  {label:20s} [{bar:<50s}] {pct:.1f}%")
        lines.append("```")

        return "\n".join(lines)

    def _md_section_4_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Section 4: EU Taxonomy alignment."""
        tax = data.get("taxonomy_alignment", {})
        min_pct = tax.get("minimum_alignment_pct", 0.0)
        objectives = tax.get("objective_breakdown", [])
        sc_criteria = tax.get("substantial_contribution_criteria", [])
        dnsh_criteria = tax.get("dnsh_criteria", [])
        social_safeguards = tax.get("minimum_social_safeguards", "")
        fossil_gas = tax.get("fossil_gas_included", False)
        nuclear = tax.get("nuclear_included", False)
        verified = tax.get("third_party_verification", False)

        lines: List[str] = [
            "## 4. EU Taxonomy Alignment\n",
            f"**Minimum taxonomy alignment commitment:** {min_pct:.1f}%\n",
            f"**Third-party verification:** {'Yes' if verified else 'No'}\n",
        ]

        # Fossil gas / nuclear disclosure
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

        if sc_criteria:
            lines.append("### Substantial Contribution Criteria\n")
            for c in sc_criteria:
                lines.append(f"- {c}")
            lines.append("")

        if dnsh_criteria:
            lines.append("### DNSH Criteria per Objective\n")
            for c in dnsh_criteria:
                lines.append(f"- {c}")
            lines.append("")

        if social_safeguards:
            lines.append(f"### Minimum Social Safeguards\n\n{social_safeguards}")

        return "\n".join(lines)

    def _md_section_5_monitoring(self, data: Dict[str, Any]) -> str:
        """Section 5: Monitoring of sustainable investment objective."""
        mon = data.get("monitoring", {})
        methodology = mon.get("methodology", "")
        frequency = mon.get("frequency", "quarterly")
        kpis = mon.get("kpis", [])
        escalation = mon.get("escalation_process", "")
        reporting = mon.get("reporting_to_investors", "")

        lines: List[str] = [
            "## 5. Monitoring of Sustainable Investment Objective\n",
            f"**Monitoring Frequency:** {frequency}\n",
        ]

        if methodology:
            lines.append(f"### Methodology\n\n{methodology}\n")

        if kpis:
            lines.append("### Key Performance Indicators\n")
            lines.append("| KPI | Target | Unit | Frequency |")
            lines.append("|-----|--------|------|-----------|")
            for kpi in kpis:
                lines.append(
                    f"| {kpi.get('name', '')} | "
                    f"{kpi.get('target', '')} | "
                    f"{kpi.get('unit', '')} | "
                    f"{kpi.get('frequency', '')} |"
                )
            lines.append("")

        if escalation:
            lines.append(f"### Escalation Process\n\n{escalation}\n")

        if reporting:
            lines.append(f"### Investor Reporting\n\n{reporting}")

        return "\n".join(lines)

    def _md_section_6_dnsh(self, data: Dict[str, Any]) -> str:
        """Section 6: Do No Significant Harm approach."""
        dnsh = data.get("dnsh_approach", {})
        methodology = dnsh.get("methodology", "")
        indicators = dnsh.get("pai_indicators_considered", [])
        thresholds = dnsh.get("thresholds", [])
        oecd_ungp = dnsh.get("alignment_with_oecd_ungp", "")
        pai_coverage = dnsh.get("mandatory_pai_coverage", 100.0)

        lines: List[str] = [
            "## 6. Do No Significant Harm (DNSH)\n",
            "The sustainable investments made by this product do not "
            "significantly harm any environmental or social objective.\n",
            f"**Mandatory PAI Coverage:** {pai_coverage:.1f}%\n",
        ]

        if methodology:
            lines.append(f"### Methodology\n\n{methodology}\n")

        if indicators:
            lines.append("### PAI Indicators Considered\n")
            for i, ind in enumerate(indicators, 1):
                lines.append(f"{i}. {ind}")
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

    def _md_section_7_data_sources(self, data: Dict[str, Any]) -> str:
        """Section 7: Data sources and processing."""
        ds = data.get("data_sources", {})
        sources = ds.get("sources", [])
        estimation = ds.get("estimation_methodology", "")
        quality = ds.get("data_quality_score")
        estimated_pct = ds.get("proportion_estimated", 0.0)

        lines: List[str] = [
            "## 7. Data Sources and Processing\n",
        ]

        if sources:
            lines.append("| Source | Provider | Coverage | Type | Quality | Update Freq. |")
            lines.append("|--------|----------|----------|------|---------|--------------|")
            for s in sources:
                name = s.get("source_name", "")
                provider = s.get("provider", "")
                coverage = s.get("coverage_pct", 0.0)
                dtype = s.get("data_type", "")
                quality_r = s.get("quality_rating", "")
                freq = s.get("update_frequency", "")
                lines.append(
                    f"| {name} | {provider} | {coverage:.1f}% | "
                    f"{dtype} | {quality_r} | {freq} |"
                )
            lines.append("")

        if estimation:
            lines.append(f"### Estimation Methodology\n\n{estimation}\n")

        lines.append(f"**Proportion of estimated data:** {estimated_pct:.1f}%\n")

        if quality is not None:
            lines.append(f"**Data Quality Score:** {quality:.1f}/100")

        return "\n".join(lines)

    def _md_section_8_limitations(self, data: Dict[str, Any]) -> str:
        """Section 8: Limitations to methodologies and data."""
        ds = data.get("data_sources", {})
        limitations = ds.get("limitations", [])

        lines: List[str] = [
            "## 8. Limitations to Methodologies and Data\n",
        ]

        if limitations:
            for lim in limitations:
                lines.append(f"- {lim}")
            lines.append("")
            lines.append(
                "These limitations do not affect the ability of the product to "
                "meet its sustainable investment objective."
            )
        else:
            lines.append("No material limitations have been identified.")

        return "\n".join(lines)

    def _md_section_9_due_diligence(self, data: Dict[str, Any]) -> str:
        """Section 9: Due diligence process."""
        dd = data.get("due_diligence", {})
        description = dd.get("description", "")
        internal = dd.get("internal_resources", [])
        external = dd.get("external_resources", [])
        frequency = dd.get("frequency", "")
        esg_int = dd.get("esg_integration", "")
        controversies = dd.get("controversies_monitoring", "")

        lines: List[str] = [
            "## 9. Due Diligence\n",
        ]

        if description:
            lines.append(f"{description}\n")

        if esg_int:
            lines.append(f"### ESG Integration\n\n{esg_int}\n")

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
            lines.append(f"**Review Frequency:** {frequency}\n")

        if controversies:
            lines.append(f"### Controversies Monitoring\n\n{controversies}")

        return "\n".join(lines)

    def _md_section_10_engagement_benchmark(self, data: Dict[str, Any]) -> str:
        """Section 10: Engagement policies and designated benchmark."""
        strat = data.get("investment_strategy", {})
        engagement = strat.get("engagement_policy", "")
        bm = data.get("benchmark", {})
        has_bm = bm.get("has_benchmark", False)
        bm_name = bm.get("benchmark_name", "")
        bm_type = bm.get("benchmark_type", "")
        bm_provider = bm.get("benchmark_provider", "")
        alignment_desc = bm.get("alignment_description", "")
        methodology_summary = bm.get("methodology_summary", "")
        deviation_policy = bm.get("deviation_policy", "")
        no_bm_rationale = bm.get("no_benchmark_rationale", "")

        lines: List[str] = [
            "## 10. Engagement Policies and Designated Benchmark\n",
        ]

        # Engagement
        lines.append("### Engagement Policies\n")
        if engagement:
            lines.append(f"{engagement}\n")
        else:
            lines.append(
                "The engagement policy is described in the product's "
                "investment strategy documentation.\n"
            )

        # Benchmark
        lines.append("### Designated Reference Benchmark\n")
        if has_bm:
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            lines.append(f"| **Benchmark** | {bm_name} |")
            if bm_type:
                lines.append(f"| **Type** | {bm_type} |")
            if bm_provider:
                lines.append(f"| **Provider** | {bm_provider} |")
            lines.append("")

            if alignment_desc:
                lines.append(f"**Alignment with Objective:**\n{alignment_desc}\n")
            if methodology_summary:
                lines.append(f"**Methodology:**\n{methodology_summary}\n")
            if deviation_policy:
                lines.append(f"**Deviation Policy:**\n{deviation_policy}")
        else:
            lines.append("No EU Climate Benchmark has been designated for this product.\n")
            if no_bm_rationale:
                lines.append(
                    f"**How the sustainable objective is attained:**\n{no_bm_rationale}"
                )
            else:
                lines.append(
                    "The sustainable investment objective is attained through "
                    "the investment strategy described in Section 2."
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_objective(self, data: Dict[str, Any]) -> str:
        """Build HTML sustainable objective section."""
        pi = data.get("product_info", {})
        obj = data.get("sustainable_objective", {})
        obj_type = obj.get("objective_type", "environmental")
        obj_desc = obj.get("objective_description", "")
        env_objs = obj.get("environmental_objectives", [])
        soc_objs = obj.get("social_objectives", [])
        sdgs = obj.get("sdg_alignment", [])
        paris = obj.get("paris_alignment", False)

        parts: List[str] = [
            '<div class="section objective-box">',
            "<h2>1. Sustainable Investment Objective</h2>",
            '<p class="objective-statement"><strong>This financial product has '
            "sustainable investment as its objective.</strong></p>",
            '<table class="info-table">',
            f"<tr><td><strong>Product Name</strong></td>"
            f"<td>{_esc(pi.get('product_name', 'N/A'))}</td></tr>",
            f"<tr><td><strong>ISIN</strong></td>"
            f"<td>{_esc(pi.get('isin', '') or 'N/A')}</td></tr>",
            f"<tr><td><strong>LEI</strong></td>"
            f"<td>{_esc(pi.get('lei', '') or 'N/A')}</td></tr>",
            f"<tr><td><strong>Classification</strong></td>"
            f"<td>{_esc(self._format_classification(pi.get('sfdr_classification', '')))}</td></tr>",
            f"<tr><td><strong>Fund Type</strong></td>"
            f"<td>{_esc(pi.get('fund_type', '') or 'N/A')}</td></tr>",
            f"<tr><td><strong>Currency</strong></td>"
            f"<td>{_esc(pi.get('currency', 'EUR'))}</td></tr>",
            f"<tr><td><strong>Management Company</strong></td>"
            f"<td>{_esc(pi.get('management_company', '') or 'N/A')}</td></tr>",
            f"<tr><td><strong>Domicile</strong></td>"
            f"<td>{_esc(pi.get('domicile', '') or 'N/A')}</td></tr>",
            "</table>",
            f"<h3>Objective Type: {_esc(self._format_objective_type(obj_type))}</h3>",
        ]

        if obj_desc:
            parts.append(f"<p>{_esc(obj_desc)}</p>")

        if paris:
            parts.append(
                '<p class="highlight"><strong>Paris Agreement Alignment:</strong> '
                "This product targets alignment with the Paris Agreement.</p>"
            )

        if env_objs:
            parts.append("<h3>Environmental Objectives</h3><ul>")
            for item in env_objs:
                parts.append(f"<li>{_esc(item)}</li>")
            parts.append("</ul>")

        if soc_objs:
            parts.append("<h3>Social Objectives</h3><ul>")
            for item in soc_objs:
                parts.append(f"<li>{_esc(item)}</li>")
            parts.append("</ul>")

        if sdgs:
            parts.append("<h3>SDG Alignment</h3>")
            parts.append('<table class="data-table">')
            parts.append("<tr><th>SDG</th><th>Name</th><th>Relevance</th></tr>")
            for sdg in sdgs:
                parts.append(
                    f"<tr><td>SDG {_esc(str(sdg.get('sdg_number', '')))}</td>"
                    f"<td>{_esc(str(sdg.get('sdg_name', '')))}</td>"
                    f"<td>{_esc(str(sdg.get('relevance', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_2_strategy(self, data: Dict[str, Any]) -> str:
        """Build HTML investment strategy section."""
        strat = data.get("investment_strategy", {})
        parts: List[str] = ['<div class="section"><h2>2. Investment Strategy</h2>']

        approach = strat.get("approach", "")
        if approach:
            parts.append(f"<h3>Approach</h3><p>{_esc(approach)}</p>")

        commitments = strat.get("binding_commitments", [])
        if commitments:
            parts.append("<h3>Binding Commitments</h3><ul>")
            for c in commitments:
                parts.append(f"<li>{_esc(c)}</li>")
            parts.append("</ul>")

        positive = strat.get("positive_screening", [])
        if positive:
            parts.append("<h3>Positive Screening</h3><ul>")
            for p in positive:
                parts.append(f"<li>{_esc(p)}</li>")
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

        safeguards = strat.get("minimum_safeguards", "")
        if safeguards:
            parts.append(f"<h3>Minimum Safeguards</h3><p>{_esc(safeguards)}</p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_3_proportions(self, data: Dict[str, Any]) -> str:
        """Build HTML proportions section."""
        prop = data.get("proportions", {})
        sustainable = prop.get("sustainable_total_pct", 100.0)

        parts: List[str] = [
            '<div class="section"><h2>3. Proportion of Investments</h2>',
            f"<p><strong>Sustainable investments:</strong> {sustainable:.1f}%</p>",
            '<table class="data-table">',
            "<tr><th>Category</th><th>Proportion</th><th>Visual</th></tr>",
        ]

        items = [
            ("Taxonomy-aligned", prop.get("taxonomy_aligned_pct", 0.0), "#2ecc71"),
            ("Other environmental", prop.get("other_environmental_pct", 0.0), "#27ae60"),
            ("Social", prop.get("social_pct", 0.0), "#3498db"),
            ("Cash/Hedging", prop.get("not_sustainable_pct", 0.0), "#95a5a6"),
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

    def _html_section_4_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Build HTML taxonomy alignment section."""
        tax = data.get("taxonomy_alignment", {})
        min_pct = tax.get("minimum_alignment_pct", 0.0)
        objectives = tax.get("objective_breakdown", [])
        fossil_gas = tax.get("fossil_gas_included", False)
        nuclear = tax.get("nuclear_included", False)
        verified = tax.get("third_party_verification", False)

        parts: List[str] = [
            '<div class="section"><h2>4. EU Taxonomy Alignment</h2>',
            f"<p><strong>Minimum taxonomy alignment:</strong> {min_pct:.1f}%</p>",
            f"<p><strong>Third-party verified:</strong> {'Yes' if verified else 'No'}</p>",
        ]

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

        parts.append("</div>")
        return "".join(parts)

    def _html_section_5_monitoring(self, data: Dict[str, Any]) -> str:
        """Build HTML monitoring section."""
        mon = data.get("monitoring", {})
        frequency = mon.get("frequency", "quarterly")
        kpis = mon.get("kpis", [])
        methodology = mon.get("methodology", "")

        parts: List[str] = [
            '<div class="section"><h2>5. Monitoring</h2>',
            f"<p><strong>Frequency:</strong> {_esc(frequency)}</p>",
        ]

        if methodology:
            parts.append(f"<p>{_esc(methodology)}</p>")

        if kpis:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>KPI</th><th>Target</th><th>Unit</th><th>Frequency</th></tr>")
            for kpi in kpis:
                parts.append(
                    f"<tr><td>{_esc(str(kpi.get('name', '')))}</td>"
                    f"<td>{_esc(str(kpi.get('target', '')))}</td>"
                    f"<td>{_esc(str(kpi.get('unit', '')))}</td>"
                    f"<td>{_esc(str(kpi.get('frequency', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_6_dnsh(self, data: Dict[str, Any]) -> str:
        """Build HTML DNSH section."""
        dnsh = data.get("dnsh_approach", {})
        methodology = dnsh.get("methodology", "")
        indicators = dnsh.get("pai_indicators_considered", [])
        thresholds = dnsh.get("thresholds", [])

        parts: List[str] = [
            '<div class="section"><h2>6. Do No Significant Harm (DNSH)</h2>',
            "<p>The sustainable investments do not significantly harm any "
            "environmental or social objective.</p>",
        ]

        if methodology:
            parts.append(f"<h3>Methodology</h3><p>{_esc(methodology)}</p>")

        if indicators:
            parts.append("<h3>PAI Indicators</h3><ol>")
            for ind in indicators:
                parts.append(f"<li>{_esc(ind)}</li>")
            parts.append("</ol>")

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

    def _html_section_7_data_sources(self, data: Dict[str, Any]) -> str:
        """Build HTML data sources section."""
        ds = data.get("data_sources", {})
        sources = ds.get("sources", [])

        parts: List[str] = [
            '<div class="section"><h2>7. Data Sources and Processing</h2>'
        ]

        if sources:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Source</th><th>Provider</th><th>Coverage</th>"
                "<th>Type</th><th>Quality</th></tr>"
            )
            for s in sources:
                parts.append(
                    f"<tr><td>{_esc(s.get('source_name', ''))}</td>"
                    f"<td>{_esc(s.get('provider', ''))}</td>"
                    f"<td>{s.get('coverage_pct', 0.0):.1f}%</td>"
                    f"<td>{_esc(s.get('data_type', ''))}</td>"
                    f"<td>{_esc(s.get('quality_rating', ''))}</td></tr>"
                )
            parts.append("</table>")

        estimation = ds.get("estimation_methodology", "")
        if estimation:
            parts.append(f"<h3>Estimation Methodology</h3><p>{_esc(estimation)}</p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_8_limitations(self, data: Dict[str, Any]) -> str:
        """Build HTML limitations section."""
        ds = data.get("data_sources", {})
        limitations = ds.get("limitations", [])

        parts: List[str] = ['<div class="section"><h2>8. Limitations</h2>']
        if limitations:
            parts.append("<ul>")
            for lim in limitations:
                parts.append(f"<li>{_esc(lim)}</li>")
            parts.append("</ul>")
            parts.append(
                "<p>These limitations do not affect the ability to meet "
                "the sustainable investment objective.</p>"
            )
        else:
            parts.append("<p>No material limitations identified.</p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_9_due_diligence(self, data: Dict[str, Any]) -> str:
        """Build HTML due diligence section."""
        dd = data.get("due_diligence", {})
        parts: List[str] = ['<div class="section"><h2>9. Due Diligence</h2>']

        desc = dd.get("description", "")
        if desc:
            parts.append(f"<p>{_esc(desc)}</p>")

        esg_int = dd.get("esg_integration", "")
        if esg_int:
            parts.append(f"<h3>ESG Integration</h3><p>{_esc(esg_int)}</p>")

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

    def _html_section_10_engagement_benchmark(self, data: Dict[str, Any]) -> str:
        """Build HTML engagement and benchmark section."""
        strat = data.get("investment_strategy", {})
        engagement = strat.get("engagement_policy", "")
        bm = data.get("benchmark", {})
        has_bm = bm.get("has_benchmark", False)
        bm_name = bm.get("benchmark_name", "")
        bm_type = bm.get("benchmark_type", "")
        alignment_desc = bm.get("alignment_description", "")
        no_bm_rationale = bm.get("no_benchmark_rationale", "")

        parts: List[str] = [
            '<div class="section"><h2>10. Engagement Policies and Benchmark</h2>'
        ]

        parts.append("<h3>Engagement Policies</h3>")
        if engagement:
            parts.append(f"<p>{_esc(engagement)}</p>")

        parts.append("<h3>Designated Reference Benchmark</h3>")
        if has_bm:
            parts.append(f"<p><strong>{_esc(bm_name)}</strong></p>")
            if bm_type:
                parts.append(f"<p>Type: {_esc(bm_type)}</p>")
            if alignment_desc:
                parts.append(f"<p>{_esc(alignment_desc)}</p>")
        else:
            parts.append("<p>No EU Climate Benchmark designated.</p>")
            if no_bm_rationale:
                parts.append(f"<p>{_esc(no_bm_rationale)}</p>")

        parts.append("</div>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    def _format_classification(self, classification: str) -> str:
        """Format SFDR classification for display."""
        mapping = {
            "article_9": "Article 9 (Dark Green)",
            "article_9_transitional": "Article 9 (Dark Green - Transitional)",
        }
        return mapping.get(classification, classification)

    def _format_objective_type(self, obj_type: str) -> str:
        """Format objective type for display."""
        mapping = {
            "environmental": "Environmental Objective",
            "social": "Social Objective",
            "environmental_and_social": "Environmental and Social Objectives",
        }
        return mapping.get(obj_type, obj_type)

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
            "h1 { color: #1a5276; border-bottom: 3px solid #1abc9c; padding-bottom: 10px; }\n"
            "h2 { color: #1a5276; margin-top: 30px; border-bottom: 1px solid #bdc3c7; "
            "padding-bottom: 5px; }\n"
            "h3 { color: #2c3e50; }\n"
            ".section { margin-bottom: 30px; padding: 15px; "
            "background: #fafafa; border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".objective-box { background: #e8f8f5; border-color: #1abc9c; }\n"
            ".objective-statement { font-size: 1.1em; color: #0e6655; font-weight: bold; }\n"
            ".highlight { color: #117864; background: #d1f2eb; padding: 8px; "
            "border-radius: 4px; }\n"
            ".info-table, .data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".info-table td, .data-table td, .data-table th { padding: 8px 12px; "
            "border: 1px solid #ddd; }\n"
            ".data-table th { background: #1a5276; color: white; text-align: left; }\n"
            ".info-table tr:nth-child(even), .data-table tr:nth-child(even) { "
            "background: #f2f3f4; }\n"
            ".provenance { margin-top: 40px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            ".footer { margin-top: 30px; font-size: 0.85em; color: #7f8c8d; "
            "border-top: 1px solid #bdc3c7; padding-top: 10px; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f'<p><strong>{self.RTS_REF}</strong></p>\n'
            '<p class="objective-statement">This financial product has sustainable '
            "investment as its objective.</p>\n"
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
