"""
AnnexIIIWebsiteTemplate - SFDR RTS Annex III website disclosure template.

This module implements the Annex III website disclosure template for
PACK-010 SFDR Article 8 products. It generates the mandatory website
disclosure required under SFDR Delegated Regulation (EU) 2022/1288,
covering 12 sections from summary through designated reference benchmark.

Example:
    >>> template = AnnexIIIWebsiteTemplate()
    >>> data = WebsiteDisclosureData(product_name="ESG Equity Fund", ...)
    >>> html = template.render_html(data.model_dump())
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

class WebProductInfo(BaseModel):
    """Product identification for website disclosure."""

    product_name: str = Field(..., min_length=1, description="Legal product name")
    isin: str = Field("", description="ISIN code")
    lei: str = Field("", description="Legal Entity Identifier")
    sfdr_classification: str = Field("article_8", description="SFDR classification")
    management_company: str = Field("", description="Management company name")
    last_updated: str = Field("", description="Last update date of the disclosure")

    @field_validator("sfdr_classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        """Validate SFDR classification."""
        allowed = {"article_8", "article_8_plus", "article_9"}
        if v not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v


class WebSummary(BaseModel):
    """Summary section content."""

    overview: str = Field("", description="High-level product summary")
    es_characteristics_summary: str = Field(
        "", description="Summary of E/S characteristics promoted"
    )
    sustainable_investment_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Minimum sustainable investment %"
    )
    taxonomy_alignment_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Minimum taxonomy alignment %"
    )


class WebMonitoring(BaseModel):
    """Monitoring methodology section."""

    methodology_description: str = Field(
        "", description="Description of monitoring methodology"
    )
    indicators: List[str] = Field(
        default_factory=list, description="Sustainability indicators monitored"
    )
    frequency: str = Field("", description="Monitoring frequency")
    reporting_to_investors: str = Field(
        "", description="How results are reported to investors"
    )
    thresholds: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alert thresholds: {indicator, threshold, action}",
    )


class WebMethodology(BaseModel):
    """Methodologies for E/S characteristics measurement."""

    description: str = Field("", description="Methodology overview")
    esg_rating_providers: List[str] = Field(
        default_factory=list, description="ESG rating providers used"
    )
    scoring_approach: str = Field("", description="Scoring/rating approach")
    weighting_methodology: str = Field("", description="Weighting methodology")
    review_frequency: str = Field("", description="Methodology review frequency")


class WebDataSource(BaseModel):
    """Data source for website disclosure."""

    name: str = Field("", description="Source name")
    provider: str = Field("", description="Provider name")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage %")
    data_type: str = Field("", description="reported, estimated, modeled")
    update_frequency: str = Field("", description="Update frequency")


class WebDataSources(BaseModel):
    """Data sources and processing section."""

    sources: List[WebDataSource] = Field(default_factory=list)
    processing_description: str = Field(
        "", description="How data is processed"
    )
    proportion_estimated: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="% of data that is estimated"
    )
    quality_assurance: str = Field("", description="Data quality assurance process")


class WebLimitations(BaseModel):
    """Limitations section."""

    methodology_limitations: List[str] = Field(
        default_factory=list, description="Methodology limitations"
    )
    data_limitations: List[str] = Field(
        default_factory=list, description="Data limitations"
    )
    mitigation_measures: List[str] = Field(
        default_factory=list, description="Measures to mitigate limitations"
    )
    impact_on_attainment: str = Field(
        "", description="Impact of limitations on E/S characteristic attainment"
    )


class WebDueDiligence(BaseModel):
    """Due diligence section."""

    description: str = Field("", description="Due diligence process description")
    underlying_assets_review: str = Field(
        "", description="Review process for underlying assets"
    )
    internal_controls: List[str] = Field(
        default_factory=list, description="Internal controls applied"
    )
    external_verification: str = Field("", description="External verification")


class WebEngagement(BaseModel):
    """Engagement policies section."""

    engagement_description: str = Field("", description="Engagement policy description")
    voting_policy: str = Field("", description="Voting policy")
    escalation_process: str = Field("", description="Escalation process")
    engagement_outcomes: List[str] = Field(
        default_factory=list, description="Key engagement outcomes"
    )


class WebInvestmentStrategy(BaseModel):
    """Investment strategy section for website disclosure."""

    strategy_description: str = Field("", description="Strategy description")
    binding_elements: List[str] = Field(
        default_factory=list, description="Binding elements"
    )
    exclusions: List[str] = Field(default_factory=list, description="Exclusions")
    good_governance: str = Field("", description="Good governance assessment")


class WebProportions(BaseModel):
    """Proportion of investments."""

    sustainable_pct: float = Field(0.0, ge=0.0, le=100.0)
    taxonomy_aligned_pct: float = Field(0.0, ge=0.0, le=100.0)
    other_env_pct: float = Field(0.0, ge=0.0, le=100.0)
    social_pct: float = Field(0.0, ge=0.0, le=100.0)
    not_sustainable_pct: float = Field(0.0, ge=0.0, le=100.0)


class WebReferenceBenchmark(BaseModel):
    """Reference benchmark information."""

    designated: bool = Field(False, description="Whether a benchmark is designated")
    name: str = Field("", description="Benchmark name")
    how_aligned: str = Field(
        "", description="How the benchmark is aligned with E/S characteristics"
    )
    how_differs: str = Field(
        "", description="How it differs from a broad market index"
    )


class WebsiteDisclosureData(BaseModel):
    """Complete input data for Annex III website disclosure."""

    product_info: WebProductInfo
    summary: WebSummary = Field(default_factory=WebSummary)
    es_characteristics: List[str] = Field(
        default_factory=list, description="E/S characteristics list"
    )
    investment_strategy: WebInvestmentStrategy = Field(
        default_factory=WebInvestmentStrategy
    )
    proportions: WebProportions = Field(default_factory=WebProportions)
    monitoring: WebMonitoring = Field(default_factory=WebMonitoring)
    methodology: WebMethodology = Field(default_factory=WebMethodology)
    data_sources: WebDataSources = Field(default_factory=WebDataSources)
    limitations: WebLimitations = Field(default_factory=WebLimitations)
    due_diligence: WebDueDiligence = Field(default_factory=WebDueDiligence)
    engagement: WebEngagement = Field(default_factory=WebEngagement)
    reference_benchmark: WebReferenceBenchmark = Field(
        default_factory=WebReferenceBenchmark
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class AnnexIIIWebsiteTemplate:
    """
    SFDR RTS Annex III website disclosure template for Article 8 products.

    Generates the mandatory website disclosure required under SFDR
    Delegated Regulation (EU) 2022/1288 Annex III, covering 12 sections.

    Example:
        >>> template = AnnexIIIWebsiteTemplate()
        >>> html = template.render_html(data)
    """

    PACK_ID = "PACK-010"
    TEMPLATE_NAME = "annex_iii_website"
    VERSION = "1.0"
    REGULATION_REF = "Regulation (EU) 2019/2088"
    RTS_REF = "Delegated Regulation (EU) 2022/1288, Annex III"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AnnexIIIWebsiteTemplate."""
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render website disclosure in the specified format.

        Args:
            data: Report data matching WebsiteDisclosureData schema.
            fmt: Output format.

        Returns:
            Rendered content.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'.")

    # ------------------------------------------------------------------ #
    #  Markdown rendering
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render website disclosure as Markdown."""
        pi = data.get("product_info", {})
        sections: List[str] = [
            (
                f"# Website Disclosure (Article 8)\n\n"
                f"**{self.RTS_REF}**\n\n"
                f"**Product:** {pi.get('product_name', 'Unknown')}\n\n"
                f"**Pack:** {self.PACK_ID} | "
                f"**Template:** {self.TEMPLATE_NAME} v{self.VERSION}\n\n"
                f"**Generated:** {self.generated_at}"
            ),
            self._md_section_1_summary(data),
            self._md_section_2_no_sustainable_obj(data),
            self._md_section_3_es_characteristics(data),
            self._md_section_4_investment_strategy(data),
            self._md_section_5_proportions(data),
            self._md_section_6_monitoring(data),
            self._md_section_7_methodologies(data),
            self._md_section_8_data_sources(data),
            self._md_section_9_limitations(data),
            self._md_section_10_due_diligence(data),
            self._md_section_11_engagement(data),
            self._md_section_12_benchmark(data),
        ]

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(provenance_hash)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render website disclosure as self-contained HTML."""
        sections: List[str] = [
            self._html_section_1_summary(data),
            self._html_section_2_no_sustainable_obj(data),
            self._html_section_3_es_characteristics(data),
            self._html_section_4_investment_strategy(data),
            self._html_section_5_proportions(data),
            self._html_section_6_monitoring(data),
            self._html_section_7_methodologies(data),
            self._html_section_8_data_sources(data),
            self._html_section_9_limitations(data),
            self._html_section_10_due_diligence(data),
            self._html_section_11_engagement(data),
            self._html_section_12_benchmark(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html("SFDR Website Disclosure", body, provenance_hash)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render website disclosure as structured JSON."""
        report: Dict[str, Any] = {
            "report_type": "sfdr_annex_iii_website",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "product_info": data.get("product_info", {}),
            "summary": data.get("summary", {}),
            "es_characteristics": data.get("es_characteristics", []),
            "investment_strategy": data.get("investment_strategy", {}),
            "proportions": data.get("proportions", {}),
            "monitoring": data.get("monitoring", {}),
            "methodology": data.get("methodology", {}),
            "data_sources": data.get("data_sources", {}),
            "limitations": data.get("limitations", {}),
            "due_diligence": data.get("due_diligence", {}),
            "engagement": data.get("engagement", {}),
            "reference_benchmark": data.get("reference_benchmark", {}),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_section_1_summary(self, data: Dict[str, Any]) -> str:
        """Section 1: Summary."""
        summary = data.get("summary", {})
        lines = ["## 1. Summary\n"]
        overview = summary.get("overview", "")
        if overview:
            lines.append(f"{overview}\n")
        es_summary = summary.get("es_characteristics_summary", "")
        if es_summary:
            lines.append(f"**E/S Characteristics:** {es_summary}\n")
        sust_pct = summary.get("sustainable_investment_pct")
        if sust_pct is not None:
            lines.append(f"**Minimum Sustainable Investment:** {sust_pct:.1f}%\n")
        tax_pct = summary.get("taxonomy_alignment_pct")
        if tax_pct is not None:
            lines.append(f"**Minimum Taxonomy Alignment:** {tax_pct:.1f}%")
        return "\n".join(lines)

    def _md_section_2_no_sustainable_obj(self, data: Dict[str, Any]) -> str:
        """Section 2: No sustainable investment objective statement."""
        pi = data.get("product_info", {})
        classification = pi.get("sfdr_classification", "article_8")
        lines = ["## 2. Sustainable Investment Objective\n"]
        if classification == "article_8":
            lines.append(
                "This financial product promotes environmental or social "
                "characteristics, but does not have as its objective sustainable investment."
            )
        elif classification == "article_8_plus":
            summary = data.get("summary", {})
            pct = summary.get("sustainable_investment_pct", 0)
            lines.append(
                f"This financial product promotes environmental or social "
                f"characteristics and commits to a minimum of {pct:.1f}% "
                f"sustainable investments."
            )
        return "\n".join(lines)

    def _md_section_3_es_characteristics(self, data: Dict[str, Any]) -> str:
        """Section 3: E/S characteristics."""
        chars = data.get("es_characteristics", [])
        lines = ["## 3. Environmental/Social Characteristics\n"]
        if chars:
            for c in chars:
                lines.append(f"- {c}")
        else:
            lines.append("*No characteristics specified.*")
        return "\n".join(lines)

    def _md_section_4_investment_strategy(self, data: Dict[str, Any]) -> str:
        """Section 4: Investment strategy."""
        strat = data.get("investment_strategy", {})
        lines = ["## 4. Investment Strategy\n"]
        desc = strat.get("strategy_description", "")
        if desc:
            lines.append(f"{desc}\n")
        binding = strat.get("binding_elements", [])
        if binding:
            lines.append("### Binding Elements\n")
            for b in binding:
                lines.append(f"- {b}")
            lines.append("")
        exclusions = strat.get("exclusions", [])
        if exclusions:
            lines.append("### Exclusions\n")
            for e in exclusions:
                lines.append(f"- {e}")
            lines.append("")
        governance = strat.get("good_governance", "")
        if governance:
            lines.append(f"### Good Governance\n\n{governance}")
        return "\n".join(lines)

    def _md_section_5_proportions(self, data: Dict[str, Any]) -> str:
        """Section 5: Proportion of investments."""
        p = data.get("proportions", {})
        lines = [
            "## 5. Proportion of Investments\n",
            "| Category | Proportion |",
            "|----------|-----------|",
            f"| Sustainable Total | {p.get('sustainable_pct', 0.0):.1f}% |",
            f"| Taxonomy-aligned | {p.get('taxonomy_aligned_pct', 0.0):.1f}% |",
            f"| Other environmental | {p.get('other_env_pct', 0.0):.1f}% |",
            f"| Social | {p.get('social_pct', 0.0):.1f}% |",
            f"| Not sustainable | {p.get('not_sustainable_pct', 0.0):.1f}% |",
        ]
        return "\n".join(lines)

    def _md_section_6_monitoring(self, data: Dict[str, Any]) -> str:
        """Section 6: Monitoring of E/S characteristics."""
        mon = data.get("monitoring", {})
        lines = ["## 6. Monitoring of E/S Characteristics\n"]
        desc = mon.get("methodology_description", "")
        if desc:
            lines.append(f"{desc}\n")
        indicators = mon.get("indicators", [])
        if indicators:
            lines.append("### Indicators Monitored\n")
            for ind in indicators:
                lines.append(f"- {ind}")
            lines.append("")
        freq = mon.get("frequency", "")
        if freq:
            lines.append(f"**Frequency:** {freq}\n")
        reporting = mon.get("reporting_to_investors", "")
        if reporting:
            lines.append(f"**Reporting to Investors:** {reporting}")
        return "\n".join(lines)

    def _md_section_7_methodologies(self, data: Dict[str, Any]) -> str:
        """Section 7: Methodologies."""
        meth = data.get("methodology", {})
        lines = ["## 7. Methodologies for E/S Characteristics\n"]
        desc = meth.get("description", "")
        if desc:
            lines.append(f"{desc}\n")
        providers = meth.get("esg_rating_providers", [])
        if providers:
            lines.append("### ESG Rating Providers\n")
            for p in providers:
                lines.append(f"- {p}")
            lines.append("")
        scoring = meth.get("scoring_approach", "")
        if scoring:
            lines.append(f"**Scoring Approach:** {scoring}\n")
        weighting = meth.get("weighting_methodology", "")
        if weighting:
            lines.append(f"**Weighting:** {weighting}")
        return "\n".join(lines)

    def _md_section_8_data_sources(self, data: Dict[str, Any]) -> str:
        """Section 8: Data sources and processing."""
        ds = data.get("data_sources", {})
        sources = ds.get("sources", [])
        lines = ["## 8. Data Sources and Processing\n"]
        if sources:
            lines.append("| Source | Provider | Coverage | Type | Update Frequency |")
            lines.append("|--------|----------|----------|------|------------------|")
            for s in sources:
                lines.append(
                    f"| {s.get('name', '')} | {s.get('provider', '')} | "
                    f"{s.get('coverage_pct', 0.0):.1f}% | "
                    f"{s.get('data_type', '')} | {s.get('update_frequency', '')} |"
                )
            lines.append("")
        processing = ds.get("processing_description", "")
        if processing:
            lines.append(f"### Processing\n\n{processing}\n")
        est = ds.get("proportion_estimated")
        if est is not None:
            lines.append(f"**Estimated Data:** {est:.1f}%\n")
        qa = ds.get("quality_assurance", "")
        if qa:
            lines.append(f"**Quality Assurance:** {qa}")
        return "\n".join(lines)

    def _md_section_9_limitations(self, data: Dict[str, Any]) -> str:
        """Section 9: Limitations."""
        lim = data.get("limitations", {})
        lines = ["## 9. Limitations to Methodologies and Data\n"]
        meth_lim = lim.get("methodology_limitations", [])
        if meth_lim:
            lines.append("### Methodology Limitations\n")
            for l in meth_lim:
                lines.append(f"- {l}")
            lines.append("")
        data_lim = lim.get("data_limitations", [])
        if data_lim:
            lines.append("### Data Limitations\n")
            for l in data_lim:
                lines.append(f"- {l}")
            lines.append("")
        mitigation = lim.get("mitigation_measures", [])
        if mitigation:
            lines.append("### Mitigation Measures\n")
            for m in mitigation:
                lines.append(f"- {m}")
            lines.append("")
        impact = lim.get("impact_on_attainment", "")
        if impact:
            lines.append(f"**Impact on Attainment:** {impact}")
        return "\n".join(lines)

    def _md_section_10_due_diligence(self, data: Dict[str, Any]) -> str:
        """Section 10: Due diligence."""
        dd = data.get("due_diligence", {})
        lines = ["## 10. Due Diligence\n"]
        desc = dd.get("description", "")
        if desc:
            lines.append(f"{desc}\n")
        review = dd.get("underlying_assets_review", "")
        if review:
            lines.append(f"### Underlying Assets Review\n\n{review}\n")
        controls = dd.get("internal_controls", [])
        if controls:
            lines.append("### Internal Controls\n")
            for c in controls:
                lines.append(f"- {c}")
            lines.append("")
        ext = dd.get("external_verification", "")
        if ext:
            lines.append(f"**External Verification:** {ext}")
        return "\n".join(lines)

    def _md_section_11_engagement(self, data: Dict[str, Any]) -> str:
        """Section 11: Engagement policies."""
        eng = data.get("engagement", {})
        lines = ["## 11. Engagement Policies\n"]
        desc = eng.get("engagement_description", "")
        if desc:
            lines.append(f"{desc}\n")
        voting = eng.get("voting_policy", "")
        if voting:
            lines.append(f"### Voting Policy\n\n{voting}\n")
        escalation = eng.get("escalation_process", "")
        if escalation:
            lines.append(f"### Escalation Process\n\n{escalation}\n")
        outcomes = eng.get("engagement_outcomes", [])
        if outcomes:
            lines.append("### Engagement Outcomes\n")
            for o in outcomes:
                lines.append(f"- {o}")
        return "\n".join(lines)

    def _md_section_12_benchmark(self, data: Dict[str, Any]) -> str:
        """Section 12: Designated reference benchmark."""
        rb = data.get("reference_benchmark", {})
        lines = ["## 12. Designated Reference Benchmark\n"]
        if rb.get("designated"):
            lines.append(f"**Benchmark:** {rb.get('name', 'N/A')}\n")
            how_aligned = rb.get("how_aligned", "")
            if how_aligned:
                lines.append(f"**Alignment:** {how_aligned}\n")
            how_differs = rb.get("how_differs", "")
            if how_differs:
                lines.append(f"**Differences from Broad Market Index:** {how_differs}")
        else:
            lines.append("No reference benchmark has been designated for this product.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML summary section."""
        summary = data.get("summary", {})
        parts = ['<div class="section"><h2>1. Summary</h2>']
        overview = summary.get("overview", "")
        if overview:
            parts.append(f"<p>{_esc(overview)}</p>")
        es_summary = summary.get("es_characteristics_summary", "")
        if es_summary:
            parts.append(f"<p><strong>E/S Characteristics:</strong> {_esc(es_summary)}</p>")
        sust = summary.get("sustainable_investment_pct")
        if sust is not None:
            parts.append(f"<p><strong>Minimum Sustainable Investment:</strong> {sust:.1f}%</p>")
        parts.append("</div>")
        return "".join(parts)

    def _html_section_2_no_sustainable_obj(self, data: Dict[str, Any]) -> str:
        """Build HTML no-sustainable-objective section."""
        pi = data.get("product_info", {})
        classification = pi.get("sfdr_classification", "article_8")
        parts = ['<div class="section disclaimer-box">']
        parts.append("<h2>2. Sustainable Investment Objective</h2>")
        if classification == "article_8":
            parts.append(
                '<p class="disclaimer">This financial product promotes environmental or '
                "social characteristics, but does not have as its objective sustainable "
                "investment.</p>"
            )
        elif classification == "article_8_plus":
            summary = data.get("summary", {})
            pct = summary.get("sustainable_investment_pct", 0)
            parts.append(
                f'<p class="disclaimer">This product commits to {pct:.1f}% '
                f"sustainable investments.</p>"
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_section_3_es_characteristics(self, data: Dict[str, Any]) -> str:
        """Build HTML E/S characteristics section."""
        chars = data.get("es_characteristics", [])
        parts = ['<div class="section"><h2>3. E/S Characteristics</h2>']
        if chars:
            parts.append("<ul>")
            for c in chars:
                parts.append(f"<li>{_esc(c)}</li>")
            parts.append("</ul>")
        else:
            parts.append("<p><em>No characteristics specified.</em></p>")
        parts.append("</div>")
        return "".join(parts)

    def _html_section_4_investment_strategy(self, data: Dict[str, Any]) -> str:
        """Build HTML investment strategy section."""
        strat = data.get("investment_strategy", {})
        parts = ['<div class="section"><h2>4. Investment Strategy</h2>']
        desc = strat.get("strategy_description", "")
        if desc:
            parts.append(f"<p>{_esc(desc)}</p>")
        binding = strat.get("binding_elements", [])
        if binding:
            parts.append("<h3>Binding Elements</h3><ul>")
            for b in binding:
                parts.append(f"<li>{_esc(b)}</li>")
            parts.append("</ul>")
        exclusions = strat.get("exclusions", [])
        if exclusions:
            parts.append("<h3>Exclusions</h3><ul>")
            for e in exclusions:
                parts.append(f"<li>{_esc(e)}</li>")
            parts.append("</ul>")
        parts.append("</div>")
        return "".join(parts)

    def _html_section_5_proportions(self, data: Dict[str, Any]) -> str:
        """Build HTML proportions section."""
        p = data.get("proportions", {})
        parts = ['<div class="section"><h2>5. Proportion of Investments</h2>']
        items = [
            ("Taxonomy-aligned", p.get("taxonomy_aligned_pct", 0.0), "#2ecc71"),
            ("Other environmental", p.get("other_env_pct", 0.0), "#27ae60"),
            ("Social", p.get("social_pct", 0.0), "#3498db"),
            ("Not sustainable", p.get("not_sustainable_pct", 0.0), "#95a5a6"),
        ]
        for label, pct, color in items:
            bar_w = max(int(pct * 3), 0)
            parts.append(
                f'<div style="margin:8px 0;">'
                f'<span style="display:inline-block;width:170px;">{_esc(label)}</span>'
                f'<div style="display:inline-block;background:{color};'
                f'width:{bar_w}px;height:18px;border-radius:3px;'
                f'vertical-align:middle;margin-right:8px;"></div>'
                f"<span>{pct:.1f}%</span></div>"
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_section_6_monitoring(self, data: Dict[str, Any]) -> str:
        """Build HTML monitoring section."""
        mon = data.get("monitoring", {})
        parts = ['<div class="section"><h2>6. Monitoring</h2>']
        desc = mon.get("methodology_description", "")
        if desc:
            parts.append(f"<p>{_esc(desc)}</p>")
        indicators = mon.get("indicators", [])
        if indicators:
            parts.append("<h3>Indicators</h3><ul>")
            for ind in indicators:
                parts.append(f"<li>{_esc(ind)}</li>")
            parts.append("</ul>")
        freq = mon.get("frequency", "")
        if freq:
            parts.append(f"<p><strong>Frequency:</strong> {_esc(freq)}</p>")
        parts.append("</div>")
        return "".join(parts)

    def _html_section_7_methodologies(self, data: Dict[str, Any]) -> str:
        """Build HTML methodologies section."""
        meth = data.get("methodology", {})
        parts = ['<div class="section"><h2>7. Methodologies</h2>']
        desc = meth.get("description", "")
        if desc:
            parts.append(f"<p>{_esc(desc)}</p>")
        providers = meth.get("esg_rating_providers", [])
        if providers:
            parts.append("<h3>ESG Rating Providers</h3><ul>")
            for p in providers:
                parts.append(f"<li>{_esc(p)}</li>")
            parts.append("</ul>")
        parts.append("</div>")
        return "".join(parts)

    def _html_section_8_data_sources(self, data: Dict[str, Any]) -> str:
        """Build HTML data sources section."""
        ds = data.get("data_sources", {})
        sources = ds.get("sources", [])
        parts = ['<div class="section"><h2>8. Data Sources</h2>']
        if sources:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Source</th><th>Provider</th><th>Coverage</th>"
                "<th>Type</th><th>Update</th></tr>"
            )
            for s in sources:
                parts.append(
                    f"<tr><td>{_esc(s.get('name', ''))}</td>"
                    f"<td>{_esc(s.get('provider', ''))}</td>"
                    f"<td>{s.get('coverage_pct', 0.0):.1f}%</td>"
                    f"<td>{_esc(s.get('data_type', ''))}</td>"
                    f"<td>{_esc(s.get('update_frequency', ''))}</td></tr>"
                )
            parts.append("</table>")
        parts.append("</div>")
        return "".join(parts)

    def _html_section_9_limitations(self, data: Dict[str, Any]) -> str:
        """Build HTML limitations section."""
        lim = data.get("limitations", {})
        parts = ['<div class="section"><h2>9. Limitations</h2>']
        meth_lim = lim.get("methodology_limitations", [])
        if meth_lim:
            parts.append("<h3>Methodology</h3><ul>")
            for l in meth_lim:
                parts.append(f"<li>{_esc(l)}</li>")
            parts.append("</ul>")
        data_lim = lim.get("data_limitations", [])
        if data_lim:
            parts.append("<h3>Data</h3><ul>")
            for l in data_lim:
                parts.append(f"<li>{_esc(l)}</li>")
            parts.append("</ul>")
        parts.append("</div>")
        return "".join(parts)

    def _html_section_10_due_diligence(self, data: Dict[str, Any]) -> str:
        """Build HTML due diligence section."""
        dd = data.get("due_diligence", {})
        parts = ['<div class="section"><h2>10. Due Diligence</h2>']
        desc = dd.get("description", "")
        if desc:
            parts.append(f"<p>{_esc(desc)}</p>")
        controls = dd.get("internal_controls", [])
        if controls:
            parts.append("<h3>Internal Controls</h3><ul>")
            for c in controls:
                parts.append(f"<li>{_esc(c)}</li>")
            parts.append("</ul>")
        parts.append("</div>")
        return "".join(parts)

    def _html_section_11_engagement(self, data: Dict[str, Any]) -> str:
        """Build HTML engagement section."""
        eng = data.get("engagement", {})
        parts = ['<div class="section"><h2>11. Engagement Policies</h2>']
        desc = eng.get("engagement_description", "")
        if desc:
            parts.append(f"<p>{_esc(desc)}</p>")
        voting = eng.get("voting_policy", "")
        if voting:
            parts.append(f"<h3>Voting</h3><p>{_esc(voting)}</p>")
        outcomes = eng.get("engagement_outcomes", [])
        if outcomes:
            parts.append("<h3>Outcomes</h3><ul>")
            for o in outcomes:
                parts.append(f"<li>{_esc(o)}</li>")
            parts.append("</ul>")
        parts.append("</div>")
        return "".join(parts)

    def _html_section_12_benchmark(self, data: Dict[str, Any]) -> str:
        """Build HTML benchmark section."""
        rb = data.get("reference_benchmark", {})
        parts = ['<div class="section"><h2>12. Reference Benchmark</h2>']
        if rb.get("designated"):
            parts.append(f"<p><strong>{_esc(rb.get('name', ''))}</strong></p>")
            aligned = rb.get("how_aligned", "")
            if aligned:
                parts.append(f"<p>{_esc(aligned)}</p>")
        else:
            parts.append("<p>No reference benchmark designated.</p>")
        parts.append("</div>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

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
        """Wrap HTML body in a complete document."""
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>\n"
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px auto; "
            "color: #2c3e50; line-height: 1.6; max-width: 1000px; }\n"
            "h1 { color: #1a5276; border-bottom: 3px solid #2ecc71; padding-bottom: 10px; }\n"
            "h2 { color: #1a5276; margin-top: 30px; border-bottom: 1px solid #bdc3c7; }\n"
            ".section { margin-bottom: 25px; padding: 15px; background: #fafafa; "
            "border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".disclaimer-box { background: #fef9e7; border-color: #f1c40f; }\n"
            ".disclaimer { font-weight: bold; color: #7d6608; }\n"
            ".data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".data-table td, .data-table th { padding: 8px 12px; border: 1px solid #ddd; }\n"
            ".data-table th { background: #2c3e50; color: white; text-align: left; }\n"
            ".data-table tr:nth-child(even) { background: #f2f3f4; }\n"
            ".provenance { margin-top: 40px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            ".footer { margin-top: 20px; font-size: 0.85em; color: #7f8c8d; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f"<p>Pack: {self.PACK_ID} | Template: {self.TEMPLATE_NAME} v{self.VERSION} | "
            f"Generated: {self.generated_at}</p>\n"
            f"{body}\n"
            f'<div class="provenance">Provenance Hash (SHA-256): {provenance_hash}</div>\n'
            f'<div class="footer">Generated by GreenLang {self.PACK_ID}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n"
            "</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _esc(value: str) -> str:
    """Escape HTML special characters."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
