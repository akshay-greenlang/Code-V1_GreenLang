"""
Supply Chain Map Report Template - PACK-007 EUDR Professional Pack

This module generates supply chain visualization reports with multi-tier supply chain graphs,
tier breakdown tables, concentration analysis, origin mapping, critical path summaries, and
diversification options for EUDR compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from supply_chain_map_report import SupplyChainMapReportTemplate, ReportData
    >>> data = ReportData(
    ...     operator_name="Global Timber Inc",
    ...     report_date="2026-03-15",
    ...     total_tiers=4
    ... )
    >>> template = SupplyChainMapReportTemplate()
    >>> report = template.render(data, format="markdown")
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReportConfig(BaseModel):
    """Configuration for Supply Chain Map Report generation."""

    include_tier_breakdown: bool = Field(
        default=True,
        description="Include tier breakdown analysis"
    )
    include_concentration_analysis: bool = Field(
        default=True,
        description="Include supplier concentration analysis"
    )
    include_origin_mapping: bool = Field(
        default=True,
        description="Include geographic origin mapping"
    )
    include_diversification_options: bool = Field(
        default=True,
        description="Include supplier diversification recommendations"
    )
    max_graph_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum depth for supply chain graph"
    )


class SupplyChainNode(BaseModel):
    """Supply chain node (supplier/facility)."""

    node_id: str = Field(..., description="Node identifier")
    node_name: str = Field(..., description="Node name (supplier/facility)")
    node_type: str = Field(..., description="SUPPLIER, PROCESSOR, HARVESTER, etc.")
    tier: int = Field(..., ge=0, description="Tier level (0=operator, 1=direct, etc.)")
    country: str = Field(..., description="Country of operation")
    risk_level: str = Field(..., description="HIGH, MEDIUM, LOW")
    certification_status: str = Field(..., description="Certification status")
    volume_tonnes: float = Field(..., ge=0, description="Volume sourced (tonnes)")


class SupplyChainEdge(BaseModel):
    """Supply chain edge (relationship)."""

    from_node_id: str = Field(..., description="Source node ID")
    to_node_id: str = Field(..., description="Destination node ID")
    relationship_type: str = Field(..., description="SUPPLIES_TO, PROCESSES_FOR, etc.")
    volume_tonnes: float = Field(..., ge=0, description="Volume through this link")
    traceability_level: str = Field(..., description="FULL, PARTIAL, LIMITED")


class TierSummary(BaseModel):
    """Supply chain tier summary."""

    tier: int = Field(..., ge=0, description="Tier level")
    tier_name: str = Field(..., description="Tier name (e.g., Direct Suppliers)")
    total_entities: int = Field(..., ge=0, description="Number of entities in tier")
    total_volume_tonnes: float = Field(..., ge=0, description="Total volume from tier")
    avg_risk_score: float = Field(..., ge=0, le=100, description="Average risk score")
    certified_entities: int = Field(..., ge=0, description="Certified entities in tier")
    countries_represented: int = Field(..., ge=0, description="Number of countries")
    traceability_coverage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Traceability coverage percentage"
    )


class ConcentrationMetric(BaseModel):
    """Supplier concentration metric."""

    metric_name: str = Field(..., description="Metric name")
    metric_value: float = Field(..., description="Metric value")
    interpretation: str = Field(..., description="Interpretation of the metric")
    risk_assessment: str = Field(..., description="HIGH, MEDIUM, LOW concentration risk")


class GeographicOrigin(BaseModel):
    """Geographic origin summary."""

    country: str = Field(..., description="Country of origin")
    region: Optional[str] = Field(None, description="Region/state within country")
    volume_tonnes: float = Field(..., ge=0, description="Volume from this origin")
    percentage_of_total: float = Field(..., ge=0, le=100, description="Percentage of total volume")
    risk_level: str = Field(..., description="HIGH, MEDIUM, LOW")
    supplier_count: int = Field(..., ge=0, description="Number of suppliers in this origin")


class CriticalPath(BaseModel):
    """Critical supply chain path."""

    path_id: str = Field(..., description="Path identifier")
    description: str = Field(..., description="Path description")
    nodes: List[str] = Field(..., description="Node IDs in path")
    total_volume_tonnes: float = Field(..., ge=0, description="Volume through this path")
    percentage_of_total: float = Field(..., ge=0, le=100, description="Percentage of total volume")
    highest_risk_node: str = Field(..., description="Highest risk node in path")
    criticality_score: float = Field(..., ge=0, le=100, description="Path criticality score")


class DiversificationOption(BaseModel):
    """Supplier diversification option."""

    option_id: str = Field(..., description="Option identifier")
    current_situation: str = Field(..., description="Current concentration/risk")
    proposed_action: str = Field(..., description="Proposed diversification action")
    expected_benefit: str = Field(..., description="Expected benefit")
    implementation_complexity: str = Field(..., description="HIGH, MEDIUM, LOW")
    estimated_timeline: str = Field(..., description="Estimated timeline")
    priority: str = Field(..., description="HIGH, MEDIUM, LOW")


class ReportData(BaseModel):
    """Data model for Supply Chain Map Report."""

    operator_name: str = Field(..., description="Operator name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    commodity: str = Field(..., description="Primary commodity")
    total_tiers: int = Field(..., ge=1, description="Total number of tiers mapped")
    nodes: List[SupplyChainNode] = Field(
        default_factory=list,
        description="Supply chain nodes"
    )
    edges: List[SupplyChainEdge] = Field(
        default_factory=list,
        description="Supply chain edges"
    )
    tier_summaries: List[TierSummary] = Field(
        default_factory=list,
        description="Tier summaries"
    )
    concentration_metrics: List[ConcentrationMetric] = Field(
        default_factory=list,
        description="Concentration metrics"
    )
    geographic_origins: List[GeographicOrigin] = Field(
        default_factory=list,
        description="Geographic origins"
    )
    critical_paths: List[CriticalPath] = Field(
        default_factory=list,
        description="Critical supply chain paths"
    )
    diversification_options: List[DiversificationOption] = Field(
        default_factory=list,
        description="Diversification options"
    )
    mapping_completeness: float = Field(
        ...,
        ge=0,
        le=100,
        description="Supply chain mapping completeness percentage"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class SupplyChainMapReportTemplate:
    """
    Supply Chain Map Report Template for EUDR Professional Pack.

    Generates comprehensive supply chain visualization reports with multi-tier graphs,
    tier breakdown, concentration analysis, origin mapping, and diversification options.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = SupplyChainMapReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Supply Chain Map" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Supply Chain Map Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the supply chain map report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Supply Chain Map Report for {data.operator_name} in {format} format"
        )

        if format == "markdown":
            content = self._render_markdown(data)
        elif format == "html":
            content = self._render_html(data)
        elif format == "json":
            content = self._render_json(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Add provenance hash
        content_hash = self._calculate_hash(content)
        logger.info(f"Report generated with hash: {content_hash}")

        return content

    def _render_markdown(self, data: ReportData) -> str:
        """Render report in Markdown format."""
        sections = []

        # Header
        sections.append(f"# Supply Chain Map Report")
        sections.append(f"")
        sections.append(f"**Operator:** {data.operator_name}")
        sections.append(f"**Commodity:** {data.commodity}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Tiers Mapped:** {data.total_tiers}")
        sections.append(f"**Mapping Completeness:** {data.mapping_completeness:.1f}%")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        sections.append(
            f"This supply chain map report visualizes the multi-tier supply chain for "
            f"**{data.commodity}** sourced by **{data.operator_name}**. The analysis covers "
            f"{data.total_tiers} tiers with {len(data.nodes)} total entities and "
            f"{len(data.edges)} supply relationships. Mapping completeness is "
            f"**{data.mapping_completeness:.1f}%**."
        )
        sections.append(f"")

        # Supply Chain Overview
        sections.append(f"## Supply Chain Overview")
        sections.append(f"")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Total Entities | {len(data.nodes):,} |")
        sections.append(f"| Total Relationships | {len(data.edges):,} |")
        sections.append(f"| Tiers Mapped | {data.total_tiers} |")
        sections.append(f"| Mapping Completeness | {data.mapping_completeness:.1f}% |")

        if data.nodes:
            total_volume = sum(node.volume_tonnes for node in data.nodes if node.tier == 1)
            sections.append(f"| Total Volume (Direct) | {total_volume:,.0f} tonnes |")

        sections.append(f"")

        # Supply Chain Graph (Text-Based)
        if data.nodes and data.edges:
            sections.append(f"## Supply Chain Graph")
            sections.append(f"")
            sections.append(f"```")
            sections.append(self._create_text_graph(data.nodes, data.edges))
            sections.append(f"```")
            sections.append(f"")

        # Tier Breakdown
        if self.config.include_tier_breakdown and data.tier_summaries:
            sections.append(f"## Tier Breakdown Analysis")
            sections.append(f"")
            sections.append(
                f"Detailed analysis of each supply chain tier:"
            )
            sections.append(f"")

            sections.append(
                f"| Tier | Name | Entities | Volume (t) | Avg Risk | Certified | Countries | Traceability |"
            )
            sections.append(
                f"|------|------|----------|------------|----------|-----------|-----------|--------------|"
            )

            for tier in sorted(data.tier_summaries, key=lambda x: x.tier):
                sections.append(
                    f"| {tier.tier} | {tier.tier_name} | {tier.total_entities} | "
                    f"{tier.total_volume_tonnes:,.0f} | {tier.avg_risk_score:.1f} | "
                    f"{tier.certified_entities} | {tier.countries_represented} | "
                    f"{tier.traceability_coverage:.1f}% |"
                )
            sections.append(f"")

        # Concentration Analysis
        if self.config.include_concentration_analysis and data.concentration_metrics:
            sections.append(f"## Supplier Concentration Analysis")
            sections.append(f"")
            sections.append(
                f"Analysis of supplier concentration risks:"
            )
            sections.append(f"")

            sections.append(
                f"| Metric | Value | Interpretation | Risk |"
            )
            sections.append(
                f"|--------|-------|----------------|------|"
            )

            for metric in data.concentration_metrics:
                sections.append(
                    f"| {metric.metric_name} | {metric.metric_value:.2f} | "
                    f"{metric.interpretation} | {metric.risk_assessment} |"
                )
            sections.append(f"")

            # Key insights
            high_risk_metrics = [m for m in data.concentration_metrics if m.risk_assessment == "HIGH"]
            if high_risk_metrics:
                sections.append(f"### High Concentration Risk Areas")
                sections.append(f"")
                for metric in high_risk_metrics:
                    sections.append(f"- **{metric.metric_name}**: {metric.interpretation}")
                sections.append(f"")

        # Geographic Origin Mapping
        if self.config.include_origin_mapping and data.geographic_origins:
            sections.append(f"## Geographic Origin Mapping")
            sections.append(f"")
            sections.append(
                f"Distribution of supply across geographic origins:"
            )
            sections.append(f"")

            sections.append(
                f"| Country | Region | Volume (t) | % of Total | Risk | Suppliers |"
            )
            sections.append(
                f"|---------|--------|------------|------------|------|-----------|"
            )

            for origin in sorted(data.geographic_origins, key=lambda x: x.volume_tonnes, reverse=True):
                region_display = origin.region or "N/A"
                sections.append(
                    f"| {origin.country} | {region_display} | {origin.volume_tonnes:,.0f} | "
                    f"{origin.percentage_of_total:.1f}% | {origin.risk_level} | "
                    f"{origin.supplier_count} |"
                )
            sections.append(f"")

            # Top origins
            top_5 = sorted(data.geographic_origins, key=lambda x: x.volume_tonnes, reverse=True)[:5]
            top_5_pct = sum(o.percentage_of_total for o in top_5)
            sections.append(
                f"**Top 5 Origins:** Represent {top_5_pct:.1f}% of total volume"
            )
            sections.append(f"")

        # Critical Path Summary
        if data.critical_paths:
            sections.append(f"## Critical Supply Chain Paths")
            sections.append(f"")
            sections.append(
                f"High-volume or high-risk paths through the supply chain:"
            )
            sections.append(f"")

            sections.append(
                f"| Path ID | Description | Volume (t) | % of Total | Highest Risk Node | Criticality |"
            )
            sections.append(
                f"|---------|-------------|------------|------------|-------------------|-------------|"
            )

            for path in sorted(data.critical_paths, key=lambda x: x.criticality_score, reverse=True):
                sections.append(
                    f"| {path.path_id} | {path.description} | {path.total_volume_tonnes:,.0f} | "
                    f"{path.percentage_of_total:.1f}% | {path.highest_risk_node} | "
                    f"{path.criticality_score:.1f} |"
                )
            sections.append(f"")

            # Path details
            top_path = max(data.critical_paths, key=lambda x: x.criticality_score)
            sections.append(f"### Most Critical Path: {top_path.path_id}")
            sections.append(f"")
            sections.append(f"**Path:** {' → '.join(top_path.nodes)}")
            sections.append(f"")
            sections.append(
                f"This path represents {top_path.percentage_of_total:.1f}% of total volume "
                f"({top_path.total_volume_tonnes:,.0f} tonnes) with a criticality score of "
                f"{top_path.criticality_score:.1f}/100."
            )
            sections.append(f"")

        # Diversification Options
        if self.config.include_diversification_options and data.diversification_options:
            sections.append(f"## Supplier Diversification Options")
            sections.append(f"")
            sections.append(
                f"Recommended actions to reduce concentration risk:"
            )
            sections.append(f"")

            # High priority options
            high_priority = [opt for opt in data.diversification_options if opt.priority == "HIGH"]
            if high_priority:
                sections.append(f"### High Priority Actions")
                sections.append(f"")
                for opt in high_priority:
                    sections.append(f"**{opt.option_id}**")
                    sections.append(f"- **Current Situation:** {opt.current_situation}")
                    sections.append(f"- **Proposed Action:** {opt.proposed_action}")
                    sections.append(f"- **Expected Benefit:** {opt.expected_benefit}")
                    sections.append(f"- **Timeline:** {opt.estimated_timeline}")
                    sections.append(f"- **Complexity:** {opt.implementation_complexity}")
                    sections.append(f"")

            # All options table
            sections.append(f"### All Diversification Options")
            sections.append(f"")
            sections.append(
                f"| Option ID | Current Situation | Proposed Action | Complexity | Timeline | Priority |"
            )
            sections.append(
                f"|-----------|-------------------|-----------------|------------|----------|----------|"
            )

            for opt in sorted(data.diversification_options, key=lambda x: (x.priority == "HIGH", x.priority == "MEDIUM"), reverse=True):
                sections.append(
                    f"| {opt.option_id} | {opt.current_situation[:30]}... | "
                    f"{opt.proposed_action[:30]}... | {opt.implementation_complexity} | "
                    f"{opt.estimated_timeline} | {opt.priority} |"
                )
            sections.append(f"")

        # Recommendations
        sections.append(f"## Recommendations")
        sections.append(f"")

        if data.mapping_completeness < 90:
            sections.append(
                f"1. **Improve Mapping Completeness:** Current mapping is {data.mapping_completeness:.1f}%. "
                f"Target 90%+ for comprehensive visibility."
            )

        high_risk_nodes = [n for n in data.nodes if n.risk_level == "HIGH"]
        if high_risk_nodes:
            sections.append(
                f"2. **Address High-Risk Nodes:** {len(high_risk_nodes)} entities classified as "
                f"high-risk require immediate due diligence."
            )

        if data.diversification_options:
            sections.append(
                f"3. **Implement Diversification:** {len([o for o in data.diversification_options if o.priority == 'HIGH'])} "
                f"high-priority diversification actions identified."
            )

        sections.append(
            f"4. **Enhance Traceability:** Focus on improving traceability in lower tiers (Tier 3+)."
        )
        sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EUDR Professional Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Supply Chain Map Report - {data.operator_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #16a085; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #16a085; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #16a085; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        pre {{ background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Supply Chain Map Report</h1>

    <div class="summary">
        <p><strong>Operator:</strong> {data.operator_name}</p>
        <p><strong>Commodity:</strong> {data.commodity}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Tiers Mapped:</strong> {data.total_tiers}</p>
        <p><strong>Mapping Completeness:</strong> <span class="metric">{data.mapping_completeness:.1f}%</span></p>
    </div>

    <h2>Executive Summary</h2>
    <p>
        This supply chain map report visualizes the multi-tier supply chain for <strong>{data.commodity}</strong>
        sourced by {data.operator_name}. The analysis covers {data.total_tiers} tiers with
        {len(data.nodes)} total entities and {len(data.edges)} supply relationships.
    </p>
"""

        if data.tier_summaries:
            html += f"""
    <h2>Tier Breakdown Analysis</h2>
    <table>
        <tr><th>Tier</th><th>Name</th><th>Entities</th><th>Volume (t)</th><th>Avg Risk</th><th>Certified</th><th>Countries</th><th>Traceability</th></tr>
"""
            for tier in sorted(data.tier_summaries, key=lambda x: x.tier):
                html += f"""        <tr>
            <td>{tier.tier}</td>
            <td>{tier.tier_name}</td>
            <td>{tier.total_entities}</td>
            <td>{tier.total_volume_tonnes:,.0f}</td>
            <td>{tier.avg_risk_score:.1f}</td>
            <td>{tier.certified_entities}</td>
            <td>{tier.countries_represented}</td>
            <td>{tier.traceability_coverage:.1f}%</td>
        </tr>
"""
            html += f"""    </table>
"""

        if data.geographic_origins:
            html += f"""
    <h2>Geographic Origin Mapping</h2>
    <table>
        <tr><th>Country</th><th>Region</th><th>Volume (t)</th><th>% of Total</th><th>Risk</th><th>Suppliers</th></tr>
"""
            for origin in sorted(data.geographic_origins, key=lambda x: x.volume_tonnes, reverse=True)[:20]:
                html += f"""        <tr>
            <td>{origin.country}</td>
            <td>{origin.region or 'N/A'}</td>
            <td>{origin.volume_tonnes:,.0f}</td>
            <td>{origin.percentage_of_total:.1f}%</td>
            <td>{origin.risk_level}</td>
            <td>{origin.supplier_count}</td>
        </tr>
"""
            html += f"""    </table>
"""

        html += f"""
    <div class="footer">
        <p><em>Report generated on {data.report_date} using GreenLang EUDR Professional Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "supply_chain_map",
            "operator_name": data.operator_name,
            "commodity": data.commodity,
            "report_date": data.report_date,
            "total_tiers": data.total_tiers,
            "mapping_completeness": data.mapping_completeness,
            "nodes": [node.dict() for node in data.nodes],
            "edges": [edge.dict() for edge in data.edges],
            "tier_summaries": [tier.dict() for tier in data.tier_summaries],
            "concentration_metrics": [metric.dict() for metric in data.concentration_metrics],
            "geographic_origins": [origin.dict() for origin in data.geographic_origins],
            "critical_paths": [path.dict() for path in data.critical_paths],
            "diversification_options": [opt.dict() for opt in data.diversification_options],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.dict(),
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_text_graph(self, nodes: List[SupplyChainNode], edges: List[SupplyChainEdge]) -> str:
        """Create text-based supply chain graph visualization."""
        graph_lines = []
        graph_lines.append("Supply Chain Structure (Text-Based Visualization)")
        graph_lines.append("")

        # Group nodes by tier
        tiers = {}
        for node in nodes:
            if node.tier not in tiers:
                tiers[node.tier] = []
            tiers[node.tier].append(node)

        # Display by tier
        for tier_level in sorted(tiers.keys())[:self.config.max_graph_depth]:
            tier_nodes = tiers[tier_level]
            graph_lines.append(f"Tier {tier_level}:")
            for node in tier_nodes[:10]:  # Limit to 10 nodes per tier
                risk_indicator = "!" if node.risk_level == "HIGH" else "~" if node.risk_level == "MEDIUM" else "✓"
                graph_lines.append(
                    f"  [{risk_indicator}] {node.node_name} ({node.country}) - {node.volume_tonnes:,.0f}t"
                )
            if len(tier_nodes) > 10:
                graph_lines.append(f"  ... and {len(tier_nodes) - 10} more")
            graph_lines.append("")

        graph_lines.append(f"Legend: [!] High Risk  [~] Medium Risk  [✓] Low Risk")

        return "\n".join(graph_lines)
