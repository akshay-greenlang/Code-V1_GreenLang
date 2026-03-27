# -*- coding: utf-8 -*-
"""
OwnershipStructureReport - Corporate structure visualization for PACK-050.

Generates corporate structure data including entity hierarchy tree, ownership
percentages (direct and effective), control type per entity, JV partner
details, and organisational boundary overlay.

Sections:
    1. Structure Summary (total entities, entity types, boundary approach)
    2. Entity Hierarchy Tree (parent-child with depth levels)
    3. Ownership Details (direct %, effective %, control type)
    4. Joint Venture Partners (JV entities with partner details)
    5. Boundary Overlay (which entities are in/out of boundary)
    6. Provenance Footer

Output Formats: Markdown, HTML, JSON, CSV

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib, logging, uuid, json, time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class JvPartner(BaseModel):
    """Joint venture partner details."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    partner_name: str = Field("")
    partner_id: str = Field("")
    ownership_pct: Decimal = Field(Decimal("0"))
    is_operator: bool = Field(False)


class HierarchyNode(BaseModel):
    """Single entity in the corporate hierarchy."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_id: str = Field("")
    entity_name: str = Field("")
    entity_type: str = Field("", description="parent, subsidiary, jv, associate, spe")
    parent_entity_id: str = Field("", description="Empty for root entity")
    depth: int = Field(0, ge=0)
    country_code: str = Field("")
    direct_ownership_pct: Decimal = Field(Decimal("100"))
    effective_ownership_pct: Decimal = Field(Decimal("100"))
    control_type: str = Field("", description="operational, financial, equity, none")
    in_boundary: bool = Field(True)
    boundary_rationale: str = Field("")
    jv_partners: List[Dict[str, Any]] = Field(default_factory=list)
    sector: str = Field("")
    incorporation_date: str = Field("")


class OwnershipStructureReportInput(BaseModel):
    """Complete input for the ownership structure report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    consolidation_approach: str = Field("operational_control")
    root_entity_name: str = Field("")
    hierarchy: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class OwnershipStructureReportOutput(BaseModel):
    """Rendered ownership structure report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    consolidation_approach: str = Field("")
    root_entity_name: str = Field("")
    total_entities: int = Field(0)
    subsidiaries_count: int = Field(0)
    jv_count: int = Field(0)
    associate_count: int = Field(0)
    in_boundary_count: int = Field(0)
    out_boundary_count: int = Field(0)
    max_depth: int = Field(0)
    hierarchy: List[HierarchyNode] = Field(default_factory=list)
    jv_details: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field("")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class OwnershipStructureReport:
    """
    Corporate ownership structure report template for PACK-050.

    Produces a corporate hierarchy visualization with ownership chain
    calculations, JV partner details, and boundary overlay.

    Example:
        >>> tpl = OwnershipStructureReport()
        >>> report = tpl.render(data)
        >>> md = tpl.export_markdown(report)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> OwnershipStructureReportOutput:
        """Render ownership structure from input data."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = OwnershipStructureReportInput(**data) if isinstance(data, dict) else data

        nodes = [HierarchyNode(**h) if isinstance(h, dict) else h for h in inp.hierarchy]

        # Resolve JV partners on each node
        for node in nodes:
            if node.jv_partners:
                node.jv_partners = [
                    JvPartner(**p).model_dump() if isinstance(p, dict) else p
                    for p in node.jv_partners
                ]

        # Compute statistics
        subs = [n for n in nodes if n.entity_type == "subsidiary"]
        jvs = [n for n in nodes if n.entity_type == "jv"]
        assocs = [n for n in nodes if n.entity_type == "associate"]
        in_bound = [n for n in nodes if n.in_boundary]
        out_bound = [n for n in nodes if not n.in_boundary]
        max_depth = max((n.depth for n in nodes), default=0)

        # Collect JV detail records
        jv_details: List[Dict[str, Any]] = []
        for node in jvs:
            for partner in node.jv_partners:
                p = partner if isinstance(partner, dict) else partner.model_dump() if hasattr(partner, "model_dump") else partner
                jv_details.append({
                    "entity_name": node.entity_name,
                    "entity_id": node.entity_id,
                    "partner_name": p.get("partner_name", ""),
                    "partner_ownership_pct": str(p.get("ownership_pct", "0")),
                    "is_operator": p.get("is_operator", False),
                })

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = OwnershipStructureReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            consolidation_approach=inp.consolidation_approach,
            root_entity_name=inp.root_entity_name,
            total_entities=len(nodes),
            subsidiaries_count=len(subs),
            jv_count=len(jvs),
            associate_count=len(assocs),
            in_boundary_count=len(in_bound),
            out_boundary_count=len(out_bound),
            max_depth=max_depth,
            hierarchy=nodes,
            jv_details=jv_details,
            provenance_hash=prov,
        )
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return output

    # ------------------------------------------------------------------
    # CONVENIENCE RENDER METHODS
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(r)

    def render_html(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_html(r)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_json(r)

    # ------------------------------------------------------------------
    # EXPORT METHODS
    # ------------------------------------------------------------------

    def export_markdown(self, r: OwnershipStructureReportOutput) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        approach_label = r.consolidation_approach.replace("_", " ").title()
        lines.append(f"# Ownership Structure Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Approach:** {approach_label}")
        lines.append(f"**Root Entity:** {r.root_entity_name}")
        lines.append("")

        # Summary
        lines.append("## Structure Summary")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Total Entities | {r.total_entities} |")
        lines.append(f"| Subsidiaries | {r.subsidiaries_count} |")
        lines.append(f"| Joint Ventures | {r.jv_count} |")
        lines.append(f"| Associates | {r.associate_count} |")
        lines.append(f"| In Boundary | {r.in_boundary_count} |")
        lines.append(f"| Out of Boundary | {r.out_boundary_count} |")
        lines.append(f"| Max Hierarchy Depth | {r.max_depth} |")
        lines.append("")

        # Hierarchy tree
        lines.append("## Entity Hierarchy")
        lines.append("| Depth | Entity | Type | Direct % | Effective % | Control | In Boundary |")
        lines.append("|-------|--------|------|----------|-------------|---------|-------------|")
        for n in r.hierarchy:
            indent = "  " * n.depth
            boundary_status = "Yes" if n.in_boundary else "No"
            lines.append(
                f"| {n.depth} | {indent}{n.entity_name} | {n.entity_type} | "
                f"{n.direct_ownership_pct}% | {n.effective_ownership_pct}% | "
                f"{n.control_type} | {boundary_status} |"
            )
        lines.append("")

        # JV details
        if r.jv_details:
            lines.append("## Joint Venture Partner Details")
            lines.append("| JV Entity | Partner | Partner Ownership | Operator |")
            lines.append("|-----------|---------|-------------------|----------|")
            for jv in r.jv_details:
                is_op = "Yes" if jv.get("is_operator") else "No"
                lines.append(
                    f"| {jv.get('entity_name', '')} | {jv.get('partner_name', '')} | "
                    f"{jv.get('partner_ownership_pct', '0')}% | {is_op} |"
                )
            lines.append("")

        # Boundary overlay
        out_nodes = [n for n in r.hierarchy if not n.in_boundary]
        if out_nodes:
            lines.append("## Entities Excluded from Boundary")
            lines.append("| Entity | Type | Rationale |")
            lines.append("|--------|------|-----------|")
            for n in out_nodes:
                lines.append(f"| {n.entity_name} | {n.entity_type} | {n.boundary_rationale} |")
            lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: OwnershipStructureReportOutput) -> str:
        """Export report as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Ownership Structure - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: OwnershipStructureReportOutput) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: OwnershipStructureReportOutput) -> str:
        """Export hierarchy as CSV."""
        lines_out = [
            "entity_id,entity_name,entity_type,parent_entity_id,depth,"
            "direct_ownership_pct,effective_ownership_pct,control_type,in_boundary"
        ]
        for n in r.hierarchy:
            lines_out.append(
                f"{n.entity_id},{n.entity_name},{n.entity_type},{n.parent_entity_id},"
                f"{n.depth},{n.direct_ownership_pct},{n.effective_ownership_pct},"
                f"{n.control_type},{n.in_boundary}"
            )
        return "\n".join(lines_out)


__all__ = [
    "OwnershipStructureReport",
    "OwnershipStructureReportInput",
    "OwnershipStructureReportOutput",
    "HierarchyNode",
    "JvPartner",
]
