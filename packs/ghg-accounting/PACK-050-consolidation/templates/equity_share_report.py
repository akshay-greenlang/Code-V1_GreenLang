# -*- coding: utf-8 -*-
"""
EquityShareReport - Equity share approach details for PACK-050.

Provides per-entity equity percentage, raw emissions, equity-adjusted emissions,
multi-tier ownership chain visualization, JV split details, associate inclusion
logic, and partner reconciliation (sum of all partner shares = 100%).

Sections:
    1. Equity Summary (total raw, total adjusted, adjustment impact)
    2. Entity Equity Table (entity, equity %, raw, adjusted, delta)
    3. Multi-Tier Ownership Chains (intermediate entity chain)
    4. JV Splits (JV entity, partners, equity %, allocated emissions)
    5. Associate Inclusion (associates above threshold)
    6. Partner Reconciliation (partner share sums = 100%)
    7. Provenance Footer

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

class EntityEquityLine(BaseModel):
    """Equity share details for a single entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_id: str = Field("")
    entity_name: str = Field("")
    entity_type: str = Field("")
    equity_pct: Decimal = Field(Decimal("100"))
    raw_tco2e: Decimal = Field(Decimal("0"))
    adjusted_tco2e: Decimal = Field(Decimal("0"))
    delta_tco2e: Decimal = Field(Decimal("0"))


class OwnershipChainStep(BaseModel):
    """One step in a multi-tier ownership chain."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tier: int = Field(0)
    entity_name: str = Field("")
    ownership_pct: Decimal = Field(Decimal("100"))
    cumulative_pct: Decimal = Field(Decimal("100"))


class OwnershipChain(BaseModel):
    """Complete multi-tier ownership chain for one entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_entity: str = Field("")
    effective_equity_pct: Decimal = Field(Decimal("100"))
    chain: List[OwnershipChainStep] = Field(default_factory=list)


class JvSplit(BaseModel):
    """JV allocation split details."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    jv_entity_name: str = Field("")
    jv_total_tco2e: Decimal = Field(Decimal("0"))
    partner_name: str = Field("")
    partner_equity_pct: Decimal = Field(Decimal("0"))
    partner_allocated_tco2e: Decimal = Field(Decimal("0"))


class PartnerReconciliation(BaseModel):
    """Reconciliation that partner shares sum to 100%."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_name: str = Field("")
    total_partner_pct: Decimal = Field(Decimal("0"))
    is_reconciled: bool = Field(True)
    partner_count: int = Field(0)
    variance_pct: Decimal = Field(Decimal("0"))


class EquityShareReportInput(BaseModel):
    """Complete input for the equity share report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    total_raw_tco2e: Decimal = Field(Decimal("0"))
    total_adjusted_tco2e: Decimal = Field(Decimal("0"))
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    ownership_chains: List[Dict[str, Any]] = Field(default_factory=list)
    jv_splits: List[Dict[str, Any]] = Field(default_factory=list)
    reconciliations: List[Dict[str, Any]] = Field(default_factory=list)
    associate_threshold_pct: Decimal = Field(Decimal("20"))


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class EquityShareReportOutput(BaseModel):
    """Rendered equity share report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    total_raw_tco2e: Decimal = Field(Decimal("0"))
    total_adjusted_tco2e: Decimal = Field(Decimal("0"))
    adjustment_impact_tco2e: Decimal = Field(Decimal("0"))
    adjustment_impact_pct: Decimal = Field(Decimal("0"))
    entity_count: int = Field(0)
    entities: List[EntityEquityLine] = Field(default_factory=list)
    ownership_chains: List[OwnershipChain] = Field(default_factory=list)
    jv_splits: List[JvSplit] = Field(default_factory=list)
    associates_included: List[EntityEquityLine] = Field(default_factory=list)
    reconciliations: List[PartnerReconciliation] = Field(default_factory=list)
    all_reconciled: bool = Field(True)
    associate_threshold_pct: Decimal = Field(Decimal("20"))
    provenance_hash: str = Field("")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class EquityShareReport:
    """
    Equity share approach detail report template for PACK-050.

    Produces a detailed equity share analysis with multi-tier chains,
    JV split details, and partner reconciliation checks.

    Example:
        >>> tpl = EquityShareReport()
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

    def render(self, data: Dict[str, Any]) -> EquityShareReportOutput:
        """Render equity share report from input data."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = EquityShareReportInput(**data) if isinstance(data, dict) else data

        entities = [EntityEquityLine(**e) if isinstance(e, dict) else e for e in inp.entities]
        for e in entities:
            if e.delta_tco2e == Decimal("0"):
                e.delta_tco2e = e.adjusted_tco2e - e.raw_tco2e

        chains_raw = inp.ownership_chains
        chains: List[OwnershipChain] = []
        for c in chains_raw:
            if isinstance(c, dict):
                chain_steps = [
                    OwnershipChainStep(**s) if isinstance(s, dict) else s
                    for s in c.get("chain", [])
                ]
                chains.append(OwnershipChain(
                    target_entity=c.get("target_entity", ""),
                    effective_equity_pct=Decimal(str(c.get("effective_equity_pct", "100"))),
                    chain=chain_steps,
                ))
            else:
                chains.append(c)

        jv_splits = [JvSplit(**j) if isinstance(j, dict) else j for j in inp.jv_splits]
        recons = [PartnerReconciliation(**r) if isinstance(r, dict) else r for r in inp.reconciliations]

        # Associates: entities with equity between threshold and 50%
        threshold = inp.associate_threshold_pct
        associates = [e for e in entities if e.entity_type == "associate" and e.equity_pct >= threshold]

        # Check reconciliation
        for rc in recons:
            rc.variance_pct = (rc.total_partner_pct - Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            rc.is_reconciled = abs(rc.variance_pct) <= Decimal("0.01")
        all_reconciled = all(rc.is_reconciled for rc in recons) if recons else True

        # Impact
        raw_total = inp.total_raw_tco2e or sum(e.raw_tco2e for e in entities)
        adj_total = inp.total_adjusted_tco2e or sum(e.adjusted_tco2e for e in entities)
        impact = adj_total - raw_total
        impact_pct = Decimal("0")
        if raw_total > Decimal("0"):
            impact_pct = (impact / raw_total * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = EquityShareReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            total_raw_tco2e=raw_total,
            total_adjusted_tco2e=adj_total,
            adjustment_impact_tco2e=impact,
            adjustment_impact_pct=impact_pct,
            entity_count=len(entities),
            entities=entities,
            ownership_chains=chains,
            jv_splits=jv_splits,
            associates_included=associates,
            reconciliations=recons,
            all_reconciled=all_reconciled,
            associate_threshold_pct=threshold,
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

    def export_markdown(self, r: EquityShareReportOutput) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        lines.append(f"# Equity Share Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Entities:** {r.entity_count}")
        lines.append("")

        # Summary
        lines.append("## Equity Adjustment Summary")
        lines.append("| Metric | tCO2e |")
        lines.append("|--------|------:|")
        lines.append(f"| Total Raw Emissions | {r.total_raw_tco2e:,.2f} |")
        lines.append(f"| Total Equity-Adjusted | {r.total_adjusted_tco2e:,.2f} |")
        lines.append(f"| Adjustment Impact | {r.adjustment_impact_tco2e:,.2f} ({r.adjustment_impact_pct}%) |")
        lines.append("")

        # Entity table
        lines.append("## Entity Equity Details")
        lines.append("| Entity | Type | Equity % | Raw tCO2e | Adjusted tCO2e | Delta |")
        lines.append("|--------|------|----------|-----------|----------------|-------|")
        for e in r.entities:
            lines.append(
                f"| {e.entity_name} | {e.entity_type} | {e.equity_pct}% | "
                f"{e.raw_tco2e:,.0f} | {e.adjusted_tco2e:,.0f} | {e.delta_tco2e:,.0f} |"
            )
        lines.append("")

        # Ownership chains
        if r.ownership_chains:
            lines.append("## Multi-Tier Ownership Chains")
            for chain in r.ownership_chains:
                lines.append(f"\n### {chain.target_entity} (Effective: {chain.effective_equity_pct}%)")
                lines.append("| Tier | Entity | Ownership % | Cumulative % |")
                lines.append("|------|--------|-------------|--------------|")
                for step in chain.chain:
                    lines.append(
                        f"| {step.tier} | {step.entity_name} | "
                        f"{step.ownership_pct}% | {step.cumulative_pct}% |"
                    )
            lines.append("")

        # JV splits
        if r.jv_splits:
            lines.append("## Joint Venture Splits")
            lines.append("| JV Entity | JV Total | Partner | Equity % | Allocated tCO2e |")
            lines.append("|-----------|----------|---------|----------|-----------------|")
            for j in r.jv_splits:
                lines.append(
                    f"| {j.jv_entity_name} | {j.jv_total_tco2e:,.0f} | "
                    f"{j.partner_name} | {j.partner_equity_pct}% | "
                    f"{j.partner_allocated_tco2e:,.0f} |"
                )
            lines.append("")

        # Associates
        if r.associates_included:
            lines.append(f"## Associates Included (threshold >= {r.associate_threshold_pct}%)")
            lines.append("| Associate | Equity % | Adjusted tCO2e |")
            lines.append("|-----------|----------|----------------|")
            for a in r.associates_included:
                lines.append(f"| {a.entity_name} | {a.equity_pct}% | {a.adjusted_tco2e:,.0f} |")
            lines.append("")

        # Reconciliation
        if r.reconciliations:
            status = "PASS" if r.all_reconciled else "FAIL"
            lines.append(f"## Partner Reconciliation (Overall: **{status}**)")
            lines.append("| Entity | Partners | Total Partner % | Variance | Status |")
            lines.append("|--------|----------|-----------------|----------|--------|")
            for rc in r.reconciliations:
                st = "PASS" if rc.is_reconciled else "FAIL"
                lines.append(
                    f"| {rc.entity_name} | {rc.partner_count} | "
                    f"{rc.total_partner_pct}% | {rc.variance_pct}% | {st} |"
                )
            lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: EquityShareReportOutput) -> str:
        """Export report as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Equity Share Report - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: EquityShareReportOutput) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: EquityShareReportOutput) -> str:
        """Export equity entity data as CSV."""
        lines_out = [
            "entity_id,entity_name,entity_type,equity_pct,"
            "raw_tco2e,adjusted_tco2e,delta_tco2e"
        ]
        for e in r.entities:
            lines_out.append(
                f"{e.entity_id},{e.entity_name},{e.entity_type},{e.equity_pct},"
                f"{e.raw_tco2e},{e.adjusted_tco2e},{e.delta_tco2e}"
            )
        return "\n".join(lines_out)


__all__ = [
    "EquityShareReport",
    "EquityShareReportInput",
    "EquityShareReportOutput",
    "EntityEquityLine",
    "OwnershipChain",
    "OwnershipChainStep",
    "JvSplit",
    "PartnerReconciliation",
]
