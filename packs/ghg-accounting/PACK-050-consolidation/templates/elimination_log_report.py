# -*- coding: utf-8 -*-
"""
EliminationLogReport - Intercompany elimination details for PACK-050.

Provides the complete intercompany elimination log including the transfer
register (seller entity, buyer entity, energy type, quantity, emission factor,
emissions), elimination entries, matching verification status, and net impact
on the consolidated total.

Sections:
    1. Elimination Summary (total transfers, total eliminated, net impact)
    2. Transfer Register (seller, buyer, energy, quantity, EF, emissions)
    3. Elimination Entries (matched pairs with elimination amounts)
    4. Matching Verification (matched, unmatched, partial)
    5. Net Impact Analysis (pre-elimination vs post-elimination totals)
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

class TransferRecord(BaseModel):
    """Single intercompany energy/emission transfer."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    transfer_id: str = Field("")
    seller_entity: str = Field("")
    buyer_entity: str = Field("")
    energy_type: str = Field("", description="electricity, steam, heat, cooling, fuel")
    quantity: Decimal = Field(Decimal("0"))
    quantity_unit: str = Field("MWh")
    emission_factor: Decimal = Field(Decimal("0"))
    emission_factor_unit: str = Field("tCO2e/MWh")
    emissions_tco2e: Decimal = Field(Decimal("0"))
    period: str = Field("")
    verified: bool = Field(False)


class EliminationEntry(BaseModel):
    """A matched elimination entry."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    elimination_id: str = Field("")
    transfer_id: str = Field("")
    seller_entity: str = Field("")
    buyer_entity: str = Field("")
    scope_seller: str = Field("", description="scope_1, scope_2, scope_3")
    scope_buyer: str = Field("", description="scope_2, scope_3")
    eliminated_tco2e: Decimal = Field(Decimal("0"))
    match_status: str = Field("matched", description="matched, unmatched, partial")
    notes: str = Field("")


class MatchVerification(BaseModel):
    """Verification status for a transfer pair."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    transfer_id: str = Field("")
    seller_entity: str = Field("")
    buyer_entity: str = Field("")
    seller_reported_tco2e: Decimal = Field(Decimal("0"))
    buyer_reported_tco2e: Decimal = Field(Decimal("0"))
    variance_tco2e: Decimal = Field(Decimal("0"))
    variance_pct: Decimal = Field(Decimal("0"))
    status: str = Field("matched")


class EliminationLogReportInput(BaseModel):
    """Complete input for the elimination log report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    pre_elimination_tco2e: Decimal = Field(Decimal("0"))
    post_elimination_tco2e: Decimal = Field(Decimal("0"))
    transfers: List[Dict[str, Any]] = Field(default_factory=list)
    eliminations: List[Dict[str, Any]] = Field(default_factory=list)
    verifications: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class EliminationLogReportOutput(BaseModel):
    """Rendered elimination log report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    transfer_count: int = Field(0)
    elimination_count: int = Field(0)
    total_transferred_tco2e: Decimal = Field(Decimal("0"))
    total_eliminated_tco2e: Decimal = Field(Decimal("0"))
    pre_elimination_tco2e: Decimal = Field(Decimal("0"))
    post_elimination_tco2e: Decimal = Field(Decimal("0"))
    net_impact_tco2e: Decimal = Field(Decimal("0"))
    net_impact_pct: Decimal = Field(Decimal("0"))
    transfers: List[TransferRecord] = Field(default_factory=list)
    eliminations: List[EliminationEntry] = Field(default_factory=list)
    verifications: List[MatchVerification] = Field(default_factory=list)
    matched_count: int = Field(0)
    unmatched_count: int = Field(0)
    partial_count: int = Field(0)
    provenance_hash: str = Field("")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class EliminationLogReport:
    """
    Intercompany elimination log report template for PACK-050.

    Produces a complete elimination audit trail with transfer register,
    matched elimination entries, and verification status.

    Example:
        >>> tpl = EliminationLogReport()
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

    def render(self, data: Dict[str, Any]) -> EliminationLogReportOutput:
        """Render elimination log from input data."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = EliminationLogReportInput(**data) if isinstance(data, dict) else data

        transfers = [TransferRecord(**t) if isinstance(t, dict) else t for t in inp.transfers]
        elims = [EliminationEntry(**e) if isinstance(e, dict) else e for e in inp.eliminations]
        verifs = [MatchVerification(**v) if isinstance(v, dict) else v for v in inp.verifications]

        # Compute verification variances
        for v in verifs:
            if v.variance_tco2e == Decimal("0"):
                v.variance_tco2e = abs(v.seller_reported_tco2e - v.buyer_reported_tco2e)
            if v.variance_pct == Decimal("0") and v.seller_reported_tco2e > Decimal("0"):
                v.variance_pct = (
                    v.variance_tco2e / v.seller_reported_tco2e * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        total_transferred = sum(t.emissions_tco2e for t in transfers)
        total_eliminated = sum(e.eliminated_tco2e for e in elims)

        matched = sum(1 for v in verifs if v.status == "matched")
        unmatched = sum(1 for v in verifs if v.status == "unmatched")
        partial = sum(1 for v in verifs if v.status == "partial")

        pre = inp.pre_elimination_tco2e
        post = inp.post_elimination_tco2e
        net_impact = pre - post
        net_impact_pct = Decimal("0")
        if pre > Decimal("0"):
            net_impact_pct = (net_impact / pre * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = EliminationLogReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            transfer_count=len(transfers),
            elimination_count=len(elims),
            total_transferred_tco2e=total_transferred,
            total_eliminated_tco2e=total_eliminated,
            pre_elimination_tco2e=pre,
            post_elimination_tco2e=post,
            net_impact_tco2e=net_impact,
            net_impact_pct=net_impact_pct,
            transfers=transfers,
            eliminations=elims,
            verifications=verifs,
            matched_count=matched,
            unmatched_count=unmatched,
            partial_count=partial,
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

    def export_markdown(self, r: EliminationLogReportOutput) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        lines.append(f"# Elimination Log Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Transfers:** {r.transfer_count} | **Eliminations:** {r.elimination_count}")
        lines.append("")

        # Summary
        lines.append("## Elimination Summary")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Total Transferred | {r.total_transferred_tco2e:,.2f} tCO2e |")
        lines.append(f"| Total Eliminated | {r.total_eliminated_tco2e:,.2f} tCO2e |")
        lines.append(f"| Pre-Elimination Total | {r.pre_elimination_tco2e:,.2f} tCO2e |")
        lines.append(f"| Post-Elimination Total | {r.post_elimination_tco2e:,.2f} tCO2e |")
        lines.append(f"| Net Impact | {r.net_impact_tco2e:,.2f} tCO2e ({r.net_impact_pct}%) |")
        lines.append(f"| Matched | {r.matched_count} | Unmatched | {r.unmatched_count} | Partial | {r.partial_count} |")
        lines.append("")

        # Transfer register
        if r.transfers:
            lines.append("## Transfer Register")
            lines.append("| ID | Seller | Buyer | Energy Type | Qty | Unit | EF | Emissions tCO2e | Verified |")
            lines.append("|----|--------|-------|-------------|-----|------|----|-----------------|----------|")
            for t in r.transfers:
                v_status = "Yes" if t.verified else "No"
                lines.append(
                    f"| {t.transfer_id} | {t.seller_entity} | {t.buyer_entity} | "
                    f"{t.energy_type} | {t.quantity:,.2f} | {t.quantity_unit} | "
                    f"{t.emission_factor} | {t.emissions_tco2e:,.2f} | {v_status} |"
                )
            lines.append("")

        # Elimination entries
        if r.eliminations:
            lines.append("## Elimination Entries")
            lines.append("| ID | Seller | Buyer | Seller Scope | Buyer Scope | Eliminated tCO2e | Status |")
            lines.append("|----|--------|-------|-------------|-------------|------------------|--------|")
            for e in r.eliminations:
                lines.append(
                    f"| {e.elimination_id} | {e.seller_entity} | {e.buyer_entity} | "
                    f"{e.scope_seller} | {e.scope_buyer} | {e.eliminated_tco2e:,.2f} | {e.match_status} |"
                )
            lines.append("")

        # Matching verification
        if r.verifications:
            lines.append("## Matching Verification")
            lines.append("| Transfer | Seller | Buyer | Seller tCO2e | Buyer tCO2e | Variance | % | Status |")
            lines.append("|----------|--------|-------|-------------|-------------|----------|---|--------|")
            for v in r.verifications:
                lines.append(
                    f"| {v.transfer_id} | {v.seller_entity} | {v.buyer_entity} | "
                    f"{v.seller_reported_tco2e:,.2f} | {v.buyer_reported_tco2e:,.2f} | "
                    f"{v.variance_tco2e:,.2f} | {v.variance_pct}% | {v.status} |"
                )
            lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: EliminationLogReportOutput) -> str:
        """Export report as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Elimination Log - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: EliminationLogReportOutput) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: EliminationLogReportOutput) -> str:
        """Export transfer register as CSV."""
        lines_out = [
            "transfer_id,seller_entity,buyer_entity,energy_type,"
            "quantity,quantity_unit,emission_factor,emissions_tco2e,verified"
        ]
        for t in r.transfers:
            lines_out.append(
                f"{t.transfer_id},{t.seller_entity},{t.buyer_entity},{t.energy_type},"
                f"{t.quantity},{t.quantity_unit},{t.emission_factor},"
                f"{t.emissions_tco2e},{t.verified}"
            )
        return "\n".join(lines_out)


__all__ = [
    "EliminationLogReport",
    "EliminationLogReportInput",
    "EliminationLogReportOutput",
    "TransferRecord",
    "EliminationEntry",
    "MatchVerification",
]
