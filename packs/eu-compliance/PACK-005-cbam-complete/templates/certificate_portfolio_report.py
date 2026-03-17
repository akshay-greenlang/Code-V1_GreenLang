"""
CertificatePortfolioReport - CBAM certificate portfolio management template.

This module implements the certificate portfolio report for PACK-005 CBAM Complete.
It generates comprehensive reports covering certificate inventory, cost analysis,
expiry timelines, budget tracking, quarterly holding compliance, purchase/surrender
history, and ETS price trends.

Example:
    >>> template = CertificatePortfolioReport()
    >>> data = CertificatePortfolioData(
    ...     portfolio_summary=PortfolioSummary(total_purchased=1000, ...),
    ...     cost_analysis=CostAnalysis(weighted_avg_cost_eur=82.50, ...),
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class PortfolioSummary(BaseModel):
    """Summary of certificate inventory."""

    total_purchased: int = Field(0, ge=0, description="Total certificates purchased")
    total_held: int = Field(0, ge=0, description="Currently held certificates")
    total_surrendered: int = Field(0, ge=0, description="Total certificates surrendered")
    total_resold: int = Field(0, ge=0, description="Total certificates re-sold")
    total_cancelled: int = Field(0, ge=0, description="Total certificates cancelled")
    total_expired: int = Field(0, ge=0, description="Total certificates expired")
    current_balance: int = Field(0, ge=0, description="Current certificate balance")
    reporting_period: str = Field("", description="Reporting period label")


class CostAnalysis(BaseModel):
    """Certificate cost analysis data."""

    weighted_avg_cost_eur: float = Field(0.0, ge=0.0, description="Weighted average cost per certificate")
    mark_to_market_value_eur: float = Field(0.0, ge=0.0, description="Mark-to-market portfolio value")
    unrealized_pnl_eur: float = Field(0.0, description="Unrealized profit/loss")
    total_spend_ytd_eur: float = Field(0.0, ge=0.0, description="Total spend year-to-date")
    current_ets_price_eur: float = Field(0.0, ge=0.0, description="Current EU ETS price")
    avg_purchase_price_eur: float = Field(0.0, ge=0.0, description="Average purchase price")


class ExpiringCertificate(BaseModel):
    """Certificate expiry record."""

    certificate_id: str = Field("", description="Certificate identifier")
    quantity: int = Field(0, ge=0, description="Number of certificates")
    expiry_date: str = Field("", description="Expiry date ISO format")
    days_until_expiry: int = Field(0, ge=0, description="Days until expiry")
    recommended_action: str = Field("hold", description="Recommended action: surrender, extend, hold")


class ExpiryTimeline(BaseModel):
    """Certificate expiry timeline."""

    expiring_30_days: int = Field(0, ge=0, description="Certificates expiring in 30 days")
    expiring_60_days: int = Field(0, ge=0, description="Certificates expiring in 60 days")
    expiring_90_days: int = Field(0, ge=0, description="Certificates expiring in 90 days")
    expiring_180_days: int = Field(0, ge=0, description="Certificates expiring in 180 days")
    certificates: List[ExpiringCertificate] = Field(default_factory=list, description="Expiring certificate details")


class BudgetVsActual(BaseModel):
    """Budget versus actual certificate spending."""

    annual_budget_eur: float = Field(0.0, ge=0.0, description="Annual budget allocation")
    actual_spend_eur: float = Field(0.0, ge=0.0, description="Actual spend to date")
    variance_eur: float = Field(0.0, description="Variance (positive = under budget)")
    variance_pct: float = Field(0.0, description="Variance percentage")
    forecast_full_year_eur: float = Field(0.0, ge=0.0, description="Projected full year spend")
    months_elapsed: int = Field(0, ge=0, le=12, description="Months elapsed in fiscal year")


class QuarterlyHolding(BaseModel):
    """Quarterly holding compliance record."""

    quarter: str = Field("", description="Quarter label (e.g. Q1 2026)")
    required_certificates: int = Field(0, ge=0, description="Required by 50% threshold")
    actual_held: int = Field(0, ge=0, description="Actually held")
    threshold_pct: float = Field(50.0, description="Required threshold percentage")
    status: str = Field("PASS", description="PASS or FAIL")
    remediation_notes: str = Field("", description="Notes if FAIL")


class PurchaseRecord(BaseModel):
    """Certificate purchase record."""

    purchase_date: str = Field("", description="Purchase date ISO format")
    quantity: int = Field(0, ge=0, description="Number of certificates purchased")
    price_per_cert_eur: float = Field(0.0, ge=0.0, description="Price per certificate")
    total_cost_eur: float = Field(0.0, ge=0.0, description="Total cost")
    order_reference: str = Field("", description="Order reference number")
    counterparty: str = Field("", description="Counterparty or exchange")


class SurrenderRecord(BaseModel):
    """Certificate surrender record."""

    surrender_date: str = Field("", description="Surrender date ISO format")
    quantity: int = Field(0, ge=0, description="Number of certificates surrendered")
    declaration_id: str = Field("", description="Linked CBAM declaration ID")
    obligation_period: str = Field("", description="Obligation period covered")
    nca_reference: str = Field("", description="NCA reference number")


class PriceTrendEntry(BaseModel):
    """Weekly ETS price trend entry."""

    week_date: str = Field("", description="Week start date")
    closing_price_eur: float = Field(0.0, ge=0.0, description="Closing price")
    ma_4_week: Optional[float] = Field(None, description="4-week moving average")
    ma_12_week: Optional[float] = Field(None, description="12-week moving average")
    volume_traded: Optional[int] = Field(None, description="Volume traded")


class CertificatePortfolioData(BaseModel):
    """Complete input data for certificate portfolio report."""

    portfolio_summary: PortfolioSummary = Field(default_factory=PortfolioSummary)
    cost_analysis: CostAnalysis = Field(default_factory=CostAnalysis)
    expiry_timeline: ExpiryTimeline = Field(default_factory=ExpiryTimeline)
    budget_vs_actual: BudgetVsActual = Field(default_factory=BudgetVsActual)
    quarterly_holdings: List[QuarterlyHolding] = Field(default_factory=list)
    purchase_history: List[PurchaseRecord] = Field(default_factory=list)
    surrender_history: List[SurrenderRecord] = Field(default_factory=list)
    price_trend: List[PriceTrendEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class CertificatePortfolioReport:
    """
    CBAM certificate portfolio report template.

    Generates comprehensive portfolio reports covering certificate inventory
    management, cost analysis, expiry timelines, budget tracking, quarterly
    holding compliance checks, purchase/surrender history, and price trends.

    Attributes:
        config: Optional configuration dictionary.
        pack_id: Pack identifier (PACK-005).
        template_name: Template name for metadata.
        version: Template version.

    Example:
        >>> template = CertificatePortfolioReport()
        >>> md = template.render_markdown(data)
        >>> assert "Portfolio Summary" in md
    """

    PACK_ID = "PACK-005"
    TEMPLATE_NAME = "certificate_portfolio_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize CertificatePortfolioReport.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - currency (str): Currency code (default: EUR).
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the certificate portfolio report as Markdown.

        Args:
            data: Report data dictionary matching CertificatePortfolioData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header())
        sections.append(self._md_portfolio_summary(data))
        sections.append(self._md_cost_analysis(data))
        sections.append(self._md_expiry_timeline(data))
        sections.append(self._md_budget_vs_actual(data))
        sections.append(self._md_quarterly_holding(data))
        sections.append(self._md_purchase_history(data))
        sections.append(self._md_surrender_history(data))
        sections.append(self._md_price_trend(data))

        content = "\n\n".join(sections)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the certificate portfolio report as self-contained HTML.

        Args:
            data: Report data dictionary matching CertificatePortfolioData schema.

        Returns:
            Complete HTML document string with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_header())
        sections.append(self._html_portfolio_summary(data))
        sections.append(self._html_cost_analysis(data))
        sections.append(self._html_expiry_timeline(data))
        sections.append(self._html_budget_vs_actual(data))
        sections.append(self._html_quarterly_holding(data))
        sections.append(self._html_purchase_history(data))
        sections.append(self._html_surrender_history(data))
        sections.append(self._html_price_trend(data))

        body = "\n".join(sections)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="CBAM Certificate Portfolio Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the certificate portfolio report as structured JSON.

        Args:
            data: Report data dictionary matching CertificatePortfolioData schema.

        Returns:
            Dictionary with all report sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_certificate_portfolio",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "portfolio_summary": self._json_portfolio_summary(data),
            "cost_analysis": self._json_cost_analysis(data),
            "expiry_timeline": self._json_expiry_timeline(data),
            "budget_vs_actual": self._json_budget_vs_actual(data),
            "quarterly_holdings": self._json_quarterly_holding(data),
            "purchase_history": self._json_purchase_history(data),
            "surrender_history": self._json_surrender_history(data),
            "price_trend": self._json_price_trend(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self) -> str:
        """Build Markdown report header."""
        return (
            "# CBAM Certificate Portfolio Report\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_portfolio_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown portfolio summary section."""
        ps = data.get("portfolio_summary", {})
        period = ps.get("reporting_period", "N/A")

        return (
            "## 1. Portfolio Summary\n\n"
            f"**Reporting Period:** {period}\n\n"
            "| Metric | Count |\n"
            "|--------|-------|\n"
            f"| Total Purchased | {self._fmt_int(ps.get('total_purchased', 0))} |\n"
            f"| Currently Held | {self._fmt_int(ps.get('total_held', 0))} |\n"
            f"| Total Surrendered | {self._fmt_int(ps.get('total_surrendered', 0))} |\n"
            f"| Total Re-sold | {self._fmt_int(ps.get('total_resold', 0))} |\n"
            f"| Total Cancelled | {self._fmt_int(ps.get('total_cancelled', 0))} |\n"
            f"| Total Expired | {self._fmt_int(ps.get('total_expired', 0))} |\n"
            f"| **Current Balance** | **{self._fmt_int(ps.get('current_balance', 0))}** |"
        )

    def _md_cost_analysis(self, data: Dict[str, Any]) -> str:
        """Build Markdown cost analysis section."""
        ca = data.get("cost_analysis", {})
        cur = self._currency()

        pnl = ca.get("unrealized_pnl_eur", 0.0)
        pnl_sign = "+" if pnl >= 0 else ""

        return (
            "## 2. Cost Analysis\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Weighted Average Cost | {self._fmt_cur(ca.get('weighted_avg_cost_eur', 0.0), cur)}/cert |\n"
            f"| Mark-to-Market Value | {self._fmt_cur(ca.get('mark_to_market_value_eur', 0.0), cur)} |\n"
            f"| Unrealized P&L | {pnl_sign}{self._fmt_cur(pnl, cur)} |\n"
            f"| Total Spend YTD | {self._fmt_cur(ca.get('total_spend_ytd_eur', 0.0), cur)} |\n"
            f"| Current ETS Price | {self._fmt_cur(ca.get('current_ets_price_eur', 0.0), cur)}/tCO2e |\n"
            f"| Avg Purchase Price | {self._fmt_cur(ca.get('avg_purchase_price_eur', 0.0), cur)}/cert |"
        )

    def _md_expiry_timeline(self, data: Dict[str, Any]) -> str:
        """Build Markdown expiry timeline section."""
        et = data.get("expiry_timeline", {})
        certs = et.get("certificates", [])

        summary = (
            "## 3. Expiry Timeline\n\n"
            "### Expiry Summary\n\n"
            "| Window | Certificates Expiring |\n"
            "|--------|-----------------------|\n"
            f"| Within 30 days | {self._fmt_int(et.get('expiring_30_days', 0))} |\n"
            f"| Within 60 days | {self._fmt_int(et.get('expiring_60_days', 0))} |\n"
            f"| Within 90 days | {self._fmt_int(et.get('expiring_90_days', 0))} |\n"
            f"| Within 180 days | {self._fmt_int(et.get('expiring_180_days', 0))} |"
        )

        if not certs:
            return summary

        detail = (
            "\n\n### Expiry Details\n\n"
            "| Certificate ID | Quantity | Expiry Date | Days Left | Action |\n"
            "|----------------|----------|-------------|-----------|--------|\n"
        )
        rows: List[str] = []
        for c in certs:
            rows.append(
                f"| {c.get('certificate_id', '')} | "
                f"{self._fmt_int(c.get('quantity', 0))} | "
                f"{self._fmt_date(c.get('expiry_date', ''))} | "
                f"{c.get('days_until_expiry', 0)} | "
                f"{c.get('recommended_action', 'hold').upper()} |"
            )

        return summary + detail + "\n".join(rows)

    def _md_budget_vs_actual(self, data: Dict[str, Any]) -> str:
        """Build Markdown budget vs actual section."""
        bva = data.get("budget_vs_actual", {})
        cur = self._currency()

        variance = bva.get("variance_eur", 0.0)
        var_pct = bva.get("variance_pct", 0.0)
        var_sign = "+" if variance >= 0 else ""
        status = "UNDER BUDGET" if variance >= 0 else "OVER BUDGET"

        return (
            "## 4. Budget vs Actual\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Annual Budget | {self._fmt_cur(bva.get('annual_budget_eur', 0.0), cur)} |\n"
            f"| Actual Spend to Date | {self._fmt_cur(bva.get('actual_spend_eur', 0.0), cur)} |\n"
            f"| Variance | {var_sign}{self._fmt_cur(variance, cur)} ({var_sign}{var_pct:.1f}%) |\n"
            f"| Forecast Full Year | {self._fmt_cur(bva.get('forecast_full_year_eur', 0.0), cur)} |\n"
            f"| Months Elapsed | {bva.get('months_elapsed', 0)} / 12 |\n"
            f"| **Status** | **{status}** |"
        )

    def _md_quarterly_holding(self, data: Dict[str, Any]) -> str:
        """Build Markdown quarterly holding compliance section."""
        holdings = data.get("quarterly_holdings", [])

        header = (
            "## 5. Quarterly Holding Compliance\n\n"
            "50% threshold check per quarter per CBAM Regulation Article 22.\n\n"
            "| Quarter | Required (50%) | Actual Held | Status | Remediation |\n"
            "|---------|----------------|-------------|--------|-------------|\n"
        )

        rows: List[str] = []
        for h in holdings:
            status = h.get("status", "PASS")
            status_fmt = f"**{status}**"
            notes = h.get("remediation_notes", "") or "-"
            rows.append(
                f"| {h.get('quarter', '')} | "
                f"{self._fmt_int(h.get('required_certificates', 0))} | "
                f"{self._fmt_int(h.get('actual_held', 0))} | "
                f"{status_fmt} | "
                f"{notes} |"
            )

        return header + "\n".join(rows)

    def _md_purchase_history(self, data: Dict[str, Any]) -> str:
        """Build Markdown purchase history section."""
        purchases = data.get("purchase_history", [])
        cur = self._currency()

        header = (
            "## 6. Purchase History\n\n"
            "| Date | Quantity | Price/Cert | Total Cost | Order Ref | Counterparty |\n"
            "|------|----------|------------|------------|-----------|-------------|\n"
        )

        rows: List[str] = []
        for p in purchases:
            rows.append(
                f"| {self._fmt_date(p.get('purchase_date', ''))} | "
                f"{self._fmt_int(p.get('quantity', 0))} | "
                f"{self._fmt_cur(p.get('price_per_cert_eur', 0.0), cur)} | "
                f"{self._fmt_cur(p.get('total_cost_eur', 0.0), cur)} | "
                f"{p.get('order_reference', '')} | "
                f"{p.get('counterparty', '')} |"
            )

        if not rows:
            return header + "| *No purchases recorded* | | | | | |"

        return header + "\n".join(rows)

    def _md_surrender_history(self, data: Dict[str, Any]) -> str:
        """Build Markdown surrender history section."""
        surrenders = data.get("surrender_history", [])

        header = (
            "## 7. Surrender History\n\n"
            "| Date | Quantity | Declaration ID | Obligation Period | NCA Reference |\n"
            "|------|----------|----------------|-------------------|---------------|\n"
        )

        rows: List[str] = []
        for s in surrenders:
            rows.append(
                f"| {self._fmt_date(s.get('surrender_date', ''))} | "
                f"{self._fmt_int(s.get('quantity', 0))} | "
                f"{s.get('declaration_id', '')} | "
                f"{s.get('obligation_period', '')} | "
                f"{s.get('nca_reference', '')} |"
            )

        if not rows:
            return header + "| *No surrenders recorded* | | | | |"

        return header + "\n".join(rows)

    def _md_price_trend(self, data: Dict[str, Any]) -> str:
        """Build Markdown ETS price trend section with moving averages."""
        trend = data.get("price_trend", [])
        cur = self._currency()

        header = (
            "## 8. ETS Price Trend\n\n"
            "| Week | Closing Price | 4-Week MA | 12-Week MA | Volume |\n"
            "|------|---------------|-----------|------------|--------|\n"
        )

        rows: List[str] = []
        for entry in trend:
            ma4 = entry.get("ma_4_week")
            ma12 = entry.get("ma_12_week")
            vol = entry.get("volume_traded")

            ma4_str = self._fmt_cur(ma4, cur) if ma4 is not None else "-"
            ma12_str = self._fmt_cur(ma12, cur) if ma12 is not None else "-"
            vol_str = self._fmt_int(vol) if vol is not None else "-"

            rows.append(
                f"| {self._fmt_date(entry.get('week_date', ''))} | "
                f"{self._fmt_cur(entry.get('closing_price_eur', 0.0), cur)} | "
                f"{ma4_str} | "
                f"{ma12_str} | "
                f"{vol_str} |"
            )

        if not rows:
            return header + "| *No price data available* | | | | |"

        return header + "\n".join(rows)

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: {self.TEMPLATE_NAME} v{self.VERSION} | "
            f"Pack: {self.PACK_ID}*\n\n"
            f"*Provenance Hash: `{provenance_hash}`*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self) -> str:
        """Build HTML report header."""
        return (
            '<div class="report-header">'
            '<h1>CBAM Certificate Portfolio Report</h1>'
            f'<div class="meta-item">Pack: {self.PACK_ID} | '
            f'Template: {self.TEMPLATE_NAME} | Version: {self.VERSION}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
        )

    def _html_portfolio_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML portfolio summary section."""
        ps = data.get("portfolio_summary", {})
        period = ps.get("reporting_period", "N/A")

        metrics = [
            ("Total Purchased", ps.get("total_purchased", 0)),
            ("Currently Held", ps.get("total_held", 0)),
            ("Total Surrendered", ps.get("total_surrendered", 0)),
            ("Total Re-sold", ps.get("total_resold", 0)),
            ("Total Cancelled", ps.get("total_cancelled", 0)),
            ("Total Expired", ps.get("total_expired", 0)),
        ]

        cards = ""
        for label, val in metrics:
            cards += (
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{self._fmt_int(val)}</div>'
                f'</div>'
            )

        balance = ps.get("current_balance", 0)
        cards += (
            f'<div class="kpi-card" style="border:2px solid #1a5276">'
            f'<div class="kpi-label"><strong>Current Balance</strong></div>'
            f'<div class="kpi-value" style="color:#1a5276">{self._fmt_int(balance)}</div>'
            f'</div>'
        )

        return (
            '<div class="section">'
            f'<h2>1. Portfolio Summary</h2>'
            f'<p>Reporting Period: <strong>{period}</strong></p>'
            f'<div class="kpi-grid">{cards}</div>'
            '</div>'
        )

    def _html_cost_analysis(self, data: Dict[str, Any]) -> str:
        """Build HTML cost analysis section."""
        ca = data.get("cost_analysis", {})
        cur = self._currency()
        pnl = ca.get("unrealized_pnl_eur", 0.0)
        pnl_color = "#2ecc71" if pnl >= 0 else "#e74c3c"
        pnl_sign = "+" if pnl >= 0 else ""

        rows = (
            f'<tr><td>Weighted Average Cost</td>'
            f'<td class="num">{self._fmt_cur(ca.get("weighted_avg_cost_eur", 0.0), cur)}/cert</td></tr>'
            f'<tr><td>Mark-to-Market Value</td>'
            f'<td class="num">{self._fmt_cur(ca.get("mark_to_market_value_eur", 0.0), cur)}</td></tr>'
            f'<tr><td>Unrealized P&amp;L</td>'
            f'<td class="num" style="color:{pnl_color}">{pnl_sign}{self._fmt_cur(pnl, cur)}</td></tr>'
            f'<tr><td>Total Spend YTD</td>'
            f'<td class="num">{self._fmt_cur(ca.get("total_spend_ytd_eur", 0.0), cur)}</td></tr>'
            f'<tr><td>Current ETS Price</td>'
            f'<td class="num">{self._fmt_cur(ca.get("current_ets_price_eur", 0.0), cur)}/tCO2e</td></tr>'
            f'<tr><td>Avg Purchase Price</td>'
            f'<td class="num">{self._fmt_cur(ca.get("avg_purchase_price_eur", 0.0), cur)}/cert</td></tr>'
        )

        return (
            '<div class="section"><h2>2. Cost Analysis</h2>'
            '<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )

    def _html_expiry_timeline(self, data: Dict[str, Any]) -> str:
        """Build HTML expiry timeline section."""
        et = data.get("expiry_timeline", {})
        certs = et.get("certificates", [])

        windows = [
            ("30 days", et.get("expiring_30_days", 0), "#e74c3c"),
            ("60 days", et.get("expiring_60_days", 0), "#f39c12"),
            ("90 days", et.get("expiring_90_days", 0), "#f1c40f"),
            ("180 days", et.get("expiring_180_days", 0), "#2ecc71"),
        ]

        cards = ""
        for label, count, color in windows:
            cards += (
                f'<div class="kpi-card" style="border-top:3px solid {color}">'
                f'<div class="kpi-label">Within {label}</div>'
                f'<div class="kpi-value">{self._fmt_int(count)}</div>'
                f'</div>'
            )

        detail_rows = ""
        for c in certs:
            days = c.get("days_until_expiry", 0)
            color = "#e74c3c" if days <= 30 else "#f39c12" if days <= 60 else "#2ecc71"
            action = c.get("recommended_action", "hold").upper()
            detail_rows += (
                f'<tr><td>{c.get("certificate_id", "")}</td>'
                f'<td class="num">{self._fmt_int(c.get("quantity", 0))}</td>'
                f'<td>{self._fmt_date(c.get("expiry_date", ""))}</td>'
                f'<td class="num" style="color:{color}">{days}</td>'
                f'<td><strong>{action}</strong></td></tr>'
            )

        detail_table = ""
        if detail_rows:
            detail_table = (
                '<h3>Expiry Details</h3>'
                '<table><thead><tr>'
                '<th>Certificate ID</th><th>Quantity</th><th>Expiry Date</th>'
                '<th>Days Left</th><th>Action</th>'
                f'</tr></thead><tbody>{detail_rows}</tbody></table>'
            )

        return (
            '<div class="section"><h2>3. Expiry Timeline</h2>'
            f'<div class="kpi-grid">{cards}</div>'
            f'{detail_table}</div>'
        )

    def _html_budget_vs_actual(self, data: Dict[str, Any]) -> str:
        """Build HTML budget vs actual section."""
        bva = data.get("budget_vs_actual", {})
        cur = self._currency()
        variance = bva.get("variance_eur", 0.0)
        var_pct = bva.get("variance_pct", 0.0)
        budget = bva.get("annual_budget_eur", 0.0)
        actual = bva.get("actual_spend_eur", 0.0)
        months = bva.get("months_elapsed", 0)
        utilization_pct = (actual / budget * 100) if budget > 0 else 0.0
        color = "#2ecc71" if variance >= 0 else "#e74c3c"
        var_sign = "+" if variance >= 0 else ""
        status = "UNDER BUDGET" if variance >= 0 else "OVER BUDGET"

        bar_color = "#2ecc71" if utilization_pct <= 100 else "#e74c3c"
        bar_width = min(utilization_pct, 100)

        return (
            '<div class="section"><h2>4. Budget vs Actual</h2>'
            f'<div class="kpi-grid">'
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Annual Budget</div>'
            f'<div class="kpi-value">{self._fmt_cur(budget, cur)}</div></div>'
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Actual Spend</div>'
            f'<div class="kpi-value">{self._fmt_cur(actual, cur)}</div></div>'
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Variance</div>'
            f'<div class="kpi-value" style="color:{color}">'
            f'{var_sign}{self._fmt_cur(variance, cur)}</div>'
            f'<div class="kpi-unit">{var_sign}{var_pct:.1f}%</div></div>'
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Status ({months}/12 months)</div>'
            f'<div class="kpi-value" style="color:{color}">{status}</div></div>'
            f'</div>'
            f'<div style="margin-top:12px">'
            f'<div style="font-size:13px;color:#7f8c8d">Budget Utilization</div>'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{bar_width:.0f}%;background:{bar_color}"></div>'
            f'</div>'
            f'<div style="font-size:12px;color:#95a5a6">{utilization_pct:.1f}% utilized</div>'
            f'</div></div>'
        )

    def _html_quarterly_holding(self, data: Dict[str, Any]) -> str:
        """Build HTML quarterly holding compliance section."""
        holdings = data.get("quarterly_holdings", [])

        rows_html = ""
        for h in holdings:
            status = h.get("status", "PASS")
            color = "#2ecc71" if status == "PASS" else "#e74c3c"
            notes = h.get("remediation_notes", "") or "-"
            rows_html += (
                f'<tr><td>{h.get("quarter", "")}</td>'
                f'<td class="num">{self._fmt_int(h.get("required_certificates", 0))}</td>'
                f'<td class="num">{self._fmt_int(h.get("actual_held", 0))}</td>'
                f'<td style="color:{color};font-weight:bold">{status}</td>'
                f'<td>{notes}</td></tr>'
            )

        return (
            '<div class="section"><h2>5. Quarterly Holding Compliance</h2>'
            '<p>50% threshold check per quarter per CBAM Regulation Article 22.</p>'
            '<table><thead><tr>'
            '<th>Quarter</th><th>Required (50%)</th><th>Actual Held</th>'
            '<th>Status</th><th>Remediation</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_purchase_history(self, data: Dict[str, Any]) -> str:
        """Build HTML purchase history section."""
        purchases = data.get("purchase_history", [])
        cur = self._currency()

        rows_html = ""
        for p in purchases:
            rows_html += (
                f'<tr><td>{self._fmt_date(p.get("purchase_date", ""))}</td>'
                f'<td class="num">{self._fmt_int(p.get("quantity", 0))}</td>'
                f'<td class="num">{self._fmt_cur(p.get("price_per_cert_eur", 0.0), cur)}</td>'
                f'<td class="num">{self._fmt_cur(p.get("total_cost_eur", 0.0), cur)}</td>'
                f'<td>{p.get("order_reference", "")}</td>'
                f'<td>{p.get("counterparty", "")}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="6"><em>No purchases recorded</em></td></tr>'

        return (
            '<div class="section"><h2>6. Purchase History</h2>'
            '<table><thead><tr>'
            '<th>Date</th><th>Quantity</th><th>Price/Cert</th>'
            '<th>Total Cost</th><th>Order Ref</th><th>Counterparty</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_surrender_history(self, data: Dict[str, Any]) -> str:
        """Build HTML surrender history section."""
        surrenders = data.get("surrender_history", [])

        rows_html = ""
        for s in surrenders:
            rows_html += (
                f'<tr><td>{self._fmt_date(s.get("surrender_date", ""))}</td>'
                f'<td class="num">{self._fmt_int(s.get("quantity", 0))}</td>'
                f'<td>{s.get("declaration_id", "")}</td>'
                f'<td>{s.get("obligation_period", "")}</td>'
                f'<td>{s.get("nca_reference", "")}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="5"><em>No surrenders recorded</em></td></tr>'

        return (
            '<div class="section"><h2>7. Surrender History</h2>'
            '<table><thead><tr>'
            '<th>Date</th><th>Quantity</th><th>Declaration ID</th>'
            '<th>Obligation Period</th><th>NCA Reference</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_price_trend(self, data: Dict[str, Any]) -> str:
        """Build HTML ETS price trend section."""
        trend = data.get("price_trend", [])
        cur = self._currency()

        rows_html = ""
        for entry in trend:
            ma4 = entry.get("ma_4_week")
            ma12 = entry.get("ma_12_week")
            vol = entry.get("volume_traded")

            ma4_str = self._fmt_cur(ma4, cur) if ma4 is not None else "-"
            ma12_str = self._fmt_cur(ma12, cur) if ma12 is not None else "-"
            vol_str = self._fmt_int(vol) if vol is not None else "-"

            rows_html += (
                f'<tr><td>{self._fmt_date(entry.get("week_date", ""))}</td>'
                f'<td class="num">{self._fmt_cur(entry.get("closing_price_eur", 0.0), cur)}</td>'
                f'<td class="num">{ma4_str}</td>'
                f'<td class="num">{ma12_str}</td>'
                f'<td class="num">{vol_str}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="5"><em>No price data available</em></td></tr>'

        return (
            '<div class="section"><h2>8. ETS Price Trend</h2>'
            '<table><thead><tr>'
            '<th>Week</th><th>Closing Price</th><th>4-Week MA</th>'
            '<th>12-Week MA</th><th>Volume</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_portfolio_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON portfolio summary."""
        ps = data.get("portfolio_summary", {})
        return {
            "reporting_period": ps.get("reporting_period", ""),
            "total_purchased": ps.get("total_purchased", 0),
            "total_held": ps.get("total_held", 0),
            "total_surrendered": ps.get("total_surrendered", 0),
            "total_resold": ps.get("total_resold", 0),
            "total_cancelled": ps.get("total_cancelled", 0),
            "total_expired": ps.get("total_expired", 0),
            "current_balance": ps.get("current_balance", 0),
        }

    def _json_cost_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON cost analysis."""
        ca = data.get("cost_analysis", {})
        return {
            "weighted_avg_cost_eur": round(ca.get("weighted_avg_cost_eur", 0.0), 2),
            "mark_to_market_value_eur": round(ca.get("mark_to_market_value_eur", 0.0), 2),
            "unrealized_pnl_eur": round(ca.get("unrealized_pnl_eur", 0.0), 2),
            "total_spend_ytd_eur": round(ca.get("total_spend_ytd_eur", 0.0), 2),
            "current_ets_price_eur": round(ca.get("current_ets_price_eur", 0.0), 2),
            "avg_purchase_price_eur": round(ca.get("avg_purchase_price_eur", 0.0), 2),
        }

    def _json_expiry_timeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON expiry timeline."""
        et = data.get("expiry_timeline", {})
        return {
            "expiring_30_days": et.get("expiring_30_days", 0),
            "expiring_60_days": et.get("expiring_60_days", 0),
            "expiring_90_days": et.get("expiring_90_days", 0),
            "expiring_180_days": et.get("expiring_180_days", 0),
            "certificates": [
                {
                    "certificate_id": c.get("certificate_id", ""),
                    "quantity": c.get("quantity", 0),
                    "expiry_date": c.get("expiry_date", ""),
                    "days_until_expiry": c.get("days_until_expiry", 0),
                    "recommended_action": c.get("recommended_action", "hold"),
                }
                for c in et.get("certificates", [])
            ],
        }

    def _json_budget_vs_actual(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON budget vs actual."""
        bva = data.get("budget_vs_actual", {})
        variance = bva.get("variance_eur", 0.0)
        return {
            "annual_budget_eur": round(bva.get("annual_budget_eur", 0.0), 2),
            "actual_spend_eur": round(bva.get("actual_spend_eur", 0.0), 2),
            "variance_eur": round(variance, 2),
            "variance_pct": round(bva.get("variance_pct", 0.0), 2),
            "forecast_full_year_eur": round(bva.get("forecast_full_year_eur", 0.0), 2),
            "months_elapsed": bva.get("months_elapsed", 0),
            "status": "UNDER_BUDGET" if variance >= 0 else "OVER_BUDGET",
        }

    def _json_quarterly_holding(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON quarterly holding compliance."""
        return [
            {
                "quarter": h.get("quarter", ""),
                "required_certificates": h.get("required_certificates", 0),
                "actual_held": h.get("actual_held", 0),
                "threshold_pct": h.get("threshold_pct", 50.0),
                "status": h.get("status", "PASS"),
                "remediation_notes": h.get("remediation_notes", ""),
            }
            for h in data.get("quarterly_holdings", [])
        ]

    def _json_purchase_history(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON purchase history."""
        return [
            {
                "purchase_date": p.get("purchase_date", ""),
                "quantity": p.get("quantity", 0),
                "price_per_cert_eur": round(p.get("price_per_cert_eur", 0.0), 2),
                "total_cost_eur": round(p.get("total_cost_eur", 0.0), 2),
                "order_reference": p.get("order_reference", ""),
                "counterparty": p.get("counterparty", ""),
            }
            for p in data.get("purchase_history", [])
        ]

    def _json_surrender_history(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON surrender history."""
        return [
            {
                "surrender_date": s.get("surrender_date", ""),
                "quantity": s.get("quantity", 0),
                "declaration_id": s.get("declaration_id", ""),
                "obligation_period": s.get("obligation_period", ""),
                "nca_reference": s.get("nca_reference", ""),
            }
            for s in data.get("surrender_history", [])
        ]

    def _json_price_trend(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON price trend."""
        return [
            {
                "week_date": e.get("week_date", ""),
                "closing_price_eur": round(e.get("closing_price_eur", 0.0), 2),
                "ma_4_week": round(e["ma_4_week"], 2) if e.get("ma_4_week") is not None else None,
                "ma_12_week": round(e["ma_12_week"], 2) if e.get("ma_12_week") is not None else None,
                "volume_traded": e.get("volume_traded"),
            }
            for e in data.get("price_trend", [])
        ]

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _compute_provenance_hash(self, content: str) -> str:
        """Compute SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _currency(self) -> str:
        """Get configured currency code."""
        return self.config.get("currency", "EUR")

    def _fmt_int(self, value: Union[int, float, None]) -> str:
        """Format integer with thousand separators."""
        if value is None:
            return "0"
        return f"{int(value):,}"

    def _fmt_num(self, value: Union[int, float], decimals: int = 2) -> str:
        """Format number with thousand separators and fixed decimals."""
        return f"{value:,.{decimals}f}"

    def _fmt_cur(self, value: Union[int, float], currency: str = "EUR") -> str:
        """Format currency value."""
        return f"{currency} {value:,.2f}"

    def _fmt_pct(self, value: Union[int, float]) -> str:
        """Format percentage value."""
        return f"{value:.1f}%"

    def _fmt_date(self, dt: Union[datetime, str]) -> str:
        """Format datetime to ISO date string."""
        if isinstance(dt, str):
            return dt[:10] if dt else ""
        return dt.strftime("%Y-%m-%d")

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = self._get_css()
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f'Pack: {self.PACK_ID} | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )

    def _get_css(self) -> str:
        """Return inline CSS for HTML reports."""
        return (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 8px 0;font-size:24px}"
            ".meta-item{font-size:13px;opacity:0.8}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin-bottom:16px}"
            ".kpi-card{background:#fff;padding:20px;border-radius:8px;text-align:center;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".kpi-label{font-size:13px;color:#7f8c8d;margin-bottom:4px}"
            ".kpi-value{font-size:28px;font-weight:700;color:#1a5276}"
            ".kpi-unit{font-size:12px;color:#95a5a6;margin-top:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            ".section h3{margin:16px 0 8px 0;font-size:15px;color:#2c3e50}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;margin:4px 0}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
