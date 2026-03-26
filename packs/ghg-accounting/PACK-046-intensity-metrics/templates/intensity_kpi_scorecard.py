# -*- coding: utf-8 -*-
"""
IntensityKPIScorecard - Traffic-Light KPI Scorecard for PACK-046.

Generates a KPI scorecard with traffic-light indicators for each
intensity metric against its target. Includes overall score summary,
individual metric cards, trend indicators, action items, and next
review date.

Sections:
    1. Scorecard Summary (overall score)
    2. Metric Cards (per metric: current, target, status, trend)
    3. Trend Indicators
    4. Action Items
    5. Next Review Date

Statuses:
    - ON_TARGET (green): Current value meets or exceeds target
    - AT_RISK (amber): Current value within warning threshold
    - OFF_TARGET (red): Current value exceeds off-target threshold
    - NO_TARGET (grey): No target set for this metric

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured scorecard data)

Author: GreenLang Team
Version: 46.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    MD = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"


class KPIStatus(str, Enum):
    """KPI traffic light status."""
    ON_TARGET = "on_target"
    AT_RISK = "at_risk"
    OFF_TARGET = "off_target"
    NO_TARGET = "no_target"


class TrendDirection(str, Enum):
    """Trend direction for metric."""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class MetricCard(BaseModel):
    """Single metric in the KPI scorecard."""
    metric_id: str = Field("", description="Unique metric identifier")
    metric_name: str = Field(..., description="Human-readable metric name")
    unit: str = Field("", description="Metric unit")
    current_value: float = Field(0.0, description="Current period value")
    target_value: Optional[float] = Field(None, description="Target value")
    threshold_amber: Optional[float] = Field(
        None, description="Amber/warning threshold"
    )
    threshold_red: Optional[float] = Field(
        None, description="Red/off-target threshold"
    )
    prior_value: Optional[float] = Field(None, description="Prior period value")
    status: KPIStatus = Field(KPIStatus.NO_TARGET, description="Traffic light status")
    trend: TrendDirection = Field(TrendDirection.UNKNOWN, description="Trend direction")
    trend_periods: int = Field(0, description="Number of periods in trend")
    notes: str = Field("", description="Metric-specific notes")
    lower_is_better: bool = Field(
        True, description="Whether lower values are better (typical for intensity)"
    )


class ScorecardActionItem(BaseModel):
    """Action item from scorecard review."""
    priority: int = Field(1, ge=1, le=5, description="Priority (1=highest)")
    metric_name: str = Field("", description="Related metric name")
    action: str = Field(..., description="Action description")
    owner: str = Field("", description="Responsible party")
    due_date: str = Field("", description="Due date")


class KPIScorecardInput(BaseModel):
    """Complete input model for IntensityKPIScorecard."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    overall_score: Optional[float] = Field(
        None, description="Overall scorecard score (0-100)"
    )
    overall_status: KPIStatus = Field(
        KPIStatus.AT_RISK, description="Overall scorecard status"
    )
    metric_cards: List[MetricCard] = Field(
        default_factory=list, description="Individual metric cards"
    )
    action_items: List[ScorecardActionItem] = Field(
        default_factory=list, description="Action items from review"
    )
    next_review_date: str = Field("", description="Next scheduled review date")
    reviewer: str = Field("", description="Reviewer name")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _status_label(status: KPIStatus) -> str:
    """Return human-readable label for KPI status."""
    mapping = {
        KPIStatus.ON_TARGET: "ON TARGET",
        KPIStatus.AT_RISK: "AT RISK",
        KPIStatus.OFF_TARGET: "OFF TARGET",
        KPIStatus.NO_TARGET: "NO TARGET",
    }
    return mapping.get(status, "UNKNOWN")


def _status_css(status: KPIStatus) -> str:
    """Return CSS class for KPI status."""
    mapping = {
        KPIStatus.ON_TARGET: "kpi-green",
        KPIStatus.AT_RISK: "kpi-amber",
        KPIStatus.OFF_TARGET: "kpi-red",
        KPIStatus.NO_TARGET: "kpi-grey",
    }
    return mapping.get(status, "kpi-grey")


def _status_color(status: KPIStatus) -> str:
    """Return hex colour for KPI status."""
    mapping = {
        KPIStatus.ON_TARGET: "#2a9d8f",
        KPIStatus.AT_RISK: "#e9c46a",
        KPIStatus.OFF_TARGET: "#e76f51",
        KPIStatus.NO_TARGET: "#888888",
    }
    return mapping.get(status, "#888888")


def _trend_label(trend: TrendDirection) -> str:
    """Return human-readable trend label."""
    mapping = {
        TrendDirection.IMPROVING: "Improving",
        TrendDirection.STABLE: "Stable",
        TrendDirection.DETERIORATING: "Deteriorating",
        TrendDirection.UNKNOWN: "Unknown",
    }
    return mapping.get(trend, "Unknown")


def _trend_arrow(trend: TrendDirection) -> str:
    """Return text arrow for trend direction."""
    mapping = {
        TrendDirection.IMPROVING: "v (improving)",
        TrendDirection.STABLE: "- (stable)",
        TrendDirection.DETERIORATING: "^ (deteriorating)",
        TrendDirection.UNKNOWN: "? (unknown)",
    }
    return mapping.get(trend, "?")


def _trend_html_arrow(trend: TrendDirection) -> str:
    """Return HTML arrow for trend direction."""
    mapping = {
        TrendDirection.IMPROVING: "&#9660; Improving",
        TrendDirection.STABLE: "&#9654; Stable",
        TrendDirection.DETERIORATING: "&#9650; Deteriorating",
        TrendDirection.UNKNOWN: "&#8211; Unknown",
    }
    return mapping.get(trend, "&#8211;")


def _trend_css(trend: TrendDirection) -> str:
    """Return CSS class for trend."""
    mapping = {
        TrendDirection.IMPROVING: "trend-improving",
        TrendDirection.STABLE: "trend-stable",
        TrendDirection.DETERIORATING: "trend-deteriorating",
        TrendDirection.UNKNOWN: "trend-unknown",
    }
    return mapping.get(trend, "trend-unknown")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class IntensityKPIScorecard:
    """
    Traffic-light KPI scorecard template.

    Renders a per-metric scorecard with traffic-light indicators
    (on-target / at-risk / off-target / no-target), trend direction,
    action items, and next review date. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = IntensityKPIScorecard()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IntensityKPIScorecard."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render KPI scorecard as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render KPI scorecard as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render KPI scorecard as JSON dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_summary(data),
            self._md_metric_cards(data),
            self._md_action_items(data),
            self._md_review_info(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        overall = KPIStatus(self._get_val(data, "overall_status", "at_risk"))
        return (
            f"# Intensity KPI Scorecard - {company}\n\n"
            f"**Period:** {period} | "
            f"**Overall Status:** {_status_label(overall)} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown scorecard summary."""
        overall_score = data.get("overall_score")
        cards = data.get("metric_cards", [])
        on_target = sum(1 for c in cards if c.get("status") == "on_target")
        at_risk = sum(1 for c in cards if c.get("status") == "at_risk")
        off_target = sum(1 for c in cards if c.get("status") == "off_target")
        no_target = sum(1 for c in cards if c.get("status") == "no_target")
        total = len(cards)
        lines = [
            "## 1. Scorecard Summary",
            "",
        ]
        if overall_score is not None:
            lines.append(f"**Overall Score:** {overall_score:.0f}/100")
            lines.append("")
        lines.extend([
            "| Status | Count | % |",
            "|--------|-------|---|",
            f"| ON TARGET (green) | {on_target} | {(on_target/total*100) if total else 0:.0f}% |",
            f"| AT RISK (amber) | {at_risk} | {(at_risk/total*100) if total else 0:.0f}% |",
            f"| OFF TARGET (red) | {off_target} | {(off_target/total*100) if total else 0:.0f}% |",
            f"| NO TARGET (grey) | {no_target} | {(no_target/total*100) if total else 0:.0f}% |",
            f"| **Total Metrics** | **{total}** | **100%** |",
        ])
        return "\n".join(lines)

    def _md_metric_cards(self, data: Dict[str, Any]) -> str:
        """Render Markdown metric cards table."""
        cards = data.get("metric_cards", [])
        if not cards:
            return "## 2. Metric Details\n\nNo metrics available."
        lines = [
            "## 2. Metric Details",
            "",
            "| Metric | Current | Target | Status | Trend | Notes |",
            "|--------|---------|--------|--------|-------|-------|",
        ]
        for c in cards:
            name = c.get("metric_name", "")
            unit = c.get("unit", "")
            current = c.get("current_value", 0)
            target = c.get("target_value")
            status = KPIStatus(c.get("status", "no_target"))
            trend = TrendDirection(c.get("trend", "unknown"))
            notes = c.get("notes", "")
            target_str = f"{target:,.4f}" if target is not None else "N/A"
            lines.append(
                f"| {name} ({unit}) | {current:,.4f} | {target_str} | "
                f"**{_status_label(status)}** | {_trend_arrow(trend)} | {notes} |"
            )
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render Markdown action items."""
        actions = data.get("action_items", [])
        if not actions:
            return ""
        lines = [
            "## 3. Action Items",
            "",
            "| Priority | Metric | Action | Owner | Due Date |",
            "|----------|--------|--------|-------|----------|",
        ]
        for a in actions:
            priority = a.get("priority", 1)
            metric = a.get("metric_name", "-")
            action = a.get("action", "")
            owner = a.get("owner", "-")
            due = a.get("due_date", "-")
            lines.append(f"| P{priority} | {metric} | {action} | {owner} | {due} |")
        return "\n".join(lines)

    def _md_review_info(self, data: Dict[str, Any]) -> str:
        """Render Markdown review information."""
        next_review = self._get_val(data, "next_review_date", "")
        reviewer = self._get_val(data, "reviewer", "")
        if not next_review and not reviewer:
            return ""
        lines = ["## 4. Review Information", ""]
        if next_review:
            lines.append(f"**Next Review Date:** {next_review}")
        if reviewer:
            lines.append(f"**Reviewer:** {reviewer}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_summary(data),
            self._html_metric_cards(data),
            self._html_action_items(data),
            self._html_review_info(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>KPI Scorecard - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".kpi-green{color:#2a9d8f;font-weight:700;}\n"
            ".kpi-amber{color:#e9c46a;font-weight:700;}\n"
            ".kpi-red{color:#e76f51;font-weight:700;}\n"
            ".kpi-grey{color:#888;font-weight:700;}\n"
            ".trend-improving{color:#2a9d8f;}\n"
            ".trend-stable{color:#888;}\n"
            ".trend-deteriorating{color:#e76f51;}\n"
            ".trend-unknown{color:#ccc;}\n"
            ".score-box{background:#f0f4f8;border-radius:12px;padding:2rem;"
            "text-align:center;margin:1rem 0;}\n"
            ".score-value{font-size:3rem;font-weight:700;}\n"
            ".score-label{font-size:1rem;color:#555;}\n"
            ".kpi-card{display:inline-block;background:#fff;border-radius:8px;"
            "border:1px solid #ddd;padding:1rem 1.5rem;margin:0.5rem;"
            "text-align:center;min-width:200px;vertical-align:top;}\n"
            ".kpi-card-value{font-size:1.3rem;font-weight:700;color:#1b263b;}\n"
            ".kpi-card-name{font-size:0.85rem;color:#555;margin-bottom:0.5rem;}\n"
            ".kpi-card-status{font-size:0.8rem;padding:2px 8px;border-radius:4px;"
            "display:inline-block;margin-top:0.3rem;}\n"
            ".status-on-target{background:#e8f5e9;color:#2a9d8f;}\n"
            ".status-at-risk{background:#fff8e1;color:#f9a825;}\n"
            ".status-off-target{background:#fbe9e7;color:#e76f51;}\n"
            ".status-no-target{background:#f5f5f5;color:#888;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        overall = KPIStatus(self._get_val(data, "overall_status", "at_risk"))
        css = _status_css(overall)
        return (
            '<div class="section">\n'
            f"<h1>Intensity KPI Scorecard &mdash; {company}</h1>\n"
            f"<p><strong>Period:</strong> {period} | "
            f'<strong>Overall:</strong> <span class="{css}">'
            f"{_status_label(overall)}</span></p>\n<hr>\n</div>"
        )

    def _html_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML scorecard summary with visual score."""
        overall_score = data.get("overall_score")
        overall_status = KPIStatus(self._get_val(data, "overall_status", "at_risk"))
        color = _status_color(overall_status)
        cards = data.get("metric_cards", [])
        total = len(cards)
        on_target = sum(1 for c in cards if c.get("status") == "on_target")
        at_risk = sum(1 for c in cards if c.get("status") == "at_risk")
        off_target = sum(1 for c in cards if c.get("status") == "off_target")
        score_html = ""
        if overall_score is not None:
            score_html = (
                f'<div class="score-box" style="border:3px solid {color};">\n'
                f'<div class="score-value" style="color:{color};">{overall_score:.0f}</div>\n'
                f'<div class="score-label">Overall Score (out of 100)</div>\n'
                "</div>\n"
            )
        summary_table = (
            "<table><thead><tr><th>Status</th><th>Count</th></tr></thead>\n<tbody>"
            f'<tr><td class="kpi-green">ON TARGET</td><td>{on_target}</td></tr>\n'
            f'<tr><td class="kpi-amber">AT RISK</td><td>{at_risk}</td></tr>\n'
            f'<tr><td class="kpi-red">OFF TARGET</td><td>{off_target}</td></tr>\n'
            f"<tr><td><strong>Total</strong></td><td><strong>{total}</strong></td></tr>\n"
            "</tbody></table>\n"
        )
        return (
            '<div class="section">\n<h2>1. Scorecard Summary</h2>\n'
            f"{score_html}{summary_table}</div>"
        )

    def _html_metric_cards(self, data: Dict[str, Any]) -> str:
        """Render HTML metric cards as visual cards."""
        cards = data.get("metric_cards", [])
        if not cards:
            return ""
        card_html = ""
        for c in cards:
            name = c.get("metric_name", "")
            unit = c.get("unit", "")
            current = c.get("current_value", 0)
            target = c.get("target_value")
            status = KPIStatus(c.get("status", "no_target"))
            trend = TrendDirection(c.get("trend", "unknown"))
            color = _status_color(status)
            status_css_class = {
                KPIStatus.ON_TARGET: "status-on-target",
                KPIStatus.AT_RISK: "status-at-risk",
                KPIStatus.OFF_TARGET: "status-off-target",
                KPIStatus.NO_TARGET: "status-no-target",
            }.get(status, "status-no-target")
            target_str = f"Target: {target:,.4f}" if target is not None else "No target set"
            t_css = _trend_css(trend)
            t_html = _trend_html_arrow(trend)
            card_html += (
                f'<div class="kpi-card" style="border-top:4px solid {color};">\n'
                f'<div class="kpi-card-name">{name} ({unit})</div>\n'
                f'<div class="kpi-card-value">{current:,.4f}</div>\n'
                f"<div>{target_str}</div>\n"
                f'<div class="kpi-card-status {status_css_class}">'
                f"{_status_label(status)}</div>\n"
                f'<div class="{t_css}" style="font-size:0.8rem;margin-top:0.3rem;">'
                f"{t_html}</div>\n"
                "</div>\n"
            )
        return f'<div class="section">\n<h2>2. Metric Details</h2>\n<div>{card_html}</div>\n</div>'

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Render HTML action items table."""
        actions = data.get("action_items", [])
        if not actions:
            return ""
        rows = ""
        for a in actions:
            priority = a.get("priority", 1)
            metric = a.get("metric_name", "-")
            action = a.get("action", "")
            owner = a.get("owner", "-")
            due = a.get("due_date", "-")
            rows += (
                f"<tr><td>P{priority}</td><td>{metric}</td><td>{action}</td>"
                f"<td>{owner}</td><td>{due}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Action Items</h2>\n'
            "<table><thead><tr><th>Priority</th><th>Metric</th>"
            "<th>Action</th><th>Owner</th><th>Due Date</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_review_info(self, data: Dict[str, Any]) -> str:
        """Render HTML review information."""
        next_review = self._get_val(data, "next_review_date", "")
        reviewer = self._get_val(data, "reviewer", "")
        if not next_review and not reviewer:
            return ""
        content = ""
        if next_review:
            content += f"<p><strong>Next Review Date:</strong> {next_review}</p>\n"
        if reviewer:
            content += f"<p><strong>Reviewer:</strong> {reviewer}</p>\n"
        return f'<div class="section">\n<h2>4. Review Information</h2>\n{content}</div>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render KPI scorecard as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        cards = data.get("metric_cards", [])
        total = len(cards)
        return {
            "template": "intensity_kpi_scorecard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "overall_score": data.get("overall_score"),
            "overall_status": self._get_val(data, "overall_status", "at_risk"),
            "summary": {
                "total_metrics": total,
                "on_target": sum(1 for c in cards if c.get("status") == "on_target"),
                "at_risk": sum(1 for c in cards if c.get("status") == "at_risk"),
                "off_target": sum(1 for c in cards if c.get("status") == "off_target"),
                "no_target": sum(1 for c in cards if c.get("status") == "no_target"),
            },
            "metric_cards": cards,
            "action_items": data.get("action_items", []),
            "next_review_date": self._get_val(data, "next_review_date", ""),
            "reviewer": self._get_val(data, "reviewer", ""),
        }
