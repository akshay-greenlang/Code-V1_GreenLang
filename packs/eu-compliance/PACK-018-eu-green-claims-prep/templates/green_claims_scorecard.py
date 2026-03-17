# -*- coding: utf-8 -*-
"""
GreenClaimsScorecardTemplate - EU Green Claims Executive Scorecard

Renders an executive-level scorecard summarising overall Green Claims Directive
readiness with traffic-light indicators for each compliance dimension. Covers
substantiation rate, evidence completeness, greenwashing risk, label compliance,
benchmark comparison, and trend analysis in a concise dashboard format.

Sections:
    1. Score Overview - Composite score with traffic-light status
    2. Substantiation Rate - Claim substantiation percentage
    3. Evidence Completeness - Evidence coverage assessment
    4. Greenwashing Risk - Aggregated greenwashing risk indicator
    5. Label Compliance - Eco-label compliance summary
    6. Benchmark Comparison - Peer and sector comparison
    7. Trend Analysis - Score progression across periods
    8. Provenance - Data lineage and hash chain

PACK Reference: PACK-018 EU Green Claims Prep Pack
Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["GreenClaimsScorecardTemplate"]

_SECTIONS: List[Dict[str, Any]] = [
    {"id": "score_overview", "title": "Score Overview", "order": 1},
    {"id": "substantiation_rate", "title": "Substantiation Rate", "order": 2},
    {"id": "evidence_completeness", "title": "Evidence Completeness", "order": 3},
    {"id": "greenwashing_risk", "title": "Greenwashing Risk", "order": 4},
    {"id": "label_compliance", "title": "Label Compliance", "order": 5},
    {"id": "benchmark_comparison", "title": "Benchmark Comparison", "order": 6},
    {"id": "trend_analysis", "title": "Trend Analysis", "order": 7},
    {"id": "provenance", "title": "Provenance", "order": 8},
]

_TRAFFIC_LIGHT_THRESHOLDS = {
    "green": 80.0,
    "amber": 50.0,
}


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _traffic_light(score: float) -> str:
    """Return traffic-light indicator for a score (0-100)."""
    if score >= _TRAFFIC_LIGHT_THRESHOLDS["green"]:
        return "GREEN"
    elif score >= _TRAFFIC_LIGHT_THRESHOLDS["amber"]:
        return "AMBER"
    else:
        return "RED"


class GreenClaimsScorecardTemplate:
    """
    EU Green Claims Directive - Executive Scorecard.

    Renders a concise executive dashboard with traffic-light indicators
    for substantiation rate, evidence completeness, greenwashing risk,
    label compliance, and benchmark comparison. Tracks score progression
    over reporting periods and produces a composite readiness score.

    Example:
        >>> tpl = GreenClaimsScorecardTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GreenClaimsScorecardTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render scorecard as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_score_overview(data),
            self._md_substantiation_rate(data),
            self._md_evidence_completeness(data),
            self._md_greenwashing_risk(data),
            self._md_label_compliance(data),
            self._md_benchmark_comparison(data),
            self._md_trend_analysis(data),
            self._md_provenance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render scorecard as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_score_overview(data),
            self._html_dimension_cards(data),
            self._html_benchmark(data),
            self._html_trend(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Green Claims Scorecard - EU Green Claims</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render scorecard as structured JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "green_claims_scorecard",
            "directive_reference": "EU Green Claims Directive 2023/0085",
            "version": "18.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "score_overview": self._section_score_overview(data),
            "substantiation_rate": self._section_substantiation_rate(data),
            "evidence_completeness": self._section_evidence_completeness(data),
            "greenwashing_risk": self._section_greenwashing_risk(data),
            "label_compliance": self._section_label_compliance(data),
            "benchmark_comparison": self._section_benchmark(data),
            "trend_analysis": self._section_trend_analysis(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def get_sections(self) -> List[Dict[str, Any]]:
        """Return list of available section definitions."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("scores"):
            errors.append("scores dict is required with dimension scores")
        if not data.get("reporting_period"):
            warnings.append("reporting_period missing; will default to empty")
        if not data.get("benchmark"):
            warnings.append("benchmark data missing; comparison will be limited")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section builders (dict)
    # ------------------------------------------------------------------

    def _section_score_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build score overview section."""
        scores = data.get("scores", {})
        dimensions = self._build_dimensions(scores)
        composite = self._composite_score(dimensions)
        return {
            "title": "Score Overview",
            "composite_score": composite,
            "traffic_light": _traffic_light(composite),
            "dimensions": dimensions,
            "assessment_date": data.get(
                "assessment_date", _utcnow().isoformat()
            ),
        }

    def _section_substantiation_rate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build substantiation rate section."""
        scores = data.get("scores", {})
        sub = scores.get("substantiation", {})
        total = sub.get("total_claims", 0)
        substantiated = sub.get("substantiated_claims", 0)
        rate = round(substantiated / total * 100, 1) if total > 0 else 0.0
        return {
            "title": "Substantiation Rate",
            "total_claims": total,
            "substantiated_claims": substantiated,
            "unsubstantiated_claims": total - substantiated,
            "rate_pct": rate,
            "traffic_light": _traffic_light(rate),
            "by_type": sub.get("by_type", {}),
        }

    def _section_evidence_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build evidence completeness section."""
        scores = data.get("scores", {})
        ev = scores.get("evidence", {})
        score = ev.get("completeness_pct", 0.0)
        return {
            "title": "Evidence Completeness",
            "completeness_pct": score,
            "traffic_light": _traffic_light(score),
            "total_evidence_items": ev.get("total_items", 0),
            "valid_items": ev.get("valid_items", 0),
            "expired_items": ev.get("expired_items", 0),
            "missing_items": ev.get("missing_items", 0),
        }

    def _section_greenwashing_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build greenwashing risk section."""
        scores = data.get("scores", {})
        gw = scores.get("greenwashing", {})
        risk_score = gw.get("risk_score", 0.0)
        inverted = max(0.0, 100.0 - risk_score * 10.0)
        return {
            "title": "Greenwashing Risk",
            "risk_score": round(risk_score, 1),
            "risk_level": gw.get("risk_level", "low"),
            "inverted_score_pct": round(inverted, 1),
            "traffic_light": _traffic_light(inverted),
            "sins_triggered": gw.get("sins_triggered", 0),
            "ucpd_flags": gw.get("ucpd_flags", 0),
        }

    def _section_label_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build label compliance section."""
        scores = data.get("scores", {})
        lb = scores.get("labels", {})
        rate = lb.get("compliance_rate_pct", 0.0)
        return {
            "title": "Label Compliance",
            "total_labels": lb.get("total_labels", 0),
            "compliant_labels": lb.get("compliant_labels", 0),
            "non_compliant_labels": lb.get("non_compliant_labels", 0),
            "compliance_rate_pct": rate,
            "traffic_light": _traffic_light(rate),
            "expired_certificates": lb.get("expired_certificates", 0),
        }

    def _section_benchmark(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build benchmark comparison section."""
        benchmark = data.get("benchmark", {})
        return {
            "title": "Benchmark Comparison",
            "entity_score": benchmark.get("entity_score", 0.0),
            "sector_average": benchmark.get("sector_average", 0.0),
            "sector_median": benchmark.get("sector_median", 0.0),
            "sector_best": benchmark.get("sector_best", 0.0),
            "percentile_rank": benchmark.get("percentile_rank", 0),
            "sector": benchmark.get("sector", ""),
            "peer_count": benchmark.get("peer_count", 0),
        }

    def _section_trend_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build trend analysis section."""
        trend_data = data.get("trend_data", [])
        return {
            "title": "Trend Analysis",
            "periods": [
                {
                    "period": t.get("period", ""),
                    "composite_score": round(
                        t.get("composite_score", 0.0), 1
                    ),
                    "traffic_light": _traffic_light(
                        t.get("composite_score", 0.0)
                    ),
                    "substantiation_rate": round(
                        t.get("substantiation_rate", 0.0), 1
                    ),
                    "evidence_completeness": round(
                        t.get("evidence_completeness", 0.0), 1
                    ),
                }
                for t in trend_data
            ],
            "overall_trend": data.get("overall_trend", "stable"),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Green Claims Scorecard - EU Green Claims Directive\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Directive:** EU Green Claims Directive 2023/0085"
        )

    def _md_score_overview(self, data: Dict[str, Any]) -> str:
        """Render score overview as markdown."""
        sec = self._section_score_overview(data)
        lines = [
            f"## {sec['title']}\n",
            f"### Composite Score: {sec['composite_score']:.1f}% "
            f"[{sec['traffic_light']}]\n",
            "| Dimension | Score | Status |",
            "|-----------|------:|--------|",
        ]
        for d in sec["dimensions"]:
            lines.append(
                f"| {d['name']} | {d['score']:.1f}% | {d['traffic_light']} |"
            )
        return "\n".join(lines)

    def _md_substantiation_rate(self, data: Dict[str, Any]) -> str:
        """Render substantiation rate as markdown."""
        sec = self._section_substantiation_rate(data)
        return (
            f"## {sec['title']} [{sec['traffic_light']}]\n\n"
            f"**Rate:** {sec['rate_pct']:.1f}%  \n"
            f"**Substantiated:** "
            f"{sec['substantiated_claims']}/{sec['total_claims']}  \n"
            f"**Unsubstantiated:** {sec['unsubstantiated_claims']}"
        )

    def _md_evidence_completeness(self, data: Dict[str, Any]) -> str:
        """Render evidence completeness as markdown."""
        sec = self._section_evidence_completeness(data)
        return (
            f"## {sec['title']} [{sec['traffic_light']}]\n\n"
            f"**Completeness:** {sec['completeness_pct']:.1f}%\n\n"
            f"| Metric | Count |\n|--------|------:|\n"
            f"| Total Items | {sec['total_evidence_items']} |\n"
            f"| Valid | {sec['valid_items']} |\n"
            f"| Expired | {sec['expired_items']} |\n"
            f"| Missing | {sec['missing_items']} |"
        )

    def _md_greenwashing_risk(self, data: Dict[str, Any]) -> str:
        """Render greenwashing risk as markdown."""
        sec = self._section_greenwashing_risk(data)
        return (
            f"## {sec['title']} [{sec['traffic_light']}]\n\n"
            f"**Risk Score:** {sec['risk_score']:.1f} / 10.0  \n"
            f"**Risk Level:** {sec['risk_level']}  \n"
            f"**Sins Triggered:** {sec['sins_triggered']}  \n"
            f"**UCPD Flags:** {sec['ucpd_flags']}"
        )

    def _md_label_compliance(self, data: Dict[str, Any]) -> str:
        """Render label compliance as markdown."""
        sec = self._section_label_compliance(data)
        return (
            f"## {sec['title']} [{sec['traffic_light']}]\n\n"
            f"**Compliance Rate:** {sec['compliance_rate_pct']:.1f}%  \n"
            f"**Compliant:** "
            f"{sec['compliant_labels']}/{sec['total_labels']}  \n"
            f"**Non-Compliant:** {sec['non_compliant_labels']}  \n"
            f"**Expired Certificates:** {sec['expired_certificates']}"
        )

    def _md_benchmark_comparison(self, data: Dict[str, Any]) -> str:
        """Render benchmark comparison as markdown."""
        sec = self._section_benchmark(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Sector:** {sec['sector']}  \n"
            f"**Peer Count:** {sec['peer_count']}\n\n"
            f"| Metric | Score |\n|--------|------:|\n"
            f"| Entity Score | {sec['entity_score']:.1f}% |\n"
            f"| Sector Average | {sec['sector_average']:.1f}% |\n"
            f"| Sector Median | {sec['sector_median']:.1f}% |\n"
            f"| Sector Best | {sec['sector_best']:.1f}% |\n"
            f"| Percentile Rank | {sec['percentile_rank']}th |"
        )

    def _md_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render trend analysis as markdown."""
        sec = self._section_trend_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Overall Trend:** {sec['overall_trend']}\n",
            "| Period | Composite | Status | Substantiation | Evidence |",
            "|--------|--------:|--------|-------------:|---------:|",
        ]
        for p in sec["periods"]:
            lines.append(
                f"| {p['period']} | {p['composite_score']:.1f}% "
                f"| {p['traffic_light']} "
                f"| {p['substantiation_rate']:.1f}% "
                f"| {p['evidence_completeness']:.1f}% |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        """Render provenance section as markdown."""
        prov = _compute_hash(data)
        return (
            f"## Provenance\n\n"
            f"**Input Data Hash:** `{prov}`  \n"
            f"**Template Version:** 18.0.0  \n"
            f"**Generated At:** "
            f"{self.generated_at.isoformat() if self.generated_at else ''}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Scorecard generated by PACK-018 "
            f"EU Green Claims Prep Pack on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#1b5e20;border-bottom:2px solid #1b5e20;padding-bottom:.3em}"
            "h2{color:#2e7d32;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8f5e9}"
            ".tl-green{color:#2e7d32;font-weight:bold}"
            ".tl-amber{color:#e65100;font-weight:bold}"
            ".tl-red{color:#c62828;font-weight:bold}"
            ".card{border:1px solid #ccc;padding:1em;margin:.5em;"
            "display:inline-block;min-width:200px;vertical-align:top}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Green Claims Scorecard - EU Green Claims Directive</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> "
            f"| {data.get('reporting_period', '')}</p>"
        )

    def _html_score_overview(self, data: Dict[str, Any]) -> str:
        """Render score overview HTML."""
        sec = self._section_score_overview(data)
        css_class = f"tl-{sec['traffic_light'].lower()}"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css_class}'>Composite Score: "
            f"{sec['composite_score']:.1f}% [{sec['traffic_light']}]</p>"
        )

    def _html_dimension_cards(self, data: Dict[str, Any]) -> str:
        """Render dimension cards HTML."""
        sec = self._section_score_overview(data)
        cards = ""
        for d in sec["dimensions"]:
            css_class = f"tl-{d['traffic_light'].lower()}"
            cards += (
                f"<div class='card'><h3>{d['name']}</h3>"
                f"<p class='{css_class}'>"
                f"{d['score']:.1f}% [{d['traffic_light']}]</p></div>\n"
            )
        return f"<div class='cards'>\n{cards}</div>"

    def _html_benchmark(self, data: Dict[str, Any]) -> str:
        """Render benchmark comparison HTML."""
        sec = self._section_benchmark(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Score</th></tr>"
            f"<tr><td>Entity</td><td>{sec['entity_score']:.1f}%</td></tr>"
            f"<tr><td>Sector Average</td>"
            f"<td>{sec['sector_average']:.1f}%</td></tr>"
            f"<tr><td>Sector Median</td>"
            f"<td>{sec['sector_median']:.1f}%</td></tr>"
            f"<tr><td>Sector Best</td>"
            f"<td>{sec['sector_best']:.1f}%</td></tr>"
            f"<tr><td>Percentile</td>"
            f"<td>{sec['percentile_rank']}th</td></tr></table>"
        )

    def _html_trend(self, data: Dict[str, Any]) -> str:
        """Render trend analysis HTML."""
        sec = self._section_trend_analysis(data)
        rows = "".join(
            f"<tr><td>{p['period']}</td>"
            f"<td>{p['composite_score']:.1f}%</td>"
            f"<td class='tl-{p['traffic_light'].lower()}'>"
            f"{p['traffic_light']}</td></tr>"
            for p in sec["periods"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Overall Trend: {sec['overall_trend']}</p>\n"
            f"<table><tr><th>Period</th><th>Score</th><th>Status</th></tr>"
            f"{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_dimensions(self, scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build dimension list from scores dict."""
        dims: List[Dict[str, Any]] = []

        sub = scores.get("substantiation", {})
        total = sub.get("total_claims", 0)
        sub_count = sub.get("substantiated_claims", 0)
        sub_rate = round(sub_count / total * 100, 1) if total > 0 else 0.0
        dims.append({
            "name": "Substantiation Rate", "score": sub_rate,
            "traffic_light": _traffic_light(sub_rate),
        })

        ev_pct = scores.get("evidence", {}).get("completeness_pct", 0.0)
        dims.append({
            "name": "Evidence Completeness", "score": ev_pct,
            "traffic_light": _traffic_light(ev_pct),
        })

        gw_raw = scores.get("greenwashing", {}).get("risk_score", 0.0)
        gw_inv = max(0.0, 100.0 - gw_raw * 10.0)
        dims.append({
            "name": "Greenwashing Risk", "score": round(gw_inv, 1),
            "traffic_light": _traffic_light(gw_inv),
        })

        lb_rate = scores.get("labels", {}).get("compliance_rate_pct", 0.0)
        dims.append({
            "name": "Label Compliance", "score": lb_rate,
            "traffic_light": _traffic_light(lb_rate),
        })

        return dims

    def _composite_score(self, dimensions: List[Dict[str, Any]]) -> float:
        """Calculate weighted composite score from dimensions."""
        if not dimensions:
            return 0.0
        total = sum(d["score"] for d in dimensions)
        return round(total / len(dimensions), 1)
