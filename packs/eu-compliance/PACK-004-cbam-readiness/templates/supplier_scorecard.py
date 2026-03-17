"""
SupplierScorecardTemplate - CBAM supplier data quality scorecard template.

This module implements the supplier data quality scorecard for CBAM compliance.
It generates formatted scorecards with supplier summary tables, per-supplier
detail views, data quality distribution, missing data heatmaps, improvement
recommendations, and submission history timelines.

Example:
    >>> template = SupplierScorecardTemplate()
    >>> data = {"suppliers": [...], "quality_distribution": {...}, ...}
    >>> markdown = template.render_markdown(data)
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class SupplierScorecardTemplate:
    """
    CBAM supplier data quality scorecard template.

    Generates formatted scorecards assessing supplier data quality for
    CBAM compliance. Includes summary tables, per-supplier details,
    quality tier distribution, missing data analysis, and recommendations.

    Attributes:
        config: Optional configuration dictionary.
        generated_at: Timestamp of report generation.
    """

    QUALITY_TIERS: Dict[str, Dict[str, Any]] = {
        "excellent": {"label": "Excellent", "min_score": 90, "color": "#2ecc71"},
        "good": {"label": "Good", "min_score": 70, "color": "#27ae60"},
        "fair": {"label": "Fair", "min_score": 50, "color": "#f39c12"},
        "poor": {"label": "Poor", "min_score": 0, "color": "#e74c3c"},
    }

    DATA_FIELDS: List[str] = [
        "installation_id",
        "emission_factor",
        "production_volume",
        "fuel_consumption",
        "electricity_consumption",
        "direct_emissions",
        "indirect_emissions",
        "verification_status",
        "cn_code_mapping",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize SupplierScorecardTemplate.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - quality_threshold (float): Minimum acceptable quality score.
                - data_fields (list[str]): Override default data fields.
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the supplier scorecard as Markdown.

        Args:
            data: Scorecard data dictionary containing:
                - suppliers (list[dict]): supplier records with quality scores
                - quality_distribution (dict): counts by tier
                - missing_data_matrix (list[dict]): supplier x field matrix
                - recommendations (list[dict]): improvement recommendations
                - submission_history (list[dict]): historical submissions

        Returns:
            Formatted Markdown string.
        """
        sections: List[str] = []

        sections.append(self._md_header())
        sections.append(self._md_supplier_summary(data))
        sections.append(self._md_supplier_detail(data))
        sections.append(self._md_quality_distribution(data))
        sections.append(self._md_missing_data_heatmap(data))
        sections.append(self._md_recommendations(data))
        sections.append(self._md_submission_history(data))
        sections.append(self._md_provenance_footer())

        content = "\n\n".join(sections)
        provenance_hash = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the supplier scorecard as self-contained HTML.

        Args:
            data: Scorecard data dictionary (same schema as render_markdown).

        Returns:
            Complete HTML document string with inline CSS.
        """
        sections: List[str] = []

        sections.append(self._html_header())
        sections.append(self._html_supplier_summary(data))
        sections.append(self._html_supplier_detail(data))
        sections.append(self._html_quality_distribution(data))
        sections.append(self._html_missing_data_heatmap(data))
        sections.append(self._html_recommendations(data))
        sections.append(self._html_submission_history(data))

        body = "\n".join(sections)
        provenance_hash = self._generate_provenance_hash(body)

        return self._wrap_html(
            title="CBAM Supplier Data Quality Scorecard",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the supplier scorecard as a structured dict.

        Args:
            data: Scorecard data dictionary (same schema as render_markdown).

        Returns:
            Dictionary with all scorecard sections and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_supplier_scorecard",
            "generated_at": self.generated_at,
            "supplier_summary": self._json_supplier_summary(data),
            "supplier_detail": self._json_supplier_detail(data),
            "quality_distribution": self._json_quality_distribution(data),
            "missing_data_matrix": self._json_missing_data_matrix(data),
            "recommendations": self._json_recommendations(data),
            "submission_history": self._json_submission_history(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self) -> str:
        """Build Markdown header."""
        return (
            "# CBAM Supplier Data Quality Scorecard\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_supplier_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown supplier summary table."""
        suppliers: List[Dict[str, Any]] = data.get("suppliers", [])

        header = (
            "## Supplier Summary\n\n"
            "| Supplier | Country | Installations | Data Quality Score | "
            "Quality Tier | Verification Status |\n"
            "|----------|---------|---------------|-------------------|"
            "-------------|---------------------|\n"
        )

        rows: List[str] = []
        for s in sorted(suppliers, key=lambda x: x.get("data_quality_score", 0), reverse=True):
            score = s.get("data_quality_score", 0.0)
            tier = self._get_quality_tier(score)

            rows.append(
                f"| {s.get('name', '')} | "
                f"{s.get('country', '')} | "
                f"{s.get('installation_count', 0)} | "
                f"{self._format_number(score, 1)}/100 | "
                f"{tier} | "
                f"{s.get('verification_status', 'N/A')} |"
            )

        return header + "\n".join(rows)

    def _md_supplier_detail(self, data: Dict[str, Any]) -> str:
        """Build Markdown per-supplier detail sections."""
        suppliers: List[Dict[str, Any]] = data.get("suppliers", [])

        section = "## Supplier Details\n"

        for s in suppliers:
            installations: List[Dict[str, Any]] = s.get("installations", [])
            cn_codes: List[str] = s.get("cn_codes", [])
            completeness = s.get("data_completeness_pct", 0.0)
            last_submission = s.get("last_submission_date", "N/A")

            section += (
                f"\n### {s.get('name', 'Unknown')}\n\n"
                f"- **Country:** {s.get('country', 'N/A')}\n"
                f"- **CN Codes:** {', '.join(self._format_cn_code(c) for c in cn_codes) if cn_codes else 'N/A'}\n"
                f"- **Data Completeness:** {self._format_percentage(completeness)}\n"
                f"- **Last Submission:** {self._format_date(last_submission)}\n\n"
            )

            if installations:
                section += (
                    "| Installation ID | Name | Specific Emissions | Method |\n"
                    "|-----------------|------|-------------------|--------|\n"
                )
                for inst in installations:
                    section += (
                        f"| {inst.get('installation_id', '')} | "
                        f"{inst.get('name', '')} | "
                        f"{self._format_number(inst.get('specific_emissions', 0.0), 4)} tCO2e/t | "
                        f"{inst.get('method', '')} |\n"
                    )

        return section

    def _md_quality_distribution(self, data: Dict[str, Any]) -> str:
        """Build Markdown data quality distribution section."""
        distribution: Dict[str, int] = data.get("quality_distribution", {})
        total = sum(distribution.values()) if distribution else 1

        section = (
            "## Data Quality Distribution\n\n"
            "| Tier | Count | Percentage | Bar |\n"
            "|------|-------|------------|-----|\n"
        )

        for tier_key, tier_info in self.QUALITY_TIERS.items():
            count = distribution.get(tier_key, 0)
            pct = (count / total * 100) if total > 0 else 0.0
            bar_len = int(pct / 5)  # Scale to ~20 char max
            bar = "#" * bar_len

            section += (
                f"| {tier_info['label']} | {count} | "
                f"{self._format_percentage(pct)} | {bar} |\n"
            )

        return section

    def _md_missing_data_heatmap(self, data: Dict[str, Any]) -> str:
        """Build Markdown missing data heatmap (supplier x field matrix)."""
        matrix: List[Dict[str, Any]] = data.get("missing_data_matrix", [])
        fields = self.config.get("data_fields", self.DATA_FIELDS)

        if not matrix:
            return "## Missing Data Heatmap\n\n*No missing data information available.*"

        field_headers = " | ".join(f[:12] for f in fields)
        separator = " | ".join("---" for _ in fields)

        header = (
            "## Missing Data Heatmap\n\n"
            f"| Supplier | {field_headers} |\n"
            f"|----------|{separator}|\n"
        )

        rows: List[str] = []
        for entry in matrix:
            supplier_name = entry.get("supplier", "")
            field_values: List[str] = []
            for field in fields:
                has_data = entry.get("fields", {}).get(field, False)
                field_values.append("OK" if has_data else "MISS")

            rows.append(f"| {supplier_name} | {' | '.join(field_values)} |")

        return header + "\n".join(rows)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Build Markdown improvement recommendations section."""
        recommendations: List[Dict[str, Any]] = data.get("recommendations", [])

        if not recommendations:
            return "## Improvement Recommendations\n\n*No recommendations at this time.*"

        section = "## Improvement Recommendations\n\n"

        for i, rec in enumerate(recommendations, 1):
            priority = rec.get("priority", "medium").upper()
            section += (
                f"### {i}. {rec.get('supplier', 'General')} - {rec.get('title', '')}\n\n"
                f"**Priority:** {priority}\n\n"
                f"{rec.get('description', '')}\n\n"
                f"**Action:** {rec.get('action', 'N/A')}\n\n"
                f"**Deadline:** {self._format_date(rec.get('deadline', 'N/A'))}\n\n"
            )

        return section.rstrip()

    def _md_submission_history(self, data: Dict[str, Any]) -> str:
        """Build Markdown submission history timeline."""
        history: List[Dict[str, Any]] = data.get("submission_history", [])

        if not history:
            return "## Submission History\n\n*No submission history available.*"

        header = (
            "## Submission History\n\n"
            "| Date | Supplier | Data Type | Quality Score | Status |\n"
            "|------|----------|-----------|---------------|--------|\n"
        )

        rows: List[str] = []
        for entry in sorted(history, key=lambda x: x.get("date", ""), reverse=True):
            rows.append(
                f"| {self._format_date(entry.get('date', ''))} | "
                f"{entry.get('supplier', '')} | "
                f"{entry.get('data_type', '')} | "
                f"{self._format_number(entry.get('quality_score', 0.0), 1)}/100 | "
                f"{entry.get('status', '')} |"
            )

        return header + "\n".join(rows)

    def _md_provenance_footer(self) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: SupplierScorecardTemplate v1.0*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self) -> str:
        """Build HTML header."""
        return (
            '<div class="report-header">'
            '<h1>CBAM Supplier Data Quality Scorecard</h1>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
        )

    def _html_supplier_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML supplier summary table."""
        suppliers: List[Dict[str, Any]] = data.get("suppliers", [])

        rows_html = ""
        for s in sorted(suppliers, key=lambda x: x.get("data_quality_score", 0), reverse=True):
            score = s.get("data_quality_score", 0.0)
            tier = self._get_quality_tier(score)
            tier_color = self._get_tier_color(score)

            rows_html += (
                f'<tr><td>{s.get("name", "")}</td>'
                f'<td>{s.get("country", "")}</td>'
                f'<td class="num">{s.get("installation_count", 0)}</td>'
                f'<td class="num">{self._format_number(score, 1)}/100</td>'
                f'<td><span class="tier-badge" style="background:{tier_color}">'
                f'{tier}</span></td>'
                f'<td>{s.get("verification_status", "N/A")}</td></tr>'
            )

        return (
            '<div class="section"><h2>Supplier Summary</h2>'
            '<table><thead><tr>'
            '<th>Supplier</th><th>Country</th><th>Installations</th>'
            '<th>Quality Score</th><th>Tier</th><th>Verification</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_supplier_detail(self, data: Dict[str, Any]) -> str:
        """Build HTML per-supplier detail cards."""
        suppliers: List[Dict[str, Any]] = data.get("suppliers", [])

        cards_html = ""
        for s in suppliers:
            score = s.get("data_quality_score", 0.0)
            tier_color = self._get_tier_color(score)
            cn_codes = s.get("cn_codes", [])
            installations = s.get("installations", [])

            inst_rows = ""
            for inst in installations:
                inst_rows += (
                    f'<tr><td class="code">{inst.get("installation_id", "")}</td>'
                    f'<td>{inst.get("name", "")}</td>'
                    f'<td class="num">{self._format_number(inst.get("specific_emissions", 0.0), 4)}</td>'
                    f'<td>{inst.get("method", "")}</td></tr>'
                )

            cards_html += (
                f'<div class="supplier-card" style="border-left:4px solid {tier_color}">'
                f'<h3>{s.get("name", "")}</h3>'
                f'<div class="supplier-meta">'
                f'<span>Country: {s.get("country", "N/A")}</span>'
                f'<span>CN Codes: {", ".join(self._format_cn_code(c) for c in cn_codes) if cn_codes else "N/A"}</span>'
                f'<span>Completeness: {self._format_percentage(s.get("data_completeness_pct", 0.0))}</span>'
                f'<span>Last submission: {self._format_date(s.get("last_submission_date", "N/A"))}</span>'
                f'</div>'
            )

            if inst_rows:
                cards_html += (
                    '<table><thead><tr>'
                    '<th>Installation ID</th><th>Name</th>'
                    '<th>Specific Emissions (tCO2e/t)</th><th>Method</th>'
                    f'</tr></thead><tbody>{inst_rows}</tbody></table>'
                )

            cards_html += '</div>'

        return f'<div class="section"><h2>Supplier Details</h2>{cards_html}</div>'

    def _html_quality_distribution(self, data: Dict[str, Any]) -> str:
        """Build HTML data quality distribution pie chart (as bars)."""
        distribution: Dict[str, int] = data.get("quality_distribution", {})
        total = sum(distribution.values()) if distribution else 1

        bars_html = ""
        for tier_key, tier_info in self.QUALITY_TIERS.items():
            count = distribution.get(tier_key, 0)
            pct = (count / total * 100) if total > 0 else 0.0

            bars_html += (
                f'<div class="dist-item">'
                f'<div class="dist-label">{tier_info["label"]} ({count})</div>'
                f'<div class="progress-bar">'
                f'<div class="progress-fill" '
                f'style="width:{pct}%;background:{tier_info["color"]}"></div>'
                f'</div>'
                f'<div class="dist-pct">{self._format_percentage(pct)}</div>'
                f'</div>'
            )

        return (
            '<div class="section"><h2>Data Quality Distribution</h2>'
            f'<div class="dist-container">{bars_html}</div></div>'
        )

    def _html_missing_data_heatmap(self, data: Dict[str, Any]) -> str:
        """Build HTML missing data heatmap."""
        matrix: List[Dict[str, Any]] = data.get("missing_data_matrix", [])
        fields = self.config.get("data_fields", self.DATA_FIELDS)

        if not matrix:
            return (
                '<div class="section"><h2>Missing Data Heatmap</h2>'
                '<p class="note">No missing data information available.</p></div>'
            )

        headers_html = "".join(
            f'<th class="rotated"><span>{f.replace("_", " ").title()}</span></th>'
            for f in fields
        )

        rows_html = ""
        for entry in matrix:
            cells = ""
            for field in fields:
                has_data = entry.get("fields", {}).get(field, False)
                color = "#2ecc71" if has_data else "#e74c3c"
                label = "OK" if has_data else "MISS"
                cells += f'<td class="heatmap-cell" style="background:{color};color:#fff">{label}</td>'

            rows_html += (
                f'<tr><td>{entry.get("supplier", "")}</td>{cells}</tr>'
            )

        return (
            '<div class="section"><h2>Missing Data Heatmap</h2>'
            f'<table class="heatmap"><thead><tr><th>Supplier</th>{headers_html}'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Build HTML improvement recommendations."""
        recommendations: List[Dict[str, Any]] = data.get("recommendations", [])

        if not recommendations:
            return (
                '<div class="section"><h2>Improvement Recommendations</h2>'
                '<p class="note">No recommendations at this time.</p></div>'
            )

        cards_html = ""
        priority_colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71"}

        for rec in recommendations:
            priority = rec.get("priority", "medium").lower()
            color = priority_colors.get(priority, "#95a5a6")

            cards_html += (
                f'<div class="rec-card" style="border-left:4px solid {color}">'
                f'<div class="rec-header">'
                f'<strong>{rec.get("supplier", "General")}</strong> - '
                f'{rec.get("title", "")}'
                f'<span class="priority-badge" style="background:{color}">'
                f'{priority.upper()}</span></div>'
                f'<p>{rec.get("description", "")}</p>'
                f'<div class="rec-action"><strong>Action:</strong> {rec.get("action", "N/A")}</div>'
                f'<div class="rec-deadline"><strong>Deadline:</strong> '
                f'{self._format_date(rec.get("deadline", "N/A"))}</div>'
                f'</div>'
            )

        return f'<div class="section"><h2>Improvement Recommendations</h2>{cards_html}</div>'

    def _html_submission_history(self, data: Dict[str, Any]) -> str:
        """Build HTML submission history timeline."""
        history: List[Dict[str, Any]] = data.get("submission_history", [])

        if not history:
            return (
                '<div class="section"><h2>Submission History</h2>'
                '<p class="note">No submission history available.</p></div>'
            )

        rows_html = ""
        for entry in sorted(history, key=lambda x: x.get("date", ""), reverse=True):
            score = entry.get("quality_score", 0.0)
            tier_color = self._get_tier_color(score)

            rows_html += (
                f'<tr><td>{self._format_date(entry.get("date", ""))}</td>'
                f'<td>{entry.get("supplier", "")}</td>'
                f'<td>{entry.get("data_type", "")}</td>'
                f'<td class="num" style="color:{tier_color}">'
                f'{self._format_number(score, 1)}/100</td>'
                f'<td>{entry.get("status", "")}</td></tr>'
            )

        return (
            '<div class="section"><h2>Submission History</h2>'
            '<table><thead><tr>'
            '<th>Date</th><th>Supplier</th><th>Data Type</th>'
            '<th>Quality Score</th><th>Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_supplier_summary(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON supplier summary."""
        suppliers: List[Dict[str, Any]] = data.get("suppliers", [])

        return [
            {
                "name": s.get("name", ""),
                "country": s.get("country", ""),
                "installation_count": s.get("installation_count", 0),
                "data_quality_score": round(s.get("data_quality_score", 0.0), 1),
                "quality_tier": self._get_quality_tier(s.get("data_quality_score", 0.0)),
                "verification_status": s.get("verification_status", "N/A"),
            }
            for s in sorted(suppliers, key=lambda x: x.get("data_quality_score", 0), reverse=True)
        ]

    def _json_supplier_detail(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON supplier detail."""
        suppliers: List[Dict[str, Any]] = data.get("suppliers", [])

        return [
            {
                "name": s.get("name", ""),
                "country": s.get("country", ""),
                "cn_codes": [self._format_cn_code(c) for c in s.get("cn_codes", [])],
                "data_completeness_pct": round(s.get("data_completeness_pct", 0.0), 2),
                "last_submission_date": self._format_date(s.get("last_submission_date", "")),
                "installations": [
                    {
                        "installation_id": inst.get("installation_id", ""),
                        "name": inst.get("name", ""),
                        "specific_emissions_tco2e_per_t": round(inst.get("specific_emissions", 0.0), 4),
                        "method": inst.get("method", ""),
                    }
                    for inst in s.get("installations", [])
                ],
            }
            for s in suppliers
        ]

    def _json_quality_distribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON quality distribution."""
        distribution: Dict[str, int] = data.get("quality_distribution", {})
        total = sum(distribution.values()) if distribution else 0

        result: Dict[str, Any] = {"total_suppliers": total, "tiers": {}}
        for tier_key, tier_info in self.QUALITY_TIERS.items():
            count = distribution.get(tier_key, 0)
            pct = (count / total * 100) if total > 0 else 0.0
            result["tiers"][tier_key] = {
                "label": tier_info["label"],
                "count": count,
                "percentage": round(pct, 2),
            }

        return result

    def _json_missing_data_matrix(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON missing data matrix."""
        matrix: List[Dict[str, Any]] = data.get("missing_data_matrix", [])
        fields = self.config.get("data_fields", self.DATA_FIELDS)

        result: List[Dict[str, Any]] = []
        for entry in matrix:
            field_status: Dict[str, bool] = {}
            for field in fields:
                field_status[field] = entry.get("fields", {}).get(field, False)

            missing_count = sum(1 for v in field_status.values() if not v)
            result.append({
                "supplier": entry.get("supplier", ""),
                "fields": field_status,
                "missing_count": missing_count,
                "completeness_pct": round(
                    (len(fields) - missing_count) / len(fields) * 100 if fields else 0, 2
                ),
            })

        return result

    def _json_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON recommendations."""
        recommendations: List[Dict[str, Any]] = data.get("recommendations", [])
        return [
            {
                "supplier": rec.get("supplier", "General"),
                "title": rec.get("title", ""),
                "priority": rec.get("priority", "medium"),
                "description": rec.get("description", ""),
                "action": rec.get("action", ""),
                "deadline": self._format_date(rec.get("deadline", "")),
            }
            for rec in recommendations
        ]

    def _json_submission_history(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON submission history."""
        history: List[Dict[str, Any]] = data.get("submission_history", [])
        return [
            {
                "date": self._format_date(entry.get("date", "")),
                "supplier": entry.get("supplier", ""),
                "data_type": entry.get("data_type", ""),
                "quality_score": round(entry.get("quality_score", 0.0), 1),
                "status": entry.get("status", ""),
            }
            for entry in sorted(history, key=lambda x: x.get("date", ""), reverse=True)
        ]

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _get_quality_tier(self, score: float) -> str:
        """Determine quality tier label from score."""
        for tier_key in ["excellent", "good", "fair", "poor"]:
            if score >= self.QUALITY_TIERS[tier_key]["min_score"]:
                return self.QUALITY_TIERS[tier_key]["label"]
        return "Poor"

    def _get_tier_color(self, score: float) -> str:
        """Get CSS color for quality tier from score."""
        for tier_key in ["excellent", "good", "fair", "poor"]:
            if score >= self.QUALITY_TIERS[tier_key]["min_score"]:
                return self.QUALITY_TIERS[tier_key]["color"]
        return "#e74c3c"

    def _generate_provenance_hash(self, content: str) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _format_number(self, value: Union[int, float], decimals: int = 2) -> str:
        """Format a numeric value with thousand separators and fixed decimals."""
        return f"{value:,.{decimals}f}"

    def _format_percentage(self, value: Union[int, float]) -> str:
        """Format a percentage value."""
        return f"{value:.2f}%"

    def _format_date(self, dt: Union[datetime, str]) -> str:
        """Format a datetime to ISO date string."""
        if isinstance(dt, str):
            return dt[:10] if len(dt) >= 10 else dt
        return dt.strftime("%Y-%m-%d")

    def _format_currency(self, value: Union[int, float], currency: str = "EUR") -> str:
        """Format a currency value."""
        return f"{currency} {value:,.2f}"

    def _format_cn_code(self, code: str) -> str:
        """Format a CN code to standard XXXX.XX format."""
        clean = code.replace(".", "").replace(" ", "")
        if len(clean) >= 6:
            return f"{clean[:4]}.{clean[4:6]}"
        elif len(clean) == 4:
            return f"{clean}.00"
        return code

    def _html_progress_bar(self, pct: float, color: str) -> str:
        """Generate an inline HTML progress bar."""
        width = max(0, min(100, pct))
        return (
            f'<div class="progress-bar">'
            f'<div class="progress-fill" '
            f'style="width:{width}%;background:{color}"></div>'
            f'</div>'
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 8px 0;font-size:24px}"
            ".meta-item{font-size:13px;opacity:0.8}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".code{font-family:monospace;font-size:13px}"
            ".tier-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:12px}"
            ".supplier-card{background:#f8f9fa;padding:16px;border-radius:8px;"
            "margin-bottom:12px}"
            ".supplier-card h3{margin:0 0 8px 0;font-size:16px;color:#2c3e50}"
            ".supplier-meta{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;"
            "color:#7f8c8d;margin-bottom:12px}"
            ".dist-container{margin:16px 0}"
            ".dist-item{display:flex;align-items:center;gap:12px;margin-bottom:8px}"
            ".dist-label{width:140px;font-size:14px}"
            ".dist-pct{width:60px;font-size:13px;text-align:right}"
            ".progress-bar{flex:1;background:#ecf0f1;border-radius:4px;height:16px;"
            "overflow:hidden}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".heatmap{font-size:12px}"
            ".heatmap-cell{text-align:center;font-weight:bold;padding:4px 8px}"
            ".rotated span{writing-mode:vertical-rl;transform:rotate(180deg);"
            "font-size:11px}"
            ".rec-card{background:#f8f9fa;padding:12px 16px;border-radius:8px;"
            "margin-bottom:12px}"
            ".rec-header{font-size:15px;margin-bottom:8px;display:flex;"
            "align-items:center;gap:8px;flex-wrap:wrap}"
            ".priority-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px}"
            ".rec-card p{margin:4px 0;font-size:14px;color:#555}"
            ".rec-action,.rec-deadline{font-size:13px;color:#7f8c8d;margin-top:4px}"
            ".note{color:#7f8c8d;font-style:italic}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )

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
            f'Template: SupplierScorecardTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
