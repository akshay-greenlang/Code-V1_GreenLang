"""
PredictiveInsightsTemplate - AI forecast visualization for CSRD Enterprise Pack.

This module implements the predictive insights template rendering emission
forecasts with confidence intervals, gap-to-target analysis, anomaly detection
timelines, feature importance charts, Monte Carlo distributions, risk matrices,
and model performance metrics.

Example:
    >>> template = PredictiveInsightsTemplate()
    >>> data = {"forecasts": [...], "gap_analysis": {...}, "anomalies": [...]}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class PredictiveInsightsTemplate:
    """
    AI-driven predictive insights template.

    Renders emission forecasts, gap-to-target analysis, anomaly detection,
    feature importance, Monte Carlo distributions, risk matrices, and
    model performance metrics.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RISK_CATEGORIES = [
        "Scope 1 - Stationary",
        "Scope 1 - Mobile",
        "Scope 1 - Fugitive",
        "Scope 2 - Electricity",
        "Scope 3 - Supply Chain",
        "Scope 3 - Transport",
        "Scope 3 - Waste",
    ]

    LIKELIHOOD_LEVELS = ["Very Low", "Low", "Medium", "High", "Very High"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PredictiveInsightsTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render predictive insights as Markdown.

        Args:
            data: Insights data with forecasts, gap_analysis, anomalies,
                  feature_importance, monte_carlo, risk_matrix, model_metrics,
                  and prediction_vs_actual.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._render_md_header(data))
        sections.append(self._render_md_forecast(data))
        sections.append(self._render_md_gap_analysis(data))
        sections.append(self._render_md_anomalies(data))
        sections.append(self._render_md_feature_importance(data))
        sections.append(self._render_md_monte_carlo(data))
        sections.append(self._render_md_risk_matrix(data))
        sections.append(self._render_md_model_metrics(data))
        sections.append(self._render_md_prediction_vs_actual(data))
        sections.append(self._render_md_footer(data))

        content = "\n\n".join(sections)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- Provenance: {provenance} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render predictive insights as self-contained HTML.

        Args:
            data: Insights data dict.

        Returns:
            Complete HTML string with inline styles.
        """
        self.generated_at = datetime.utcnow()
        css = self._build_css()
        body_parts: List[str] = []

        body_parts.append(self._render_html_header(data))
        body_parts.append(self._render_html_forecast(data))
        body_parts.append(self._render_html_gap_analysis(data))
        body_parts.append(self._render_html_anomalies(data))
        body_parts.append(self._render_html_feature_importance(data))
        body_parts.append(self._render_html_monte_carlo(data))
        body_parts.append(self._render_html_risk_matrix(data))
        body_parts.append(self._render_html_model_metrics(data))
        body_parts.append(self._render_html_prediction_vs_actual(data))
        body_parts.append(self._render_html_footer(data))

        body_html = "\n".join(body_parts)
        provenance = self._generate_provenance_hash(body_html)

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>Predictive Insights</title>\n<style>\n{css}\n</style>\n"
            "</head>\n<body>\n"
            f"<div class=\"insights-container\">\n{body_html}\n</div>\n"
            f"<!-- Provenance: {provenance} -->\n"
            "</body>\n</html>"
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render predictive insights as structured JSON.

        Args:
            data: Insights data dict.

        Returns:
            Structured dict with all insight sections.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "predictive_insights",
            "version": "1.0.0",
            "generated_at": self.generated_at.isoformat(),
            "emission_forecast": self._build_json_forecast(data),
            "gap_analysis": self._build_json_gap_analysis(data),
            "anomalies": self._build_json_anomalies(data),
            "feature_importance": self._build_json_feature_importance(data),
            "monte_carlo": self._build_json_monte_carlo(data),
            "risk_matrix": self._build_json_risk_matrix(data),
            "model_metrics": self._build_json_model_metrics(data),
            "prediction_vs_actual": self._build_json_prediction_vs_actual(data),
        }
        provenance = self._generate_provenance_hash(json.dumps(result, default=str))
        result["provenance_hash"] = provenance
        return result

    # ------------------------------------------------------------------
    # Markdown renderers
    # ------------------------------------------------------------------

    def _render_md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        title = data.get("title", "Predictive Insights Report")
        ts = self._format_date(self.generated_at)
        return f"# {title}\n\n**Generated:** {ts}\n\n---"

    def _render_md_forecast(self, data: Dict[str, Any]) -> str:
        """Render emission forecast table with confidence intervals."""
        forecasts: List[Dict[str, Any]] = data.get("forecasts", [])
        if not forecasts:
            return "## Emission Forecast\n\n_No forecast data available._"

        lines = [
            "## Emission Forecast with Confidence Intervals",
            "",
            "| Period | Forecast (tCO2e) | Lower 95% CI | Upper 95% CI | Lower 80% CI | Upper 80% CI |",
            "|--------|-----------------|-------------|-------------|-------------|-------------|",
        ]
        for f in forecasts:
            period = f.get("period", "-")
            forecast = self._format_number(f.get("forecast", 0))
            lo95 = self._format_number(f.get("lower_95", 0))
            hi95 = self._format_number(f.get("upper_95", 0))
            lo80 = self._format_number(f.get("lower_80", 0))
            hi80 = self._format_number(f.get("upper_80", 0))
            lines.append(f"| {period} | {forecast} | {lo95} | {hi95} | {lo80} | {hi80} |")

        return "\n".join(lines)

    def _render_md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap-to-target analysis."""
        gap = data.get("gap_analysis", {})
        if not gap:
            return "## Gap-to-Target Analysis\n\n_No gap analysis data available._"

        trajectories: List[Dict[str, Any]] = gap.get("trajectories", [])
        lines = [
            "## Gap-to-Target Analysis (Actual vs SBTi Trajectory)",
            "",
            f"**Base Year:** {gap.get('base_year', '-')}",
            f"**Target Year:** {gap.get('target_year', '-')}",
            f"**Reduction Target:** {self._format_percentage(gap.get('reduction_target_pct', 0))}",
            f"**Current Gap:** {self._format_number(gap.get('current_gap_tco2e', 0))} tCO2e",
            "",
            "| Year | Actual (tCO2e) | Target (tCO2e) | Gap (tCO2e) | On Track |",
            "|------|---------------|----------------|-------------|----------|",
        ]
        for t in trajectories:
            year = t.get("year", "-")
            actual = self._format_number(t.get("actual", 0))
            target = self._format_number(t.get("target", 0))
            gap_val = self._format_number(t.get("gap", 0))
            on_track = "Yes" if t.get("on_track", False) else "No"
            lines.append(f"| {year} | {actual} | {target} | {gap_val} | {on_track} |")

        return "\n".join(lines)

    def _render_md_anomalies(self, data: Dict[str, Any]) -> str:
        """Render anomaly detection timeline."""
        anomalies: List[Dict[str, Any]] = data.get("anomalies", [])
        if not anomalies:
            return "## Anomaly Detection\n\n_No anomalies detected._"

        lines = [
            "## Anomaly Detection Timeline",
            "",
            "| Date | Category | Value (tCO2e) | Expected | Deviation | Severity |",
            "|------|----------|--------------|----------|-----------|----------|",
        ]
        for a in anomalies:
            date = a.get("date", "-")
            category = a.get("category", "-")
            value = self._format_number(a.get("value", 0))
            expected = self._format_number(a.get("expected", 0))
            deviation = self._format_percentage(a.get("deviation_pct", 0))
            severity = a.get("severity", "medium").upper()
            lines.append(
                f"| {date} | {category} | {value} | {expected} | {deviation} | {severity} |"
            )

        return "\n".join(lines)

    def _render_md_feature_importance(self, data: Dict[str, Any]) -> str:
        """Render feature importance ranking."""
        features: List[Dict[str, Any]] = data.get("feature_importance", [])
        if not features:
            return "## Feature Importance\n\n_No feature importance data available._"

        lines = [
            "## Feature Importance (SHAP-style)",
            "",
            "| Rank | Feature | Importance | Direction |",
            "|------|---------|-----------|-----------|",
        ]
        sorted_features = sorted(
            features, key=lambda x: abs(x.get("importance", 0)), reverse=True
        )
        for rank, f in enumerate(sorted_features, 1):
            name = f.get("feature", "-")
            importance = self._format_number(abs(f.get("importance", 0)), 4)
            direction = "Increasing" if f.get("importance", 0) > 0 else "Decreasing"
            bar = self._text_bar(abs(f.get("importance", 0)), max_val=1.0)
            lines.append(f"| {rank} | {name} | {importance} {bar} | {direction} |")

        return "\n".join(lines)

    def _render_md_monte_carlo(self, data: Dict[str, Any]) -> str:
        """Render Monte Carlo distribution summary."""
        mc = data.get("monte_carlo", {})
        if not mc:
            return "## Monte Carlo Simulation\n\n_No simulation data available._"

        lines = [
            "## Monte Carlo Distribution",
            "",
            f"**Simulations:** {self._format_number(mc.get('n_simulations', 0), 0)}",
            f"**Mean Outcome:** {self._format_number(mc.get('mean', 0))} tCO2e",
            f"**Median Outcome:** {self._format_number(mc.get('median', 0))} tCO2e",
            f"**Std Deviation:** {self._format_number(mc.get('std_dev', 0))} tCO2e",
            f"**P5 (5th percentile):** {self._format_number(mc.get('p5', 0))} tCO2e",
            f"**P25:** {self._format_number(mc.get('p25', 0))} tCO2e",
            f"**P75:** {self._format_number(mc.get('p75', 0))} tCO2e",
            f"**P95 (95th percentile):** {self._format_number(mc.get('p95', 0))} tCO2e",
            "",
            "### Histogram Bins",
            "",
            "| Bin Range (tCO2e) | Frequency | Relative % |",
            "|-------------------|-----------|-----------|",
        ]
        for b in mc.get("histogram_bins", []):
            lo = self._format_number(b.get("lower", 0))
            hi = self._format_number(b.get("upper", 0))
            freq = b.get("frequency", 0)
            pct = self._format_percentage(b.get("relative_pct", 0))
            lines.append(f"| {lo} - {hi} | {freq} | {pct} |")

        return "\n".join(lines)

    def _render_md_risk_matrix(self, data: Dict[str, Any]) -> str:
        """Render risk category matrix."""
        matrix: List[Dict[str, Any]] = data.get("risk_matrix", [])
        if not matrix:
            return "## Risk Category Matrix\n\n_No risk matrix data available._"

        lines = [
            "## Risk Category Matrix",
            "",
            "| Category | Likelihood | Impact | Risk Score | Mitigation |",
            "|----------|-----------|--------|-----------|-----------|",
        ]
        for r in matrix:
            category = r.get("category", "-")
            likelihood = r.get("likelihood", "-")
            impact = r.get("impact", "-")
            score = self._format_number(r.get("risk_score", 0), 1)
            mitigation = r.get("mitigation", "-")
            lines.append(
                f"| {category} | {likelihood} | {impact} | {score} | {mitigation} |"
            )

        return "\n".join(lines)

    def _render_md_model_metrics(self, data: Dict[str, Any]) -> str:
        """Render model performance metrics table."""
        metrics = data.get("model_metrics", {})
        if not metrics:
            return "## Model Performance\n\n_No model metrics available._"

        models: List[Dict[str, Any]] = metrics.get("models", [])
        lines = [
            "## Model Performance Metrics",
            "",
            "| Model | R-Squared | MAE (tCO2e) | RMSE (tCO2e) | MAPE (%) | Training Samples |",
            "|-------|----------|------------|-------------|---------|-----------------|",
        ]
        for m in models:
            name = m.get("name", "-")
            r2 = self._format_number(m.get("r_squared", 0), 4)
            mae = self._format_number(m.get("mae", 0))
            rmse = self._format_number(m.get("rmse", 0))
            mape = self._format_percentage(m.get("mape", 0))
            samples = self._format_number(m.get("training_samples", 0), 0)
            lines.append(f"| {name} | {r2} | {mae} | {rmse} | {mape} | {samples} |")

        return "\n".join(lines)

    def _render_md_prediction_vs_actual(self, data: Dict[str, Any]) -> str:
        """Render prediction versus actual comparison."""
        comparisons: List[Dict[str, Any]] = data.get("prediction_vs_actual", [])
        if not comparisons:
            return "## Prediction vs Actual\n\n_No comparison data available._"

        lines = [
            "## Prediction vs Actual Comparison",
            "",
            "| Period | Predicted (tCO2e) | Actual (tCO2e) | Error (tCO2e) | Error (%) |",
            "|--------|------------------|---------------|--------------|----------|",
        ]
        for c in comparisons:
            period = c.get("period", "-")
            predicted = self._format_number(c.get("predicted", 0))
            actual = self._format_number(c.get("actual", 0))
            error = self._format_number(c.get("error", 0))
            error_pct = self._format_percentage(c.get("error_pct", 0))
            lines.append(f"| {period} | {predicted} | {actual} | {error} | {error_pct} |")

        return "\n".join(lines)

    def _render_md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        ts = self._format_date(self.generated_at)
        return f"---\n_Predictive Insights generated at {ts} | PACK-003 CSRD Enterprise_"

    # ------------------------------------------------------------------
    # HTML renderers
    # ------------------------------------------------------------------

    def _build_css(self) -> str:
        """Build inline CSS for predictive insights."""
        return """
:root {
    --primary: #1a56db; --primary-light: #e1effe; --success: #057a55;
    --warning: #e3a008; --danger: #e02424; --info: #1c64f2;
    --bg: #f9fafb; --card-bg: #fff; --text: #1f2937;
    --text-muted: #6b7280; --border: #e5e7eb;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --forecast-band: rgba(26, 86, 219, 0.1);
    --anomaly-color: #e02424;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); }
.insights-container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.insights-header { background: linear-gradient(135deg, #1a56db, #7c3aed);
    color: #fff; padding: 32px; border-radius: 12px; margin-bottom: 24px; }
.insights-header h1 { font-size: 28px; }
.insights-header .subtitle { opacity: 0.85; margin-top: 4px; font-size: 14px; }
.section { margin-bottom: 24px; background: var(--card-bg); border-radius: 10px;
    padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.section-title { font-size: 18px; font-weight: 600; color: var(--primary);
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid var(--primary); }
table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
th { background: var(--primary-light); color: var(--primary); padding: 10px 12px;
    text-align: left; font-size: 12px; font-weight: 600; }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
tr:hover { background: #f3f4f6; }
.summary-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 16px; }
.stat-card { border: 1px solid var(--border); border-radius: 8px; padding: 14px;
    text-align: center; }
.stat-card .stat-value { font-size: 24px; font-weight: 700; color: var(--primary); }
.stat-card .stat-label { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.confidence-band { display: flex; align-items: center; height: 20px; position: relative; }
.confidence-band .band-95 { background: rgba(26,86,219,0.15); height: 100%;
    position: absolute; border-radius: 4px; }
.confidence-band .band-80 { background: rgba(26,86,219,0.3); height: 100%;
    position: absolute; border-radius: 4px; }
.confidence-band .forecast-point { width: 8px; height: 8px; background: var(--primary);
    border-radius: 50%; position: absolute; top: 6px; z-index: 2; }
.bar-chart-row { display: flex; align-items: center; margin-bottom: 6px; }
.bar-chart-row .bar-label { width: 180px; font-size: 12px; text-align: right;
    padding-right: 10px; }
.bar-chart-row .bar { height: 22px; border-radius: 4px; min-width: 2px;
    transition: width 0.3s; }
.bar-chart-row .bar.positive { background: var(--danger); }
.bar-chart-row .bar.negative { background: var(--success); }
.bar-chart-row .bar-value { font-size: 11px; padding-left: 6px; color: var(--text-muted); }
.risk-cell { padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 600;
    display: inline-block; text-align: center; }
.risk-cell.very-high { background: #fde8e8; color: #e02424; }
.risk-cell.high { background: #feecdc; color: #d97706; }
.risk-cell.medium { background: #fef9c3; color: #92400e; }
.risk-cell.low { background: #d1fae5; color: #057a55; }
.risk-cell.very-low { background: #e1effe; color: #1a56db; }
.anomaly-marker { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 4px; }
.anomaly-marker.critical { background: var(--danger); }
.anomaly-marker.high { background: #f59e0b; }
.anomaly-marker.medium { background: var(--warning); }
.anomaly-marker.low { background: var(--info); }
.on-track { color: var(--success); font-weight: 600; }
.off-track { color: var(--danger); font-weight: 600; }
.histogram-bar { display: inline-block; height: 16px; background: var(--primary);
    border-radius: 2px; margin-right: 4px; vertical-align: middle; }
.footer { text-align: center; color: var(--text-muted); font-size: 12px;
    padding: 16px 0; margin-top: 24px; border-top: 1px solid var(--border); }
"""

    def _render_html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        title = self._escape_html(data.get("title", "Predictive Insights Report"))
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"insights-header\">\n"
            f"  <h1>{title}</h1>\n"
            f"  <div class=\"subtitle\">Generated: {ts}</div>\n"
            f"</div>"
        )

    def _render_html_forecast(self, data: Dict[str, Any]) -> str:
        """Render HTML emission forecast table with confidence bands."""
        forecasts: List[Dict[str, Any]] = data.get("forecasts", [])
        if not forecasts:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Emission Forecast</h2>\n"
                "  <p>No forecast data available.</p>\n</div>"
            )

        rows = ""
        for f in forecasts:
            period = f.get("period", "-")
            forecast = self._format_number(f.get("forecast", 0))
            lo95 = self._format_number(f.get("lower_95", 0))
            hi95 = self._format_number(f.get("upper_95", 0))
            lo80 = self._format_number(f.get("lower_80", 0))
            hi80 = self._format_number(f.get("upper_80", 0))
            rows += (
                f"<tr><td>{period}</td><td><strong>{forecast}</strong></td>"
                f"<td>{lo95}</td><td>{hi95}</td>"
                f"<td>{lo80}</td><td>{hi80}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Emission Forecast with Confidence Intervals</h2>\n"
            "  <table><thead><tr>"
            "<th>Period</th><th>Forecast (tCO2e)</th>"
            "<th>Lower 95% CI</th><th>Upper 95% CI</th>"
            "<th>Lower 80% CI</th><th>Upper 80% CI</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap-to-target analysis."""
        gap = data.get("gap_analysis", {})
        if not gap:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Gap-to-Target Analysis</h2>\n"
                "  <p>No gap analysis data available.</p>\n</div>"
            )

        summary_html = (
            "<div class=\"summary-stats\">\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{gap.get('base_year', '-')}</div>"
            f"<div class=\"stat-label\">Base Year</div></div>\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{gap.get('target_year', '-')}</div>"
            f"<div class=\"stat-label\">Target Year</div></div>\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{self._format_percentage(gap.get('reduction_target_pct', 0))}</div>"
            f"<div class=\"stat-label\">Reduction Target</div></div>\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{self._format_number(gap.get('current_gap_tco2e', 0))}</div>"
            f"<div class=\"stat-label\">Current Gap (tCO2e)</div></div>\n"
            "</div>\n"
        )

        trajectories = gap.get("trajectories", [])
        rows = ""
        for t in trajectories:
            on_track = t.get("on_track", False)
            track_cls = "on-track" if on_track else "off-track"
            track_text = "On Track" if on_track else "Off Track"
            rows += (
                f"<tr><td>{t.get('year', '-')}</td>"
                f"<td>{self._format_number(t.get('actual', 0))}</td>"
                f"<td>{self._format_number(t.get('target', 0))}</td>"
                f"<td>{self._format_number(t.get('gap', 0))}</td>"
                f"<td class=\"{track_cls}\">{track_text}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Gap-to-Target Analysis (Actual vs SBTi)</h2>\n"
            f"  {summary_html}\n"
            "  <table><thead><tr>"
            "<th>Year</th><th>Actual (tCO2e)</th><th>Target (tCO2e)</th>"
            "<th>Gap (tCO2e)</th><th>Status</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_anomalies(self, data: Dict[str, Any]) -> str:
        """Render HTML anomaly detection timeline."""
        anomalies: List[Dict[str, Any]] = data.get("anomalies", [])
        if not anomalies:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Anomaly Detection</h2>\n"
                "  <p>No anomalies detected.</p>\n</div>"
            )

        rows = ""
        for a in anomalies:
            severity = a.get("severity", "medium")
            marker = f"<span class=\"anomaly-marker {severity}\"></span>"
            rows += (
                f"<tr><td>{a.get('date', '-')}</td>"
                f"<td>{a.get('category', '-')}</td>"
                f"<td>{self._format_number(a.get('value', 0))}</td>"
                f"<td>{self._format_number(a.get('expected', 0))}</td>"
                f"<td>{self._format_percentage(a.get('deviation_pct', 0))}</td>"
                f"<td>{marker}{severity.upper()}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Anomaly Detection Timeline</h2>\n"
            "  <table><thead><tr>"
            "<th>Date</th><th>Category</th><th>Value (tCO2e)</th>"
            "<th>Expected</th><th>Deviation</th><th>Severity</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_feature_importance(self, data: Dict[str, Any]) -> str:
        """Render HTML feature importance bar chart."""
        features: List[Dict[str, Any]] = data.get("feature_importance", [])
        if not features:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Feature Importance</h2>\n"
                "  <p>No feature importance data available.</p>\n</div>"
            )

        sorted_features = sorted(
            features, key=lambda x: abs(x.get("importance", 0)), reverse=True
        )
        max_importance = max(
            (abs(f.get("importance", 0)) for f in sorted_features), default=1
        )

        bars = ""
        for f in sorted_features:
            name = self._escape_html(f.get("feature", "-"))
            imp = f.get("importance", 0)
            abs_imp = abs(imp)
            width_pct = (abs_imp / max_importance * 100) if max_importance else 0
            bar_cls = "positive" if imp > 0 else "negative"
            bars += (
                f"<div class=\"bar-chart-row\">\n"
                f"  <div class=\"bar-label\">{name}</div>\n"
                f"  <div class=\"bar {bar_cls}\" style=\"width:{width_pct:.1f}%\"></div>\n"
                f"  <span class=\"bar-value\">{self._format_number(abs_imp, 4)}</span>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Feature Importance (SHAP-style)</h2>\n"
            f"  {bars}\n"
            "</div>"
        )

    def _render_html_monte_carlo(self, data: Dict[str, Any]) -> str:
        """Render HTML Monte Carlo distribution."""
        mc = data.get("monte_carlo", {})
        if not mc:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Monte Carlo Simulation</h2>\n"
                "  <p>No simulation data available.</p>\n</div>"
            )

        stats_html = (
            "<div class=\"summary-stats\">\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{self._format_number(mc.get('n_simulations', 0), 0)}</div>"
            f"<div class=\"stat-label\">Simulations</div></div>\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{self._format_number(mc.get('mean', 0))}</div>"
            f"<div class=\"stat-label\">Mean (tCO2e)</div></div>\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{self._format_number(mc.get('median', 0))}</div>"
            f"<div class=\"stat-label\">Median (tCO2e)</div></div>\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{self._format_number(mc.get('std_dev', 0))}</div>"
            f"<div class=\"stat-label\">Std Dev</div></div>\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{self._format_number(mc.get('p5', 0))}</div>"
            f"<div class=\"stat-label\">P5</div></div>\n"
            f"  <div class=\"stat-card\"><div class=\"stat-value\">"
            f"{self._format_number(mc.get('p95', 0))}</div>"
            f"<div class=\"stat-label\">P95</div></div>\n"
            "</div>\n"
        )

        bins = mc.get("histogram_bins", [])
        max_freq = max((b.get("frequency", 0) for b in bins), default=1)
        hist_rows = ""
        for b in bins:
            lo = self._format_number(b.get("lower", 0))
            hi = self._format_number(b.get("upper", 0))
            freq = b.get("frequency", 0)
            pct = self._format_percentage(b.get("relative_pct", 0))
            bar_width = (freq / max_freq * 200) if max_freq else 0
            hist_rows += (
                f"<tr><td>{lo} - {hi}</td><td>{freq}</td><td>"
                f"<span class=\"histogram-bar\" style=\"width:{bar_width:.0f}px\"></span>"
                f" {pct}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Monte Carlo Distribution</h2>\n"
            f"  {stats_html}\n"
            "  <table><thead><tr>"
            "<th>Bin Range (tCO2e)</th><th>Frequency</th><th>Distribution</th>"
            "</tr></thead>\n"
            f"  <tbody>{hist_rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_risk_matrix(self, data: Dict[str, Any]) -> str:
        """Render HTML risk category matrix."""
        matrix: List[Dict[str, Any]] = data.get("risk_matrix", [])
        if not matrix:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Risk Category Matrix</h2>\n"
                "  <p>No risk matrix data available.</p>\n</div>"
            )

        rows = ""
        for r in matrix:
            likelihood = r.get("likelihood", "-")
            likelihood_cls = likelihood.lower().replace(" ", "-")
            impact = r.get("impact", "-")
            impact_cls = impact.lower().replace(" ", "-")
            rows += (
                f"<tr><td>{r.get('category', '-')}</td>"
                f"<td><span class=\"risk-cell {likelihood_cls}\">{likelihood}</span></td>"
                f"<td><span class=\"risk-cell {impact_cls}\">{impact}</span></td>"
                f"<td>{self._format_number(r.get('risk_score', 0), 1)}</td>"
                f"<td>{self._escape_html(r.get('mitigation', '-'))}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Risk Category Matrix</h2>\n"
            "  <table><thead><tr>"
            "<th>Category</th><th>Likelihood</th><th>Impact</th>"
            "<th>Risk Score</th><th>Mitigation</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_model_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML model performance metrics."""
        metrics = data.get("model_metrics", {})
        models: List[Dict[str, Any]] = metrics.get("models", []) if metrics else []
        if not models:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Model Performance</h2>\n"
                "  <p>No model metrics available.</p>\n</div>"
            )

        rows = ""
        for m in models:
            rows += (
                f"<tr><td><strong>{self._escape_html(m.get('name', '-'))}</strong></td>"
                f"<td>{self._format_number(m.get('r_squared', 0), 4)}</td>"
                f"<td>{self._format_number(m.get('mae', 0))}</td>"
                f"<td>{self._format_number(m.get('rmse', 0))}</td>"
                f"<td>{self._format_percentage(m.get('mape', 0))}</td>"
                f"<td>{self._format_number(m.get('training_samples', 0), 0)}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Model Performance Metrics</h2>\n"
            "  <table><thead><tr>"
            "<th>Model</th><th>R-Squared</th><th>MAE</th>"
            "<th>RMSE</th><th>MAPE</th><th>Samples</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_prediction_vs_actual(self, data: Dict[str, Any]) -> str:
        """Render HTML prediction vs actual comparison."""
        comparisons: List[Dict[str, Any]] = data.get("prediction_vs_actual", [])
        if not comparisons:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Prediction vs Actual</h2>\n"
                "  <p>No comparison data available.</p>\n</div>"
            )

        rows = ""
        for c in comparisons:
            error = c.get("error", 0)
            error_cls = "on-track" if abs(c.get("error_pct", 0)) < 5 else "off-track"
            rows += (
                f"<tr><td>{c.get('period', '-')}</td>"
                f"<td>{self._format_number(c.get('predicted', 0))}</td>"
                f"<td>{self._format_number(c.get('actual', 0))}</td>"
                f"<td class=\"{error_cls}\">{self._format_number(error)}</td>"
                f"<td class=\"{error_cls}\">"
                f"{self._format_percentage(c.get('error_pct', 0))}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Prediction vs Actual Comparison</h2>\n"
            "  <table><thead><tr>"
            "<th>Period</th><th>Predicted (tCO2e)</th><th>Actual (tCO2e)</th>"
            "<th>Error (tCO2e)</th><th>Error (%)</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"footer\">"
            f"Predictive Insights generated at {ts} | PACK-003 CSRD Enterprise"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _build_json_forecast(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON forecast section."""
        return data.get("forecasts", [])

    def _build_json_gap_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON gap analysis section."""
        return data.get("gap_analysis", {})

    def _build_json_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON anomalies section."""
        return data.get("anomalies", [])

    def _build_json_feature_importance(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build JSON feature importance section."""
        features = data.get("feature_importance", [])
        return sorted(
            features, key=lambda x: abs(x.get("importance", 0)), reverse=True
        )

    def _build_json_monte_carlo(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON Monte Carlo section."""
        return data.get("monte_carlo", {})

    def _build_json_risk_matrix(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON risk matrix section."""
        return data.get("risk_matrix", [])

    def _build_json_model_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON model metrics section."""
        return data.get("model_metrics", {})

    def _build_json_prediction_vs_actual(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build JSON prediction vs actual section."""
        return data.get("prediction_vs_actual", [])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_provenance_hash(content: str) -> str:
        """Generate SHA-256 provenance hash.

        Args:
            content: Content to hash.

        Returns:
            Hexadecimal SHA-256 hash.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_number(value: Union[int, float], decimals: int = 2) -> str:
        """Format numeric value with thousands separator.

        Args:
            value: Numeric value.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _format_percentage(value: Union[int, float]) -> str:
        """Format value as percentage.

        Args:
            value: Numeric value.

        Returns:
            Percentage string.
        """
        return f"{value:.1f}%"

    @staticmethod
    def _format_date(dt: Optional[datetime]) -> str:
        """Format datetime as string.

        Args:
            dt: Datetime object.

        Returns:
            Formatted date string.
        """
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Raw text.

        Returns:
            HTML-safe string.
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    @staticmethod
    def _text_bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
        """Create a text-based bar for Markdown tables.

        Args:
            value: Current value.
            max_val: Maximum value for scaling.
            width: Bar width in characters.

        Returns:
            Text bar string.
        """
        filled = int((value / max_val) * width) if max_val else 0
        return "|" + "=" * filled + " " * (width - filled) + "|"
