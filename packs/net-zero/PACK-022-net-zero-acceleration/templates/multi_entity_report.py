# -*- coding: utf-8 -*-
"""
MultiEntityReportTemplate - Multi-entity consolidated emissions for PACK-022.

Renders a multi-entity consolidated emissions report with group summary,
consolidation method, entity hierarchy, entity-level emissions, intercompany
eliminations, completeness matrix, scope splits, target allocation vs
performance, structural changes, and base year recalculation notes.

Sections:
    1. Group Summary
    2. Consolidation Method
    3. Entity Hierarchy Diagram Data
    4. Entity-Level Emissions
    5. Intercompany Eliminations
    6. Completeness Matrix
    7. Scope Split by Entity
    8. Target Allocation vs Performance
    9. Structural Changes (M&A/divestiture)
   10. Base Year Recalculation Notes

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)


def _pct_of(part: Any, total: Any) -> Decimal:
    try:
        p = Decimal(str(part))
        t = Decimal(str(total))
        if t == 0:
            return Decimal("0.00")
        return (p / t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception:
        return Decimal("0.00")


class MultiEntityReportTemplate:
    """
    Multi-entity consolidated emissions report template.

    Consolidates emissions across group entities with hierarchy,
    intercompany eliminations, completeness tracking, target allocation,
    and structural change adjustments.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_group_summary(data),
            self._md_consolidation_method(data),
            self._md_entity_hierarchy(data),
            self._md_entity_emissions(data),
            self._md_intercompany(data),
            self._md_completeness_matrix(data),
            self._md_scope_split(data),
            self._md_target_allocation(data),
            self._md_structural_changes(data),
            self._md_base_year_recalc(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_group_summary(data),
            self._html_consolidation_method(data),
            self._html_entity_hierarchy(data),
            self._html_entity_emissions(data),
            self._html_intercompany(data),
            self._html_completeness_matrix(data),
            self._html_scope_split(data),
            self._html_target_allocation(data),
            self._html_structural_changes(data),
            self._html_base_year_recalc(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Multi-Entity Consolidated Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        entities = data.get("entities", [])
        total = sum(Decimal(str(e.get("total_tco2e", 0))) for e in entities)

        result: Dict[str, Any] = {
            "template": "multi_entity_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "group_summary": {
                "entity_count": len(entities),
                "total_consolidated_tco2e": str(total),
                "consolidation_method": data.get("consolidation_method", {}).get("method", ""),
            },
            "consolidation_method": data.get("consolidation_method", {}),
            "hierarchy": data.get("hierarchy", []),
            "entities": entities,
            "intercompany_eliminations": data.get("intercompany_eliminations", []),
            "completeness_matrix": data.get("completeness_matrix", []),
            "scope_split": data.get("scope_split", []),
            "target_allocation": data.get("target_allocation", []),
            "structural_changes": data.get("structural_changes", []),
            "base_year_recalculation": data.get("base_year_recalculation", {}),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Multi-Entity Consolidated Emissions Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_group_summary(self, data: Dict[str, Any]) -> str:
        entities = data.get("entities", [])
        total = sum(Decimal(str(e.get("total_tco2e", 0))) for e in entities)
        elim = sum(Decimal(str(e.get("eliminated_tco2e", 0))) for e in data.get("intercompany_eliminations", []))
        net = total - elim
        method = data.get("consolidation_method", {}).get("method", "Operational Control")
        return (
            "## 1. Group Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Number of Entities | {len(entities)} |\n"
            f"| Consolidation Method | {method} |\n"
            f"| Gross Consolidated Emissions | {_dec_comma(total)} tCO2e |\n"
            f"| Intercompany Eliminations | {_dec_comma(elim)} tCO2e |\n"
            f"| Net Consolidated Emissions | {_dec_comma(net)} tCO2e |"
        )

    def _md_consolidation_method(self, data: Dict[str, Any]) -> str:
        cm = data.get("consolidation_method", {})
        return (
            "## 2. Consolidation Method\n\n"
            f"- **Method:** {cm.get('method', 'Operational Control')}\n"
            f"- **Standard:** {cm.get('standard', 'GHG Protocol Corporate Standard')}\n"
            f"- **Rationale:** {cm.get('rationale', 'N/A')}\n"
            f"- **Equity Threshold:** {_dec(cm.get('equity_threshold_pct', 0))}%\n"
            f"- **Joint Ventures Treatment:** {cm.get('jv_treatment', 'Proportional')}"
        )

    def _md_entity_hierarchy(self, data: Dict[str, Any]) -> str:
        hierarchy = data.get("hierarchy", [])
        lines = [
            "## 3. Entity Hierarchy\n",
            "| Entity | Parent | Level | Ownership (%) | Control Type | Country |",
            "|--------|--------|:-----:|:-------------:|:------------:|---------|",
        ]
        for h in hierarchy:
            lines.append(
                f"| {h.get('name', '-')} | {h.get('parent', '-')} "
                f"| {h.get('level', 0)} "
                f"| {_dec(h.get('ownership_pct', 100))}% "
                f"| {h.get('control_type', '-')} "
                f"| {h.get('country', '-')} |"
            )
        if not hierarchy:
            lines.append("| _No hierarchy data_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_entity_emissions(self, data: Dict[str, Any]) -> str:
        entities = data.get("entities", [])
        total = sum(Decimal(str(e.get("total_tco2e", 0))) for e in entities)
        lines = [
            "## 4. Entity-Level Emissions\n",
            "| Entity | Scope 1 (tCO2e) | Scope 2 (tCO2e) | Scope 3 (tCO2e) | Total (tCO2e) | Share (%) |",
            "|--------|----------------:|----------------:|----------------:|--------------:|:---------:|",
        ]
        for e in entities:
            ent_total = Decimal(str(e.get("total_tco2e", 0)))
            lines.append(
                f"| {e.get('name', '-')} "
                f"| {_dec_comma(e.get('scope1_tco2e', 0))} "
                f"| {_dec_comma(e.get('scope2_tco2e', 0))} "
                f"| {_dec_comma(e.get('scope3_tco2e', 0))} "
                f"| {_dec_comma(ent_total)} "
                f"| {_dec(_pct_of(ent_total, total))}% |"
            )
        if entities:
            lines.append(f"| **Group Total** | | | | **{_dec_comma(total)}** | **100.00%** |")
        else:
            lines.append("| _No entity data_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_intercompany(self, data: Dict[str, Any]) -> str:
        eliminations = data.get("intercompany_eliminations", [])
        lines = [
            "## 5. Intercompany Eliminations\n",
            "| From Entity | To Entity | Type | Eliminated (tCO2e) | Reason |",
            "|-------------|-----------|------|-------------------:|--------|",
        ]
        for el in eliminations:
            lines.append(
                f"| {el.get('from_entity', '-')} | {el.get('to_entity', '-')} "
                f"| {el.get('type', '-')} "
                f"| {_dec_comma(el.get('eliminated_tco2e', 0))} "
                f"| {el.get('reason', '-')} |"
            )
        if not eliminations:
            lines.append("| _No eliminations_ | - | - | - | - |")
        total_elim = sum(Decimal(str(e.get("eliminated_tco2e", 0))) for e in eliminations)
        if eliminations:
            lines.append(f"\n**Total Eliminations:** {_dec_comma(total_elim)} tCO2e")
        return "\n".join(lines)

    def _md_completeness_matrix(self, data: Dict[str, Any]) -> str:
        matrix = data.get("completeness_matrix", [])
        lines = [
            "## 6. Completeness Matrix\n",
            "| Entity | Scope 1 | Scope 2 | Scope 3 | Data Quality | Reporting Period |",
            "|--------|:-------:|:-------:|:-------:|:------------:|:----------------:|",
        ]
        for row in matrix:
            s1 = "Complete" if row.get("scope1_complete", False) else "Incomplete"
            s2 = "Complete" if row.get("scope2_complete", False) else "Incomplete"
            s3 = "Complete" if row.get("scope3_complete", False) else "Incomplete"
            lines.append(
                f"| {row.get('entity', '-')} "
                f"| {s1} | {s2} | {s3} "
                f"| {row.get('data_quality', '-')} "
                f"| {row.get('reporting_period', '-')} |"
            )
        if not matrix:
            lines.append("| _No completeness data_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_scope_split(self, data: Dict[str, Any]) -> str:
        splits = data.get("scope_split", [])
        lines = [
            "## 7. Scope Split by Entity\n",
            "| Entity | Scope 1 (%) | Scope 2 (%) | Scope 3 (%) |",
            "|--------|:----------:|:----------:|:----------:|",
        ]
        for s in splits:
            lines.append(
                f"| {s.get('entity', '-')} "
                f"| {_dec(s.get('scope1_pct', 0))}% "
                f"| {_dec(s.get('scope2_pct', 0))}% "
                f"| {_dec(s.get('scope3_pct', 0))}% |"
            )
        if not splits:
            lines.append("| _No scope split data_ | - | - | - |")
        return "\n".join(lines)

    def _md_target_allocation(self, data: Dict[str, Any]) -> str:
        allocations = data.get("target_allocation", [])
        lines = [
            "## 8. Target Allocation vs Performance\n",
            "| Entity | Allocated Target (tCO2e) | Actual (tCO2e) | Variance (tCO2e) | On Track |",
            "|--------|--------------------------:|---------------:|-----------------:|:--------:|",
        ]
        for a in allocations:
            target = Decimal(str(a.get("target_tco2e", 0)))
            actual = Decimal(str(a.get("actual_tco2e", 0)))
            variance = actual - target
            on_track = "YES" if variance <= 0 else "NO"
            lines.append(
                f"| {a.get('entity', '-')} "
                f"| {_dec_comma(target)} "
                f"| {_dec_comma(actual)} "
                f"| {_dec_comma(variance)} "
                f"| {on_track} |"
            )
        if not allocations:
            lines.append("| _No allocation data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_structural_changes(self, data: Dict[str, Any]) -> str:
        changes = data.get("structural_changes", [])
        lines = [
            "## 9. Structural Changes (M&A / Divestiture)\n",
            "| Date | Entity | Change Type | Impact (tCO2e) | Ownership Change (%) | Treatment |",
            "|------|--------|-------------|:--------------:|:--------------------:|-----------|",
        ]
        for ch in changes:
            lines.append(
                f"| {ch.get('date', '-')} | {ch.get('entity', '-')} "
                f"| {ch.get('change_type', '-')} "
                f"| {_dec_comma(ch.get('impact_tco2e', 0))} "
                f"| {_dec(ch.get('ownership_change_pct', 0))}% "
                f"| {ch.get('treatment', '-')} |"
            )
        if not changes:
            lines.append("| _No structural changes_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_base_year_recalc(self, data: Dict[str, Any]) -> str:
        recalc = data.get("base_year_recalculation", {})
        triggers = recalc.get("triggers", [])
        lines = [
            "## 10. Base Year Recalculation Notes\n",
            f"**Base Year:** {recalc.get('base_year', 'N/A')}  \n"
            f"**Recalculation Required:** {'Yes' if recalc.get('required', False) else 'No'}  \n"
            f"**Significance Threshold:** {_dec(recalc.get('threshold_pct', 5))}%  \n"
            f"**Policy:** {recalc.get('policy', 'GHG Protocol base year recalculation guidance')}\n",
        ]
        if triggers:
            lines.append("| Trigger | Impact (tCO2e) | Significance (%) | Recalculate |")
            lines.append("|---------|:--------------:|:----------------:|:-----------:|")
            for t in triggers:
                impact = Decimal(str(t.get("impact_tco2e", 0)))
                sig = Decimal(str(t.get("significance_pct", 0)))
                recalc_needed = "Yes" if t.get("recalculate", False) else "No"
                lines.append(
                    f"| {t.get('trigger', '-')} "
                    f"| {_dec_comma(impact)} "
                    f"| {_dec(sig)}% "
                    f"| {recalc_needed} |"
                )
        else:
            lines.append("_No recalculation triggers identified._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*Consolidation per GHG Protocol Corporate Standard.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".complete{color:#1b5e20;font-weight:600;}"
            ".incomplete{color:#c62828;font-weight:600;}"
            ".on-track{color:#1b5e20;font-weight:600;}"
            ".off-track{color:#c62828;font-weight:600;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Multi-Entity Consolidated Emissions Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_group_summary(self, data: Dict[str, Any]) -> str:
        entities = data.get("entities", [])
        total = sum(Decimal(str(e.get("total_tco2e", 0))) for e in entities)
        elim = sum(Decimal(str(e.get("eliminated_tco2e", 0))) for e in data.get("intercompany_eliminations", []))
        net = total - elim
        return (
            f'<h2>1. Group Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Entities</div>'
            f'<div class="card-value">{len(entities)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Gross Emissions</div>'
            f'<div class="card-value">{_dec_comma(total)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Eliminations</div>'
            f'<div class="card-value">{_dec_comma(elim)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Net Consolidated</div>'
            f'<div class="card-value">{_dec_comma(net)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_consolidation_method(self, data: Dict[str, Any]) -> str:
        cm = data.get("consolidation_method", {})
        return (
            f'<h2>2. Consolidation Method</h2>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Method</td><td>{cm.get("method", "Operational Control")}</td></tr>\n'
            f'<tr><td>Standard</td><td>{cm.get("standard", "GHG Protocol")}</td></tr>\n'
            f'<tr><td>Rationale</td><td>{cm.get("rationale", "N/A")}</td></tr>\n'
            f'<tr><td>Equity Threshold</td><td>{_dec(cm.get("equity_threshold_pct", 0))}%</td></tr>\n'
            f'<tr><td>JV Treatment</td><td>{cm.get("jv_treatment", "Proportional")}</td></tr>\n'
            f'</table>'
        )

    def _html_entity_hierarchy(self, data: Dict[str, Any]) -> str:
        hierarchy = data.get("hierarchy", [])
        rows = ""
        for h in hierarchy:
            indent = "&nbsp;" * (h.get("level", 0) * 4)
            rows += (
                f'<tr><td>{indent}{h.get("name", "-")}</td>'
                f'<td>{h.get("parent", "-")}</td>'
                f'<td>{h.get("level", 0)}</td>'
                f'<td>{_dec(h.get("ownership_pct", 100))}%</td>'
                f'<td>{h.get("control_type", "-")}</td>'
                f'<td>{h.get("country", "-")}</td></tr>\n'
            )
        return (
            f'<h2>3. Entity Hierarchy</h2>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Parent</th><th>Level</th>'
            f'<th>Ownership</th><th>Control</th><th>Country</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_entity_emissions(self, data: Dict[str, Any]) -> str:
        entities = data.get("entities", [])
        total = sum(Decimal(str(e.get("total_tco2e", 0))) for e in entities)
        rows = ""
        for e in entities:
            ent_total = Decimal(str(e.get("total_tco2e", 0)))
            rows += (
                f'<tr><td><strong>{e.get("name", "-")}</strong></td>'
                f'<td>{_dec_comma(e.get("scope1_tco2e", 0))}</td>'
                f'<td>{_dec_comma(e.get("scope2_tco2e", 0))}</td>'
                f'<td>{_dec_comma(e.get("scope3_tco2e", 0))}</td>'
                f'<td><strong>{_dec_comma(ent_total)}</strong></td>'
                f'<td>{_dec(_pct_of(ent_total, total))}%</td></tr>\n'
            )
        return (
            f'<h2>4. Entity-Level Emissions</h2>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Scope 1</th><th>Scope 2</th>'
            f'<th>Scope 3</th><th>Total (tCO2e)</th><th>Share</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_intercompany(self, data: Dict[str, Any]) -> str:
        eliminations = data.get("intercompany_eliminations", [])
        rows = ""
        for el in eliminations:
            rows += (
                f'<tr><td>{el.get("from_entity", "-")}</td>'
                f'<td>{el.get("to_entity", "-")}</td>'
                f'<td>{el.get("type", "-")}</td>'
                f'<td>{_dec_comma(el.get("eliminated_tco2e", 0))}</td>'
                f'<td>{el.get("reason", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. Intercompany Eliminations</h2>\n'
            f'<table>\n'
            f'<tr><th>From</th><th>To</th><th>Type</th>'
            f'<th>Eliminated (tCO2e)</th><th>Reason</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_completeness_matrix(self, data: Dict[str, Any]) -> str:
        matrix = data.get("completeness_matrix", [])
        rows = ""
        for row in matrix:
            s1 = row.get("scope1_complete", False)
            s2 = row.get("scope2_complete", False)
            s3 = row.get("scope3_complete", False)
            rows += (
                f'<tr><td>{row.get("entity", "-")}</td>'
                f'<td class="{"complete" if s1 else "incomplete"}">{"Complete" if s1 else "Incomplete"}</td>'
                f'<td class="{"complete" if s2 else "incomplete"}">{"Complete" if s2 else "Incomplete"}</td>'
                f'<td class="{"complete" if s3 else "incomplete"}">{"Complete" if s3 else "Incomplete"}</td>'
                f'<td>{row.get("data_quality", "-")}</td>'
                f'<td>{row.get("reporting_period", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. Completeness Matrix</h2>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Scope 1</th><th>Scope 2</th>'
            f'<th>Scope 3</th><th>Data Quality</th><th>Period</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope_split(self, data: Dict[str, Any]) -> str:
        splits = data.get("scope_split", [])
        rows = ""
        for s in splits:
            rows += (
                f'<tr><td>{s.get("entity", "-")}</td>'
                f'<td>{_dec(s.get("scope1_pct", 0))}%</td>'
                f'<td>{_dec(s.get("scope2_pct", 0))}%</td>'
                f'<td>{_dec(s.get("scope3_pct", 0))}%</td></tr>\n'
            )
        return (
            f'<h2>7. Scope Split by Entity</h2>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Scope 1 (%)</th><th>Scope 2 (%)</th>'
            f'<th>Scope 3 (%)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_target_allocation(self, data: Dict[str, Any]) -> str:
        allocations = data.get("target_allocation", [])
        rows = ""
        for a in allocations:
            target = Decimal(str(a.get("target_tco2e", 0)))
            actual = Decimal(str(a.get("actual_tco2e", 0)))
            variance = actual - target
            on_track = variance <= 0
            cls = "on-track" if on_track else "off-track"
            label = "On Track" if on_track else "Off Track"
            rows += (
                f'<tr><td>{a.get("entity", "-")}</td>'
                f'<td>{_dec_comma(target)}</td>'
                f'<td>{_dec_comma(actual)}</td>'
                f'<td>{_dec_comma(variance)}</td>'
                f'<td class="{cls}">{label}</td></tr>\n'
            )
        return (
            f'<h2>8. Target Allocation vs Performance</h2>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Target (tCO2e)</th><th>Actual (tCO2e)</th>'
            f'<th>Variance</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_structural_changes(self, data: Dict[str, Any]) -> str:
        changes = data.get("structural_changes", [])
        rows = ""
        for ch in changes:
            rows += (
                f'<tr><td>{ch.get("date", "-")}</td>'
                f'<td>{ch.get("entity", "-")}</td>'
                f'<td>{ch.get("change_type", "-")}</td>'
                f'<td>{_dec_comma(ch.get("impact_tco2e", 0))}</td>'
                f'<td>{_dec(ch.get("ownership_change_pct", 0))}%</td>'
                f'<td>{ch.get("treatment", "-")}</td></tr>\n'
            )
        return (
            f'<h2>9. Structural Changes</h2>\n'
            f'<table>\n'
            f'<tr><th>Date</th><th>Entity</th><th>Change</th>'
            f'<th>Impact (tCO2e)</th><th>Ownership Change</th><th>Treatment</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_base_year_recalc(self, data: Dict[str, Any]) -> str:
        recalc = data.get("base_year_recalculation", {})
        triggers = recalc.get("triggers", [])
        rows = ""
        for t in triggers:
            recalc_needed = t.get("recalculate", False)
            cls = "off-track" if recalc_needed else "on-track"
            label = "Yes" if recalc_needed else "No"
            rows += (
                f'<tr><td>{t.get("trigger", "-")}</td>'
                f'<td>{_dec_comma(t.get("impact_tco2e", 0))}</td>'
                f'<td>{_dec(t.get("significance_pct", 0))}%</td>'
                f'<td class="{cls}">{label}</td></tr>\n'
            )
        required = recalc.get("required", False)
        req_cls = "off-track" if required else "on-track"
        return (
            f'<h2>10. Base Year Recalculation</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Base Year</div>'
            f'<div class="card-value">{recalc.get("base_year", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Recalculation Required</div>'
            f'<div class="card-value {req_cls}">{"Yes" if required else "No"}</div></div>\n'
            f'  <div class="card"><div class="card-label">Threshold</div>'
            f'<div class="card-value">{_dec(recalc.get("threshold_pct", 5))}%</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Trigger</th><th>Impact (tCO2e)</th><th>Significance</th>'
            f'<th>Recalculate</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'Consolidation per GHG Protocol Corporate Standard.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
