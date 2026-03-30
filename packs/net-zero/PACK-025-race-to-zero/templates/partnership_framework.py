# -*- coding: utf-8 -*-
"""
PartnershipFrameworkTemplate - Race to Zero partnership framework for PACK-025.

Renders the partnership framework document covering partner organization
profiles, joint reduction commitments, collaboration governance, data sharing
protocols, performance tracking, and accountability framework.

Sections:
    1. Partnership Overview
    2. Partner Organization Profiles
    3. Joint Reduction Commitments
    4. Collaboration Governance Structure
    5. Data Sharing Protocols
    6. Performance Tracking Dashboard
    7. Accountability Framework

Author: GreenLang Team
Version: 25.0.0
Pack: PACK-025 Race to Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"
_PACK_ID = "PACK-025"
_TEMPLATE_ID = "partnership_framework"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(r).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg:
            ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0:
                f = "," + f
            f = ch + f
        if neg:
            f = "-" + f
        if len(parts) > 1:
            f += "." + parts[1]
        return f
    except Exception:
        return str(val)

def _pct(val: Any) -> str:
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)

def _safe_div(n: Any, d: Any) -> float:
    try:
        dv = float(d)
        return float(n) / dv if dv != 0 else 0.0
    except Exception:
        return 0.0

class PartnershipFrameworkTemplate:
    """Race to Zero partnership framework template for PACK-025.

    Generates partnership framework documents with partner profiles,
    joint commitments, governance structures, data protocols,
    performance dashboards, and accountability mechanisms.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the partnership framework as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_partner_profiles(data),
            self._md_joint_commitments(data),
            self._md_governance(data),
            self._md_data_sharing(data),
            self._md_performance_tracking(data),
            self._md_accountability(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the partnership framework as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_partners(data),
            self._html_commitments(data),
            self._html_governance(data),
            self._html_data_sharing(data),
            self._html_performance(data),
            self._html_accountability(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Race to Zero - Partnership Framework</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the partnership framework as structured JSON."""
        self.generated_at = utcnow()
        partners = data.get("partners", [])
        commitments = data.get("joint_commitments", [])
        total_combined = sum(p.get("emissions_tco2e", 0) for p in partners)
        total_reduction = sum(c.get("reduction_tco2e", 0) for c in commitments)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "partnership_name": data.get("partnership_name", ""),
            "partners": partners,
            "partner_count": len(partners),
            "combined_emissions_tco2e": total_combined,
            "joint_commitments": commitments,
            "total_joint_reduction_tco2e": total_reduction,
            "governance": data.get("governance", {}),
            "data_sharing": data.get("data_sharing", {}),
            "kpis": data.get("kpis", []),
            "accountability": data.get("accountability", {}),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data for Excel/openpyxl export."""
        self.generated_at = utcnow()
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        # Sheet 1: Partner Profiles
        partners = data.get("partners", [])
        p_rows: List[Dict[str, Any]] = []
        for p in partners:
            p_rows.append({
                "Organization": p.get("name", ""),
                "Sector": p.get("sector", ""),
                "Country": p.get("country", ""),
                "Emissions (tCO2e)": p.get("emissions_tco2e", 0),
                "Target Year": p.get("target_year", ""),
                "SBTi": "Yes" if p.get("sbti_validated") else "No",
                "Race to Zero": "Yes" if p.get("race_to_zero") else "No",
                "Role": p.get("role", ""),
            })
        sheets["Partner Profiles"] = p_rows

        # Sheet 2: Joint Commitments
        commitments = data.get("joint_commitments", [])
        c_rows: List[Dict[str, Any]] = []
        for c in commitments:
            c_rows.append({
                "Commitment": c.get("commitment", ""),
                "Partners": c.get("partners", ""),
                "Baseline (tCO2e)": c.get("baseline_tco2e", 0),
                "Target Reduction (tCO2e)": c.get("reduction_tco2e", 0),
                "Target Year": c.get("target_year", ""),
                "Status": c.get("status", "Active"),
            })
        sheets["Joint Commitments"] = c_rows

        # Sheet 3: Performance KPIs
        kpis = data.get("kpis", [])
        kpi_rows: List[Dict[str, Any]] = []
        for kpi in kpis:
            kpi_rows.append({
                "KPI": kpi.get("name", ""),
                "Target": kpi.get("target", ""),
                "Current": kpi.get("current", ""),
                "Status": kpi.get("status", ""),
                "Trend": kpi.get("trend", ""),
            })
        sheets["Performance KPIs"] = kpi_rows

        # Sheet 4: Data Sharing Protocol
        protocols = data.get("data_sharing", {}).get("protocols", [])
        ds_rows: List[Dict[str, Any]] = []
        for proto in protocols:
            ds_rows.append({
                "Data Type": proto.get("data_type", ""),
                "Frequency": proto.get("frequency", ""),
                "Format": proto.get("format", ""),
                "Classification": proto.get("classification", ""),
                "Platform": proto.get("platform", ""),
            })
        sheets["Data Sharing"] = ds_rows

        return sheets

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        partnership = data.get("partnership_name", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Race to Zero -- Partnership Framework\n\n"
            f"**Lead Organization:** {org}  \n"
            f"**Partnership:** {partnership}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        partners = data.get("partners", [])
        commitments = data.get("joint_commitments", [])
        total_emissions = sum(p.get("emissions_tco2e", 0) for p in partners)
        total_reduction = sum(c.get("reduction_tco2e", 0) for c in commitments)
        partnership = data.get("partnership_name", "Climate Action Partnership")

        return (
            f"## 1. Partnership Overview\n\n"
            f"**{partnership}** brings together {len(partners)} organizations committed to "
            f"the Race to Zero campaign, with combined emissions of "
            f"{_dec_comma(total_emissions)} tCO2e and joint reduction commitments of "
            f"{_dec_comma(total_reduction)} tCO2e.\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Partner Count | {len(partners)} |\n"
            f"| Combined Emissions | {_dec_comma(total_emissions)} tCO2e |\n"
            f"| Joint Reduction Target | {_dec_comma(total_reduction)} tCO2e |\n"
            f"| Sectors Represented | {len(set(p.get('sector', '') for p in partners))} |\n"
            f"| Countries Represented | {len(set(p.get('country', '') for p in partners))} |\n"
            f"| SBTi Validated Partners | {sum(1 for p in partners if p.get('sbti_validated'))} |\n"
            f"| Partnership Established | {data.get('established_date', 'N/A')} |"
        )

    def _md_partner_profiles(self, data: Dict[str, Any]) -> str:
        partners = data.get("partners", [])
        lines = ["## 2. Partner Organization Profiles\n"]

        if partners:
            lines.extend([
                "| # | Organization | Sector | Country | Emissions (tCO2e) | Target | SBTi | R2Z |",
                "|---|-------------|--------|---------|------------------:|:------:|:----:|:---:|",
            ])
            for i, p in enumerate(partners, 1):
                lines.append(
                    f"| {i} | {p.get('name', '-')} "
                    f"| {p.get('sector', '-')} "
                    f"| {p.get('country', '-')} "
                    f"| {_dec_comma(p.get('emissions_tco2e', 0))} "
                    f"| {p.get('target_year', '-')} "
                    f"| {'Yes' if p.get('sbti_validated') else 'No'} "
                    f"| {'Yes' if p.get('race_to_zero') else 'No'} |"
                )

            # Partner detail blocks
            for p in partners:
                if p.get("description"):
                    lines.append(
                        f"\n### {p.get('name', '')}\n"
                        f"{p.get('description', '')}\n"
                        f"- **Role in Partnership:** {p.get('role', 'Participant')}\n"
                        f"- **Key Contribution:** {p.get('contribution', 'Collaborative reduction')}"
                    )
        else:
            lines.append("_Partner profiles to be populated._")

        return "\n".join(lines)

    def _md_joint_commitments(self, data: Dict[str, Any]) -> str:
        commitments = data.get("joint_commitments", [])
        lines = ["## 3. Joint Reduction Commitments\n"]

        if commitments:
            lines.extend([
                "| # | Commitment | Partners | Baseline (tCO2e) | Reduction (tCO2e) | Year | Status |",
                "|---|-----------|----------|-----------------:|-----------------:|:----:|:------:|",
            ])
            for i, c in enumerate(commitments, 1):
                lines.append(
                    f"| {i} | {c.get('commitment', '-')} "
                    f"| {c.get('partners', '-')} "
                    f"| {_dec_comma(c.get('baseline_tco2e', 0))} "
                    f"| {_dec_comma(c.get('reduction_tco2e', 0))} "
                    f"| {c.get('target_year', '-')} "
                    f"| {c.get('status', 'Active')} |"
                )
        else:
            lines.append("_Joint commitments to be defined._")

        return "\n".join(lines)

    def _md_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance", {})
        bodies = gov.get("bodies", [])

        lines = [
            "## 4. Collaboration Governance Structure\n",
            f"**Decision Model:** {gov.get('decision_model', 'Consensus-based')}\n"
            f"**Meeting Cadence:** {gov.get('meeting_cadence', 'Quarterly')}\n"
            f"**Secretariat:** {gov.get('secretariat', 'Lead organization')}\n",
        ]

        if bodies:
            lines.extend([
                "### Governance Bodies\n",
                "| Body | Role | Chair | Frequency | Members |",
                "|------|------|-------|:---------:|:-------:|",
            ])
            for body in bodies:
                lines.append(
                    f"| {body.get('name', '-')} | {body.get('role', '-')} "
                    f"| {body.get('chair', '-')} | {body.get('frequency', '-')} "
                    f"| {body.get('members', '-')} |"
                )
        else:
            lines.extend([
                "### Default Governance Structure\n",
                "| Level | Body | Frequency | Responsibility |",
                "|-------|------|:---------:|----------------|",
                "| Strategic | Steering Committee | Quarterly | Strategy, targets, resource allocation |",
                "| Operational | Working Group | Monthly | Project coordination, data sharing |",
                "| Technical | Expert Panel | As needed | Methodology review, best practices |",
            ])

        return "\n".join(lines)

    def _md_data_sharing(self, data: Dict[str, Any]) -> str:
        ds = data.get("data_sharing", {})
        protocols = ds.get("protocols", [])

        lines = [
            "## 5. Data Sharing Protocols\n",
            f"**Platform:** {ds.get('platform', 'GreenLang Platform')}\n"
            f"**Security Standard:** {ds.get('security', 'ISO 27001 / SOC 2')}\n"
            f"**Data Classification:** {ds.get('classification_framework', 'Confidential / Restricted / Public')}\n",
        ]

        if protocols:
            lines.extend([
                "### Data Sharing Schedule\n",
                "| Data Type | Frequency | Format | Classification | Sharing Level |",
                "|-----------|:---------:|--------|:--------------:|:-------------:|",
            ])
            for p in protocols:
                lines.append(
                    f"| {p.get('data_type', '-')} | {p.get('frequency', '-')} "
                    f"| {p.get('format', '-')} | {p.get('classification', '-')} "
                    f"| {p.get('sharing_level', '-')} |"
                )
        else:
            lines.extend([
                "### Default Data Sharing Requirements\n",
                "| Data | Frequency | Purpose |",
                "|------|:---------:|---------|",
                "| GHG Inventory (S1+S2+S3) | Annual | Joint progress tracking |",
                "| Reduction Project Status | Quarterly | Coordination and learning |",
                "| Best Practices | Ongoing | Knowledge sharing |",
                "| Verification Outcomes | Annual | Accountability and transparency |",
            ])

        return "\n".join(lines)

    def _md_performance_tracking(self, data: Dict[str, Any]) -> str:
        kpis = data.get("kpis", [])
        lines = ["## 6. Performance Tracking Dashboard\n"]

        if kpis:
            lines.extend([
                "| # | KPI | Target | Current | Status | Trend |",
                "|---|-----|--------|---------|:------:|:-----:|",
            ])
            for i, kpi in enumerate(kpis, 1):
                lines.append(
                    f"| {i} | {kpi.get('name', '-')} "
                    f"| {kpi.get('target', '-')} "
                    f"| {kpi.get('current', '-')} "
                    f"| {kpi.get('status', '-')} "
                    f"| {kpi.get('trend', '-')} |"
                )
        else:
            lines.extend([
                "### Key Performance Indicators\n",
                "| KPI | Description | Measurement |",
                "|-----|-------------|-------------|",
                "| Combined Emissions Reduction | Total YoY reduction across partners | tCO2e |",
                "| Target Alignment Rate | % of partners on track with targets | % |",
                "| Data Sharing Compliance | % of partners meeting data deadlines | % |",
                "| Joint Project Impact | Emissions reduced through joint projects | tCO2e |",
                "| Engagement Score | Active participation in governance | Score/10 |",
            ])

        return "\n".join(lines)

    def _md_accountability(self, data: Dict[str, Any]) -> str:
        acct = data.get("accountability", {})
        mechanisms = acct.get("mechanisms", [])

        lines = [
            "## 7. Accountability Framework\n",
            f"**Escalation Process:** {acct.get('escalation', 'Working Group -> Steering Committee -> Public reporting')}\n"
            f"**Non-Compliance Threshold:** {acct.get('threshold', 'Two consecutive quarters off-track')}\n"
            f"**Review Cycle:** {acct.get('review_cycle', 'Annual comprehensive review')}\n",
        ]

        if mechanisms:
            lines.extend([
                "### Accountability Mechanisms\n",
                "| # | Mechanism | Trigger | Action | Owner |",
                "|---|-----------|---------|--------|-------|",
            ])
            for i, m in enumerate(mechanisms, 1):
                lines.append(
                    f"| {i} | {m.get('mechanism', '-')} "
                    f"| {m.get('trigger', '-')} "
                    f"| {m.get('action', '-')} "
                    f"| {m.get('owner', '-')} |"
                )
        else:
            lines.extend([
                "### Default Accountability Mechanisms\n",
                "1. **Quarterly Progress Reviews** -- Status reporting against joint targets",
                "2. **Annual Performance Assessment** -- Comprehensive review of all partners",
                "3. **Peer Review** -- Cross-partner verification of claims and progress",
                "4. **Public Transparency** -- Annual public disclosure of partnership results",
                "5. **Corrective Action Plans** -- Required for partners off-track for 2+ quarters",
            ])

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-025 Race to Zero Pack on {ts}.*  \n"
            f"*Partnership framework aligned with Race to Zero campaign requirements.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".partner-card{border:2px solid #c8e6c9;border-radius:10px;padding:16px;margin:10px 0;}"
            ".partner-name{font-size:1.1em;font-weight:700;color:#1b5e20;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        partnership = data.get("partnership_name", "Climate Action Partnership")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Race to Zero -- Partnership Framework</h1>\n'
            f'<p><strong>Lead:</strong> {org} | '
            f'<strong>Partnership:</strong> {partnership} | '
            f'<strong>Date:</strong> {ts}</p>'
        )

    def _html_overview(self, data: Dict[str, Any]) -> str:
        partners = data.get("partners", [])
        total_emissions = sum(p.get("emissions_tco2e", 0) for p in partners)
        return (
            f'<h2>1. Partnership Overview</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Partners</div>'
            f'<div class="card-value">{len(partners)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Combined Emissions</div>'
            f'<div class="card-value">{_dec_comma(total_emissions)}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Sectors</div>'
            f'<div class="card-value">{len(set(p.get("sector", "") for p in partners))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Countries</div>'
            f'<div class="card-value">{len(set(p.get("country", "") for p in partners))}</div></div>\n'
            f'</div>'
        )

    def _html_partners(self, data: Dict[str, Any]) -> str:
        partners = data.get("partners", [])
        rows = ""
        for p in partners:
            rows += (f'<tr><td>{p.get("name", "-")}</td><td>{p.get("sector", "-")}</td>'
                     f'<td>{_dec_comma(p.get("emissions_tco2e", 0))}</td>'
                     f'<td>{p.get("target_year", "-")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>Partner data pending</em></td></tr>'
        return (
            f'<h2>2. Partner Profiles</h2>\n'
            f'<table><tr><th>Organization</th><th>Sector</th><th>Emissions</th><th>Target</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_commitments(self, data: Dict[str, Any]) -> str:
        commitments = data.get("joint_commitments", [])
        rows = ""
        for c in commitments:
            rows += (f'<tr><td>{c.get("commitment", "-")}</td>'
                     f'<td>{_dec_comma(c.get("reduction_tco2e", 0))}</td>'
                     f'<td>{c.get("target_year", "-")}</td>'
                     f'<td>{c.get("status", "Active")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>Commitments pending</em></td></tr>'
        return (
            f'<h2>3. Joint Commitments</h2>\n'
            f'<table><tr><th>Commitment</th><th>Reduction (tCO2e)</th>'
            f'<th>Target Year</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance", {})
        bodies = gov.get("bodies", [])
        rows = ""
        for b in bodies:
            rows += (f'<tr><td>{b.get("name", "-")}</td><td>{b.get("role", "-")}</td>'
                     f'<td>{b.get("frequency", "-")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="3"><em>Governance structure pending</em></td></tr>'
        return (
            f'<h2>4. Governance</h2>\n'
            f'<table><tr><th>Body</th><th>Role</th><th>Frequency</th></tr>\n{rows}</table>'
        )

    def _html_data_sharing(self, data: Dict[str, Any]) -> str:
        ds = data.get("data_sharing", {})
        protocols = ds.get("protocols", [])
        rows = ""
        for p in protocols:
            rows += (f'<tr><td>{p.get("data_type", "-")}</td><td>{p.get("frequency", "-")}</td>'
                     f'<td>{p.get("format", "-")}</td><td>{p.get("classification", "-")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>Data sharing protocols pending</em></td></tr>'
        return (
            f'<h2>5. Data Sharing</h2>\n'
            f'<table><tr><th>Data Type</th><th>Frequency</th><th>Format</th><th>Classification</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_performance(self, data: Dict[str, Any]) -> str:
        kpis = data.get("kpis", [])
        rows = ""
        for kpi in kpis:
            rows += (f'<tr><td>{kpi.get("name", "-")}</td><td>{kpi.get("target", "-")}</td>'
                     f'<td>{kpi.get("current", "-")}</td><td>{kpi.get("status", "-")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>KPIs to be defined</em></td></tr>'
        return (
            f'<h2>6. Performance Tracking</h2>\n'
            f'<table><tr><th>KPI</th><th>Target</th><th>Current</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_accountability(self, data: Dict[str, Any]) -> str:
        acct = data.get("accountability", {})
        return (
            f'<h2>7. Accountability</h2>\n'
            f'<table><tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Escalation</td><td>{acct.get("escalation", "Working Group -> Steering Committee")}</td></tr>\n'
            f'<tr><td>Threshold</td><td>{acct.get("threshold", "Two quarters off-track")}</td></tr>\n'
            f'<tr><td>Review Cycle</td><td>{acct.get("review_cycle", "Annual")}</td></tr>\n'
            f'</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-025 Race to Zero Pack on {ts}'
            f'</div>'
        )
