# -*- coding: utf-8 -*-
"""
SMEAccountingGuideTemplate - Accounting software integration guide for PACK-026.

Renders a step-by-step guide for connecting accounting software
(Xero, QuickBooks, Sage) to GreenLang for automated carbon tracking,
with GL account code mappings, spend category to Scope 3 mappings,
carbon cost allocation, monthly P&L carbon tracking, and tax
deduction guidance.

Sections:
    1. Software Connection Guide (Xero/QuickBooks/Sage)
    2. Step-by-Step Setup Instructions (ASCII art placeholders)
    3. Spend Category Mapping (GL codes -> Scope 3)
    4. Carbon Cost Allocation Methodology
    5. Monthly P&L Carbon Tracking
    6. Tax Deduction Guidance

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"
_TEMPLATE_ID = "sme_accounting_guide"

_PRIMARY = "#1b5e20"
_SECONDARY = "#2e7d32"
_ACCENT = "#43a047"
_LIGHT = "#e8f5e9"
_LIGHTER = "#f1f8e9"
_CARD_BG = "#c8e6c9"

# ---------------------------------------------------------------------------
# Accounting software profiles
# ---------------------------------------------------------------------------
_ACCOUNTING_SOFTWARE = {
    "xero": {
        "name": "Xero",
        "api_type": "OAuth 2.0 REST API",
        "key_features": [
            "Automatic bank feed import",
            "Chart of accounts sync",
            "Invoice and bill categorization",
            "Multi-currency support",
        ],
        "setup_steps": [
            "Log in to your Xero account",
            "Navigate to Settings > Connected Apps",
            "Search for 'GreenLang Carbon Tracker'",
            "Click 'Connect' and authorize access",
            "Select the organization to connect",
            "Map your chart of accounts to emission categories",
            "Enable automatic monthly sync",
        ],
    },
    "quickbooks": {
        "name": "QuickBooks",
        "api_type": "OAuth 2.0 REST API",
        "key_features": [
            "Expense categorization sync",
            "Vendor classification",
            "P&L report integration",
            "Receipt data extraction",
        ],
        "setup_steps": [
            "Log in to QuickBooks Online",
            "Go to Apps > Find Apps",
            "Search for 'GreenLang'",
            "Click 'Get App Now'",
            "Authorize the connection",
            "Review and confirm account mappings",
            "Set sync frequency (daily/weekly/monthly)",
        ],
    },
    "sage": {
        "name": "Sage",
        "api_type": "Sage Business Cloud API",
        "key_features": [
            "Nominal ledger sync",
            "Purchase ledger integration",
            "VAT-aware categorization",
            "Departmental breakdown",
        ],
        "setup_steps": [
            "Log in to Sage Business Cloud",
            "Navigate to Connected Services",
            "Find 'GreenLang Carbon Tracker'",
            "Click 'Connect' and sign in",
            "Select your company file",
            "Map nominal codes to emission categories",
            "Configure sync schedule",
        ],
    },
}

# ---------------------------------------------------------------------------
# Default GL -> Scope 3 mappings
# ---------------------------------------------------------------------------
_DEFAULT_GL_MAPPINGS = [
    {"gl_range": "4000-4999", "category": "Cost of Sales", "scope3_cat": "1", "scope3_name": "Purchased Goods & Services", "ef_method": "Spend-based"},
    {"gl_range": "5000-5099", "category": "Raw Materials", "scope3_cat": "1", "scope3_name": "Purchased Goods & Services", "ef_method": "Spend-based"},
    {"gl_range": "5100-5199", "category": "Packaging", "scope3_cat": "1", "scope3_name": "Purchased Goods & Services", "ef_method": "Spend-based"},
    {"gl_range": "6000-6099", "category": "Staff Travel (air)", "scope3_cat": "6", "scope3_name": "Business Travel", "ef_method": "Distance-based"},
    {"gl_range": "6100-6199", "category": "Staff Travel (rail)", "scope3_cat": "6", "scope3_name": "Business Travel", "ef_method": "Distance-based"},
    {"gl_range": "6200-6299", "category": "Hotels", "scope3_cat": "6", "scope3_name": "Business Travel", "ef_method": "Spend-based"},
    {"gl_range": "6300-6399", "category": "Subsistence", "scope3_cat": "6", "scope3_name": "Business Travel", "ef_method": "Spend-based"},
    {"gl_range": "6400-6499", "category": "Freight & Shipping", "scope3_cat": "4", "scope3_name": "Upstream Transportation", "ef_method": "Weight-distance"},
    {"gl_range": "6500-6599", "category": "Office Supplies", "scope3_cat": "1", "scope3_name": "Purchased Goods & Services", "ef_method": "Spend-based"},
    {"gl_range": "6600-6699", "category": "IT Equipment", "scope3_cat": "2", "scope3_name": "Capital Goods", "ef_method": "Spend-based"},
    {"gl_range": "7000-7099", "category": "Electricity", "scope": "Scope 2", "scope3_name": "N/A (Scope 2)", "ef_method": "Activity-based (kWh)"},
    {"gl_range": "7100-7199", "category": "Gas", "scope": "Scope 1", "scope3_name": "N/A (Scope 1)", "ef_method": "Activity-based (kWh)"},
    {"gl_range": "7200-7299", "category": "Water", "scope3_cat": "1", "scope3_name": "Purchased Goods & Services", "ef_method": "Activity-based (m3)"},
    {"gl_range": "7300-7399", "category": "Waste Disposal", "scope3_cat": "5", "scope3_name": "Waste Generated", "ef_method": "Weight-based"},
    {"gl_range": "7400-7499", "category": "Vehicle Fuel", "scope": "Scope 1", "scope3_name": "N/A (Scope 1)", "ef_method": "Activity-based (litres)"},
    {"gl_range": "8000-8099", "category": "Professional Services", "scope3_cat": "1", "scope3_name": "Purchased Goods & Services", "ef_method": "Spend-based"},
    {"gl_range": "8100-8199", "category": "Marketing", "scope3_cat": "1", "scope3_name": "Purchased Goods & Services", "ef_method": "Spend-based"},
    {"gl_range": "8200-8299", "category": "Insurance", "scope3_cat": "1", "scope3_name": "Purchased Goods & Services", "ef_method": "Spend-based"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


# ===========================================================================
# Template Class
# ===========================================================================

class SMEAccountingGuideTemplate:
    """
    SME accounting software integration guide template.

    Renders step-by-step guides for connecting Xero/QuickBooks/Sage
    to GreenLang, with GL account mappings, carbon cost allocation,
    P&L carbon tracking, and tax deduction guidance across
    Markdown, HTML, and JSON formats.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the accounting guide as Markdown."""
        self.generated_at = _utcnow()
        software = data.get("accounting_software", "xero").lower()
        sections: List[str] = [
            self._md_header(data),
            self._md_software_overview(data, software),
            self._md_setup_steps(data, software),
            self._md_gl_mappings(data),
            self._md_industry_examples(data),
            self._md_carbon_cost_allocation(data),
            self._md_monthly_pl(data),
            self._md_tax_guidance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the accounting guide as HTML."""
        self.generated_at = _utcnow()
        software = data.get("accounting_software", "xero").lower()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_software_overview(data, software),
            self._html_setup_steps(data, software),
            self._html_gl_mappings(data),
            self._html_carbon_cost_allocation(data),
            self._html_monthly_pl(data),
            self._html_tax_guidance(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>SME Accounting Integration Guide</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the accounting guide as structured JSON."""
        self.generated_at = _utcnow()
        software = data.get("accounting_software", "xero").lower()
        sw_info = _ACCOUNTING_SOFTWARE.get(software, _ACCOUNTING_SOFTWARE["xero"])
        custom_mappings = data.get("custom_gl_mappings", [])
        all_mappings = custom_mappings if custom_mappings else _DEFAULT_GL_MAPPINGS

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {
                "name": data.get("org_name", ""),
                "accounting_software": sw_info["name"],
                "sector": data.get("sector", ""),
            },
            "software_setup": {
                "software": sw_info["name"],
                "api_type": sw_info["api_type"],
                "features": sw_info["key_features"],
                "steps": sw_info["setup_steps"],
            },
            "gl_mappings": all_mappings,
            "carbon_cost_allocation": {
                "method": data.get("allocation_method", "Proportional to emissions"),
                "internal_carbon_price": data.get("internal_carbon_price", 0),
                "currency": data.get("currency", "GBP"),
            },
            "tax_deductions": data.get("tax_deductions", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        software = data.get("accounting_software", "Xero")
        return (
            f"# Accounting Software Integration Guide\n\n"
            f"**Organization:** {data.get('org_name', 'Your Company')}  \n"
            f"**Software:** {software}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_software_overview(self, data: Dict[str, Any], software: str) -> str:
        sw_info = _ACCOUNTING_SOFTWARE.get(software, _ACCOUNTING_SOFTWARE["xero"])
        features = "\n".join(f"- {f}" for f in sw_info["key_features"])

        lines = [
            f"## 1. Connecting {sw_info['name']} to GreenLang\n",
            f"**API Type:** {sw_info['api_type']}\n",
            f"**Key Features:**\n{features}\n",
            f"### How It Works\n",
            f"```",
            f"+------------------+     +-----------------+     +------------------+",
            f"|                  |     |                 |     |                  |",
            f"|   {sw_info['name']:^14}   | --> |   GreenLang     | --> |  Carbon Report   |",
            f"|   Accounting     |     |   Carbon Engine |     |  Dashboard       |",
            f"|                  |     |                 |     |                  |",
            f"+------------------+     +-----------------+     +------------------+",
            f"     Your data            Auto-categorize         Emissions report",
            f"     stays safe           & calculate EFs         per GL code",
            f"```",
        ]
        return "\n".join(lines)

    def _md_setup_steps(self, data: Dict[str, Any], software: str) -> str:
        sw_info = _ACCOUNTING_SOFTWARE.get(software, _ACCOUNTING_SOFTWARE["xero"])
        steps = sw_info["setup_steps"]

        lines = [f"## 2. Step-by-Step Setup\n"]
        for idx, step in enumerate(steps, 1):
            lines.append(f"### Step {idx}: {step}\n")
            # ASCII art placeholder for screenshot
            lines.append(f"```")
            lines.append(f"+{'=' * 50}+")
            lines.append(f"|{'':^50}|")
            lines.append(f"|{'[Screenshot: ' + step[:30] + '...]':^50}|")
            lines.append(f"|{'':^50}|")
            lines.append(f"+{'=' * 50}+")
            lines.append(f"```")
            lines.append("")

        return "\n".join(lines)

    def _md_gl_mappings(self, data: Dict[str, Any]) -> str:
        custom = data.get("custom_gl_mappings", [])
        mappings = custom if custom else _DEFAULT_GL_MAPPINGS

        lines = [
            "## 3. Spend Category Mapping\n",
            "GL account codes are mapped to emission scopes and categories:\n",
            "| GL Range | Category | Scope / S3 Cat | Emission Factor Method |",
            "|----------|----------|:--------------:|------------------------|",
        ]
        for m in mappings:
            scope_info = m.get("scope", f"Scope 3 Cat {m.get('scope3_cat', '?')}")
            lines.append(
                f"| {m.get('gl_range', '')} "
                f"| {m.get('category', '')} "
                f"| {scope_info} "
                f"| {m.get('ef_method', '')} |"
            )

        return "\n".join(lines)

    def _md_industry_examples(self, data: Dict[str, Any]) -> str:
        sector = data.get("sector", "General")
        examples = data.get("industry_examples", [])

        if not examples:
            # Provide defaults based on sector
            examples = [
                {"gl_code": "4100", "description": "Main product costs", "scope3_cat": "1",
                 "ef_note": "Use sector-specific emission factors where available"},
                {"gl_code": "6050", "description": "Staff mileage claims", "scope3_cat": "6",
                 "ef_note": "Distance-based: 0.21 kgCO2e/km (average car)"},
                {"gl_code": "7050", "description": "Electricity bills", "scope3_cat": "Scope 2",
                 "ef_note": "Use actual kWh from bills, grid factor varies by country"},
            ]

        lines = [
            f"### Industry Examples ({sector})\n",
            "| GL Code | Description | Scope/Cat | Note |",
            "|---------|-------------|-----------|------|",
        ]
        for ex in examples:
            lines.append(
                f"| {ex.get('gl_code', '')} "
                f"| {ex.get('description', '')} "
                f"| {ex.get('scope3_cat', '')} "
                f"| {ex.get('ef_note', '')} |"
            )

        return "\n".join(lines)

    def _md_carbon_cost_allocation(self, data: Dict[str, Any]) -> str:
        currency = data.get("currency", "GBP")
        icp = float(data.get("internal_carbon_price", 75))

        lines = [
            "## 4. Carbon Cost Allocation\n",
            "### Internal Carbon Price\n",
            f"Recommended internal carbon price: **{currency} {_dec_comma(icp)}/tCO2e**\n",
            "### Allocation Methodology\n",
            f"1. **Calculate emissions** per cost centre / department",
            f"2. **Multiply** emissions by internal carbon price",
            f"3. **Allocate** carbon cost to each department's P&L",
            f"4. **Track** monthly to create carbon budget accountability\n",
            "### Example\n",
            f"| Department | Emissions (tCO2e) | Carbon Cost ({currency}) |",
            f"|------------|------------------:|------------------------:|",
        ]
        depts = data.get("departments", [
            {"name": "Operations", "tco2e": 150},
            {"name": "Sales & Marketing", "tco2e": 45},
            {"name": "IT", "tco2e": 30},
            {"name": "Admin", "tco2e": 15},
        ])
        for dept in depts:
            tco2e = float(dept.get("tco2e", 0))
            cost = tco2e * icp
            lines.append(
                f"| {dept.get('name', '')} | {_dec_comma(tco2e)} | {currency} {_dec_comma(cost)} |"
            )
        total_e = sum(float(d.get("tco2e", 0)) for d in depts)
        total_c = total_e * icp
        lines.append(
            f"| **Total** | **{_dec_comma(total_e)}** | **{currency} {_dec_comma(total_c)}** |"
        )

        return "\n".join(lines)

    def _md_monthly_pl(self, data: Dict[str, Any]) -> str:
        currency = data.get("currency", "GBP")

        lines = [
            "## 5. Monthly P&L Carbon Tracking\n",
            "Add a carbon column to your monthly P&L to track emissions alongside costs:\n",
            f"| P&L Line | {currency} Amount | tCO2e | Carbon Cost ({currency}) |",
            f"|----------|-----------:|------:|------------------------:|",
        ]
        pl_lines = data.get("pl_lines", [
            {"line": "Revenue", "amount": 500000, "tco2e": 0, "carbon_cost": 0},
            {"line": "Cost of Sales", "amount": -250000, "tco2e": 120, "carbon_cost": 9000},
            {"line": "Energy", "amount": -24000, "tco2e": 80, "carbon_cost": 6000},
            {"line": "Travel", "amount": -15000, "tco2e": 25, "carbon_cost": 1875},
            {"line": "Office Costs", "amount": -8000, "tco2e": 5, "carbon_cost": 375},
            {"line": "Gross Profit", "amount": 203000, "tco2e": 230, "carbon_cost": 17250},
        ])
        for pl in pl_lines:
            lines.append(
                f"| {pl.get('line', '')} "
                f"| {currency} {_dec_comma(pl.get('amount', 0))} "
                f"| {_dec_comma(pl.get('tco2e', 0))} "
                f"| {currency} {_dec_comma(pl.get('carbon_cost', 0))} |"
            )

        return "\n".join(lines)

    def _md_tax_guidance(self, data: Dict[str, Any]) -> str:
        currency = data.get("currency", "GBP")
        region = data.get("region", "UK")

        lines = [
            "## 6. Tax Deduction Guidance\n",
            f"### Capital Allowances for Energy Efficiency ({region})\n",
        ]

        deductions = data.get("tax_deductions", [
            {"item": "LED Lighting", "allowance": "100% First Year (ECA)", "notes": "Energy Technology List eligible"},
            {"item": "Heat Pumps", "allowance": "100% First Year (ECA)", "notes": "Must be on ETL"},
            {"item": "Solar PV", "allowance": "50% First Year (Super-deduction until 2026)", "notes": "Check current rates"},
            {"item": "EV Charging Points", "allowance": "100% First Year", "notes": "Workplace charging scheme"},
            {"item": "Insulation", "allowance": "Standard 18% WDA", "notes": "Integral features"},
            {"item": "Building Energy Mgmt Systems", "allowance": "100% First Year (ECA)", "notes": "Energy Technology List"},
        ])

        lines.append("| Investment | Tax Allowance | Notes |")
        lines.append("|------------|---------------|-------|")
        for d in deductions:
            lines.append(
                f"| {d.get('item', '')} "
                f"| {d.get('allowance', '')} "
                f"| {d.get('notes', '')} |"
            )

        lines.append(f"\n> **Note:** Tax rules change frequently. Always consult with your "
                     f"accountant or tax advisor for the latest capital allowance rates.")

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}*  \n"
            f"*Accounting software integration guide for automated carbon tracking.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "*, *::before, *::after{box-sizing:border-box;}"
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:900px;margin:0 auto;background:#fff;padding:32px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:#388e3c;margin-top:16px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid {_CARD_BG};padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".step-card{{background:{_LIGHTER};border:1px solid {_CARD_BG};"
            f"border-left:5px solid {_ACCENT};border-radius:0 10px 10px 0;"
            f"padding:14px;margin:10px 0;}}"
            f".step-num{{display:inline-block;background:{_PRIMARY};color:#fff;width:28px;"
            f"height:28px;border-radius:50%;text-align:center;line-height:28px;"
            f"font-weight:700;font-size:0.85em;margin-right:8px;}}"
            f".screenshot{{background:#f5f5f5;border:2px dashed {_CARD_BG};"
            f"border-radius:8px;padding:24px;text-align:center;color:#9e9e9e;"
            f"font-size:0.85em;margin:8px 0;}}"
            f".flow-diagram{{display:flex;align-items:center;justify-content:center;"
            f"gap:12px;margin:16px 0;flex-wrap:wrap;}}"
            f".flow-box{{background:{_LIGHT};border:2px solid {_ACCENT};border-radius:8px;"
            f"padding:12px 16px;text-align:center;min-width:160px;}}"
            f".flow-arrow{{font-size:1.5em;color:{_ACCENT};font-weight:700;}}"
            f".tip-box{{background:{_LIGHTER};border-left:4px solid {_ACCENT};"
            f"padding:12px 16px;border-radius:0 8px 8px 0;margin:10px 0;font-size:0.9em;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#689f38;font-size:0.8em;text-align:center;}}"
            f"@media(max-width:600px){{.flow-diagram{{flex-direction:column;}}"
            f".report{{padding:16px;}}}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        software = data.get("accounting_software", "Xero")
        return (
            f'<h1>Accounting Software Integration Guide</h1>\n'
            f'<p><strong>{data.get("org_name", "Your Company")}</strong> | '
            f'{software} | Generated: {ts}</p>'
        )

    def _html_software_overview(self, data: Dict[str, Any], software: str) -> str:
        sw_info = _ACCOUNTING_SOFTWARE.get(software, _ACCOUNTING_SOFTWARE["xero"])

        features_html = ""
        for f in sw_info["key_features"]:
            features_html += f'<li>{f}</li>\n'

        return (
            f'<h2>1. Connecting {sw_info["name"]} to GreenLang</h2>\n'
            f'<p><strong>API Type:</strong> {sw_info["api_type"]}</p>\n'
            f'<p><strong>Key Features:</strong></p>\n<ul>{features_html}</ul>\n'
            f'<div class="flow-diagram">\n'
            f'  <div class="flow-box"><strong>{sw_info["name"]}</strong><br>Your Accounting Data</div>\n'
            f'  <div class="flow-arrow">--></div>\n'
            f'  <div class="flow-box"><strong>GreenLang</strong><br>Carbon Engine</div>\n'
            f'  <div class="flow-arrow">--></div>\n'
            f'  <div class="flow-box"><strong>Reports</strong><br>Emissions Dashboard</div>\n'
            f'</div>'
        )

    def _html_setup_steps(self, data: Dict[str, Any], software: str) -> str:
        sw_info = _ACCOUNTING_SOFTWARE.get(software, _ACCOUNTING_SOFTWARE["xero"])
        steps = sw_info["setup_steps"]

        cards = ""
        for idx, step in enumerate(steps, 1):
            cards += (
                f'<div class="step-card">\n'
                f'  <span class="step-num">{idx}</span>'
                f'  <strong>{step}</strong>\n'
                f'  <div class="screenshot">[Screenshot: {step}]</div>\n'
                f'</div>\n'
            )

        return f'<h2>2. Step-by-Step Setup</h2>\n{cards}'

    def _html_gl_mappings(self, data: Dict[str, Any]) -> str:
        custom = data.get("custom_gl_mappings", [])
        mappings = custom if custom else _DEFAULT_GL_MAPPINGS

        rows = ""
        for m in mappings:
            scope_info = m.get("scope", f"S3 Cat {m.get('scope3_cat', '?')}")
            rows += (
                f'<tr><td>{m.get("gl_range", "")}</td>'
                f'<td>{m.get("category", "")}</td>'
                f'<td>{scope_info}</td>'
                f'<td>{m.get("ef_method", "")}</td></tr>\n'
            )

        return (
            f'<h2>3. Spend Category Mapping</h2>\n'
            f'<p>GL account codes mapped to emission scopes and categories:</p>\n'
            f'<table>\n'
            f'<tr><th>GL Range</th><th>Category</th><th>Scope/Cat</th><th>EF Method</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_carbon_cost_allocation(self, data: Dict[str, Any]) -> str:
        currency = data.get("currency", "GBP")
        icp = float(data.get("internal_carbon_price", 75))

        depts = data.get("departments", [
            {"name": "Operations", "tco2e": 150},
            {"name": "Sales & Marketing", "tco2e": 45},
            {"name": "IT", "tco2e": 30},
            {"name": "Admin", "tco2e": 15},
        ])

        rows = ""
        total_e = 0.0
        for dept in depts:
            tco2e = float(dept.get("tco2e", 0))
            total_e += tco2e
            cost = tco2e * icp
            rows += (
                f'<tr><td>{dept.get("name", "")}</td>'
                f'<td>{_dec_comma(tco2e)} tCO2e</td>'
                f'<td>{currency} {_dec_comma(cost)}</td></tr>\n'
            )
        rows += (
            f'<tr><th>Total</th><th>{_dec_comma(total_e)} tCO2e</th>'
            f'<th>{currency} {_dec_comma(total_e * icp)}</th></tr>\n'
        )

        return (
            f'<h2>4. Carbon Cost Allocation</h2>\n'
            f'<p><strong>Internal Carbon Price:</strong> {currency} {_dec_comma(icp)}/tCO2e</p>\n'
            f'<table>\n'
            f'<tr><th>Department</th><th>Emissions</th><th>Carbon Cost</th></tr>\n'
            f'{rows}</table>\n'
            f'<div class="tip-box"><strong>Method:</strong> Calculate emissions per department, '
            f'multiply by internal carbon price, allocate to each P&L.</div>'
        )

    def _html_monthly_pl(self, data: Dict[str, Any]) -> str:
        currency = data.get("currency", "GBP")
        pl_lines = data.get("pl_lines", [
            {"line": "Revenue", "amount": 500000, "tco2e": 0, "carbon_cost": 0},
            {"line": "Cost of Sales", "amount": -250000, "tco2e": 120, "carbon_cost": 9000},
            {"line": "Energy", "amount": -24000, "tco2e": 80, "carbon_cost": 6000},
            {"line": "Travel", "amount": -15000, "tco2e": 25, "carbon_cost": 1875},
            {"line": "Office Costs", "amount": -8000, "tco2e": 5, "carbon_cost": 375},
            {"line": "Gross Profit", "amount": 203000, "tco2e": 230, "carbon_cost": 17250},
        ])

        rows = ""
        for pl in pl_lines:
            rows += (
                f'<tr><td>{"<strong>" + pl.get("line", "") + "</strong>" if "Profit" in pl.get("line", "") else pl.get("line", "")}</td>'
                f'<td>{currency} {_dec_comma(pl.get("amount", 0))}</td>'
                f'<td>{_dec_comma(pl.get("tco2e", 0))}</td>'
                f'<td>{currency} {_dec_comma(pl.get("carbon_cost", 0))}</td></tr>\n'
            )

        return (
            f'<h2>5. Monthly P&L Carbon Tracking</h2>\n'
            f'<p>Add a carbon column to your monthly P&L:</p>\n'
            f'<table>\n'
            f'<tr><th>P&L Line</th><th>Amount</th><th>tCO2e</th><th>Carbon Cost</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_tax_guidance(self, data: Dict[str, Any]) -> str:
        deductions = data.get("tax_deductions", [
            {"item": "LED Lighting", "allowance": "100% First Year (ECA)", "notes": "Energy Technology List eligible"},
            {"item": "Heat Pumps", "allowance": "100% First Year (ECA)", "notes": "Must be on ETL"},
            {"item": "Solar PV", "allowance": "50% First Year", "notes": "Check current rates"},
            {"item": "EV Charging Points", "allowance": "100% First Year", "notes": "Workplace charging scheme"},
            {"item": "Insulation", "allowance": "Standard 18% WDA", "notes": "Integral features"},
            {"item": "BEMS", "allowance": "100% First Year (ECA)", "notes": "Energy Technology List"},
        ])

        rows = ""
        for d in deductions:
            rows += (
                f'<tr><td>{d.get("item", "")}</td>'
                f'<td>{d.get("allowance", "")}</td>'
                f'<td>{d.get("notes", "")}</td></tr>\n'
            )

        return (
            f'<h2>6. Tax Deduction Guidance</h2>\n'
            f'<h3>Capital Allowances for Energy Efficiency</h3>\n'
            f'<table>\n'
            f'<tr><th>Investment</th><th>Tax Allowance</th><th>Notes</th></tr>\n'
            f'{rows}</table>\n'
            f'<div class="tip-box"><strong>Note:</strong> Tax rules change frequently. '
            f'Always consult your accountant for current capital allowance rates.</div>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}<br>'
            f'Accounting integration guide for automated carbon tracking'
            f'</div>'
        )
