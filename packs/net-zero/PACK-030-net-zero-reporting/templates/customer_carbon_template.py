# -*- coding: utf-8 -*-
"""
CustomerCarbonTemplate - Customer Carbon Footprint Template for PACK-030.

Renders a customer-facing product carbon footprint report covering product
lifecycle emissions, supply chain carbon intensity, Scope 3 Category 11
(use of sold products) analysis, reduction initiatives, carbon labeling
data, and customer engagement metrics. Multi-format output (MD, HTML,
JSON, PDF) with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  Product Carbon Footprint
    3.  Lifecycle Stage Breakdown
    4.  Supply Chain Emissions
    5.  Use-Phase Emissions (Scope 3 Cat 11)
    6.  End-of-Life Emissions (Scope 3 Cat 12)
    7.  Reduction Initiatives
    8.  Carbon Label / PCF Data
    9.  Customer Engagement Metrics
    10. Audit Trail & Provenance

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
"""

import hashlib, json, logging, uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_MODULE_VERSION = "30.0.0"
_PACK_ID = "PACK-030"
_TEMPLATE_ID = "customer_carbon"
_PRIMARY = "#1b5e20"
_SECONDARY = "#2e7d32"
_ACCENT = "#66bb6a"
_LIGHT = "#e8f5e9"
_LIGHTER = "#f1f8f2"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow(): return datetime.now(timezone.utc).replace(microsecond=0)
def _new_uuid(): return str(uuid.uuid4())
def _compute_hash(data):
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
def _dec(val, places=2):
    try: return str(Decimal(str(val)).quantize(Decimal("0." + "0" * places), rounding=ROUND_HALF_UP))
    except: return str(val)
def _dec_comma(val, places=2):
    try:
        rounded = Decimal(str(val)).quantize(Decimal("0." + "0" * places if places > 0 else "0"), rounding=ROUND_HALF_UP)
        parts = str(rounded).split("."); ip = parts[0]; neg = ip.startswith("-")
        if neg: ip = ip[1:]
        fmt = ""
        for i, c in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0: fmt = "," + fmt
            fmt = c + fmt
        if neg: fmt = "-" + fmt
        return fmt + ("." + parts[1] if len(parts) > 1 else "")
    except: return str(val)
def _pct_change(current, baseline):
    c, b = Decimal(str(current)), Decimal(str(baseline))
    if b == 0: return Decimal("0.00")
    return ((c - b) / b * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Lifecycle Stages
# ---------------------------------------------------------------------------

LIFECYCLE_STAGES: List[str] = [
    "Raw Materials Extraction",
    "Manufacturing & Processing",
    "Packaging",
    "Transportation & Distribution",
    "Use Phase",
    "End of Life",
]

CARBON_LABEL_STANDARDS: List[str] = [
    "ISO 14067:2018",
    "PAS 2050",
    "GHG Protocol Product Standard",
    "EU PEF (Product Environmental Footprint)",
]


class CustomerCarbonTemplate:
    """Customer carbon footprint template for PACK-030. Supports MD, HTML, JSON, PDF."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # -----------------------------------------------------------------------
    # Public Render Methods
    # -----------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_product_footprint(data), self._md_lifecycle(data),
            self._md_supply_chain(data), self._md_use_phase(data),
            self._md_end_of_life(data), self._md_initiatives(data),
            self._md_carbon_label(data), self._md_engagement(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_product_footprint(data), self._html_lifecycle(data),
            self._html_initiatives(data), self._html_carbon_label(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Customer Carbon Report - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n'
            f'<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {_compute_hash(body)} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        products = data.get("products", [])
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION, "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""), "reporting_year": data.get("reporting_year", ""),
            "stakeholder": "customer",
            "products": products,
            "supply_chain": data.get("supply_chain", {}),
            "lifecycle_total_kgco2e": _dec(sum(
                float(p.get("footprint_kgco2e", 0)) for p in products
            )),
            "initiatives": data.get("initiatives", []),
            "carbon_label": data.get("carbon_label", {}),
            "engagement": data.get("engagement", {}),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "format": "pdf",
            "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {
                "title": f"Customer Carbon Report - {data.get('org_name', '')}",
                "author": "GreenLang PACK-030",
            },
        }

    # -----------------------------------------------------------------------
    # Markdown Section Renderers
    # -----------------------------------------------------------------------

    def _md_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Customer Carbon Footprint Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Audience:** Customers & Downstream Partners  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data):
        products = data.get("products", [])
        total = sum(float(p.get("footprint_kgco2e", 0)) for p in products)
        prev_total = sum(float(p.get("previous_kgco2e", 0)) for p in products)
        yoy = float(_pct_change(total, prev_total)) if prev_total else 0
        initiatives = data.get("initiatives", [])
        total_reduction = sum(float(i.get("reduction_kgco2e", 0)) for i in initiatives)
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Products Covered | {len(products)} |",
            f"| Total Product Carbon | {_dec_comma(total, 1)} kgCO2e |",
            f"| YoY Change | {'+' if yoy > 0 else ''}{_dec(yoy)}% |",
            f"| Reduction from Initiatives | {_dec_comma(total_reduction, 0)} kgCO2e |",
            f"| Carbon Label Standard | {data.get('carbon_label', {}).get('standard', 'ISO 14067')} |",
        ]
        return "\n".join(lines)

    def _md_product_footprint(self, data):
        products = data.get("products", [])
        lines = [
            "## 2. Product Carbon Footprint\n",
            "| # | Product | Unit | PCF (kgCO2e) | YoY (%) | Intensity |",
            "|---|---------|------|-------------:|--------:|-----------|",
        ]
        for i, p in enumerate(products, 1):
            pcf = float(p.get("footprint_kgco2e", 0))
            prev = float(p.get("previous_kgco2e", 0))
            yoy = float(_pct_change(pcf, prev)) if prev else 0
            lines.append(
                f"| {i} | {p.get('name', '')} | {p.get('unit', 'per unit')} | "
                f"{_dec_comma(pcf, 2)} | {'+' if yoy > 0 else ''}{_dec(yoy)}% | "
                f"{p.get('intensity', '')} |"
            )
        if not products:
            lines.append("| - | _No products listed_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_lifecycle(self, data):
        lifecycle = data.get("lifecycle_breakdown", {})
        lines = [
            "## 3. Lifecycle Stage Breakdown\n",
            "| Stage | Emissions (kgCO2e) | Share (%) |",
            "|-------|-----------:|--------:|",
        ]
        total = sum(float(lifecycle.get(stage, 0)) for stage in LIFECYCLE_STAGES)
        for stage in LIFECYCLE_STAGES:
            val = float(lifecycle.get(stage, 0))
            share = val / total * 100 if total else 0
            lines.append(f"| {stage} | {_dec_comma(val, 1)} | {_dec(share)}% |")
        lines.append(f"| **Total** | **{_dec_comma(total, 1)}** | **100%** |")
        return "\n".join(lines)

    def _md_supply_chain(self, data):
        sc = data.get("supply_chain", {})
        suppliers = sc.get("suppliers", [])
        lines = [
            "## 4. Supply Chain Emissions\n",
            f"**Total Supply Chain Emissions:** {_dec_comma(sc.get('total_tco2e', 0), 0)} tCO2e  \n"
            f"**Supplier Coverage:** {sc.get('coverage_pct', 0)}%\n",
            "| # | Supplier | Category | Emissions (tCO2e) | Share (%) | SBTi |",
            "|---|----------|----------|------------------:|----------:|:----:|",
        ]
        for i, s in enumerate(suppliers, 1):
            lines.append(
                f"| {i} | {s.get('name', '')} | {s.get('category', '')} | "
                f"{_dec_comma(s.get('emissions_tco2e', 0), 0)} | "
                f"{_dec(s.get('share_pct', 0))}% | {s.get('sbti', 'No')} |"
            )
        if not suppliers:
            lines.append("| - | _No supplier data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_use_phase(self, data):
        use = data.get("use_phase", {})
        products = use.get("products", [])
        lines = [
            "## 5. Use-Phase Emissions (Scope 3 Category 11)\n",
            f"**Methodology:** {use.get('methodology', 'Expected lifetime use')}\n",
            "| # | Product | Lifetime Use | Annual Use (kgCO2e) | Lifetime (kgCO2e) |",
            "|---|---------|-------------|--------------------:|-----------------:|",
        ]
        for i, p in enumerate(products, 1):
            lines.append(
                f"| {i} | {p.get('name', '')} | {p.get('lifetime_years', '')} yrs | "
                f"{_dec_comma(p.get('annual_kgco2e', 0), 1)} | "
                f"{_dec_comma(p.get('lifetime_kgco2e', 0), 0)} |"
            )
        if not products:
            lines.append("| - | _No use-phase data_ | - | - | - |")
        return "\n".join(lines)

    def _md_end_of_life(self, data):
        eol = data.get("end_of_life", {})
        pathways = eol.get("pathways", [])
        lines = [
            "## 6. End-of-Life Emissions (Scope 3 Category 12)\n",
            "| # | Pathway | Share (%) | Emissions (kgCO2e) |",
            "|---|---------|----------:|-----------:|",
        ]
        for i, p in enumerate(pathways, 1):
            lines.append(
                f"| {i} | {p.get('pathway', '')} | {_dec(p.get('share_pct', 0))}% | "
                f"{_dec_comma(p.get('emissions_kgco2e', 0), 1)} |"
            )
        if not pathways:
            lines.extend([
                "| 1 | Recycling | - | - |",
                "| 2 | Landfill | - | - |",
                "| 3 | Incineration | - | - |",
            ])
        return "\n".join(lines)

    def _md_initiatives(self, data):
        initiatives = data.get("initiatives", [])
        lines = [
            "## 7. Reduction Initiatives\n",
            "| # | Initiative | Impact (kgCO2e) | Status | Timeline |",
            "|---|-----------|----------------:|--------|----------|",
        ]
        for i, init in enumerate(initiatives, 1):
            lines.append(
                f"| {i} | {init.get('name', '')} | "
                f"{_dec_comma(init.get('reduction_kgco2e', 0), 0)} | "
                f"{init.get('status', 'In Progress')} | {init.get('timeline', '')} |"
            )
        if not initiatives:
            lines.append("| - | _No initiatives listed_ | - | - | - |")
        return "\n".join(lines)

    def _md_carbon_label(self, data):
        cl = data.get("carbon_label", {})
        lines = [
            "## 8. Carbon Label / PCF Data\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Standard | {cl.get('standard', 'ISO 14067:2018')} |",
            f"| Scope | {cl.get('scope', 'Cradle-to-grave')} |",
            f"| Verification | {cl.get('verification', 'Third-party verified')} |",
            f"| Label Type | {cl.get('label_type', 'Product Carbon Footprint')} |",
            f"| Comparability | {cl.get('comparability', 'Industry average benchmark')} |",
            f"| Data Source | {cl.get('data_source', 'Primary + secondary')} |",
        ]
        lines.append("\n### Applicable Standards\n")
        for std in CARBON_LABEL_STANDARDS:
            applied = "Applied" if cl.get(std.split(":")[0].lower().replace(" ", "_"), False) else "Reference"
            lines.append(f"- **{std}**: {applied}")
        return "\n".join(lines)

    def _md_engagement(self, data):
        eng = data.get("engagement", {})
        lines = [
            "## 9. Customer Engagement Metrics\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Customers Reached | {_dec_comma(eng.get('customers_reached', 0), 0)} |",
            f"| Label Awareness | {eng.get('label_awareness_pct', 'N/A')}% |",
            f"| Low-Carbon Product Uptake | {eng.get('low_carbon_uptake_pct', 'N/A')}% |",
            f"| Customer Feedback Score | {eng.get('feedback_score', 'N/A')}/5 |",
            f"| Carbon Offset Opt-In | {eng.get('offset_optin_pct', 'N/A')}% |",
        ]
        return "\n".join(lines)

    def _md_audit(self, data):
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f"## 10. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n"
            f"| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n"
            f"| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n"
            f"| Data Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n"
            f"*Customer product carbon footprint report.*"
        )

    # -----------------------------------------------------------------------
    # HTML Section Renderers
    # -----------------------------------------------------------------------

    def _css(self):
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            f"background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_ACCENT};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};"
            f"padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #a5d6a7;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#e8f5e9;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));"
            f"gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});"
            f"border-radius:10px;padding:18px;text-align:center;"
            f"border-left:4px solid {_ACCENT};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_SECONDARY};}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};"
            f"color:{_SECONDARY};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Customer Carbon Footprint</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'{data.get("reporting_year", "")} | {ts}</p>'
        )

    def _html_executive_summary(self, data):
        products = data.get("products", [])
        total = sum(float(p.get("footprint_kgco2e", 0)) for p in products)
        initiatives = data.get("initiatives", [])
        total_reduction = sum(float(i.get("reduction_kgco2e", 0)) for i in initiatives)
        return (
            f'<h2>1. Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Products</div>'
            f'<div class="card-value">{len(products)}</div></div>\n'
            f'<div class="card"><div class="card-label">Total PCF</div>'
            f'<div class="card-value">{_dec_comma(total, 0)}</div>'
            f'<div class="card-unit">kgCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Reduction</div>'
            f'<div class="card-value">{_dec_comma(total_reduction, 0)}</div>'
            f'<div class="card-unit">kgCO2e</div></div>\n</div>'
        )

    def _html_product_footprint(self, data):
        products = data.get("products", [])
        rows = ""
        for i, p in enumerate(products, 1):
            rows += (
                f'<tr><td>{i}</td><td>{p.get("name", "")}</td>'
                f'<td>{_dec_comma(p.get("footprint_kgco2e", 0), 2)}</td>'
                f'<td>{p.get("unit", "per unit")}</td></tr>\n'
            )
        return (
            f'<h2>2. Product Carbon Footprint</h2>\n<table>\n'
            f'<tr><th>#</th><th>Product</th><th>PCF (kgCO2e)</th><th>Unit</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_lifecycle(self, data):
        lifecycle = data.get("lifecycle_breakdown", {})
        rows = ""
        total = sum(float(lifecycle.get(stage, 0)) for stage in LIFECYCLE_STAGES)
        for stage in LIFECYCLE_STAGES:
            val = float(lifecycle.get(stage, 0))
            share = val / total * 100 if total else 0
            rows += f'<tr><td>{stage}</td><td>{_dec_comma(val, 1)}</td><td>{_dec(share)}%</td></tr>\n'
        return (
            f'<h2>3. Lifecycle Breakdown</h2>\n<table>\n'
            f'<tr><th>Stage</th><th>Emissions</th><th>Share</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_initiatives(self, data):
        initiatives = data.get("initiatives", [])
        rows = ""
        for i, init in enumerate(initiatives, 1):
            rows += (
                f'<tr><td>{i}</td><td>{init.get("name", "")}</td>'
                f'<td>{_dec_comma(init.get("reduction_kgco2e", 0), 0)}</td>'
                f'<td>{init.get("status", "")}</td></tr>\n'
            )
        return (
            f'<h2>4. Reduction Initiatives</h2>\n<table>\n'
            f'<tr><th>#</th><th>Initiative</th><th>Impact (kgCO2e)</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_carbon_label(self, data):
        cl = data.get("carbon_label", {})
        return (
            f'<h2>5. Carbon Label</h2>\n<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Standard</td><td>{cl.get("standard", "ISO 14067")}</td></tr>\n'
            f'<tr><td>Scope</td><td>{cl.get("scope", "Cradle-to-grave")}</td></tr>\n'
            f'<tr><td>Verification</td><td>{cl.get("verification", "Third-party")}</td></tr>\n'
            f'</table>'
        )

    def _html_audit(self, data):
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f'<h2>6. Audit</h2>\n<table>\n'
            f'<tr><th>Param</th><th>Value</th></tr>\n'
            f'<tr><td>ID</td><td><code>{rid}</code></td></tr>\n'
            f'<tr><td>Generated</td><td>{ts}</td></tr>\n'
            f'<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'
        )

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-030 on {ts} '
            f'- Customer Carbon Report</div>'
        )
