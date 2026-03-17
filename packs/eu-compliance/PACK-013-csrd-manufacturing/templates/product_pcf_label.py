# -*- coding: utf-8 -*-
"""
ProductPCFLabelTemplate - Product Carbon Footprint Label

Generates product carbon footprint labels with product info, total PCF,
lifecycle breakdown, BOM hotspots, DPP QR code data, and ISO 14067
compliance statement.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _esc(value: str) -> str:
    """Escape HTML special characters."""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class ProductPCFLabelData(BaseModel):
    """Data model for product PCF label."""
    product_id: str = Field(default="")
    product_name: str = Field(default="")
    functional_unit: str = Field(default="1 unit")
    total_pcf_kgco2e: float = Field(default=0.0)
    lifecycle_breakdown: Dict[str, float] = Field(default_factory=dict)
    bom_hotspots: List[Dict[str, Any]] = Field(default_factory=list)
    dpp_data: Dict[str, Any] = Field(default_factory=dict)
    data_quality_rating: str = Field(default="MEDIUM")
    methodology: str = Field(default="ISO 14067:2018")
    allocation_method: str = Field(default="mass")


class ProductPCFLabelTemplate:
    """
    Product carbon footprint label template.

    Generates PCF labels with lifecycle breakdown, BOM hotspots,
    DPP data, and ISO 14067 compliance statements for Digital
    Product Passports and product-level carbon declarations.
    """

    PACK_ID = "PACK-013"
    TEMPLATE_NAME = "product_pcf_label"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """Render label in specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format '{fmt}'.")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render as GitHub-flavored Markdown."""
        sections: List[str] = []
        pid = data.get("product_id", "")
        pname = data.get("product_name", "Product")
        total = data.get("total_pcf_kgco2e", 0.0)
        fu = data.get("functional_unit", "1 unit")
        method = data.get("methodology", "ISO 14067:2018")

        sections.append(
            f"# Product Carbon Footprint Label\n\n"
            f"**Product:** {pname} (`{pid}`)\n\n"
            f"**Functional Unit:** {fu}\n\n"
            f"**Total PCF:** **{total:,.2f} kgCO2e** per {fu}"
        )

        # Lifecycle breakdown
        breakdown = data.get("lifecycle_breakdown", {})
        if breakdown:
            rows = ["## Lifecycle Breakdown\n",
                     "| Stage | kgCO2e | Share (%) |",
                     "|-------|--------|-----------|"]
            for stage, value in breakdown.items():
                pct = (value / total * 100) if total > 0 else 0.0
                bar = self._bar(pct)
                rows.append(f"| {stage.replace('_', ' ').title()} | {value:,.2f} | {pct:,.1f}% {bar} |")
            sections.append("\n".join(rows))

        # BOM hotspots
        hotspots = data.get("bom_hotspots", [])
        if hotspots:
            rows = ["## BOM Hotspots\n",
                     "| Component | Material | kgCO2e | Share (%) |",
                     "|-----------|----------|--------|-----------|"]
            for h in hotspots[:10]:
                em = h.get("emissions_kgco2e", 0.0)
                pct = (em / total * 100) if total > 0 else 0.0
                rows.append(
                    f"| {h.get('component_name', '')} | {h.get('material', '')} | "
                    f"{em:,.2f} | {pct:,.1f}% |"
                )
            sections.append("\n".join(rows))

        # DPP data
        dpp = data.get("dpp_data", {})
        if dpp:
            sections.append(
                f"## Digital Product Passport Data\n\n"
                f"| Field | Value |\n|-------|-------|\n"
                f"| PCF (kgCO2e) | {dpp.get('pcf_kgco2e', total):,.2f} |\n"
                f"| Methodology | {dpp.get('methodology', method)} |\n"
                f"| Allocation | {dpp.get('allocation_method', 'mass')} |\n"
                f"| Data Quality | {dpp.get('data_quality_rating', 'MEDIUM')} |\n"
                f"| Generated | {dpp.get('generated_at', self.generated_at)} |"
            )

        # ISO 14067 compliance
        sections.append(
            f"## Compliance Statement\n\n"
            f"This product carbon footprint has been calculated in accordance with "
            f"**{method}** (Carbon footprint of products - Requirements and guidelines "
            f"for quantification). The system boundary covers cradle-to-grave lifecycle "
            f"stages. Allocation method: {data.get('allocation_method', 'mass')}-based."
        )

        content = "\n\n".join(sections)
        ph = self._provenance(content)
        content += f"\n\n---\n\n*Generated by GreenLang {self.PACK_ID}*\n\n"
        content += f"**Provenance:** `{ph}`\n\n<!-- provenance_hash: {ph} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render as self-contained HTML."""
        pname = _esc(data.get("product_name", "Product"))
        total = data.get("total_pcf_kgco2e", 0.0)
        body = (
            f'<div class="section"><h2>Product Carbon Footprint</h2>'
            f'<p style="font-size:2em;color:#1a5276"><strong>{total:,.2f} kgCO2e</strong></p>'
            f'<p>per {_esc(data.get("functional_unit", "1 unit"))}</p></div>'
        )

        breakdown = data.get("lifecycle_breakdown", {})
        if breakdown:
            body += '<div class="section"><h2>Lifecycle Breakdown</h2><table class="data-table">'
            body += '<tr><th>Stage</th><th>kgCO2e</th><th>Share</th></tr>'
            for stage, val in breakdown.items():
                pct = (val / total * 100) if total > 0 else 0.0
                body += f'<tr><td>{_esc(stage.replace("_", " ").title())}</td><td>{val:,.2f}</td><td>{pct:,.1f}%</td></tr>'
            body += '</table></div>'

        ph = self._provenance(body)
        return self._wrap_html(f"PCF Label - {pname}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        report = {"report_type": self.TEMPLATE_NAME, "pack_id": self.PACK_ID,
                  "version": self.VERSION, "generated_at": self.generated_at, **data}
        report["provenance_hash"] = self._provenance(json.dumps(report, default=str, sort_keys=True))
        return report

    @staticmethod
    def _bar(pct: float) -> str:
        """Create a simple text bar for percentage."""
        filled = int(pct / 5)
        return "[" + "#" * filled + "-" * (20 - filled) + "]"

    @staticmethod
    def _provenance(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _wrap_html(self, title: str, body: str, ph: str) -> str:
        return (
            f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<title>{_esc(title)}</title>'
            f'<style>body{{font-family:sans-serif;max-width:800px;margin:40px auto}}'
            f'.data-table{{width:100%;border-collapse:collapse}}'
            f'.data-table td,.data-table th{{padding:8px;border:1px solid #ddd}}'
            f'.data-table th{{background:#1a5276;color:#fff}}'
            f'.section{{margin:20px 0;padding:15px;background:#fafafa;border-radius:6px}}</style>'
            f'</head><body><h1>{_esc(title)}</h1>{body}'
            f'<div style="margin-top:30px;font-family:monospace;font-size:0.85em">'
            f'Provenance: {ph}</div><!-- provenance_hash: {ph} --></body></html>'
        )
