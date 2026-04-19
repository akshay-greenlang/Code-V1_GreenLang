# -*- coding: utf-8 -*-
"""
DocumentationIndex - Documentation Completeness Tracker for PACK-044.

Generates a documentation index covering documentation completeness
by category, missing documents, version tracking, and compliance
documentation requirements.

Sections:
    1. Documentation Overview
    2. Document Inventory
    3. Completeness by Category
    4. Missing Documents
    5. Compliance Requirements

Author: GreenLang Team
Version: 44.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "44.0.0"


class DocumentationIndex:
    """
    Documentation index template for GHG inventory management.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DocumentationIndex."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render documentation index as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_inventory(data),
            self._md_completeness(data),
            self._md_missing(data),
            self._md_compliance_reqs(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render documentation index as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overview(data),
            self._html_inventory(data),
            self._html_completeness(data),
            self._html_missing(data),
            self._html_compliance_reqs(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render documentation index as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        return {
            "template": "documentation_index",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_provenance(data),
            "company_name": self._get_val(data, "company_name", ""),
            "doc_overview": data.get("doc_overview", {}),
            "documents": data.get("documents", []),
            "completeness_by_category": data.get("completeness_by_category", []),
            "missing_documents": data.get("missing_documents", []),
            "compliance_requirements": data.get("compliance_requirements", []),
        }

    def _md_header(self, data: Dict[str, Any]) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return f"# Documentation Index - {company}\n\n**Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n---"

    def _md_overview(self, data: Dict[str, Any]) -> str:
        overview = data.get("doc_overview", {})
        if not overview:
            return "## 1. Documentation Overview\n\nNo overview data."
        return (
            "## 1. Documentation Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Documents | {overview.get('total_documents', 0)} |\n"
            f"| Complete | {overview.get('complete', 0)} |\n"
            f"| In Progress | {overview.get('in_progress', 0)} |\n"
            f"| Missing | {overview.get('missing', 0)} |\n"
            f"| Completeness | {overview.get('completeness_pct', 0):.0f}% |"
        )

    def _md_inventory(self, data: Dict[str, Any]) -> str:
        docs = data.get("documents", [])
        if not docs:
            return ""
        lines = [
            "## 2. Document Inventory", "",
            "| Document | Category | Version | Status | Last Updated | Owner |",
            "|---------|----------|---------|--------|-------------|-------|",
        ]
        for d in docs:
            lines.append(
                f"| {d.get('name', '')} | {d.get('category', '')} | "
                f"{d.get('version', '-')} | **{d.get('status', '')}** | "
                f"{d.get('last_updated', '-')} | {d.get('owner', '-')} |"
            )
        return "\n".join(lines)

    def _md_completeness(self, data: Dict[str, Any]) -> str:
        cats = data.get("completeness_by_category", [])
        if not cats:
            return ""
        lines = [
            "## 3. Completeness by Category", "",
            "| Category | Required | Complete | Completeness |",
            "|---------|---------|----------|-------------|",
        ]
        for c in cats:
            req = c.get("required", 0)
            comp = c.get("complete", 0)
            pct = (comp / req * 100) if req > 0 else 0.0
            lines.append(f"| {c.get('category', '')} | {req} | {comp} | {pct:.0f}% |")
        return "\n".join(lines)

    def _md_missing(self, data: Dict[str, Any]) -> str:
        missing = data.get("missing_documents", [])
        if not missing:
            return ""
        lines = [
            "## 4. Missing Documents", "",
            "| Document | Category | Required By | Priority |",
            "|---------|----------|-----------|----------|",
        ]
        for m in missing:
            lines.append(
                f"| {m.get('name', '')} | {m.get('category', '')} | "
                f"{m.get('required_by', '-')} | **{m.get('priority', 'medium')}** |"
            )
        return "\n".join(lines)

    def _md_compliance_reqs(self, data: Dict[str, Any]) -> str:
        reqs = data.get("compliance_requirements", [])
        if not reqs:
            return ""
        lines = [
            "## 5. Compliance Documentation Requirements", "",
            "| Framework | Required Document | Status | Notes |",
            "|----------|------------------|--------|-------|",
        ]
        for r in reqs:
            lines.append(
                f"| {r.get('framework', '')} | {r.get('document', '')} | "
                f"**{r.get('status', '')}** | {r.get('notes', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return f"---\n\n*Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}*\n*Provenance Hash: `{provenance}`*"

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f'<meta charset="UTF-8"><title>Documentation Index - {company}</title>\n'
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#1a1a2e;line-height:1.6;}"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;}h2{color:#1b263b;margin-top:2rem;}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}"
            "th{background:#f0f4f8;font-weight:600;}tr:nth-child(even){background:#fafbfc;}"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return f'<div><h1>Documentation Index &mdash; {company}</h1><hr></div>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        overview = data.get("doc_overview", {})
        if not overview:
            return ""
        return f'<div><h2>1. Overview</h2><p>Documents: {overview.get("total_documents", 0)} | Complete: {overview.get("completeness_pct", 0):.0f}%</p></div>'

    def _html_inventory(self, data: Dict[str, Any]) -> str:
        docs = data.get("documents", [])
        if not docs:
            return ""
        rows = ""
        for d in docs:
            rows += f"<tr><td>{d.get('name', '')}</td><td>{d.get('category', '')}</td><td>{d.get('version', '-')}</td><td><strong>{d.get('status', '')}</strong></td></tr>\n"
        return '<div><h2>2. Documents</h2>\n<table><thead><tr><th>Name</th><th>Category</th><th>Version</th><th>Status</th></tr></thead>\n<tbody>' + rows + '</tbody></table></div>'

    def _html_completeness(self, data: Dict[str, Any]) -> str:
        cats = data.get("completeness_by_category", [])
        if not cats:
            return ""
        rows = ""
        for c in cats:
            req = c.get("required", 0)
            comp = c.get("complete", 0)
            pct = (comp / req * 100) if req > 0 else 0.0
            rows += f"<tr><td>{c.get('category', '')}</td><td>{req}</td><td>{comp}</td><td>{pct:.0f}%</td></tr>\n"
        return '<div><h2>3. Completeness</h2>\n<table><thead><tr><th>Category</th><th>Required</th><th>Complete</th><th>%</th></tr></thead>\n<tbody>' + rows + '</tbody></table></div>'

    def _html_missing(self, data: Dict[str, Any]) -> str:
        missing = data.get("missing_documents", [])
        if not missing:
            return ""
        rows = ""
        for m in missing:
            rows += f"<tr><td>{m.get('name', '')}</td><td>{m.get('category', '')}</td><td><strong>{m.get('priority', 'medium')}</strong></td></tr>\n"
        return '<div><h2>4. Missing</h2>\n<table><thead><tr><th>Document</th><th>Category</th><th>Priority</th></tr></thead>\n<tbody>' + rows + '</tbody></table></div>'

    def _html_compliance_reqs(self, data: Dict[str, Any]) -> str:
        reqs = data.get("compliance_requirements", [])
        if not reqs:
            return ""
        rows = ""
        for r in reqs:
            rows += f"<tr><td>{r.get('framework', '')}</td><td>{r.get('document', '')}</td><td><strong>{r.get('status', '')}</strong></td></tr>\n"
        return '<div><h2>5. Compliance Requirements</h2>\n<table><thead><tr><th>Framework</th><th>Document</th><th>Status</th></tr></thead>\n<tbody>' + rows + '</tbody></table></div>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return f'<div style="font-size:0.85rem;color:#666;"><hr><p>Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}</p><p class="provenance">Provenance Hash: {provenance}</p></div>'
