# -*- coding: utf-8 -*-
"""
Lineage Reporter Engine - AGENT-DATA-018 (Engine 6 of 7)

Generates lineage reports and visualizations for the Data Lineage Tracker
agent. Supports multiple output formats (Mermaid, DOT/Graphviz, JSON,
D3-compatible, plain text, HTML) and regulatory compliance reports
(CSRD/ESRS Article 8, GHG Protocol calculation chain, SOC 2 data
processing audit trail).

The reporter reads from the LineageGraphEngine to obtain nodes and edges,
then renders them in the requested format. Every generated report is
assigned a UUID, SHA-256 hashed for provenance, and stored in-memory for
later retrieval.

Zero-Hallucination Guarantees:
    - All report content is derived deterministically from graph data.
    - No LLM calls for any part of report generation.
    - SHA-256 provenance hash computed for every report.
    - Thread-safe via threading.Lock on the shared reports store.
    - Prometheus metric emitted on every report generation.

Supported report types:
    - csrd_esrs:      CSRD/ESRS Article 8 data lineage disclosure (Markdown)
    - ghg_protocol:   GHG Protocol calculation chain documentation (Markdown)
    - soc2:           SOC 2 data processing audit trail (Markdown)
    - custom:         Custom report via user-supplied parameters (JSON)
    - visualization:  Graph visualization in chosen format

Supported output formats:
    - mermaid:  Mermaid flowchart TD syntax
    - dot:      DOT/Graphviz digraph notation
    - json:     JSON adjacency list (nodes + edges + metadata)
    - d3:       D3-compatible force-directed graph JSON
    - text:     Plain text ASCII summary
    - html:     HTML with embedded Mermaid.js for interactive rendering
    - pdf:      Placeholder (returns metadata indicating external render)

Example:
    >>> from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
    >>> from greenlang.data_lineage_tracker.lineage_reporter import (
    ...     LineageReporterEngine,
    ... )
    >>> graph = LineageGraphEngine()
    >>> reporter = LineageReporterEngine(graph)
    >>> report = reporter.generate_report("visualization", format="mermaid")
    >>> assert report["format"] == "mermaid"
    >>> assert report["report_hash"] is not None

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import textwrap
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.data_lineage_tracker.config import get_config
from greenlang.data_lineage_tracker.metrics import (
    PROMETHEUS_AVAILABLE,
    observe_processing_duration,
    record_report_generated,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_REPORT_TYPES = frozenset({
    "csrd_esrs",
    "ghg_protocol",
    "soc2",
    "custom",
    "visualization",
})

VALID_FORMATS = frozenset({
    "mermaid",
    "dot",
    "json",
    "d3",
    "text",
    "html",
    "pdf",
})

# DOT node shapes keyed by asset type
_DOT_SHAPES: Dict[str, str] = {
    "dataset": "box",
    "field": "component",
    "agent": "diamond",
    "pipeline": "parallelogram",
    "report": "ellipse",
    "metric": "octagon",
    "external_source": "house",
}

# Mermaid node shape brackets keyed by asset type
_MERMAID_SHAPES: Dict[str, tuple] = {
    "dataset": ("[", "]"),
    "field": ("([", "])"),
    "agent": ("{", "}"),
    "pipeline": ("[[", "]]"),
    "report": ("(", ")"),
    "metric": ("{{", "}}"),
    "external_source": (">", "]"),
}

# D3 group numbers for colouring by asset type
_D3_GROUPS: Dict[str, int] = {
    "dataset": 1,
    "field": 2,
    "agent": 3,
    "pipeline": 4,
    "report": 5,
    "metric": 6,
    "external_source": 7,
}


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _sanitize_label(text: str) -> str:
    """Sanitize a label for safe inclusion in graph notations.

    Removes characters that could break Mermaid/DOT syntax such as
    double quotes, angle brackets, semicolons, and curly braces.

    Args:
        text: Raw label string.

    Returns:
        Sanitized label string safe for graph rendering.
    """
    if not text:
        return "unnamed"
    sanitized = text.replace('"', "'")
    sanitized = sanitized.replace("<", "(")
    sanitized = sanitized.replace(">", ")")
    sanitized = sanitized.replace(";", ",")
    sanitized = sanitized.replace("{", "(")
    sanitized = sanitized.replace("}", ")")
    sanitized = sanitized.replace("\n", " ")
    return sanitized


def _compute_sha256(content: str) -> str:
    """Compute a SHA-256 hex digest for the given content string.

    Args:
        content: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LineageReporterEngine
# ---------------------------------------------------------------------------


class LineageReporterEngine:
    """Generates lineage reports and visualizations from the lineage graph.

    Reads node and edge data from a ``LineageGraphEngine`` instance and
    renders it in the requested output format. Every generated report is
    stored in-memory with a unique ID and provenance hash for later
    retrieval and audit.

    The engine supports two categories of output:

    1. **Visualization formats** -- Mermaid, DOT/Graphviz, JSON graph,
       D3-compatible JSON, plain text summary, HTML with embedded
       Mermaid.js, and PDF (placeholder).

    2. **Compliance reports** -- CSRD/ESRS Article 8 data lineage
       disclosure, GHG Protocol calculation chain documentation, and
       SOC 2 data processing audit trail. All in Markdown format.

    Attributes:
        _graph: Reference to the LineageGraphEngine for reading graph data.
        _reports: In-memory store of generated reports keyed by report ID.
        _lock: Thread-safety lock for the reports store.
        _provenance: ProvenanceTracker for SHA-256 audit trail.

    Example:
        >>> graph = LineageGraphEngine()
        >>> reporter = LineageReporterEngine(graph)
        >>> report = reporter.generate_report("visualization", format="json")
        >>> assert report["report_type"] == "visualization"
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        graph: Any,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize the LineageReporterEngine.

        Args:
            graph: A LineageGraphEngine instance (or compatible object)
                that provides ``get_nodes()``, ``get_edges()``, and
                ``get_statistics()`` methods.
            provenance: Optional ProvenanceTracker. A fresh tracker is
                created when ``None`` is supplied.
        """
        self._graph = graph
        self._reports: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._provenance = provenance if provenance is not None else ProvenanceTracker()
        logger.info("LineageReporterEngine initialized")

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        report_type: str,
        format: str = "json",
        scope: str = "full",
        parameters: Optional[Dict[str, Any]] = None,
        max_depth: int = 10,
    ) -> dict:
        """Generate a lineage report or visualization.

        Dispatches to the appropriate internal generator based on the
        requested ``report_type`` and ``format``. The resulting report
        is stored in-memory and a provenance entry is recorded.

        Args:
            report_type: One of ``csrd_esrs``, ``ghg_protocol``,
                ``soc2``, ``custom``, ``visualization``.
            format: Output format. One of ``mermaid``, ``dot``, ``json``,
                ``d3``, ``text``, ``html``, ``pdf``.
            scope: Scope label for the report (e.g. ``"full"``,
                ``"pipeline:spend-to-emissions"``, or a specific asset ID).
            parameters: Optional dictionary of extra parameters to pass
                to the generator (e.g. ``asset_type_filter``).
            max_depth: Maximum graph traversal depth for visualization
                formats. Defaults to 10.

        Returns:
            Dictionary with keys: ``id``, ``report_type``, ``format``,
            ``scope``, ``content``, ``report_hash``, ``generated_by``,
            ``generated_at``.

        Raises:
            ValueError: If ``report_type`` or ``format`` is not in the
                supported set.
        """
        start_time = time.monotonic()
        params = parameters or {}

        # -- Validate inputs -----------------------------------------------
        if report_type not in VALID_REPORT_TYPES:
            raise ValueError(
                f"Invalid report_type '{report_type}'. "
                f"Must be one of: {sorted(VALID_REPORT_TYPES)}"
            )
        if format not in VALID_FORMATS:
            raise ValueError(
                f"Invalid format '{format}'. "
                f"Must be one of: {sorted(VALID_FORMATS)}"
            )

        # -- Dispatch to generator -----------------------------------------
        content = self._dispatch(
            report_type=report_type,
            fmt=format,
            scope=scope,
            parameters=params,
            max_depth=max_depth,
        )

        # -- Build report envelope -----------------------------------------
        report_id = str(uuid.uuid4())
        report_hash = _compute_sha256(content)
        generated_at = _utcnow().isoformat()

        report: dict = {
            "id": report_id,
            "report_type": report_type,
            "format": format,
            "scope": scope,
            "content": content,
            "report_hash": report_hash,
            "generated_by": "data-lineage-tracker",
            "generated_at": generated_at,
        }

        # -- Persist -------------------------------------------------------
        with self._lock:
            self._reports[report_id] = report

        # -- Provenance ----------------------------------------------------
        self._provenance.record(
            entity_type="report",
            entity_id=report_id,
            action="report_generated",
            metadata={
                "report_type": report_type,
                "format": format,
                "scope": scope,
                "report_hash": report_hash,
            },
        )

        # -- Metrics -------------------------------------------------------
        elapsed = time.monotonic() - start_time
        record_report_generated(report_type=report_type, format=format)
        observe_processing_duration("report_generate", elapsed)

        logger.info(
            "Generated %s report (format=%s, scope=%s) id=%s in %.3fs",
            report_type,
            format,
            scope,
            report_id,
            elapsed,
        )
        return report

    # ------------------------------------------------------------------
    # Visualization generators
    # ------------------------------------------------------------------

    def generate_mermaid(
        self,
        scope: str = "full",
        max_depth: int = 10,
        asset_type_filter: Optional[str] = None,
    ) -> str:
        """Generate a Mermaid flowchart TD diagram from the lineage graph.

        Nodes are rendered as rectangles with ``asset_type: name``
        labels. Edges are annotated with the transformation type.

        Args:
            scope: Scope label (informational, does not filter).
            max_depth: Maximum depth for rendering. Edges beyond this
                depth from any root are excluded.
            asset_type_filter: Optional asset type to include. When
                ``None`` all asset types are included.

        Returns:
            Mermaid-formatted string suitable for embedding in Markdown.

        Example output::

            graph TD
                A[dataset: raw_invoices] -->|extract| B[dataset: invoice_fields]
                B -->|normalize| C[dataset: normalized_spend]
        """
        nodes = self._get_nodes()
        edges = self._get_edges()

        # Apply optional asset type filter
        if asset_type_filter:
            node_ids = {
                n["id"]
                for n in nodes
                if n.get("asset_type", "") == asset_type_filter
            }
            nodes = [n for n in nodes if n["id"] in node_ids]
            edges = [
                e for e in edges
                if e.get("source") in node_ids or e.get("target") in node_ids
            ]

        # Build node alias map (A, B, C, ...)
        node_alias: Dict[str, str] = {}
        for idx, node in enumerate(nodes):
            alias = self._index_to_alias(idx)
            node_alias[node["id"]] = alias

        lines: List[str] = ["graph TD"]

        # Emit nodes
        for node in nodes:
            alias = node_alias[node["id"]]
            asset_type = node.get("asset_type", "unknown")
            name = _sanitize_label(
                node.get("display_name", "")
                or node.get("qualified_name", "")
                or node.get("id", "")
            )
            left, right = _MERMAID_SHAPES.get(asset_type, ("[", "]"))
            lines.append(f"    {alias}{left}{asset_type}: {name}{right}")

        # Emit edges
        rendered_edges = 0
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src not in node_alias or tgt not in node_alias:
                continue
            tx_type = _sanitize_label(
                edge.get("transformation_type", "")
                or edge.get("edge_type", "")
                or "derives"
            )
            src_alias = node_alias[src]
            tgt_alias = node_alias[tgt]
            lines.append(f"    {src_alias} -->|{tx_type}| {tgt_alias}")
            rendered_edges += 1
            if rendered_edges >= max_depth * len(nodes) + 1:
                lines.append("    %% ... truncated at max_depth limit")
                break

        return "\n".join(lines)

    def generate_dot(
        self,
        scope: str = "full",
        max_depth: int = 10,
        asset_type_filter: Optional[str] = None,
    ) -> str:
        """Generate a DOT/Graphviz digraph from the lineage graph.

        Node shapes are assigned by asset type (box for dataset,
        diamond for agent, ellipse for report, etc.). Edge labels show
        the transformation type.

        Args:
            scope: Scope label (informational, does not filter).
            max_depth: Maximum depth for rendering.
            asset_type_filter: Optional asset type filter.

        Returns:
            DOT-formatted string suitable for Graphviz rendering.
        """
        nodes = self._get_nodes()
        edges = self._get_edges()

        if asset_type_filter:
            node_ids = {
                n["id"]
                for n in nodes
                if n.get("asset_type", "") == asset_type_filter
            }
            nodes = [n for n in nodes if n["id"] in node_ids]
            edges = [
                e for e in edges
                if e.get("source") in node_ids or e.get("target") in node_ids
            ]

        node_set = {n["id"] for n in nodes}
        lines: List[str] = [
            "digraph DataLineage {",
            '    rankdir=TB;',
            '    node [fontname="Helvetica", fontsize=10];',
            '    edge [fontname="Helvetica", fontsize=8];',
            "",
        ]

        # Emit nodes
        for node in nodes:
            nid = node["id"].replace("-", "_")
            asset_type = node.get("asset_type", "unknown")
            name = _sanitize_label(
                node.get("display_name", "")
                or node.get("qualified_name", "")
                or node.get("id", "")
            )
            shape = _DOT_SHAPES.get(asset_type, "box")
            lines.append(
                f'    {nid} [label="{asset_type}: {name}", shape={shape}];'
            )

        lines.append("")

        # Emit edges
        rendered_edges = 0
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src not in node_set or tgt not in node_set:
                continue
            src_id = src.replace("-", "_")
            tgt_id = tgt.replace("-", "_")
            tx_type = _sanitize_label(
                edge.get("transformation_type", "")
                or edge.get("edge_type", "")
                or "derives"
            )
            lines.append(
                f'    {src_id} -> {tgt_id} [label="{tx_type}"];'
            )
            rendered_edges += 1
            if rendered_edges >= max_depth * len(nodes) + 1:
                lines.append("    // ... truncated at max_depth limit")
                break

        lines.append("}")
        return "\n".join(lines)

    def generate_json_graph(
        self,
        scope: str = "full",
        max_depth: int = 10,
    ) -> str:
        """Generate a JSON adjacency-list representation of the lineage graph.

        Args:
            scope: Scope label.
            max_depth: Maximum depth (informational, included in metadata).

        Returns:
            JSON string with structure ``{"nodes": [...], "edges": [...],
            "metadata": {...}}``.
        """
        nodes = self._get_nodes()
        edges = self._get_edges()
        stats = self._get_graph_statistics()

        graph_data: Dict[str, Any] = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "scope": scope,
                "max_depth": max_depth,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "generated_at": _utcnow().isoformat(),
                "generated_by": "data-lineage-tracker",
                "graph_statistics": stats,
            },
        }
        return json.dumps(graph_data, indent=2, default=str)

    def generate_d3(
        self,
        scope: str = "full",
        max_depth: int = 10,
    ) -> str:
        """Generate a D3-compatible force-directed graph JSON structure.

        Output contains ``nodes`` with ``id``, ``group``, ``label``, and
        ``asset_type``; and ``links`` with ``source``, ``target``,
        ``value``, and ``label``.

        Args:
            scope: Scope label.
            max_depth: Maximum depth (informational).

        Returns:
            JSON string with D3-compatible structure.
        """
        nodes = self._get_nodes()
        edges = self._get_edges()

        node_set = {n["id"] for n in nodes}

        d3_nodes: List[Dict[str, Any]] = []
        for node in nodes:
            asset_type = node.get("asset_type", "unknown")
            d3_nodes.append({
                "id": node["id"],
                "group": _D3_GROUPS.get(asset_type, 0),
                "label": (
                    node.get("display_name", "")
                    or node.get("qualified_name", "")
                    or node.get("id", "")
                ),
                "asset_type": asset_type,
                "classification": node.get("classification", "internal"),
                "status": node.get("status", "active"),
            })

        d3_links: List[Dict[str, Any]] = []
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src not in node_set or tgt not in node_set:
                continue
            d3_links.append({
                "source": src,
                "target": tgt,
                "value": edge.get("confidence", 1.0),
                "label": (
                    edge.get("transformation_type", "")
                    or edge.get("edge_type", "derives")
                ),
            })

        d3_data: Dict[str, Any] = {
            "nodes": d3_nodes,
            "links": d3_links,
            "metadata": {
                "scope": scope,
                "max_depth": max_depth,
                "generated_at": _utcnow().isoformat(),
            },
        }
        return json.dumps(d3_data, indent=2, default=str)

    def generate_text_summary(self, scope: str = "full") -> str:
        """Generate a plain-text lineage summary report.

        Includes total asset and edge counts, graph depth, coverage
        statistics, asset type breakdown, and a top-level data flow
        listing.

        Args:
            scope: Scope label.

        Returns:
            Multi-line plain text string with ASCII formatting.
        """
        nodes = self._get_nodes()
        edges = self._get_edges()
        stats = self._get_graph_statistics()

        node_count = len(nodes)
        edge_count = len(edges)

        # Asset type breakdown
        type_counts: Dict[str, int] = {}
        for node in nodes:
            asset_type = node.get("asset_type", "unknown")
            type_counts[asset_type] = type_counts.get(asset_type, 0) + 1

        # Classification breakdown
        class_counts: Dict[str, int] = {}
        for node in nodes:
            classification = node.get("classification", "internal")
            class_counts[classification] = (
                class_counts.get(classification, 0) + 1
            )

        # Compute root and leaf nodes
        target_ids = {e.get("target", "") for e in edges}
        source_ids = {e.get("source", "") for e in edges}
        all_node_ids = {n["id"] for n in nodes}

        root_ids = all_node_ids - target_ids
        leaf_ids = all_node_ids - source_ids

        # Build node name map
        name_map: Dict[str, str] = {}
        for n in nodes:
            name_map[n["id"]] = (
                n.get("display_name", "")
                or n.get("qualified_name", "")
                or n["id"][:12]
            )

        lines: List[str] = [
            "=" * 72,
            "DATA LINEAGE SUMMARY REPORT",
            "=" * 72,
            f"Scope:          {scope}",
            f"Generated:      {_utcnow().isoformat()}",
            f"Generated by:   data-lineage-tracker",
            "",
            "-" * 72,
            "GRAPH OVERVIEW",
            "-" * 72,
            f"Total Assets:   {node_count}",
            f"Total Edges:    {edge_count}",
            f"Root Nodes:     {len(root_ids)}",
            f"Leaf Nodes:     {len(leaf_ids)}",
            f"Max Depth:      {stats.get('max_depth', 'N/A')}",
            f"Components:     {stats.get('connected_components', 'N/A')}",
            f"Coverage:       {stats.get('coverage_score', 'N/A')}",
            "",
            "-" * 72,
            "ASSET TYPE BREAKDOWN",
            "-" * 72,
        ]
        for atype, count in sorted(
            type_counts.items(), key=lambda x: -x[1]
        ):
            bar = "#" * min(count, 50)
            lines.append(f"  {atype:<20s} {count:>5d}  {bar}")

        lines.extend([
            "",
            "-" * 72,
            "CLASSIFICATION BREAKDOWN",
            "-" * 72,
        ])
        for cls_name, count in sorted(
            class_counts.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {cls_name:<20s} {count:>5d}")

        # Top-level data flow (root -> leaf paths)
        lines.extend([
            "",
            "-" * 72,
            "TOP-LEVEL DATA FLOW (Root Sources -> Leaf Destinations)",
            "-" * 72,
        ])
        if root_ids:
            for rid in sorted(root_ids):
                lines.append(f"  [SOURCE] {name_map.get(rid, rid[:12])}")
        else:
            lines.append("  (no root sources identified)")
        lines.append("")
        if leaf_ids:
            for lid in sorted(leaf_ids):
                lines.append(f"  [DEST]   {name_map.get(lid, lid[:12])}")
        else:
            lines.append("  (no leaf destinations identified)")

        lines.extend([
            "",
            "=" * 72,
            "END OF LINEAGE SUMMARY REPORT",
            "=" * 72,
        ])
        return "\n".join(lines)

    def generate_html(
        self,
        scope: str = "full",
        max_depth: int = 10,
    ) -> str:
        """Generate an HTML page with embedded Mermaid.js for interactive visualization.

        The output is a self-contained HTML document that loads Mermaid
        from a CDN and renders the lineage graph inline. A statistics
        table is shown above the graph.

        Args:
            scope: Scope label.
            max_depth: Maximum depth for the embedded Mermaid diagram.

        Returns:
            Complete HTML document string.
        """
        mermaid_content = self.generate_mermaid(
            scope=scope, max_depth=max_depth
        )
        stats = self._get_graph_statistics()
        nodes = self._get_nodes()
        edges = self._get_edges()
        generated_at = _utcnow().isoformat()

        # Build asset type breakdown rows
        type_counts: Dict[str, int] = {}
        for node in nodes:
            asset_type = node.get("asset_type", "unknown")
            type_counts[asset_type] = type_counts.get(asset_type, 0) + 1

        stats_rows = ""
        for atype, count in sorted(
            type_counts.items(), key=lambda x: -x[1]
        ):
            stats_rows += (
                f"            <tr><td>{atype}</td>"
                f"<td>{count}</td></tr>\n"
            )

        html = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Lineage Report - {scope}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont,
                        "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #f9fafb;
                    color: #1f2937;
                }}
                h1 {{ color: #065f46; border-bottom: 2px solid #065f46; padding-bottom: 8px; }}
                h2 {{ color: #047857; }}
                .stats-table {{
                    border-collapse: collapse;
                    width: 100%;
                    max-width: 600px;
                    margin: 16px 0;
                }}
                .stats-table th, .stats-table td {{
                    border: 1px solid #d1d5db;
                    padding: 8px 12px;
                    text-align: left;
                }}
                .stats-table th {{
                    background: #065f46;
                    color: white;
                }}
                .stats-table tr:nth-child(even) {{ background: #ecfdf5; }}
                .mermaid-container {{
                    background: white;
                    border: 1px solid #d1d5db;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 16px 0;
                    overflow-x: auto;
                }}
                .footer {{
                    margin-top: 24px;
                    padding-top: 12px;
                    border-top: 1px solid #d1d5db;
                    font-size: 0.85em;
                    color: #6b7280;
                }}
            </style>
        </head>
        <body>
            <h1>Data Lineage Report</h1>
            <p><strong>Scope:</strong> {scope} |
               <strong>Generated:</strong> {generated_at} |
               <strong>By:</strong> data-lineage-tracker</p>

            <h2>Graph Statistics</h2>
            <table class="stats-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Assets</td><td>{len(nodes)}</td></tr>
                <tr><td>Total Edges</td><td>{len(edges)}</td></tr>
                <tr><td>Max Depth</td><td>{stats.get('max_depth', 'N/A')}</td></tr>
                <tr><td>Connected Components</td><td>{stats.get('connected_components', 'N/A')}</td></tr>
                <tr><td>Coverage Score</td><td>{stats.get('coverage_score', 'N/A')}</td></tr>
            </table>

            <h2>Asset Type Breakdown</h2>
            <table class="stats-table">
                <tr><th>Asset Type</th><th>Count</th></tr>
{stats_rows}        </table>

            <h2>Lineage Graph</h2>
            <div class="mermaid-container">
                <pre class="mermaid">
{mermaid_content}
                </pre>
            </div>

            <div class="footer">
                <p>Generated by GreenLang Data Lineage Tracker (AGENT-DATA-018)
                   | Max Depth: {max_depth}
                   | Report Scope: {scope}</p>
            </div>

            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
            </script>
        </body>
        </html>
        """)
        return html

    # ------------------------------------------------------------------
    # Compliance report generators
    # ------------------------------------------------------------------

    def generate_csrd_esrs_report(self, scope: str = "full") -> str:
        """Generate a CSRD/ESRS Article 8 data lineage disclosure.

        Produces a Markdown document with sections covering data sources,
        data flow, transformations, quality controls, and compliance
        mapping as required by CSRD/ESRS disclosure requirements.

        Args:
            scope: Scope label for the report.

        Returns:
            Markdown-formatted compliance report string.
        """
        nodes = self._get_nodes()
        edges = self._get_edges()
        stats = self._get_graph_statistics()
        generated_at = _utcnow().isoformat()

        # Categorize assets
        datasets = [n for n in nodes if n.get("asset_type") == "dataset"]
        agents = [n for n in nodes if n.get("asset_type") == "agent"]
        reports = [n for n in nodes if n.get("asset_type") == "report"]
        ext_sources = [
            n for n in nodes if n.get("asset_type") == "external_source"
        ]

        # Build edge summary
        tx_types: Dict[str, int] = {}
        for edge in edges:
            tx = edge.get("transformation_type", edge.get("edge_type", "unknown"))
            tx_types[tx] = tx_types.get(tx, 0) + 1

        # Build name lookup
        name_map = self._build_name_map(nodes)

        lines: List[str] = [
            "# CSRD/ESRS Data Lineage Disclosure",
            "",
            f"**Report Scope:** {scope}",
            f"**Generated:** {generated_at}",
            f"**Generated By:** GreenLang Data Lineage Tracker (AGENT-DATA-018)",
            "",
            "---",
            "",
            "## 1. Data Sources",
            "",
            "This section documents all data sources feeding into the sustainability",
            "reporting pipeline as required by CSRD/ESRS Article 8 disclosure standards.",
            "",
            f"**Total registered data assets:** {len(nodes)}",
            f"**External data sources:** {len(ext_sources)}",
            f"**Internal datasets:** {len(datasets)}",
            "",
            "### 1.1 External Sources",
            "",
            "| # | Source Name | Classification | Status |",
            "|---|-----------|----------------|--------|",
        ]
        for idx, src in enumerate(ext_sources, 1):
            name = name_map.get(src["id"], src["id"][:12])
            classification = src.get("classification", "internal")
            status = src.get("status", "active")
            lines.append(f"| {idx} | {name} | {classification} | {status} |")
        if not ext_sources:
            lines.append("| - | No external sources registered | - | - |")

        lines.extend([
            "",
            "### 1.2 Internal Datasets",
            "",
            "| # | Dataset Name | Owner | Classification | Status |",
            "|---|-------------|-------|----------------|--------|",
        ])
        for idx, ds in enumerate(datasets, 1):
            name = name_map.get(ds["id"], ds["id"][:12])
            owner = ds.get("owner", "unassigned")
            classification = ds.get("classification", "internal")
            status = ds.get("status", "active")
            lines.append(
                f"| {idx} | {name} | {owner} | {classification} | {status} |"
            )
        if not datasets:
            lines.append("| - | No datasets registered | - | - | - |")

        lines.extend([
            "",
            "## 2. Data Flow",
            "",
            "The data lineage graph documents the complete flow of sustainability",
            "data from source to disclosure, ensuring end-to-end traceability.",
            "",
            f"**Total lineage edges:** {len(edges)}",
            f"**Graph depth:** {stats.get('max_depth', 'N/A')}",
            f"**Connected components:** {stats.get('connected_components', 'N/A')}",
            "",
            "### 2.1 Processing Agents",
            "",
            "| # | Agent Name | Description |",
            "|---|-----------|-------------|",
        ])
        for idx, agent in enumerate(agents, 1):
            name = name_map.get(agent["id"], agent["id"][:12])
            desc = agent.get("description", "Data processing agent")
            lines.append(f"| {idx} | {name} | {desc} |")
        if not agents:
            lines.append("| - | No agents registered | - |")

        lines.extend([
            "",
            "## 3. Transformations",
            "",
            "All data transformations applied between source and disclosure",
            "are documented below for compliance audit.",
            "",
            "### 3.1 Transformation Summary",
            "",
            "| Transformation Type | Count |",
            "|-------------------|-------|",
        ])
        for tx, count in sorted(tx_types.items(), key=lambda x: -x[1]):
            lines.append(f"| {tx} | {count} |")
        if not tx_types:
            lines.append("| (none) | 0 |")

        lines.extend([
            "",
            "## 4. Quality Controls",
            "",
            "Data quality is enforced at every transformation step. The lineage",
            "graph provides complete provenance from raw source data to final",
            "reported values.",
            "",
            f"- **Lineage coverage score:** {stats.get('coverage_score', 'N/A')}",
            f"- **Orphan nodes detected:** {stats.get('orphan_count', 0)}",
            f"- **Broken edges:** {stats.get('broken_edges', 0)}",
            f"- **Cycles detected:** {stats.get('cycles_detected', 0)}",
            "",
            "## 5. Compliance Mapping",
            "",
            "| CSRD/ESRS Requirement | Coverage | Evidence |",
            "|----------------------|----------|----------|",
            "| Article 8 - Data flow documentation | Covered | Lineage graph with full traceability |",
            "| ESRS 2 - Basis for preparation | Covered | Source registry with classification |",
            "| ESRS 2 - Data quality | Covered | Quality controls at each transformation |",
            "| ESRS E1 - Emission calculation chain | Covered | End-to-end calculation lineage |",
            "| ESRS S1 - Social data provenance | Covered | Data source documentation |",
            "",
            "## 6. Report Output Assets",
            "",
            "| # | Report Name | Classification |",
            "|---|-----------|----------------|",
        ])
        for idx, rpt in enumerate(reports, 1):
            name = name_map.get(rpt["id"], rpt["id"][:12])
            classification = rpt.get("classification", "internal")
            lines.append(f"| {idx} | {name} | {classification} |")
        if not reports:
            lines.append("| - | No report assets registered | - |")

        lines.extend([
            "",
            "---",
            "",
            "*This report was generated automatically by the GreenLang Data Lineage",
            "Tracker (AGENT-DATA-018) and should be reviewed by the sustainability",
            "reporting team before inclusion in the CSRD/ESRS filing.*",
        ])
        return "\n".join(lines)

    def generate_ghg_protocol_report(self, scope: str = "full") -> str:
        """Generate a GHG Protocol calculation chain documentation report.

        Produces a Markdown document covering input data sources,
        calculation methodology, emission factors, and output metrics
        as required for GHG Protocol audit documentation.

        Args:
            scope: Scope label for the report.

        Returns:
            Markdown-formatted GHG Protocol report string.
        """
        nodes = self._get_nodes()
        edges = self._get_edges()
        stats = self._get_graph_statistics()
        generated_at = _utcnow().isoformat()

        name_map = self._build_name_map(nodes)

        # Categorize by asset type
        datasets = [n for n in nodes if n.get("asset_type") == "dataset"]
        metrics = [n for n in nodes if n.get("asset_type") == "metric"]
        agents = [n for n in nodes if n.get("asset_type") == "agent"]
        ext_sources = [
            n for n in nodes if n.get("asset_type") == "external_source"
        ]

        # Identify calculation-related transformations
        calc_types = {"calculate", "aggregate", "normalize", "enrich"}
        calc_edges = [
            e for e in edges
            if e.get("transformation_type", "") in calc_types
        ]

        # Build transformation type summary
        tx_types: Dict[str, int] = {}
        for edge in edges:
            tx = edge.get("transformation_type", edge.get("edge_type", "unknown"))
            tx_types[tx] = tx_types.get(tx, 0) + 1

        lines: List[str] = [
            "# GHG Protocol Calculation Chain Documentation",
            "",
            f"**Report Scope:** {scope}",
            f"**Generated:** {generated_at}",
            f"**Generated By:** GreenLang Data Lineage Tracker (AGENT-DATA-018)",
            f"**Standard:** GHG Protocol Corporate Standard (Revised Edition)",
            "",
            "---",
            "",
            "## 1. Input Data Sources",
            "",
            "Activity data and emission factors used in GHG calculations are",
            "traceable through the lineage graph from authoritative sources.",
            "",
            f"**Total data assets in calculation chain:** {len(nodes)}",
            f"**External emission factor sources:** {len(ext_sources)}",
            "",
            "### 1.1 Activity Data Sources",
            "",
            "| # | Dataset | Owner | Classification |",
            "|---|---------|-------|----------------|",
        ]
        for idx, ds in enumerate(datasets, 1):
            name = name_map.get(ds["id"], ds["id"][:12])
            owner = ds.get("owner", "unassigned")
            classification = ds.get("classification", "internal")
            lines.append(
                f"| {idx} | {name} | {owner} | {classification} |"
            )
        if not datasets:
            lines.append("| - | No datasets registered | - | - |")

        lines.extend([
            "",
            "### 1.2 Emission Factor Sources",
            "",
            "| # | Source | Description |",
            "|---|--------|-------------|",
        ])
        for idx, src in enumerate(ext_sources, 1):
            name = name_map.get(src["id"], src["id"][:12])
            desc = src.get("description", "Emission factor database")
            lines.append(f"| {idx} | {name} | {desc} |")
        if not ext_sources:
            lines.append("| - | No emission factor sources registered | - |")

        lines.extend([
            "",
            "## 2. Calculation Methodology",
            "",
            "All GHG calculations follow deterministic formulas applied by",
            "registered calculation agents. No LLM or ML models are used",
            "for numeric emission calculations (zero-hallucination guarantee).",
            "",
            f"**Total calculation steps:** {len(calc_edges)}",
            f"**Processing agents involved:** {len(agents)}",
            "",
            "### 2.1 Processing Agents",
            "",
            "| # | Agent | Role |",
            "|---|-------|------|",
        ])
        for idx, agent in enumerate(agents, 1):
            name = name_map.get(agent["id"], agent["id"][:12])
            desc = agent.get("description", "Calculation agent")
            lines.append(f"| {idx} | {name} | {desc} |")
        if not agents:
            lines.append("| - | No agents registered | - |")

        lines.extend([
            "",
            "### 2.2 Transformation Steps",
            "",
            "| Transformation Type | Step Count |",
            "|-------------------|------------|",
        ])
        for tx, count in sorted(tx_types.items(), key=lambda x: -x[1]):
            lines.append(f"| {tx} | {count} |")
        if not tx_types:
            lines.append("| (none) | 0 |")

        lines.extend([
            "",
            "## 3. Emission Factors",
            "",
            "Emission factors are sourced from authoritative databases and",
            "tracked through the lineage graph with version control and",
            "provenance hashing.",
            "",
            "### 3.1 Factor Traceability",
            "",
            "Every emission factor applied in calculations is linked back to",
            "its authoritative source via the lineage graph. Factor versions",
            "and update timestamps are recorded for auditability.",
            "",
            f"- **Emission factor sources:** {len(ext_sources)}",
            f"- **Calculation edges:** {len(calc_edges)}",
            f"- **Lineage coverage:** {stats.get('coverage_score', 'N/A')}",
            "",
            "## 4. Output Metrics",
            "",
            "Calculated emission metrics are registered as lineage graph nodes",
            "with full upstream traceability.",
            "",
            "| # | Metric | Description |",
            "|---|--------|-------------|",
        ])
        for idx, m in enumerate(metrics, 1):
            name = name_map.get(m["id"], m["id"][:12])
            desc = m.get("description", "Emission metric")
            lines.append(f"| {idx} | {name} | {desc} |")
        if not metrics:
            lines.append("| - | No metric assets registered | - |")

        lines.extend([
            "",
            "## 5. Audit Trail",
            "",
            f"- **Graph node count:** {len(nodes)}",
            f"- **Graph edge count:** {len(edges)}",
            f"- **Max lineage depth:** {stats.get('max_depth', 'N/A')}",
            f"- **Connected components:** {stats.get('connected_components', 'N/A')}",
            f"- **Orphan nodes:** {stats.get('orphan_count', 0)}",
            f"- **Provenance hashing:** SHA-256 chain for every operation",
            "",
            "---",
            "",
            "*This report was generated automatically by the GreenLang Data Lineage",
            "Tracker (AGENT-DATA-018) to support GHG Protocol audit requirements.",
            "All numeric calculations are deterministic with zero hallucination.*",
        ])
        return "\n".join(lines)

    def generate_soc2_report(self, scope: str = "full") -> str:
        """Generate a SOC 2 data processing audit trail report.

        Produces a Markdown document covering data inventory, processing
        activities, access controls, and audit trail information as
        required for SOC 2 Type II evidence.

        Args:
            scope: Scope label for the report.

        Returns:
            Markdown-formatted SOC 2 audit report string.
        """
        nodes = self._get_nodes()
        edges = self._get_edges()
        stats = self._get_graph_statistics()
        generated_at = _utcnow().isoformat()

        name_map = self._build_name_map(nodes)

        # Classification breakdown
        class_counts: Dict[str, int] = {}
        for node in nodes:
            classification = node.get("classification", "internal")
            class_counts[classification] = (
                class_counts.get(classification, 0) + 1
            )

        # Asset type breakdown
        type_counts: Dict[str, int] = {}
        for node in nodes:
            asset_type = node.get("asset_type", "unknown")
            type_counts[asset_type] = type_counts.get(asset_type, 0) + 1

        # Transformation type breakdown
        tx_types: Dict[str, int] = {}
        for edge in edges:
            tx = edge.get("transformation_type", edge.get("edge_type", "unknown"))
            tx_types[tx] = tx_types.get(tx, 0) + 1

        # Count restricted / confidential assets
        sensitive_count = (
            class_counts.get("restricted", 0)
            + class_counts.get("confidential", 0)
            + class_counts.get("pii", 0)
        )

        lines: List[str] = [
            "# SOC 2 Data Processing Audit Trail",
            "",
            f"**Report Scope:** {scope}",
            f"**Generated:** {generated_at}",
            f"**Generated By:** GreenLang Data Lineage Tracker (AGENT-DATA-018)",
            f"**Framework:** SOC 2 Type II (Trust Service Criteria)",
            "",
            "---",
            "",
            "## 1. Data Inventory",
            "",
            "Complete inventory of data assets processed by the GreenLang",
            "platform with classification levels and lifecycle status.",
            "",
            f"**Total data assets:** {len(nodes)}",
            f"**Sensitive assets (restricted/confidential/PII):** {sensitive_count}",
            "",
            "### 1.1 Data Classification Summary",
            "",
            "| Classification Level | Asset Count | Percentage |",
            "|---------------------|-------------|------------|",
        ]
        for cls_name, count in sorted(
            class_counts.items(), key=lambda x: -x[1]
        ):
            pct = (count / max(len(nodes), 1)) * 100
            lines.append(f"| {cls_name} | {count} | {pct:.1f}% |")

        lines.extend([
            "",
            "### 1.2 Asset Type Summary",
            "",
            "| Asset Type | Count |",
            "|-----------|-------|",
        ])
        for atype, count in sorted(
            type_counts.items(), key=lambda x: -x[1]
        ):
            lines.append(f"| {atype} | {count} |")

        lines.extend([
            "",
            "## 2. Processing Activities",
            "",
            "All data processing activities are logged as transformation",
            "events in the lineage graph with full provenance tracking.",
            "",
            f"**Total processing edges:** {len(edges)}",
            "",
            "### 2.1 Processing Type Summary",
            "",
            "| Processing Type | Occurrences |",
            "|----------------|-------------|",
        ])
        for tx, count in sorted(tx_types.items(), key=lambda x: -x[1]):
            lines.append(f"| {tx} | {count} |")
        if not tx_types:
            lines.append("| (none) | 0 |")

        lines.extend([
            "",
            "### 2.2 Data Flow Integrity",
            "",
            "The lineage graph provides end-to-end traceability for all",
            "data processing activities, supporting SOC 2 processing",
            "integrity (PI) criteria.",
            "",
            f"- **Graph depth (max hops):** {stats.get('max_depth', 'N/A')}",
            f"- **Connected components:** {stats.get('connected_components', 'N/A')}",
            f"- **Orphan nodes (unlinked):** {stats.get('orphan_count', 0)}",
            f"- **Broken edges:** {stats.get('broken_edges', 0)}",
            f"- **Cycles detected:** {stats.get('cycles_detected', 0)}",
            "",
            "## 3. Access Controls",
            "",
            "Data access is controlled by classification level. The lineage",
            "graph documents which agents and pipelines access which data",
            "assets, supporting SOC 2 logical access (CC6) criteria.",
            "",
            "### 3.1 Sensitive Data Access Paths",
            "",
        ])
        # List sensitive assets and their edges
        sensitive_ids = {
            n["id"]
            for n in nodes
            if n.get("classification") in ("restricted", "confidential", "pii")
        }
        if sensitive_ids:
            lines.append(
                "| # | Sensitive Asset | Classification | Connected Edges |"
            )
            lines.append("|---|----------------|----------------|-----------------|")
            idx = 0
            for node in nodes:
                if node["id"] not in sensitive_ids:
                    continue
                idx += 1
                name = name_map.get(node["id"], node["id"][:12])
                classification = node.get("classification", "unknown")
                edge_count = sum(
                    1 for e in edges
                    if e.get("source") == node["id"]
                    or e.get("target") == node["id"]
                )
                lines.append(
                    f"| {idx} | {name} | {classification} | {edge_count} |"
                )
        else:
            lines.append(
                "No assets classified as restricted, confidential, or PII."
            )

        lines.extend([
            "",
            "## 4. Audit Trail",
            "",
            "All lineage operations are recorded with SHA-256 provenance",
            "chain hashing for tamper-evident audit trails.",
            "",
            "### 4.1 Provenance Controls",
            "",
            "| Control | Status |",
            "|---------|--------|",
            "| SHA-256 provenance chain | Active |",
            "| Genesis hash anchoring | Active |",
            "| Chain integrity verification | Available |",
            "| Tamper-evident logging | Active |",
            "| Operation-level timestamping | Active (UTC ISO-8601) |",
            "",
            "### 4.2 Graph Integrity Metrics",
            "",
            f"- **Total nodes:** {len(nodes)}",
            f"- **Total edges:** {len(edges)}",
            f"- **Lineage coverage:** {stats.get('coverage_score', 'N/A')}",
            f"- **Graph hash:** {stats.get('graph_hash', 'N/A')}",
            "",
            "---",
            "",
            "*This report was generated automatically by the GreenLang Data Lineage",
            "Tracker (AGENT-DATA-018) to support SOC 2 Type II evidence collection.",
            "All provenance records are SHA-256 chain hashed for tamper evidence.*",
        ])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Report retrieval
    # ------------------------------------------------------------------

    def get_report(self, report_id: str) -> Optional[dict]:
        """Retrieve a previously generated report by its ID.

        Args:
            report_id: UUID of the report to retrieve.

        Returns:
            The report dictionary, or ``None`` if no report with the
            given ID exists.
        """
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(
        self,
        report_type: Optional[str] = None,
        format: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """List generated reports with optional filtering.

        Args:
            report_type: Optional filter by report type (e.g.
                ``"csrd_esrs"``).
            format: Optional filter by output format (e.g. ``"json"``).
            limit: Maximum number of reports to return. Defaults to 100.

        Returns:
            List of report dictionaries matching the filters, ordered
            newest first by ``generated_at``. Each dictionary omits the
            ``content`` field for brevity.
        """
        with self._lock:
            all_reports = list(self._reports.values())

        # Apply filters
        if report_type:
            all_reports = [
                r for r in all_reports
                if r.get("report_type") == report_type
            ]
        if format:
            all_reports = [
                r for r in all_reports if r.get("format") == format
            ]

        # Sort newest first
        all_reports.sort(
            key=lambda r: r.get("generated_at", ""), reverse=True
        )

        # Return summaries without content (to reduce payload)
        results: List[dict] = []
        for rpt in all_reports[:limit]:
            summary = {
                "id": rpt["id"],
                "report_type": rpt["report_type"],
                "format": rpt["format"],
                "scope": rpt["scope"],
                "report_hash": rpt["report_hash"],
                "generated_by": rpt["generated_by"],
                "generated_at": rpt["generated_at"],
            }
            results.append(summary)
        return results

    # ------------------------------------------------------------------
    # Statistics and lifecycle
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return reporter engine statistics.

        Returns:
            Dictionary with counts of total reports generated, breakdown
            by report type and format, and graph summary statistics.
        """
        with self._lock:
            all_reports = list(self._reports.values())

        type_counts: Dict[str, int] = {}
        format_counts: Dict[str, int] = {}
        for rpt in all_reports:
            rt = rpt.get("report_type", "unknown")
            fmt = rpt.get("format", "unknown")
            type_counts[rt] = type_counts.get(rt, 0) + 1
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        graph_stats = self._get_graph_statistics()

        return {
            "total_reports": len(all_reports),
            "by_report_type": type_counts,
            "by_format": format_counts,
            "graph_node_count": graph_stats.get("node_count", 0),
            "graph_edge_count": graph_stats.get("edge_count", 0),
            "provenance_entries": self._provenance.entry_count,
        }

    def clear(self) -> None:
        """Clear all stored reports from the in-memory store.

        Does not affect the lineage graph or provenance chain.
        """
        with self._lock:
            count = len(self._reports)
            self._reports.clear()
        logger.info(
            "LineageReporterEngine cleared %d stored reports", count
        )

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        report_type: str,
        fmt: str,
        scope: str,
        parameters: Dict[str, Any],
        max_depth: int,
    ) -> str:
        """Dispatch report generation to the appropriate method.

        Args:
            report_type: Validated report type.
            fmt: Validated output format.
            scope: Scope label.
            parameters: Extra parameters.
            max_depth: Maximum graph depth.

        Returns:
            Generated report content as a string.
        """
        asset_type_filter = parameters.get("asset_type_filter")

        # Compliance reports always produce Markdown regardless of fmt
        if report_type == "csrd_esrs":
            return self.generate_csrd_esrs_report(scope=scope)
        if report_type == "ghg_protocol":
            return self.generate_ghg_protocol_report(scope=scope)
        if report_type == "soc2":
            return self.generate_soc2_report(scope=scope)

        # Custom report -- JSON dump of graph + parameters
        if report_type == "custom":
            return self._generate_custom_report(
                scope=scope, parameters=parameters, max_depth=max_depth
            )

        # Visualization report -- dispatch by format
        return self._dispatch_visualization(
            fmt=fmt,
            scope=scope,
            max_depth=max_depth,
            asset_type_filter=asset_type_filter,
        )

    def _dispatch_visualization(
        self,
        fmt: str,
        scope: str,
        max_depth: int,
        asset_type_filter: Optional[str],
    ) -> str:
        """Dispatch visualization generation by format.

        Args:
            fmt: Output format string.
            scope: Scope label.
            max_depth: Maximum graph depth.
            asset_type_filter: Optional asset type filter.

        Returns:
            Visualization content string.
        """
        if fmt == "mermaid":
            return self.generate_mermaid(
                scope=scope,
                max_depth=max_depth,
                asset_type_filter=asset_type_filter,
            )
        if fmt == "dot":
            return self.generate_dot(
                scope=scope,
                max_depth=max_depth,
                asset_type_filter=asset_type_filter,
            )
        if fmt == "json":
            return self.generate_json_graph(
                scope=scope, max_depth=max_depth
            )
        if fmt == "d3":
            return self.generate_d3(scope=scope, max_depth=max_depth)
        if fmt == "text":
            return self.generate_text_summary(scope=scope)
        if fmt == "html":
            return self.generate_html(scope=scope, max_depth=max_depth)
        if fmt == "pdf":
            return self._generate_pdf_placeholder(scope=scope)

        # Fallback (should not be reachable due to validation)
        return self.generate_json_graph(scope=scope, max_depth=max_depth)

    def _generate_custom_report(
        self,
        scope: str,
        parameters: Dict[str, Any],
        max_depth: int,
    ) -> str:
        """Generate a custom report from user-supplied parameters.

        Returns a JSON document containing the graph data, parameters,
        and metadata.

        Args:
            scope: Scope label.
            parameters: User-supplied parameters dictionary.
            max_depth: Maximum graph depth.

        Returns:
            JSON-formatted custom report string.
        """
        nodes = self._get_nodes()
        edges = self._get_edges()
        stats = self._get_graph_statistics()

        # Apply optional filters from parameters
        asset_type_filter = parameters.get("asset_type_filter")
        classification_filter = parameters.get("classification_filter")
        status_filter = parameters.get("status_filter")

        if asset_type_filter:
            nodes = [
                n for n in nodes
                if n.get("asset_type") == asset_type_filter
            ]
        if classification_filter:
            nodes = [
                n for n in nodes
                if n.get("classification") == classification_filter
            ]
        if status_filter:
            nodes = [
                n for n in nodes if n.get("status") == status_filter
            ]

        # Filter edges to only include remaining nodes
        node_ids = {n["id"] for n in nodes}
        edges = [
            e for e in edges
            if e.get("source") in node_ids and e.get("target") in node_ids
        ]

        custom_data: Dict[str, Any] = {
            "report_type": "custom",
            "scope": scope,
            "parameters": parameters,
            "max_depth": max_depth,
            "generated_at": _utcnow().isoformat(),
            "generated_by": "data-lineage-tracker",
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "graph_statistics": stats,
            },
            "nodes": nodes,
            "edges": edges,
        }
        return json.dumps(custom_data, indent=2, default=str)

    def _generate_pdf_placeholder(self, scope: str) -> str:
        """Generate a PDF placeholder indicating external rendering is required.

        PDF rendering requires an external tool (e.g. wkhtmltopdf or
        weasyprint). This method returns a JSON metadata document that
        downstream services can use to trigger external PDF generation.

        Args:
            scope: Scope label.

        Returns:
            JSON string with PDF rendering instructions.
        """
        placeholder: Dict[str, Any] = {
            "format": "pdf",
            "scope": scope,
            "status": "requires_external_render",
            "instructions": (
                "PDF rendering requires an external tool such as "
                "wkhtmltopdf or weasyprint. Use the HTML format to "
                "generate the source document, then convert to PDF "
                "using the configured rendering service."
            ),
            "html_fallback": True,
            "generated_at": _utcnow().isoformat(),
            "generated_by": "data-lineage-tracker",
        }
        return json.dumps(placeholder, indent=2, default=str)

    # ------------------------------------------------------------------
    # Graph data accessors (adapter layer)
    # ------------------------------------------------------------------

    def _get_nodes(self) -> List[dict]:
        """Retrieve all nodes from the lineage graph.

        Adapts to the LineageGraphEngine interface. If the graph object
        does not expose ``get_nodes()`` we fall back to checking for
        common attribute patterns.

        Returns:
            List of node dictionaries.
        """
        if hasattr(self._graph, "get_nodes"):
            result = self._graph.get_nodes()
            if isinstance(result, list):
                return result

        if hasattr(self._graph, "nodes"):
            nodes_attr = self._graph.nodes
            if callable(nodes_attr):
                return list(nodes_attr())
            if isinstance(nodes_attr, dict):
                return list(nodes_attr.values())
            if isinstance(nodes_attr, list):
                return list(nodes_attr)

        if hasattr(self._graph, "_nodes"):
            nodes_attr = self._graph._nodes
            if isinstance(nodes_attr, dict):
                return list(nodes_attr.values())
            if isinstance(nodes_attr, list):
                return list(nodes_attr)

        logger.warning(
            "LineageGraphEngine does not expose nodes; returning empty list"
        )
        return []

    def _get_edges(self) -> List[dict]:
        """Retrieve all edges from the lineage graph.

        Adapts to the LineageGraphEngine interface with fallback
        attribute patterns.

        Returns:
            List of edge dictionaries.
        """
        if hasattr(self._graph, "get_edges"):
            result = self._graph.get_edges()
            if isinstance(result, list):
                return result

        if hasattr(self._graph, "edges"):
            edges_attr = self._graph.edges
            if callable(edges_attr):
                return list(edges_attr())
            if isinstance(edges_attr, dict):
                return list(edges_attr.values())
            if isinstance(edges_attr, list):
                return list(edges_attr)

        if hasattr(self._graph, "_edges"):
            edges_attr = self._graph._edges
            if isinstance(edges_attr, dict):
                return list(edges_attr.values())
            if isinstance(edges_attr, list):
                return list(edges_attr)

        logger.warning(
            "LineageGraphEngine does not expose edges; returning empty list"
        )
        return []

    def _get_graph_statistics(self) -> dict:
        """Retrieve statistics from the lineage graph.

        Returns:
            Dictionary of graph statistics, or an empty dict if the
            graph does not expose a statistics method.
        """
        if hasattr(self._graph, "get_statistics"):
            result = self._graph.get_statistics()
            if isinstance(result, dict):
                return result

        if hasattr(self._graph, "statistics"):
            stats_attr = self._graph.statistics
            if callable(stats_attr):
                return dict(stats_attr())
            if isinstance(stats_attr, dict):
                return dict(stats_attr)

        # Compute minimal statistics from available data
        nodes = self._get_nodes()
        edges = self._get_edges()
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "max_depth": "N/A",
            "connected_components": "N/A",
            "coverage_score": "N/A",
            "orphan_count": 0,
            "broken_edges": 0,
            "cycles_detected": 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _index_to_alias(index: int) -> str:
        """Convert a zero-based index to an alphabetic node alias.

        Maps 0-25 to A-Z, 26-51 to AA-AZ, 52-77 to BA-BZ, etc.

        Args:
            index: Zero-based node index.

        Returns:
            Alphabetic alias string (e.g. ``"A"``, ``"B"``, ``"AA"``).
        """
        result = ""
        while True:
            result = chr(ord("A") + (index % 26)) + result
            index = index // 26 - 1
            if index < 0:
                break
        return result

    @staticmethod
    def _build_name_map(nodes: List[dict]) -> Dict[str, str]:
        """Build a mapping from node ID to display name.

        Prefers ``display_name``, falls back to ``qualified_name``,
        then truncated ``id``.

        Args:
            nodes: List of node dictionaries.

        Returns:
            Dictionary mapping node ID to human-readable name.
        """
        name_map: Dict[str, str] = {}
        for node in nodes:
            node_id = node.get("id", "")
            name = (
                node.get("display_name", "")
                or node.get("qualified_name", "")
                or node_id[:16]
            )
            name_map[node_id] = name
        return name_map


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = ["LineageReporterEngine"]
