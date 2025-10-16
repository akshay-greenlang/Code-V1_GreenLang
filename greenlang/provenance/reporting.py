"""
GreenLang Provenance - Reporting Module
Audit report generation and visualization.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging

from .records import ProvenanceRecord
from .validation import validate_provenance

logger = logging.getLogger(__name__)


# ============================================================================
# MARKDOWN REPORT GENERATION
# ============================================================================

def generate_markdown_report(provenance: ProvenanceRecord) -> str:
    """
    Generate human-readable Markdown audit report from provenance record.

    Args:
        provenance: ProvenanceRecord to report on

    Returns:
        Markdown-formatted audit report

    Example:
        >>> report = generate_markdown_report(provenance)
        >>> print(report)
        >>> # Or save to file:
        >>> with open("audit_report.md", "w") as f:
        ...     f.write(report)
    """
    lines = []

    # Header
    lines.append("# PROVENANCE AUDIT REPORT")
    lines.append("")
    lines.append(f"**Record ID:** {provenance.record_id}")
    lines.append(f"**Generated:** {provenance.generated_at}")
    lines.append("")

    # Input File Integrity (if available)
    if provenance.input_file_hash:
        lines.append("## Input File Integrity")
        lines.append("")
        lines.append(f"- **File:** {provenance.input_file_hash.get('file_name', 'N/A')}")
        lines.append(f"- **Size:** {provenance.input_file_hash.get('human_readable_size', 'N/A')}")
        lines.append(f"- **SHA256:** `{provenance.input_file_hash.get('hash_value', 'N/A')}`")
        lines.append(f"- **Hashed:** {provenance.input_file_hash.get('hash_timestamp', 'N/A')}")
        lines.append("")

    # Execution Environment
    lines.append("## Execution Environment")
    lines.append("")

    env = provenance.environment
    if "python" in env:
        py = env["python"]
        if "version_info" in py:
            vi = py["version_info"]
            lines.append(f"- **Python:** {vi.get('major', '?')}.{vi.get('minor', '?')}.{vi.get('micro', '?')}")
        if "implementation" in py:
            lines.append(f"- **Implementation:** {py['implementation']}")

    if "system" in env:
        sys = env["system"]
        lines.append(f"- **OS:** {sys.get('os', 'N/A')} {sys.get('release', '')}")
        lines.append(f"- **Machine:** {sys.get('machine', 'N/A')}")
        if "cpu_count" in sys:
            lines.append(f"- **CPUs:** {sys['cpu_count']}")

    if "process" in env:
        proc = env["process"]
        lines.append(f"- **User:** {proc.get('user', 'N/A')}")
        lines.append(f"- **Working Directory:** {proc.get('cwd', 'N/A')}")

    lines.append("")

    # Dependencies
    lines.append("## Dependencies")
    lines.append("")
    for pkg, version in provenance.dependencies.items():
        lines.append(f"- **{pkg}**: {version}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    if provenance.configuration:
        lines.append("```json")
        lines.append(json.dumps(provenance.configuration, indent=2))
        lines.append("```")
    else:
        lines.append("*No configuration recorded*")
    lines.append("")

    # Agent Execution
    if provenance.agent_execution:
        lines.append("## Agent Execution")
        lines.append("")
        for execution in provenance.agent_execution:
            agent_name = execution.get('agent_name', 'Unknown Agent')
            lines.append(f"### {agent_name}")
            lines.append(f"- **Started:** {execution.get('start_time', 'N/A')}")
            lines.append(f"- **Ended:** {execution.get('end_time', 'N/A')}")
            lines.append(f"- **Duration:** {execution.get('duration_seconds', 'N/A')}s")
            lines.append(f"- **Input Records:** {execution.get('input_records', 0)}")
            lines.append(f"- **Output Records:** {execution.get('output_records', 0)}")
            lines.append("")

    # Data Lineage
    if provenance.data_lineage:
        lines.append("## Data Lineage")
        lines.append("")
        for event in provenance.data_lineage:
            step = event.get('step', '?')
            stage = event.get('stage', 'Unknown')
            description = event.get('description', '')
            lines.append(f"{step}. **{stage}**: {description}")
        lines.append("")

    # Validation Results
    if provenance.validation_results:
        lines.append("## Validation Results")
        lines.append("")
        is_valid = provenance.validation_results.get('is_valid', False)
        status = '✓ PASS' if is_valid else '✗ FAIL'
        lines.append(f"- **Status:** {status}")
        lines.append(f"- **Errors:** {len(provenance.validation_results.get('errors', []))}")
        lines.append(f"- **Warnings:** {len(provenance.validation_results.get('warnings', []))}")
        lines.append("")

    # Metadata
    if provenance.metadata:
        lines.append("## Additional Metadata")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(provenance.metadata, indent=2))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def generate_audit_report(
    provenance: ProvenanceRecord,
    format: str = "markdown",
    output_path: Optional[str] = None
) -> str:
    """
    Generate audit report in specified format.

    Args:
        provenance: ProvenanceRecord to report on
        format: Report format ('markdown', 'json', 'html')
        output_path: Optional path to save report

    Returns:
        Report content as string

    Example:
        >>> report = generate_audit_report(provenance, format="markdown")
        >>> # Or save directly:
        >>> generate_audit_report(provenance, format="html", output_path="report.html")
    """
    if format == "markdown":
        content = generate_markdown_report(provenance)
    elif format == "json":
        content = provenance.to_json(indent=2)
    elif format == "html":
        content = generate_html_report(provenance)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if output_path:
        with open(output_path, 'w') as f:
            f.write(content)
        logger.info(f"Audit report saved to {output_path}")

    return content


def generate_html_report(provenance: ProvenanceRecord) -> str:
    """
    Generate HTML audit report.

    Args:
        provenance: ProvenanceRecord to report on

    Returns:
        HTML report
    """
    # Convert markdown to HTML structure
    markdown = generate_markdown_report(provenance)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Provenance Audit Report - {provenance.record_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }}
        h3 {{ color: #7f8c8d; }}
        code {{ background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .status-pass {{ color: #27ae60; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        ul {{ line-height: 1.8; }}
        .metadata {{ background: #ecf0f1; padding: 15px; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PROVENANCE AUDIT REPORT</h1>
        <p><strong>Record ID:</strong> {provenance.record_id}</p>
        <p><strong>Generated:</strong> {provenance.generated_at}</p>
"""

    # Add sections from markdown (simplified HTML conversion)
    markdown_lines = markdown.split('\n')
    in_code_block = False

    for line in markdown_lines:
        if line.startswith('```'):
            if in_code_block:
                html += "</pre>\n"
                in_code_block = False
            else:
                html += "<pre>"
                in_code_block = True
        elif in_code_block:
            html += line + "\n"
        elif line.startswith('###'):
            html += f"<h3>{line[4:]}</h3>\n"
        elif line.startswith('##'):
            html += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith('#'):
            continue  # Skip main header (already added)
        elif line.startswith('- '):
            html += f"<li>{line[2:]}</li>\n"
        elif line.strip():
            html += f"<p>{line}</p>\n"

    html += """
    </div>
</body>
</html>"""

    return html


def generate_summary_report(provenance_records: List[ProvenanceRecord]) -> str:
    """
    Generate summary report for multiple provenance records.

    Args:
        provenance_records: List of ProvenanceRecords

    Returns:
        Summary report in Markdown format

    Example:
        >>> records = [record1, record2, record3]
        >>> summary = generate_summary_report(records)
        >>> print(summary)
    """
    lines = []

    lines.append("# PROVENANCE SUMMARY REPORT")
    lines.append("")
    lines.append(f"**Total Records:** {len(provenance_records)}")
    lines.append("")

    # Validation summary
    lines.append("## Validation Summary")
    lines.append("")
    valid_count = 0
    invalid_count = 0

    for record in provenance_records:
        result = validate_provenance(record)
        if result["is_valid"]:
            valid_count += 1
        else:
            invalid_count += 1

    lines.append(f"- **Valid Records:** {valid_count}")
    lines.append(f"- **Invalid Records:** {invalid_count}")
    lines.append("")

    # Individual records
    lines.append("## Individual Records")
    lines.append("")

    for record in provenance_records:
        result = validate_provenance(record)
        status = "✓" if result["is_valid"] else "✗"
        lines.append(f"- {status} **{record.record_id}** ({record.generated_at})")

    lines.append("")

    return "\n".join(lines)
