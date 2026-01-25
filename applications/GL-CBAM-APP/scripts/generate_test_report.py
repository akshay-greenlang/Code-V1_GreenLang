#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GL-CBAM-APP - Test Report Generator

Purpose: Generate comprehensive test execution report with coverage analysis,
         test results summary, and validation status.

Usage:
    python scripts/generate_test_report.py                    # Generate report
    python scripts/generate_test_report.py --format html      # HTML format
    python scripts/generate_test_report.py --format markdown  # Markdown format
    python scripts/generate_test_report.py --format json      # JSON format

Inputs:
    - test-results/*.xml (JUnit XML)
    - coverage.json (coverage data)
    - .pytest_cache (pytest cache)

Outputs:
    - test-report.html (default)
    - test-report.md
    - test-report.json

Version: 1.0.0
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
from greenlang.determinism import DeterministicClock

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

# Report templates
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GL-CBAM Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            opacity: 0.9;
            margin-top: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .metric {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric.pass {{ color: #10b981; }}
        .metric.fail {{ color: #ef4444; }}
        .metric.warning {{ color: #f59e0b; }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.875em;
            font-weight: 600;
        }}
        .badge.success {{ background: #d1fae5; color: #065f46; }}
        .badge.failure {{ background: #fee2e2; color: #991b1b; }}
        .badge.skipped {{ background: #fef3c7; color: #92400e; }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #6b7280;
            font-size: 0.875em;
        }}
        .recommendation {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 15px 0;
        }}
        .recommendation.success {{
            background: #d1fae5;
            border-left-color: #10b981;
        }}
        .recommendation.error {{
            background: #fee2e2;
            border-left-color: #ef4444;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GL-CBAM Test Report</h1>
        <div class="subtitle">Comprehensive Test Execution & Validation Report</div>
        <div class="subtitle">Generated: {timestamp}</div>
    </div>

    <div class="summary">
        <div class="card">
            <h3>Total Tests</h3>
            <div class="metric">{total_tests}</div>
        </div>
        <div class="card">
            <h3>Passed</h3>
            <div class="metric pass">{passed_tests}</div>
        </div>
        <div class="card">
            <h3>Failed</h3>
            <div class="metric fail">{failed_tests}</div>
        </div>
        <div class="card">
            <h3>Code Coverage</h3>
            <div class="metric {coverage_class}">{coverage_percent}%</div>
        </div>
    </div>

    {sections}

    <div class="footer">
        <p>Generated by GL-CBAM Test Report Generator v1.0.0</p>
        <p>GL-CBAM-APP - Carbon Border Adjustment Mechanism</p>
    </div>
</body>
</html>
"""

# ============================================================================
# Data Parsers
# ============================================================================

def parse_junit_xml(xml_path: Path) -> Dict[str, Any]:
    """Parse JUnit XML test results."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract test suite information
        testsuite = root.find('.//testsuite') or root

        total = int(testsuite.get('tests', 0))
        failures = int(testsuite.get('failures', 0))
        errors = int(testsuite.get('errors', 0))
        skipped = int(testsuite.get('skipped', 0))
        time = float(testsuite.get('time', 0))

        passed = total - failures - errors - skipped

        # Extract individual test cases
        test_cases = []
        for testcase in root.findall('.//testcase'):
            test_info = {
                'name': testcase.get('name'),
                'classname': testcase.get('classname'),
                'time': float(testcase.get('time', 0)),
                'status': 'passed'
            }

            # Check for failures/errors
            if testcase.find('failure') is not None:
                test_info['status'] = 'failed'
                test_info['message'] = testcase.find('failure').get('message', '')
            elif testcase.find('error') is not None:
                test_info['status'] = 'error'
                test_info['message'] = testcase.find('error').get('message', '')
            elif testcase.find('skipped') is not None:
                test_info['status'] = 'skipped'

            test_cases.append(test_info)

        return {
            'total': total,
            'passed': passed,
            'failed': failures + errors,
            'skipped': skipped,
            'duration': time,
            'test_cases': test_cases
        }

    except Exception as e:
        print(f"Warning: Could not parse JUnit XML: {e}")
        return {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'duration': 0,
            'test_cases': []
        }


def parse_coverage_json(json_path: Path) -> Dict[str, Any]:
    """Parse coverage.json file."""
    try:
        with open(json_path) as f:
            data = json.load(f)

        totals = data.get('totals', {})
        percent = totals.get('percent_covered', 0)

        return {
            'percent_covered': round(percent, 2),
            'lines_covered': totals.get('covered_lines', 0),
            'lines_total': totals.get('num_statements', 0),
            'branches_covered': totals.get('covered_branches', 0),
            'branches_total': totals.get('num_branches', 0),
            'files': data.get('files', {})
        }

    except Exception as e:
        print(f"Warning: Could not parse coverage.json: {e}")
        return {
            'percent_covered': 0,
            'lines_covered': 0,
            'lines_total': 0,
            'branches_covered': 0,
            'branches_total': 0,
            'files': {}
        }


# ============================================================================
# Report Generation
# ============================================================================

def generate_html_report(data: Dict[str, Any]) -> str:
    """Generate HTML test report."""
    test_results = data.get('test_results', {})
    coverage = data.get('coverage', {})

    # Calculate coverage class
    coverage_percent = coverage.get('percent_covered', 0)
    if coverage_percent >= 80:
        coverage_class = 'pass'
    elif coverage_percent >= 60:
        coverage_class = 'warning'
    else:
        coverage_class = 'fail'

    # Generate sections
    sections = []

    # Test Results Section
    sections.append(f"""
    <div class="section">
        <h2>Test Execution Summary</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {test_results.get('passed', 0) / max(test_results.get('total', 1), 1) * 100}%">
                {test_results.get('passed', 0)}/{test_results.get('total', 0)} Passed
            </div>
        </div>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{test_results.get('total', 0)}</td></tr>
            <tr><td>Passed</td><td><span class="badge success">{test_results.get('passed', 0)}</span></td></tr>
            <tr><td>Failed</td><td><span class="badge failure">{test_results.get('failed', 0)}</span></td></tr>
            <tr><td>Skipped</td><td><span class="badge skipped">{test_results.get('skipped', 0)}</span></td></tr>
            <tr><td>Duration</td><td>{test_results.get('duration', 0):.2f} seconds</td></tr>
            <tr><td>Success Rate</td><td>{(test_results.get('passed', 0) / max(test_results.get('total', 1), 1) * 100):.1f}%</td></tr>
        </table>
    </div>
    """)

    # Coverage Section
    sections.append(f"""
    <div class="section">
        <h2>Code Coverage</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {coverage_percent}%">
                {coverage_percent}%
            </div>
        </div>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Coverage Percentage</td><td><strong>{coverage_percent}%</strong></td></tr>
            <tr><td>Lines Covered</td><td>{coverage.get('lines_covered', 0)} / {coverage.get('lines_total', 0)}</td></tr>
            <tr><td>Branches Covered</td><td>{coverage.get('branches_covered', 0)} / {coverage.get('branches_total', 0)}</td></tr>
            <tr><td>Files Analyzed</td><td>{len(coverage.get('files', {}))}</td></tr>
        </table>
    </div>
    """)

    # Recommendations
    recommendations = []
    if test_results.get('failed', 0) > 0:
        recommendations.append(
            '<div class="recommendation error"><strong>Action Required:</strong> '
            f'{test_results.get("failed", 0)} test(s) failed. Review failures and fix issues.</div>'
        )
    else:
        recommendations.append(
            '<div class="recommendation success"><strong>All Tests Passed!</strong> '
            'No test failures detected.</div>'
        )

    if coverage_percent < 80:
        recommendations.append(
            '<div class="recommendation"><strong>Coverage Improvement:</strong> '
            f'Current coverage is {coverage_percent}%. Target is 80%+. Add more tests.</div>'
        )
    else:
        recommendations.append(
            '<div class="recommendation success"><strong>Coverage Target Met!</strong> '
            f'Coverage is {coverage_percent}% (≥80%).</div>'
        )

    sections.append(f"""
    <div class="section">
        <h2>Recommendations</h2>
        {''.join(recommendations)}
    </div>
    """)

    # Fill template
    html = HTML_TEMPLATE.format(
        timestamp=DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_tests=test_results.get('total', 0),
        passed_tests=test_results.get('passed', 0),
        failed_tests=test_results.get('failed', 0),
        coverage_percent=coverage_percent,
        coverage_class=coverage_class,
        sections=''.join(sections)
    )

    return html


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generate Markdown test report."""
    test_results = data.get('test_results', {})
    coverage = data.get('coverage', {})

    md = f"""# GL-CBAM Test Report

**Generated:** {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Execution Summary

| Metric | Value |
|--------|-------|
| Total Tests | {test_results.get('total', 0)} |
| Passed | {test_results.get('passed', 0)} ✓ |
| Failed | {test_results.get('failed', 0)} ✗ |
| Skipped | {test_results.get('skipped', 0)} - |
| Duration | {test_results.get('duration', 0):.2f}s |
| Success Rate | {(test_results.get('passed', 0) / max(test_results.get('total', 1), 1) * 100):.1f}% |

## Code Coverage

| Metric | Value |
|--------|-------|
| Coverage Percentage | **{coverage.get('percent_covered', 0):.2f}%** |
| Lines Covered | {coverage.get('lines_covered', 0)} / {coverage.get('lines_total', 0)} |
| Branches Covered | {coverage.get('branches_covered', 0)} / {coverage.get('branches_total', 0)} |
| Files Analyzed | {len(coverage.get('files', {}))} |

## Status

"""

    if test_results.get('failed', 0) == 0:
        md += "✓ **ALL TESTS PASSED**\n\n"
    else:
        md += f"✗ **{test_results.get('failed', 0)} TEST(S) FAILED**\n\n"

    if coverage.get('percent_covered', 0) >= 80:
        md += "✓ **COVERAGE TARGET MET** (≥80%)\n\n"
    else:
        md += f"⚠ **COVERAGE BELOW TARGET** ({coverage.get('percent_covered', 0):.1f}% < 80%)\n\n"

    md += "\n---\n\nGenerated by GL-CBAM Test Report Generator v1.0.0\n"

    return md


def generate_json_report(data: Dict[str, Any]) -> str:
    """Generate JSON test report."""
    return json.dumps(data, indent=2)


# ============================================================================
# Main Report Generator
# ============================================================================

def main():
    """Generate test report."""
    parser = argparse.ArgumentParser(description="Generate GL-CBAM test report")
    parser.add_argument("--format", choices=["html", "markdown", "json"], default="html",
                       help="Output format (default: html)")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("--junit-xml", help="Path to JUnit XML file")
    parser.add_argument("--coverage-json", default="coverage.json", help="Path to coverage.json")

    args = parser.parse_args()

    print("="*70)
    print("GL-CBAM Test Report Generator")
    print("="*70)

    # Find latest JUnit XML if not specified
    if not args.junit_xml:
        results_dir = PROJECT_ROOT / "test-results"
        if results_dir.exists():
            xml_files = list(results_dir.glob("test-results-*.xml"))
            if xml_files:
                args.junit_xml = str(sorted(xml_files)[-1])  # Latest file
                print(f"Using JUnit XML: {args.junit_xml}")

    # Parse data
    test_results = {}
    if args.junit_xml and Path(args.junit_xml).exists():
        print(f"Parsing test results: {args.junit_xml}")
        test_results = parse_junit_xml(Path(args.junit_xml))
    else:
        print("Warning: No JUnit XML file found")

    coverage = {}
    coverage_path = PROJECT_ROOT / args.coverage_json
    if coverage_path.exists():
        print(f"Parsing coverage data: {coverage_path}")
        coverage = parse_coverage_json(coverage_path)
    else:
        print("Warning: No coverage.json file found")

    # Compile data
    data = {
        'metadata': {
            'generated_at': DeterministicClock.now().isoformat(),
            'format': args.format
        },
        'test_results': test_results,
        'coverage': coverage
    }

    # Generate report
    print(f"Generating {args.format.upper()} report...")

    if args.format == "html":
        report_content = generate_html_report(data)
        default_output = "test-report.html"
    elif args.format == "markdown":
        report_content = generate_markdown_report(data)
        default_output = "test-report.md"
    else:  # json
        report_content = generate_json_report(data)
        default_output = "test-report.json"

    # Write output
    output_path = PROJECT_ROOT / (args.output or default_output)
    with open(output_path, 'w') as f:
        f.write(report_content)

    print(f"✓ Report generated: {output_path}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
