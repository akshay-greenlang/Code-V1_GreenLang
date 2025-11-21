#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Infrastructure Usage Report Generator

Scans applications and generates comprehensive reports on GreenLang infrastructure usage.
Calculates metrics, generates visualizations, and produces HTML dashboards.
"""

import argparse
import ast
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from datetime import datetime
from greenlang.determinism import DeterministicClock


class UsageAnalyzer:
    """Analyzes GreenLang infrastructure usage."""

    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(int))
        self.file_stats = {}
        self.greenlang_imports = set()
        self.non_greenlang_imports = set()

    def analyze_file(self, file_path: str) -> Dict:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Count lines of code (excluding comments and blank lines)
            loc = 0
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    loc += 1

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {'error': 'syntax_error', 'loc': loc}

            # Extract imports
            greenlang_imports = []
            other_imports = []
            greenlang_usage = defaultdict(int)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith('greenlang'):
                            greenlang_imports.append(alias.name)
                            self.greenlang_imports.add(alias.name)
                            greenlang_usage[alias.name] += 1
                        else:
                            other_imports.append(alias.name)
                            self.non_greenlang_imports.add(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('greenlang'):
                        for alias in node.names:
                            import_path = f"{node.module}.{alias.name}"
                            greenlang_imports.append(import_path)
                            self.greenlang_imports.add(import_path)
                            greenlang_usage[node.module] += 1
                    elif node.module:
                        other_imports.append(node.module)
                        self.non_greenlang_imports.add(node.module)

            # Count agent classes
            agent_classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if inherits from Agent
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == 'Agent':
                            agent_classes.append(node.name)
                        elif isinstance(base, ast.Attribute):
                            if base.attr == 'Agent':
                                agent_classes.append(node.name)

            # Calculate IUM (Infrastructure Usage Metric)
            total_imports = len(greenlang_imports) + len(other_imports)
            ium = (len(greenlang_imports) / total_imports * 100) if total_imports > 0 else 0

            return {
                'file_path': file_path,
                'loc': loc,
                'greenlang_imports': greenlang_imports,
                'other_imports': other_imports,
                'greenlang_usage': dict(greenlang_usage),
                'agent_classes': agent_classes,
                'ium': ium,
                'total_imports': total_imports
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_directory(self, directory: str, exclude_patterns: List[str] = None) -> Dict:
        """Analyze all Python files in a directory."""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'venv', 'env', 'node_modules', '.greenlang', 'test']

        results = {
            'directory': directory,
            'files': [],
            'summary': {
                'total_files': 0,
                'total_loc': 0,
                'files_using_greenlang': 0,
                'total_greenlang_imports': 0,
                'total_other_imports': 0,
                'total_agent_classes': 0,
                'average_ium': 0,
            },
            'by_component': defaultdict(int),
            'file_details': []
        }

        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_stats = self.analyze_file(file_path)

                    if 'error' not in file_stats:
                        results['files'].append(file_path)
                        results['file_details'].append(file_stats)
                        results['summary']['total_files'] += 1
                        results['summary']['total_loc'] += file_stats['loc']

                        if file_stats['greenlang_imports']:
                            results['summary']['files_using_greenlang'] += 1

                        results['summary']['total_greenlang_imports'] += len(file_stats['greenlang_imports'])
                        results['summary']['total_other_imports'] += len(file_stats['other_imports'])
                        results['summary']['total_agent_classes'] += len(file_stats['agent_classes'])

                        # Count by component
                        for imp in file_stats['greenlang_imports']:
                            component = imp.split('.')[1] if '.' in imp else imp
                            results['by_component'][component] += 1

        # Calculate average IUM
        if results['file_details']:
            total_ium = sum(f['ium'] for f in results['file_details'])
            results['summary']['average_ium'] = total_ium / len(results['file_details'])

        # Calculate overall IUM
        total_imports = results['summary']['total_greenlang_imports'] + results['summary']['total_other_imports']
        results['summary']['overall_ium'] = (
            results['summary']['total_greenlang_imports'] / total_imports * 100
        ) if total_imports > 0 else 0

        # Calculate adoption rate
        results['summary']['adoption_rate'] = (
            results['summary']['files_using_greenlang'] / results['summary']['total_files'] * 100
        ) if results['summary']['total_files'] > 0 else 0

        return results


class ReportGenerator:
    """Generates various report formats."""

    def generate_text_report(self, analysis: Dict) -> str:
        """Generate text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("GreenLang Infrastructure Usage Report")
        lines.append("=" * 80)
        lines.append(f"Generated: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Directory: {analysis['directory']}")
        lines.append("")

        # Summary
        summary = analysis['summary']
        lines.append("SUMMARY:")
        lines.append(f"  Total files analyzed: {summary['total_files']}")
        lines.append(f"  Total lines of code: {summary['total_loc']:,}")
        lines.append(f"  Files using GreenLang: {summary['files_using_greenlang']} ({summary['adoption_rate']:.1f}%)")
        lines.append(f"  GreenLang imports: {summary['total_greenlang_imports']}")
        lines.append(f"  Other imports: {summary['total_other_imports']}")
        lines.append(f"  Agent classes: {summary['total_agent_classes']}")
        lines.append(f"  Overall IUM: {summary['overall_ium']:.1f}%")
        lines.append(f"  Average IUM per file: {summary['average_ium']:.1f}%")
        lines.append("")

        # By component
        if analysis['by_component']:
            lines.append("USAGE BY COMPONENT:")
            for component, count in sorted(analysis['by_component'].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {component}: {count}")
            lines.append("")

        # Top files
        top_files = sorted(
            analysis['file_details'],
            key=lambda x: x['ium'],
            reverse=True
        )[:10]

        if top_files:
            lines.append("TOP 10 FILES BY IUM:")
            for i, file_info in enumerate(top_files, 1):
                rel_path = os.path.relpath(file_info['file_path'], analysis['directory'])
                lines.append(f"  {i}. {rel_path}")
                lines.append(f"     IUM: {file_info['ium']:.1f}% | LOC: {file_info['loc']} | " +
                           f"GL Imports: {len(file_info['greenlang_imports'])}")
            lines.append("")

        return '\n'.join(lines)

    def generate_json_report(self, analysis: Dict) -> str:
        """Generate JSON report."""
        return json.dumps(analysis, indent=2)

    def generate_html_report(self, analysis: Dict) -> str:
        """Generate HTML dashboard."""
        summary = analysis['summary']

        # Prepare data for charts
        component_data = [
            {'name': comp, 'count': count}
            for comp, count in sorted(analysis['by_component'].items(), key=lambda x: x[1], reverse=True)
        ]

        # Top files
        top_files = sorted(
            analysis['file_details'],
            key=lambda x: x['ium'],
            reverse=True
        )[:10]

        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>GreenLang Infrastructure Usage Report</title>
    <meta charset="utf-8">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #2c5f2d 0%, #4caf50 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}

        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c5f2d;
            margin: 10px 0;
        }}

        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .section {{
            padding: 30px;
        }}

        .section h2 {{
            color: #2c5f2d;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #4caf50;
        }}

        .chart-container {{
            margin: 20px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .bar {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}

        .bar-label {{
            min-width: 150px;
            font-weight: 500;
        }}

        .bar-visual {{
            flex: 1;
            background: #e0e0e0;
            height: 30px;
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        }}

        .bar-fill {{
            background: linear-gradient(90deg, #4caf50 0%, #2c5f2d 100%);
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
            transition: width 1s ease;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th {{
            background: #2c5f2d;
            color: white;
            padding: 15px;
            text-align: left;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}

        tr:hover {{
            background: #f5f5f5;
        }}

        .progress-ring {{
            display: inline-block;
            position: relative;
            width: 150px;
            height: 150px;
        }}

        .progress-ring-circle {{
            transition: stroke-dashoffset 1s;
            transform: rotate(-90deg);
            transform-origin: 50% 50%;
        }}

        .progress-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2em;
            font-weight: bold;
            color: #2c5f2d;
        }}

        .gauges {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin: 30px 0;
        }}

        .gauge {{
            text-align: center;
            margin: 20px;
        }}

        .gauge-label {{
            margin-top: 10px;
            font-weight: 500;
            color: #666;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GreenLang Infrastructure Usage Report</h1>
            <p>Generated: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Directory: {analysis['directory']}</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Total Files</div>
                <div class="metric-value">{summary['total_files']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Lines of Code</div>
                <div class="metric-value">{summary['total_loc']:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Overall IUM</div>
                <div class="metric-value">{summary['overall_ium']:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Adoption Rate</div>
                <div class="metric-value">{summary['adoption_rate']:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GL Imports</div>
                <div class="metric-value">{summary['total_greenlang_imports']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Agent Classes</div>
                <div class="metric-value">{summary['total_agent_classes']}</div>
            </div>
        </div>

        <div class="section">
            <h2>Infrastructure Usage Metrics</h2>
            <div class="gauges">
                <div class="gauge">
                    <svg class="progress-ring" width="150" height="150">
                        <circle class="progress-ring-circle" stroke="#e0e0e0" stroke-width="15" fill="transparent" r="60" cx="75" cy="75"/>
                        <circle class="progress-ring-circle" stroke="#4caf50" stroke-width="15" fill="transparent" r="60" cx="75" cy="75"
                                stroke-dasharray="{376.99}" stroke-dashoffset="{376.99 * (1 - summary['overall_ium'] / 100)}"/>
                    </svg>
                    <div class="progress-text">{summary['overall_ium']:.0f}%</div>
                    <div class="gauge-label">Overall IUM</div>
                </div>

                <div class="gauge">
                    <svg class="progress-ring" width="150" height="150">
                        <circle class="progress-ring-circle" stroke="#e0e0e0" stroke-width="15" fill="transparent" r="60" cx="75" cy="75"/>
                        <circle class="progress-ring-circle" stroke="#4caf50" stroke-width="15" fill="transparent" r="60" cx="75" cy="75"
                                stroke-dasharray="{376.99}" stroke-dashoffset="{376.99 * (1 - summary['adoption_rate'] / 100)}"/>
                    </svg>
                    <div class="progress-text">{summary['adoption_rate']:.0f}%</div>
                    <div class="gauge-label">Adoption Rate</div>
                </div>

                <div class="gauge">
                    <svg class="progress-ring" width="150" height="150">
                        <circle class="progress-ring-circle" stroke="#e0e0e0" stroke-width="15" fill="transparent" r="60" cx="75" cy="75"/>
                        <circle class="progress-ring-circle" stroke="#4caf50" stroke-width="15" fill="transparent" r="60" cx="75" cy="75"
                                stroke-dasharray="{376.99}" stroke-dashoffset="{376.99 * (1 - summary['average_ium'] / 100)}"/>
                    </svg>
                    <div class="progress-text">{summary['average_ium']:.0f}%</div>
                    <div class="gauge-label">Average IUM</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Usage by Component</h2>
            <div class="chart-container">
'''

        # Add component bars
        if component_data:
            max_count = max(c['count'] for c in component_data)
            for comp in component_data:
                percentage = (comp['count'] / max_count * 100) if max_count > 0 else 0
                html += f'''                <div class="bar">
                    <div class="bar-label">{comp['name']}</div>
                    <div class="bar-visual">
                        <div class="bar-fill" style="width: {percentage}%">{comp['count']}</div>
                    </div>
                </div>
'''

        html += '''            </div>
        </div>

        <div class="section">
            <h2>Top 10 Files by IUM</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>File</th>
                        <th>IUM</th>
                        <th>LOC</th>
                        <th>GL Imports</th>
                        <th>Other Imports</th>
                    </tr>
                </thead>
                <tbody>
'''

        for i, file_info in enumerate(top_files, 1):
            rel_path = os.path.relpath(file_info['file_path'], analysis['directory'])
            html += f'''                    <tr>
                        <td>{i}</td>
                        <td>{rel_path}</td>
                        <td>{file_info['ium']:.1f}%</td>
                        <td>{file_info['loc']}</td>
                        <td>{len(file_info['greenlang_imports'])}</td>
                        <td>{len(file_info['other_imports'])}</td>
                    </tr>
'''

        html += '''                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by GreenLang Migration Toolkit</p>
            <p>For more information, visit the GreenLang documentation</p>
        </div>
    </div>
</body>
</html>'''

        return html


def main():
    parser = argparse.ArgumentParser(
        description="GreenLang Usage Report Generator - Generate infrastructure usage reports"
    )

    parser.add_argument(
        'directory',
        help='Directory to analyze'
    )

    parser.add_argument(
        '--format',
        choices=['text', 'json', 'html'],
        default='html',
        help='Report format (default: html)'
    )

    parser.add_argument(
        '--output',
        help='Output file path'
    )

    args = parser.parse_args()

    # Analyze directory
    print(f"Analyzing {args.directory}...")
    analyzer = UsageAnalyzer()
    analysis = analyzer.analyze_directory(args.directory)

    print(f"Analyzed {analysis['summary']['total_files']} files")

    # Generate report
    print(f"Generating {args.format} report...")
    generator = ReportGenerator()

    if args.format == 'text':
        report = generator.generate_text_report(analysis)
    elif args.format == 'json':
        report = generator.generate_json_report(analysis)
    elif args.format == 'html':
        report = generator.generate_html_report(analysis)

    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Report saved to {args.output}")

        if args.format == 'html':
            print(f"Open in browser: file://{os.path.abspath(args.output)}")
    else:
        print("\n" + report)


if __name__ == '__main__':
    main()
