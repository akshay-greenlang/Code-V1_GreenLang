#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency Graph Visualizer

Visualize infrastructure dependencies across apps.
Shows which apps use which infrastructure components.
"""

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from greenlang.determinism import sorted_listdir


class DependencyAnalyzer:
    """Analyze infrastructure dependencies."""

    def __init__(self):
        self.app_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.component_usage: Dict[str, Set[str]] = defaultdict(set)
        self.import_counts: Dict[str, int] = defaultdict(int)

    def analyze_file(self, file_path: str, app_name: str):
        """Analyze a single file for dependencies."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find infrastructure imports
            infrastructure_imports = [
                'ChatSession',
                'BaseAgent',
                'CacheManager',
                'ValidationFramework',
                'Logger',
                'ConfigManager',
                'DatabaseConnector',
                'APIClient',
                'TaskQueue',
                'MetricsCollector',
                'PromptTemplate',
                'ResponseParser',
                'ErrorHandler',
                'DataLoader',
                'AgentPipeline'
            ]

            for component in infrastructure_imports:
                if component in content:
                    self.app_dependencies[app_name].add(component)
                    self.component_usage[component].add(app_name)
                    self.import_counts[component] += 1

        except Exception as e:
            pass

    def analyze_directory(self, directory: str, app_name: str = None):
        """Analyze entire directory."""
        if app_name is None:
            app_name = os.path.basename(directory)

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', 'node_modules']]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.analyze_file(file_path, app_name)

    def get_graph_data(self) -> Dict[str, any]:
        """Get graph data for visualization."""
        nodes = []
        edges = []

        # Add app nodes
        for app in self.app_dependencies.keys():
            nodes.append({
                "id": app,
                "label": app,
                "type": "app",
                "size": len(self.app_dependencies[app]) * 10
            })

        # Add component nodes
        for component in self.component_usage.keys():
            nodes.append({
                "id": component,
                "label": component,
                "type": "component",
                "size": len(self.component_usage[component]) * 15
            })

        # Add edges
        for app, components in self.app_dependencies.items():
            for component in components:
                edges.append({
                    "source": app,
                    "target": component,
                    "weight": 1
                })

        return {
            "nodes": nodes,
            "edges": edges
        }


class GraphVisualizer:
    """Visualize dependency graph."""

    @staticmethod
    def generate_dot(analyzer: DependencyAnalyzer) -> str:
        """Generate Graphviz DOT format."""
        dot = ["digraph dependencies {"]
        dot.append("  rankdir=LR;")
        dot.append("  node [shape=box];")
        dot.append("")

        # App nodes
        dot.append("  // Applications")
        for app in analyzer.app_dependencies.keys():
            dot.append(f'  "{app}" [style=filled, fillcolor=lightblue];')

        # Component nodes
        dot.append("\n  // Infrastructure Components")
        for component in analyzer.component_usage.keys():
            count = len(analyzer.component_usage[component])
            color = "lightgreen" if count > 3 else "lightyellow"
            dot.append(f'  "{component}" [style=filled, fillcolor={color}];')

        # Edges
        dot.append("\n  // Dependencies")
        for app, components in analyzer.app_dependencies.items():
            for component in components:
                dot.append(f'  "{app}" -> "{component}";')

        dot.append("}")
        return "\n".join(dot)

    @staticmethod
    def generate_html(analyzer: DependencyAnalyzer) -> str:
        """Generate interactive HTML with D3.js."""
        graph_data = analyzer.get_graph_data()

        html = """<!DOCTYPE html>
<html>
<head>
    <title>Infrastructure Dependency Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #f5f5f5; }
        #graph { background: white; border: 1px solid #ddd; }
        .node { cursor: pointer; }
        .node.app { fill: #3498db; }
        .node.component { fill: #2ecc71; }
        .node text { fill: white; font-size: 12px; font-weight: bold; pointer-events: none; }
        .link { stroke: #999; stroke-width: 2px; opacity: 0.6; }
        .link:hover { stroke: #333; opacity: 1; }
        h1 { color: #2c3e50; }
        .stats { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Infrastructure Dependency Graph</h1>

    <div class="stats">
        <strong>Applications:</strong> """ + str(len(analyzer.app_dependencies)) + """ |
        <strong>Components:</strong> """ + str(len(analyzer.component_usage)) + """
    </div>

    <svg id="graph" width="1200" height="800"></svg>

    <script>
        const data = """ + json.dumps(graph_data) + """;

        const width = 1200;
        const height = 800;

        const svg = d3.select("#graph");

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.edges).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.append("g")
            .selectAll("line")
            .data(data.edges)
            .enter().append("line")
            .attr("class", "link");

        const node = svg.append("g")
            .selectAll("g")
            .data(data.nodes)
            .enter().append("g")
            .attr("class", d => `node ${d.type}`)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", d => d.size || 20)
            .attr("class", d => d.type);

        node.append("text")
            .attr("dx", 0)
            .attr("dy", 5)
            .attr("text-anchor", "middle")
            .text(d => d.label);

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    </script>
</body>
</html>
"""
        return html

    @staticmethod
    def generate_text(analyzer: DependencyAnalyzer) -> str:
        """Generate text report."""
        output = []
        output.append("=" * 80)
        output.append("INFRASTRUCTURE DEPENDENCY GRAPH")
        output.append("=" * 80)

        output.append("\nAPPLICATION DEPENDENCIES:")
        output.append("-" * 80)

        for app in sorted(analyzer.app_dependencies.keys()):
            components = analyzer.app_dependencies[app]
            output.append(f"\n{app} ({len(components)} components)")
            for component in sorted(components):
                output.append(f"  → {component}")

        output.append("\n" + "=" * 80)
        output.append("\nCOMPONENT USAGE:")
        output.append("-" * 80)

        # Sort by usage count
        sorted_components = sorted(
            analyzer.component_usage.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for component, apps in sorted_components:
            output.append(f"\n{component} (used by {len(apps)} apps)")
            for app in sorted(apps):
                output.append(f"  ← {app}")

        output.append("\n" + "=" * 80)
        return "\n".join(output)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Visualize infrastructure dependencies')
    parser.add_argument('--directory', default='.', help='Root directory to scan')
    parser.add_argument('--apps', nargs='+', help='Specific apps to analyze')
    parser.add_argument('--format', choices=['text', 'dot', 'html', 'json'], default='text')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--interactive', action='store_true', help='Generate interactive HTML')

    args = parser.parse_args()

    # Analyze dependencies
    analyzer = DependencyAnalyzer()

    if args.apps:
        for app in args.apps:
            if os.path.exists(app):
                print(f"Analyzing {app}...")
                analyzer.analyze_directory(app)
    else:
        # Find all GL-* apps
        for item in sorted_listdir(args.directory):
            if item.startswith('GL-') and os.path.isdir(item):
                print(f"Analyzing {item}...")
                analyzer.analyze_directory(item)

    # Generate output
    visualizer = GraphVisualizer()

    if args.format == 'dot':
        output = visualizer.generate_dot(analyzer)
    elif args.format == 'html' or args.interactive:
        output = visualizer.generate_html(analyzer)
    elif args.format == 'json':
        output = json.dumps(analyzer.get_graph_data(), indent=2)
    else:
        output = visualizer.generate_text(analyzer)

    # Save or print
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\nOutput saved to: {args.output}")

        if args.format == 'html' or args.interactive:
            print(f"Open in browser: file://{os.path.abspath(args.output)}")
    else:
        print(output)


if __name__ == '__main__':
    main()
