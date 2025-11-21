#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Coverage Audit Script for GreenLang Phase 5

This script uses coverage.py to analyze current test coverage and generate
detailed reports with actionable items to achieve 95%+ coverage.

Features:
- Analyzes current coverage by module
- Identifies uncovered functions, branches, and edge cases
- Prioritizes gaps by criticality
- Outputs JSON report with actionable items
- Generates HTML and console reports

Usage:
    python scripts/audit_test_coverage.py [--output coverage_audit.json]
"""

import argparse
import ast
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import subprocess
import re
from greenlang.determinism import DeterministicClock

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CoverageGap:
    """Represents a coverage gap with priority and details."""
    file_path: str
    module: str
    function_name: str
    line_number: int
    gap_type: str  # 'uncovered_function', 'uncovered_branch', 'uncovered_edge_case'
    priority: int  # 1 (critical) to 5 (low)
    description: str
    complexity_score: int
    suggested_tests: List[str]


@dataclass
class ModuleCoverage:
    """Coverage statistics for a module."""
    module_name: str
    total_statements: int
    covered_statements: int
    missing_statements: int
    total_branches: int
    covered_branches: int
    missing_branches: int
    coverage_percent: float
    branch_coverage_percent: float
    uncovered_functions: List[str]
    priority: int


@dataclass
class CoverageAuditReport:
    """Complete coverage audit report."""
    timestamp: str
    overall_coverage: float
    overall_branch_coverage: float
    total_statements: int
    covered_statements: int
    total_branches: int
    covered_branches: int
    module_coverage: List[ModuleCoverage]
    coverage_gaps: List[CoverageGap]
    priority_gaps: Dict[int, int]  # priority level -> count
    recommended_actions: List[str]
    summary: str


class CoverageAnalyzer:
    """Analyzes test coverage and identifies gaps."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.source_dirs = [
            project_root / "greenlang",
            project_root / "core" / "greenlang"
        ]
        self.coverage_data = None
        self.ast_cache: Dict[str, ast.Module] = {}

    def run_coverage(self) -> bool:
        """Run coverage analysis on the test suite."""
        print("Running coverage analysis...")
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=greenlang",
                "--cov=core.greenlang",
                "--cov-report=json:coverage.json",
                "--cov-report=html:.coverage_html",
                "--cov-report=term",
                "--cov-branch",
                "-v",
                str(self.project_root / "tests")
            ]
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Warning: Tests failed but continuing with coverage analysis")
                print(f"STDOUT: {result.stdout[:500]}")
                print(f"STDERR: {result.stderr[:500]}")

            return True
        except Exception as e:
            print(f"Error running coverage: {e}")
            return False

    def load_coverage_data(self) -> bool:
        """Load coverage data from JSON report."""
        coverage_file = self.project_root / "coverage.json"
        if not coverage_file.exists():
            print(f"Coverage data not found at {coverage_file}")
            return False

        try:
            with open(coverage_file, 'r') as f:
                self.coverage_data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading coverage data: {e}")
            return False

    def parse_python_file(self, file_path: Path) -> Optional[ast.Module]:
        """Parse Python file and return AST."""
        if str(file_path) in self.ast_cache:
            return self.ast_cache[str(file_path)]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content, filename=str(file_path))
            self.ast_cache[str(file_path)] = tree
            return tree
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def extract_functions(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract all functions from AST."""
        functions = []

        class FunctionVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                complexity = self.calculate_complexity(node)
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'is_async': False,
                    'complexity': complexity,
                    'decorators': [self.get_decorator_name(d) for d in node.decorator_list],
                    'args': len(node.args.args),
                    'returns': node.returns is not None
                })
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                complexity = self.calculate_complexity(node)
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'is_async': True,
                    'complexity': complexity,
                    'decorators': [self.get_decorator_name(d) for d in node.decorator_list],
                    'args': len(node.args.args),
                    'returns': node.returns is not None
                })
                self.generic_visit(node)

            @staticmethod
            def calculate_complexity(node):
                """Calculate cyclomatic complexity."""
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                return complexity

            @staticmethod
            def get_decorator_name(decorator):
                if isinstance(decorator, ast.Name):
                    return decorator.id
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                    return decorator.func.id
                return str(decorator)

        visitor = FunctionVisitor()
        visitor.visit(tree)
        return functions

    def calculate_priority(self, file_path: str, function_name: str, complexity: int) -> int:
        """Calculate priority for a coverage gap (1=critical, 5=low)."""
        # Core modules are high priority
        if 'core/workflow' in file_path or 'orchestrator' in file_path:
            return 1
        if 'agents' in file_path and 'demo' not in file_path:
            return 2
        if 'policy' in file_path or 'security' in file_path:
            return 1
        if 'config' in file_path or 'auth' in file_path:
            return 2

        # High complexity functions are higher priority
        if complexity > 10:
            return 2
        if complexity > 5:
            return 3

        # Default priority based on location
        if 'cli' in file_path:
            return 4
        if 'utils' in file_path or 'helpers' in file_path:
            return 5

        return 3

    def suggest_tests(self, function_name: str, complexity: int, file_path: str) -> List[str]:
        """Suggest test cases for a function."""
        suggestions = []

        # Basic test
        suggestions.append(f"Test {function_name} with valid inputs")

        # Error handling
        if complexity > 3:
            suggestions.append(f"Test {function_name} with invalid/edge case inputs")
            suggestions.append(f"Test {function_name} error handling and exceptions")

        # Async functions
        if 'async' in file_path.lower() or function_name.startswith('async_'):
            suggestions.append(f"Test {function_name} concurrency and async behavior")

        # Agent functions
        if 'agent' in file_path.lower():
            suggestions.append(f"Test {function_name} with different agent configurations")
            suggestions.append(f"Test {function_name} output validation")

        # Workflow functions
        if 'workflow' in file_path.lower() or 'orchestrat' in file_path.lower():
            suggestions.append(f"Test {function_name} workflow state transitions")
            suggestions.append(f"Test {function_name} rollback scenarios")

        return suggestions

    def analyze_module_coverage(self) -> List[ModuleCoverage]:
        """Analyze coverage by module."""
        if not self.coverage_data:
            return []

        modules = defaultdict(lambda: {
            'total_statements': 0,
            'covered_statements': 0,
            'total_branches': 0,
            'covered_branches': 0,
            'files': []
        })

        files_data = self.coverage_data.get('files', {})

        for file_path, file_coverage in files_data.items():
            # Normalize path
            if 'greenlang' not in file_path:
                continue

            # Extract module name
            path_obj = Path(file_path)
            parts = path_obj.parts
            if 'greenlang' in parts:
                idx = parts.index('greenlang')
                if idx + 1 < len(parts):
                    module_name = parts[idx + 1]
                else:
                    module_name = 'greenlang'
            else:
                module_name = 'other'

            summary = file_coverage.get('summary', {})

            modules[module_name]['total_statements'] += summary.get('num_statements', 0)
            modules[module_name]['covered_statements'] += summary.get('covered_lines', 0)
            modules[module_name]['total_branches'] += summary.get('num_branches', 0)
            modules[module_name]['covered_branches'] += summary.get('covered_branches', 0)
            modules[module_name]['files'].append(file_path)

        result = []
        for module_name, data in modules.items():
            total_stmts = data['total_statements']
            covered_stmts = data['covered_statements']
            total_branches = data['total_branches']
            covered_branches = data['covered_branches']

            coverage_pct = (covered_stmts / total_stmts * 100) if total_stmts > 0 else 100
            branch_pct = (covered_branches / total_branches * 100) if total_branches > 0 else 100

            # Determine priority based on coverage
            if coverage_pct < 80:
                priority = 1
            elif coverage_pct < 90:
                priority = 2
            else:
                priority = 3

            result.append(ModuleCoverage(
                module_name=module_name,
                total_statements=total_stmts,
                covered_statements=covered_stmts,
                missing_statements=total_stmts - covered_stmts,
                total_branches=total_branches,
                covered_branches=covered_branches,
                missing_branches=total_branches - covered_branches,
                coverage_percent=coverage_pct,
                branch_coverage_percent=branch_pct,
                uncovered_functions=[],
                priority=priority
            ))

        return sorted(result, key=lambda x: x.coverage_percent)

    def identify_coverage_gaps(self) -> List[CoverageGap]:
        """Identify specific coverage gaps."""
        if not self.coverage_data:
            return []

        gaps = []
        files_data = self.coverage_data.get('files', {})

        for file_path, file_coverage in files_data.items():
            # Skip test files
            if 'test' in file_path.lower() or '__pycache__' in file_path:
                continue

            if 'greenlang' not in file_path:
                continue

            # Get missing lines and branches
            missing_lines = set(file_coverage.get('missing_lines', []))
            excluded_lines = set(file_coverage.get('excluded_lines', []))

            # Parse the file to find functions
            path_obj = Path(file_path)
            if not path_obj.exists():
                continue

            tree = self.parse_python_file(path_obj)
            if not tree:
                continue

            functions = self.extract_functions(tree)

            for func in functions:
                # Check if function is covered
                func_lines = set(range(func['line'], func['line'] + 10))  # Approximate
                if func_lines & missing_lines:
                    # Function has uncovered lines
                    priority = self.calculate_priority(file_path, func['name'], func['complexity'])
                    suggestions = self.suggest_tests(func['name'], func['complexity'], file_path)

                    module = self.extract_module_name(file_path)

                    gap = CoverageGap(
                        file_path=file_path,
                        module=module,
                        function_name=func['name'],
                        line_number=func['line'],
                        gap_type='uncovered_function',
                        priority=priority,
                        description=f"Function '{func['name']}' has uncovered lines (complexity: {func['complexity']})",
                        complexity_score=func['complexity'],
                        suggested_tests=suggestions
                    )
                    gaps.append(gap)

        return sorted(gaps, key=lambda x: (x.priority, -x.complexity_score))

    def extract_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        path_obj = Path(file_path)
        parts = path_obj.parts
        if 'greenlang' in parts:
            idx = parts.index('greenlang')
            if idx + 1 < len(parts):
                return '.'.join(parts[idx:idx+2])
        return 'greenlang'

    def generate_recommendations(self, gaps: List[CoverageGap],
                                module_coverage: List[ModuleCoverage]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Overall coverage recommendation
        low_coverage_modules = [m for m in module_coverage if m.coverage_percent < 90]
        if low_coverage_modules:
            recommendations.append(
                f"Focus on {len(low_coverage_modules)} modules with <90% coverage: "
                f"{', '.join([m.module_name for m in low_coverage_modules[:5]])}"
            )

        # Priority gaps
        p1_gaps = [g for g in gaps if g.priority == 1]
        if p1_gaps:
            recommendations.append(
                f"Address {len(p1_gaps)} critical (P1) coverage gaps in core modules"
            )

        # High complexity functions
        complex_gaps = [g for g in gaps if g.complexity_score > 10]
        if complex_gaps:
            recommendations.append(
                f"Add tests for {len(complex_gaps)} high-complexity functions (complexity > 10)"
            )

        # Branch coverage
        low_branch_modules = [m for m in module_coverage if m.branch_coverage_percent < 85]
        if low_branch_modules:
            recommendations.append(
                f"Improve branch coverage in {len(low_branch_modules)} modules"
            )

        # Specific recommendations
        recommendations.extend([
            "Add integration tests for agent combinations",
            "Create E2E tests for critical user journeys",
            "Implement chaos engineering tests for resilience",
            "Add property-based tests for core algorithms",
            "Create performance regression tests"
        ])

        return recommendations

    def generate_report(self) -> CoverageAuditReport:
        """Generate complete coverage audit report."""
        if not self.coverage_data:
            raise ValueError("Coverage data not loaded")

        # Calculate overall statistics
        totals = self.coverage_data.get('totals', {})
        overall_coverage = totals.get('percent_covered', 0)
        total_statements = totals.get('num_statements', 0)
        covered_statements = totals.get('covered_lines', 0)
        total_branches = totals.get('num_branches', 0)
        covered_branches = totals.get('covered_branches', 0)

        branch_coverage = (covered_branches / total_branches * 100) if total_branches > 0 else 0

        # Analyze modules and gaps
        module_coverage = self.analyze_module_coverage()
        coverage_gaps = self.identify_coverage_gaps()

        # Priority distribution
        priority_dist = defaultdict(int)
        for gap in coverage_gaps:
            priority_dist[gap.priority] += 1

        # Generate recommendations
        recommendations = self.generate_recommendations(coverage_gaps, module_coverage)

        # Summary
        summary = f"""
Coverage Audit Summary:
- Overall Coverage: {overall_coverage:.2f}%
- Branch Coverage: {branch_coverage:.2f}%
- Total Statements: {total_statements}
- Covered Statements: {covered_statements}
- Missing Statements: {total_statements - covered_statements}
- Total Gaps: {len(coverage_gaps)}
- P1 Gaps: {priority_dist[1]}
- P2 Gaps: {priority_dist[2]}
- P3 Gaps: {priority_dist[3]}
"""

        return CoverageAuditReport(
            timestamp=DeterministicClock.now().isoformat(),
            overall_coverage=overall_coverage,
            overall_branch_coverage=branch_coverage,
            total_statements=total_statements,
            covered_statements=covered_statements,
            total_branches=total_branches,
            covered_branches=covered_branches,
            module_coverage=module_coverage,
            coverage_gaps=coverage_gaps[:100],  # Limit to top 100
            priority_gaps=dict(priority_dist),
            recommended_actions=recommendations,
            summary=summary
        )

    def export_report(self, report: CoverageAuditReport, output_path: Path):
        """Export report to JSON file."""
        # Convert dataclasses to dicts
        report_dict = {
            'timestamp': report.timestamp,
            'overall_coverage': report.overall_coverage,
            'overall_branch_coverage': report.overall_branch_coverage,
            'total_statements': report.total_statements,
            'covered_statements': report.covered_statements,
            'total_branches': report.total_branches,
            'covered_branches': report.covered_branches,
            'module_coverage': [asdict(m) for m in report.module_coverage],
            'coverage_gaps': [asdict(g) for g in report.coverage_gaps],
            'priority_gaps': report.priority_gaps,
            'recommended_actions': report.recommended_actions,
            'summary': report.summary
        }

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        print(f"\nReport exported to: {output_path}")

    def print_summary(self, report: CoverageAuditReport):
        """Print summary to console."""
        print("\n" + "="*80)
        print("COVERAGE AUDIT REPORT")
        print("="*80)
        print(report.summary)

        print("\nTop 10 Modules Needing Coverage:")
        print("-" * 80)
        for module in report.module_coverage[:10]:
            print(f"  {module.module_name:30s} {module.coverage_percent:6.2f}% "
                  f"(Branch: {module.branch_coverage_percent:6.2f}%)")

        print("\nTop 10 Priority Gaps:")
        print("-" * 80)
        for gap in report.coverage_gaps[:10]:
            print(f"  [P{gap.priority}] {gap.module}::{gap.function_name} "
                  f"(complexity: {gap.complexity_score})")
            print(f"      {gap.description}")

        print("\nRecommended Actions:")
        print("-" * 80)
        for i, action in enumerate(report.recommended_actions, 1):
            print(f"  {i}. {action}")

        print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Audit test coverage for GreenLang')
    parser.add_argument(
        '--output',
        default='coverage_audit.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--skip-run',
        action='store_true',
        help='Skip running tests, use existing coverage data'
    )

    args = parser.parse_args()

    analyzer = CoverageAnalyzer(PROJECT_ROOT)

    # Run coverage if not skipped
    if not args.skip_run:
        if not analyzer.run_coverage():
            print("Failed to run coverage analysis")
            sys.exit(1)

    # Load coverage data
    if not analyzer.load_coverage_data():
        print("Failed to load coverage data")
        sys.exit(1)

    # Generate report
    try:
        report = analyzer.generate_report()

        # Print summary
        analyzer.print_summary(report)

        # Export to JSON
        output_path = PROJECT_ROOT / args.output
        analyzer.export_report(report, output_path)

        print(f"\n✓ Coverage audit complete!")
        print(f"  Overall Coverage: {report.overall_coverage:.2f}%")
        print(f"  Branch Coverage: {report.overall_branch_coverage:.2f}%")
        print(f"  Total Gaps: {len(report.coverage_gaps)}")

        # Exit code based on coverage
        if report.overall_coverage < 95:
            print(f"\n⚠ Coverage below 95% target")
            sys.exit(1)
        else:
            print(f"\n✓ Coverage meets 95% target!")
            sys.exit(0)

    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
