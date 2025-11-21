#!/usr/bin/env python3
# -*- coding: utf-8 -*-

logger = logging.getLogger(__name__)
"""
Infrastructure Usage Metrics (IUM) Calculator
==============================================

Calculates what percentage of code uses GreenLang infrastructure vs custom implementations.

Usage:
    python calculate_ium.py [--app APP_NAME] [--output FORMAT]

Metrics:
    - Infrastructure imports vs total imports
    - Agent inheritance compliance
    - LLM calls through ChatSession vs direct
    - Auth operations through greenlang.auth vs custom
    - Cache operations through CacheManager vs direct
"""

import logging
import ast
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class FileMetrics:
    """Metrics for a single file"""
    file_path: str
    total_imports: int
    greenlang_imports: int
    forbidden_imports: int
    agent_classes: int
    compliant_agents: int
    llm_calls: int
    greenlang_llm_calls: int
    auth_ops: int
    greenlang_auth_ops: int
    cache_ops: int
    greenlang_cache_ops: int
    db_ops: int
    greenlang_db_ops: int

    @property
    def import_score(self) -> float:
        """Calculate import compliance score"""
        if self.total_imports == 0:
            return 100.0
        return (self.greenlang_imports / self.total_imports) * 100

    @property
    def agent_score(self) -> float:
        """Calculate agent compliance score"""
        if self.agent_classes == 0:
            return 100.0
        return (self.compliant_agents / self.agent_classes) * 100

    @property
    def llm_score(self) -> float:
        """Calculate LLM compliance score"""
        if self.llm_calls == 0:
            return 100.0
        return (self.greenlang_llm_calls / self.llm_calls) * 100

    @property
    def auth_score(self) -> float:
        """Calculate auth compliance score"""
        if self.auth_ops == 0:
            return 100.0
        return (self.greenlang_auth_ops / self.auth_ops) * 100

    @property
    def cache_score(self) -> float:
        """Calculate cache compliance score"""
        if self.cache_ops == 0:
            return 100.0
        return (self.greenlang_cache_ops / self.cache_ops) * 100

    @property
    def db_score(self) -> float:
        """Calculate database compliance score"""
        if self.db_ops == 0:
            return 100.0
        return (self.greenlang_db_ops / self.db_ops) * 100

    @property
    def overall_score(self) -> float:
        """Calculate overall infrastructure usage score"""
        scores = []
        weights = []

        # Import score (weight: 2)
        scores.append(self.import_score)
        weights.append(2)

        # Agent score (weight: 3)
        if self.agent_classes > 0:
            scores.append(self.agent_score)
            weights.append(3)

        # LLM score (weight: 3)
        if self.llm_calls > 0:
            scores.append(self.llm_score)
            weights.append(3)

        # Auth score (weight: 2)
        if self.auth_ops > 0:
            scores.append(self.auth_score)
            weights.append(2)

        # Cache score (weight: 1)
        if self.cache_ops > 0:
            scores.append(self.cache_score)
            weights.append(1)

        # DB score (weight: 1)
        if self.db_ops > 0:
            scores.append(self.db_score)
            weights.append(1)

        if not scores:
            return 100.0

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight


class IUMAnalyzer(ast.NodeVisitor):
    """AST analyzer for infrastructure usage metrics"""

    FORBIDDEN_MODULES = {
        'openai', 'anthropic', 'redis', 'pymongo', 'motor',
        'jose', 'jwt', 'pyjwt', 'passlib', 'bcrypt'
    }

    LLM_PATTERNS = {'completion', 'chat', 'generate', 'llm', 'gpt', 'claude'}
    AUTH_PATTERNS = {'auth', 'login', 'token', 'password', 'hash', 'verify'}
    CACHE_PATTERNS = {'cache', 'get', 'set', 'redis'}
    DB_PATTERNS = {'find', 'insert', 'update', 'delete', 'query', 'execute'}

    def __init__(self):
        self.metrics = FileMetrics(
            file_path="",
            total_imports=0,
            greenlang_imports=0,
            forbidden_imports=0,
            agent_classes=0,
            compliant_agents=0,
            llm_calls=0,
            greenlang_llm_calls=0,
            auth_ops=0,
            greenlang_auth_ops=0,
            cache_ops=0,
            greenlang_cache_ops=0,
            db_ops=0,
            greenlang_db_ops=0
        )
        self.has_greenlang_intelligence = False
        self.has_greenlang_auth = False
        self.has_greenlang_cache = False
        self.has_greenlang_db = False

    def visit_Import(self, node: ast.Import):
        """Track imports"""
        for alias in node.names:
            self.metrics.total_imports += 1
            module = alias.name

            # Track GreenLang imports
            if module.startswith('greenlang'):
                self.metrics.greenlang_imports += 1

                if 'intelligence' in module:
                    self.has_greenlang_intelligence = True
                elif 'auth' in module:
                    self.has_greenlang_auth = True
                elif 'cache' in module:
                    self.has_greenlang_cache = True
                elif 'db' in module or 'database' in module:
                    self.has_greenlang_db = True

            # Track forbidden imports
            root = module.split('.')[0]
            if root in self.FORBIDDEN_MODULES:
                self.metrics.forbidden_imports += 1

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from-imports"""
        if node.module:
            self.metrics.total_imports += len(node.names)
            module = node.module

            # Track GreenLang imports
            if module.startswith('greenlang'):
                self.metrics.greenlang_imports += len(node.names)

                if 'intelligence' in module:
                    self.has_greenlang_intelligence = True
                elif 'auth' in module:
                    self.has_greenlang_auth = True
                elif 'cache' in module:
                    self.has_greenlang_cache = True
                elif 'db' in module or 'database' in module:
                    self.has_greenlang_db = True

            # Track forbidden imports
            root = module.split('.')[0]
            if root in self.FORBIDDEN_MODULES:
                self.metrics.forbidden_imports += len(node.names)

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Track agent classes"""
        if 'agent' in node.name.lower():
            self.metrics.agent_classes += 1

            # Check inheritance
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == 'Agent':
                    self.metrics.compliant_agents += 1
                    break
                elif isinstance(base, ast.Attribute):
                    if 'Agent' in self._get_attr_name(base):
                        self.metrics.compliant_agents += 1
                        break

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Track function calls for patterns"""
        call_str = ast.unparse(node).lower()

        # Check LLM calls
        if any(pattern in call_str for pattern in self.LLM_PATTERNS):
            self.metrics.llm_calls += 1
            if self.has_greenlang_intelligence:
                self.metrics.greenlang_llm_calls += 1

        # Check auth operations
        if any(pattern in call_str for pattern in self.AUTH_PATTERNS):
            self.metrics.auth_ops += 1
            if self.has_greenlang_auth:
                self.metrics.greenlang_auth_ops += 1

        # Check cache operations
        if any(pattern in call_str for pattern in self.CACHE_PATTERNS):
            self.metrics.cache_ops += 1
            if self.has_greenlang_cache:
                self.metrics.greenlang_cache_ops += 1

        # Check DB operations
        if any(pattern in call_str for pattern in self.DB_PATTERNS):
            self.metrics.db_ops += 1
            if self.has_greenlang_db:
                self.metrics.greenlang_db_ops += 1

        self.generic_visit(node)

    @staticmethod
    def _get_attr_name(node):
        """Get full attribute name"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{IUMAnalyzer._get_attr_name(node.value)}.{node.attr}"
        return ""


def analyze_file(file_path: Path) -> FileMetrics:
    """Analyze a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))
        analyzer = IUMAnalyzer()
        analyzer.visit(tree)

        analyzer.metrics.file_path = str(file_path)
        return analyzer.metrics

    except Exception:
        # Return empty metrics on error
        return FileMetrics(
            file_path=str(file_path),
            total_imports=0, greenlang_imports=0, forbidden_imports=0,
            agent_classes=0, compliant_agents=0,
            llm_calls=0, greenlang_llm_calls=0,
            auth_ops=0, greenlang_auth_ops=0,
            cache_ops=0, greenlang_cache_ops=0,
            db_ops=0, greenlang_db_ops=0
        )


def aggregate_metrics(metrics_list: List[FileMetrics]) -> Dict:
    """Aggregate metrics across files"""
    if not metrics_list:
        return {
            'total_files': 0,
            'percentage': 100.0
        }

    total = FileMetrics(
        file_path="AGGREGATE",
        total_imports=sum(m.total_imports for m in metrics_list),
        greenlang_imports=sum(m.greenlang_imports for m in metrics_list),
        forbidden_imports=sum(m.forbidden_imports for m in metrics_list),
        agent_classes=sum(m.agent_classes for m in metrics_list),
        compliant_agents=sum(m.compliant_agents for m in metrics_list),
        llm_calls=sum(m.llm_calls for m in metrics_list),
        greenlang_llm_calls=sum(m.greenlang_llm_calls for m in metrics_list),
        auth_ops=sum(m.auth_ops for m in metrics_list),
        greenlang_auth_ops=sum(m.greenlang_auth_ops for m in metrics_list),
        cache_ops=sum(m.cache_ops for m in metrics_list),
        greenlang_cache_ops=sum(m.greenlang_cache_ops for m in metrics_list),
        db_ops=sum(m.db_ops for m in metrics_list),
        greenlang_db_ops=sum(m.greenlang_db_ops for m in metrics_list)
    )

    return {
        'total_files': len(metrics_list),
        'percentage': total.overall_score,
        'details': {
            'imports': {
                'total': total.total_imports,
                'greenlang': total.greenlang_imports,
                'forbidden': total.forbidden_imports,
                'percentage': total.import_score
            },
            'agents': {
                'total': total.agent_classes,
                'compliant': total.compliant_agents,
                'percentage': total.agent_score
            },
            'llm': {
                'total': total.llm_calls,
                'greenlang': total.greenlang_llm_calls,
                'percentage': total.llm_score
            },
            'auth': {
                'total': total.auth_ops,
                'greenlang': total.greenlang_auth_ops,
                'percentage': total.auth_score
            },
            'cache': {
                'total': total.cache_ops,
                'greenlang': total.greenlang_cache_ops,
                'percentage': total.cache_score
            },
            'database': {
                'total': total.db_ops,
                'greenlang': total.greenlang_db_ops,
                'percentage': total.db_score
            }
        }
    }


def format_markdown(report: Dict) -> str:
    """Format report as markdown"""
    lines = []
    lines.append("# Infrastructure Usage Metrics Report")
    lines.append("")

    overall = report['overall']
    lines.append(f"## Overall Score: {overall['percentage']:.1f}%")
    lines.append("")
    lines.append(f"- Total Files Analyzed: {overall['total_files']}")
    lines.append("")

    lines.append("## Detailed Breakdown")
    lines.append("")

    for category, data in overall['details'].items():
        lines.append(f"### {category.title()}")
        lines.append(f"- **Score:** {data['percentage']:.1f}%")
        lines.append(f"- **Total Operations:** {data['total']}")

        if category == 'imports':
            lines.append(f"- **GreenLang Imports:** {data['greenlang']}")
            lines.append(f"- **Forbidden Imports:** {data['forbidden']}")
        else:
            lines.append(f"- **Using GreenLang:** {data['greenlang']}")

        lines.append("")

    # Per-app breakdown
    if 'by_app' in report:
        lines.append("## By Application")
        lines.append("")

        for app, metrics in report['by_app'].items():
            lines.append(f"### {app}")
            lines.append(f"- **Score:** {metrics['percentage']:.1f}%")
            lines.append(f"- **Files:** {metrics['total_files']}")
            lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Calculate Infrastructure Usage Metrics'
    )
    parser.add_argument(
        '--app',
        help='Specific app to analyze (e.g., GL-CBAM-APP)'
    )
    parser.add_argument(
        '--path',
        default='.',
        help='Path to analyze (default: current directory)'
    )
    parser.add_argument(
        '--output',
        choices=['json', 'markdown', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--output-file',
        help='Output file for JSON report'
    )
    parser.add_argument(
        '--markdown-file',
        help='Output file for markdown report'
    )

    args = parser.parse_args()

    # Determine paths to analyze
    root_path = Path(args.path)

    if args.app:
        # Analyze specific app
        app_paths = list(root_path.glob(f"**/{args.app}/**/*.py"))
        if not app_paths:
            logger.error(f"No files found for app {args.app}", file=sys.stderr)
            return 1

        metrics = [analyze_file(p) for p in app_paths]
        report = {
            'app': args.app,
            'overall': aggregate_metrics(metrics),
            'files': [asdict(m) for m in metrics]
        }

    else:
        # Analyze all apps
        all_metrics = []
        by_app = {}

        for py_file in root_path.rglob('*.py'):
            # Skip unwanted directories
            if any(part.startswith('.') or part in {'venv', 'env', 'build', 'dist', '__pycache__'}
                   for part in py_file.parts):
                continue

            metrics = analyze_file(py_file)
            all_metrics.append(metrics)

            # Try to determine app
            for part in py_file.parts:
                if part.startswith('GL-') or part.endswith('-APP'):
                    by_app.setdefault(part, []).append(metrics)
                    break

        report = {
            'overall': aggregate_metrics(all_metrics),
            'by_app': {app: aggregate_metrics(mlist) for app, mlist in by_app.items()},
            'files': [asdict(m) for m in all_metrics]
        }

    # Output JSON
    if args.output in ['json', 'both']:
        json_output = json.dumps(report, indent=2)

        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(json_output)
            print(f"JSON report written to {args.output_file}")
        else:
            print(json_output)

    # Output Markdown
    if args.output in ['markdown', 'both']:
        md_output = format_markdown(report)

        if args.markdown_file:
            with open(args.markdown_file, 'w') as f:
                f.write(md_output)
            print(f"Markdown report written to {args.markdown_file}")
        else:
            print(md_output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
