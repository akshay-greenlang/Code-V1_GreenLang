#!/usr/bin/env python3
"""
Automatic Code Recommendation Engine

Analyzes code and suggests infrastructure improvements.
Detects anti-patterns and recommends appropriate infrastructure components.
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class Recommendation:
    """Code recommendation."""
    file_path: str
    line_number: int
    severity: str  # "error", "warning", "info"
    category: str
    title: str
    description: str
    current_code: str
    suggested_code: str
    why: str
    benefits: List[str]
    related_infrastructure: str
    auto_fixable: bool


class PatternDetector:
    """Detect code patterns that should use infrastructure."""

    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.split('\n')
        self.tree = None
        self.recommendations: List[Recommendation] = []

        try:
            self.tree = ast.parse(content)
        except SyntaxError:
            pass

    def detect_all(self) -> List[Recommendation]:
        """Run all pattern detectors."""

        self.detect_direct_llm_usage()
        self.detect_custom_caching()
        self.detect_manual_validation()
        self.detect_custom_agents()
        self.detect_direct_openai()
        self.detect_manual_config()
        self.detect_print_statements()
        self.detect_raw_requests()
        self.detect_manual_retry()
        self.detect_hardcoded_prompts()
        self.detect_manual_error_handling()
        self.detect_custom_logging()

        return self.recommendations

    def add_recommendation(self, **kwargs):
        """Add a recommendation."""
        self.recommendations.append(Recommendation(**kwargs))

    def detect_direct_openai(self):
        """Detect direct OpenAI API usage."""
        if not self.tree:
            return

        # Look for openai.ChatCompletion or openai.Completion
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'attr') and hasattr(node.func.value, 'id'):
                    if node.func.value.id == 'openai' and node.func.attr in ['ChatCompletion', 'Completion']:
                        line_num = node.lineno

                        self.add_recommendation(
                            file_path=self.file_path,
                            line_number=line_num,
                            severity="warning",
                            category="LLM",
                            title="Direct OpenAI API usage detected",
                            description="Using OpenAI API directly instead of ChatSession infrastructure",
                            current_code=self.lines[line_num - 1] if line_num <= len(self.lines) else "",
                            suggested_code="""from shared.infrastructure.llm import ChatSession

session = ChatSession(provider='openai', model='gpt-4')
response = session.chat("Your prompt here")""",
                            why="ChatSession provides provider abstraction, token counting, caching, and consistent error handling",
                            benefits=[
                                "Switch between OpenAI/Anthropic/Google without code changes",
                                "Automatic token counting and cost tracking",
                                "Built-in retry logic and error handling",
                                "Conversation history management",
                                "Integration with caching and monitoring"
                            ],
                            related_infrastructure="shared.infrastructure.llm.ChatSession",
                            auto_fixable=True
                        )

    def detect_direct_llm_usage(self):
        """Detect direct Anthropic/Google AI usage."""
        patterns = [
            (r'anthropic\.', 'Anthropic'),
            (r'google\.generativeai', 'Google AI'),
            (r'from anthropic import', 'Anthropic'),
        ]

        for i, line in enumerate(self.lines, 1):
            for pattern, provider in patterns:
                if re.search(pattern, line):
                    self.add_recommendation(
                        file_path=self.file_path,
                        line_number=i,
                        severity="warning",
                        category="LLM",
                        title=f"Direct {provider} usage detected",
                        description=f"Using {provider} API directly instead of ChatSession",
                        current_code=line.strip(),
                        suggested_code=f"""from shared.infrastructure.llm import ChatSession

session = ChatSession(provider='{provider.lower().replace(' ', '')}')
response = session.chat("Your prompt")""",
                        why="ChatSession provides unified interface across all LLM providers",
                        benefits=[
                            "Provider-agnostic code",
                            "Easy A/B testing between providers",
                            "Centralized configuration",
                            "Cost tracking across providers"
                        ],
                        related_infrastructure="shared.infrastructure.llm.ChatSession",
                        auto_fixable=True
                    )

    def detect_custom_caching(self):
        """Detect custom caching implementations."""
        cache_patterns = [
            r'cache\s*=\s*\{\}',
            r'@lru_cache',
            r'redis\.Redis\(',
            r'memcache',
            r'if .+ in cache:',
            r'cache\[.+\]\s*='
        ]

        for i, line in enumerate(self.lines, 1):
            for pattern in cache_patterns:
                if re.search(pattern, line):
                    self.add_recommendation(
                        file_path=self.file_path,
                        line_number=i,
                        severity="info",
                        category="Caching",
                        title="Custom caching detected",
                        description="Using custom caching instead of CacheManager infrastructure",
                        current_code=line.strip(),
                        suggested_code="""from shared.infrastructure.cache import CacheManager

cache = CacheManager(ttl=3600)

# Decorator
@cache.cached(key_prefix="my_function")
def my_function(args):
    return expensive_operation(args)

# Manual
cache.set("key", value, ttl=1800)
result = cache.get("key")""",
                        why="CacheManager provides distributed caching, TTL management, and cache warming",
                        benefits=[
                            "Redis-backed distributed caching",
                            "Automatic TTL and invalidation",
                            "Cache warming and preloading",
                            "In-memory fallback",
                            "Metrics and monitoring"
                        ],
                        related_infrastructure="shared.infrastructure.cache.CacheManager",
                        auto_fixable=False
                    )

    def detect_manual_validation(self):
        """Detect manual validation code."""
        validation_patterns = [
            r'if not .+:',
            r'raise ValueError\(',
            r'assert isinstance\(',
            r'if len\(.+\) == 0:',
            r'if .+ is None:',
            r'\.get\(.+\) or raise'
        ]

        # Count validation patterns
        validation_count = 0
        for line in self.lines:
            for pattern in validation_patterns:
                if re.search(pattern, line):
                    validation_count += 1

        # If lots of validation, suggest framework
        if validation_count >= 3:
            self.add_recommendation(
                file_path=self.file_path,
                line_number=1,
                severity="info",
                category="Validation",
                title="Manual validation detected",
                description=f"Found {validation_count} manual validation checks",
                current_code="Multiple if/raise validation statements",
                suggested_code="""from shared.infrastructure.validation import ValidationFramework, Field

class MyDataValidator(ValidationFramework):
    schema = {
        "field1": Field(type=str, required=True),
        "field2": Field(type=int, min_value=0, max_value=100),
        "email": Field(type=str, pattern=r'^[\\w\\.]+@[\\w\\.]+$')
    }

    def validate_business_rules(self, data):
        if data['field1'] == 'invalid':
            raise ValueError("Invalid field1")

validator = MyDataValidator()
result = validator.validate(data)
if not result.is_valid:
    print(result.errors)""",
                why="ValidationFramework provides consistent validation with detailed error messages",
                benefits=[
                    "Schema-based validation",
                    "Custom business rules",
                    "Detailed error reporting",
                    "Type checking and conversion",
                    "Integration with Pydantic"
                ],
                related_infrastructure="shared.infrastructure.validation.ValidationFramework",
                auto_fixable=False
            )

    def detect_custom_agents(self):
        """Detect custom agent classes not using BaseAgent."""
        if not self.tree:
            return

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                # Check if class has "Agent" in name but doesn't inherit from BaseAgent
                if 'agent' in node.name.lower():
                    base_names = [base.id for base in node.bases if hasattr(base, 'id')]

                    if 'BaseAgent' not in base_names:
                        self.add_recommendation(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            severity="warning",
                            category="Agents",
                            title="Custom agent class detected",
                            description=f"Agent class '{node.name}' not inheriting from BaseAgent",
                            current_code=f"class {node.name}:",
                            suggested_code=f"""from shared.infrastructure.agents import BaseAgent

class {node.name}(BaseAgent):
    def execute(self, input_data: dict) -> dict:
        # Your agent logic
        return {{"status": "success", "result": result}}

    def validate_input(self, input_data: dict) -> bool:
        return True  # Add validation""",
                            why="BaseAgent provides standard interface and built-in functionality",
                            benefits=[
                                "Consistent agent interface",
                                "Built-in batch processing",
                                "Validation framework",
                                "Logging and monitoring",
                                "Error handling",
                                "Pipeline integration"
                            ],
                            related_infrastructure="shared.infrastructure.agents.BaseAgent",
                            auto_fixable=False
                        )

    def detect_manual_config(self):
        """Detect manual config loading."""
        config_patterns = [
            r'os\.getenv\(',
            r'os\.environ\[',
            r'dotenv\.load_dotenv\(',
            r'with open\(.+config',
            r'ConfigParser\('
        ]

        for i, line in enumerate(self.lines, 1):
            for pattern in config_patterns:
                if re.search(pattern, line):
                    self.add_recommendation(
                        file_path=self.file_path,
                        line_number=i,
                        severity="info",
                        category="Configuration",
                        title="Manual configuration loading",
                        description="Loading config manually instead of using ConfigManager",
                        current_code=line.strip(),
                        suggested_code="""from shared.infrastructure.config import ConfigManager

config = ConfigManager()
config.load()

api_key = config.get("API_KEY")
db_url = config.get("DATABASE_URL")
config.require(["API_KEY", "DATABASE_URL"])""",
                        why="ConfigManager provides centralized, validated configuration",
                        benefits=[
                            "Environment-based configs",
                            "Required key validation",
                            "Type conversion",
                            "Secrets management",
                            "Default values"
                        ],
                        related_infrastructure="shared.infrastructure.config.ConfigManager",
                        auto_fixable=True
                    )

    def detect_print_statements(self):
        """Detect print() instead of proper logging."""
        for i, line in enumerate(self.lines, 1):
            if re.search(r'\bprint\s*\(', line) and 'logger' not in self.content[:500]:
                self.add_recommendation(
                    file_path=self.file_path,
                    line_number=i,
                    severity="info",
                    category="Logging",
                    title="Using print() instead of Logger",
                    description="Print statements should use structured logging",
                    current_code=line.strip(),
                    suggested_code="""from shared.infrastructure.logging import Logger

logger = Logger(name=__name__)
logger.info("Your message", extra={"context": "value"})""",
                    why="Logger provides structured, searchable logs with correlation",
                    benefits=[
                        "Structured logging",
                        "Correlation IDs",
                        "Log levels",
                        "JSON/text output",
                        "Integration with monitoring"
                    ],
                    related_infrastructure="shared.infrastructure.logging.Logger",
                    auto_fixable=True
                )
                break  # Only recommend once per file

    def detect_raw_requests(self):
        """Detect raw requests library usage."""
        if re.search(r'import requests|from requests import', self.content):
            for i, line in enumerate(self.lines, 1):
                if re.search(r'requests\.(get|post|put|delete)', line):
                    self.add_recommendation(
                        file_path=self.file_path,
                        line_number=i,
                        severity="info",
                        category="HTTP",
                        title="Raw requests usage",
                        description="Using requests directly instead of APIClient",
                        current_code=line.strip(),
                        suggested_code="""from shared.infrastructure.http import APIClient

client = APIClient(
    base_url="https://api.example.com",
    auth_token=config.get("API_TOKEN"),
    retry_count=3
)

response = client.get("/endpoint")""",
                        why="APIClient provides retry, rate limiting, and error handling",
                        benefits=[
                            "Automatic retry logic",
                            "Rate limiting",
                            "Timeout handling",
                            "Authentication management",
                            "Request/response logging"
                        ],
                        related_infrastructure="shared.infrastructure.http.APIClient",
                        auto_fixable=False
                    )
                    break

    def detect_manual_retry(self):
        """Detect manual retry logic."""
        retry_patterns = [
            r'for .+ in range\(.+\):.*try:',
            r'while .+:.*try:',
            r'retry_count\s*=',
            r'except.*continue'
        ]

        for i, line in enumerate(self.lines, 1):
            for pattern in retry_patterns:
                if re.search(pattern, line):
                    self.add_recommendation(
                        file_path=self.file_path,
                        line_number=i,
                        severity="info",
                        category="Error Handling",
                        title="Manual retry logic detected",
                        description="Using custom retry instead of ErrorHandler",
                        current_code=line.strip(),
                        suggested_code="""from shared.infrastructure.errors import ErrorHandler

handler = ErrorHandler(retry_count=3, backoff_factor=2)

@handler.with_retry
def risky_operation():
    return api.call()

# Or with fallback
result = handler.try_with_fallback(
    primary=lambda: expensive_operation(),
    fallback=lambda: cached_result()
)""",
                        why="ErrorHandler provides consistent retry with exponential backoff",
                        benefits=[
                            "Configurable retry logic",
                            "Exponential backoff",
                            "Fallback strategies",
                            "Error reporting",
                            "Circuit breaker pattern"
                        ],
                        related_infrastructure="shared.infrastructure.errors.ErrorHandler",
                        auto_fixable=False
                    )
                    break

    def detect_hardcoded_prompts(self):
        """Detect hardcoded LLM prompts."""
        # Look for long f-strings or triple-quoted strings
        for i, line in enumerate(self.lines, 1):
            if re.search(r'f["\'].*{.*}.*["\']', line) and ('prompt' in line.lower() or 'instruction' in line.lower()):
                self.add_recommendation(
                    file_path=self.file_path,
                    line_number=i,
                    severity="info",
                    category="LLM",
                    title="Hardcoded prompt detected",
                    description="Using hardcoded prompts instead of PromptTemplate",
                    current_code=line.strip()[:100] + "...",
                    suggested_code="""from shared.infrastructure.llm import PromptTemplate

template = PromptTemplate(
    name="my_prompt",
    template='''Analyze {data_type}:

    Data: {data}

    Provide: {requirements}''',
    version="1.0"
)

prompt = template.render(
    data_type="emissions",
    data=emission_data,
    requirements="total and trends"
)""",
                    why="PromptTemplate enables versioning, testing, and optimization",
                    benefits=[
                        "Prompt versioning",
                        "Variable validation",
                        "A/B testing",
                        "Centralized prompt management",
                        "Performance tracking"
                    ],
                    related_infrastructure="shared.infrastructure.llm.PromptTemplate",
                    auto_fixable=False
                )
                break

    def detect_custom_logging(self):
        """Detect custom logging setup."""
        logging_patterns = [
            r'logging\.basicConfig\(',
            r'logging\.getLogger\(',
            r'logger\s*=\s*logging\.'
        ]

        for i, line in enumerate(self.lines, 1):
            for pattern in logging_patterns:
                if re.search(pattern, line):
                    self.add_recommendation(
                        file_path=self.file_path,
                        line_number=i,
                        severity="info",
                        category="Logging",
                        title="Custom logging setup",
                        description="Using basic logging instead of infrastructure Logger",
                        current_code=line.strip(),
                        suggested_code="""from shared.infrastructure.logging import Logger

logger = Logger(name=__name__)

logger.info("Message", extra={"key": "value"})
logger.error("Error occurred", exc_info=True)

# With correlation
with logger.correlation("request-123"):
    logger.info("Processing...")""",
                        why="Infrastructure Logger provides structured logging with correlation",
                        benefits=[
                            "Structured JSON logs",
                            "Correlation IDs",
                            "Context propagation",
                            "Integration with monitoring",
                            "Consistent format"
                        ],
                        related_infrastructure="shared.infrastructure.logging.Logger",
                        auto_fixable=True
                    )
                    break


class CodeRecommender:
    """Main recommendation engine."""

    def __init__(self):
        self.all_recommendations: List[Recommendation] = []

    def analyze_file(self, file_path: str) -> List[Recommendation]:
        """Analyze a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            detector = PatternDetector(file_path, content)
            return detector.detect_all()

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []

    def analyze_directory(self, directory: str) -> List[Recommendation]:
        """Analyze all Python files in a directory."""
        recommendations = []

        for root, dirs, files in os.walk(directory):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', 'node_modules', '.venv']]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    recs = self.analyze_file(file_path)
                    recommendations.extend(recs)

        return recommendations

    def generate_report(self, recommendations: List[Recommendation], format: str = 'text') -> str:
        """Generate report from recommendations."""

        if format == 'json':
            return self._generate_json_report(recommendations)
        elif format == 'html':
            return self._generate_html_report(recommendations)
        else:
            return self._generate_text_report(recommendations)

    def _generate_text_report(self, recommendations: List[Recommendation]) -> str:
        """Generate text report."""
        if not recommendations:
            return "No recommendations found. Code looks good!"

        # Group by category
        by_category = {}
        for rec in recommendations:
            if rec.category not in by_category:
                by_category[rec.category] = []
            by_category[rec.category].append(rec)

        output = []
        output.append("=" * 80)
        output.append("CODE RECOMMENDATIONS REPORT")
        output.append("=" * 80)
        output.append(f"\nTotal recommendations: {len(recommendations)}")
        output.append(f"Categories: {', '.join(by_category.keys())}\n")

        for category, recs in by_category.items():
            output.append(f"\n{category.upper()} ({len(recs)} issues)")
            output.append("-" * 80)

            for i, rec in enumerate(recs, 1):
                output.append(f"\n{i}. {rec.title}")
                output.append(f"   File: {rec.file_path}:{rec.line_number}")
                output.append(f"   Severity: {rec.severity.upper()}")
                output.append(f"\n   {rec.description}")
                output.append(f"\n   Why: {rec.why}")
                output.append(f"\n   Benefits:")
                for benefit in rec.benefits:
                    output.append(f"   - {benefit}")
                output.append(f"\n   Infrastructure: {rec.related_infrastructure}")
                output.append(f"   Auto-fixable: {'Yes' if rec.auto_fixable else 'No'}")
                output.append(f"\n   Suggested code:")
                for line in rec.suggested_code.split('\n'):
                    output.append(f"   {line}")
                output.append("")

        return "\n".join(output)

    def _generate_json_report(self, recommendations: List[Recommendation]) -> str:
        """Generate JSON report."""
        data = {
            "total_recommendations": len(recommendations),
            "by_severity": self._count_by_field(recommendations, 'severity'),
            "by_category": self._count_by_field(recommendations, 'category'),
            "auto_fixable_count": sum(1 for r in recommendations if r.auto_fixable),
            "recommendations": [asdict(r) for r in recommendations]
        }
        return json.dumps(data, indent=2)

    def _generate_html_report(self, recommendations: List[Recommendation]) -> str:
        """Generate HTML report."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Code Recommendations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }
        h1 { color: #2c3e50; }
        .summary { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .recommendation { border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 5px; }
        .recommendation.error { border-left: 4px solid #e74c3c; }
        .recommendation.warning { border-left: 4px solid #f39c12; }
        .recommendation.info { border-left: 4px solid #3498db; }
        .title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        .meta { color: #7f8c8d; font-size: 14px; margin-bottom: 10px; }
        .code { background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 3px; overflow-x: auto; }
        .benefits { margin: 10px 0; }
        .benefits li { margin: 5px 0; }
        .tag { display: inline-block; padding: 3px 8px; margin: 2px; background: #3498db; color: white; border-radius: 3px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Code Recommendations Report</h1>
        <div class="summary">
            <strong>Total Recommendations:</strong> {total}<br>
            <strong>Auto-fixable:</strong> {auto_fixable}<br>
            <strong>Categories:</strong> {categories}
        </div>
"""

        total = len(recommendations)
        auto_fixable = sum(1 for r in recommendations if r.auto_fixable)
        categories = ', '.join(set(r.category for r in recommendations))

        html = html.format(total=total, auto_fixable=auto_fixable, categories=categories)

        for rec in recommendations:
            html += f"""
        <div class="recommendation {rec.severity}">
            <div class="title">{rec.title}</div>
            <div class="meta">
                {rec.file_path}:{rec.line_number} |
                <span class="tag">{rec.category}</span>
                <span class="tag">{rec.severity}</span>
                {('<span class="tag">Auto-fixable</span>' if rec.auto_fixable else '')}
            </div>
            <p>{rec.description}</p>
            <p><strong>Why:</strong> {rec.why}</p>
            <div class="benefits">
                <strong>Benefits:</strong>
                <ul>
                {''.join(f'<li>{b}</li>' for b in rec.benefits)}
                </ul>
            </div>
            <p><strong>Infrastructure:</strong> <code>{rec.related_infrastructure}</code></p>
            <strong>Suggested Code:</strong>
            <pre class="code">{rec.suggested_code}</pre>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html

    def _count_by_field(self, recommendations: List[Recommendation], field: str) -> Dict[str, int]:
        """Count recommendations by field."""
        counts = {}
        for rec in recommendations:
            value = getattr(rec, field)
            counts[value] = counts.get(value, 0) + 1
        return counts


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Automatic code recommendation engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a file
  greenlang recommend mycode.py

  # Analyze directory
  greenlang recommend GL-CBAM-APP/

  # Generate HTML report
  greenlang recommend . --format html --output recommendations.html

  # Show only auto-fixable
  greenlang recommend . --auto-fixable-only

  # Filter by category
  greenlang recommend . --category LLM
        """
    )

    parser.add_argument('path', help='File or directory to analyze')
    parser.add_argument('--format', choices=['text', 'json', 'html'], default='text', help='Output format')
    parser.add_argument('--output', help='Output file (default: stdout)')
    parser.add_argument('--auto-fixable-only', action='store_true', help='Show only auto-fixable issues')
    parser.add_argument('--category', help='Filter by category')
    parser.add_argument('--severity', choices=['error', 'warning', 'info'], help='Filter by severity')

    args = parser.parse_args()

    # Analyze
    recommender = CodeRecommender()

    if os.path.isfile(args.path):
        recommendations = recommender.analyze_file(args.path)
    else:
        recommendations = recommender.analyze_directory(args.path)

    # Filter
    if args.auto_fixable_only:
        recommendations = [r for r in recommendations if r.auto_fixable]

    if args.category:
        recommendations = [r for r in recommendations if r.category.lower() == args.category.lower()]

    if args.severity:
        recommendations = [r for r in recommendations if r.severity == args.severity]

    # Generate report
    report = recommender.generate_report(recommendations, args.format)

    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
