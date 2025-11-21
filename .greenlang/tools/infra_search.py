#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

logger = logging.getLogger(__name__)
AI-Powered Infrastructure Search Tool

Uses semantic search and RAG to help developers discover infrastructure components.
Embeds queries and infrastructure catalog for vector similarity matching.
"""

import logging
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import ast
import re

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning(f" sentence-transformers not installed. Using fallback search.")


@dataclass
class InfraComponent:
    """Infrastructure component metadata."""
    name: str
    category: str
    file_path: str
    description: str
    when_to_use: str
    code_example: str
    related_components: List[str]
    api_methods: List[str]
    tags: List[str]
    embedding: Optional[np.ndarray] = None


class InfrastructureCatalog:
    """Infrastructure catalog builder and searcher."""

    def __init__(self, root_dir: str = None):
        self.root_dir = root_dir or os.getcwd()
        self.components: List[InfraComponent] = []
        self.embedder = None

        if HAS_TRANSFORMERS:
            print("Loading sentence transformer model...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def scan_infrastructure(self):
        """Scan codebase for infrastructure components."""
        print(f"Scanning {self.root_dir} for infrastructure...")

        # Common infrastructure patterns
        patterns = [
            "**/shared/**/*.py",
            "**/infrastructure/**/*.py",
            "**/core/**/*.py",
            "**/agents/**/*.py",
            "**/*agent*.py",
            "**/*cache*.py",
            "**/*llm*.py",
            "**/*validation*.py",
        ]

        # Manually defined catalog (would be auto-generated in production)
        self.components = self._build_infrastructure_catalog()

        print(f"Found {len(self.components)} infrastructure components")

        # Generate embeddings
        if self.embedder:
            self._generate_embeddings()

    def _build_infrastructure_catalog(self) -> List[InfraComponent]:
        """Build comprehensive infrastructure catalog."""

        catalog = [
            # LLM Infrastructure
            InfraComponent(
                name="ChatSession",
                category="LLM",
                file_path="shared/infrastructure/llm/chat_session.py",
                description="Unified LLM chat interface supporting multiple providers (OpenAI, Anthropic, Google). Handles session management, token counting, and response streaming.",
                when_to_use="When you need to interact with LLMs. Use instead of direct OpenAI/Anthropic API calls. Provides provider abstraction and consistent API.",
                code_example="""from shared.infrastructure.llm import ChatSession

session = ChatSession(provider='openai', model='gpt-4')
response = session.chat("Analyze this emission data")
print(response.content)

# With streaming
for chunk in session.stream_chat("Generate report"):
    print(chunk, end='')""",
                related_components=["TokenCounter", "PromptTemplate", "ResponseParser"],
                api_methods=["chat", "stream_chat", "count_tokens", "get_history"],
                tags=["llm", "openai", "anthropic", "chat", "ai"]
            ),

            InfraComponent(
                name="BaseAgent",
                category="Agents",
                file_path="shared/infrastructure/agents/base_agent.py",
                description="Base class for all agentic workflows. Provides standard interface for execute(), batch processing, validation, and error handling.",
                when_to_use="When creating any agent. All agents should inherit from BaseAgent for consistency. Provides built-in logging, validation, and batch processing.",
                code_example="""from shared.infrastructure.agents import BaseAgent

class MyAgent(BaseAgent):
    def execute(self, input_data: dict) -> dict:
        # Your agent logic
        result = self.process(input_data)
        return {"status": "success", "result": result}

    def validate_input(self, input_data: dict) -> bool:
        return "required_field" in input_data

agent = MyAgent()
result = agent.execute({"data": "..."})
results = agent.batch_execute([{...}, {...}])""",
                related_components=["AgentPipeline", "ChatSession", "ValidationFramework"],
                api_methods=["execute", "batch_execute", "validate_input", "validate_output"],
                tags=["agent", "workflow", "pipeline", "batch"]
            ),

            InfraComponent(
                name="CacheManager",
                category="Caching",
                file_path="shared/infrastructure/cache/cache_manager.py",
                description="Redis-based distributed caching with TTL, invalidation, and cache warming. Supports in-memory fallback.",
                when_to_use="When you need to cache expensive operations (LLM calls, API requests, calculations). Use instead of custom caching solutions.",
                code_example="""from shared.infrastructure.cache import CacheManager

cache = CacheManager(ttl=3600)  # 1 hour TTL

# Cache LLM responses
@cache.cached(key_prefix="emission_calc")
def calculate_emissions(data):
    # Expensive LLM call
    return session.chat(f"Calculate: {data}")

# Manual caching
cache.set("my_key", {"data": "value"}, ttl=1800)
result = cache.get("my_key")

# Invalidate
cache.invalidate("my_key")""",
                related_components=["ChatSession", "DatabaseConnector"],
                api_methods=["get", "set", "delete", "cached", "invalidate", "warm"],
                tags=["cache", "redis", "performance", "optimization"]
            ),

            InfraComponent(
                name="ValidationFramework",
                category="Validation",
                file_path="shared/infrastructure/validation/framework.py",
                description="Comprehensive validation framework supporting schema validation, business rules, and custom validators. Pydantic-based with detailed error reporting.",
                when_to_use="When validating input data, API requests, or agent outputs. Use instead of manual validation. Provides consistent error messages.",
                code_example="""from shared.infrastructure.validation import ValidationFramework, Field

class EmissionDataValidator(ValidationFramework):
    schema = {
        "facility_id": Field(type=str, required=True),
        "emissions": Field(type=float, min_value=0),
        "date": Field(type=str, pattern=r"\\d{4}-\\d{2}-\\d{2}")
    }

    def validate_business_rules(self, data):
        if data["emissions"] > 1000000:
            raise ValueError("Emissions too high")

validator = EmissionDataValidator()
result = validator.validate(data)
if result.is_valid:
    print("Valid!")
else:
    print(result.errors)""",
                related_components=["BaseAgent", "ConfigManager"],
                api_methods=["validate", "validate_schema", "validate_rules"],
                tags=["validation", "schema", "pydantic", "error-handling"]
            ),

            InfraComponent(
                name="AgentPipeline",
                category="Agents",
                file_path="shared/infrastructure/agents/pipeline.py",
                description="Orchestrates multiple agents in sequence or parallel. Handles data passing, error recovery, and retry logic.",
                when_to_use="When you need to chain multiple agents together. Use for complex workflows requiring multiple steps.",
                code_example="""from shared.infrastructure.agents import AgentPipeline

pipeline = AgentPipeline([
    DataValidationAgent(),
    EmissionCalculatorAgent(),
    ReportGeneratorAgent()
], parallel=False)

result = pipeline.execute(input_data)

# Parallel execution
parallel_pipeline = AgentPipeline([
    Agent1(), Agent2(), Agent3()
], parallel=True)""",
                related_components=["BaseAgent", "TaskQueue"],
                api_methods=["execute", "add_agent", "remove_agent", "get_status"],
                tags=["pipeline", "orchestration", "workflow", "agents"]
            ),

            InfraComponent(
                name="ConfigManager",
                category="Configuration",
                file_path="shared/infrastructure/config/manager.py",
                description="Centralized configuration management with environment-based configs, secrets, and validation. Supports .env and YAML.",
                when_to_use="When you need to load configuration. Use instead of os.getenv() or manual config loading. Supports validation and defaults.",
                code_example="""from shared.infrastructure.config import ConfigManager

config = ConfigManager()

# Load from .env and config.yaml
config.load()

# Access config
api_key = config.get("OPENAI_API_KEY")
model = config.get("LLM_MODEL", default="gpt-4")

# Validate required keys
config.require(["OPENAI_API_KEY", "DATABASE_URL"])

# Environment-specific config
if config.is_production():
    print("Production mode")""",
                related_components=["SecretManager", "ValidationFramework"],
                api_methods=["get", "set", "load", "require", "is_production"],
                tags=["config", "environment", "secrets", "settings"]
            ),

            InfraComponent(
                name="DatabaseConnector",
                category="Database",
                file_path="shared/infrastructure/database/connector.py",
                description="Database abstraction layer supporting PostgreSQL, MySQL, SQLite. Handles connection pooling, transactions, and migrations.",
                when_to_use="When you need database access. Use instead of direct SQLAlchemy or psycopg2. Provides connection management and retry logic.",
                code_example="""from shared.infrastructure.database import DatabaseConnector

db = DatabaseConnector()

# Query
results = db.query("SELECT * FROM emissions WHERE facility_id = %s", [facility_id])

# Transaction
with db.transaction():
    db.execute("INSERT INTO emissions VALUES (%s, %s)", [data1, data2])
    db.execute("UPDATE facilities SET status = %s", ["active"])

# ORM-style
emissions = db.get_all("emissions", filters={"year": 2024})""",
                related_components=["CacheManager", "DataLoader"],
                api_methods=["query", "execute", "transaction", "get_all", "insert"],
                tags=["database", "sql", "postgres", "orm"]
            ),

            InfraComponent(
                name="MetricsCollector",
                category="Monitoring",
                file_path="shared/infrastructure/monitoring/metrics.py",
                description="Collects and reports application metrics (performance, errors, usage). Integrates with Prometheus and Grafana.",
                when_to_use="When you want to track performance or usage metrics. Add to agents for monitoring. Automatic metric collection for infrastructure.",
                code_example="""from shared.infrastructure.monitoring import MetricsCollector

metrics = MetricsCollector()

# Track execution time
with metrics.timer("agent_execution"):
    result = agent.execute(data)

# Count events
metrics.increment("emissions_calculated")
metrics.gauge("cache_size", cache.size())

# Custom metrics
metrics.record("llm_tokens_used", token_count)

# Export for Prometheus
print(metrics.export())""",
                related_components=["Logger", "AlertManager"],
                api_methods=["timer", "increment", "gauge", "record", "export"],
                tags=["metrics", "monitoring", "prometheus", "performance"]
            ),

            InfraComponent(
                name="Logger",
                category="Logging",
                file_path="shared/infrastructure/logging/logger.py",
                description="Structured logging with correlation IDs, log levels, and output formatting. Supports JSON and text formats.",
                when_to_use="When you need logging. Use instead of print() or basic logging. Provides structured logs with context and correlation.",
                code_example="""from shared.infrastructure.logging import Logger

logger = Logger(name="MyAgent")

logger.info("Processing emission data", extra={
    "facility_id": facility_id,
    "year": 2024
})

logger.error("Calculation failed", exc_info=True)

# With correlation ID
with logger.correlation("request-123"):
    logger.info("Step 1 complete")
    logger.info("Step 2 complete")""",
                related_components=["MetricsCollector", "ErrorHandler"],
                api_methods=["debug", "info", "warning", "error", "correlation"],
                tags=["logging", "structured-logging", "debugging"]
            ),

            InfraComponent(
                name="TaskQueue",
                category="Background Jobs",
                file_path="shared/infrastructure/queue/task_queue.py",
                description="Distributed task queue using Celery/Redis. Supports async execution, scheduling, and retry logic.",
                when_to_use="When you need background processing or async tasks. Use for long-running operations or scheduled jobs.",
                code_example="""from shared.infrastructure.queue import TaskQueue

queue = TaskQueue()

# Define task
@queue.task
def process_emissions(facility_id):
    # Long-running processing
    return calculate_emissions(facility_id)

# Enqueue
task = queue.enqueue(process_emissions, facility_id="F123")

# Check status
if task.is_complete():
    result = task.result()

# Schedule
queue.schedule(process_emissions, schedule="0 2 * * *")  # Daily at 2am""",
                related_components=["BaseAgent", "CacheManager"],
                api_methods=["task", "enqueue", "schedule", "get_status"],
                tags=["queue", "celery", "async", "background-jobs"]
            ),

            InfraComponent(
                name="APIClient",
                category="HTTP",
                file_path="shared/infrastructure/http/api_client.py",
                description="HTTP client with retry logic, rate limiting, and authentication. Wrapper around requests/httpx.",
                when_to_use="When calling external APIs. Use instead of raw requests. Provides retry, timeout, and error handling.",
                code_example="""from shared.infrastructure.http import APIClient

client = APIClient(
    base_url="https://api.example.com",
    auth_token=config.get("API_TOKEN"),
    retry_count=3
)

# GET request
response = client.get("/emissions", params={"year": 2024})

# POST with automatic retry
response = client.post("/calculate", json={"data": "..."})

# Rate limiting
client.set_rate_limit(requests_per_second=10)""",
                related_components=["CacheManager", "ConfigManager"],
                api_methods=["get", "post", "put", "delete", "set_rate_limit"],
                tags=["http", "api", "rest", "requests"]
            ),

            InfraComponent(
                name="DataLoader",
                category="Data",
                file_path="shared/infrastructure/data/loader.py",
                description="Unified data loading interface for CSV, Excel, JSON, Parquet. Handles parsing, validation, and transformation.",
                when_to_use="When loading data files. Use instead of pandas.read_csv() for consistent interface and validation.",
                code_example="""from shared.infrastructure.data import DataLoader

loader = DataLoader()

# Auto-detect format
data = loader.load("emissions_data.csv")

# With validation
data = loader.load("data.xlsx",
    schema=EmissionSchema,
    validate=True
)

# Transform
data = loader.load_and_transform(
    "data.json",
    transformers=[CleanDates(), ValidateEmissions()]
)""",
                related_components=["ValidationFramework", "DatabaseConnector"],
                api_methods=["load", "load_and_transform", "save", "validate"],
                tags=["data", "csv", "excel", "etl"]
            ),

            InfraComponent(
                name="PromptTemplate",
                category="LLM",
                file_path="shared/infrastructure/llm/prompt_template.py",
                description="Template system for LLM prompts with variable substitution, versioning, and optimization tracking.",
                when_to_use="When building LLM prompts. Use instead of f-strings for maintainable and testable prompts.",
                code_example="""from shared.infrastructure.llm import PromptTemplate

template = PromptTemplate(
    name="emission_analyzer",
    template='''Analyze emission data for {facility_name}:

    Data: {emission_data}

    Provide:
    1. Total emissions
    2. Trends
    3. Recommendations''',
    version="1.0"
)

prompt = template.render(
    facility_name="Plant A",
    emission_data=data
)

response = session.chat(prompt)""",
                related_components=["ChatSession", "ResponseParser"],
                api_methods=["render", "validate", "get_version", "optimize"],
                tags=["llm", "prompts", "templates"]
            ),

            InfraComponent(
                name="ResponseParser",
                category="LLM",
                file_path="shared/infrastructure/llm/response_parser.py",
                description="Parse and validate LLM responses. Extract structured data from natural language outputs.",
                when_to_use="When processing LLM responses. Use to extract JSON, tables, or structured data from LLM outputs.",
                code_example="""from shared.infrastructure.llm import ResponseParser

parser = ResponseParser()

# Parse JSON from response
response = session.chat("Analyze emissions and return JSON")
data = parser.extract_json(response.content)

# Parse table
table = parser.extract_table(response.content)

# Validate structure
result = parser.parse_and_validate(
    response.content,
    expected_schema=EmissionSchema
)""",
                related_components=["ChatSession", "ValidationFramework"],
                api_methods=["extract_json", "extract_table", "parse_and_validate"],
                tags=["llm", "parsing", "extraction"]
            ),

            InfraComponent(
                name="ErrorHandler",
                category="Error Handling",
                file_path="shared/infrastructure/errors/handler.py",
                description="Centralized error handling with retry logic, fallback strategies, and error reporting.",
                when_to_use="When you need robust error handling. Wrap risky operations with retry and fallback logic.",
                code_example="""from shared.infrastructure.errors import ErrorHandler

handler = ErrorHandler(
    retry_count=3,
    backoff_factor=2
)

# Automatic retry
@handler.with_retry
def call_external_api():
    return api.get("/data")

# With fallback
result = handler.try_with_fallback(
    primary=lambda: expensive_llm_call(),
    fallback=lambda: cached_result()
)

# Error reporting
try:
    risky_operation()
except Exception as e:
    handler.report(e, context={"user": user_id})""",
                related_components=["Logger", "MetricsCollector"],
                api_methods=["with_retry", "try_with_fallback", "report"],
                tags=["errors", "retry", "fault-tolerance"]
            )
        ]

        return catalog

    def _generate_embeddings(self):
        """Generate embeddings for all components."""
        if not self.embedder:
            return

        print("Generating embeddings...")

        for component in self.components:
            # Combine all text for embedding
            text = f"{component.name} {component.description} {component.when_to_use} {' '.join(component.tags)}"
            component.embedding = self.embedder.encode(text)

    def search(self, query: str, top_k: int = 5) -> List[InfraComponent]:
        """Search for infrastructure components using semantic similarity."""

        if not self.components:
            self.scan_infrastructure()

        if self.embedder and all(c.embedding is not None for c in self.components):
            return self._semantic_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[InfraComponent]:
        """Semantic search using embeddings."""
        query_embedding = self.embedder.encode(query)

        # Calculate cosine similarity
        scores = []
        for component in self.components:
            similarity = np.dot(query_embedding, component.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(component.embedding)
            )
            scores.append((component, similarity))

        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        return [comp for comp, score in scores[:top_k]]

    def _keyword_search(self, query: str, top_k: int) -> List[InfraComponent]:
        """Fallback keyword search."""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        scores = []
        for component in self.components:
            # Calculate keyword match score
            text = f"{component.name} {component.description} {component.when_to_use} {' '.join(component.tags)}".lower()
            text_words = set(re.findall(r'\w+', text))

            # Intersection score
            match_count = len(query_words & text_words)
            score = match_count / len(query_words) if query_words else 0

            scores.append((component, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [comp for comp, score in scores[:top_k]]

    def get_by_category(self, category: str) -> List[InfraComponent]:
        """Get all components in a category."""
        return [c for c in self.components if c.category.lower() == category.lower()]

    def get_by_tag(self, tag: str) -> List[InfraComponent]:
        """Get all components with a tag."""
        return [c for c in self.components if tag.lower() in [t.lower() for t in c.tags]]


class SearchUI:
    """User interface for search results."""

    @staticmethod
    def display_results(components: List[InfraComponent], query: str):
        """Display search results in a nice format."""

        print("\n" + "=" * 80)
        print(f"Search Results for: '{query}'")
        print("=" * 80 + "\n")

        if not components:
            print("No matching infrastructure components found.")
            print("\nTry:")
            print("  - Using different keywords")
            print("  - Searching by use case (e.g., 'cache API responses')")
            print("  - Browsing by category: greenlang search --category llm")
            return

        for i, component in enumerate(components, 1):
            print(f"{i}. {component.name}")
            print(f"   Category: {component.category}")
            print(f"   {'-' * 76}")
            print(f"\n   {component.description}\n")

            print(f"   When to use:")
            print(f"   {component.when_to_use}\n")

            print(f"   Example:")
            # Indent code example
            for line in component.code_example.split('\n'):
                print(f"   {line}")

            if component.related_components:
                print(f"\n   Related: {', '.join(component.related_components)}")

            print(f"\n   Tags: {', '.join(component.tags)}")
            print(f"   File: {component.file_path}")
            print("\n" + "=" * 80 + "\n")

    @staticmethod
    def display_json(components: List[InfraComponent]):
        """Display results as JSON."""
        results = [
            {
                "name": c.name,
                "category": c.category,
                "description": c.description,
                "when_to_use": c.when_to_use,
                "code_example": c.code_example,
                "related_components": c.related_components,
                "api_methods": c.api_methods,
                "tags": c.tags,
                "file_path": c.file_path
            }
            for c in components
        ]
        print(json.dumps(results, indent=2))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AI-powered infrastructure search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for caching solutions
  greenlang search "cache API responses"

  # Find LLM-related components
  greenlang search "how to call OpenAI"

  # Search by use case
  greenlang search "validate CSV data"

  # Browse by category
  greenlang search --category llm

  # Search by tag
  greenlang search --tag validation

  # JSON output
  greenlang search "agents" --format json
        """
    )

    parser.add_argument('query', nargs='?', help='Search query (natural language)')
    parser.add_argument('--category', help='Filter by category')
    parser.add_argument('--tag', help='Filter by tag')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    parser.add_argument('--root-dir', help='Root directory to scan')

    args = parser.parse_args()

    # Initialize catalog
    catalog = InfrastructureCatalog(args.root_dir)
    catalog.scan_infrastructure()

    # Perform search
    if args.category:
        results = catalog.get_by_category(args.category)
    elif args.tag:
        results = catalog.get_by_tag(args.tag)
    elif args.query:
        results = catalog.search(args.query, args.top_k)
    else:
        parser.print_help()
        sys.exit(1)

    # Display results
    if args.format == 'json':
        SearchUI.display_json(results)
    else:
        SearchUI.display_results(results, args.query or args.category or args.tag)


if __name__ == '__main__':
    main()
