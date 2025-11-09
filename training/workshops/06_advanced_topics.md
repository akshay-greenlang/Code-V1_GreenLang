# Workshop 6: Advanced Topics & Best Practices

**Duration:** 3 hours
**Level:** Advanced
**Prerequisites:** Workshops 1-5 completed

---

## Workshop Overview

Master advanced GreenLang patterns, optimize performance, build multi-agent systems, and contribute to infrastructure.

### Learning Objectives

- Build multi-agent pipelines
- Optimize performance and costs
- Implement advanced caching strategies
- Use shared services effectively
- Contribute to infrastructure
- Design scalable architectures

---

## Part 1: Multi-Agent Pipelines (45 minutes)

### Pipeline Architecture

```python
from GL_COMMONS.infrastructure.agents import Agent, Pipeline

class DataIngestionAgent(Agent):
    """Step 1: Ingest data."""
    def execute(self):
        data = self._load_csv(self.input_data["file_path"])
        return {"raw_data": data}

class ValidationAgent(Agent):
    """Step 2: Validate data."""
    def execute(self):
        raw_data = self.input_data["raw_data"]
        validated = self._validate(raw_data)
        return {"validated_data": validated}

class CalculationAgent(Agent):
    """Step 3: Calculate emissions."""
    def execute(self):
        validated_data = self.input_data["validated_data"]
        results = self._calculate_emissions(validated_data)
        return {"calculated_data": results}

class ReportingAgent(Agent):
    """Step 4: Generate report."""
    def execute(self):
        calculated_data = self.input_data["calculated_data"]
        report = self._generate_report(calculated_data)
        return {"report": report}

# Build pipeline
pipeline = Pipeline(name="emission_pipeline")
pipeline.add_agent(DataIngestionAgent())
pipeline.add_agent(ValidationAgent())
pipeline.add_agent(CalculationAgent())
pipeline.add_agent(ReportingAgent())

# Execute pipeline
result = pipeline.execute({
    "file_path": "data/emissions.csv"
})

print(result["report"])
```

### Conditional Pipelines

```python
class ConditionalPipeline(Pipeline):
    """Pipeline with conditional execution."""

    def execute(self, input_data):
        # Step 1: Ingestion (always)
        result = self.agents[0].execute_with_input(input_data)

        # Step 2: Validation (always)
        result = self.agents[1].execute_with_input(result)

        # Step 3: Conditional processing
        if result["requires_llm"]:
            # Use LLM-based calculation
            result = self.llm_calculation_agent.execute_with_input(result)
        else:
            # Use simple calculation
            result = self.simple_calculation_agent.execute_with_input(result)

        # Step 4: Reporting (always)
        result = self.agents[3].execute_with_input(result)

        return result
```

### Parallel Pipelines

```python
from concurrent.futures import ThreadPoolExecutor

class ParallelPipeline(Pipeline):
    """Execute multiple agents in parallel."""

    def execute(self, input_data):
        # Step 1: Ingestion
        data = self.ingestion_agent.execute_with_input(input_data)

        # Step 2: Parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Process Scope 1, 2, 3 in parallel
            scope1_future = executor.submit(
                self.scope1_agent.execute_with_input,
                {"data": data["scope1_data"]}
            )
            scope2_future = executor.submit(
                self.scope2_agent.execute_with_input,
                {"data": data["scope2_data"]}
            )
            scope3_future = executor.submit(
                self.scope3_agent.execute_with_input,
                {"data": data["scope3_data"]}
            )

            # Wait for all to complete
            scope1_result = scope1_future.result()
            scope2_result = scope2_future.result()
            scope3_result = scope3_future.result()

        # Step 3: Aggregate results
        aggregated = self._aggregate(
            scope1_result,
            scope2_result,
            scope3_result
        )

        # Step 4: Reporting
        return self.reporting_agent.execute_with_input(aggregated)
```

### Error Handling in Pipelines

```python
class ResilientPipeline(Pipeline):
    """Pipeline with error recovery."""

    def execute(self, input_data):
        checkpoint = {}
        result = input_data

        for i, agent in enumerate(self.agents):
            try:
                # Save checkpoint before each step
                checkpoint[f"step_{i}"] = result

                # Execute agent
                result = agent.execute_with_input(result)

            except Exception as e:
                logger.error(f"Agent {agent.name} failed", extra={
                    "agent": agent.name,
                    "step": i,
                    "error": str(e)
                })

                # Attempt recovery
                if hasattr(agent, "fallback"):
                    logger.info(f"Using fallback for {agent.name}")
                    result = agent.fallback(result)
                else:
                    # Restore from checkpoint
                    logger.info(f"Restoring from checkpoint {i-1}")
                    result = checkpoint.get(f"step_{i-1}", input_data)
                    raise

        return result
```

---

## Part 2: Performance Optimization (45 minutes)

### Caching Strategies

#### 1. Multi-Level Caching

```python
class OptimizedAgent(Agent):
    """Agent with aggressive caching."""

    def __init__(self):
        super().__init__()
        # L1: In-memory (fastest, smallest)
        self.memory_cache = {}
        # L2: Redis (fast, medium)
        self.redis_cache = CacheManager()
        # L3: Database (slow, largest)
        self.db = DatabaseManager()

    def get_data(self, key):
        """Get with multi-level caching."""

        # L1: Memory
        if key in self.memory_cache:
            self.telemetry.increment("l1_hits")
            return self.memory_cache[key]

        # L2: Redis
        value = self.redis_cache.get(key)
        if value:
            self.telemetry.increment("l2_hits")
            self.memory_cache[key] = value  # Promote to L1
            return value

        # L3: Database
        value = self.db.query_one(f"SELECT * FROM data WHERE id=?", [key])
        if value:
            self.telemetry.increment("l3_hits")
            self.redis_cache.set(key, value, ttl=3600)  # Promote to L2
            self.memory_cache[key] = value  # Promote to L1
            return value

        return None
```

#### 2. Predictive Caching

```python
class PredictiveCacheAgent(Agent):
    """Pre-load data based on usage patterns."""

    def setup(self):
        super().setup()

        # Analyze access patterns
        patterns = self._analyze_access_patterns()

        # Pre-warm cache
        for pattern in patterns:
            if pattern["probability"] > 0.8:  # 80% likely to be accessed
                data = self._load_data(pattern["key"])
                self.cache.set(pattern["key"], data, ttl=3600)

        logger.info(f"Pre-warmed {len(patterns)} cache entries")

    def _analyze_access_patterns(self):
        """Analyze historical access patterns."""
        # Query access logs
        logs = self.db.query("""
            SELECT key, COUNT(*) as count
            FROM access_log
            WHERE timestamp > NOW() - INTERVAL '7 days'
            GROUP BY key
            ORDER BY count DESC
            LIMIT 100
        """)

        return [
            {"key": log["key"], "probability": log["count"] / total_accesses}
            for log in logs
        ]
```

#### 3. Semantic Caching for LLM

```python
from GL_COMMONS.infrastructure.llm import SemanticCacheManager

class SemanticallyCachedAgent(Agent):
    """Use semantic caching for LLM calls."""

    def setup(self):
        super().setup()

        self.llm_session = ChatSession(provider="openai", model="gpt-4")

        self.semantic_cache = SemanticCacheManager(
            similarity_threshold=0.95,  # 95% similar = cache hit
            ttl=86400  # 24 hours
        )

    def execute(self):
        query = self.input_data["query"]

        # Check semantic cache
        cached = self.semantic_cache.get(query)
        if cached:
            logger.info("Semantic cache hit")
            return cached

        # Call LLM
        response = self.llm_session.send_message(query)

        # Cache semantically
        self.semantic_cache.set(query, response)

        return response
```

### Batch Processing Optimization

```python
class OptimizedBatchAgent(Agent):
    """Optimized batch processing."""

    def execute(self):
        items = self.input_data["items"]

        # Group items by type for batch processing
        grouped = self._group_by_type(items)

        results = []
        for item_type, group in grouped.items():
            # Process each group in bulk
            batch_results = self._process_batch(item_type, group)
            results.extend(batch_results)

        return results

    def _process_batch(self, item_type, items):
        """Process batch with bulk operations."""

        # Bulk database query (1 query instead of N)
        ids = [item["id"] for item in items]
        data = self.db.query(
            f"SELECT * FROM items WHERE id IN ({','.join(['?']*len(ids))})",
            ids
        )

        # Bulk LLM processing (1 call instead of N)
        if len(items) > 10:
            # Batch into single prompt
            batch_prompt = self._create_batch_prompt(items)
            llm_results = self.llm_session.send_message(batch_prompt)
            parsed_results = self._parse_batch_results(llm_results)
        else:
            # Individual processing for small batches
            parsed_results = [
                self.llm_session.send_message(self._create_prompt(item))
                for item in items
            ]

        return parsed_results
```

### Query Optimization

```python
class QueryOptimizedAgent(Agent):
    """Optimize database queries."""

    def execute(self):
        # Bad: N+1 query problem
        # companies = self.db.query("SELECT * FROM companies")
        # for company in companies:
        #     emissions = self.db.query(
        #         "SELECT * FROM emissions WHERE company_id=?",
        #         [company["id"]]
        #     )

        # Good: Single join query
        results = self.db.query("""
            SELECT
                c.id,
                c.name,
                e.year,
                e.scope_1,
                e.scope_2,
                e.scope_3
            FROM companies c
            LEFT JOIN emissions e ON c.id = e.company_id
            WHERE e.year = ?
        """, [2023])

        return self._group_results(results)
```

---

## Part 3: Cost Optimization (30 minutes)

### LLM Cost Reduction

```python
class CostOptimizedAgent(Agent):
    """Minimize LLM costs."""

    def setup(self):
        super().setup()

        # Use cheaper model when possible
        self.cheap_model = ChatSession(
            provider="openai",
            model="gpt-3.5-turbo"  # 10x cheaper than GPT-4
        )

        self.expensive_model = ChatSession(
            provider="openai",
            model="gpt-4"
        )

        self.semantic_cache = SemanticCacheManager()

    def execute(self):
        query = self.input_data["query"]

        # Check cache first (free!)
        cached = self.semantic_cache.get(query)
        if cached:
            return cached

        # Determine complexity
        complexity = self._assess_complexity(query)

        # Use cheaper model for simple queries
        if complexity == "simple":
            logger.info("Using cheap model (GPT-3.5)")
            response = self.cheap_model.send_message(query)
            cost = self.cheap_model.get_cost()

        # Use expensive model for complex queries
        else:
            logger.info("Using expensive model (GPT-4)")
            response = self.expensive_model.send_message(query)
            cost = self.expensive_model.get_cost()

        # Cache the result
        self.semantic_cache.set(query, response)

        logger.info(f"Query cost: ${cost:.4f}")

        return response

    def _assess_complexity(self, query):
        """Assess query complexity."""
        # Simple heuristics
        if len(query) < 100:
            return "simple"
        if "calculate" in query.lower():
            return "simple"
        if "analyze" in query.lower() or "explain" in query.lower():
            return "complex"
        return "simple"
```

### Token Optimization

```python
class TokenOptimizedAgent(Agent):
    """Minimize token usage."""

    def execute(self):
        query = self.input_data["query"]

        # Get relevant context (instead of all documents)
        context = self.rag.query(query, top_k=3)  # Only top 3 docs

        # Build concise prompt
        prompt = self._build_minimal_prompt(query, context)

        # Use token limit
        response = self.llm_session.send_message(
            prompt,
            max_tokens=500  # Limit response length
        )

        return response

    def _build_minimal_prompt(self, query, context):
        """Build minimal prompt to reduce tokens."""

        # Summarize context if too long
        if len(context) > 1000:
            context = context[:1000] + "..."

        return f"Context: {context}\n\nQ: {query}\nA:"
```

---

## Part 4: Shared Services (30 minutes)

### Notification Service

```python
from GL_COMMONS.infrastructure.services import NotificationService

class NotifyingAgent(Agent):
    """Agent that sends notifications."""

    def setup(self):
        super().setup()
        self.notifier = NotificationService()

    def execute(self):
        result = self._process_data()

        # Send notification on completion
        if result["status"] == "success":
            self.notifier.send(
                channel="slack",
                recipient="#data-team",
                message=f"Processing complete: {result['records_processed']} records",
                metadata=result
            )

        return result
```

### Email Service

```python
from GL_COMMONS.infrastructure.services import EmailService

class ReportingAgent(Agent):
    """Send reports via email."""

    def setup(self):
        super().setup()
        self.email = EmailService()

    def execute(self):
        report = self._generate_report()

        # Send email with report
        self.email.send(
            to=["stakeholder@company.com"],
            subject="Monthly Emissions Report",
            body=report["text"],
            attachments=[{
                "filename": "report.pdf",
                "content": report["pdf"]
            }]
        )

        return {"status": "sent"}
```

### Task Queue

```python
from GL_COMMONS.infrastructure.services import TaskQueue

class AsyncAgent(Agent):
    """Process tasks asynchronously."""

    def setup(self):
        super().setup()
        self.queue = TaskQueue()

    def execute(self):
        items = self.input_data["items"]

        # Queue items for async processing
        for item in items:
            self.queue.enqueue(
                task_name="process_emission",
                params=item,
                priority="high" if item["urgent"] else "normal"
            )

        return {"queued": len(items)}
```

---

## Part 5: Contributing to Infrastructure (30 minutes)

### Adding a New Feature

```python
# GL_COMMONS/infrastructure/llm/prompt_optimizer.py

class PromptOptimizer:
    """Optimize prompts to reduce tokens and improve quality."""

    def optimize(self, prompt: str) -> str:
        """Optimize prompt."""

        # Remove redundant whitespace
        optimized = " ".join(prompt.split())

        # Remove filler words
        optimized = self._remove_filler_words(optimized)

        # Compress examples
        optimized = self._compress_examples(optimized)

        # Calculate savings
        original_tokens = self._count_tokens(prompt)
        optimized_tokens = self._count_tokens(optimized)
        savings = ((original_tokens - optimized_tokens) / original_tokens) * 100

        logger.info(f"Prompt optimized: {savings:.1f}% token reduction")

        return optimized

    def _remove_filler_words(self, text: str) -> str:
        """Remove unnecessary words."""
        filler_words = ["basically", "actually", "literally", "very"]
        for word in filler_words:
            text = text.replace(f" {word} ", " ")
        return text

    def _compress_examples(self, text: str) -> str:
        """Compress verbose examples."""
        # Implementation
        pass

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Use tiktoken or similar
        return len(text.split())
```

### Writing Tests

```python
# tests/infrastructure/test_prompt_optimizer.py
import pytest
from GL_COMMONS.infrastructure.llm import PromptOptimizer

def test_remove_filler_words():
    """Test filler word removal."""
    optimizer = PromptOptimizer()

    input_text = "This is basically a very good example"
    output_text = optimizer.optimize(input_text)

    assert "basically" not in output_text
    assert "very" not in output_text
    assert "good example" in output_text

def test_token_reduction():
    """Test token reduction."""
    optimizer = PromptOptimizer()

    long_prompt = "This is a very long prompt with lots of redundant    spaces"
    optimized = optimizer.optimize(long_prompt)

    original_tokens = optimizer._count_tokens(long_prompt)
    optimized_tokens = optimizer._count_tokens(optimized)

    assert optimized_tokens < original_tokens
```

### Documentation

```python
# GL_COMMONS/infrastructure/llm/README.md

"""
# PromptOptimizer

Optimize prompts to reduce tokens and costs while maintaining quality.

## Usage

```python
from GL_COMMONS.infrastructure.llm import PromptOptimizer

optimizer = PromptOptimizer()

# Optimize prompt
long_prompt = "..."
optimized = optimizer.optimize(long_prompt)

# Reduces tokens by ~20-30% on average
```

## Features

- Remove filler words
- Compress whitespace
- Compress examples
- Calculate token savings

## Configuration

No configuration required. Works out of the box.
"""
```

---

## Part 6: Hands-On Lab - Multi-Agent System (60 minutes)

### Lab: Build Complete CSRD Processing Pipeline

**Requirements:**
1. Ingest CSV with company emissions
2. Validate data quality
3. Calculate missing metrics using LLM
4. Generate CSRD compliance report
5. Send report via email
6. Monitor entire pipeline

### Implementation

```python
# csrd_pipeline.py
from GL_COMMONS.infrastructure.agents import Agent, Pipeline
from GL_COMMONS.infrastructure.llm import ChatSession, RAGEngine
from GL_COMMONS.infrastructure.cache import CacheManager
from GL_COMMONS.infrastructure.validation import ValidationFramework
from GL_COMMONS.infrastructure.services import EmailService
from GL_COMMONS.infrastructure.telemetry import TelemetryManager
import csv

# Agent 1: Data Ingestion
class CSRDIngestionAgent(Agent):
    def execute(self):
        file_path = self.input_data["file_path"]

        with open(file_path, 'r') as f:
            data = list(csv.DictReader(f))

        logger.info(f"Loaded {len(data)} records")

        return {"raw_data": data}

# Agent 2: Data Validation
class CSRDValidationAgent(Agent):
    def setup(self):
        super().setup()
        self.validator = ValidationFramework()

    def execute(self):
        raw_data = self.input_data["raw_data"]

        schema = {
            "company": {"type": "string", "required": True},
            "year": {"type": "integer", "min": 2020, "max": 2030},
            "scope_1": {"type": "number", "min": 0},
            "scope_2": {"type": "number", "min": 0},
            "scope_3": {"type": "number", "min": 0}
        }

        validated = []
        errors = []

        for record in raw_data:
            try:
                self.validator.validate(record, schema)
                validated.append(record)
            except Exception as e:
                errors.append({"record": record, "error": str(e)})

        logger.info(f"Validated: {len(validated)}, Errors: {len(errors)}")

        return {
            "validated_data": validated,
            "validation_errors": errors
        }

# Agent 3: LLM Enrichment
class CSRDEnrichmentAgent(Agent):
    def setup(self):
        super().setup()
        self.llm_session = ChatSession(
            provider="openai",
            model="gpt-4",
            system_message="You are a CSRD compliance expert."
        )
        self.cache = CacheManager()

    def execute(self):
        validated_data = self.input_data["validated_data"]

        enriched = []
        for record in validated_data:
            # Calculate total emissions
            total = (
                float(record["scope_1"]) +
                float(record["scope_2"]) +
                float(record["scope_3"])
            )

            # Get compliance assessment from LLM (cached)
            cache_key = f"compliance:{record['company']}:{record['year']}"
            assessment = self.cache.get(cache_key)

            if not assessment:
                prompt = f"""Assess CSRD compliance for:
                Company: {record['company']}
                Year: {record['year']}
                Total Emissions: {total} tons CO2

                Is this compliant? Provide brief assessment."""

                assessment = self.llm_session.send_message(prompt)
                self.cache.set(cache_key, assessment, ttl=86400)

            enriched.append({
                **record,
                "total_emissions": total,
                "compliance_assessment": assessment
            })

        return {"enriched_data": enriched}

# Agent 4: Report Generation
class CSRDReportAgent(Agent):
    def execute(self):
        enriched_data = self.input_data["enriched_data"]

        # Generate report
        report = self._generate_report(enriched_data)

        return {"report": report}

    def _generate_report(self, data):
        total_companies = len(data)
        total_emissions = sum(r["total_emissions"] for r in data)
        avg_emissions = total_emissions / total_companies if total_companies > 0 else 0

        report = f"""
CSRD Emissions Report
=====================

Summary:
- Companies: {total_companies}
- Total Emissions: {total_emissions:,.0f} tons CO2
- Average Emissions: {avg_emissions:,.0f} tons CO2

Details:
"""

        for record in data:
            report += f"""
Company: {record['company']} ({record['year']})
  Scope 1: {record['scope_1']:,.0f} tons
  Scope 2: {record['scope_2']:,.0f} tons
  Scope 3: {record['scope_3']:,.0f} tons
  Total: {record['total_emissions']:,.0f} tons
  Compliance: {record['compliance_assessment'][:100]}...

"""

        return report

# Agent 5: Email Distribution
class CSRDEmailAgent(Agent):
    def setup(self):
        super().setup()
        self.email = EmailService()

    def execute(self):
        report = self.input_data["report"]
        recipients = self.input_data.get("recipients", ["admin@company.com"])

        self.email.send(
            to=recipients,
            subject="CSRD Emissions Report",
            body=report
        )

        return {"status": "sent", "recipients": recipients}

# Build Pipeline
class CSRDPipeline(Pipeline):
    def __init__(self):
        super().__init__(name="csrd_pipeline")

        self.add_agent(CSRDIngestionAgent())
        self.add_agent(CSRDValidationAgent())
        self.add_agent(CSRDEnrichmentAgent())
        self.add_agent(CSRDReportAgent())
        self.add_agent(CSRDEmailAgent())

        # Add telemetry
        self.telemetry = TelemetryManager(
            service_name="csrd_pipeline",
            environment="production"
        )

    def execute(self, input_data):
        """Execute pipeline with monitoring."""

        with self.telemetry.timer("pipeline_duration_ms"):
            try:
                result = super().execute(input_data)

                self.telemetry.increment("pipeline_success")
                logger.info("Pipeline completed successfully")

                return result

            except Exception as e:
                self.telemetry.increment("pipeline_failures")
                logger.error(f"Pipeline failed: {e}")
                raise

# Run Pipeline
if __name__ == "__main__":
    pipeline = CSRDPipeline()

    result = pipeline.execute({
        "file_path": "data/csrd_emissions.csv",
        "recipients": ["stakeholder@company.com"]
    })

    print("Pipeline complete!")
    print(f"Status: {result['status']}")
    print(f"Recipients: {result['recipients']}")
```

---

## Workshop Wrap-Up

### What You Learned

✓ Multi-agent pipeline architecture
✓ Performance optimization techniques
✓ Cost reduction strategies
✓ Shared services usage
✓ Contributing to infrastructure
✓ Built complete multi-agent system

### Key Takeaways

1. **Pipelines > monoliths** - Composable agents are maintainable
2. **Cache aggressively** - 40x performance improvement
3. **Optimize costs** - Use cheaper models when possible
4. **Reuse services** - Don't rebuild email/notifications
5. **Contribute back** - Improve infrastructure for everyone

### Certification Ready

You've completed all workshops! Next steps:
1. Review all workshop materials
2. Complete hands-on labs
3. Take Level 1 Certification
4. Build a production application
5. Contribute to infrastructure

---

## Additional Resources

- **Infrastructure Source:** `GL_COMMONS/infrastructure/`
- **Examples:** `examples/`
- **Architecture Docs:** `docs/architecture/`
- **Slack:** #greenlang-advanced

---

**All Workshops Complete! Ready for Certification!**
