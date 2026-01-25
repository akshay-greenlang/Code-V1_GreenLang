# GreenLang Infrastructure Catalog

**Complete Reference for All Reusable Infrastructure Components**

Version: 1.0.0
Last Updated: November 9, 2025
Status: Production Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [LLM Infrastructure](#llm-infrastructure)
3. [Agent Framework](#agent-framework)
4. [Data Storage & Caching](#data-storage--caching)
5. [Authentication & Authorization](#authentication--authorization)
6. [API Frameworks](#api-frameworks)
7. [Validation & Security](#validation--security)
8. [Monitoring & Telemetry](#monitoring--telemetry)
9. [Configuration Management](#configuration-management)
10. [Pipeline & Orchestration](#pipeline--orchestration)
11. [Data Processing](#data-processing)
12. [Reporting & Output](#reporting--output)
13. [ERP Connectors](#erp-connectors)
14. [Emissions & Climate Data](#emissions--climate-data)
15. [Machine Learning](#machine-learning)
16. [Testing Infrastructure](#testing-infrastructure)
17. [Deployment & Infrastructure](#deployment--infrastructure)
18. [CLI Framework](#cli-framework)
19. [Pack System](#pack-system)
20. [Migration Patterns](#migration-patterns)

---

## Introduction

### Purpose of This Catalog

This catalog is the **single source of truth** for all reusable infrastructure components in the GreenLang ecosystem. Before writing ANY custom code, developers MUST consult this catalog to find existing infrastructure that solves their problem.

### GreenLang-First Architecture Policy

**POLICY:** Always use GreenLang infrastructure. Never build custom when infrastructure exists.

**Enforcement:**
- Pre-commit hook checks for infrastructure usage
- Code review requires infrastructure justification
- Architecture Decision Record (ADR) required for custom code
- Quarterly audits track Infrastructure Usage Metrics (IUM)

**Target:** 80%+ of application code uses GreenLang infrastructure

### How to Use This Catalog

1. **Search First:** Use Ctrl+F to find keywords related to your need
2. **Read the Component:** Understand purpose, API, and use cases
3. **Check Examples:** See real-world usage patterns
4. **Use, Don't Build:** Import and use the infrastructure
5. **Request Features:** If infrastructure is incomplete, request enhancement

### Component Status Legend

- ✅ **Production Ready:** Battle-tested, stable API, full documentation
- 🚧 **Beta:** Functional but API may change, use with caution
- 📝 **Planned:** Documented but not yet implemented
- ⚠️ **Deprecated:** Legacy code, do not use for new projects

---

## LLM Infrastructure

### ChatSession - LLM Abstraction Layer

**Location:** `greenlang/intelligence/chat_session.py`
**Status:** ✅ Production Ready
**Lines of Code:** 1,200+

#### Purpose

Unified interface for interacting with multiple LLM providers (OpenAI GPT-4, Anthropic Claude-3) with temperature=0 for reproducibility, tool-first architecture for zero hallucination, and complete provenance tracking.

#### Use Cases

- AI-powered narrative generation (CSRD reports, recommendations)
- Intelligent categorization (spend, waste, products)
- Entity resolution and fuzzy matching
- Natural language queries to structured data
- Contextual explanations and insights

#### When to Use vs. When to Build Custom

**Use ChatSession when:**
- You need LLM capabilities (text generation, classification, Q&A)
- You want provider-agnostic code (switch between GPT-4, Claude, etc.)
- You need reproducible results (temperature=0, seed-based)
- You want complete provenance (track every token, cost, latency)

**Build custom when:**
- You need numeric calculations (NEVER use LLM for math)
- You need 100% deterministic outcomes (use database lookups)
- You need regulatory compliance (use zero-hallucination architecture)

#### API Documentation

```python
from greenlang.intelligence import ChatSession

# Initialize session
session = ChatSession(
    provider="openai",  # or "anthropic"
    model="gpt-4",
    temperature=0.0,  # For reproducibility
    seed=42,  # For deterministic results
    max_tokens=2000,
    track_provenance=True
)

# Simple completion
response = session.complete(
    prompt="Explain the GHG Protocol Scope 3 standard in 2 sentences.",
    system="You are a climate compliance expert."
)

# Tool-first approach (recommended for data extraction)
response = session.complete_with_tools(
    prompt="Extract emissions data from this invoice: {invoice_text}",
    tools=[
        {
            "name": "extract_emissions",
            "description": "Extract emissions data",
            "parameters": {
                "type": "object",
                "properties": {
                    "emissions_tco2": {"type": "number"},
                    "scope": {"type": "string"}
                }
            }
        }
    ]
)

# Streaming responses
for chunk in session.stream(prompt="Generate a long report..."):
    print(chunk, end="")

# Access provenance
print(f"Total tokens: {session.total_tokens}")
print(f"Total cost: ${session.total_cost:.4f}")
print(f"Latency: {session.last_latency_ms}ms")
```

#### Configuration

```yaml
# config/llm_config.yaml
llm:
  default_provider: openai
  openai:
    api_key: ${OPENAI_API_KEY}  # From environment
    model: gpt-4
    temperature: 0.0
    max_tokens: 2000
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-3-opus
    temperature: 0.0
    max_tokens: 4000
  provenance:
    track_tokens: true
    track_cost: true
    track_latency: true
    log_prompts: true  # For debugging only
```

#### Code Examples

**Example 1: Narrative Generation (CSRD Reports)**

```python
from greenlang.intelligence import ChatSession

session = ChatSession(provider="openai", model="gpt-4", temperature=0.0)

# Generate materiality narrative
narrative = session.complete(
    prompt=f"""
    Based on these materiality assessment results:
    {materiality_data}

    Generate a 2-paragraph narrative explaining why Climate Change
    is material from both impact and financial perspectives.
    """,
    system="You are an ESRS sustainability reporting expert."
)

print(narrative.content)
# Cost tracking
print(f"Cost: ${narrative.cost:.4f}")
```

**Example 2: Spend Categorization (Scope 3)**

```python
from greenlang.intelligence import ChatSession

session = ChatSession(provider="openai", temperature=0.0)

# Categorize spend using tool-first approach
result = session.complete_with_tools(
    prompt=f"Categorize this purchase: {purchase_description}",
    tools=[
        {
            "name": "categorize_spend",
            "description": "Categorize procurement spend",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["steel", "cement", "electronics", "services"]
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["category", "confidence"]
            }
        }
    ]
)

category = result.tool_calls[0].arguments["category"]
confidence = result.tool_calls[0].arguments["confidence"]
```

**Example 3: Entity Resolution**

```python
from greenlang.intelligence import ChatSession

session = ChatSession(provider="anthropic", model="claude-3-opus")

# Fuzzy match supplier names
result = session.complete(
    prompt=f"""
    Are these two companies the same?
    Company 1: {supplier_name_1}
    Company 2: {supplier_name_2}

    Consider: spelling variations, abbreviations, legal suffixes.
    """,
    system="You are an expert at entity resolution."
)

# Parse yes/no response
is_match = "yes" in result.content.lower()
```

#### Migration Guide

**From Custom OpenAI Code:**

```python
# BEFORE (Custom Code)
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.0
)

content = response.choices[0].message.content

# AFTER (GreenLang Infrastructure)
from greenlang.intelligence import ChatSession

session = ChatSession(provider="openai", model="gpt-4", temperature=0.0)
response = session.complete(
    prompt="Hello!",
    system="You are a helper."
)

content = response.content
# Bonus: automatic provenance tracking!
```

#### Performance Characteristics

- **Latency:** 500-5000ms (depends on provider, prompt size)
- **Throughput:** 100+ requests/minute (with rate limiting)
- **Cost:** $0.01-0.10 per request (GPT-4), $0.05-0.20 (Claude-3)
- **Token Limits:** 4K-128K depending on model

#### Best Practices

1. **Always use temperature=0** for reproducible results
2. **Always use tools** for structured data extraction (no string parsing!)
3. **Never use LLM for calculations** (use database lookups + Python arithmetic)
4. **Always track provenance** (enable in production)
5. **Always set max_tokens** (prevent runaway costs)
6. **Always use system prompts** (better results than pure user prompts)
7. **Cache responses** when possible (identical prompts = reuse results)

#### Common Anti-Patterns

❌ **DON'T: Use LLM for calculations**
```python
# WRONG
response = session.complete("What is 2.5 tons CO2 * $50/ton?")
# LLM might hallucinate: "125" or "$125" or "approximately $125"
```

✅ **DO: Use Python arithmetic**
```python
# CORRECT
result = 2.5 * 50  # 125.0 (deterministic)
```

❌ **DON'T: Parse unstructured text**
```python
# WRONG
response = session.complete("Extract the emissions value from: {text}")
# Then regex parse the response (brittle!)
```

✅ **DO: Use tool calling**
```python
# CORRECT
response = session.complete_with_tools(
    prompt=f"Extract emissions from: {text}",
    tools=[extract_emissions_tool]
)
value = response.tool_calls[0].arguments["emissions_tco2"]
```

#### Related Components

- **RAGManager:** For retrieval-augmented generation
- **EmbeddingService:** For semantic search
- **ProvenanceTracker:** For audit trails
- **CacheManager:** For response caching

---

### RAGManager - Retrieval-Augmented Generation

**Location:** `greenlang/intelligence/rag_manager.py`
**Status:** ✅ Production Ready
**Lines of Code:** 800+

#### Purpose

Implement retrieval-augmented generation (RAG) to ground LLM responses in factual, domain-specific knowledge. Reduces hallucination by providing relevant context from vector database before generating responses.

#### Use Cases

- Technical documentation Q&A
- Regulatory compliance lookups (ESRS data points, GHG Protocol standards)
- Best practice recommendations (from knowledge base)
- Citation-backed reports

#### API Documentation

```python
from greenlang.intelligence import RAGManager

# Initialize with Weaviate vector DB
rag = RAGManager(
    vector_db="weaviate",
    collection="climate_knowledge",
    embedding_model="text-embedding-ada-002"
)

# Index documents
rag.index_documents([
    {"id": "ghg-scope3", "text": "GHG Protocol Scope 3 Standard...", "metadata": {"source": "GHG Protocol"}},
    {"id": "esrs-e1", "text": "ESRS E1 Climate Change standard...", "metadata": {"source": "EFRAG"}}
])

# Query with RAG
response = rag.query(
    question="What are the 15 Scope 3 categories?",
    top_k=3,  # Retrieve top 3 relevant documents
    include_citations=True
)

print(response.answer)
print(f"Sources: {response.citations}")
```

#### Best Practices

1. **Always include citations** for audit trails
2. **Chunk documents** to 500-1000 words for better retrieval
3. **Use metadata filtering** to scope searches (e.g., only ESRS standards)
4. **Re-rank results** for better relevance
5. **Cache embeddings** to avoid recomputing

---

### EmbeddingService - Semantic Embeddings

**Location:** `greenlang/intelligence/embedding_service.py`
**Status:** ✅ Production Ready
**Lines of Code:** 400+

#### Purpose

Generate semantic embeddings for text, enabling similarity search, clustering, and classification. Used by RAG, entity resolution, and duplicate detection.

#### API Documentation

```python
from greenlang.intelligence import EmbeddingService

embedder = EmbeddingService(
    provider="openai",
    model="text-embedding-ada-002"
)

# Single text
embedding = embedder.embed("Scope 3 emissions from purchased goods")
# Returns: numpy array of 1536 dimensions

# Batch embedding (more efficient)
embeddings = embedder.embed_batch([
    "Scope 1 emissions",
    "Scope 2 emissions",
    "Scope 3 emissions"
])

# Similarity search
query_emb = embedder.embed("upstream value chain emissions")
similarities = embedder.cosine_similarity(query_emb, embeddings)
# Returns: [0.65, 0.72, 0.89] - Scope 3 is most similar!
```

#### Performance

- **Latency:** 50-200ms per text
- **Batch size:** Up to 2,048 texts per request
- **Cost:** $0.0001 per 1K tokens (OpenAI)

---

## Agent Framework

### Agent Base Class

**Location:** `greenlang/sdk/base.py`
**Status:** ✅ Production Ready
**Lines of Code:** 600+

#### Purpose

Base class for all GreenLang agents. Provides standardized lifecycle, error handling, provenance tracking, and telemetry. All application agents should inherit from this class.

#### Use Cases

- Build calculation agents (emissions, metrics, benchmarks)
- Build data processing agents (intake, validation, aggregation)
- Build intelligence agents (materiality, recommendations, forecasting)
- Build reporting agents (PDF, XBRL, dashboards)

#### When to Use vs. When to Build Custom

**Use Agent Base when:**
- Building ANY agent in GreenLang ecosystem
- You need standardized error handling
- You need provenance tracking
- You need telemetry and monitoring
- You want automatic retry logic
- You want input/output validation

**Build custom when:**
- Never. All agents should use this base class.

#### API Documentation

```python
from greenlang.sdk.base import Agent
from pydantic import BaseModel
from typing import Dict, Any

class MyAgentInput(BaseModel):
    data: str
    threshold: float = 0.5

class MyAgentOutput(BaseModel):
    result: float
    confidence: float

class MyCustomAgent(Agent):
    """
    Custom agent that inherits from Agent base.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="my-custom-agent",
            version="1.0.0",
            description="Does amazing things",
            config=config
        )

    def execute(self, input_data: MyAgentInput) -> MyAgentOutput:
        """
        Main execution logic. Must be implemented by all agents.
        """
        # Your custom logic here
        result = len(input_data.data) * input_data.threshold
        confidence = 0.95

        return MyAgentOutput(result=result, confidence=confidence)

    def validate_input(self, input_data: Any) -> bool:
        """
        Optional: custom input validation
        """
        return isinstance(input_data, MyAgentInput)

    def validate_output(self, output_data: Any) -> bool:
        """
        Optional: custom output validation
        """
        return isinstance(output_data, MyAgentOutput)

# Usage
agent = MyCustomAgent()
result = agent.run(MyAgentInput(data="test", threshold=0.8))
print(result.result)

# Provenance automatically tracked
print(f"Agent: {agent.name} v{agent.version}")
print(f"Execution time: {agent.last_execution_time_ms}ms")
print(f"Status: {agent.last_status}")
```

#### Built-in Features

**1. Automatic Provenance Tracking**
```python
# After execution
print(agent.execution_history)
# [
#   {
#     "timestamp": "2025-11-09T10:30:00Z",
#     "input_hash": "abc123...",
#     "output_hash": "def456...",
#     "duration_ms": 150,
#     "status": "success"
#   }
# ]
```

**2. Automatic Error Handling**
```python
# Errors are caught and logged
try:
    result = agent.run(bad_input)
except AgentExecutionError as e:
    print(f"Agent failed: {e}")
    print(f"Error details: {e.details}")
```

**3. Automatic Retry Logic**
```python
agent = MyAgent(config={
    "retry": {
        "max_attempts": 3,
        "backoff_factor": 2,
        "retry_on": ["timeout", "rate_limit"]
    }
})

# Will retry up to 3 times on transient failures
result = agent.run(input_data)
```

**4. Automatic Telemetry**
```python
# Metrics automatically sent to monitoring system
# - agent.{name}.execution_count
# - agent.{name}.execution_time_ms
# - agent.{name}.error_count
# - agent.{name}.success_rate
```

**5. Automatic Input/Output Validation**
```python
# Pydantic models automatically validated
class StrictInput(BaseModel):
    value: int  # Must be int, not string

agent.run(StrictInput(value="123"))  # ValidationError!
agent.run(StrictInput(value=123))    # Success!
```

#### Code Examples

**Example 1: Emissions Calculator Agent**

```python
from greenlang.sdk.base import Agent
from pydantic import BaseModel
from typing import List

class EmissionInput(BaseModel):
    fuel_type: str
    consumption: float
    unit: str

class EmissionOutput(BaseModel):
    emissions_tco2: float
    emission_factor: float
    source: str

class EmissionsCalculatorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="emissions-calculator",
            version="2.0.0",
            description="Calculate GHG emissions from fuel consumption"
        )
        # Load emission factors from database
        self.emission_factors = self._load_factors()

    def execute(self, input_data: EmissionInput) -> EmissionOutput:
        # Deterministic calculation (no LLM!)
        factor = self.emission_factors.get(input_data.fuel_type)
        if not factor:
            raise ValueError(f"Unknown fuel type: {input_data.fuel_type}")

        # Convert units if needed
        consumption_kwh = self._convert_to_kwh(
            input_data.consumption,
            input_data.unit
        )

        # Calculate emissions
        emissions = consumption_kwh * factor.kgco2_per_kwh / 1000

        return EmissionOutput(
            emissions_tco2=emissions,
            emission_factor=factor.kgco2_per_kwh,
            source=factor.source
        )

    def _load_factors(self):
        # Load from database (cached)
        pass

    def _convert_to_kwh(self, value, unit):
        # Unit conversion logic
        pass

# Usage
agent = EmissionsCalculatorAgent()
result = agent.run(EmissionInput(
    fuel_type="natural_gas",
    consumption=1000,
    unit="therms"
))
print(f"Emissions: {result.emissions_tco2:.2f} tCO2")
```

**Example 2: Data Validation Agent**

```python
from greenlang.sdk.base import Agent
from pydantic import BaseModel
from typing import List, Dict

class ValidationRule(BaseModel):
    field: str
    rule_type: str  # "range", "enum", "regex", "custom"
    parameters: Dict

class ValidationInput(BaseModel):
    data: List[Dict]
    rules: List[ValidationRule]

class ValidationOutput(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class DataValidationAgent(Agent):
    def execute(self, input_data: ValidationInput) -> ValidationOutput:
        errors = []
        warnings = []

        for record in input_data.data:
            for rule in input_data.rules:
                result = self._apply_rule(record, rule)
                if result.level == "error":
                    errors.append(result.message)
                elif result.level == "warning":
                    warnings.append(result.message)

        return ValidationOutput(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

#### Migration Guide

**From Custom Agent Code:**

```python
# BEFORE (Custom Code)
class MyAgent:
    def __init__(self):
        self.name = "my-agent"

    def process(self, data):
        try:
            result = self._do_work(data)
            self._log(f"Success: {result}")
            return result
        except Exception as e:
            self._log(f"Error: {e}")
            raise

# AFTER (GreenLang Infrastructure)
from greenlang.sdk.base import Agent

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            name="my-agent",
            version="1.0.0"
        )

    def execute(self, input_data):
        # Error handling, logging, provenance all automatic!
        return self._do_work(input_data)

# Bonus features:
# - Automatic retry on transient failures
# - Provenance tracking
# - Telemetry to monitoring system
# - Input/output validation
# - Standardized error messages
```

#### Performance Characteristics

- **Overhead:** <5ms per agent execution (provenance tracking)
- **Memory:** ~50KB per agent instance
- **Scalability:** 10,000+ agents per process

#### Best Practices

1. **Always inherit from Agent base** - Never create standalone agent classes
2. **Use Pydantic models** for input/output - Automatic validation
3. **Keep execute() pure** - No side effects, deterministic when possible
4. **Use config for dependencies** - DB connections, API keys, etc.
5. **Write unit tests** - Test execute() with various inputs
6. **Version your agents** - Increment version on breaking changes

---

### AsyncAgent - Asynchronous Agent Base

**Location:** `greenlang/agents/async_agent_base.py`
**Status:** ✅ Production Ready
**Lines of Code:** 400+

#### Purpose

Async/await version of Agent base class for I/O-bound operations (API calls, database queries, file I/O). Use when you need to execute multiple operations concurrently.

#### API Documentation

```python
from greenlang.agents.async_agent_base import AsyncAgent
from pydantic import BaseModel
import asyncio

class MyAsyncAgent(AsyncAgent):
    async def execute(self, input_data):
        # Async operations
        result1 = await self._fetch_from_api()
        result2 = await self._query_database()

        # Concurrent operations
        results = await asyncio.gather(
            self._process_chunk_1(),
            self._process_chunk_2(),
            self._process_chunk_3()
        )

        return {"results": results}

# Usage
agent = MyAsyncAgent()
result = await agent.run(input_data)

# Or run synchronously
result = asyncio.run(agent.run(input_data))
```

---

### AgentSpec v2 - Declarative Agent Definition

**Location:** `greenlang/agents/agentspec_v2_base.py`
**Status:** ✅ Production Ready
**Lines of Code:** 500+

#### Purpose

Define agents declaratively using YAML specifications instead of writing Python code. Ideal for simple agents that follow standard patterns.

#### Example YAML Specification

```yaml
# specs/my_agent_spec.yaml
agent:
  name: emissions-calculator
  version: 2.0.0
  description: Calculate emissions from fuel consumption

inputs:
  - name: fuel_type
    type: string
    required: true
    enum: [natural_gas, electricity, diesel, gasoline]

  - name: consumption
    type: number
    required: true
    minimum: 0

  - name: unit
    type: string
    required: true
    enum: [kwh, therms, gallons, liters]

outputs:
  - name: emissions_tco2
    type: number

  - name: emission_factor
    type: number

  - name: source
    type: string

calculations:
  - step: lookup_factor
    operation: database_query
    query: "SELECT * FROM emission_factors WHERE fuel_type = {{fuel_type}}"
    output: factor_record

  - step: convert_units
    operation: python_function
    function: convert_to_kwh
    args: [consumption, unit]
    output: consumption_kwh

  - step: calculate_emissions
    operation: multiply
    args: [consumption_kwh, factor_record.kgco2_per_kwh]
    output: emissions_kg

  - step: convert_to_tons
    operation: divide
    args: [emissions_kg, 1000]
    output: emissions_tco2

validations:
  - field: consumption
    rule: positive
    message: "Consumption must be positive"

  - field: emissions_tco2
    rule: range
    min: 0
    max: 1000000
    message: "Emissions out of reasonable range"
```

#### Usage

```python
from greenlang.agents.agentspec_v2_base import load_agent_from_spec

# Load agent from YAML
agent = load_agent_from_spec("specs/my_agent_spec.yaml")

# Use like any other agent
result = agent.run({
    "fuel_type": "natural_gas",
    "consumption": 1000,
    "unit": "therms"
})

print(result.emissions_tco2)
```

---

## Data Storage & Caching

### CacheManager - Distributed Caching

**Location:** `greenlang/cache/cache_manager.py`
**Status:** ✅ Production Ready
**Lines of Code:** 600+

#### Purpose

Distributed caching layer using Redis for high-performance data storage and retrieval. Reduces database load, improves response times, and enables rate limiting and session management.

#### Use Cases

- Cache LLM responses (identical prompts = reuse results)
- Cache database query results
- Cache emission factors (rarely change)
- Session storage (user sessions, auth tokens)
- Rate limiting (track API calls per user)
- Distributed locks (prevent concurrent operations)

#### API Documentation

```python
from greenlang.cache import CacheManager

# Initialize cache
cache = CacheManager(
    host="localhost",
    port=6379,
    db=0,
    password=None,
    default_ttl=3600  # 1 hour
)

# Basic operations
cache.set("key", "value", ttl=300)  # 5 minutes
value = cache.get("key")
cache.delete("key")

# JSON serialization (automatic)
cache.set("user:123", {"name": "John", "role": "admin"})
user = cache.get("user:123")  # Returns dict

# Batch operations
cache.mset({
    "key1": "value1",
    "key2": "value2",
    "key3": "value3"
})
values = cache.mget(["key1", "key2", "key3"])

# Hash operations (for nested data)
cache.hset("user:123", "name", "John")
cache.hset("user:123", "email", "john@example.com")
name = cache.hget("user:123", "name")

# List operations (for queues)
cache.lpush("tasks", "task1")
cache.lpush("tasks", "task2")
task = cache.rpop("tasks")  # FIFO queue

# Set operations (for unique items)
cache.sadd("suppliers", "supplier1")
cache.sadd("suppliers", "supplier2")
is_member = cache.sismember("suppliers", "supplier1")

# Sorted sets (for leaderboards)
cache.zadd("emissions_rank", {"supplier1": 1000, "supplier2": 500})
top_emitters = cache.zrange("emissions_rank", 0, 9, desc=True)

# Atomic operations
cache.incr("api_calls:user:123")  # Thread-safe counter
cache.incrby("total_emissions", 100)
```

#### Code Examples

**Example 1: Cache LLM Responses**

```python
from greenlang.cache import CacheManager
from greenlang.intelligence import ChatSession
import hashlib

cache = CacheManager()
llm = ChatSession(provider="openai")

def get_llm_response(prompt: str, system: str = None):
    # Create cache key from prompt
    cache_key = f"llm:{hashlib.sha256(prompt.encode()).hexdigest()}"

    # Check cache first
    cached = cache.get(cache_key)
    if cached:
        return cached["response"]

    # Cache miss - call LLM
    response = llm.complete(prompt=prompt, system=system)

    # Cache for 1 hour
    cache.set(cache_key, {
        "response": response.content,
        "cost": response.cost,
        "tokens": response.total_tokens
    }, ttl=3600)

    return response.content

# First call: LLM API call
result1 = get_llm_response("Explain Scope 3")

# Second call: Instant (from cache)
result2 = get_llm_response("Explain Scope 3")
```

**Example 2: Rate Limiting**

```python
from greenlang.cache import CacheManager
from datetime import datetime

cache = CacheManager()

def check_rate_limit(user_id: str, max_calls: int = 100, window_seconds: int = 3600):
    """
    Check if user has exceeded rate limit.
    Returns (allowed: bool, remaining: int, reset_time: datetime)
    """
    key = f"rate_limit:{user_id}"

    # Increment counter
    calls = cache.incr(key)

    # Set expiry on first call
    if calls == 1:
        cache.expire(key, window_seconds)

    # Check limit
    if calls > max_calls:
        ttl = cache.ttl(key)
        reset_time = datetime.now() + timedelta(seconds=ttl)
        return False, 0, reset_time

    remaining = max_calls - calls
    reset_time = datetime.now() + timedelta(seconds=cache.ttl(key))
    return True, remaining, reset_time

# Usage
allowed, remaining, reset = check_rate_limit("user123")
if not allowed:
    raise RateLimitExceeded(f"Try again at {reset}")
```

**Example 3: Distributed Lock**

```python
from greenlang.cache import CacheManager
import time

cache = CacheManager()

def process_supplier_data(supplier_id: str):
    """
    Process supplier data with distributed lock to prevent concurrent processing.
    """
    lock_key = f"lock:supplier:{supplier_id}"
    lock_value = str(time.time())

    # Try to acquire lock (expires after 60 seconds)
    acquired = cache.set(lock_key, lock_value, ttl=60, nx=True)

    if not acquired:
        raise LockError("Another process is already processing this supplier")

    try:
        # Do work
        result = _process_supplier(supplier_id)
        return result
    finally:
        # Release lock
        cache.delete(lock_key)
```

#### Performance Characteristics

- **Latency:** <1ms for simple operations (local Redis)
- **Throughput:** 100,000+ ops/sec (single instance)
- **Memory:** ~1KB per cached item (depends on value size)
- **Persistence:** Optional (RDB snapshots or AOF)

#### Best Practices

1. **Always set TTL** - Prevent unbounded memory growth
2. **Use namespaced keys** - Prevent key collisions (e.g., `emissions:supplier:123`)
3. **Serialize complex objects** - Use JSON, not pickle (cross-language compatibility)
4. **Use atomic operations** - For counters and locks (INCR, SET NX)
5. **Monitor memory usage** - Redis is in-memory, can run out of RAM
6. **Use connection pooling** - Reuse connections (built-in to CacheManager)

---

### DatabaseManager - Database Abstraction

**Location:** `greenlang/db/database_manager.py`
**Status:** ✅ Production Ready
**Lines of Code:** 800+

#### Purpose

Unified interface for database operations (PostgreSQL, MySQL, SQLite) with connection pooling, automatic retry, query builder, and migration support.

#### API Documentation

```python
from greenlang.db import DatabaseManager

# Initialize database connection
db = DatabaseManager(
    provider="postgresql",
    host="localhost",
    port=5432,
    database="greenlang",
    user="greenlang",
    password="${DB_PASSWORD}",  # From environment
    pool_size=20,
    max_overflow=10
)

# Execute query
results = db.query("SELECT * FROM emission_factors WHERE fuel_type = %s", ["natural_gas"])

# Execute with dict result
results = db.query_dict("SELECT fuel_type, kgco2_per_kwh FROM emission_factors")
# [{"fuel_type": "natural_gas", "kgco2_per_kwh": 0.18}, ...]

# Insert
db.execute(
    "INSERT INTO emissions (supplier_id, tco2, date) VALUES (%s, %s, %s)",
    ["SUP-001", 100.5, "2025-01-01"]
)

# Bulk insert (faster)
db.bulk_insert(
    "emissions",
    ["supplier_id", "tco2", "date"],
    [
        ["SUP-001", 100.5, "2025-01-01"],
        ["SUP-002", 200.3, "2025-01-01"],
        ["SUP-003", 150.7, "2025-01-01"]
    ]
)

# Transaction
with db.transaction():
    db.execute("UPDATE suppliers SET status = 'active' WHERE id = %s", ["SUP-001"])
    db.execute("INSERT INTO audit_log (action, supplier_id) VALUES (%s, %s)", ["activate", "SUP-001"])
    # Commits automatically if no exception
    # Rolls back on exception
```

---

## Authentication & Authorization

### AuthManager - Authentication & Authorization

**Location:** `greenlang/auth/auth_manager.py`
**Status:** ✅ Production Ready
**Lines of Code:** 700+

#### Purpose

Centralized authentication and authorization with JWT tokens, RBAC (Role-Based Access Control), API key management, and audit logging.

#### API Documentation

```python
from greenlang.auth import AuthManager

auth = AuthManager(
    secret_key="${JWT_SECRET}",
    algorithm="HS256",
    access_token_expiry=3600,  # 1 hour
    refresh_token_expiry=604800  # 7 days
)

# Create user
user = auth.create_user(
    username="john@example.com",
    password="secure_password",
    roles=["user", "analyst"]
)

# Authenticate
token_pair = auth.authenticate(
    username="john@example.com",
    password="secure_password"
)
access_token = token_pair.access_token
refresh_token = token_pair.refresh_token

# Verify token
payload = auth.verify_token(access_token)
print(f"User ID: {payload['user_id']}")
print(f"Roles: {payload['roles']}")

# Check permission
has_permission = auth.check_permission(
    user_id=payload['user_id'],
    resource="emissions",
    action="write"
)

# Refresh token
new_token_pair = auth.refresh_token(refresh_token)
```

---

## API Frameworks

### FastAPI Integration

**Location:** `greenlang/api/fastapi_integration.py`
**Status:** ✅ Production Ready
**Lines of Code:** 500+

#### Purpose

Pre-configured FastAPI application with authentication middleware, CORS, rate limiting, error handling, and API versioning.

#### API Documentation

```python
from greenlang.api import create_app
from fastapi import Depends
from greenlang.auth import get_current_user

# Create FastAPI app with GreenLang defaults
app = create_app(
    title="My Climate API",
    version="1.0.0",
    auth_enabled=True,
    rate_limit_enabled=True,
    cors_enabled=True
)

# Define endpoints
@app.get("/api/v1/emissions/{supplier_id}")
async def get_emissions(
    supplier_id: str,
    user = Depends(get_current_user)  # Automatic authentication
):
    # User is authenticated, check permission
    if "analyst" not in user.roles:
        raise PermissionDenied("Requires analyst role")

    # Fetch emissions
    emissions = db.query(
        "SELECT * FROM emissions WHERE supplier_id = %s",
        [supplier_id]
    )

    return {"supplier_id": supplier_id, "emissions": emissions}

# Automatic rate limiting (100 requests per minute per user)
# Automatic error handling (returns JSON errors)
# Automatic CORS (configurable origins)
# Automatic OpenAPI docs at /docs
```

---

## Validation & Security

### ValidationFramework - Data Validation

**Location:** `greenlang/validation/validation_framework.py`
**Status:** ✅ Production Ready
**Lines of Code:** 900+

#### Purpose

Declarative data validation with 50+ built-in rules, custom rules, and detailed error messages. Validates input data before processing.

#### API Documentation

```python
from greenlang.validation import ValidationFramework, ValidationRule

# Define validation rules
rules = [
    ValidationRule(
        field="consumption",
        rule_type="positive",
        message="Consumption must be positive"
    ),
    ValidationRule(
        field="fuel_type",
        rule_type="enum",
        parameters={"allowed": ["natural_gas", "electricity", "diesel"]},
        message="Invalid fuel type"
    ),
    ValidationRule(
        field="date",
        rule_type="date_range",
        parameters={"min": "2020-01-01", "max": "2030-12-31"},
        message="Date out of range"
    )
]

validator = ValidationFramework(rules=rules)

# Validate data
data = {
    "consumption": 1000,
    "fuel_type": "natural_gas",
    "date": "2025-01-01"
}

result = validator.validate(data)

if result.is_valid:
    print("Data is valid!")
else:
    for error in result.errors:
        print(f"Error: {error.field} - {error.message}")
```

#### Built-in Validation Rules

**Numeric:**
- `positive` - Value > 0
- `non_negative` - Value >= 0
- `range` - Value between min and max
- `multiple_of` - Value is multiple of X

**String:**
- `enum` - Value in allowed list
- `regex` - Matches regex pattern
- `min_length` - String length >= min
- `max_length` - String length <= max
- `email` - Valid email format
- `url` - Valid URL format

**Date/Time:**
- `date_format` - Matches date format (YYYY-MM-DD)
- `date_range` - Date between min and max
- `future_date` - Date is in future
- `past_date` - Date is in past

**Business Logic:**
- `unique` - Value unique in dataset
- `reference_exists` - Foreign key exists
- `sum_equals` - Sum of fields equals X
- `percentage` - Value between 0 and 100

---

## Monitoring & Telemetry

### TelemetryManager - Metrics & Tracing

**Location:** `greenlang/monitoring/telemetry_manager.py`
**Status:** ✅ Production Ready
**Lines of Code:** 600+

#### Purpose

Unified telemetry for metrics (Prometheus), tracing (OpenTelemetry), and logging (structured logs). Automatic instrumentation of agents, APIs, and database queries.

#### API Documentation

```python
from greenlang.monitoring import TelemetryManager

telemetry = TelemetryManager(
    service_name="csrd-reporting",
    environment="production",
    prometheus_port=9090,
    jaeger_endpoint="http://localhost:14268"
)

# Record metric
telemetry.record_counter("emissions_calculated", labels={"supplier": "SUP-001"})
telemetry.record_histogram("calculation_time_ms", 150, labels={"category": "scope3"})
telemetry.record_gauge("active_suppliers", 1234)

# Trace span
with telemetry.trace_span("calculate_emissions") as span:
    span.set_attribute("supplier_id", "SUP-001")
    result = calculate_emissions()
    span.set_attribute("result_tco2", result)

# Structured logging
telemetry.log_info("Emissions calculated", extra={
    "supplier_id": "SUP-001",
    "emissions_tco2": 100.5,
    "category": "scope3_cat1"
})
```

---

## Configuration Management

### ConfigManager - Configuration Management

**Location:** `greenlang/config/config_manager.py`
**Status:** ✅ Production Ready
**Lines of Code:** 500+

#### Purpose

Centralized configuration management with environment-specific configs, secret management (no hardcoded secrets), and validation.

#### API Documentation

```python
from greenlang.config import ConfigManager

config = ConfigManager(
    config_file="config/app_config.yaml",
    environment="production",  # or "development", "staging"
    secrets_provider="env"  # or "vault", "aws_secrets"
)

# Get config values
db_host = config.get("database.host")
api_key = config.get_secret("openai.api_key")  # Never logs value

# Get with default
max_retries = config.get("retry.max_attempts", default=3)

# Get nested config
llm_config = config.get_section("llm")
# Returns: {"provider": "openai", "model": "gpt-4", ...}
```

#### Example Configuration File

```yaml
# config/app_config.yaml
environment: ${ENV}  # From environment variable

database:
  host: ${DB_HOST:localhost}  # Default to localhost
  port: ${DB_PORT:5432}
  name: greenlang_${ENV}
  user: greenlang
  password: ${DB_PASSWORD}  # Secret from environment

llm:
  provider: openai
  model: gpt-4
  temperature: 0.0
  max_tokens: 2000
  api_key: ${OPENAI_API_KEY}  # Secret

cache:
  provider: redis
  host: ${REDIS_HOST:localhost}
  port: ${REDIS_PORT:6379}
  ttl: 3600

monitoring:
  enabled: true
  prometheus_port: 9090
  jaeger_endpoint: ${JAEGER_ENDPOINT}
```

---

## Pipeline & Orchestration

### PipelineOrchestrator - Multi-Agent Pipelines

**Location:** `greenlang/pipeline/orchestrator.py`
**Status:** ✅ Production Ready
**Lines of Code:** 700+

#### Purpose

Orchestrate multi-agent pipelines with dependency management, parallel execution, error handling, and checkpointing.

#### API Documentation

```python
from greenlang.pipeline import PipelineOrchestrator, PipelineStep

# Define pipeline
pipeline = PipelineOrchestrator(
    name="csrd-reporting-pipeline",
    version="1.0.0"
)

# Add steps
pipeline.add_step(
    PipelineStep(
        name="intake",
        agent=IntakeAgent(),
        dependencies=[]  # No dependencies
    )
)

pipeline.add_step(
    PipelineStep(
        name="materiality",
        agent=MaterialityAgent(),
        dependencies=["intake"]  # Runs after intake
    )
)

pipeline.add_step(
    PipelineStep(
        name="calculate",
        agent=CalculatorAgent(),
        dependencies=["intake"]  # Runs after intake (parallel with materiality)
    )
)

pipeline.add_step(
    PipelineStep(
        name="report",
        agent=ReportingAgent(),
        dependencies=["materiality", "calculate"]  # Runs after both complete
    )
)

# Execute pipeline
result = pipeline.run(input_data={
    "esg_data_file": "data.csv",
    "company_profile": "company.json"
})

# Results keyed by step name
intake_result = result["intake"]
materiality_result = result["materiality"]
final_report = result["report"]
```

---

## Data Processing

### DataTransformer - ETL Transformations

**Location:** `greenlang/data/transformer.py`
**Status:** ✅ Production Ready
**Lines of Code:** 600+

#### Purpose

Common data transformation operations (clean, normalize, aggregate, pivot) with pandas integration.

#### API Documentation

```python
from greenlang.data import DataTransformer
import pandas as pd

transformer = DataTransformer()

# Load data
df = pd.read_csv("emissions_data.csv")

# Clean data
df_clean = transformer.clean(
    df,
    operations=[
        {"type": "drop_nulls", "columns": ["emissions_tco2"]},
        {"type": "remove_duplicates", "subset": ["supplier_id", "date"]},
        {"type": "fill_nulls", "column": "data_quality", "value": "medium"},
        {"type": "clip", "column": "emissions_tco2", "min": 0, "max": 1000000}
    ]
)

# Normalize data
df_normalized = transformer.normalize(
    df_clean,
    operations=[
        {"type": "lowercase", "columns": ["supplier_name"]},
        {"type": "trim_whitespace", "columns": ["supplier_name"]},
        {"type": "standardize_date", "column": "date", "format": "%Y-%m-%d"}
    ]
)

# Aggregate data
df_agg = transformer.aggregate(
    df_normalized,
    group_by=["supplier_id"],
    aggregations={
        "emissions_tco2": "sum",
        "data_quality": "first",
        "date": "max"
    }
)
```

---

## Reporting & Output

### ReportGenerator - PDF/Excel/XBRL Reports

**Location:** `greenlang/reporting/report_generator.py`
**Status:** ✅ Production Ready
**Lines of Code:** 1,000+

#### Purpose

Generate professional reports in PDF, Excel, XBRL formats with templates, charts, and branding.

#### API Documentation

```python
from greenlang.reporting import ReportGenerator

generator = ReportGenerator(
    template_dir="templates/reports",
    output_dir="output/reports"
)

# Generate PDF report
report = generator.generate_pdf(
    template="csrd_report_template.html",
    data={
        "company_name": "Acme Manufacturing",
        "reporting_period": "2024",
        "emissions_data": emissions_df,
        "charts": [chart1, chart2, chart3]
    },
    output_file="csrd_report_2024.pdf",
    branding={
        "logo": "company_logo.png",
        "primary_color": "#0066CC"
    }
)

# Generate Excel report
excel = generator.generate_excel(
    sheets={
        "Summary": summary_df,
        "Scope 1": scope1_df,
        "Scope 2": scope2_df,
        "Scope 3": scope3_df
    },
    output_file="emissions_report_2024.xlsx",
    formatting={
        "header_color": "#0066CC",
        "freeze_panes": True,
        "auto_filter": True
    }
)

# Generate XBRL report (ESEF-compliant)
xbrl = generator.generate_xbrl(
    taxonomy="esrs-2024",
    data_points={
        "E1-1": 12500,  # Scope 1 emissions
        "E1-2": 8300,   # Scope 2 emissions
        "E1-6": 185000  # Total energy
    },
    output_file="sustainability_statement.xhtml"
)
```

---

## ERP Connectors

### SAP Connector

**Location:** `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/connectors/sap/`
**Status:** ✅ Production Ready
**Lines of Code:** 2,000+

#### Purpose

Native integration with SAP S/4HANA for procurement, logistics, and emissions data extraction via OData API.

#### API Documentation

```python
from vcci_scope3.connectors.sap import SAPConnector

sap = SAPConnector(
    base_url="https://sap-system.example.com:50000",
    client="100",
    username="${SAP_USERNAME}",
    password="${SAP_PASSWORD}",
    use_oauth=True
)

# Extract procurement data
procurement = sap.extract_procurement_data(
    fiscal_year=2024,
    filters={
        "company_code": "1000",
        "purchasing_org": "US01"
    }
)

# Returns: DataFrame with columns
# - po_number, item, material, supplier, quantity, unit, amount, currency, date

# Extract logistics data
logistics = sap.extract_logistics_data(
    shipment_date_from="2024-01-01",
    shipment_date_to="2024-12-31"
)

# Extract supplier master data
suppliers = sap.extract_supplier_master()
```

---

### Oracle ERP Connector

**Location:** `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/connectors/oracle/`
**Status:** ✅ Production Ready
**Lines of Code:** 1,800+

#### Purpose

Native integration with Oracle ERP Cloud via REST API.

---

### Workday Connector

**Location:** `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/connectors/workday/`
**Status:** ✅ Production Ready
**Lines of Code:** 1,200+

#### Purpose

Native integration with Workday for business travel and employee commuting data.

---

## Emissions & Climate Data

### EmissionFactorDatabase

**Location:** `greenlang/data/emission_factors.py`
**Status:** ✅ Production Ready
**Lines of Code:** 500+ (plus 100,000+ factor records)

#### Purpose

Centralized database of emission factors from authoritative sources (DEFRA, EPA, IEA, IPCC, Ecoinvent) with version control and lineage tracking.

#### API Documentation

```python
from greenlang.data import EmissionFactorDatabase

efdb = EmissionFactorDatabase(
    sources=["defra_2024", "epa_2024", "ghg_protocol"]
)

# Get emission factor
factor = efdb.get_factor(
    activity="electricity_consumption",
    region="UK",
    year=2024
)

print(f"Factor: {factor.kgco2e_per_kwh}")
print(f"Source: {factor.source}")
print(f"Vintage: {factor.vintage}")
print(f"Uncertainty: ±{factor.uncertainty_pct}%")

# Search factors
factors = efdb.search(
    activity_type="transport",
    mode="road_freight",
    region="EU"
)

# Get latest version
latest = efdb.get_latest("natural_gas_combustion", region="US")
```

---

## Machine Learning

### ForecastAgent - Time Series Forecasting

**Location:** `greenlang/agents/forecast_agent_sarima.py`
**Status:** ✅ Production Ready
**Lines of Code:** 600+

#### Purpose

SARIMA-based time series forecasting for emissions trends, energy consumption, and business metrics.

#### API Documentation

```python
from greenlang.agents import ForecastAgentSARIMA
import pandas as pd

agent = ForecastAgentSARIMA(
    auto_tune=True,  # Automatically find best parameters
    seasonality=12   # Monthly data
)

# Historical data
historical = pd.DataFrame({
    "date": pd.date_range("2020-01-01", periods=48, freq="M"),
    "emissions_tco2": [100, 105, 98, ...]  # 48 months
})

# Forecast next 12 months
forecast = agent.run({
    "historical_data": historical,
    "forecast_periods": 12,
    "confidence_level": 0.95
})

print(forecast.predictions)  # [110, 115, 108, ...]
print(forecast.lower_bound)  # [105, 110, 103, ...]
print(forecast.upper_bound)  # [115, 120, 113, ...]
print(forecast.metrics.mape)  # Mean Absolute Percentage Error
```

---

### AnomalyAgent - Outlier Detection

**Location:** `greenlang/agents/anomaly_agent_iforest.py`
**Status:** ✅ Production Ready
**Lines of Code:** 500+

#### Purpose

Isolation Forest-based anomaly detection for data quality, fraud detection, and outlier identification.

#### API Documentation

```python
from greenlang.agents import AnomalyAgentIForest

agent = AnomalyAgentIForest(
    contamination=0.05,  # Expect 5% anomalies
    random_state=42
)

# Detect anomalies
result = agent.run({
    "data": emissions_df,
    "features": ["emissions_tco2", "spend_usd", "data_quality_score"]
})

# Flag anomalies
anomalies = result.anomalies  # Boolean array
anomaly_scores = result.scores  # Anomaly scores (-1 to 1)

# Investigate
for idx in anomalies[anomalies == True].index:
    print(f"Anomaly: {emissions_df.loc[idx]}")
```

---

## Testing Infrastructure

### TestFixtures - Reusable Test Data

**Location:** `tests/fixtures/`
**Status:** ✅ Production Ready

#### Purpose

Centralized test fixtures (sample data, mock objects, factories) for consistent testing across all apps.

#### API Documentation

```python
from greenlang.testing import EmissionFactorFixtures, SupplierFixtures

# Get test emission factors
factors = EmissionFactorFixtures.get_factors(count=10)

# Get test suppliers
suppliers = SupplierFixtures.create_suppliers(count=100, country="US")

# Get test procurement data
procurement = ProcurementFixtures.create_procurement_data(
    num_suppliers=50,
    num_items=1000,
    year=2024
)
```

---

## Deployment & Infrastructure

### KubernetesManifests

**Location:** `infrastructure/kubernetes/`
**Status:** ✅ Production Ready
**Files:** 77 YAML manifests

#### Purpose

Production-ready Kubernetes manifests for deploying GreenLang applications with autoscaling, monitoring, and secrets management.

#### Components

- Deployments (API, Workers, Agents)
- Services (LoadBalancer, ClusterIP)
- ConfigMaps (Configuration)
- Secrets (API Keys, DB Passwords)
- HorizontalPodAutoscalers
- NetworkPolicies
- IngressRoutes

---

## CLI Framework

### Typer CLI Framework

**Location:** `greenlang/cli/`
**Status:** ✅ Production Ready
**Lines of Code:** 2,000+

#### Purpose

Production CLI framework built on Typer with 24+ commands for managing packs, agents, validation, and deployments.

#### Commands

```bash
# Agent management
gl agent create my-agent
gl agent run my-agent --input data.json
gl agent list
gl agent validate my-agent

# Pack management
gl pack create my-pack
gl pack build my-pack
gl pack publish my-pack
gl pack install emissions-core

# Validation
gl validate data.csv --rules rules.yaml
gl validate agent my-agent

# Deployment
gl deploy local
gl deploy kubernetes --namespace production

# Security
gl sbom generate
gl verify signature artifact.tar.gz

# Utilities
gl doctor  # Health check
gl version
gl config show
```

---

## Pack System

### Pack Management

**Location:** `greenlang/packs/`
**Status:** ✅ Production Ready
**Packs Available:** 22

#### Purpose

Modular, reusable climate intelligence components that can be installed, versioned, and shared.

#### Available Packs

1. **emissions-core** - Core emissions calculations
2. **boiler-solar** - Solar thermal for industrial heating
3. **boiler_replacement** - Boiler optimization
4. **hvac-measures** - HVAC efficiency
5. **industrial_process_heat** - Process heat optimization
6. **cement-lca** - Cement lifecycle assessment
7. ... (16 more)

#### Pack Structure

```
my-pack/
├── pack.yaml           # Pack metadata
├── agents/             # Agent implementations
├── data/               # Reference data
├── schemas/            # JSON schemas
├── tests/              # Unit tests
├── README.md           # Documentation
└── requirements.txt    # Dependencies
```

#### Usage

```python
from greenlang.packs import PackManager

pm = PackManager()

# Install pack
pm.install("emissions-core")

# Use pack
from emissions_core import CarbonAgent

agent = CarbonAgent()
result = agent.run({"fuel_type": "natural_gas", "consumption": 1000})
```

---

## Migration Patterns

### Common Migration Patterns

#### Pattern 1: Replace Custom LLM Code

**Before:**
```python
import openai
response = openai.ChatCompletion.create(...)
content = response.choices[0].message.content
```

**After:**
```python
from greenlang.intelligence import ChatSession
session = ChatSession(provider="openai")
response = session.complete(prompt="...")
content = response.content
# Bonus: automatic provenance tracking!
```

#### Pattern 2: Replace Custom Agent Code

**Before:**
```python
class MyAgent:
    def process(self, data):
        # Custom error handling, logging, etc.
        pass
```

**After:**
```python
from greenlang.sdk.base import Agent

class MyAgent(Agent):
    def execute(self, input_data):
        # Error handling, logging automatic!
        pass
```

#### Pattern 3: Replace Custom Validation

**Before:**
```python
def validate_data(data):
    errors = []
    if data["value"] < 0:
        errors.append("Value must be positive")
    # ... 50 more rules
    return errors
```

**After:**
```python
from greenlang.validation import ValidationFramework

validator = ValidationFramework(rules=[
    ValidationRule(field="value", rule_type="positive")
])
result = validator.validate(data)
```

---

## Infrastructure Usage Metrics (IUM)

### Calculating IUM Score

**Formula:**
```
IUM = (Infrastructure LOC / Total Application LOC) * 100
```

**Target:** 80%+

**Measurement:**
```bash
# Count infrastructure imports
grep -r "from greenlang\." app/ | wc -l

# Count total LOC
find app/ -name "*.py" -exec wc -l {} + | tail -1
```

**Example:**
- Total App LOC: 10,000
- Infrastructure Usage: 8,500
- IUM Score: 85% ✅ (Target: 80%)

---

## Support & Resources

### Getting Help

- **Documentation:** This catalog (you're reading it!)
- **Examples:** `examples/` directory
- **Community:** Discord, GitHub Discussions
- **Enterprise Support:** enterprise@greenlang.io

### Contributing to Infrastructure

1. Identify gap or limitation
2. Propose enhancement (GitHub Issue)
3. Submit PR with tests and documentation
4. Update this catalog

### Reporting Issues

**Infrastructure Bug:**
- File issue in main GreenLang repo
- Tag: `infrastructure`, `bug`
- Include: version, code snippet, expected vs. actual

**Missing Feature:**
- File issue in main GreenLang repo
- Tag: `infrastructure`, `enhancement`
- Include: use case, desired API, examples

---

## Appendix

### Version History

- **1.0.0** (2025-11-09): Initial catalog with 100+ components
- Future versions will track infrastructure additions/changes

### Glossary

- **IUM:** Infrastructure Usage Metric
- **ADR:** Architecture Decision Record
- **Zero Hallucination:** Never use LLM for numeric calculations
- **Tool-First:** Use LLM tools for structured data extraction
- **Provenance:** Complete lineage tracking from input to output

### License

This catalog and all GreenLang infrastructure: MIT License

---

**Last Updated:** November 9, 2025
**Maintainer:** GreenLang Infrastructure Team
**Contact:** infrastructure@greenlang.io

---

**Remember:** Always search this catalog before writing custom code. The infrastructure you need probably already exists!
