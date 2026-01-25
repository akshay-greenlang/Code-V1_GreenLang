# Workshop 2: Using LLM & AI Infrastructure

**Duration:** 3 hours
**Level:** Intermediate
**Prerequisites:** Workshop 1 completed

---

## Workshop Overview

Deep dive into GreenLang's LLM infrastructure. Learn how to build production-ready AI features using ChatSession, RAG Engine, and Semantic Caching.

### Learning Objectives

- Master ChatSession API for all LLM providers
- Implement RAG (Retrieval Augmented Generation)
- Use semantic caching to reduce costs
- Build a complete LLM-powered agent
- Handle errors, retries, and rate limits
- Track costs and performance

---

## Part 1: ChatSession Deep Dive (40 minutes)

### Why ChatSession?

**The Problem:**
Every LLM provider has different APIs:
- OpenAI uses `client.chat.completions.create()`
- Anthropic uses `client.messages.create()`
- Different parameters, error handling, retry logic

**The Solution:**
One unified interface that works with all providers.

### ChatSession Architecture

```python
from GL_COMMONS.infrastructure.llm import ChatSession

# Under the hood:
# - Provider abstraction layer
# - Automatic retry with exponential backoff
# - Token counting and cost tracking
# - Response caching
# - Structured logging
# - Error handling
```

### Basic Usage

```python
from GL_COMMONS.infrastructure.llm import ChatSession

# Create a session
session = ChatSession(
    provider="openai",           # openai, anthropic, azure
    model="gpt-4",              # model name
    system_message="You are a carbon footprint expert",
    temperature=0.7,            # creativity (0-1)
    max_tokens=1000             # response length limit
)

# Send a message
response = session.send_message("Calculate emissions for 100km flight")

# Access metadata
print(f"Tokens used: {session.get_token_count()}")
print(f"Cost: ${session.get_cost()}")
print(f"Response time: {session.get_response_time()}ms")
```

### Advanced Features

#### 1. Conversation Context

```python
# ChatSession maintains conversation history
session = ChatSession(provider="openai", model="gpt-4")

# First message
session.send_message("What is Scope 3 emissions?")

# Follow-up (context maintained)
response = session.send_message("Give me an example")
# LLM knows "example" refers to Scope 3

# Get full conversation
history = session.get_conversation_history()
for msg in history:
    print(f"{msg['role']}: {msg['content']}")
```

#### 2. Streaming Responses

```python
# For long responses, stream token by token
session = ChatSession(provider="openai", model="gpt-4")

for chunk in session.stream_message("Write a detailed report on carbon accounting"):
    print(chunk, end="", flush=True)
# Prints: "Carbon accounting is..."
```

#### 3. Function Calling

```python
# Define functions the LLM can call
functions = [
    {
        "name": "calculate_emissions",
        "description": "Calculate CO2 emissions",
        "parameters": {
            "type": "object",
            "properties": {
                "distance_km": {"type": "number"},
                "transport_mode": {"type": "string"}
            }
        }
    }
]

session = ChatSession(
    provider="openai",
    model="gpt-4",
    functions=functions
)

response = session.send_message("Calculate emissions for 500km by car")

# Response includes function call
if response.function_call:
    name = response.function_call.name
    args = response.function_call.arguments
    # Execute the function
    result = calculate_emissions(**args)
```

#### 4. JSON Mode

```python
# Force JSON responses
session = ChatSession(
    provider="openai",
    model="gpt-4",
    response_format={"type": "json_object"}
)

response = session.send_message(
    "Extract: Company: Tesla, Year: 2023, Emissions: 1.2M tons"
)

import json
data = json.loads(response)
# {"company": "Tesla", "year": 2023, "emissions": 1200000}
```

#### 5. Multi-Provider Support

```python
# Same code, different providers
providers = ["openai", "anthropic", "azure"]

for provider in providers:
    session = ChatSession(
        provider=provider,
        model="gpt-4" if provider == "openai" else "claude-3-opus"
    )
    response = session.send_message("Hello")
    print(f"{provider}: {response}")
```

---

## Part 2: RAG Engine (Retrieval Augmented Generation) (45 minutes)

### What is RAG?

**Problem:** LLMs don't know your company data.

**Solution:** Retrieve relevant documents, inject into prompt.

```
User Query → Retrieve Docs → Augment Prompt → LLM Response
```

### RAGEngine Architecture

```python
from GL_COMMONS.infrastructure.llm import RAGEngine

# Components:
# 1. Document Store (vector database)
# 2. Embedding Model (text → vectors)
# 3. Retrieval (find similar docs)
# 4. Generation (LLM with context)
```

### Basic RAG Setup

```python
from GL_COMMONS.infrastructure.llm import RAGEngine

# Initialize RAG
rag = RAGEngine(
    embedding_model="text-embedding-3-small",  # OpenAI
    llm_provider="openai",
    llm_model="gpt-4",
    chunk_size=1000,           # Split docs into chunks
    chunk_overlap=200,         # Overlap for context
    top_k=5                    # Retrieve top 5 matches
)

# Index documents
documents = [
    {
        "id": "doc1",
        "content": "GreenLang CSRD compliance requires Scope 1, 2, and 3 emissions...",
        "metadata": {"type": "policy", "date": "2024-01"}
    },
    {
        "id": "doc2",
        "content": "Carbon accounting follows GHG Protocol standards...",
        "metadata": {"type": "guideline", "date": "2024-02"}
    }
]

rag.index_documents(documents)
```

### Querying with RAG

```python
# Query with automatic context retrieval
response = rag.query(
    "What emissions are required for CSRD?",
    include_sources=True
)

print(response.answer)
# "CSRD compliance requires Scope 1, 2, and 3 emissions..."

print(response.sources)
# [{"id": "doc1", "score": 0.95, "content": "..."}]
```

### Advanced RAG Patterns

#### 1. Hybrid Search (Keyword + Semantic)

```python
rag = RAGEngine(
    search_type="hybrid",  # keyword + vector
    alpha=0.7              # 70% semantic, 30% keyword
)

# Better for exact terms like "CSRD" or "Scope 3"
response = rag.query("CSRD Scope 3 requirements")
```

#### 2. Re-ranking

```python
# Retrieve more, then re-rank
rag = RAGEngine(
    top_k=20,              # Retrieve 20 candidates
    rerank_top_k=5,        # Re-rank to top 5
    rerank_model="cross-encoder"
)

# More accurate but slower
response = rag.query("complex query about carbon accounting")
```

#### 3. Metadata Filtering

```python
# Only search specific documents
response = rag.query(
    "What are the guidelines?",
    filters={"type": "guideline", "date": {"$gte": "2024-01"}}
)

# Only returns docs matching filter
```

#### 4. Citation Mode

```python
rag = RAGEngine(
    citation_mode=True
)

response = rag.query("CSRD requirements")
print(response.answer)
# "CSRD requires Scope 1, 2, and 3 emissions[1].
#  The reporting deadline is March 2025[2]."

print(response.citations)
# [1] doc1:paragraph2
# [2] doc3:paragraph5
```

---

## Part 3: Semantic Caching (30 minutes)

### Why Semantic Caching?

**Problem:**
```python
"What is carbon footprint?"  → $0.01
"What's a carbon footprint?"  → $0.01  # Same meaning, new cost!
"Define carbon footprint"     → $0.01  # Same meaning, new cost!
```

**Solution:** Cache by meaning, not exact text.

### SemanticCacheManager

```python
from GL_COMMONS.infrastructure.llm import SemanticCacheManager

cache = SemanticCacheManager(
    similarity_threshold=0.95,  # 95% similar = cache hit
    ttl=3600                    # Cache for 1 hour
)

# First query
query1 = "What is carbon footprint?"
response1 = cache.get(query1)  # None (not cached)

# Make LLM call
actual_response = session.send_message(query1)

# Cache the response
cache.set(query1, actual_response)

# Similar query
query2 = "What's a carbon footprint?"  # 97% similar
response2 = cache.get(query2)  # Returns cached response!

# Cost savings: $0.01 instead of $0.02
```

### Integration with ChatSession

```python
from GL_COMMONS.infrastructure.llm import ChatSession, SemanticCacheManager

# Automatic semantic caching
session = ChatSession(
    provider="openai",
    model="gpt-4",
    semantic_cache=True,  # Enable semantic caching
    cache_ttl=3600
)

# First call: $0.01
response1 = session.send_message("What is Scope 3?")

# Similar call: $0.00 (cached)
response2 = session.send_message("What's Scope 3 emissions?")

# Cache stats
print(f"Cache hit rate: {session.get_cache_hit_rate()}%")
print(f"Cost saved: ${session.get_cost_saved()}")
```

### Cache Invalidation

```python
# Clear specific query
cache.invalidate("What is carbon footprint?")

# Clear by pattern
cache.invalidate_pattern("carbon*")

# Clear by age
cache.invalidate_older_than(days=7)

# Clear all
cache.clear()
```

---

## Part 4: Error Handling & Resilience (25 minutes)

### Automatic Retry Logic

```python
session = ChatSession(
    provider="openai",
    model="gpt-4",
    max_retries=3,              # Retry up to 3 times
    retry_delay=1.0,            # Start with 1 second
    exponential_backoff=True    # 1s, 2s, 4s, 8s...
)

# Automatically retries on:
# - Rate limits (429)
# - Server errors (500, 502, 503)
# - Timeout errors
# - Network errors
```

### Rate Limit Handling

```python
from GL_COMMONS.infrastructure.llm import RateLimiter

# Respect rate limits
limiter = RateLimiter(
    max_requests_per_minute=60,
    max_tokens_per_minute=90000
)

session = ChatSession(
    provider="openai",
    model="gpt-4",
    rate_limiter=limiter
)

# Automatically waits if rate limit reached
for i in range(100):
    response = session.send_message(f"Query {i}")
    # Will pause when rate limit approached
```

### Error Recovery

```python
from GL_COMMONS.infrastructure.llm import ChatSession, LLMError

session = ChatSession(provider="openai", model="gpt-4")

try:
    response = session.send_message("Hello")

except LLMError as e:
    # Base exception for all LLM errors
    print(f"Error: {e.message}")
    print(f"Provider: {e.provider}")
    print(f"Retry possible: {e.retryable}")

    if e.retryable:
        # Wait and retry manually
        time.sleep(5)
        response = session.send_message("Hello")
```

### Fallback Providers

```python
from GL_COMMONS.infrastructure.llm import ChatSession

# Try OpenAI first, fallback to Anthropic
session = ChatSession(
    provider="openai",
    model="gpt-4",
    fallback_provider="anthropic",
    fallback_model="claude-3-opus"
)

# If OpenAI fails, automatically uses Anthropic
response = session.send_message("Hello")
```

---

## Part 5: Cost & Performance Monitoring (20 minutes)

### Token Tracking

```python
session = ChatSession(provider="openai", model="gpt-4")

response = session.send_message("Long prompt here...")

# Get token counts
stats = session.get_token_stats()
print(f"Input tokens: {stats['input_tokens']}")
print(f"Output tokens: {stats['output_tokens']}")
print(f"Total tokens: {stats['total_tokens']}")
```

### Cost Tracking

```python
# Per session
print(f"This session cost: ${session.get_cost():.4f}")

# Across all sessions
from GL_COMMONS.infrastructure.llm import CostTracker

tracker = CostTracker()
print(f"Today's total: ${tracker.get_daily_cost():.2f}")
print(f"This month: ${tracker.get_monthly_cost():.2f}")

# By provider
costs = tracker.get_cost_breakdown()
# {"openai": 45.23, "anthropic": 12.45}
```

### Performance Metrics

```python
# Response time
print(f"Response time: {session.get_response_time()}ms")

# Cache performance
cache_stats = session.get_cache_stats()
print(f"Hit rate: {cache_stats['hit_rate']}%")
print(f"Saved: ${cache_stats['cost_saved']:.2f}")

# Provider health
from GL_COMMONS.infrastructure.llm import ProviderHealthMonitor

health = ProviderHealthMonitor()
status = health.get_status("openai")
print(f"OpenAI latency: {status['avg_latency']}ms")
print(f"Error rate: {status['error_rate']}%")
```

---

## Part 6: Hands-On Lab - Build an LLM-Powered Agent (60 minutes)

### Lab Overview

Build a complete agent that:
1. Accepts user queries about carbon emissions
2. Uses RAG to find relevant documentation
3. Uses LLM to generate answers
4. Caches responses semantically
5. Tracks costs and performance

### Step 1: Create Agent Structure

```python
# carbon_qa_agent.py
from GL_COMMONS.infrastructure.agents import Agent
from GL_COMMONS.infrastructure.llm import ChatSession, RAGEngine, SemanticCacheManager
import logging

logger = logging.getLogger(__name__)

class CarbonQAAgent(Agent):
    """Agent that answers carbon accounting questions using RAG."""

    def __init__(self):
        super().__init__(
            name="carbon_qa_agent",
            version="1.0.0"
        )

        # Initialize LLM components
        self.rag = None
        self.session = None
        self.cache = None

    def setup(self):
        """Initialize infrastructure components."""
        # TODO: Initialize RAGEngine
        # TODO: Initialize ChatSession
        # TODO: Initialize SemanticCacheManager
        pass

    def execute(self):
        """Main execution logic."""
        # Get query from input
        query = self.input_data.get("query")

        if not query:
            raise ValueError("No query provided")

        # TODO: Check semantic cache
        # TODO: If not cached, use RAG + LLM
        # TODO: Cache the result
        # TODO: Return answer with sources

        pass

    def teardown(self):
        """Cleanup resources."""
        if self.session:
            self.session.close()
```

### Step 2: Implement Setup

```python
def setup(self):
    """Initialize infrastructure components."""

    # RAG Engine for document retrieval
    self.rag = RAGEngine(
        embedding_model="text-embedding-3-small",
        llm_provider="openai",
        llm_model="gpt-4",
        chunk_size=1000,
        top_k=3
    )

    # Load carbon accounting documents
    documents = self._load_documents()
    self.rag.index_documents(documents)

    # Chat Session for LLM
    self.session = ChatSession(
        provider="openai",
        model="gpt-4",
        system_message="""You are a carbon accounting expert.
        Answer questions using the provided context.
        Be accurate and cite sources.""",
        temperature=0.3  # Lower for factual responses
    )

    # Semantic cache
    self.cache = SemanticCacheManager(
        similarity_threshold=0.95,
        ttl=3600
    )

    logger.info("Agent setup complete")

def _load_documents(self):
    """Load carbon accounting documentation."""
    # In real implementation, load from database or files
    return [
        {
            "id": "scope1",
            "content": "Scope 1 emissions are direct GHG emissions from sources owned or controlled by the company...",
            "metadata": {"category": "scopes"}
        },
        {
            "id": "scope2",
            "content": "Scope 2 emissions are indirect GHG emissions from purchased electricity, steam, heating, and cooling...",
            "metadata": {"category": "scopes"}
        },
        {
            "id": "scope3",
            "content": "Scope 3 emissions are all other indirect emissions in the value chain...",
            "metadata": {"category": "scopes"}
        }
    ]
```

### Step 3: Implement Execute

```python
def execute(self):
    """Main execution logic."""

    query = self.input_data.get("query")
    logger.info(f"Processing query: {query}")

    # Check semantic cache
    cached_response = self.cache.get(query)
    if cached_response:
        logger.info("Cache hit!")
        return {
            "answer": cached_response["answer"],
            "sources": cached_response["sources"],
            "cached": True,
            "cost": 0
        }

    # Use RAG to get relevant context
    rag_response = self.rag.query(
        query,
        include_sources=True
    )

    # Build prompt with context
    context = "\n\n".join([
        f"Source {i+1}: {doc['content']}"
        for i, doc in enumerate(rag_response.sources)
    ])

    prompt = f"""Context:
{context}

Question: {query}

Please answer based on the context above. Cite sources by number."""

    # Get LLM response
    answer = self.session.send_message(prompt)

    # Prepare result
    result = {
        "answer": answer,
        "sources": [
            {
                "id": src["id"],
                "score": src["score"],
                "content": src["content"][:200] + "..."
            }
            for src in rag_response.sources
        ],
        "cached": False,
        "cost": self.session.get_cost(),
        "tokens": self.session.get_token_count()
    }

    # Cache for future queries
    self.cache.set(query, {
        "answer": answer,
        "sources": result["sources"]
    })

    logger.info(f"Query processed. Cost: ${result['cost']:.4f}")

    return result
```

### Step 4: Test the Agent

```python
# test_carbon_qa_agent.py
from carbon_qa_agent import CarbonQAAgent

def test_agent():
    """Test the Carbon QA Agent."""

    agent = CarbonQAAgent()

    # Test queries
    queries = [
        "What are Scope 1 emissions?",
        "What is Scope 1?",  # Similar - should cache hit
        "Explain Scope 2 emissions",
        "What's the difference between Scope 1 and 2?"
    ]

    agent.setup()

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        result = agent.execute_with_input({"query": query})

        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {len(result['sources'])} documents")
        for i, src in enumerate(result['sources']):
            print(f"  [{i+1}] {src['id']} (score: {src['score']:.2f})")

        print(f"\nCached: {result['cached']}")
        print(f"Cost: ${result['cost']:.4f}")
        print(f"Tokens: {result['tokens']}")

    agent.teardown()

if __name__ == "__main__":
    test_agent()
```

### Expected Output

```
============================================================
Query: What are Scope 1 emissions?
============================================================

Answer: Scope 1 emissions are direct GHG emissions from sources owned or controlled by the company[1]. These include emissions from company vehicles, on-site fuel combustion, and manufacturing processes.

Sources: 3 documents
  [1] scope1 (score: 0.98)
  [2] ghg_protocol (score: 0.85)
  [3] reporting_guide (score: 0.72)

Cached: False
Cost: $0.0234
Tokens: 456

============================================================
Query: What is Scope 1?
============================================================

Answer: Scope 1 emissions are direct GHG emissions from sources owned or controlled by the company[1]. These include emissions from company vehicles, on-site fuel combustion, and manufacturing processes.

Sources: 3 documents
  [1] scope1 (score: 0.98)
  [2] ghg_protocol (score: 0.85)
  [3] reporting_guide (score: 0.72)

Cached: True  ← Semantic cache hit!
Cost: $0.0000 ← Saved money!
Tokens: 0
```

---

## Part 7: Best Practices (20 minutes)

### 1. System Messages Matter

```python
# Bad: Vague system message
system_message = "You are helpful"

# Good: Specific with constraints
system_message = """You are a carbon accounting expert specializing in CSRD compliance.

Rules:
- Only answer based on provided context
- Cite sources by number [1], [2], etc.
- If unsure, say "I don't have enough information"
- Be concise but accurate
- Use technical terms correctly"""
```

### 2. Temperature Selection

```python
# Factual questions: Low temperature
session = ChatSession(
    model="gpt-4",
    temperature=0.1  # Deterministic, factual
)

# Creative tasks: Higher temperature
session = ChatSession(
    model="gpt-4",
    temperature=0.8  # Creative, varied
)
```

### 3. Token Optimization

```python
# Bad: Sending huge context every time
context = load_entire_database()  # 50,000 tokens
prompt = f"{context}\n\nQuestion: {query}"

# Good: Use RAG to get only relevant chunks
rag_response = rag.query(query, top_k=3)  # 1,000 tokens
context = format_rag_results(rag_response)
prompt = f"{context}\n\nQuestion: {query}"
```

### 4. Error Handling

```python
from GL_COMMONS.infrastructure.llm import ChatSession, LLMError

try:
    response = session.send_message(query)

except LLMError as e:
    if e.error_code == "rate_limit":
        # Wait and retry
        time.sleep(e.retry_after)
        response = session.send_message(query)

    elif e.error_code == "context_length_exceeded":
        # Reduce prompt size
        shorter_prompt = truncate_prompt(query)
        response = session.send_message(shorter_prompt)

    else:
        # Log and fail gracefully
        logger.error(f"LLM error: {e}")
        return {"error": "Unable to process query"}
```

### 5. Cost Monitoring

```python
# Set budget limits
session = ChatSession(
    model="gpt-4",
    cost_limit=10.00  # Max $10 per session
)

# Monitor in production
from GL_COMMONS.infrastructure.llm import CostAlertManager

alerts = CostAlertManager()
alerts.set_alert(
    threshold=100.00,  # $100/day
    notification="slack:#eng-alerts"
)
```

---

## Workshop Wrap-Up

### What You Learned

✓ ChatSession API for unified LLM access
✓ RAG Engine for context-aware responses
✓ Semantic caching to reduce costs
✓ Error handling and resilience
✓ Cost and performance monitoring
✓ Built a complete LLM-powered agent

### Key Takeaways

1. **Always use ChatSession** - Never import openai/anthropic directly
2. **RAG for accuracy** - Don't rely on LLM knowledge alone
3. **Cache semantically** - Save money on similar queries
4. **Monitor costs** - LLMs can get expensive fast
5. **Handle errors gracefully** - Rate limits will happen

### Homework Assignment

**Task:** Build a document Q&A system
1. Index 10+ documents using RAGEngine
2. Create an agent that answers questions
3. Implement semantic caching
4. Track and report costs
5. Submit PR with implementation

**Bonus:**
- Add hybrid search
- Implement citation mode
- Add multi-provider fallback

---

## Next Steps

1. Workshop 3: Building Agents (advanced agent patterns)
2. Review code examples in `examples/llm/`
3. Read LLM infrastructure docs: `docs/infrastructure/LLM.md`
4. Join #llm-infrastructure on Slack

---

**Workshop Complete! Ready for Workshop 3: Agent Framework**
