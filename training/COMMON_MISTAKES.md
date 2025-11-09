# Common Mistakes & Solutions Guide

**Version:** 1.0
**Last Updated:** 2025-11-09

---

## Introduction

This guide documents the 25+ most common mistakes developers make when adopting GreenLang-First, along with clear solutions.

**Remember:** We all make mistakes. The goal is to learn and improve.

---

## Category 1: Direct API Usage (Most Common)

### Mistake 1: Using `openai` Package Directly

**WRONG:**
```python
import openai

client = openai.Client(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Why Wrong:**
- No automatic caching
- No retry logic
- No cost tracking
- No token counting
- No standardized error handling
- Can't switch providers easily

**CORRECT:**
```python
from GL_COMMONS.infrastructure.llm import ChatSession

session = ChatSession(
    provider="openai",
    model="gpt-4"
)
response = session.send_message("Hello")

# Bonus: Get metrics
print(f"Cost: ${session.get_cost():.4f}")
print(f"Tokens: {session.get_token_count()}")
```

**Impact:** Using ChatSession reduces LLM costs by 30-40% through caching alone.

---

### Mistake 2: Using `anthropic` Package Directly

**WRONG:**
```python
from anthropic import Anthropic

client = Anthropic(api_key="...")
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

**CORRECT:**
```python
from GL_COMMONS.infrastructure.llm import ChatSession

session = ChatSession(
    provider="anthropic",
    model="claude-3-opus",
    max_tokens=1024
)
response = session.send_message("Hello")
```

**Why It Matters:** Same ChatSession interface works for OpenAI, Anthropic, Azure. Change provider with one parameter.

---

### Mistake 3: Using `redis` Package Directly

**WRONG:**
```python
import redis

r = redis.Redis(
    host='localhost',
    port=6379,
    db=0
)
r.set('key', 'value')
value = r.get('key')
```

**Why Wrong:**
- No connection pooling
- No error handling
- No serialization handling
- No TTL management
- No cache stats

**CORRECT:**
```python
from GL_COMMONS.infrastructure.cache import CacheManager

cache = CacheManager()
cache.set('key', 'value', ttl=3600)
value = cache.get('key')

# Bonus: Get stats
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']}%")
```

---

### Mistake 4: Using `psycopg2` Directly

**WRONG:**
```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="admin",
    password="admin123"
)
cursor = conn.cursor()
cursor.execute("SELECT * FROM users")
results = cursor.fetchall()
cursor.close()
conn.close()
```

**Why Wrong:**
- No connection pooling
- No query caching
- Manual connection management
- No query building helpers
- No transaction helpers

**CORRECT:**
```python
from GL_COMMONS.infrastructure.database import DatabaseManager

db = DatabaseManager()
results = db.query("SELECT * FROM users")

# Or with parameters
results = db.query(
    "SELECT * FROM users WHERE id = ?",
    [user_id]
)
```

---

## Category 2: Agent Pattern Violations

### Mistake 5: Not Inheriting from Agent Base Class

**WRONG:**
```python
class MyDataProcessor:
    def __init__(self):
        self.data = []

    def process(self):
        # Processing logic
        pass
```

**Why Wrong:**
- No lifecycle management
- No standardized error handling
- No automatic logging
- No metrics tracking
- Can't use agent infrastructure

**CORRECT:**
```python
from GL_COMMONS.infrastructure.agents import Agent

class MyDataProcessor(Agent):
    def setup(self):
        # Initialize resources
        self.data = []

    def execute(self):
        # Processing logic
        return result

    def teardown(self):
        # Cleanup
        pass
```

---

### Mistake 6: Ignoring Agent Templates

**WRONG:**
```python
from GL_COMMONS.infrastructure.agents import Agent

class CalculatorAgent(Agent):
    def setup(self):
        self.cache = CacheManager()

    def execute(self):
        # Manual cache checking
        cached = self.cache.get(...)
        if cached:
            return cached

        # Calculate
        result = self._calculate()

        # Manual caching
        self.cache.set(..., result)
        return result
```

**Why Wrong:** Reinventing functionality that exists in CalculatorAgent template.

**CORRECT:**
```python
from GL_COMMONS.infrastructure.agents.templates import CalculatorAgent

class MyCalculator(CalculatorAgent):
    def calculate(self, input_data):
        # Just implement calculation
        # Caching, validation, metrics automatic!
        return result
```

---

### Mistake 7: Missing Lifecycle Methods

**WRONG:**
```python
class MyAgent(Agent):
    def execute(self):
        # Initializing here instead of setup()
        self.llm = ChatSession(...)
        self.cache = CacheManager()

        # Process
        result = self._process()

        # Cleanup here instead of teardown()
        self.llm.close()

        return result
```

**CORRECT:**
```python
class MyAgent(Agent):
    def setup(self):
        # Initialize ONCE
        self.llm = ChatSession(...)
        self.cache = CacheManager()

    def execute(self):
        # Just process (can run multiple times)
        return self._process()

    def teardown(self):
        # Cleanup ONCE
        if self.llm:
            self.llm.close()
```

---

## Category 3: Validation & Error Handling

### Mistake 8: Custom Validation Logic

**WRONG:**
```python
def process_data(data):
    # Manual validation
    if not data:
        raise ValueError("Data is empty")

    if not isinstance(data.get('amount'), (int, float)):
        raise TypeError("Amount must be a number")

    if data.get('amount') < 0:
        raise ValueError("Amount must be positive")

    if len(data.get('name', '')) < 2:
        raise ValueError("Name too short")

    # ... 50 more lines of validation
```

**Why Wrong:**
- Duplicates validation logic
- Hard to maintain
- Inconsistent error messages
- No schema reuse

**CORRECT:**
```python
from GL_COMMONS.infrastructure.validation import ValidationFramework

validator = ValidationFramework()

schema = {
    "amount": {"type": "number", "min": 0, "required": True},
    "name": {"type": "string", "min_length": 2, "required": True}
}

def process_data(data):
    # One line validation
    validator.validate(data, schema)

    # Process validated data
    ...
```

---

### Mistake 9: Missing Error Handling

**WRONG:**
```python
class MyAgent(Agent):
    def execute(self):
        # No try/except
        result = self.llm.send_message(query)
        return result
```

**Why Wrong:** LLM calls can fail (rate limits, timeouts, errors).

**CORRECT:**
```python
class MyAgent(Agent):
    def execute(self):
        try:
            result = self.llm.send_message(query)
            return result

        except LLMError as e:
            if e.retryable:
                # Wait and retry
                time.sleep(e.retry_after)
                return self.llm.send_message(query)
            else:
                logger.error(f"LLM error: {e}")
                raise

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
```

**Even Better:** Use ChatSession's automatic retry:
```python
session = ChatSession(
    provider="openai",
    model="gpt-4",
    max_retries=3,  # Automatic retry!
    retry_delay=1.0
)
```

---

### Mistake 10: Not Logging Errors

**WRONG:**
```python
try:
    result = process_data()
except Exception as e:
    # Silent failure
    return None
```

**CORRECT:**
```python
from GL_COMMONS.infrastructure.logging import LoggingService

logger = LoggingService.get_logger(__name__)

try:
    result = process_data()
except Exception as e:
    logger.error("Processing failed", extra={
        "error": str(e),
        "error_type": type(e).__name__,
        "input_data": input_data
    }, exc_info=True)
    raise
```

---

## Category 4: Caching Mistakes

### Mistake 11: No Caching at All

**WRONG:**
```python
def get_emission_factor(activity_type):
    # Calls database every time
    return db.query(
        "SELECT factor FROM emission_factors WHERE activity=?",
        [activity_type]
    )
```

**Impact:** 50-200ms per call vs 1-5ms with caching.

**CORRECT:**
```python
from GL_COMMONS.infrastructure.cache import CacheManager

cache = CacheManager()

def get_emission_factor(activity_type):
    # Check cache first
    cache_key = f"emission_factor:{activity_type}"
    cached = cache.get(cache_key)

    if cached:
        return cached

    # Load from database
    factor = db.query(
        "SELECT factor FROM emission_factors WHERE activity=?",
        [activity_type]
    )

    # Cache for 24 hours (static data)
    cache.set(cache_key, factor, ttl=86400)

    return factor
```

---

### Mistake 12: Caching with Wrong TTL

**WRONG:**
```python
# Stock price (changes constantly) cached for 24 hours
cache.set("stock_price", price, ttl=86400)

# Company info (rarely changes) cached for 1 minute
cache.set("company_info", info, ttl=60)
```

**CORRECT:**
```python
# Stock price - short TTL
cache.set("stock_price", price, ttl=60)  # 1 minute

# Company info - long TTL
cache.set("company_info", info, ttl=86400)  # 24 hours

# Emission factors - very long TTL
cache.set("emission_factor", factor, ttl=604800)  # 7 days
```

**Rule of thumb:**
- Real-time data: 1-5 minutes
- Frequently updated: 1 hour
- Daily updated: 24 hours
- Rarely changes: 7+ days
- Static data: 30+ days

---

### Mistake 13: Not Invalidating Cache

**WRONG:**
```python
def update_company(company_id, data):
    # Update database
    db.execute("UPDATE companies SET ... WHERE id=?", [company_id, ...])

    # Forgot to invalidate cache!
    # Users will see stale data
```

**CORRECT:**
```python
def update_company(company_id, data):
    # Update database
    db.execute("UPDATE companies SET ... WHERE id=?", [company_id, ...])

    # Invalidate related caches
    cache.delete(f"company:{company_id}")
    cache.delete(f"company_list")  # If exists
    cache.delete_pattern(f"company:{company_id}:*")  # All related

    logger.info(f"Invalidated cache for company {company_id}")
```

---

## Category 5: LLM-Specific Mistakes

### Mistake 14: No Semantic Caching for Similar Queries

**WRONG:**
```python
# Each query costs money, even if similar
session.send_message("What is carbon footprint?")  # $0.01
session.send_message("What's a carbon footprint?")  # $0.01 (duplicate!)
session.send_message("Define carbon footprint")     # $0.01 (duplicate!)
```

**CORRECT:**
```python
from GL_COMMONS.infrastructure.llm import SemanticCacheManager

semantic_cache = SemanticCacheManager(similarity_threshold=0.95)

# All three queries hit cache after first one
# Saves $0.02!
```

---

### Mistake 15: Using Expensive Model for Simple Tasks

**WRONG:**
```python
# Using GPT-4 ($0.03/1K tokens) for simple tasks
session = ChatSession(provider="openai", model="gpt-4")

# Simple classification
session.send_message("Is 'hello' positive or negative?")
```

**CORRECT:**
```python
# Use cheaper model for simple tasks
cheap_session = ChatSession(
    provider="openai",
    model="gpt-3.5-turbo"  # $0.003/1K tokens (10x cheaper!)
)

expensive_session = ChatSession(provider="openai", model="gpt-4")

# Route based on complexity
if is_simple_task(query):
    response = cheap_session.send_message(query)
else:
    response = expensive_session.send_message(query)
```

---

### Mistake 16: Not Setting Token Limits

**WRONG:**
```python
session = ChatSession(provider="openai", model="gpt-4")

# No limit - might generate 4000 token response ($0.12)
response = session.send_message("Explain climate change")
```

**CORRECT:**
```python
session = ChatSession(
    provider="openai",
    model="gpt-4",
    max_tokens=500  # Limit response length
)

response = session.send_message("Explain climate change")
# Max cost: $0.015 (controlled)
```

---

## Category 6: Monitoring & Telemetry

### Mistake 17: No Telemetry

**WRONG:**
```python
class MyAgent(Agent):
    def execute(self):
        result = self._process()
        return result
        # No metrics, no monitoring, no visibility!
```

**CORRECT:**
```python
from GL_COMMONS.infrastructure.telemetry import TelemetryManager

class MyAgent(Agent):
    def setup(self):
        self.telemetry = TelemetryManager(service_name="my_agent")

    def execute(self):
        # Track execution
        with self.telemetry.timer("execution_duration_ms"):
            result = self._process()

            # Record metrics
            self.telemetry.increment("executions_total")
            self.telemetry.histogram("result_size", len(result))

            return result
```

---

### Mistake 18: Poor Logging

**WRONG:**
```python
print("Processing started")  # Don't use print!
print(f"Result: {result}")
```

**CORRECT:**
```python
from GL_COMMONS.infrastructure.logging import LoggingService

logger = LoggingService.get_logger(__name__)

logger.info("Processing started", extra={
    "input_size": len(input_data),
    "agent": self.name,
    "version": self.version
})

logger.info("Processing complete", extra={
    "result_size": len(result),
    "duration_ms": duration
})
```

---

### Mistake 19: No Health Checks

**WRONG:**
```python
# Agent with no health check
# If database goes down, no way to know until it fails
```

**CORRECT:**
```python
from GL_COMMONS.infrastructure.monitoring import HealthCheck

class MyAgent(Agent):
    def get_health(self):
        """Health check endpoint."""
        checks = {
            "database": self._check_database(),
            "cache": self._check_cache(),
            "llm": self._check_llm()
        }

        all_healthy = all(
            check["status"] == "healthy"
            for check in checks.values()
        )

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": checks
        }

    def _check_database(self):
        try:
            self.db.query("SELECT 1")
            return {"status": "healthy"}
        except:
            return {"status": "unhealthy"}
```

---

## Category 7: Performance & Optimization

### Mistake 20: N+1 Query Problem

**WRONG:**
```python
# Load companies
companies = db.query("SELECT * FROM companies")

# For each company, load emissions (N queries!)
for company in companies:
    emissions = db.query(
        "SELECT * FROM emissions WHERE company_id=?",
        [company["id"]]
    )
    # Process...
```

**Impact:** 1 + N queries instead of 1 query.

**CORRECT:**
```python
# Single JOIN query
results = db.query("""
    SELECT
        c.*,
        e.*
    FROM companies c
    LEFT JOIN emissions e ON c.id = e.company_id
""")

# Process in one pass
```

---

### Mistake 21: Loading Too Much Data

**WRONG:**
```python
# Load entire table into memory
all_emissions = db.query("SELECT * FROM emissions")  # 10M rows!

# Process one by one
for emission in all_emissions:
    process(emission)
```

**CORRECT:**
```python
# Use batch processing
from GL_COMMONS.infrastructure.agents import BatchProcessor

class EmissionProcessor(BatchProcessor):
    def __init__(self):
        super().__init__(batch_size=1000)  # 1000 at a time

    def process_batch(self, batch):
        # Process batch
        return results
```

---

### Mistake 22: Not Using Bulk Operations

**WRONG:**
```python
# Insert one by one (slow!)
for record in records:
    db.execute(
        "INSERT INTO emissions VALUES (?, ?, ?)",
        [record["company"], record["year"], record["emissions"]]
    )
    # 1000 records = 1000 database calls
```

**CORRECT:**
```python
# Bulk insert (fast!)
db.bulk_insert("emissions", records)
# 1000 records = 1 database call (1000x faster)
```

---

## Category 8: Architecture & Design

### Mistake 23: Monolithic Agents

**WRONG:**
```python
class SuperAgent(Agent):
    def execute(self):
        # Does everything!
        data = self._ingest_data()
        validated = self._validate_data(data)
        calculated = self._calculate(validated)
        report = self._generate_report(calculated)
        self._send_email(report)
        self._update_database(calculated)
        return report
```

**Why Wrong:** Hard to test, maintain, and reuse.

**CORRECT:**
```python
# Separate agents with clear responsibilities
class IngestionAgent(Agent):
    def execute(self):
        return self._ingest_data()

class ValidationAgent(Agent):
    def execute(self):
        return self._validate_data(self.input_data)

class CalculationAgent(Agent):
    def execute(self):
        return self._calculate(self.input_data)

# Chain with Pipeline
pipeline = Pipeline()
pipeline.add_agent(IngestionAgent())
pipeline.add_agent(ValidationAgent())
pipeline.add_agent(CalculationAgent())
```

---

### Mistake 24: Tight Coupling

**WRONG:**
```python
class MyAgent(Agent):
    def setup(self):
        # Hardcoded dependencies
        self.db = PostgresDB("localhost", 5432)
        self.cache = RedisCache("localhost", 6379)
```

**CORRECT:**
```python
class MyAgent(Agent):
    def __init__(self, db_manager=None, cache_manager=None):
        super().__init__()
        self.db = db_manager or DatabaseManager()
        self.cache = cache_manager or CacheManager()

# Easy to test with mocks
agent = MyAgent(
    db_manager=MockDatabase(),
    cache_manager=MockCache()
)
```

---

### Mistake 25: Not Using ADRs for Custom Code

**WRONG:**
```python
# Custom code without ADR
# Implementing custom retry logic because "I know better"
for attempt in range(3):
    try:
        result = custom_api_call()
        break
    except:
        time.sleep(2 ** attempt)
```

**Why Wrong:** Violates policy, no documentation of decision.

**CORRECT:**
1. Write ADR explaining why custom code needed
2. Get approval
3. Implement with documentation
4. Add to INFRASTRUCTURE_USAGE.md

---

## Summary: Top 10 Most Common

1. Using `openai` directly → Use `ChatSession`
2. Using `redis` directly → Use `CacheManager`
3. Not inheriting from `Agent` → Always inherit
4. No caching → Cache everything reasonable
5. No error handling → Try/except with logging
6. Custom validation → Use `ValidationFramework`
7. No telemetry → Track metrics
8. Poor logging → Structured logging
9. N+1 queries → Use JOINs or bulk operations
10. Missing ADRs → Document custom code decisions

---

## Prevention Checklist

Before submitting code:
- [ ] No direct imports (`openai`, `redis`, `psycopg2`)
- [ ] All agents inherit from `Agent`
- [ ] Caching implemented where appropriate
- [ ] Error handling with logging
- [ ] Validation using framework
- [ ] Telemetry/metrics tracked
- [ ] Health checks implemented
- [ ] ADR exists for custom code
- [ ] Pre-commit hooks passing
- [ ] Tests passing

---

**Remember: These mistakes are learning opportunities. We all make them!**

**Questions? Ask in #greenlang-help**
