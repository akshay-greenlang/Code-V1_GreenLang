# Workshop 1: Introduction to GreenLang-First Architecture

**Duration:** 2 hours
**Level:** Beginner
**Prerequisites:** Basic Python knowledge

---

## Workshop Overview

Welcome to GreenLang! In this workshop, you'll learn why we built a centralized infrastructure platform and how the GreenLang-First policy ensures we never go back to the chaos of duplicate code.

### Learning Objectives

By the end of this workshop, you will:
- Understand the problems GreenLang-First solves
- Know the core policy rules and exceptions
- Install and use enforcement mechanisms
- Run your first infrastructure checks
- Know when custom code is allowed (and how to request it)

---

## Part 1: The Problem We're Solving (20 minutes)

### The "Before Times" - A Horror Story

Imagine joining a company and finding:
- 47 different implementations of LLM chat sessions
- Each with different error handling, retry logic, and token tracking
- Some using OpenAI, some Anthropic, some both
- No consistent caching, monitoring, or cost tracking
- Every developer reinventing the wheel

**Real example from our codebase (DO NOT DO THIS):**

```python
# CSRD App - Custom Implementation
import openai
client = openai.Client(api_key="...")
response = client.chat.completions.create(...)

# VCCI App - Different Custom Implementation
from anthropic import Anthropic
client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
message = client.messages.create(...)

# ESG Dashboard - Yet Another Implementation
def call_llm(prompt):
    # 200 lines of custom code
    # No error handling
    # No retry logic
    # No cost tracking
```

### The Cost of Duplication

**Real metrics from our audit:**
- 150+ hours spent writing duplicate LLM code
- $12K+ in unnecessary LLM costs (no caching)
- 23 different cache implementations (all broken differently)
- 8 production incidents from custom retry logic
- 200+ hours in code reviews asking "why not use infrastructure?"

### The GreenLang Solution

**One infrastructure. One way. Zero exceptions (mostly).**

```python
# The GreenLang Way - Always
from GL_COMMONS.infrastructure.llm import ChatSession

session = ChatSession(
    provider="openai",
    model="gpt-4",
    system_message="You are a helpful assistant"
)

response = session.send_message("Calculate carbon footprint")
# Automatic: caching, retry, token tracking, cost monitoring, logging
```

---

## Part 2: The GreenLang-First Policy (30 minutes)

### Core Principle

**NEVER write custom code when infrastructure exists.**

### The Four Commandments

#### 1. Thou Shall Use Infrastructure
```python
# WRONG - Custom LLM code
import openai
response = openai.ChatCompletion.create(...)

# RIGHT - Infrastructure
from GL_COMMONS.infrastructure.llm import ChatSession
session = ChatSession(provider="openai")
response = session.send_message(...)
```

#### 2. Thou Shall Inherit from Base Classes
```python
# WRONG - Custom agent
class MyAgent:
    def process(self):
        # Custom implementation
        pass

# RIGHT - Infrastructure
from GL_COMMONS.infrastructure.agents import Agent

class MyAgent(Agent):
    def execute(self):
        # Your business logic only
        pass
```

#### 3. Thou Shall Not Duplicate
```python
# WRONG - Rolling your own validation
def validate_data(data):
    if not data:
        raise ValueError("Empty")
    # 50 more lines...

# RIGHT - Infrastructure
from GL_COMMONS.infrastructure.validation import ValidationFramework

validator = ValidationFramework()
validator.validate(data, schema)
```

#### 4. Thou Shall Use Shared Services
```python
# WRONG - Direct database access
import psycopg2
conn = psycopg2.connect(...)

# RIGHT - Infrastructure
from GL_COMMONS.infrastructure.database import DatabaseManager

db = DatabaseManager()
results = db.execute_query(...)
```

### When Custom Code IS Allowed

**Only with an approved ADR (Architecture Decision Record).**

Valid reasons:
1. **Business-specific logic** that doesn't belong in infrastructure
2. **Performance-critical code** where infrastructure overhead matters
3. **External system integration** with unique requirements
4. **Prototype/proof-of-concept** (must migrate later)

Invalid reasons:
- "I didn't know infrastructure existed"
- "I prefer my way"
- "It's faster to write custom code"
- "I don't understand the infrastructure"

### The Exception Process

```
1. Identify need for custom code
2. Write ADR (Architecture Decision Record)
   - What: What custom code do you need?
   - Why: Why can't infrastructure work?
   - Alternatives: What infrastructure did you consider?
   - Consequences: What's the maintenance cost?
3. Submit for review
4. Get approval from Tech Lead + Infrastructure Team
5. Only then: Write custom code
6. Document in INFRASTRUCTURE_USAGE.md
```

---

## Part 3: Enforcement Mechanisms (20 minutes)

### The Three Layers of Defense

#### Layer 1: Pre-commit Hooks (Local)
Runs before you commit. Catches violations instantly.

#### Layer 2: CI/CD Pipeline (GitHub)
Runs on every push. Blocks merges if violations exist.

#### Layer 3: Code Review (Human)
Tech leads trained to spot violations.

### What Gets Checked?

1. **Import Analysis**
   - ✗ `import openai` (use ChatSession)
   - ✗ `import anthropic` (use ChatSession)
   - ✗ `import redis` (use CacheManager)
   - ✓ `from GL_COMMONS.infrastructure import ...`

2. **Inheritance Validation**
   - ✗ Class has `execute()` but doesn't inherit from Agent
   - ✓ `class MyAgent(Agent):`

3. **Pattern Detection**
   - ✗ Custom retry loops (use infrastructure)
   - ✗ Manual token counting (ChatSession does this)
   - ✗ Custom validation logic (use ValidationFramework)

4. **ADR Requirements**
   - Custom code must have corresponding ADR
   - ADR must be in approved state

---

## Part 4: Hands-On - Install Enforcement (30 minutes)

### Exercise 1: Install Pre-commit Hooks

**Step 1: Navigate to repo root**
```bash
cd C:\Users\aksha\Code-V1_GreenLang
```

**Step 2: Install pre-commit**
```bash
pip install pre-commit
```

**Step 3: Install hooks**
```bash
pre-commit install
```

**Step 4: Test the hooks**
```bash
pre-commit run --all-files
```

### Exercise 2: Your First Violation

Create a file with a violation:

```python
# test_violation.py
import openai  # This will be caught!

def chat(prompt):
    client = openai.Client()
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
```

**Try to commit:**
```bash
git add test_violation.py
git commit -m "Test violation"
```

**Expected output:**
```
GreenLang-First Policy Violation Detected!
==========================================

VIOLATION: Direct OpenAI import detected
File: test_violation.py
Line: 1

REQUIRED ACTION:
Replace with: from GL_COMMONS.infrastructure.llm import ChatSession

See: docs/GREENLANG_FIRST_POLICY.md
```

### Exercise 3: Fix the Violation

```python
# test_violation.py - FIXED
from GL_COMMONS.infrastructure.llm import ChatSession

def chat(prompt):
    session = ChatSession(
        provider="openai",
        model="gpt-4"
    )
    return session.send_message(prompt)
```

**Now commit:**
```bash
git add test_violation.py
git commit -m "Use infrastructure for LLM"
```

**Expected output:**
```
✓ GreenLang-First Policy Check ... Passed
✓ Inheritance Validation ......... Passed
✓ Import Analysis ................ Passed
[master abc123] Use infrastructure for LLM
```

---

## Part 5: Infrastructure Tour (15 minutes)

### Available Infrastructure Components

#### LLM & AI
- `ChatSession` - Unified LLM interface
- `RAGEngine` - Retrieval Augmented Generation
- `SemanticCacheManager` - Intelligent caching
- `PromptTemplateManager` - Prompt management

#### Agents
- `Agent` - Base class for all agents
- `CalculatorAgent` - Template for calculation agents
- `DataIntakeAgent` - Template for data ingestion
- `BatchProcessor` - Process items in batches

#### Data & Storage
- `CacheManager` - Redis caching
- `DatabaseManager` - Database abstraction
- `ValidationFramework` - Data validation
- `SchemaRegistry` - Schema management

#### Monitoring & Telemetry
- `TelemetryManager` - Metrics collection
- `MonitoringService` - Health checks
- `LoggingService` - Structured logging

### Quick Reference

```python
# LLM Chat
from GL_COMMONS.infrastructure.llm import ChatSession
session = ChatSession(provider="openai", model="gpt-4")
response = session.send_message("Hello")

# Caching
from GL_COMMONS.infrastructure.cache import CacheManager
cache = CacheManager()
cache.set("key", value, ttl=3600)
result = cache.get("key")

# Agent
from GL_COMMONS.infrastructure.agents import Agent

class MyAgent(Agent):
    def execute(self):
        return {"status": "success"}

# Validation
from GL_COMMONS.infrastructure.validation import ValidationFramework
validator = ValidationFramework()
validator.validate(data, schema)
```

---

## Part 6: Hands-On Lab - First Infrastructure Code (20 minutes)

### Lab: Build a Simple LLM-Powered Function

**Requirements:**
1. Create a function that uses LLM to analyze text sentiment
2. Must use GreenLang infrastructure
3. Must include proper error handling
4. Must pass enforcement checks

**Starter Code:**

```python
# sentiment_analyzer.py
from GL_COMMONS.infrastructure.llm import ChatSession
from GL_COMMONS.infrastructure.cache import CacheManager
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes text sentiment using LLM infrastructure."""

    def __init__(self):
        self.session = ChatSession(
            provider="openai",
            model="gpt-4",
            system_message="You are a sentiment analysis expert. "
                          "Respond with only: positive, negative, or neutral."
        )
        self.cache = CacheManager()

    def analyze(self, text: str) -> str:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment: positive, negative, or neutral
        """
        # TODO: Check cache first
        # TODO: If not cached, use ChatSession
        # TODO: Cache the result
        # TODO: Return sentiment
        pass

# YOUR TASK: Implement the analyze() method
```

**Solution:**

```python
def analyze(self, text: str) -> str:
    """Analyze sentiment of text."""

    # Check cache first
    cache_key = f"sentiment:{hash(text)}"
    cached = self.cache.get(cache_key)

    if cached:
        logger.info("Cache hit for sentiment analysis")
        return cached

    # Use LLM
    try:
        response = self.session.send_message(
            f"Analyze sentiment: {text}"
        )
        sentiment = response.strip().lower()

        # Validate response
        if sentiment not in ["positive", "negative", "neutral"]:
            raise ValueError(f"Invalid sentiment: {sentiment}")

        # Cache for 1 hour
        self.cache.set(cache_key, sentiment, ttl=3600)

        return sentiment

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise
```

**Test Your Code:**

```python
# test_sentiment.py
from sentiment_analyzer import SentimentAnalyzer

def test_sentiment():
    analyzer = SentimentAnalyzer()

    # Test cases
    assert analyzer.analyze("I love this!") == "positive"
    assert analyzer.analyze("This is terrible") == "negative"
    assert analyzer.analyze("It's okay") == "neutral"

    print("✓ All tests passed!")

if __name__ == "__main__":
    test_sentiment()
```

---

## Part 7: Q&A and Best Practices (15 minutes)

### Common Questions

**Q: What if infrastructure doesn't do exactly what I need?**
A: First, check if you can configure it. Second, submit a feature request. Third, if truly impossible, write an ADR for custom code.

**Q: Can I extend infrastructure classes?**
A: Yes! That's encouraged. Inherit and add your business logic.

**Q: What if I'm prototyping?**
A: Prototypes can use custom code, but must migrate to infrastructure before production.

**Q: How do I know what infrastructure exists?**
A: Check `GL_COMMONS/infrastructure/` and read `INFRASTRUCTURE_CATALOG.md`

**Q: Who approves ADRs?**
A: Tech Lead + Infrastructure Team. Usually 2-3 day turnaround.

### Best Practices

1. **Always check infrastructure first** before writing any code
2. **Read existing code** to see how others use infrastructure
3. **Ask in Slack** (#greenlang-help) if unsure
4. **Contribute back** - found a bug? Fix it in infrastructure
5. **Keep ADRs updated** - if you find a better way, update the ADR

---

## Workshop Wrap-Up

### What You Learned

✓ Why GreenLang-First exists (prevent duplication)
✓ The four core commandments
✓ When custom code is allowed (ADR required)
✓ How enforcement works (hooks, CI/CD, code review)
✓ How to install and use pre-commit hooks
✓ Basic infrastructure components
✓ Built your first infrastructure-based code

### Next Steps

1. Complete hands-on labs in your project
2. Take Workshop 2: LLM Infrastructure (deep dive)
3. Join #greenlang-help on Slack
4. Read the full policy: `docs/GREENLANG_FIRST_POLICY.md`
5. Review infrastructure catalog: `INFRASTRUCTURE_CATALOG.md`

### Homework Assignment

**Task:** Audit one of your existing Python files
1. Identify any infrastructure violations
2. Refactor to use GreenLang infrastructure
3. Submit a PR with your changes
4. Get it reviewed by a tech lead

**Deliverable:** PR link showing before/after

---

## Additional Resources

- **Policy Document:** `docs/GREENLANG_FIRST_POLICY.md`
- **Infrastructure Catalog:** `INFRASTRUCTURE_CATALOG.md`
- **ADR Template:** `docs/templates/ADR_TEMPLATE.md`
- **Slack Channel:** #greenlang-help
- **Office Hours:** Thursdays 2-3pm with Infrastructure Team

---

## Quick Reference Card

```
BEFORE WRITING CODE, ASK:
├─ Does infrastructure already do this? → Use it
├─ Can I extend infrastructure? → Inherit and extend
├─ Is this business-specific? → Needs ADR review
└─ Am I duplicating code? → STOP! Use infrastructure

ENFORCEMENT CHECKS:
1. Pre-commit hooks (local)
2. CI/CD pipeline (GitHub)
3. Code review (human)

WHEN IN DOUBT:
1. Check infrastructure catalog
2. Ask in #greenlang-help
3. Review existing code
4. Talk to Tech Lead
```

---

**Workshop Complete! Ready for Workshop 2: LLM Infrastructure**
