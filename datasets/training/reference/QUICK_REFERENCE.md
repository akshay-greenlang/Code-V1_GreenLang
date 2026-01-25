# GreenLang-First Quick Reference Card

**Version:** 1.0
**Print:** 2 pages, double-sided

---

## The Golden Rule

### NEVER WRITE CUSTOM CODE WHEN INFRASTRUCTURE EXISTS

---

## Core Components

### LLM & AI

```python
# ChatSession - Unified LLM interface
from GL_COMMONS.infrastructure.llm import ChatSession

session = ChatSession(
    provider="openai",  # openai, anthropic, azure
    model="gpt-4",
    system_message="You are...",
    temperature=0.7
)
response = session.send_message("Hello")

# RAG Engine
from GL_COMMONS.infrastructure.llm import RAGEngine

rag = RAGEngine(llm_provider="openai")
rag.index_documents(docs)
result = rag.query("question")

# Semantic Caching
from GL_COMMONS.infrastructure.llm import SemanticCacheManager

cache = SemanticCacheManager(similarity_threshold=0.95)
```

### Agents

```python
# Base Agent
from GL_COMMONS.infrastructure.agents import Agent

class MyAgent(Agent):
    def setup(self):
        # Initialize resources
        pass

    def execute(self):
        # Business logic
        return result

    def teardown(self):
        # Cleanup
        pass

# Templates
from GL_COMMONS.infrastructure.agents.templates import (
    CalculatorAgent,
    DataIntakeAgent,
    ReportingAgent
)
```

### Data & Storage

```python
# CacheManager
from GL_COMMONS.infrastructure.cache import CacheManager

cache = CacheManager()
cache.set("key", value, ttl=3600)
result = cache.get("key")

# DatabaseManager
from GL_COMMONS.infrastructure.database import DatabaseManager

db = DatabaseManager()
results = db.query("SELECT * FROM table WHERE id=?", [123])

# ValidationFramework
from GL_COMMONS.infrastructure.validation import ValidationFramework

validator = ValidationFramework()
schema = {"name": {"type": "string", "required": True}}
validator.validate(data, schema)
```

### Monitoring

```python
# Telemetry
from GL_COMMONS.infrastructure.telemetry import TelemetryManager

telemetry = TelemetryManager(service_name="my_agent")
telemetry.increment("requests_total")
telemetry.histogram("duration_ms", 125)

# Logging
from GL_COMMONS.infrastructure.logging import LoggingService

logger = LoggingService.get_logger(__name__)
logger.info("Message", extra={"key": "value"})
```

---

## Decision Tree

```
BEFORE WRITING CODE:
├─ Does infrastructure exist? → YES → Use it ✓
├─ Can I extend infrastructure? → YES → Inherit ✓
├─ Is this business-specific? → YES → Write ADR → Review → Implement
└─ Am I duplicating code? → YES → STOP! Use infrastructure ✗
```

---

## Common Patterns

### LLM Call with Caching

```python
from GL_COMMONS.infrastructure.llm import ChatSession
from GL_COMMONS.infrastructure.cache import CacheManager

session = ChatSession(provider="openai", model="gpt-4")
cache = CacheManager()

# Check cache
cached = cache.get(f"llm:{query}")
if cached:
    return cached

# Call LLM
response = session.send_message(query)

# Cache result
cache.set(f"llm:{query}", response, ttl=3600)
```

### Agent with Validation

```python
from GL_COMMONS.infrastructure.agents import Agent
from GL_COMMONS.infrastructure.validation import ValidationFramework

class ValidatedAgent(Agent):
    def setup(self):
        self.validator = ValidationFramework()

    def execute(self):
        # Validate input
        schema = {"amount": {"type": "number", "min": 0}}
        self.validator.validate(self.input_data, schema)

        # Process
        return result
```

### Batch Processing

```python
from GL_COMMONS.infrastructure.agents import BatchProcessor

class MyBatchAgent(BatchProcessor):
    def __init__(self):
        super().__init__(batch_size=100, max_workers=4)

    def process_item(self, item):
        # Process single item
        return result
```

---

## Enforcement Checklist

Before commit:
- [ ] No direct imports: `openai`, `anthropic`, `redis`, `psycopg2`
- [ ] All agents inherit from `Agent`
- [ ] Use infrastructure for: LLM, cache, database, validation
- [ ] Custom code has approved ADR
- [ ] Pre-commit hooks installed and passing

Before PR:
- [ ] CI/CD checks passing
- [ ] Code review from tech lead
- [ ] Documentation updated
- [ ] Tests passing

---

## Quick Troubleshooting

### "Pre-commit hook failed"
→ Check violation message
→ Replace custom code with infrastructure
→ Commit again

### "Import not found"
→ Install: `pip install -r requirements.txt`
→ Check path: `GL_COMMONS/infrastructure/`

### "Need custom code"
→ Write ADR: `docs/adr/ADR-XXX.md`
→ Submit for review
→ Wait for approval
→ Document in INFRASTRUCTURE_USAGE.md

---

## Resources

| Resource | Location |
|----------|----------|
| Full Policy | `docs/GREENLANG_FIRST_POLICY.md` |
| Infrastructure Catalog | `INFRASTRUCTURE_CATALOG.md` |
| Workshops | `training/workshops/` |
| Examples | `examples/` |
| Slack Help | `#greenlang-help` |
| ADR Template | `docs/templates/ADR_TEMPLATE.md` |

---

## Emergency Contacts

- **Tech Lead:** @tech-lead
- **Infrastructure Team:** @infra-team
- **Slack:** #greenlang-help
- **Office Hours:** Thursdays 2-3pm

---

**Remember: Infrastructure first. Custom code last. Always.**

---

**© GreenLang Team | Version 1.0 | Updated 2025-11**
